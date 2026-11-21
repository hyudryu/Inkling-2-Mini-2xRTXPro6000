# SPDX-License-Identifier: Apache-2.0
"""Loop message controller: the down projection for all streams in two
launches.

`Inkling2LoopDecoderLayer.blend` projects the normalized `[dest, message]`
pair of every stream to rank `r` with `torch.bmm`. For decode-sized batches
cuBLAS runs that batched (S, T, 2D) x (S, 2D, r) product on a couple of
thread blocks, taking 15-25 us for under 2 MB of weights. Here every stream
runs the reference `linear_k_groups` tiles (`ops.linear`): each 1024-wide K
group is accumulated in fp32 by its own program and the partials are summed
in order, exactly as `Inkling2MessageController.forward` (the per-stream
reference) computes it.
"""

import torch

from vllm.triton_utils import tl, triton

from .linear import _k_group_tile

# Rows per stream up to which the split-K tiles beat the batched matmul.
MAX_TOKENS = 512


@triton.jit
def _stream_k_group_kernel(
    X,  # (S, T, K) bf16
    W,  # (S, N, K) bf16
    Y,  # (S, G, T, N) fp32 partials
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    s = tl.program_id(1).to(tl.int64)
    _k_group_tile(
        X + s * T * K,
        W + s * (N * K),
        Y + s * GROUPS * T * N,
        T,
        N,
        K,
        BM,
        BN,
        True,
        tl.program_id(0) % tl.cdiv(N, BN),
        tl.program_id(0) // tl.cdiv(N, BN),
        tl.program_id(2),
    )


@triton.jit
def _stream_k_group_sum_kernel(
    Partials, Y, TN, total, GROUPS: tl.constexpr, BLOCK: tl.constexpr
):
    tl.extra.cuda.gdc_wait()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    src = Partials + (offs // TN).to(tl.int64) * (GROUPS * TN) + offs % TN
    acc = tl.load(src, mask)
    for group in range(1, GROUPS):
        acc = acc + tl.load(src + group * TN, mask)
    tl.store(Y + offs, acc, mask)


def use_stream_linear(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() == 3
        and weight.dim() == 3
        and 0 < x.shape[1] <= MAX_TOKENS
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.shape[0] == weight.shape[0]
        and x.shape[2] == weight.shape[2]
        and x.shape[2] > 1024
    )


def stream_linear_k_groups(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """bf16 `(S, T, N)`: `linear_k_groups(x[s], weight[s], bf16)` for every
    stream `s` of bf16 `x` `(S, T, K)` and `weight` `(S, N, K)`."""
    streams, tokens, k = x.shape
    n = weight.shape[1]
    groups = triton.cdiv(k, 1024)
    block_m = 16
    block_n = 32 if tokens <= 16 else 64
    partials = torch.empty(
        (streams, groups, tokens, n), dtype=torch.float32, device=x.device
    )
    out = torch.empty((streams, tokens, n), dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(n, block_n) * triton.cdiv(tokens, block_m), streams, groups)
    _stream_k_group_kernel[grid](
        x,
        weight,
        partials,
        tokens,
        n,
        k,
        groups,
        block_m,
        block_n,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
        launch_pdl=True,
    )
    total = out.numel()
    _stream_k_group_sum_kernel[(triton.cdiv(total, 1024),)](
        partials, out, tokens * n, total, groups, 1024, launch_pdl=True
    )
    return out
