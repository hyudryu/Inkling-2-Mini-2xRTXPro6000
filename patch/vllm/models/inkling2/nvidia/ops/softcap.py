# SPDX-License-Identifier: Apache-2.0
"""In-place logit soft-cap, `tanh(x / cap) * cap`, in one pass.

The stock LogitsProcessor runs three elementwise kernels over the full
`[tokens, vocab]` fp32 logits (scale, tanh, scale). This kernel does the same
three operations per element, in the same order and precision: PyTorch
divides by a scalar as a multiply by the fp32 reciprocal, and both use
libdevice's `tanhf`, so the result is bit-identical.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _softcap_kernel(X, n, inv_cap, cap, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask, other=0.0)
    tl.store(X + offs, tldevice.tanh(x * inv_cap) * cap, mask)


def use_fused_softcap(logits: torch.Tensor) -> bool:
    return logits.is_cuda and logits.dtype == torch.float32 and logits.is_contiguous()


def softcap_(logits: torch.Tensor, cap: float) -> torch.Tensor:
    n = logits.numel()
    if n:
        inv_cap = torch.tensor(1.0 / cap, dtype=torch.float32).item()
        block = 4096 if n >= 1 << 22 else 1024
        _softcap_kernel[(triton.cdiv(n, block),)](
            logits, n, inv_cap, cap, BLOCK=block, num_warps=4
        )
    return logits
