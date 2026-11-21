"""Checks for the V1 kernel changes. Run inside the serving container (the patch
is mounted there):

    docker cp tests/test_v1_kernels.py inkling-2-mini:/tmp/
    docker exec inkling-2-mini python3 /tmp/test_v1_kernels.py

Each check compares the new path with the one it replaces on random inputs.
"""
import torch

from vllm.models.inkling2.nvidia.ops.gemv import sm12x_bf16_gemv
from vllm.models.inkling2.nvidia.ops.linear import linear_k_groups
from vllm.models.inkling2.nvidia.ops.loop_fused import stream_linear_k_groups
from vllm.models.inkling2.nvidia.ops.softcap import softcap_

dev = "cuda"
bf16 = torch.bfloat16
torch.manual_seed(0)
ok = True


def report(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


# 1. Soft-cap: one in-place pass vs torch's three (must be bit-identical).
for shape in ((8, 201024), (256, 201024)):
    x = torch.randn(shape, device=dev) * 40
    x.view(-1)[:7] = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), 1e-30, -1e-30, 30.0], device=dev
    )
    ref = torch.tanh(x / 30.0) * 30.0
    got = softcap_(x.clone(), 30.0)
    same = ref.view(-1).view(torch.int32) == got.view(-1).view(torch.int32)
    report(f"softcap {shape}", same.all(), f"mismatches={int((~same).sum())}")

# 2. Loop controller down projection: all streams at once vs the per-stream
#    reference linear_k_groups (must be bit-identical) and vs cuBLAS bmm.
D, R = 2304, 64
for streams, tokens in ((3, 1), (3, 5), (3, 8), (3, 16), (3, 40), (3, 160), (3, 256)):
    down = (torch.randn(streams, R, 2 * D, device=dev) * 0.02).to(bf16)
    pair = torch.randn(streams, tokens, 2 * D, device=dev).to(bf16)
    got = stream_linear_k_groups(pair, down)
    ref = torch.stack(
        [linear_k_groups(pair[s].contiguous(), down[s].contiguous(), bf16) for s in range(streams)]
    )
    bmm = torch.bmm(pair, down.transpose(1, 2))
    exact = pair.double() @ down.double().transpose(1, 2)
    err_new = (got.double() - exact).abs().max().item()
    err_bmm = (bmm.double() - exact).abs().max().item()
    report(
        f"stream projection S={streams} T={tokens}",
        (got == ref).all() and err_new <= err_bmm * 1.5 + 1e-3,
        f"same as bmm {(got == bmm).float().mean().item() * 100:.2f}%  "
        f"max err vs fp64: new {err_new:.2e}, bmm {err_bmm:.2e}",
    )

# 3. Dynamic-tap projection: Triton GEMV vs cuBLAS F.linear (bit-identical).
for tokens in (2, 5, 8, 16):
    w = (torch.randn(32, D, device=dev) * 0.05).to(bf16)
    x = torch.randn(tokens, D, device=dev).to(bf16)
    report(
        f"dynamic taps T={tokens}",
        (torch.nn.functional.linear(x, w) == sm12x_bf16_gemv(x, w)).all(),
    )

# 4. Draft Markov head for one request: Triton GEMV vs cuBLAS addmm. Draft-only,
#    so a one-ulp difference cannot change final outputs; check it stays there.
w = (torch.randn(201024, 256, device=dev) * 0.05).to(bf16)
bias = torch.randn(201024, device=dev).to(bf16)
lat = torch.randn(1, 256, device=dev).to(bf16)
a, b = torch.addmm(bias, lat, w.T), sm12x_bf16_gemv(lat, w, bias)
ulp = (a.float().abs().clamp_min(1e-3) * 2**-7)
report(
    "markov head T=1",
    ((a.float() - b.float()).abs() <= ulp).all(),
    f"identical {(a == b).float().mean().item() * 100:.2f}%  argmax equal: {bool(a.argmax() == b.argmax())}",
)

print("ALL PASS" if ok else "FAILURES")
raise SystemExit(0 if ok else 1)
