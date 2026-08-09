"""Minimal reproduction: higher-order gradients through fused SDPA backends.

Evidence script for FUSED_ATTENTION_HIGHER_ORDER_REVIEW.md.  Standalone: it
does not import or modify any training code.  It exercises the exact IRED
training pattern

    g = grad(energy, model_input, create_graph=True)   # inner energy gradient
    grad(g.sum(), model.parameters())                  # mixed 2nd derivative

on every ``torch.nn.attention.SDPBackend`` of the installed PyTorch build,
distinguishing:

* first-order input gradient (``create_graph=False``),
* the IRED inner gradient itself (``create_graph=True``),
* the mixed second derivative into model parameters,
* the pure input-input second derivative,

and checks numerical agreement of the math backend against an explicit
attention implementation.  Problem size mirrors the S4 comparator:
sequence 80, width 64, 4 heads.

Usage: ``python fused_attention_higher_order_probe.py [--device cuda]``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

SEQ, WIDTH, HEADS, BATCH = 80, 64, 4, 2
HEAD_DIM = WIDTH // HEADS

BACKENDS = {
    "math": SDPBackend.MATH,
    "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


def make_problem(device: str, dtype: torch.dtype, use_mask: bool, seed: int = 0):
    """Small energy model: x -> q,k,v -> SDPA -> squared scalar energy."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x0 = torch.randn(BATCH, SEQ, WIDTH, generator=gen).to(device, dtype)
    w_qkv = torch.randn(3 * WIDTH, WIDTH, generator=gen).to(device, dtype) / math.sqrt(WIDTH)
    w_out = torch.randn(WIDTH, generator=gen).to(device, dtype) / math.sqrt(WIDTH)
    w_qkv.requires_grad_(True)
    w_out.requires_grad_(True)
    mask = None
    if use_mask:
        # Bool key-padding mask, broadcastable to (B, 1, 1, S): the exact
        # shape repro/model.py passes to F.scaled_dot_product_attention.
        mask = torch.ones(BATCH, 1, 1, SEQ, dtype=torch.bool, device=device)
        mask[0, 0, 0, -7:] = False
    return x0, w_qkv, w_out, mask


def split_heads(t: torch.Tensor) -> torch.Tensor:
    b, s, d = t.shape
    return t.view(b, s, HEADS, d // HEADS).transpose(1, 2)


def energy_sdpa(x: torch.Tensor, w_qkv, w_out, mask) -> torch.Tensor:
    qkv = (x @ w_qkv.t()).chunk(3, dim=-1)
    q, k, v = (split_heads(t) for t in qkv)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    o = o.transpose(1, 2).reshape(BATCH, SEQ, WIDTH)
    return (o * w_out).pow(2).sum()


def energy_explicit(x: torch.Tensor, w_qkv, w_out, mask) -> torch.Tensor:
    """Reference attention written in primitive ops (math-backend semantics)."""
    qkv = (x @ w_qkv.t()).chunk(3, dim=-1)
    q, k, v = (split_heads(t) for t in qkv)
    scores = q @ k.transpose(-2, -1) / math.sqrt(HEAD_DIM)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    o = torch.softmax(scores, dim=-1) @ v
    o = o.transpose(1, 2).reshape(BATCH, SEQ, WIDTH)
    return (o * w_out).pow(2).sum()


def probe_one(device: str, dtype: torch.dtype, use_mask: bool, backend: str | None) -> dict:
    """Run the IRED pattern on one backend; record every failure mode."""
    record: dict = {"device": device, "dtype": str(dtype), "mask": use_mask,
                    "backend": backend or "default"}
    ctx = sdpa_kernel([BACKENDS[backend]]) if backend else _nullctx()
    x0, w_qkv, w_out, mask = make_problem(device, dtype, use_mask)
    params = [w_qkv, w_out]

    with ctx:
        # 1. plain first-order input gradient (inference-style, no graph)
        x = x0.clone().requires_grad_(True)
        try:
            e = energy_sdpa(x, w_qkv, w_out, mask)
            g1 = torch.autograd.grad(e, x)[0]
            record["first_order_input_grad"] = "ok"
            record["energy"] = float(e)
        except Exception as exc:  # noqa: BLE001 - evidence capture
            record["first_order_input_grad"] = f"FAIL: {type(exc).__name__}: {exc}"
            record["ired_inner_create_graph"] = "n/a (forward failed)"
            record["mixed_param_derivative"] = "n/a"
            record["input_input_second"] = "n/a"
            return record

        # 2. IRED inner gradient with create_graph=True
        x = x0.clone().requires_grad_(True)
        try:
            e = energy_sdpa(x, w_qkv, w_out, mask)
            g = torch.autograd.grad(e, x, create_graph=True)[0]
            record["ired_inner_create_graph"] = "ok"
        except Exception as exc:  # noqa: BLE001
            record["ired_inner_create_graph"] = f"FAIL: {type(exc).__name__}: {exc}"
            record["mixed_param_derivative"] = "n/a (inner create_graph failed)"
            record["input_input_second"] = "n/a (inner create_graph failed)"
            return record

        # 3. mixed second derivative into model parameters (what trains IRED)
        try:
            mixed = torch.autograd.grad(g.sum(), params, retain_graph=True)
            record["mixed_param_derivative"] = "ok"
            record["mixed_param_grad_norm"] = float(mixed[0].norm())
        except Exception as exc:  # noqa: BLE001
            record["mixed_param_derivative"] = f"FAIL: {type(exc).__name__}: {exc}"

        # 4. pure input-input second derivative
        try:
            second = torch.autograd.grad(g.sum(), x)
            record["input_input_second"] = "ok"
            record["input_input_grad_norm"] = float(second[0].norm())
        except Exception as exc:  # noqa: BLE001
            record["input_input_second"] = f"FAIL: {type(exc).__name__}: {exc}"
    return record


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def agreement_check(device: str, dtype: torch.dtype) -> dict:
    """Forced-math SDPA vs explicit attention: energy, dE/dx, mixed d/dparams."""
    out: dict = {"device": device, "dtype": str(dtype)}
    results = {}
    for name, fn in (("math_sdpa", energy_sdpa), ("explicit", energy_explicit)):
        x0, w_qkv, w_out, mask = make_problem(device, dtype, use_mask=True)
        x = x0.clone().requires_grad_(True)
        if name == "math_sdpa":
            with sdpa_kernel([SDPBackend.MATH]):
                e = fn(x, w_qkv, w_out, mask)
                g = torch.autograd.grad(e, x, create_graph=True)[0]
        else:
            e = fn(x, w_qkv, w_out, mask)
            g = torch.autograd.grad(e, x, create_graph=True)[0]
        mixed = torch.autograd.grad(g.sum(), [w_qkv, w_out], retain_graph=True)
        results[name] = (e.detach().float(), g.detach().float(),
                         [m.detach().float() for m in mixed])
    e0, g0, m0 = results["math_sdpa"]
    e1, g1, m1 = results["explicit"]
    out["max_abs_diff_energy"] = float((e0 - e1).abs())
    out["max_abs_diff_input_grad"] = float((g0 - g1).abs().max())
    out["max_abs_diff_mixed_wqkv"] = float((m0[0] - m1[0]).abs().max())
    out["max_abs_diff_mixed_wout"] = float((m0[1] - m1[1]).abs().max())
    tol = 1e-10 if dtype == torch.float64 else 2e-4
    out["agree"] = bool(
        out["max_abs_diff_energy"] < tol
        and out["max_abs_diff_input_grad"] < tol
        and out["max_abs_diff_mixed_wqkv"] < tol
        and out["max_abs_diff_mixed_wout"] < tol
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    print(json.dumps({
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "problem": {"batch": BATCH, "seq": SEQ, "width": WIDTH, "heads": HEADS},
    }, indent=2))

    combos = []
    if args.device == "cuda":
        combos = [
            (torch.float32, True, None),
            (torch.float32, True, "math"),
            (torch.float32, True, "mem_efficient"),
            (torch.float32, True, "flash"),
            (torch.float32, True, "cudnn"),
            (torch.float32, False, "mem_efficient"),
            (torch.float16, False, None),
            (torch.float16, False, "math"),
            (torch.float16, False, "mem_efficient"),
            (torch.float16, False, "flash"),
            (torch.float16, False, "cudnn"),
            (torch.float16, True, "flash"),
        ]
    else:
        combos = [
            (torch.float32, True, None),
            (torch.float32, True, "math"),
            (torch.float32, True, "mem_efficient"),
            (torch.float32, True, "flash"),
            (torch.float32, False, None),
            (torch.float64, True, None),
            (torch.float64, True, "math"),
        ]

    for dtype, use_mask, backend in combos:
        try:
            rec = probe_one(args.device, dtype, use_mask, backend)
        except Exception:  # noqa: BLE001 - keep the matrix complete
            rec = {"device": args.device, "dtype": str(dtype), "mask": use_mask,
                   "backend": backend or "default",
                   "probe_error": traceback.format_exc(limit=2)}
        print(json.dumps(rec))

    print(json.dumps(agreement_check(args.device, torch.float64 if args.device == "cpu" else torch.float32)))

    if args.device == "cuda":
        # Report the kernel the default dispatcher actually picks for the
        # model's exact config (fp32, bool mask) via profiler kernel names.
        x0, w_qkv, w_out, mask = make_problem("cuda", torch.float32, True)
        x = x0.clone().requires_grad_(True)
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                energy_sdpa(x, w_qkv, w_out, mask).sum().backward()
            kernels = sorted({e.key for e in prof.key_averages()
                              if e.device_type == torch.profiler.DeviceType.CUDA
                              or "cuda" in e.key.lower() or "kernel" in e.key.lower()})
            print(json.dumps({"default_dispatch_cuda_fp32_masked_kernels": kernels}))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"profiler": f"unavailable: {exc}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
