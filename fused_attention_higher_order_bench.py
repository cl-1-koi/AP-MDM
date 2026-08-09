"""Tiny cost benchmark: fused vs math SDPA for the S4 IRED shape.

Measures, on the local GPU, at (batch=16, seq=80, width=64, heads=4, fp32,
bool padding mask — the S4 IRED comparator's exact per-layer shape):

* forward + first-order backward time: mem_efficient vs math backend,
* full IRED-pattern step time (inner grad with create_graph=True + mixed
  derivative into parameters) on the math backend (the only backend that can
  do it),
* peak CUDA memory of each.

Standalone evidence for FUSED_ATTENTION_HIGHER_ORDER_REVIEW.md; does not touch
training code.  Allocations are a few MB and the loop is 60 iterations.
"""

from __future__ import annotations

import json
import math
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

BATCH, SEQ, WIDTH, HEADS = 16, 80, 64, 4
HEAD_DIM = WIDTH // HEADS
ITERS, WARMUP = 50, 10


def make(device="cuda", dtype=torch.float32, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x0 = torch.randn(BATCH, SEQ, WIDTH, generator=gen).to(device, dtype)
    w = (torch.randn(3 * WIDTH, WIDTH, generator=gen) / math.sqrt(WIDTH)).to(device, dtype)
    w.requires_grad_(True)
    mask = torch.ones(BATCH, 1, 1, SEQ, dtype=torch.bool, device=device)
    mask[:, 0, 0, -7:] = False
    return x0, w, mask


def energy(x, w, mask):
    qkv = (x @ w.t()).chunk(3, dim=-1)
    q, k, v = (t.view(BATCH, SEQ, HEADS, HEAD_DIM).transpose(1, 2) for t in qkv)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return o.pow(2).sum()


def timed(fn, iters=ITERS):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3  # ms


def peak_mb(fn):
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def first_order(backend):
    x0, w, mask = make()

    def step():
        x = x0.clone().requires_grad_(True)
        with sdpa_kernel([backend]):
            torch.autograd.grad(energy(x, w, mask), [x, w])
    return step


def ired_step_math():
    x0, w, mask = make()

    def step():
        x = x0.clone().requires_grad_(True)
        with sdpa_kernel([SDPBackend.MATH]):
            g = torch.autograd.grad(energy(x, w, mask), x, create_graph=True)[0]
            torch.autograd.grad(g.sum(), w)
    return step


def main():
    torch.cuda.init()
    out = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "shape": {"batch": BATCH, "seq": SEQ, "width": WIDTH, "heads": HEADS,
                     "dtype": "float32", "mask": "bool padding"},
           "iters": ITERS}
    out["fwd+bwd_ms_mem_efficient"] = round(timed(first_order(SDPBackend.EFFICIENT_ATTENTION)), 4)
    out["fwd+bwd_ms_math"] = round(timed(first_order(SDPBackend.MATH)), 4)
    out["ired_step_ms_math_only_option"] = round(timed(ired_step_math()), 4)
    out["peak_mb_fwd+bwd_mem_efficient"] = round(peak_mb(first_order(SDPBackend.EFFICIENT_ATTENTION)), 3)
    out["peak_mb_fwd+bwd_math"] = round(peak_mb(first_order(SDPBackend.MATH)), 3)
    out["peak_mb_ired_step_math"] = round(peak_mb(ired_step_math()), 3)
    attn_floats = BATCH * HEADS * SEQ * SEQ
    out["attention_matrix_mb_fp32"] = round(attn_floats * 4 / 2**20, 3)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
