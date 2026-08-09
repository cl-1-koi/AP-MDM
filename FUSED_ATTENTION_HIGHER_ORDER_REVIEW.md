# Fused-attention higher-order-gradient review

Date: 2026-08-09. Binding spec: `KIMI_FUSED_ATTENTION_HIGHER_ORDER_REVIEW_SPEC.md`.
Author: Kimi Code CLI review unit (autonomous).

Local environment for all measurements: PyTorch 2.7.0 (CUDA 12.8 build),
Python 3.10.12, one NVIDIA A10 (sm86, 23 GB; an unrelated training job was
running — only tiny, short-lived allocations were used, per spec).

---

## 1. Verdict

**Second-order differentiation through fused scaled-dot-product attention is a
known, documented, still-open limitation of every fused SDPA backend** (CUDA
FlashAttention-2, CUDA memory-efficient/cutlass, CUDA cuDNN, and the CPU flash
kernel) in the installed PyTorch 2.7.0 and — per release notes and still-open
upstream issues — through PyTorch 2.13.0 (2026-07-08). Only the **math
backend** (a C++ composition of differentiable primitives) supports the double
backward that IRED-style training needs. PyTorch backend selection **can**
force the math kernel on GPU, it preserves the IRED objective **exactly** (it
*is* explicit attention), and at the S4 shape (seq 80, width 64, 4 heads) the
measured cost over the fused kernel is ~2–3 % step time and ~3 MB of memory.

This is a positive result: the limitation is real, the workaround is validated
locally on GPU, and the fix is a one-line context manager — not a model change.

## 2. The three gradients that must be distinguished

IRED-style training (energy network `E_θ(x, ·)`, model parameters optimized
through `grad(energy, model_input, create_graph=True)`) involves three
distinct autograd operations, with *different* backend support:

1. **First-order input gradient** `∂E/∂x` (ordinary backward, used by the
   IRED *inference* inner loop): works on **every** backend, fused included.
   This is why the gap is easy to miss — ordinary backward succeeding says
   nothing about higher-order support.
2. **Pure input–input second derivative** `∂²E/∂x²` (Hessian-vector product;
   needed by second-order inner optimizers, not by vanilla IRED): fails on all
   fused backends.
3. **Mixed second derivative into parameters** `∂²E/∂x∂θ` — obtained by
   differentiating the `create_graph=True` inner gradient w.r.t. `θ`. **This
   is what IRED training requires**, because the training loss is applied to
   the model's *gradient output*: the official IRED code computes
   `torch.autograd.grad([energy.sum()], [opt_out], create_graph=True)[0]`
   ([models.py:808](https://github.com/yilundu/ired_code_release/blob/main/models.py#L808))
   and applies the denoising MSE loss to that gradient, so `d(loss)/dθ`
   differentiates `∂E/∂(input)` a second time into `θ` through every layer,
   attention included.

Failure signature on fused backends (loud, not silent): the
`grad(..., create_graph=True)` call itself **returns normally**; the
`RuntimeError` fires only when the returned gradient is differentiated again:

```
RuntimeError: derivative for aten::_scaled_dot_product_{flash,efficient,cudnn}_attention[_for_cpu]_backward is not implemented
```

Mechanism (PyTorch source, main @
[`b9ce263`](https://github.com/pytorch/pytorch/tree/b9ce26335c42fc249582d3ad8140348655058644),
checked 2026-08-09): `tools/autograd/derivatives.yaml` registers derivatives
only for the fused **forward** ops, mapping them to the fused backward aten
calls
([L2931–2957](https://github.com/pytorch/pytorch/blob/b9ce26335c42fc249582d3ad8140348655058644/tools/autograd/derivatives.yaml#L2931));
**no derivative entry exists for any fused `*_backward` op**
([native_functions.yaml L15266, L15274, L15293, L15306](https://github.com/pytorch/pytorch/blob/b9ce26335c42fc249582d3ad8140348655058644/aten/src/ATen/native/native_functions.yaml#L15266)).
The math path
([attention.cpp L894](https://github.com/pytorch/pytorch/blob/b9ce26335c42fc249582d3ad8140348655058644/aten/src/ATen/native/transformers/attention.cpp#L894),
"Naive, composite implementation") is pure matmul/softmax primitives, hence
differentiable to arbitrary order.

## 3. Minimal local reproduction and observed output

Scripts committed with this report (standalone; no training code touched):

- `fused_attention_higher_order_probe.py` — backend × device × dtype matrix of
  the exact IRED pattern at the S4 shape (batch 2, seq 80, width 64, 4 heads),
  tested both with and without the bool key-padding mask supported by
  `repro/model.py`, plus a forced-math vs explicit-attention agreement check.
  The current fixed-length `S4Energy` invocation does not pass a padding mask;
  the no-mask rows are therefore its operational configuration. The masked
  rows cover the same block's variable-length use and reach the same
  higher-order-support verdict.
- `fused_attention_higher_order_bench.py` — time/memory cost at batch 16.

Commands:

```
python fused_attention_higher_order_probe.py --device cpu
python fused_attention_higher_order_probe.py --device cuda
python fused_attention_higher_order_bench.py
```

### 3.1 Observed: CUDA (A10, torch 2.7.0), fp32 + bool mask

| backend | 1st-order `∂E/∂x` | inner `create_graph` call | mixed `∂²E/∂x∂θ` | `∂²E/∂x²` |
|---|---|---|---|---|
| **default dispatch** | ok | ok (returns) | **FAIL** `derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented` | **FAIL** (same) |
| forced math | ok | ok | **ok** (‖g‖=134.49) | **ok** |
| forced mem_efficient | ok | ok (returns) | **FAIL** (same as default) | **FAIL** |
| forced flash | **forward fails**: `No available kernel` (fp32 unsupported) | n/a | n/a | n/a |
| forced cuDNN | **forward fails**: `No available kernel` (fp32 unsupported) | n/a | n/a | n/a |

Profiler kernel names under default dispatch for this config confirm the
dispatcher picks the **memory-efficient cutlass kernel**:
`fmha_cutlassF_f32_aligned_64x64_rf_sm80` (forward),
`fmha_cutlassB_f32_aligned_64x64_k32_sm80` (backward). Warnings emitted:
"Flash Attention does not support non-null attn_mask", "Expected query, key
and value to all be of dtype: {Half, BFloat16}".

### 3.2 Observed: CUDA, fp16 (unmasked and masked)

| backend | 1st-order | mixed `∂²E/∂x∂θ` | `∂²E/∂x²` |
|---|---|---|---|
| default (fp16, no mask) → flash | ok | **FAIL** `_scaled_dot_product_flash_attention_backward ... not implemented` | **FAIL** |
| forced math | ok | **ok** | **ok** |
| forced mem_efficient | ok | **FAIL** `_scaled_dot_product_efficient_attention_backward ...` | **FAIL** |
| forced flash | ok | **FAIL** `_scaled_dot_product_flash_attention_backward ...` | **FAIL** |
| forced cuDNN | ok | **FAIL** `_scaled_dot_product_cudnn_attention_backward ...` | **FAIL** |
| forced flash, fp16 **+ bool mask** | **forward fails** `No available kernel` (no arbitrary-mask support) | n/a | n/a |

### 3.3 Observed: CPU (torch 2.7.0)

| backend | 1st-order | mixed `∂²E/∂x∂θ` | `∂²E/∂x²` |
|---|---|---|---|
| default (fp32 **and** fp64, masked or not) → CPU flash kernel | ok | **FAIL** `_scaled_dot_product_flash_attention_for_cpu_backward ...` | **FAIL** |
| forced math (fp32/fp64) | ok | **ok** | **ok** |
| forced mem_efficient | forward fails (`No viable backend`, CUDA-only kernel) | n/a | n/a |

Note: on this build the CPU *default* dispatches to the fused CPU flash kernel
even for fp64 and even with a bool mask, so CPU-only IRED training would hit
the same wall unless math is forced. This matches upstream issue
[#181177](https://github.com/pytorch/pytorch/issues/181177) (open,
2026-04-22).

### 3.4 Observed: numerical agreement, forced-math SDPA vs explicit attention

`energy_explicit` re-implements attention as
`softmax(QKᵀ/√d ⊕ mask)V` in primitive ops. Same init, same inputs:

- CPU fp64: max |Δ| = **0.0** on energy, `∂E/∂x`, mixed `∂²E/∂x∂W_qkv`,
  `∂²E/∂x∂w_out` → `"agree": true`.
- CUDA fp32: max |Δ| = **0.0** on all four → `"agree": true`.

(Bitwise 0.0 here because the math backend executes the identical op sequence;
tolerance-based checks are appropriate for other dtypes/shapes.)

### 3.5 Observed: cost at the S4 shape (batch 16, seq 80, width 64, 4 heads, fp32, bool mask, A10)

| measurement | value |
|---|---|
| fwd + 1st-order bwd, mem_efficient | 2.229 ms / step, peak 21.5 MB |
| fwd + 1st-order bwd, math | 2.282 ms / step (+2.4 %), peak 24.7 MB (+3.3 MB) |
| full IRED step (inner `create_graph` grad + mixed param grad), math | 5.25 ms / step, peak 38.3 MB |
| materialized attention matrix (16,4,80,80) fp32 | 1.56 MB |

Interpretation: at seq 80 the fused kernel's O(S²)-memory advantage is
irrelevant (1.6 MB) and its speed advantage is within noise (~2 %). The
~2.4× step cost of the IRED pattern over a plain step is the price of the
second derivative itself — identical on any backend — and it only runs on
math/explicit attention at all.

## 4. Source-linked support matrix

Support = "second-order autograd through the attention op" (mixed or pure
second derivative), per evidence below. First-order backward works on all
listed backends.

| backend | device/dtype | 2nd-order | evidence |
|---|---|---|---|
| SDPA **math** | CPU+CUDA, fp16/32/64 | **YES** | composite primitives, [attention.cpp L894](https://github.com/pytorch/pytorch/blob/b9ce26335c42fc249582d3ad8140348655058644/aten/src/ATen/native/transformers/attention.cpp#L894); local §3.1–3.4 |
| SDPA **mem_efficient** (cutlass, ex-xFormers) | CUDA fp32/fp16 | **NO** | local §3.1; [pytorch#117974](https://github.com/pytorch/pytorch/issues/117974) (open 2024-01-22): drisspg, 2024-01-23 — *"Double backward for SDPA_Efficient_attention is not currently implemented."*; no derivative in [derivatives.yaml](https://github.com/pytorch/pytorch/blob/b9ce26335c42fc249582d3ad8140348655058644/tools/autograd/derivatives.yaml#L2931) |
| SDPA **flash** (FA-2) | CUDA fp16/bf16 only, no arbitrary mask | **NO** | local §3.2; [pytorch#116350](https://github.com/pytorch/pytorch/issues/116350) (open 2023-12-23): jbschlosser — *"The double backward for SDPA is a known implementation gap"*; drisspg — *"the current work around is to disable the 'fused' kernels and run the 'math' kernels"* |
| SDPA **cuDNN** | CUDA fp16/bf16 | **NO** | local §3.2 (`_scaled_dot_product_cudnn_attention_backward`); no derivative entry in derivatives.yaml |
| SDPA **flash_for_cpu** (CPU default in 2.7) | CPU fp32/fp64 | **NO** | local §3.3; [pytorch#181177](https://github.com/pytorch/pytorch/issues/181177) (open 2026-04-22); fix [PR #184656](https://github.com/pytorch/pytorch/pull/184656) open, unmerged, CPU-compile scope only |
| **FlashAttention** standalone (Dao-AILab, FA1/2/3) | CUDA fp16/bf16, no arbitrary mask | **NO** | issue [#699](https://github.com/Dao-AILab/flash-attention/issues/699) "Support for second order differentiation" — tridao, 2023-12: *"No we don't have plan to do that."*; JVP issue [#1672](https://github.com/Dao-AILab/flash-attention/issues/1672) (open): *"JVP would be nontrivial to implement … no plan … unless someone volunteers"*; even first-order `torch.func.grad` broken: [#2071](https://github.com/Dao-AILab/flash-attention/issues/2071) (open 2025-12-16); source: `FlashAttnFunc` implements only fwd/bwd on raw CUDA ops ([flash_attn_interface.py L828–881](https://github.com/Dao-AILab/flash-attention/blob/a369df707e19/flash_attn/flash_attn_interface.py)); [README](https://github.com/Dao-AILab/flash-attention/blob/a369df707e19/README.md) L145–146: fp16/bf16 only |
| **xFormers** `memory_efficient_attention` | CUDA | **NO** | `_fMHA.backward` decorated [`@once_differentiable`](https://github.com/facebookresearch/xformers/blob/v0.0.28/xformers/ops/fmha/__init__.py) (v0.0.28, L158) — raises on any second-order attempt; same kernel upstreamed as PyTorch mem_efficient |
| **Triton** fused-attention tutorial | CUDA | **NO** | [06-fused-attention.py](https://github.com/triton-lang/triton/blob/23689d2aa356/python/tutorials/06-fused-attention.py) `_attention` implements only forward/backward; PyTorch SDPA has **no Triton backend** on NVIDIA — registry is flash/mem_efficient/math/cuDNN/overrideable ([torch/nn/attention/__init__.py L73–79](https://github.com/pytorch/pytorch/blob/b9ce26335c42/torch/nn/attention/__init__.py#L73)) |
| **JAX** `jax.nn.dot_product_attention` (default XLA path) | TPU/GPU/CPU | **yes** (by composition) | [docs](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html); source is pure jnp matmul/softmax — arbitrary-order differentiable (inference from source composition, not a doc claim; the optional `implementation='cudnn'` custom call is restricted) |

Version drift check: PyTorch release notes 2.7.1 (2025-06-04) through 2.13.0
(2026-07-08) contain **no** second-order/SDPA-backward change (SDPA entries
are performance/fixes only, e.g. "Add Flash Attention 4 to sdpa" in 2.10);
issues #116350, #117974, #181177 remain open as of 2026-08-09. The official
SDPA doc pages ([2.7](https://pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html),
[stable](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
[sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html))
list backends and toggles but **never mention** the higher-order limitation —
the authoritative statements are the issues above.

## 5. Answers to the spec's seven questions

1. **Which SDPA backends support the needed double backward?** Only `MATH`
   (CPU and CUDA, all tested dtypes). Flash, memory-efficient, and cuDNN fused
   kernels — CUDA and the CPU flash kernel — do not, and this is a
   maintainer-acknowledged "known implementation gap" (#116350, #117974).
2. **Are failures backend-, dtype-, device-, or version-specific?** The
   failure is *fused-backend-specific*, not dtype- or device-specific: it
   reproduces on CUDA fp16 (flash, mem_efficient, cuDNN), CUDA fp32
   (mem_efficient), and CPU fp32/fp64 (flash_for_cpu). For fp32/64 or
   arbitrary masks, flash/cuDNN fail even earlier (no forward kernel). It is
   not version-fixed: unchanged from ≤2.7 through 2.13 (2026-07-08); the only
   fix PR (#184656) is CPU-compile-scoped and unmerged.
3. **Can backend selection force a correct math kernel on GPU?** Yes —
   `with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):` (or global
   `torch.backends.cuda.enable_flash_sdp(False)` /
   `enable_mem_efficient_sdp(False)` / `enable_cudnn_sdp(False)`). Verified
   locally: forced-math CUDA double backward succeeds and matches explicit
   attention exactly (§3.1, §3.4). This is also the maintainer-recommended
   workaround (drisspg in #116350/#117974).
4. **Do FlashAttention/xFormers expose supported double-backward paths?** No.
   FlashAttention: explicitly declined by the maintainer (#699); JVP also
   unplanned (#1672); `torch.func.grad` broken even first-order (#2071).
   xFormers: `once_differentiable` backward — second order raises by design.
5. **What workarounds exist?** Exact-objective: (a) force math backend;
   (b) hand-written explicit attention (matmul/softmax — what the official
   IRED and EBT repos both do); (c) custom fused kernel with a differentiable
   backward (Triton/CUDA; community prototype
   [`amorehead/jvp_flash_attention`](https://github.com/amorehead/jvp_flash_attention)
   linked from #116350/#1672); (d) reformulate using HVP/JVP — but note HVP
   still requires a differentiable first backward, so it reduces to (a)/(b) on
   PyTorch. Objective-changing (non-goals per spec, listed for completeness):
   first-order/FOMAML-style detaching ([MAML §5.2](https://arxiv.org/abs/1703.03400):
   ~33 % speedup, "nearly the same" performance), truncated unrolling
   (EBT S1/S2 variants, [arXiv:2507.02092](https://arxiv.org/abs/2507.02092)),
   DARTS-style first-order approximation
   ([arXiv:1806.09055](https://arxiv.org/abs/1806.09055)), finite differences,
   implicit-gradient/DEQ reformulations.
6. **Which workaround preserves the IRED objective exactly, and at what cost
   for seq 80 / width 64 / 4 heads?** (a) and (b) are exact — the math backend
   *is* explicit attention (verified 0.0 max diff on energy and both gradient
   kinds, §3.4). Measured local cost of forcing math at the S4 shape: +2.4 %
   step time and +3.3 MB peak for first-order; a full IRED second-order step
   costs 5.25 ms vs 2.28 ms for a plain step and peaks at 38 MB — the
   second-derivative cost itself, backend-independent, and trivially within
   the A10's 23 GB.
7. **Recommended immediate implementation + later optimization, with
   falsifiable tests.** See §6.

## 6. Ranked workarounds for the S4 IRED comparator

**R1 — immediate (recommended): force the math backend for the IRED arm.**
Wrap the energy model's forward in
`with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):` during IRED training
(inner gradient *and* outer loss backprop), leaving the other S4 arms on the
default fused dispatch. Exact objective, one-line change, no mask/dtype
restrictions (fp32 + bool mask supported), no dependency additions.
*Falsifiable validation:* (i) unit test — at the S4 shape, forced-math SDPA
matches an explicit-attention reference on energy, `∂E/∂x`, and mixed
`∂²E/∂x∂θ` to < 1e-6 (fp32; the probe in §3.4 shows 0.0) and the fused
backends raise `RuntimeError` on the mixed derivative; (ii) smoke test — one
IRED training step (`grad(E, inp, create_graph=True)` → loss on the gradient
→ `backward()` into θ) completes and produces finite, nonzero parameter
gradients.

**R2 — equivalent alternative: explicit attention module** (matmul/softmax in
place of `F.scaled_dot_product_attention`) in the IRED energy model.
Identical numerics to R1 (that is literally what the math backend computes);
only worth it if context-manager scoping proves awkward. This is what both
reference implementations already ship: IRED's `CausalSelfAttention`
([diffusion_lib/transformer.py L51–58](https://github.com/yilundu/ired_code_release/blob/main/diffusion_lib/transformer.py#L51))
and EBT's `Attention`
([model/ar_ebt_default.py](https://github.com/alexiglad/EBT/blob/main/model/ar_ebt_default.py#L154)).

**R3 — later optimization (only if profiling says attention dominates):**
benchmark first — §3.5 shows the math/fused delta at the S4 shape is ~2 %, so
a custom kernel is likely wasted effort. If a future arm scales sequence
length substantially, port a double-backward-capable Triton FlashAttention-2
(starting point: `amorehead/jvp_flash_attention`, referenced in pytorch#116350
and flash-attention#1672).
*Falsifiable validation:* value, first-order, and mixed second-derivative
agreement vs the math backend within fp16-appropriate tolerance at the target
shape, plus a measured throughput win over forced math on the same A10.

**Explicitly rejected (change the IRED objective — non-goal per spec):**
detached/first-order inner gradients, truncated unrolling, finite differences,
implicit-function-theorem reformulations. They would make fused kernels
"work" by removing the second derivative, i.e. by training a different
objective.

## 7. Commands/tests that verify numerical agreement with explicit attention

```
# full backend × dtype × device matrix + agreement check (CPU, no GPU needed)
python fused_attention_higher_order_probe.py --device cpu
# same on GPU (tiny allocation; safe alongside the running job)
python fused_attention_higher_order_probe.py --device cuda
# time/memory cost at the S4 shape
python fused_attention_higher_order_bench.py
# repo regression suite (unchanged by this work unit)
python -m pytest tests/ -q
```

The probe's `agreement_check` (§3.4) is the exact-agreement gate; when the
IRED arm lands, its test should reuse `energy_explicit` from the probe as the
reference oracle.

## 8. What could not be established (clean negatives)

- **Local Hopper/cuDNN-preferred behavior**: the A10 (sm86) had cuDNN SDPA
  "runtime disabled" in the default config; cuDNN was exercised only forced,
  fp16, unmasked. The no-double-backward conclusion for cuDNN rests on the
  local forced-run failure plus the absent derivative entry in
  derivatives.yaml — not on a Hopper run.
- **FlashAttention-on-GPU standalone behavior**: `flash_attn` is not installed
  on this stack (prebuilt wheels don't cover torch 2.7.0/cu12.8, per
  `repro/model.py`), so FA claims rest on its repo issues/source, not a local
  run.
- **Exact dispatch-priority tables** for every dtype/mask/arch combination:
  established locally for the two configurations that matter (fp32+mask →
  mem_efficient, fp16 → flash), not exhaustively.
- **JAX arbitrary-order claim** is inferred from source composition (pure jnp
  XLA path); no doc sentence states it.
- **IRED repo's PyTorch version**: unpinned upstream (no requirements file) —
  but its explicit-attention implementation makes the version moot.
- No evidence was found that *any* fused SDPA backend in *any* PyTorch release
  ≤ 2.13.0 supports double backward; absence of a release-note entry is
  negative evidence, and the tracking issues remain open.

## 9. Primary sources (all fetched/verified 2026-08-09)

PyTorch — issues: [#116350](https://github.com/pytorch/pytorch/issues/116350)
(open, 2023-12-23; jbschlosser "known implementation gap"; drisspg math-kernel
workaround), [#117974](https://github.com/pytorch/pytorch/issues/117974)
(open, 2024-01-22; efficient-attention double backward "not currently
implemented"), [#181177](https://github.com/pytorch/pytorch/issues/181177)
(open, 2026-04-22; CPU flash_for_cpu), [PR
#184656](https://github.com/pytorch/pytorch/pull/184656) (open 2026-05-21,
CPU-compile-only fix), [#148988](https://github.com/pytorch/pytorch/issues/148988)
(closed; nested-tensor-only, tangential).
PyTorch — source (main @ b9ce263): [native_functions.yaml
L15266–15306](https://github.com/pytorch/pytorch/blob/b9ce26335c42fc249582d3ad8140348655058644/aten/src/ATen/native/native_functions.yaml#L15266),
[derivatives.yaml
L2931–2957](https://github.com/pytorch/pytorch/blob/b9ce26335c42fc249582d3ad8140348655058644/tools/autograd/derivatives.yaml#L2931),
[attention.cpp
L894](https://github.com/pytorch/pytorch/blob/b9ce26335c42fc249582d3ad8140348655058644/aten/src/ATen/native/transformers/attention.cpp#L894),
[backend
registry](https://github.com/pytorch/pytorch/blob/b9ce26335c42/torch/nn/attention/__init__.py#L73).
PyTorch — docs: [SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
[sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html),
[backends](https://pytorch.org/docs/stable/backends.html);
[releases 2.7.1–2.13.0](https://github.com/pytorch/pytorch/releases).
FlashAttention: [issue #699](https://github.com/Dao-AILab/flash-attention/issues/699)
(closed; tridao declines second order), [issue
#1672](https://github.com/Dao-AILab/flash-attention/issues/1672) (JVP, open),
[issue #2071](https://github.com/Dao-AILab/flash-attention/issues/2071)
(torch.func.grad, open),
[flash_attn_interface.py](https://github.com/Dao-AILab/flash-attention/blob/a369df707e19/flash_attn/flash_attn_interface.py),
[README](https://github.com/Dao-AILab/flash-attention/blob/a369df707e19/README.md).
xFormers: [fmha/__init__.py
v0.0.28](https://github.com/facebookresearch/xformers/blob/v0.0.28/xformers/ops/fmha/__init__.py)
(`once_differentiable`). Triton: [06-fused-attention.py](https://github.com/triton-lang/triton/blob/23689d2aa356/python/tutorials/06-fused-attention.py).
JAX: [jax.nn.dot_product_attention](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html).
IRED: [arXiv:2406.11179](https://arxiv.org/abs/2406.11179) (Du, Mao,
Tenenbaum; ICML 2024); code
[yilundu/ired_code_release](https://github.com/yilundu/ired_code_release):
`create_graph=True` at
[models.py L808](https://github.com/yilundu/ired_code_release/blob/main/models.py#L808),
explicit attention at
[diffusion_lib/transformer.py L51–58](https://github.com/yilundu/ired_code_release/blob/main/diffusion_lib/transformer.py#L51);
predecessor IREM [arXiv:2206.15448](https://arxiv.org/abs/2206.15448). EBT:
[arXiv:2507.02092](https://arxiv.org/abs/2507.02092) ("this loss is
backpropagated through the entire optimization process, requiring second-order
derivatives … via Hessian-vector products"), code
[alexiglad/EBT](https://github.com/alexiglad/EBT) (torch==2.4.0 pinned;
`create_graph=learning` at
[model/nlp/ebt.py L149](https://github.com/alexiglad/EBT/blob/main/model/nlp/ebt.py#L149)).
MAML: [arXiv:1703.03400](https://arxiv.org/abs/1703.03400) ("gradient through
a gradient"; FOMAML ~33 % faster, near-equal accuracy). DARTS:
[arXiv:1806.09055](https://arxiv.org/abs/1806.09055) (first-order
approximation tradeoff).
