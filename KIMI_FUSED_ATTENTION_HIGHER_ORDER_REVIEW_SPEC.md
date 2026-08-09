# Binding spec: fused-attention higher-order-gradient review

## Question

Determine whether second-order differentiation through fused scaled-dot-product
attention is a known limitation, and identify validated GPU workarounds suitable
for IRED-style training where model parameters are optimized through
`grad(energy, model_input, create_graph=True)`.

## Required evidence

Use primary sources wherever possible: official PyTorch documentation/source and
GitHub issues, the official FlashAttention and xFormers repositories, framework
or kernel documentation, and relevant papers with implementation repositories.
Record exact links, versions/dates, and distinguish CPU, CUDA math, memory-
efficient, FlashAttention, cuDNN, and Triton backends. Check whether first-order
input gradients work separately from the mixed second derivative needed to train
energy-network parameters.

Run a minimal local reproduction against the installed PyTorch build. If a CUDA
device is available without disrupting existing jobs, use a tiny allocation;
otherwise report CPU evidence and authoritative GPU evidence without claiming a
local GPU result.

## Questions the report must answer

1. Which PyTorch SDPA backends currently support the needed double backward?
2. Are failures backend-, dtype-, device-, or version-specific?
3. Can PyTorch backend selection force a correct math kernel on GPU?
4. Do FlashAttention/xFormers expose supported double-backward paths?
5. What workarounds exist: explicit attention, backend forcing, custom autograd,
   Hessian-vector/JVP reformulations, finite differences, implicit gradients, or
   alternative IRED objectives?
6. Which workaround preserves the IRED objective exactly, and what are its
   memory/speed costs for sequence length 80, width 64, four heads?
7. Recommend one immediate implementation and one later optimization, each with
   a falsifiable validation test.

## PASS/FAIL gates

PASS requires a committed `FUSED_ATTENTION_HIGHER_ORDER_REVIEW.md` containing:

- a source-linked support matrix;
- a minimal reproduction and observed output;
- an explicit distinction between double backward w.r.t. input and the mixed
  derivative into model parameters;
- a ranked workaround recommendation for the S4 IRED comparator;
- commands/tests that verify numerical agreement with explicit attention.

FAIL is a valid clean result if authoritative evidence or network access is
insufficient; list precisely what could not be established. A clean negative is
a valid deliverable.

## Non-goals

- Do not modify training code or launch paid compute.
- Do not reinterpret or simplify the IRED objective.
- Do not rely on secondary summaries when a primary source exists.
- Do not claim a kernel supports higher-order gradients merely because ordinary
  backward succeeds.

Run the existing relevant tests if any code is added, but implementation changes
are not expected. Commit the report promptly as one results commit. Work
autonomously without asking for confirmation.
