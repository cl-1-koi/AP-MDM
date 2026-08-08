# Binding Spec: AP-MDM Sudoku Replication

Date: 2026-08-08

## Objective

Make the official AP-MDM repository capable of reproducing, measuring, and auditing the Sudoku result in arXiv:2510.06190v2, then prepare a frozen experiment handoff. The paper reports 99.28% Sudoku accuracy from 100 training puzzles. This unit must preserve the distinction between 100 source puzzles and the much larger solver-generated supervised transition set.

The immediate implementation unit ends after tested code, deterministic data generation, a CPU/GPU smoke, and a frozen production manifest. It must not launch the full production training run until an independent review has passed. A clean negative or precise blocker is a valid deliverable.

## Authoritative inputs

- Paper PDF: `/home/ubuntu/papers/2510.06190/2510.06190v2.pdf`, SHA-256 `efec4ac49e7359659426b48e21c5e6b0076eae66bbbe89eee964572ca14eba9c`.
- Paper source: `/home/ubuntu/papers/2510.06190/2510.06190v2-source.tar`, SHA-256 `891ef81fb986e31ecafb7803ef5eed3ed78d7f6a4cbd9dd186caedfc94ec085c`.
- Extracted paper source: `/home/ubuntu/papers/2510.06190/source`.
- Official code repository: `https://github.com/chr26195/AP-MDM`, commit `836b79b30286301ee1cca975db7e8468bc2a653e`.
- Training puzzles: `dataset/sudoku/sudoku-100.npy`, shape `(100, 325)`, SHA-256 `a3e262870715fe4d3371a2682fd0656baf5af76e88ccf5f31ff83d5b2ab676e0`.
- Test puzzles: `dataset/sudoku/sudoku-test-10k.npy`, shape `(10000, 325)`, SHA-256 `36f57185fbdd7cac097c8e365845b917c0e7f9bcb7b8de212b7424d8b99eeb7c`.
- Exact full-row overlap between the checked-in train and test arrays was independently measured as zero; recheck and authenticate it in code.

Read the paper and relevant source in full enough to reconstruct every Sudoku claim and method detail, especially `iclr2026_conference.tex`, `method.tex`, `training.tex` or its equivalent, `exp.tex`, and `examples.tex`. Read all official Sudoku dataset, model, loss, sampling, training, and configuration code before editing.

## Facts and discrepancies that must not be hidden

1. The paper reports a 6-layer, 4-head, width-256 encoder Transformer, FFN width 1024, maximum length 400, vocabulary 31, and approximately 1.2M parameters. The checked-in `train/configs/sudoku.yaml` instead declares 4 layers, 2 heads, and width 128. Do not overwrite or reinterpret the historical config. Add a separately named paper-faithful configuration and report exact parameter counts for both.
2. `examples.tex` says both 1,421.3 state transitions per puzzle and, for the actual 100 hard puzzles, 25,022.6 training tuples per puzzle. Measure the checked-in generator's actual count, distribution, and total. Never call the expanded tuples independent puzzles or independent samples.
3. The prose state-representation description says 32 tokens while `exp.tex`, the result table, and checked-in config say vocabulary 31. Authenticate the generated vocabulary and explain the discrepancy; do not change vocabulary merely to force agreement.
4. The checked-in training script and config contain stale absolute `/workspace/...` paths. The reproduction must use portable repository/data-root arguments without silently changing scientific settings.
5. The ARM/AO-MDM comparison numbers are imported from Kim et al. rather than trained by this repository. The first replication tests the AP-MDM result only and must not claim a controlled baseline comparison.

## Scientific contract

### Fixed primary arm

- Model: AP-MDM encoder Transformer matching the paper's Sudoku architecture: 6 blocks, 4 heads, hidden size 256, FFN ratio 4/width 1024, RoPE, maximum length 400, authenticated vocabulary size, and four output functions: token/unmask plus binary remask, insert/expansion, and delete/contraction.
- No timestep embedding/time conditioning during supervised Sudoku training.
- Data: exactly the 100 authenticated source puzzles, expanded only by the checked-in backtracking trajectory generator after its outputs are validated. The official 10,000-puzzle array is held out for evaluation only.
- Optimizer: AdamW, learning rate `1e-4`, betas `(0.9, 0.999)`, weight decay `0.01`, global batch size 256, constant schedule with 250 warmup steps, bf16, gradient clipping 1.0, and all operation-loss weights 1.0.
- Training ceiling: 1,000,000 optimizer steps or a predeclared convergence stop. Do not tune outside a separately declared diagnostic sweep.
- Sampling/evaluation: reconstruct the paper/official AP-MDM iterative generation exactly, including threshold rules and maximum steps. Freeze all thresholds before production evaluation. Do not select thresholds on the 10,000-puzzle verdict set.
- Seeds: seed 42 is the paper/config lineage. All Python, NumPy, Torch, data-loader, and sampling seeds must be recorded. A second seed is a follow-up, not a substitute for the primary replication.

### Evaluation semantics

For every held-out puzzle record at least:

- exact equality with the supplied ground-truth grid (primary paper-comparison metric unless evidence proves a different original definition);
- validity under Sudoku row/column/subgrid constraints;
- consistency with givens;
- solved/terminated, timeout, invalid token/shape, and maximum generation steps;
- number and type of remask/insert/delete/unmask operations and total forward passes.

Persist per-puzzle rows and aggregate them deterministically. Report a binomial 95% interval for exact accuracy and validity. Never score the symbolic backtracking data generator itself as model accuracy.

The paper target is 99.28% on 10,000 puzzles. Predeclare:

- **strong replication:** exact accuracy at least 98.78% (within 0.50 percentage points of the report), with no leakage or validity failure;
- **partial replication:** 97.00% to below 98.78%;
- **failure to replicate:** below 97.00% after the declared convergence/step ceiling;
- **invalid/inconclusive:** any provenance, leakage, evaluator, architecture, or execution-contract failure.

Always report the exact estimate and interval; the labels do not replace the measurement.

## Implementation requirements

1. Add a paper-faithful Sudoku config without modifying the historical config's meaning.
2. Add a portable, resumable entry point for dataset generation, training, checkpointing, and evaluation. All large datasets, caches, checkpoints, raw rows, and telemetry stay outside Git.
3. Add a sealed JSON manifest binding paper/code/data hashes, architecture signature and parameter count, vocabulary, generator counts, split hashes, optimizer, losses, seeds, thresholds, checkpoint/evaluation cadence, environment versions, hardware, output paths, and compute ceilings.
4. Add an evaluator that can score official held-out puzzles from a checkpoint and emit immutable per-puzzle rows plus aggregate statistics. It must reject train/test overlap and mismatched manifests/checkpoints.
5. Add telemetry adequate to diagnose operation-head collapse, non-emission, looping, invalid sequences, loss components, throughput, memory, checkpoint progress, and accuracy curves.
6. Provide a fail-closed supervisor for the existing local A10 and optional existing `a10-220` only. It must never create/resume RunPod or other paid capacity. The initial smoke ceiling is 15 combined A10 minutes. The later production ceiling must be estimated from measured smoke throughput and frozen by independent review before launch.
7. Make installation reproducible in an isolated environment. If FlashAttention cannot be built on the available CUDA/Torch stack, a PyTorch SDPA fallback may be implemented only if numerical/shape behavior is tested and the change is recorded; do not imply identical kernels.
8. Preserve upstream attribution and keep the author remote separate from the `cl-1-koi/AP-MDM` fork.

## Mandatory tests and smoke gates

Add a real automated test suite. At minimum test:

- source/test data hashes, shapes, uniqueness, and zero overlap;
- Sudoku decoding, validity, givens consistency, exact scoring, and malformed-output rejection;
- deterministic trajectory generation on fixtures and all atomic operations (assign, branch, contradiction, backtrack, recovery);
- measured transition accounting by source puzzle without treating transitions as independent source examples;
- vocabulary identity and legal token roles;
- paper-faithful architecture fields, exact parameter count, four-head output shapes, and time-conditioning disabled;
- each loss/head receives finite nonzero gradients on a fixture containing the relevant operation;
- sampler state transitions, termination, maximum-step behavior, and invalid-operation rejection;
- manifest authentication, immutability, and checkpoint/config binding;
- deterministic evaluation aggregation and confidence interval;
- portable paths and import safety;
- historical config remains unchanged in meaning;
- SDPA/FlashAttention backend agreement within a declared tolerance when both are available.

Run all tests plus:

1. CPU dataset/evaluator fixtures.
2. A one-batch forward/backward optimizer smoke.
3. A tiny overfit smoke demonstrating that the model can learn all four declared operation targets.
4. At most 15 combined A10 minutes for a measured end-to-end data/train/checkpoint/sample/evaluate smoke. Authenticate GPU idleness first and return it to idle afterward.

The implementation gate passes only if the suite is green, all losses/operations emit, artifacts authenticate, the tiny overfit succeeds, the end-to-end evaluator processes unseen puzzles correctly, and a production cost estimate can be made. Do not launch the full run in this implementation unit.

## Deliverables and commit discipline

1. `SUDOKU_PAPER_CODE_AUDIT_20260808.md`: exact paper/code reconciliation and unresolved ambiguities.
2. Implementation, configurations, manifest tooling, supervisor, and tests.
3. `SUDOKU_REPRO_IMPLEMENTATION_REPORT_20260808.md`: test/smoke results, hashes, actual derived-transition counts, environment, measured SPS/steps per second, cost projection, and exact production handoff.

Commit promptly because the work is timed:

- one commit for the audit/contract corrections if needed;
- one focused commit for implementation and tests after green;
- one separate commit for smoke/results and frozen handoff.

Push to the `cl-1-koi/AP-MDM` fork and leave the worktree clean. Work autonomously without asking for confirmation. No tuning beyond the declared settings is authorized. A clean negative or precise blocker is a valid deliverable.
