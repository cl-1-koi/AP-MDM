# Sudoku Discriminating Gates D1/D2

Date: 2026-08-10 UTC

## Decision

The completed S4 panel and the cross-paper review justify two independent,
bounded follow-ups. D1 tests whether the S4 keyed channel can learn retrieval
when the Sudoku-solution memorization shortcut is removed. D2 tests whether the
official AP-MDM 100k checkpoint generalizes its supervised transition operator
and whether the reconstructed fixed-point termination rule is independently
responsible for rollout failure.

Neither gate changes or extends an existing arm. Both are diagnostics, not
paper-replication claims.

## D1: random-payload keyed-retrieval control

- Code path: `repro.small_grokking`, FO-ARM.
- Base data: frozen S4 seed-42 64/128 puzzle split.
- Target mode: `random_payload`.
- Each of the ten blank-cell targets is sampled independently and uniformly
  from digits 1..4 under recorded seed `2,000,045`; the six immutable givens
  retain their puzzle digits. The keyed hint channel exposes every target.
- Architecture: 2 blocks, width 64, 4 heads; batch 256.
- AdamW 1e-4, weight decay 0.01, warmup 250, gradient clip 1.0, bf16.
- Maximum 100,000 updates; evaluations/checkpoints at 1k, 3k, 10k, 30k,
  100k; seed 42.
- Primary metric: one-forward held-out `payload_exact_rate` across all ten
  visible keyed blank-cell payloads; free-running target exactness is secondary.
  Sudoku validity is deliberately not a success metric.

Predeclared interpretation:

- `payload_exact_rate >= 0.50`: retrieval is learnable; the original S4 task
  let Sudoku memorization dominate. Stop at the first gate meeting this bound.
- `payload_exact_rate < 0.05` at 100k after train interpolation: the current
  representation/objective cannot readily learn content-addressed retrieval
  even after removing the Sudoku shortcut.
- Anything else, or failure to interpolate train, is inconclusive and does not
  authorize tuning.

## D2: held-out teacher-forced AP-MDM transition audit

- Frozen input: official author-stack `sudoku_paper` checkpoint at 100,000
  steps and its byte-verified vocabulary.
- Data: a seed-42 permutation selects 1,000 puzzles from the authenticated
  10,000-puzzle held-out array. The official solver and sample generator
  generate their transition labels; no training data or weights change.
- Evaluation: teacher-forced unmask-token accuracy, remask precision/recall,
  insert/delete false positives, complete next-transition exactness, and
  per-operation breakdown at threshold 0.5.
- Termination proxy: fixed-point stability of the official solver trajectory's
  final encoded state.
  This diagnoses our reconstructed sampler only; the released generator has no
  explicit halt target, so it cannot establish the paper's unstated stopping
  semantics.
- Batch 512, bf16, no search, no threshold sweep.

Predeclared interpretation:

- Unmask accuracy at least 99%, transition exactness at least 95%, and terminal
  stability below 50%: the learned transition operator transfers, while
  composition/termination is the binding failure of our reconstructed rollout.
- Unmask accuracy below 95% or transition exactness below 90%: the supervised
  operator itself does not generalize; continuing the same 100-puzzle training
  recipe is not warranted merely to repair halting.
- Intermediate results are mixed/inconclusive. No threshold selection or
  checkpoint selection follows from this gate.

## Execution and closure

Run a one-puzzle D2 smoke and a 100-step D1 smoke before the sealed jobs. Use
the two currently idle GPUs on retained pod `e1cr8gz83bhkpu`, one gate per GPU,
without disturbing the remaining S4 arms on the other pods. Preserve manifests,
all scheduled D1 checkpoints/evaluation rows, D2 per-puzzle rows, summaries,
logs, completion records, and SHA-256 closure reports locally before any pod
handoff or termination.
