# Sudoku order/exposure gate (2026-08-11)

## Question

The GuacaMol reconstruction established that insertion training can learn a
useful order on a forgiving sequence domain.  The S4 Sudoku runs, by contrast,
interpolate their 64 training boards but generalize poorly.  This gate asks
whether the Sudoku failure is primarily:

1. **ordering** — the digit predictor works if cells are exposed in a useful
   order;
2. **exposure/compounding** — short prefixes work but self-generated errors
   accumulate with rollout length; or
3. **representation/content** — even a rule-coded, target-blind ordering does
   not rescue the next-digit predictions.

This is a diagnostic gate, not a claim that a rule-coded Sudoku heuristic is a
scalable solver.

## Frozen contract

- Data: the seed-42 S4 split from `repro.small_grokking.frozen_splits` (64
  train, 128 held out, six givens and ten missing cells).
- Policies: the completed 1,000,000-update FO-ARM, AO-ARM, and LO-ARM
  checkpoints, evaluated without changing weights.
- State ABI: unchanged keyed, shuffled target-hint channel used during the
  original S4 runs.  Consequently this panel measures keyed retrieval and
  sequencing as well as Sudoku consistency; it is not a pure deduction test.
- Rollout orders:
  - `fixed`: lowest-index empty cell;
  - `random`: uniform empty cell, with recorded seeds;
  - `learned`: policy `cell_logits`;
  - `oracle_mrv`: fewest legal Sudoku candidates, row-major tie-break.  It uses
    only the current partial board, never the held-out answer.
- Horizons: 1, 2, 4, 8, and full (10) policy decisions.
- Replication: paired seeds use the same shuffled hint realization across all
  order arms.  The default panel uses eight repeats per held-out board.

## Outcomes

At every horizon report:

- target-prefix survival (all decisions so far equal the unique completion);
- per-decision target accuracy;
- partial Sudoku consistency (no duplicate nonzero digits);
- exact solver continuability of the partial board;
- exact and valid completion rate at the full horizon.

The primary comparisons are the native diagonal (FO/fixed, AO/random,
LO/learned), plus `oracle_mrv` applied to every frozen digit predictor.  The
complete policy-by-order matrix is retained to expose architecture/head
confounds rather than silently folding them into an “ordering” result.

## Interpretation gates

- **Oracle rescue:** oracle-prefix survival materially exceeds the policy's
  native order at matched horizon.  Next experiment: train an oracle-order
  control and compare it with learned ordering.
- **Horizon cliff:** horizons 1–2 are strong but 4–full collapse under all
  orders.  Next experiment: scheduled self-generated-prefix exposure and a
  reversible remask/backtrack arm.
- **No short-horizon signal:** oracle horizon 1 is weak on held-out boards.
  Stop order engineering; the content representation/objective is the active
  failure.
- **Late optimization plateau:** only after establishing one of the above,
  continue the implicated arm with a sealed shrink-and-perturb event.  Do not
  reinterpret a continuation as a fresh seed.

No new paid capacity is authorized by this gate.  The frozen evaluation and
first smoke tests run on an available local GPU.
