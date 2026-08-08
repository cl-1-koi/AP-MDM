# Sudoku monotone generation ladder

Date: 2026-08-08. Status: implementation complete; focused CPU tests pass;
GPU throughput gates pending release of an approved A10. This document does
not authorize new paid capacity.

## Question

Does learned non-monotone construction order provide an intermediate and more
trainable reasoning mechanism between fixed-order autoregression and full
AP-MDM remask/insert/delete backtracking?

"Intermediate" is a hypothesis, not a promotion criterion. A monotone model
may outperform the more expressive editing model at finite compute because it
has fewer operations to learn, or underperform fixed-order AR because its
latent order introduces a harder optimization problem.

## Shared contract

All arms use:

- the authenticated official 100-puzzle training array and disjoint 10,000
  puzzle test array;
- terminal puzzle/solution pairs only--no solver-generated backtracking
  transition store;
- the paper-declared six-block, four-head, width-256 DIT trunk and the released
  runtime vocabulary size of 34;
- the same 324-token spatial board representation and the same digit head;
- terminal exact Sudoku validity plus consistency with givens as the primary
  held-out outcome;
- no MCTS, external solver queries, deletion, replacement, or remasking at
  inference.

The model state is a canonical 9x9 grid. An insertion action selects an empty
cell and writes one digit. Written digits are irreversible.

## Arms

### M0: fixed-order AR

The next cell is the first empty cell in row-major order. The network predicts
only its digit. Prefix length is sampled uniformly during training.

### M1: random-order insertion

The next cell is selected uniformly from empty cells. The network predicts its
digit. This isolates the cost or benefit of exposing arbitrary construction
orders without learning an order policy.

### M2: learned-order insertion

The policy predicts both the next empty cell and its digit. A smaller
training-only Transformer observes the completed target and clue mask and
parameterizes a Plackett--Luce distribution over permutations of non-given
cells. Training implements the fixed-canvas specialization of Zhang et al.,
*Variational Learning for Insertion-based Generation* (arXiv:2606.02133v3):

- permutations sampled with Gumbel-Top-k;
- two posterior samples (`M=2`);
- a uniformly sampled construction prefix;
- exact expectation over the next remaining cell;
- REINFORCE Leave-One-Out with the paper's stop-gradient surrogate;
- posterior learning rate 1/100 of the policy learning rate by default.

Because Sudoku's 81-cell canvas has known length, termination is deterministic
when no empty cells remain. This arm tests learned order, not the paper's
variable-length stopping result.

### M3: full AP-MDM comparison

The official paper-declared AP-MDM arm may remask, insert, and delete. It is a
separate architecture/head and data-generation process. Its results are shown
beside M0-M2 but must not be described as a perfectly compute-identical
ablation.

## Measurements

At every checkpoint record:

- exact held-out solve/valid/consistent rates and immutable per-puzzle rows;
- digit NLL and next-digit accuracy;
- learned-order ELBO, RLOO surrogate, posterior prefix log probability, policy
  cell entropy, posterior next-cell entropy, and expected digit NLL;
- optimizer step, examples/s, steps/s, wall time, peak GPU memory, and
  checkpoint hash;
- for random order, the evaluation RNG seed and later a multi-seed interval.

Comparisons are reported both per optimizer step and per wall-clock/GPU FLOP.
The learned arm performs more neural work per step, so an update-count-only
comparison is insufficient.

## Execution gates

1. Focused CPU correctness tests for state encoding, monotonicity, loss
   gradients, RLOO gradients, exact evaluator behavior, manifest sealing, and
   bit-restorable checkpoints.
2. One bounded local-A10 throughput smoke per arm after the active AP-MDM run
   releases the GPU. Freeze batch sizes only after measuring memory and
   examples/s.
3. A matched short panel with identical training instances, seeds, held-out
   rows, and declared optimizer-step budget.
4. Continue only arms that emit their intended signals and show a nonzero
   learning curve. Replicate any apparent ordering advantage across seeds.

RunPod is not required for implementation or the first local smokes. If the
measured three-arm, multi-seed panel would exceed roughly 12 local A10-hours or
delay feedback by more than a day, prepare a costed parallel RunPod proposal.
Do not create or resume paid pods without explicit owner authorization.

## GPU/CPU data path

The monotone trainer copies all 100 training puzzles and solutions to the GPU
once. Batch indices, prefix masks, Gumbel permutations, partial-grid encoding,
the policy/posterior forwards, losses, and optimizer updates then remain on the
GPU. Only periodic scalar telemetry/checkpoints return to CPU.

The official AP-MDM trainer is different: its approximately 11 GiB transition
cache remains on disk/CPU, batches are collated into CPU tensors, and Lightning
copies each batch to the GPU. Forward, loss, backward, and optimizer work are
GPU operations. That transfer is a few megabytes per batch and was not the
observed bottleneck; the released epoch-boundary loader replacement was.

