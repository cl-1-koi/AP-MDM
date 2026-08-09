# Sudoku Visible-Component Gate (VC-1)

Date: 2026-08-09 UTC

## Question

Did variable-length AO-IP solve the hard star graphs more readily than Sudoku
because the star answer consists of short, visible components that only need to
be retrieved and reordered, while Sudoku must infer missing information through
global constraints?

## Matched intervention

Train two fresh `random_insertion` Sudoku policies with identical authenticated
data, architecture, optimizer, seed, online prefix sampling, update count, and
search-free decoding. The only changed variable is `condition_mode`.

1. `puzzle_only`: the normal puzzle state. Missing digits are not exposed.
2. `transposed_solution_hint`: the complete solution is intentionally exposed
   as `COLOR_1..COLOR_9` on the transposed board. To predict cell `(r,c)`, the
   policy must retrieve the disjointly encoded digit at `(c,r)` and translate
   its token class. This is an oracle retrieval/reordering control, not a Sudoku
   solver and not a claim of train/test cleanliness with respect to solutions.

The value channel remains the ordinary partial puzzle in both arms. Both arms
fill a uniformly sampled empty cell during training and use the same seeded
random cell order at evaluation.

## Frozen settings

- Authenticated upstream Sudoku train/test arrays and existing zero-overlap gate
- Arm: `random_insertion`
- Seed: 42
- Policy: released spatial DDiT trunk, width 256, 6 blocks, 4 heads
- Batch size: 64
- Updates: 78,100 (4,998,400 puzzle presentations, matching the star panel)
- AdamW and checked-in paper config: LR 1e-4, 250-step warmup
- BF16 Transformer compute
- Evaluation: 256 held-out puzzles every 5,000 updates and at update 78,100
- Search-free greedy digits; seeded random empty-cell selection
- Checkpoints every 5,000 updates plus the terminal update

Hardware may differ between arms but changes only wall-clock throughput, not the
sealed training or evaluation semantics.

## Predictions

- Visible succeeds, puzzle-only fails: supports the visible-component/inference
  distinction.
- Both succeed at the matched budget: the earlier Sudoku failure was primarily
  undertraining, not a task-mechanism distinction.
- Both fail: refutes the claim that this architecture/objective can readily use
  exposed answer components; inspect representation or optimization first.
- Puzzle-only succeeds but visible fails: the hint encoding interferes with the
  model and is not a valid control; redesign before interpreting.

Output length and constraint depth are not isolated by VC-1. If the oracle arm
succeeds, VC-2 will compare normal and length-expanded star graphs while holding
answer visibility fixed.

## Health and failure gates

- Before full launch: relevant tests pass, source commit is pushed, pod checkout
  is clean and exact, CUDA is visible, manifest seal verifies, and a 100-step
  smoke emits finite loss/gradient/throughput telemetry.
- Fail fast on non-finite loss or gradients, missing evaluation rows, manifest
  drift, checkpoint mismatch, or absent GPU telemetry.
- Do not stop merely because held-out exact remains zero before the declared
  endpoint; the star experiments transitioned only after roughly 40,000 steps.

## Artifact closure

Preserve the sealed manifest, command records, complete telemetry, every
scheduled checkpoint retained by the run contract, evaluation rows and hashes,
train summary, console log, source/deployment record, completion/failure record,
and a remote/local byte-for-byte hash audit. No pod may be released before its
arm passes closure.
