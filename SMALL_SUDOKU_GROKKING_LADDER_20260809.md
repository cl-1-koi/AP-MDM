# Small-Sudoku Grokking and Generation-Family Ladder

Date: 2026-08-09 UTC

## North-star question

Can Transformer game-state generators trained from terminal examples progress
from interpolation to a reusable algorithm, and does the construction process
change whether or when that delayed generalization occurs?

The trigger result is Sudoku VC-1c. A 6.0M-parameter keyed shuffled-answer
policy trained on only 100 unique 9x9 puzzles reached near-perfect
teacher-forced training accuracy but remained at 0/256 exact held-out solves
through 78,100 updates. This is compatible with grokking, but it is also
compatible with memorizing the 100 solutions while ignoring the content keys.
The ladder must distinguish those explanations before extending training.

## Required mechanism registry

Every mechanism below is a required arm. A stage may gate later expensive arms,
but no family may silently disappear from the overall program.

| Label | Expansion | Native process |
|---|---|---|
| FO-ARM | Fixed-Order Autoregressive Model | Fixed row-major construction |
| AO-ARM | Any-Order Autoregressive Model | Random order on a fixed canvas |
| LO-ARM | Learning-Order Autoregressive Model | Learned order on a fixed canvas |
| AO-IP | Any-Order Insertion Process | Variable-length random insertion and termination |
| IP | Insertion Process | Variable-length learned insertion and termination |
| MDM | Masked Diffusion Model | Fixed-canvas mask/denoise process |
| AP-MDM | Any-Process Masked Diffusion Model | Unmask, remask, insert/expand, and delete/contract |
| IRED | Iterative Reasoning through Energy Diffusion | Separate energy-minimization comparator |

The existing Sudoku names map as follows: `fixed_ar` is FO-ARM;
`random_insertion` on a known 81-cell canvas is AO-ARM-like, not AO-IP;
`learned_insertion` on that canvas is LO-ARM-like, not paper-native IP. The
existing star implementation contains actual variable-length AO-IP and IP.

## D0: decisive audit of the completed 9x9 keyed model

Before calling the VC-1c gap grokking, evaluate its frozen checkpoints on:

1. seen training puzzles with fresh unseen hint shuffles;
2. held-out puzzles with fresh hint shuffles;
3. shuffled hints with their content keys removed;
4. no hints;
5. counterfactual payloads in which every visible keyed digit is deliberately
   changed away from the Sudoku solution.

Record accuracy against both the supplied counterfactual payload and the
original Sudoku solution. A reusable associative-lookup circuit follows the
payload. A puzzle memorizer follows the original solution or fails when keys
are load-bearing. Full-rollout train and held-out exact rates remain separate
from the one-forward blank-cell diagnostics.

### D0 result

The frozen update-78,100 checkpoint decisively follows the memorization
explanation. On seen training puzzles with fresh hint shuffles it predicted the
normal solution on 99.879% of blank cells. When every supplied payload digit
was changed, it followed the counterfactual payload on only 0.017% of blanks
but still predicted the original solution on 99.914%; 96/100 puzzles matched
the memorized solution completely in one forward pass. Train rollout validity
was 70/100.

On held-out puzzles, keyed normal-solution accuracy was 27.825%, versus 22.525%
with shuffled hints but no keys and 22.319% with no hints. Thus keys provide a
small real signal, but the counterfactual arm was near chance at 9.321% payload
accuracy while remaining at 27.420% solution accuracy. The model did not learn
a reusable associative-copy algorithm by the matched endpoint. D0 artifacts
are byte-for-byte verified under
`/home/ubuntu/apmdm-official-data/sudoku-grokking/d0/d0-keyed-5e968f5`.

S4 must therefore include two distinct outcomes: finite-puzzle train/test
generalization for grokking, and counterfactual payload following for mechanism
identification. A held-out Sudoku improvement without payload following cannot
be attributed to content-addressed retrieval.

## S4: 4x4 mechanism and grokking phase diagram

Use 2x2 boxes. Enumerate the 288 valid completed 4x4 grids, freeze disjoint
solution splits (64 train, 128 held out, 96 unused), and generate uniquely
solvable clue masks with recorded seeds. Because the solution universe is
small, the primary S4 generalization outcome is an arbitrary keyed-payload
task over 16 cells; Sudoku validity is a secondary sanity check.

Initial architecture: two Transformer blocks, width 64, four heads, dropout
0.1. Run each required mechanism with AdamW at 1e-4, betas (0.9, 0.999),
epsilon 1e-8, gradient clip 1.0, and weight decay in {0, 0.01}. Use identical
logical train/test payloads, seed 42, and family-native state/action encodings.

Train past interpolation with logarithmic evaluations and checkpoints at
1k, 3k, 10k, 30k, 100k, 300k, and 1M updates. Record:

- train and held-out token/cell accuracy;
- complete-instance accuracy;
- counterfactual payload-following accuracy;
- hint/key ablation deltas;
- family-specific order, termination, edit, diffusion, or energy telemetry;
- examples/s, updates/s, GPU memory, wall time, and optimizer state.

A grokking event requires a delayed held-out or counterfactual jump after
training interpolation. Loss reduction alone is not a pass.

## S6: 6x6 Sudoku rule generalization

Use 2x3 boxes and disjoint completed-grid splits large enough to prevent the
solution-space saturation problem of 4x4. Start with four blocks, width 128,
four heads. Carry every mechanism through a bounded smoke; extend all healthy
arms to the first log-spaced gate and extend the full curve for arms that emit
their intended signals. Primary outcomes are exact valid Sudoku solve rate,
counterfactual key use, and the timing of train/test separation and collapse.

## S9: 9x9 continuation

Only after D0/S4 identify whether keys are used, declare a continuation of the
existing 9x9 checkpoint with its optimizer state intact. Do not alter its
sealed 78,100-update record. Continue as a new lineage to 100k, 300k, and 1M
updates, retaining the current AdamW settings, while running the same
counterfactual diagnostics. Estimated current-hardware time for 1M updates is
about seven hours.

## Comparability and interpretation

- Hold logical examples, optimizer, seed, evaluation rows, and small-model
  capacity fixed wherever a mechanism permits.
- Report family-native neural work and wall/GPU time. AP-MDM/IRED and learned
  posterior arms are not compute-identical to FO/AO arms.
- Fixed-canvas AO-ARM/LO-ARM results cannot be labeled AO-IP/IP.
- AP-MDM may consume edit trajectories; that additional supervision must be
  measured and disclosed rather than treated as terminal-pair equivalence.
- IRED is a separate energy comparator, not an insertion variant.
- Do not select checkpoints or stop horizons using the held-out verdict split.

## Execution and fleet policy

1. Run D0 immediately on the retained L40S after VC-1c artifact closure.
2. Implement and locally validate the common S4 data/evaluation ABI and every
   family adapter before provisioning the panel.
3. Benchmark one smoke per family. If each process saturates a GPU while using
   little CPU/VRAM, prefer one four-GPU pod with four independent queues to
   amortize setup. If family runtimes or dependencies conflict, use separate
   one-GPU pods.
4. Never mix arms in one process or share optimizer state. Seal each arm's
   commit, data hashes, settings, checkpoints, and artifact closure.
5. Keep a declared next arm queued on every retained pod. Terminate only after
   verified closure when no compatible follow-up is ready.
