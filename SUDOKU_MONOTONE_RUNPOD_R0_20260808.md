# Sudoku monotone ladder R0: parallel RunPod throughput gate

Date: 2026-08-08. Status: binding pre-launch amendment.

The owner explicitly authorized additional RunPod capacity after the original
local-only contract was written. This amendment authorizes only the first
three-arm throughput and signal-emission gate in `SUDOKU_MONOTONE_LADDER_20260808.md`.
It does not authorize the matched learning panel or production continuation.

## Arms and invariant inputs

Run one independent pod for each implemented monotone arm:

- `fixed_ar` (M0);
- `random_insertion` (M1);
- `learned_insertion` (M2).

All arms use the authenticated 100 training puzzles and untouched 10,000 test
puzzles, the paper-declared six-block/four-head/width-256 policy trunk, seed 42,
AdamW settings already frozen in `sudoku_paper`, BF16, and fully GPU-resident
prefix construction. No solver-generated transition store is consumed.

Each smoke is fixed at:

- 200 optimizer updates;
- batch size 64;
- telemetry every 20 updates;
- one checkpoint and held-out evaluation at update 200;
- the first 64 held-out puzzles for smoke diagnostics only;
- no resume, no checkpoint selection, and a 600-second wall-time ceiling.

Learned insertion retains `M=2` RLOO and posterior/policy learning-rate ratio
0.01. Random insertion retains seed 42. These are not tuned independently.

## Hardware and budget

Use three separately provisioned 1xA40 RunPod Secure Cloud pods with 9 allocated
vCPUs, the same PyTorch 2.4/CUDA 12.4 image, 80 GB ephemeral container disks,
and no network volume. Expected price is $0.44/hour each. The strict per-pod
ceiling is 15 minutes including setup and closure; aggregate ceiling is $0.33.

## Gates

An arm passes R0 health only if:

1. source commit and train/test hashes authenticate and the manifest explicitly
   records owner-authorized paid capacity;
2. update 200 is reached within 600 training seconds with finite loss, gradient,
   throughput, and GPU-memory telemetry;
3. the intended arm-specific metrics emit (including ELBO/RLOO/posterior and
   policy-cell statistics for learned insertion);
4. the step-200 full checkpoint reloads with the exact arm, manifest seal,
   architecture, optimizer, scheduler, generator, and RNG state;
5. 64 immutable evaluation rows and their SHA-256 are present.

Solve rate at update 200 is descriptive and cannot promote an arm. After all
three closures, compare examples/s, updates/s, wall time, peak memory, loss
movement, and intended-signal health. Only then freeze a matched short panel.

Each ordinary pod terminates after its own locally verified artifact closure.
No follow-up is queued on these pods.
