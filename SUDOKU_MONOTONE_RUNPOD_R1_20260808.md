# Sudoku monotone ladder R1: matched learning panel

Date: 2026-08-08. Status: binding pre-launch contract.

The corrected R0 throughput gate at source commit
`777fe07eea00799a30483980da4c88efc1dbb7a0` completed and passed artifact
closure for all three monotone arms. At update 200, fixed AR and random
insertion sustained about 16.9 and 16.4 updates/s (1,081 and 1,052 examples/s),
while learned insertion sustained about 4.85 updates/s (310 examples/s). Peak
allocated CUDA memory was 2.23 GB for M0/M1 and 4.80 GB for M2. All three had
zero exact solves among the first 64 held-out puzzles, as expected for a health
smoke. The learned arm emitted finite ELBO, RLOO, posterior, digit, and policy
cell signals. These measurements authorize a matched short learning panel; no
R0 solve-rate result promotes an arm.

## Frozen arms and inputs

Run one independent 1xA40 pod for each arm: `fixed_ar`,
`random_insertion`, and `learned_insertion`. Preserve the R0 architecture,
authenticated 100-puzzle training set, immutable 10,000-puzzle test set,
optimizer, seed 42, batch size 64, BF16 mode, GPU-resident data path, M=2 RLOO,
and learned-posterior learning-rate ratio 0.01. Start every arm from fresh
weights; do not resume or select a checkpoint.

Each arm receives exactly 10,000 optimizer updates. Write training telemetry
every 100 updates and evaluate the same first 256 held-out puzzles every 1,000
updates. Save a full checkpoint at update 10,000. The time ceiling is 2,700
training seconds with a 2,760-second process wrapper.

## Interpretation

The primary endpoint remains exact valid Sudoku solve rate consistent with all
givens. Also compare digit NLL/accuracy, learned-order ELBO components,
throughput, peak GPU memory, and the immutable per-puzzle curves. Update count
is the matched statistical axis; wall time and examples/s expose the learned
ordering mechanism's additional compute cost.

This is a learning-signal panel, not the final paper comparison. A zero solve
rate at 10,000 updates does not establish that an arm is incapable. Continue
only after determining whether its intended losses learn, whether held-out
validity moves, and whether the next budget is justified.

## Capacity and closure

The owner explicitly authorized additional RunPod capacity. Use three ordinary
1xA40 Secure Cloud pods at the observed $0.44/hour price. Expected aggregate
cost is below $0.50; the hard aggregate ceiling implied by three 60-minute pod
lifetimes is $1.32. Each pod has no queued follow-up and must terminate only
after its completion or failure artifacts are synced, hash-verified locally,
and recorded in the fleet ledger.
