# AP-MDM paper-declared 100k execution amendment

Date: 2026-08-08. This is an execution amendment to the immutable manifest at
`/home/ubuntu/apmdm-official-data/runs/paper-declared-100k/manifest.json`.
It does not change the model, data, optimizer, loss weights, seed, validation
split, or target step.

## Preserved state

The first process trained from scratch and preserved a full Lightning
checkpoint at global step 9,000, including model, AdamW optimizer, scheduler,
and loop state. Its SHA-256 was recorded before every resume. Attempts that
failed before step 10,000 did not overwrite that checkpoint.

## Failure 1: collator lost at epoch reconstruction

At the first training-epoch boundary, Lightning reconstructed the custom
streaming loader as an ordinary DataLoader without the AP-MDM collation
function. The following batch contained Python lists and failed at
`input_ids.shape`. The original run did not capture stderr; a bounded resume
reproduced and captured the exact traceback.

Commit `fed871b` preserves the collation function and adds a regression test.

## Failure 2: random chunk thrashing

With collation preserved, the replacement loader used a random sampler over
2,477,235 transition indices. Each worker consequently reopened unrelated
10,000-sample pickle chunks. At the epoch boundary CPU workers remained busy
while GPU utilization fell to zero. The attempt was stopped before any new
checkpoint; the step-9,000 artifact remained authoritative.

Commit `6ca361c` keeps the already re-iterable sequential chunk loader intact
instead of replacing it. It retains chunk-local shuffling and prevents both
the list batch and random disk-thrashing paths.

## Current continuation contract

The corrected continuation:

- loads the unchanged step-9,000 full checkpoint;
- runs commit `6ca361c400d6ce2624f6a4fe51fa0e1fdcdd3050` from a clean worktree;
- retains the original scientific configuration and 100,000-step target;
- subtracts all prior attempt time from the original ten-hour wall ceiling;
- captures stdout/stderr durably;
- requires a final checkpoint whose `global_step` is exactly 100,000 before
  held-out evaluation can start;
- performs the declared 256-puzzle conditional evaluation only after a clean
  training exit and idle-GPU preflight.

The generic Lightning warning that the dataloader itself lacks a serialized
state remains provenance. The checkpoint restores the loop position and the
released module reconstructs the deterministic distributed sampler position;
the two aborted continuations may repeat up to the uncheckpointed 677-step
tail. This is less than one percent of the 100k target and must be disclosed in
the final result rather than silently described as bit-identical data order.
