# AP-MDM paper-100k verdict: RunPod migration contract

Date: 2026-08-08. Status: binding before local A10 release.

## Purpose

Move the in-flight, GPU-heavy and low-CPU AP-MDM Sudoku verdict evaluation off
the local A10 so that CPU-rich A10 hosts can be reassigned to FactionWar. This
is a deterministic restart of the evaluation only. Training has already
completed at exactly 100,000 optimizer steps and is not rerun.

The local attempt began at `2026-08-08T22:00:43Z`. It emitted no result rows or
summary before migration, so there is no partial verdict to merge or resume.

## Immutable inputs and command

- Source branch: `codex/official-sudoku-arm-20260808` in `cl-1-koi/AP-MDM`.
- Scientific parent commit: `6ca361c400d6ce2624f6a4fe51fa0e1fdcdd3050`.
- Checkpoint: global step 100,000, SHA-256
  `52fc05c7f85fd4ae62a7058f277423631cbee4685595c21ee19f4abe91ab47fe`.
- Vocabulary cache SHA-256:
  `1df27c69fd08e443d905fdb0ecc2beccc3f72ed968fcaf073316450972c0ee6e`.
- Configuration: `sudoku_paper`.
- Verdict prefix: the first 256 examples of the untouched official 10,000
  puzzle test set.
- Sampler: maximum 65,536 AP-MDM iterations, batch 256, BF16, seed/config
  lineage 42.

The executable command remains:

```text
python -m repro.cli upstream-evaluate --checkpoint CHECKPOINT
  --config-name sudoku_paper --vocab-cache VOCAB --limit 256
  --batch-size 256 --max-steps 65536 --device cuda --bf16
  --eval-tag upstream-paper-100k-seed42 --out SUMMARY
```

Changing thresholds, batch membership, checkpoint, maximum sampler steps, or
merging partial output is forbidden.

## Placement and queue

Reuse only the already-running RunPod `bk223asss2nj7a` (1x A40, 9 allocated
vCPUs, $0.44/hour). Do not create or resume another pod. GO7-ENERGY-SELECTOR-E1
retains first use of the GPU. This evaluation is its explicit queued follow-up
and starts only after the Go7 runner exits, avoiding cross-job GPU timing
contention.

The evaluation ceiling is 7,200 seconds after its Python process starts, for a
maximum incremental RunPod charge of $0.88. The queue wait does not count
against the evaluation ceiling but remains visible in the fleet ledger.

## Closure and release

Before the pod is terminated, preserve and SHA-256 verify locally:

- the launch/preflight manifest and exact source commit;
- the checkpoint and vocabulary hashes;
- raw 256-row verdict artifact;
- both emitted summary copies;
- evaluation log and exit/failure record;
- recursive completion manifest;
- GO7-ENERGY-SELECTOR-E1's independently verified artifact closure.

Durable AP-MDM destination:
`/home/ubuntu/apmdm-official-data/runpod-eval3-20260808`.

The ordinary A40 has no further declared follow-up after this verdict. The
watcher must terminate it only after both Go7 and AP-MDM closures pass, then
verify the pod disappears from the account inventory.
