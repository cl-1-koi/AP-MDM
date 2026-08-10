# Binding Spec: Cross-Paper Sudoku Replication Failure Review

Date: 2026-08-10 UTC

## Objective

Perform an independent, adversarial review of why our Transformer Sudoku work
has been much harder to reproduce or generalize than the successful non-Sudoku
insertion-process results. Read the primary papers, upstream repositories,
implemented code, sealed contracts, and raw result artifacts yourself. Do not
inherit conclusions from prior summaries without checking their evidence.

A clean conclusion that the evidence is insufficient is a valid deliverable.

## Evidence that must be inspected

Primary papers and available source:

- `/home/ubuntu/papers/2606.02133/source/` — *Variational Learning for
  Insertion-based Generation*.
- `/home/ubuntu/papers/2510.06190/source/` — AP-MDM paper source.
- `/home/ubuntu/papers/ired/du24f.txt` — *Learning Iterative Reasoning through
  Energy Diffusion*.
- `/home/ubuntu/AP-MDM/` — available AP-MDM upstream code.
- `/home/ubuntu/upstream-ILM-20260809/` — audited predecessor star-graph code.

Local implementation and declared contracts:

- This repository, especially `repro/`, `IP_REPRO_CONTRACT_20260809.md`,
  `SMALL_SUDOKU_GROKKING_LADDER_20260809.md`,
  `SUDOKU_PAPER_CODE_AUDIT_20260808.md`,
  `SUDOKU_REPRO_IMPLEMENTATION_REPORT_20260808.md`, and
  `SUDOKU_VISIBLE_COMPONENT_GATE_20260809.md`.
- Commits through `8091b6e`, including the AP-MDM length correction `89d93df`
  and IRED implementation/review `58703d4`, `7b585a8`, and `8091b6e`.

Raw and durable evidence:

- `/home/ubuntu/apmdm-official-data/sudoku-grokking/`
- `/home/ubuntu/apmdm-official-data/ip-star-runpod-aoip/`
- `/home/ubuntu/apmdm-official-data/ip-star-runpod-learned/`
- `/home/ubuntu/apmdm-official-data/ip-star-runpod-fixed/`
- `/home/ubuntu/apmdm-official-data/runs/paper-declared-100k/`
- `/home/ubuntu/ired-sudoku-data/`
- `/home/ubuntu/apmdm-insertion-data/`
- `/home/ubuntu/apmdm-official-data/runpod_fleet_ledger.json`
- `/home/ubuntu/apmdm-official-data/sudoku-grokking/s4/supervisor/status.json`

Known bookkeeping facts to verify, not assume:

- The hard star-graph tests produced about 99.1% exact for variable-length
  random AO-IP and about 80.7% for learned IP on 5,000 examples; the latter is
  close to the paper's 83% claim. Our fixed arm produced 0%, but its semantics
  may not match the paper's fixed-canvas AO-ARM baseline.
- We have not run the GuacaMol/ChEMBL chemical-language gate. Do not describe
  chemical generation as replicated.
- Most S4 arms interpolate their train split, but corrected AP-MDM does not yet,
  AO-IP wd=0.01 is just below exact interpolation at its latest measured gate,
  and FO's 100% Sudoku-solution exactness does not imply 100% arbitrary
  keyed-payload accuracy.

## Required analysis

1. Build a provenance-backed truth table of what was actually tested, on which
   dataset, at what commit/config/budget, and with which primary result. Clearly
   separate faithful reproduction, reconstruction, adaptation, and diagnostic
   control.
2. Map each implemented objective and decoder to the corresponding paper
   equations or released code. Identify every material mismatch, omitted
   training ingredient, task-interface change, or ambiguity. Pay special
   attention to conditioning/addressing, corruption distribution, termination,
   rollout/inference procedure, positional representation, train-set size,
   curriculum, optimizer schedule, EMA, and whether evaluation asks for
   algorithmic inference absent from the paper's training distribution.
3. Explain why star graphs can succeed while Sudoku fails. Test at least these
   competing accounts against the evidence: visible keyed-component retrieval
   versus latent constraint solving; sequence/canvas mismatch; decoder exposure
   bias; capacity or data scale; objective/gradient implementation error;
   inadequate train interpolation; optimizer/grokking horizon; and evaluator
   mismatch.
4. Reassess what the current S4 keyed-payload task establishes. Determine
   whether it is a fair mechanism probe, an unnecessarily difficult conjunction
   of retrieval plus Sudoku, or both. Distinguish Sudoku-solution
   generalization from arbitrary payload following.
5. Rank the likely causes with confidence and direct citations to files, metric
   records, equations, or code lines. Include observations that falsify each
   favored explanation.
6. Propose the smallest discriminating experiment sequence—preferably two to
   four bounded gates—that would decide among the top explanations. Define
   predeclared outcomes and stopping rules, but do not launch experiments.

## PASS gates

- The truth table includes star graph, 9x9 Sudoku, S4 Sudoku, official AP-MDM
  reproduction, and IRED reproduction/adaptation evidence.
- Every numerical claim identifies its source artifact.
- Paper-vs-code claims cite the exact equation/section and implementation file.
- The report distinguishes non-interpolation from post-interpolation failure.
- The report explicitly answers whether any chemical-language experiment ran.
- Recommendations are discriminating tests, not an open-ended tuning list.
- Existing repository tests remain green; no production code is changed.

## Non-goals and constraints

- Analysis and documentation only. Do not launch or stop pods, alter live jobs,
  edit telemetry, or run a new training sweep.
- Do not tune beyond the bounded experiments proposed in the report.
- Do not treat loss reduction, teacher-forced token accuracy, Sudoku validity,
  exact solution reproduction, and counterfactual payload following as
  interchangeable.
- Preserve all existing user and other-worker files.

## Deliverable and commits

Write `SUDOKU_CROSSPAPER_FAILURE_REVIEW_20260810.md` in the repository root.
Run the full existing test suite and record the command/result. Commit the
review promptly as one results-document commit. Work autonomously without
asking for confirmation.
