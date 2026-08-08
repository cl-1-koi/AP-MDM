# Binding Spec: Independent AP-MDM Sudoku Fidelity Review

Date: 2026-08-08

## Objective

Adversarially review `SUDOKU_REPLICATION_SPEC_20260808.md` against arXiv:2510.06190v2 and the authors' official code at commit `836b79b30286301ee1cca975db7e8468bc2a653e`. Determine whether the contract is sufficient to implement and honestly test the paper's reported 99.28% Sudoku accuracy from 100 source puzzles.

This is an independent evidence and contract review. Do not implement code, generate the full dataset, run training/evaluation, or use a GPU. A BLOCK verdict with precise reasons is a valid deliverable.

## Required sources

Read in full the binding replication spec and every Sudoku-relevant part of:

- `/home/ubuntu/papers/2510.06190/2510.06190v2.pdf`
- `/home/ubuntu/papers/2510.06190/source/iclr2026_conference.tex`
- `/home/ubuntu/papers/2510.06190/source/method.tex`
- all paper training/loss definitions and appendices;
- `/home/ubuntu/papers/2510.06190/source/exp.tex`
- `/home/ubuntu/papers/2510.06190/source/examples.tex`
- the official repository's README, Sudoku generator/solver/loader/verifier, tokenizer/dataloader, model, loss, sampler, main entry point, scripts, and configs.

Recompute all code/data/paper hashes named in the spec. Do not rely on filenames or claims in comments.

## Review questions

1. What exactly counts as one of the claimed 100 samples? Quantify source puzzles, solver trajectories, transitions, repeated/corrupted views, batches, epochs, and effective token targets separately.
2. Is the 99.28% metric exactly reconstructed? Identify test set, denominator, exact-grid versus valid-grid semantics, uniqueness/alternate-solution handling, givens consistency, timeouts, retries, sampling randomness, and checkpoint/threshold selection.
3. Reconcile or flag the paper/code differences in architecture (6x4x256 versus checked-in 4x2x128), vocabulary (31 versus prose 32), and per-puzzle transition counts (1,421.3 versus 25,022.6).
4. Verify whether the checked-in generator, loss, and sampler actually implement the paper equations and four operations. Look specifically for labels recomputed differently from the trajectory data, loss paths that bypass precomputed labels, train/inference mismatches, implicit rules/solver use at evaluation, or operation heads that cannot emit.
5. Determine whether the official 100/10,000 arrays are independent beyond exact-row equality: duplicate underlying puzzles, solution overlap, puzzle symmetries, generation provenance, difficulty distribution, and any train-derived cache/test leakage that can be checked locally.
6. Verify the parameter-count comparison and whether the paper's 1.2M count includes all four output heads and embeddings.
7. Identify every scientifically consequential value omitted from the paper or code: epochs/selected step, convergence rule, checkpoint selection, thresholds, maximum inference steps, retries, batching, RNG, hardware, and evaluation implementation.
8. Assess whether a result from the proposed reproduction can fairly be called exact, faithful-but-inferred, partial, or only a best-effort reconstruction.
9. Review the predeclared strong/partial/fail thresholds and recommend changes only when causally justified before any production result exists.
10. Produce a minimal list of corrections or additional gates the implementation unit must adopt before GPU production.

## Deliverable

Write `SUDOKU_REPLICATION_INDEPENDENT_REVIEW_20260808.md` containing:

- a source-evidence table with exact paths/lines/hashes;
- a paper-versus-code-versus-spec identity table;
- independently verified facts, contradictions, and unknowns;
- leakage and metric risk assessment;
- a PASS, CONDITIONAL PASS, or BLOCK verdict on the implementation contract;
- binding corrections required before a full run;
- what any eventual positive or negative result would and would not establish.

Make no code changes. Commit the report as one focused commit, push to the `cl-1-koi/AP-MDM` fork, and leave the worktree clean. Work autonomously without asking for confirmation.
