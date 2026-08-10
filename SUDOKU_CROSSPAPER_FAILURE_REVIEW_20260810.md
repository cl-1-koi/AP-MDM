# Cross-Paper Sudoku Replication Failure Review

Date: 2026-08-10 UTC
Binding spec: `KIMI_SUDOKU_CROSSPAPER_FAILURE_REVIEW_SPEC_20260810.md`
Author: Kimi Code CLI review unit (autonomous)
Method: independent inspection of the primary papers, upstream repositories, this
implementation, sealed contracts, and raw durable artifacts. No compute was
launched or stopped; no production code, telemetry, or live job was touched.
All load-bearing numbers below were re-read from the raw artifacts by this
review, not inherited from prior summaries.

Revision note (second pass, 2026-08-10): incorporates reviewer corrections
(confound-safe headline, tiny-set claim scoped to the AP-MDM/IP/S4 lines,
S4-IRED interpolation labeled remotely-observed-not-durably-archived, D3
inconclusive branch, AP-MDM class = author-stack training + reconstructed
sampler/evaluator, narrowed objective-error falsification, IRED-aware bottom
line) and the newly synced S4 panel artifacts at snapshot 05:37:46Z
(`s4/panel-58703d4/`, `s4/panel-89d93df/`). First-pass evidence is retained;
stale statements are corrected in place and marked.

---

## 1. Executive summary

- **Star graphs succeeded; Sudoku failed. The leading hypothesis — supported
  by the controlled VC-1 interventions, but not an isolated, proved cause —
  is that the tasks reward different mechanisms.** The star answer is a short
  chain of node-ID tokens that are *visible* in the prompt and only need
  content-addressed retrieval and reordering; the Sudoku answer is *latent*
  and must be inferred through global constraints. The comparison remains
  confounded by answer visibility, training-set scale, task interface, and —
  in some arms — fixed-canvas versus variable-length objectives, so the
  mechanism account is retained as a hypothesis with controlled support
  (§5, item 1), not a verdict. What the evidence does exclude is a generic defect
  in the shared machinery, and only for the enumerated/unit-tested paths
  (§5, item 6). The insertion-objective family that scores 99.1% exact on hard
  star graphs (variable-length AO-IP) scores 0/256 exact on 9x9 Sudoku at a
  matched 78,100-update budget (fixed-canvas `random_insertion`, VC-1a).
  Every visibility-increasing diagnostic control confirms this gradient: an
  oracle arm that exposes the answer at a constant local offset solves
  256/256 immediately; exposing the same information at a position-dependent
  (transposed) or key-shuffled address fails 0/256.
- **On the AP-MDM/IP/S4 Sudoku lines — all trained on 64-100 instances — the
  dominant failure mode is memorization of a tiny training set, i.e.
  post-interpolation failure, not failure to fit.** With the exceptions
  enumerated in §5, item 7, every failing arm on those lines reached ~100%
  teacher-forced train accuracy and then stayed at or near 0 held-out exact.
  The D0 counterfactual diagnostics show the 9x9 keyed model reproduces the
  memorized solution (99.91% of blank cells) while ignoring deliberately
  changed payloads (0.017%). This tiny-set claim is scoped to those lines
  only: the released-IRED 9x9 arm trained on 9,000 puzzles and still
  under-reproduced its paper (§3.4), so training-set scale alone does not
  explain the Sudoku gap.
- **The official AP-MDM Sudoku claim (99.28% from 100 puzzles) did not
  reproduce at 100,000 steps: 0/256 exact, 0/256 valid.** Training was
  faithful to the authors' stack; scoring went through a *reconstructed*
  prompt-conditioned sampler/evaluator, because the released sampler cannot
  run Sudoku at all (unconditional, stops on a token the generator never
  emits, 50-step cap — §4.2). That reconstruction is the largest inference in
  the comparison. The rollout also never terminates correctly (183/256 hit
  the 65,536-step ceiling).
- **The released IRED code on its own Sudoku task also under-reproduces**:
  76.9% standard / 4.47% hard against the paper's 99.4% / 62.1% (Table 4 of
  the IRED paper). Sudoku results in this literature are fragile to details
  the papers do not state.
- **No chemical-language (GuacaMol/ChEMBL) experiment was ever run.** See §8.
- **The repository test suite is not green at the reviewed commits, and has
  not been since `47ed1ed`**: 210 passed, 1 pre-existing failed
  (`test_source_hashes_cover_every_repro_module`; the sealed-manifest source
  hash list covers 22 of 33 `repro/` modules), 1 environment-gated skip.
  Root cause and minimal future fix in §9. This review changes no code.

---

## 2. Evidence base inspected

Primary sources (read independently for this review):

- IP paper LaTeX source `/home/ubuntu/papers/2606.02133/source/` (Zhang et
  al., *Variational Learning for Insertion-based Generation*, arXiv:2606.02133v3).
- AP-MDM paper LaTeX source `/home/ubuntu/papers/2510.06190/source/`
  (arXiv:2510.06190v2).
- IRED paper text `/home/ubuntu/papers/ired/du24f.txt` (Du, Mao, Tenenbaum,
  ICML 2024).
- Upstream code: `/home/ubuntu/AP-MDM/` @ `836b79b` (remote
  `github.com/chr26195/AP-MDM.git`), `/home/ubuntu/upstream-ILM-20260809/`
  @ `6cc27f1` (remote `github.com/dhruvdcoder/ILM.git`).
- This repository through `50560f7` (review-spec commit), including
  `89d93df` (AP-MDM S4 length fix), `58703d4` / `7b585a8` / `8091b6e` (IRED
  comparator and its fused-attention review).
- Raw artifacts under `/home/ubuntu/apmdm-official-data/`,
  `/home/ubuntu/ired-sudoku-data/`, `/home/ubuntu/apmdm-insertion-data/`
  (the last contains only two shell scripts; the actual insertion artifacts
  live under `apmdm-official-data/insertion-*`).

Classification vocabulary used in the truth table:

- **Faithful reproduction**: the authors' released code and configuration run
  as released (only environment/loader repairs).
- **Reconstruction**: the paper's equations re-implemented without author
  code for the method, on the authors'/predecessor's task data.
- **Adaptation**: the mechanism ported to a different task interface or
  supervision source than the paper used.
- **Diagnostic control**: an arm whose purpose is to isolate a mechanism, not
  to reproduce a number.

---

## 3. Provenance-backed truth table

### 3.1 Hard star graphs (ILM task; IP paper's planning benchmark)

Task (IP paper `appendix.tex:293`, §"Star Graph Planning Dataset"): degree 5,
arm lengths uniform in [6,12], randomized junction, node vocabulary 56.
Predecessor ILM release verified: generator
`upstream-ILM-20260809/src/pcdd/datamodule/star_v2.py:193`
(`asymmetric_variable_armlength_star_graph`), counts 50,000/100/5,000 in
`configs/lightning_train/dataset/vstar_medium_v2.yaml:6,14,22`. Note: the
released ILM dataset configs carry a stale `_target_`
(`vstar_medium_v2.yaml:5` points at module `pcdd.datamodule.star`, which does
not exist at `6cc27f1`; the class lives in `star_v2.py:390`) — as released,
the ILM config cannot be instantiated without overriding the target.
Paper reference numbers (`sec_eval.tex:96-124`, table `tab:star-hard-results`):
FO-ARM 24.0, MDM 25.0, AO-ARM 26.5 (fixed-canvas), LO-ARM 30.1
(fixed-canvas), FlexMDM 0.0, IP 83.0 sequence accuracy.

All three local arms: regenerated frozen splits (test sha256 `1e359d18…`,
test_seed 161803, n=5,000), 78,100 updates (= 100 epochs x 781, batch 64),
AdamW 1e-4, cosine + 1,000 warmup, EMA 0.999 from step 200, search-free greedy
decoding; evaluation commit `b939ba4`, `source_dirty: false`.

| Arm | Class | Train commit | Primary result (n=5,000) | Artifact |
|---|---|---|---|---|
| AO-IP (variable-length random insertion, policy termination) | Reconstruction | `9a5fc7c` | **exact 0.991** (EMA and raw), token 0.9901, Hamming 0.159, Lev 0.0654, terminated 0.9996 | `/home/ubuntu/apmdm-official-data/ip-star-runpod-aoip/full-test-aoip-b939ba4-s42/test-evaluation.json` |
| Learned IP (Plackett-Luce posterior, M=2 RLOO) | Reconstruction | `ac97943` | **exact 0.8064 EMA / 0.8072 raw**, token 0.7802, Hamming 3.6776, Lev 1.6756, terminated 0.9768 | `/home/ubuntu/apmdm-official-data/ip-star-runpod-learned/full-test-learned-b939ba4-s42/test-evaluation.json` |
| Fixed (paper-native left-to-right FO order, variable length) | Reconstruction | `b0d54bc` | **exact 0.0**, token 0.1153, Hamming 14.3788, Lev 12.7856, terminated 1.0, predicted length 11.45 vs target 16.02 | `/home/ubuntu/apmdm-official-data/ip-star-runpod-fixed/full-test-fixed-b939ba4-s42/test-evaluation.json` |

Interpretation flags:

- The learned-IP result is 2.36 points below the paper's 83.0 (n=5,000, so the
  gap is several standard errors) — a near-replication, not an exact one. The
  paper never states star-graph split sizes (`appendix.tex:290-299` gives the
  configuration but no counts; the counts come from the ILM release).
- The AO-IP arm has **no paper star-graph target**: the paper's 26.5% belongs
  to fixed-canvas AO-ARM (`sec_eval.tex:92-93` states AO-ARM/LO-ARM/MDM
  generate on fixed-size canvases). The local manifest records this
  explicitly (`repro/ip_star_train.py:225-230`). The 99.1% AO-IP result shows
  the variable-length *task* is easy for this decoder family even with a
  random order — the learned order is not the star task's causal ingredient.
- The fixed arm's 0% against the paper's FO-ARM 24.0% is an **unresolved
  mismatch**: our fixed arm is a paper-native left-to-right AR with an
  EOS-content termination (`repro/ip_star_model.py:480-511`,
  `fixed_order_objective`), decoded greedily (`repro/ip_star_eval.py:61`). It
  systematically under-lengths (11.45 vs 16.02) while terminating 100% of the
  time. Whether the paper's 24.0% used different termination/prefix semantics
  cannot be decided from released artifacts (no author code exists;
  `IP_REPRO_CONTRACT_20260809.md:19-20`). This arm's semantics therefore
  cannot be equated with the paper's fixed-canvas AO-ARM baseline either.

Supporting diagnostics: single-example memorization controls (`memorize-*`,
2,000-5,000 steps) completed; the 8-example `overfit-learned-1923fab` control
has a manifest but no `result.json` (incomplete —
`/home/ubuntu/apmdm-official-data/ip-star-runpod-82i4fwoch34p40/`). The G0
exactness gate passed: enumerated RLOO surrogate gradient equals the exact
ELBO gradient to 1e-10 (`tests/test_ip_objective.py:44-70`).

### 3.2 9x9 Sudoku, AP-MDM task (arXiv:2510.06190v2)

Paper claim (`iclr2026_conference.tex:470-483`, table `tab:sudoku_accuracy`):
AP-MDM, 100 training samples, ~1.2M parameters, **99.28%** accuracy; ARM
87.18% and AO-MDM 89.49% rows are imported from Kim et al.
(arXiv:2502.06768), per the caption (`iclr2026_conference.tex:497`). Training
setup `exp.tex:12`: AdamW 1e-4, wd 0.01, batch 256, constant + 250 warmup,
bf16, clip 1.0, "up to 1M steps or until convergence", no EMA mentioned.
Architecture `exp.tex:6`: 6 layers, 4 heads, d=256, ffn 1024, maxlen 400,
vocab 31 — which cannot have 1.2M parameters (exact count 6,050,594;
`SUDOKU_PAPER_CODE_AUDIT_20260808.md` §1.1, pinned by
`tests/test_model.py`).

| Arm | Class | Commit | Data / budget | Primary result | Artifact |
|---|---|---|---|---|---|
| Official AP-MDM, paper-declared arch (`sudoku_paper`, 6,052,133 params, effective V=34) | **Author-stack faithful training + reconstructed sampler/evaluator** (authors' generator, loss, DIT, Lightning loop unmodified in scientific semantics; SDPA fallback for flash_attn; loader repairs `fed871b`, `6ca361c`. The conditional sampler and evaluator are *not* author-released — reconstructed from paper Algorithm 1, audit §2.3) | upstream `836b79b`; arm repo `f786925c`; eval `0648e46e` | 100 puzzles -> 2,502,258 solver transitions (2,477,235 train / 25,023 in-distribution monitor); 100,000 steps, batch 256 | **exact 0/256, valid 0/256**, givens-consistent 43/256, complete 9/256, malformed 213, timeouts 183 (65,536-step ceiling), 73 fixed-point; mean 46,884.7 forward passes; 20,996,546 unmask + 20,996,584 remask ops, **0 insert / 0 delete**; verdict "failure to replicate" | `/home/ubuntu/apmdm-official-data/runpod-eval3-kxkrwivmtw5psi/data/eval/summary_upstream-paper-100k-seed42.json`; training closure `/home/ubuntu/apmdm-official-data/runs/paper-declared-100k/` (step 100,000 reached 2026-08-08 22:00:15, `resume3_hydra/main.log`) |
| Monotone ladder R1: `fixed_ar` / `random_insertion` / `learned_insertion` (terminal pairs only, no solver transitions) | Adaptation (fixed-canvas IP specialization) | `e7a780bf` | same 100 puzzles; 10,000 updates, batch 64 | **0/256 exact and 0/256 valid at every 1,000-update gate, all three arms**; final train digit accuracy 1.0 (fixed), 0.734 (random); learned ELBO -0.157 | `/home/ubuntu/apmdm-official-data/insertion-r1-20260808/{fixed_ar-n91skvm82naziq,random_insertion-zlip5ndwmesfx7,learned_insertion-zbc6tscdjqrr4p}/curve_summary.json` |
| VC-1a `puzzle_only` | Diagnostic control | `9fc91fe` | 100 puzzles; 78,100 updates, batch 64 | 0/256 exact at all 16 gates; train digit 1.0; held-out blank-cell accuracy 0.2073 (recomputed read-only from `eval_rows_step_000078100.jsonl`; doc value 20.73% in `SUDOKU_VISIBLE_COMPONENT_GATE_20260809.md:84`) | `/home/ubuntu/apmdm-official-data/sudoku-vc1/hidden/full-hidden-9fc91fe-s42/curve_summary.json` |
| VC-1a `transposed_solution_hint` | Diagnostic control | `9fc91fe` | same | 0/256 at all gates; blank accuracy 0.2121 (recomputed) | `/home/ubuntu/apmdm-official-data/sudoku-vc1/oracle/full-oracle-9fc91fe-s42/curve_summary.json` |
| VC-1b `aligned_solution_hint` | Diagnostic control (oracle local copy) | `c1c458e` | same | **256/256 exact, valid, consistent at every gate from 5,000 through 78,100** (single blip 0.9921875 at 30,000) | `/home/ubuntu/apmdm-official-data/sudoku-vc1/aligned/full-aligned-c1c458e-s42/curve_summary.json` |
| VC-1c `keyed_shuffled_solution_hint` | Diagnostic control (content-addressed retrieval) | `f992c50` | same; vocab expanded 34 -> 115 (81 cell-key tokens) | 0/256 at all gates; train digit accuracy oscillates 0.968-1.0 in late telemetry, final 0.984375 — near-interpolation, not clean | `/home/ubuntu/apmdm-official-data/sudoku-vc1/keyed/full-keyed-f992c50-s42/curve_summary.json` |
| D0 counterfactual audit of the VC-1c checkpoint (step 78,100) | Diagnostic control | `5e968f5` | n=100 train / n=256 held-out, 5 conditions | train keyed: 0.99879 blank accuracy; **train counterfactual: payload 0.00017 vs solution 0.99914** (96/100 whole-board); held-out keyed 0.27825, unkeyed shuffled 0.22525, no-hint 0.22319; **held-out counterfactual: payload 0.09321 vs solution 0.27420**; payload exact 0.0 everywhere held-out; rollout solve 70/100 train, 0/256 held-out | `/home/ubuntu/apmdm-official-data/sudoku-grokking/d0/d0-keyed-5e968f5/diagnostics.json` |

Interpolation status (needed by the spec's non-interpolation distinction):
the official 100k arm **interpolated its transition supervision** (step 99,999
`trainer/loss` 2.03e-05, `train/unmasking` 5.4e-06, val `nll` 3.46e-4 at step
99,770 — telemetry tail parsed from
`runs/paper-declared-100k/resume3_hydra/lightning_logs/version_0/events.out.tfevents.*`),
VC-1a/b/c and monotone `fixed_ar` reached train digit accuracy ~1.0, and the
monotone `random_insertion` arm was at 0.734 at its 10,000-update endpoint
(not yet interpolated; budget-limited, not a clean post-interpolation
failure). Everything that failed held-out after fitting train is a
**post-interpolation failure**.

### 3.3 S4 (4x4) mechanism panel

Design (`SMALL_SUDOKU_GROKKING_LADDER_20260809.md:96-131`): all 288 valid 4x4
solutions enumerated; frozen disjoint splits 64 train / 128 held-out / 96
unused (`repro/small_grokking.py:154-181`); 6-given uniquely-solvable puzzles;
primary outcome is arbitrary keyed-payload following, Sudoku validity
secondary. Shared policy: 2 blocks, width 64, 4 heads, 135,813 params
(`s4-policy|L=2|H=4|d=64|V=25|state=64`, per
`s4/panel-b237f07/fo-arm-wd001-s42/manifest.json`); AdamW 1e-4, warmup 250,
clip 1.0, batch 256, bf16; eval gates 1k/3k/10k/30k/100k/300k/1M; wd in
{0, 0.01}; seed 42. Class: **adaptation/diagnostic panel** — no paper reports
this task.

Completed arms (all artifact closures verified, per
`/home/ubuntu/apmdm-official-data/sudoku-grokking/s4/supervisor/status.json`,
snapshot 2026-08-10T02:44:47Z, and local `panel-b237f07` / `panel-20ed845`
telemetry). Held-out rollout exact (n=128) and counterfactual payload
accuracy at the 1M gate:

| Arm | Held-out exact @1M | Train exact | Counterfactual payload | Payload (normal keyed) |
|---|---|---|---|---|
| fo-arm-wd0 | 0.1953125 | 1.0 (by 3k) | 0.1828 | 0.4578 |
| fo-arm-wd001 | 0.1875 | 1.0 | 0.1648 | 0.4609 |
| ao-arm-wd0 | 0.2109375 | 1.0 (by 30k) | 0.1109 | 0.6750 |
| ao-arm-wd001 | 0.234375 | 1.0 | 0.1070 | 0.6711 |
| mdm-wd0 | 0.0390625 | 1.0 (by 3k) | 0.0961 | 0.6695 |
| mdm-wd001 | 0.046875 | 1.0 | 0.1031 | 0.6734 |

Directly verified cross-check (`panel-b237f07/fo-arm-wd001-s42/telemetry.jsonl`
train-split evals): train rollout exact = 1.0 at steps 300,000 and 1,000,000
while the one-forward keyed payload accuracy on the same train split is
0.5016 / 0.5328 and counterfactual payload is 0.1703 / 0.1625. **100%
Sudoku-solution exactness does not imply arbitrary keyed-payload following**
— the spec's bookkeeping fact is confirmed from raw artifacts. Chance for an
unrelated 4-way label is 25%; all completed arms sit *below* it on
counterfactual payload while far above it on normal payload, i.e. they answer
from the memorized/learned solution distribution, not from the supplied keys.

1k-gate benchmark (`s4/bench-eea5198/`, recorded in
`SMALL_SUDOKU_GROKKING_LADDER_20260809.md:122-131`): held-out exact
11.72% / 2.34% / 13.28% for FO/AO/LO; counterfactual payload 21.09% / 16.02%
/ 11.64%; LO-ARM already predicted the original training solution on 97.66%
of counterfactual blanks — memorization-before-generalization from the start.

Completed since the first pass of this review (artifact closures verified
per `s4/supervisor/status.json`, snapshot 2026-08-10T05:37:46Z; numbers read
from the arms' local `telemetry.jsonl` under `s4/panel-58703d4/` and
`s4/panel-89d93df/`):

| Arm | Train exact @100k/300k/1M | Train payload | Held-out exact (all gates) | Held-out payload @1M | Held-out cf payload @1M |
|---|---|---|---|---|---|
| ao-ip-wd0 | 1.0 / 1.0 / 1.0 | 1.0 (train cf payload 0.0) | **0.0** (1k->1M) | 0.2797 | 0.2352 |
| ip-wd0 | 1.0 / 1.0 / 1.0 | 1.0 (train cf payload 0.0) | **0.0** (1k->1M) | 0.2805 | 0.2367 |
| ap-mdm-wd0 | 0.359 / 0.406 / 0.500 | 0.8187 | **0.0** (all gates) | 0.3414 | 0.2172 |
| ap-mdm-wd001 | 0.375 / 0.484 / 0.672 | 0.8688 | **0.0** (all gates) | 0.3398 | 0.2023 |

Read-out: the variable-length insertion arms interpolate the train split
completely — including the keyed payload mapping (train payload 1.0) — yet
follow a *changed* train payload 0.0 of the time, and solve 0 held-out
puzzles at every gate: the same memorization signature as the fixed-canvas
families, now with keys present. The corrected AP-MDM arms **never
interpolate**: train exact is still climbing (0.50 / 0.67) at the 1M
endpoint, so their 0.0 held-out exact is a genuine **non-interpolation
result** — no generalization verdict can be drawn from them.

Still running remotely at the 2026-08-10T05:37:46Z snapshot: ao-ip-wd001
@983k, ip-wd001 @854k (pod `82i4fwoch34p40`), lo-arm-wd{0,001}-relocated
@~680-699k, ired-wd{0,001}-relocated @~528-529k (pod `8du6w86kw0eevn`).
Locally archived for these arms: supervisor 3-entry train-telemetry tails
plus the <=10k pre-relocation gates. The tails (train metrics only, stale
the moment they are written): LO-ARM loss ~1e-6 at ~524k — consistent with
its locally archived train exact 1.0 at the 10k gate
(`s4/relocation-source-b237f07/`); IRED `loss_denoise` ~1.5-2e-4 with
`loss_energy_nce` ~0.49 at ~528k; ao-ip-wd001 `expected_content_nll`
~0.37-0.39 at 600k; ip-wd001 ~0.02-0.03 at 530k. For the two IRED arms an
operator remote audit after the 300k gate (2026-08-10, reported to this
review second-hand) found train exact/valid/payload = 1.0 for both wd
variants and, for wd=0.01, held-out exact 0.0 with counterfactual payload
0.23047; **these values are remotely observed but not yet durably archived
locally**, and the stale local 10k gates (train exact 0.094-0.109,
`s4/relocation-source-58703d4/`) must not be read as the current arms'
interpolation state.

The pre-fix S4 AP-MDM arms on `58703d4` failed after the 1k gate
(`s4/failed-58703d4/ap-mdm-wd{0,001}-s42/failure.json`, exit 1): the model
length 25 could not represent the corruption process's true maximum 33 —
fixed by `89d93df` (`repro/s4_apmdm.py:44-47`, `MAX_OUTPUT_LENGTH` 25 -> 33,
with regression test `tests/test_s4_apmdm.py:43-47`). Any pre-fix AP-MDM S4
number would have been invalid; the corrected arms are the ones tabulated
above (completed 1M without interpolating).

### 3.4 IRED

Paper (`du24f.txt`): energy diffusion with K=10 annealed landscapes
(`du24f.txt:516-518`), per-landscape denoising loss on the energy gradient
(Eq. (5), `du24f.txt:359-364`) plus a contrastive negative-energy term
(`du24f.txt:457-499`); inference = gradient descent with energy-decrease
acceptance over the landscapes (Algorithm 2, `du24f.txt:434-488`). Sudoku:
train on SAT-Net data (givens 31-42), harder set from RRN (givens 17-34)
(`du24f.txt:873-886`); 50,000 iterations, batch 64, Adam 1e-4
(`du24f.txt:1417-1421`); training-set size, step sizes lambda_k, EMA/WD, and
the accuracy definition are **not stated**. Paper Table 4
(`du24f.txt:791-815`): IRED 99.4% standard / 62.1% harder.

| Arm | Class | Commit | Budget | Primary result | Artifact |
|---|---|---|---|---|---|
| Released IRED SudokuEBM CNN (47,280,521 params), SAT-Net 9,000 train / 1,000 standard-val / 18,000 RRN hard-test | **Faithful-reproduction attempt** of the authors' code (upstream `3d74b85f`) inside a custom sealed harness (branch `codex/sudoku-repro-20260809` `5311cc9`); EMA 0.995/10, fp32, seed 42, 10 landscapes x 20 inner steps | `5311cc9` | 50,000 steps reached, wall 4,056 s | **standard-val 0.769** (n=1,000, Wilson [0.742, 0.794]), **hard-test 0.0447** (n=18,000, [0.0417, 0.0478]); inner-step sweep on the same checkpoint: 1->0.715, 20->0.769, 40->0.780, 80->0.778 | `/home/ubuntu/ired-sudoku-data/runpod-4naix31taf0dst/active-v2-5311cc9/eval_standard/summary.json`, `eval_hard/summary.json`, `train/training_manifest.json`, step sweep `/home/ubuntu/ired-sudoku-data/runpod-82i4fwoch34p40/step_sweep/inner_*/summary.json` |
| S4 Transformer IRED comparator (mechanism port: shared S4 Transformer condition encoder replaces the paper's 9x9 CNN; energy = squared-residual norm) | Adaptation | `58703d4`, clarified `8091b6e` | running, @~528k at the 05:37:46Z snapshot | Local durable gates are pre-relocation 10k only (train exact 0.094-0.109, held-out exact 0.0078-0.0156) and are **stale — they cannot describe the current arms**. An operator remote audit after the 300k gate reports train exact/valid/payload = 1.0 for both wd arms, wd=0.01 held-out exact 0.0 and counterfactual payload 0.23047; remotely observed, **not yet durably archived locally**. Current interpolation therefore not asserted by this review | `repro/s4_ired.py`; `s4/relocation-source-58703d4/ired-wd{0,001}-s42/telemetry.jsonl` (stale); `s4/supervisor/status.json` (train-telemetry tail only) |

The S4 comparator's mixed second derivative `∂²E/∂x∂θ` is computed through
explicit attention (`repro/model.py:174-183`, `manual_attention=True` at
`repro/s4_ired.py:100-103`) because every fused SDPA backend lacks the double
backward — established with local probes and upstream issue citations in
`FUSED_ATTENTION_HIGHER_ORDER_REVIEW.md` (verdict: math backend only, ~2-3%
step cost at the S4 shape; numerical agreement 0.0 max diff against explicit
attention). Training runs fp32 (`ops/run_s4_grokking_runpod.sh:82-85`).

### 3.5 Chemical language (GuacaMol/ChEMBL)

No run, ever. See §8.

---

## 4. Objective/decoder mapping and mismatch register

### 4.1 Insertion Process (IP paper -> `repro/ip_star_*`)

| Ingredient | Paper | Implementation | Status |
|---|---|---|---|
| Permutation -> insertion slot bijection | Lemma `lem:transform`, `sec_methods.tex:89-98` | `repro/ip_objective.py:24-37` (`permutation_to_slots`), inverse at `:40-51`; pinned example `tests/test_ip_objective.py:28-31` | match |
| Permutation-marginalized likelihood / ELBO | Theorem `thm:likelihood` `sec_methods.tex:127-144`; ELBO `eq:elbo_def` `:184-193` | exact ELBO reference `repro/ip_objective.py`; audited against enumeration (`47ed1ed`) | match |
| Plackett-Luce posterior, Gumbel-Top-k | `eq:pl_factor` `sec_methods.tex:194-201`; `appendix.tex:119` | `repro/ip_star_model.py:304-320` (`_sample_orders`), M=2 at `:416` | match |
| Exact inner expectation over next index | `eq:F_def_compact`/`eq:F_exact_compact` `sec_methods.tex:224-239` | `repro/ip_star_model.py:347-406` (`_exact_next_value`; fp32 probability arithmetic under bf16 autocast noted `:356-358`; bf16 stabilization `c5aa056`) | match |
| RLOO stop-gradient surrogate | `eq:stopgrad_objective` `appendix.tex:120-132` | `repro/ip_star_model.py:409-446`: `(log q1 - log q2) * (f1 - f2).detach()`, scale `(L+1)/2` at `:432-434`; gradient equals enumerated exact-ELBO gradient to 1e-10 (`tests/test_ip_objective.py:44-70`) | match |
| Single time-step subsampling | `sec_methods.tex:241` (i ~ Unif{1,…,L+1}) | `repro/ip_star_model.py:297-301` (`_sample_prefix_lengths`: `floor(uniform * (L+1))`) | match |
| Termination | policy-based TERM (`sec_methods.tex:33-45`); classifier variant deferred | policy TERM via appended EOS position in the location head, `repro/ip_star_model.py:207-215,236-241`; manifest note `repro/ip_star_train.py:257`; classifier variant not implemented | match for the implemented variant; **paper's GuacaMol headline (97.4% validity) is the classifier variant** — unreplicated |
| Architecture | 12L/8H/d128, SwiGLU, RoPE base 10,000, dropout 0.1; posterior 6L/4H/d64 (`appendix.tex:299`) | `repro/ip_star_model.py:25-26` (`PAPER_DECODER_SPEC`, `PAPER_POSTERIOR_SPEC`), RoPE base 10,000 at `:21`; decoder 3,167,360 params, posterior 398,720 (manifest) | match as stated; note RoPE here is the interleaved convention (`ip_star_model.py:29-56`), unlike `repro/model.py:97-112` (half-split) — harmless within one codebase but a cross-codebase port hazard |
| Optimizer / EMA | AdamW 1e-4, cosine, 1,000 warmup, batch 64, 100 epochs, EMA 0.999 from step 200 (`appendix.tex:299`) | `repro/ip_star_train.py:53-90,286-290`; EMA `:121-173` applied to the decoder only | match; weight decay 0.0 (paper states none for planning) |
| Conditioning | prefix = edge list + source + goal + graph-BOS (`sec_eval.tex:136`, `appendix.tex:290`) | same layout, `repro/ip_star_data.py` | match |
| Train/test relationship | same generator; no OOD claim | frozen split hashes, zero overlap enforced | match |
| Inference | search-free ancestral sampling, TERM stop (`sec_methods.tex:31-45`) | `repro/ip_star_eval.py:24-103` greedy, `max_tokens=32` | greedy vs sampled — a deliberate, recorded difference; greedy can only help exact match |
| Fixed-canvas specialization | **none exists in the paper** (zero occurrences of "sudoku" in the source) | 9x9/S4 fixed-canvas arms (`repro/insertion.py:477-548`, `repro/small_grokking.py:476-523`) are our adaptation: known-length canvas, no TERM action (RLOO scale L not L+1, `repro/insertion.py:525-531`), posterior over non-given cells | **adaptation, not paper content** — the contract's question ("can the paper's variable-length advantage reproduce without the fixed-canvas Sudoku specialization", `IP_REPRO_CONTRACT_20260809.md:7-14`) is answered for star graphs, and the specialization is ours |

### 4.2 AP-MDM (paper + released code -> `train/`, `repro/`)

| Ingredient | Paper / released code | This implementation | Status / mismatch |
|---|---|---|---|
| Supervised objective | 4-head loss, `method.tex:136-149`, all lambda = 1.0 (`exp.tex:12`) | `repro/losses.py:103-120` with SUBS parameterization `losses.py:31-45`; upstream path `train/diffusion.py:805-886` (precomputed labels, plain means, no time weighting) | match to released code; **the Sudoku figure caption points at the self-supervised appendix** (`iclr2026_conference.tex:497` -> `appendix:training`) although Sudoku trains supervised — paper-internal ambiguity |
| Time conditioning | none (`exp.tex:6`) | disabled and proved bit-identical for sigma 0 vs 7.5 (`tests/test_model.py`) | match |
| Corruption/training distribution | solver state-transition tuples (`exp.tex:9`, `examples.tex:11-17`) | unmodified official generator; 2,502,258 transitions validated against Algorithm 1 (`SUDOKU_PAPER_CODE_AUDIT_20260808.md` §1.2, §2.4) | match |
| Insert/delete supervision | four operations declared | **Sudoku data sets insert/delete labels identically zero** (`sudoku_generator.py` `_create_sample` call sites, e.g. `:175-178`; pinned by `tests/` per audit §2.2); heads trained on all-negative targets | paper-consistent but material: two of four heads are pinned off; any insert/delete at inference is off-distribution |
| Halting supervision | terminal state never appears as an input `x_k` (audit §2.6); `[EOS]` never emitted (`sudoku_generator.py:611-614`) | fixed-point termination predeclared | **material omitted training ingredient**: no supervised stop exists in the data; the model must generalize "emit no operation" to an unseen state |
| Inference | Algorithm 1 prompt-conditioned (`method.tex:95-128`), thresholds 0.5 (`method.tex:156`), stopping criterion open | `repro/sampler.py:124` prompt-conditioned, thresholds frozen 0.5, ceiling 65,536 | **released sampler cannot run Sudoku**: `train/diffusion.py:1196-1263` is unconditional (all-MASK prior `:377`, no prompt), stops on an EOS that never occurs (`:1252-1253`), and `sampling.max_steps: 50` (`train/configs/sudoku.yaml:116`) is ~3 orders of magnitude below hard-puzzle trajectory lengths (audit §2.3). The evaluator is a reconstruction — the largest threat to a faithful comparison |
| Transition function `g` | Algorithm 1: delete iff `ctrl[3]==1 and x_i==MASK`; remask > unmask > keep | `repro/transition.py` single implementation shared by validation, inference, tests | released artifacts disagree: `sudoku_verifier.py` deletes without the mask check (audit §2.4); inert for Sudoku (delete bits all zero) but shows released tooling drift |
| Vocabulary | 31 (`exp.tex:6`) vs 32 (`examples.tex:9`); runtime tokenizer pads to **34** (`train/apmdm_dataloader.py:109-110`); `config.model.vocab_size` is never read (`train/diffusion.py:57,77`) | repro stack builds exactly 31; upstream-arm runs effectively 34 | both recorded; numerically harmless but the released config does not describe the released model |
| Architecture/params | 6L/4H/d256 described; 1.2M reported | 6,050,594 (V=31) / 6,052,133 (V=34) exact | **unresolved paper inconsistency** (audit §1.1); the 5x-parameter comparison sentence is load-bearing on 1.2M |
| Validation split | — | upstream: non-shuffled 1% tail (`train/apmdm_dataloader.py:256-262`), i.e. the last puzzles' transitions; repro: seeded split labeled in-distribution monitor | upstream "validation" is not a distribution sample (audit §2.7) |
| Optimizer schedule / EMA | AdamW 1e-4, wd 0.01, batch 256, constant + 250 warmup, bf16, clip 1.0 (`exp.tex:12`); **no EMA mentioned**; "up to 1M steps or until convergence" with no criterion or measurement step | upstream config matches (`train/configs/sudoku.yaml:57,61,78-105`; `training.ema: 0`); our run stopped at 100,000 of the up-to-1M budget | **we ran 1/10 of the paper's stated ceiling.** The 0/256 verdict at 100k cannot distinguish "paper claim not reproducible" from "convergence between 100k and 1M under an unstated criterion" — stated plainly as an evidence limit |
| Evaluation relationship to training | test set completely unspecified (size, provenance, difficulty); "accuracy" undefined | official 10,000-puzzle array, zero overlap measured four ways, exact-equality scoring | we evaluate what is available; what the paper scored is unknowable from released artifacts (audit §3 items 5-6) |
| S4 AP-MDM port | — | `repro/s4_apmdm.py`: on-the-fly aligned corruptions of terminal grids (`:115-201`), not solver trajectories; four heads exercised; variable-length padded attention; length-cover fix `89d93df` | **adaptation**: different supervision source than the 9x9 mainline; extra edit supervision disclosed in its manifest |

### 4.3 IRED (paper -> `/home/ubuntu/ired-sudoku-data`, `repro/s4_ired.py`)

| Ingredient | Paper | Implementation | Status |
|---|---|---|---|
| Annealed landscapes | K=10, cosine schedule (`du24f.txt:516-518`) | 10 landscapes, cosine (`repro/s4_ired.py:40-60`); released-code arm per its defaults | match |
| Denoising loss on energy gradient | Eq. (5) `du24f.txt:359-364` | `repro/s4_ired.py:130-165` (`create_graph=True`; mixed second derivative tested, `tests/test_s4_ired.py:32`) | match |
| Contrastive negative-energy term | `du24f.txt:457-499`, combined `L_MSE + L_Contrast` | NCE over `-energy` logits with weight **0.05** (`repro/s4_ired.py:167-184`) | **mismatch/ambiguity**: the paper states no coefficient (weight 1 implied by Algorithm 1, `du24f.txt:474-476`); the S4 comparator uses 0.05 by design choice |
| Inference | Algorithm 2 gradient descent with energy-decrease acceptance; adaptive steps (`du24f.txt:127-132`) | `repro/s4_ired.py:206-291` (20 inner steps/landscape, pin givens, reject uphill) | match in structure; step sizes lambda_k are paper-unstated |
| Architecture | task CNNs/NLMs (Sudoku: Conv-ResNet, `du24f.txt:1493-1504`) | released-code arm uses the authors' CNN (47.28M params); S4 arm substitutes the shared Transformer encoder (declared in its manifest, `repro/s4_ired.py:489`) | 9x9 arm faithful; S4 arm an adaptation |
| Sudoku train/test | SAT-Net train, RRN harder (`du24f.txt:873-886`); counts unstated | 9,000 / 1,000 / 18,000 (manifest) | split identities match the paper's description; paper's counts unverifiable |

### 4.4 Cross-cutting training-ingredient checklist (spec-mandated)

- **Conditioning/addressing**: star = edge-list prefix with shared node-ID
  tokens (content keys by construction); 9x9 VC-1 = 4-token cells with
  value/color/marker/separator, keyed variant adds 81 cell-key tokens
  (`repro/insertion.py:87-133`, `repro/vocab.py:40,52-53`); S4 = 64-token
  canvas `(value, hint, content_key, query_key)` (`repro/small_grokking.py:40-50,184-208`).
  The star task ships retrieval keys *in the paper's own format*; the Sudoku
  tasks had to have keys added by us — and adding them did not induce their use.
- **Corruption distribution**: IP = posterior-sampled orders + uniform prefix
  (gold-derived); AP-MDM 9x9 = solver trajectories; S4 AP-MDM = synthetic
  aligned corruptions; IRED = Gaussian noise at a uniform landscape index;
  monotone arms = uniform random prefixes of gold solutions. All training is
  teacher-forced on gold-derived states; **no arm trains on its own rollouts**
  (no scheduled sampling anywhere — `repro/` contains no such path).
- **Termination**: star/S4-insertion = supervised policy TERM (in
  distribution); AP-MDM 9x9 = unsupervised fixed point (out of distribution
  by construction); monotone = deterministic when no empty cell remains;
  IRED = energy-decrease acceptance over fixed landscape count.
- **Positional representation**: 1-D RoPE everywhere (base 10,000 star;
  upstream rotary for DDiT). No 2-D row/column/box structure is exposed
  anywhere — the transposed-hint failure (VC-1a) is the direct evidence of
  what this costs.
- **Train-set size**: star 50,000; 9x9 AP-MDM/monotone/VC-1 Sudoku 100
  (x ~25k transitions for AP-MDM); S4 64 of a 288-universe; released-IRED
  9x9 9,000. Within the AP-MDM/IP/S4 lines the split is clean: every arm
  with <= 100 training instances failed, the 50,000-example star arms
  succeeded. The released-IRED arm breaks any stronger rule: with 9,000
  training puzzles it partially generalized in-distribution (0.769
  standard-val) yet failed the harder OOD set (0.0447) far below its paper —
  so scale is neither sufficient for OOD inference nor, by itself, the
  explanation of the Sudoku gap.
- **Curriculum**: none in any paper or arm.
- **Optimizer schedule**: as tabulated per arm; consistent AdamW 1e-4 lineage.
- **EMA**: star arm EMA 0.999@200 (paper-stated); IRED 9x9 EMA 0.995/10
  (harness choice; paper silent); AP-MDM none (paper silent, upstream off);
  monotone/S4/VC arms none. EMA is not a plausible cause of the Sudoku
  failures: the star raw weights match EMA (0.991 both), and VC-1a train fit
  is perfect without it.
- **Evaluation requiring inference absent from training**: 9x9/S4 held-out
  puzzles require constraint propagation never demonstrated in training
  (training shows *what* the answer is, never *how to derive* it — except
  AP-MDM's solver trajectories, which do demonstrate derivation, on only 100
  puzzles); star test requires the same retrieve-and-reorder skill training
  demonstrates 50,000 times; IRED's harder split is explicitly OOD (givens
  17-34 vs 31-42).

---

## 5. Why star graphs succeed while Sudoku fails — competing accounts tested

1. **Visible keyed-component retrieval vs latent constraint solving —
   SUPPORTED as the leading hypothesis, not a proved exclusive cause.** Star
   answers are chains of node IDs present
   in the prompt; the model retrieves and reorders visible components. Sudoku
   blanks are determined by global constraints and are invisible. The VC-1
   triad is the controlled evidence: same architecture, data, budget; only
   answer visibility/addressability varies. Aligned local oracle: 256/256
   from step 5,000. Transposed (visible but position-dependent under 1-D
   RoPE, no content key): 0/256. Keyed shuffled (visible with content keys):
   0/256 with near-interpolated train fit. D0 then shows the keyed model
   retrieves the *memorized* solution rather than the *addressed* payload
   (counterfactual payload 0.00017 vs solution 0.99914 on train).
   The star-versus-Sudoku contrast itself is *not* controlled: visibility
   co-varies with training-set scale (50,000 vs 100), interface (edge-list
   prefix vs 4-token-cell canvas), and process (variable-length with
   termination vs fixed canvas), so it cannot attribute causality alone;
   VC-1 is what supplies the within-task control.
   Falsifier check: if this account were wrong and the blocker were, say,
   canvas length or digit vocabulary, the aligned arm could not have solved
   every held-out puzzle immediately on the identical canvas.
2. **Train-set scale / memorization attractor — SUPPORTED within the
   AP-MDM/IP/S4 lines, coupled to #1, but not a universal rule.**
   Every arm on those lines has <= 100 training instances and memorized (D0;
   S4 train exact 1.0
   with counterfactual payload 0.10-0.18); the 50,000-example star arms
   generalized. Memorization is the gradient-friendlier
   solution whenever it exists; retrieval/inference is only learned when the
   training set is too large to memorize or the answer is already visible.
   The released-IRED 9x9 arm is the limiting counterexample to any stronger
   claim: 9,000 training puzzles were enough for partial in-distribution
   generalization (0.769 standard-val) but not for the harder OOD set
   (0.0447 vs the paper's 62.1%), so scale alone neither explains the Sudoku
   gap nor rescues OOD inference.
3. **Sequence/canvas representation (1-D RoPE, no 2-D geometry) —
   CONTRIBUTING, not sufficient alone.** The transposed control failed
   partly because `(r,c)->(c,r)` is not translation-equivariant under 1-D
   relative position (`SUDOKU_VISIBLE_COMPONENT_GATE_20260809.md:87-93`).
   But the aligned arm solves the same canvas perfectly, so representation
   alone does not block local retrieval; its cost is on non-local addressing.
4. **Decoder exposure bias — SECONDARY.** All arms train teacher-forced and
   roll out free-running. If exposure bias were primary, held-out
   teacher-forced accuracy would be high while rollouts collapse; instead
   VC-1a held-out blank accuracy is 0.2073 (one-forward, no compounding), so
   the per-cell conditional itself is wrong on unseen boards. For AP-MDM the
   rollout-length mismatch is more severe (training shows single steps;
   inference chains up to 65,536 with unsupervised halting — 183/256
   timeouts) and is ranked separately as cause #3 in §7.
5. **Capacity — NOT SUPPORTED as binding.** The failing 9x9 policy
   (5,979,146 params) is larger than the succeeding star decoder
   (3,167,360); the tiny 135,813-param S4 policy fits its train set exactly.
6. **Objective/gradient implementation error — FALSIFIED only for the
   enumerated IP objective and the unit-tested paths; residual uncertainty
   retained elsewhere.** What is actually ruled out: an error in the
   permutation-variational IP objective as implemented for star graphs,
   because the enumerated RLOO-surrogate gradient equals the exact ELBO
   gradient to 1e-10 (`tests/test_ip_objective.py:44-70`) and the same code
   wins on star graphs; plus the specific unit-tested paths (four-head
   AP-MDM gradients, `tests/test_losses.py`; IRED mixed second derivative
   reaching parameters, `tests/test_s4_ired.py:32`). What is *not* ruled
   out: fidelity gaps in the pieces that have no enumeration-level check —
   the reconstructed 9x9 AP-MDM sampler/evaluator (audit §2.3), the S4
   AP-MDM port's synthetic corruption distribution (`repro/s4_apmdm.py:115-201`,
   not solver trajectories), the S4 IRED comparator's 0.05 NCE coefficient
   and Transformer substitution (`repro/s4_ired.py:167-184,489`), the custom
   harness around the released IRED code, and the fixed star arm's
   unexplained 0% vs the paper's 24.0% (§3.1). These are distinct
   implementations; one test battery cannot clear them all.
7. **Inadequate train interpolation — FALSIFIED for most arms; the remaining
   cases are now final or explicitly unresolved.** Interpolated and failed
   held-out anyway (post-interpolation failures): FO/AO/MDM/LO (S4), VC-1a
   (both hint variants), VC-1c, monotone fixed_ar, the official AP-MDM 100k
   arm, and — newly archived — the S4 ao-ip-wd0 and ip-wd0 arms (train exact
   and train payload 1.0 from the 100k gate through 1M; held-out exact 0.0
   at every gate). Final **non-interpolation** results: S4 AP-MDM (completed
   1M with train exact still climbing at 0.50/0.67; no generalization
   verdict drawable) and monotone random_insertion (closed at its 10k
   endpoint, train digit 0.734). Unresolved by durable evidence: the S4 IRED
   arms — locally archived gates are stale (10k, pre-relocation) and the
   post-300k interpolation is only remotely observed (§3.3), so this review
   asserts nothing about their current fit. Concluding held-out meaning from
   any non-interpolated arm would repeat the exact confusion the spec
   forbids.
8. **Optimizer/grokking horizon — OPEN for 9x9, UNFAVORED by S4.** Star arms
   transitioned only after ~40,000 updates (noted in
   `SUDOKU_VISIBLE_COMPONENT_GATE_20260809.md:68-69`), so 78,100 updates on
   100 puzzles is not a proven-sufficient horizon. But the S4 arms ran 1M
   updates on 64 puzzles — 15,625 presentations per puzzle — with held-out
   exact flat (fo-arm-wd001 held-out exact oscillates 0.078-0.195 across the
   1k->1M gates with no grokking-shaped jump;
   counterfactual payload never rises). If grokking were coming, S4 is where
   it should have appeared first.
9. **Evaluator mismatch — FALSIFIED.** The same monotone evaluator scores the
   aligned oracle 256/256; the AP-MDM failure is mechanical (non-termination,
   213/256 malformed states), not a scoring artifact; and exact-match on
   grids has no plausible reading under which 0/256 valid is a pass. The one
   true ambiguity is what the AP-MDM *paper* scored (undefined "accuracy",
   unspecified test set) — that cuts against the paper's reportability, not
   against our evaluator.

---

## 6. What the S4 keyed-payload task actually establishes

S4 is **both** a fair mechanism probe **and** an unnecessarily difficult
conjunction, and the completed arms show which half currently dominates.

- As a probe it is well-designed for distinguishability: the primary outcome
  (arbitrary keyed-payload following) is separable from Sudoku validity, the
  counterfactual condition removes the answer from the payload, and the
  no-hint/unkeyed ablations isolate key usage (`repro/small_grokking.py:538-679`).
- But as configured it is a **conjunction that collapses to its easier half**:
  the payload *is* a valid Sudoku solution drawn from only 64 training grids,
  so memorizing the 64 solutions strictly dominates learning key-addressed
  retrieval. The measured signature is exactly that collapse: train rollout
  exact 1.0 with train one-forward payload accuracy 0.53 and counterfactual
  payload 0.16 (fo-arm-wd001, §3.3). The model solves training Sudokus and
  partially generalizes Sudoku *structure* (held-out exact 0.19-0.23) while
  treating the keys as noise (key-ablation delta only 0.11).
- **Sudoku-solution generalization vs arbitrary payload following are
  empirically distinct outcomes here**, as the spec's bookkeeping fact
  asserts: an arm can be at 100% solution exactness and ~50%/16%
  normal/counterfactual payload accuracy simultaneously. Any future S4 claim
  of "retrieval" must therefore be stated against the counterfactual payload
  metric, never against solution accuracy.
- What S4 has *not* yet established: anything about IRED (mid-run; remote
  audit only, §3.3) or about the AP-MDM family (its arms completed 1M
  *without interpolating* — a non-interpolation result carries no
  generalization verdict). The wd0 insertion arms did complete after the
  first pass of this review, and they extend the same signature: train exact
  **and** train payload 1.0 with train counterfactual payload 0.0, held-out
  exact 0.0 at every gate (§3.3) — even with keys and a variable-length
  process, the 64-solution train set is memorized rather than addressed.
  S4 still cannot separate "retrieval is unlearnable for this
  architecture/objective" from "memorization was available", because no
  completed arm was denied the memorization shortcut. Gate D1 below exists
  to make that separation.

---

## 7. Ranked causes, confidence, and falsifiers

1. **Task-mechanism gap: Sudoku requires latent multi-step constraint
   inference; the successful tasks reward retrieval/reordering of visible
   components. Confidence: high as the leading hypothesis; not proved
   exclusive.** Citations: VC-1 triad artifacts (§3.2);
   star 99.1% vs Sudoku 0/256 at matched 78,100 updates; D0 counterfactual
   split; S4 counterfactual payload at or below chance after 1M steps across
   all completed families, including the keyed insertion arms (§3.3).
   Standing confounds: the star/Sudoku contrast co-varies visibility with
   dataset scale, task interface, and fixed-canvas vs variable-length
   process; the exclusive causal reading would require the D1/D3-style
   controls, not just VC-1. An observation that
   *would* falsify it: a puzzle-only arm reaching high held-out exact at any
   budget, or a keyed arm following counterfactual payloads.
2. **On the AP-MDM/IP/S4 lines, tiny training sets make memorization the
   dominant attractor; nothing in these objectives forces algorithmic
   generalization. Confidence: high on those lines.** Citations: train-set
   sizes §4.4; D0 memorization split; S4 train exact/payload 1.0 vs
   counterfactual ~0; star's 50,000-example generalization. Falsifier: any
   <=100-example arm generalizing held-out without visible answers — not
   observed. Boundary: the released-IRED 9x9 arm (9,000 training puzzles)
   partially generalized in-distribution yet failed the harder OOD set
   (§3.4), so this cause is stated for the small-sample lines only and does
   not claim that more data alone fixes Sudoku.
3. **For the official AP-MDM 9x9 arm specifically: unsupervised halting plus
   extreme rollout composition, on top of #1-#2. Confidence: medium-high.**
   Citations: audit §2.6 (halting never supervised; `EOS` removed at
   `sudoku_generator.py:611-614`); released sampler unusable
   (`train/diffusion.py:1196-1263`, `sudoku.yaml:116`); 183/256 timeouts,
   46,884 mean forward passes (`summary_upstream-paper-100k-seed42.json`).
   The model learned the transition operator (val nll 3.5e-4) but not when to
   stop, and 100k-of-1M steps leaves the paper's own convergence clause
   untested. Falsifier: a teacher-forced per-step evaluation showing low
   per-step accuracy would move the cause back to #1/#2 entirely — that
   measurement is gate D2.
4. **1-D positional representation without board geometry handicaps
   non-local addressing. Confidence: medium.** Citation: transposed-vs-aligned
   contrast (§3.2). Falsifier: keyed shuffled *should* have rescued retrieval
   if position were the only issue — it did not, so representation is not the
   sole binding constraint.
5. **Exposure bias from purely teacher-forced training. Confidence:
   medium-low (secondary).** Citation: no self-rollout training path exists
   in `repro/`; VC-1a one-forward blank accuracy 0.2073 shows the conditional
   itself fails before compounding matters. Falsifier of primacy: the
   aligned arm free-runs 57 steps perfectly — compounding alone does not
   destroy a correct conditional.
6. **Objective/gradient implementation error. Confidence: low for the
   enumerated IP objective and unit-tested paths; unresolved for the
   unenumerated ports** (§5, item 6: reconstructed 9x9 sampler, S4 AP-MDM synthetic
   corruptions, S4 IRED coefficient/architecture substitutions, the custom
   IRED harness, and the star fixed arm's 0% vs paper 24.0% — localized,
   not systemic, but not cleared).
7. **Insufficient optimization horizon / grokking not yet reached.
   Confidence: low-medium.** Unfavored by the flat 1M-step S4 curves across
   six completed arms in four families; cannot be
   excluded for 9x9 (78.1k updates) or for the official arm (100k of 1M).
8. **Evaluator mismatch. Confidence: very low** (§5.9).
9. **EMA / optimizer hyperparameters. Confidence: very low** (§4.4 EMA
   analysis; wd 0 vs 0.01 made no qualitative S4 difference: exact
   0.195 vs 0.1875, 0.211 vs 0.234, 0.039 vs 0.047).

The honest residual uncertainty: causes #1 and #2 are entangled — every
small-train-set task also lacked visible answers, and the one large-train-set
success also had visible answers. Gates D1 and D3 are designed to decouple
them.

---

## 8. Did any chemical-language experiment run?

**No.** There is no GuacaMol/ChEMBL code in this repository, in
`/home/ubuntu/AP-MDM/`, or in `/home/ubuntu/upstream-ILM-20260809/`
(case-insensitive search: no hits), no chemical artifacts under
`/home/ubuntu/apmdm-official-data/`, and the contract's G2 gate
(`IP_REPRO_CONTRACT_20260809.md:82-95`) was never entered. The IP paper's
GuacaMol claims (IP with classifier termination: 97.4 validity / 97.3 V.U. /
95.6 V.U.N. / 97.2 KL / 89.2 FCD — `sec_eval.tex:158-191`, table
`tab:moses_guacamol`) and its paired-parentheses ordering observation remain
**unreplicated and untested**. Per the contract, the GuacaMol hyperparameter
table (`appendix.tex:319-336`) omits decoder width, heads, batch size,
tokenizer, max length, steps, and seeds, so G2 would be a reconstruction even
if entered. Nothing in this review or any prior unit should be read as
evidence about chemical generation.

---

## 9. Test suite: command, result, and the one pre-existing failure

Command (this machine, 2026-08-10, worktree at `50560f7`, clean):

```
OMP_NUM_THREADS=4 /home/ubuntu/apmdm-repro-env/venv/bin/python -m pytest tests -q
```

Result: **210 passed, 1 failed, 1 skipped, 4 warnings in 86.28 s**
(python 3.10.12, torch 2.7.0, pytest 9.1.0). The skip is the
FlashAttention-vs-SDPA agreement test, environment-gated because
`flash_attn` has no wheel for this stack (`tests/test_model.py`; documented
in `SUDOKU_REPRO_IMPLEMENTATION_REPORT_20260808.md` §5).

The failure is `tests/test_manifest_and_config.py::test_source_hashes_cover_every_repro_module`
(`tests/test_manifest_and_config.py:168-174`). Root cause, verified against
git history: `repro/manifest.py:41-75` (`HASHED_SOURCES`) lists 22 modules
and was never extended as the star-graph, S4, and IRED machinery landed. At
HEAD, 33 `repro/*.py` files exist; the 11 unbound modules are
`ip_objective.py`, `ip_star_data.py`, `ip_star_eval.py`,
`ip_star_eval_checkpoint.py`, `ip_star_model.py`, `ip_star_train.py`,
`keyed_diagnostics.py`, `s4_apmdm.py`, `s4_insertion.py`, `s4_ired.py`,
`small_grokking.py`. Commit archaeology (read-only `git show`/`ls-tree`):
listed=22/present=22 at `14f8e02` (pass); listed=22/present=23 at `47ed1ed`
(first failing commit — it added `repro/ip_objective.py` without extending
the list); listed=22/present=33 at `8091b6e` and HEAD. **The
suite has been red on this test for the entire star-graph/S4/IRED program,
and no prior report recorded it.** Practical impact is bounded but real:
manifests still bind the git commit and a whole-tree dirty flag
(`repro/manifest.py:96-105`), so a dirty tree is detected — but per-file
sha-256 binding covers only the 22 listed modules, and the seal's
defense-in-depth claim silently stopped covering the modules added after
`14f8e02`.

Disposition under this spec: fixing it requires adding the 11 paths to
`HASHED_SOURCES` in `repro/manifest.py` — production code, which this review
is forbidden to touch. The PASS gate "existing repository tests remain green"
is therefore recorded as **not satisfiable at the reviewed commits without a
prohibited change**; the failure is pre-existing (not introduced by this
review), root-caused above, and the one-line-per-module fix plus its
regression expectation (extend the list; the test then passes unmodified) is
the recommended first action of the next authorized code unit.

Read-only audit checks performed for this review (no files written):
recomputation of VC-1 held-out blank-cell accuracies from
`eval_rows_step_000078100.jsonl` against the hash-verified 10,000-puzzle test
array (hidden 0.20731, oracle 0.21212, keyed 0.25137, aligned 1.0 —
consistent with `SUDOKU_VISIBLE_COMPONENT_GATE_20260809.md:84`); direct
re-reading of every metric record cited in §3; git-history archaeology for
the failing test; structure verification of `s4/supervisor/status.json`,
D0 `diagnostics.json`, VC-1 `curve_summary.json` files, and the insertion
R1 curve summaries. Second pass (2026-08-10, snapshot 05:37:46Z): re-read of
the S4 supervisor status and of the newly synced `s4/panel-58703d4/`
(ao-ip-wd0, ip-wd0) and `s4/panel-89d93df/` (ap-mdm-wd0, ap-mdm-wd001)
telemetry, confirming completion at step 1,000,000, the insertion arms'
train exact/payload 1.0 with train counterfactual payload 0.0, and the
AP-MDM arms' non-interpolation (train exact 0.500/0.672 at 1M); confirmed
that no post-10k IRED or LO-ARM eval artifacts are durably archived locally.

---

## 10. Evidence gaps and clean negatives (recorded, not papered over)

- **The still-running S4 arms have no locally synced held-out evaluations.**
  ao-ip-wd001 (@983k), ip-wd001 (@854k), lo-arm-wd{0,001}-relocated
  (@~680-699k), and ired-wd{0,001}-relocated (@~528-529k) have, locally,
  only 3-entry supervisor train-telemetry tails
  (`s4/supervisor/status.json`, snapshot 2026-08-10T05:37:46Z) and <=10k
  pre-relocation gates. For IRED specifically, the operator's post-300k
  remote audit (train exact/valid/payload 1.0 both wd arms; wd=0.01 held-out
  exact 0.0, counterfactual payload 0.23047) is **remotely observed but not
  yet durably archived locally**; this review neither asserts nor denies it.
  Nothing about the running arms is a verdict.
- **The spec's bookkeeping line "AO-IP wd=0.01 is just below exact
  interpolation at its latest measured gate" remains not durably verified
  for that specific arm** (still running; only supervisor train telemetry
  locally). Its wd=0 sibling has since completed and *is* locally archived
  at full interpolation (train exact/payload 1.0 from the 100k gate through
  1M, `s4/panel-58703d4/ao-ip-wd0-s42/telemetry.jsonl`), so the family's
  capacity to interpolate is demonstrated; the per-arm wd=0.01 claim awaits
  durable closure.
- **The paper's star-graph FO-ARM 24.0% vs our fixed arm's 0%** is
  unresolved (§3.1); no author code or config exists to decide it.
- **The AP-MDM paper's convergence point** (which step, which criterion,
  which of the two architectures produced 99.28%) is undeterminable from
  released artifacts; our 100k-step result bounds but does not settle the
  claim.
- **IRED's unreproduced gap** (76.9/4.47 vs 99.4/62.1) cannot be attributed
  from available evidence: the paper omits the training-set size, step sizes,
  EMA, and metric definition; the authors' code was run inside a custom
  sealed harness (a deviation), and 50k steps matches the paper's stated
  budget. Both "paper overstates" and "harness differs in an unstated
  detail" remain open.
- No aggregate cost totals exist in any ledger (hourly rates only);
  `ip-star-runpod-82i4fwoch34p40/overfit-learned-1923fab/` is an incomplete
  control (manifest, no result).

---

## 11. Discriminating experiment sequence (proposed, predeclared, NOT launched)

Four bounded gates, cheapest decisive order. Each has predeclared outcomes
and a stop rule; none is an open-ended tuning sweep. All reuse existing
frozen splits, evaluators, and sealed-manifest tooling.

**D1 — Random-payload S4 control (decides memorization-shortcut vs
retrieval-inability; separates causes #1 and #2).** One arm: the S4 FO-ARM
keyed canvas unchanged, except training payloads are arbitrary uniform
16-digit strings, *not* Sudoku solutions; held-out split likewise arbitrary.
Same 135,813-param policy, batch 256, seed 42, wd 0.01, 100,000 updates
(~30 min at the measured ~81 updates/s, `SMALL_SUDOKU_GROKKING_LADDER_20260809.md:122-124`),
evals at 1k/3k/10k/30k/100k. Predeclared: **pass** = held-out payload exact
>= 50% by 100k (memorization shortcut was the blocker; S4-as-Sudoku is an
unnecessarily hard conjunction; fix the task, not the model). **Fail** =
held-out payload exact < 5% at 100k (content-addressed retrieval is not
learnable at this scale even with no shortcut; escalate to
representation/objective causes). Stop at the first gate where payload exact
exceeds 50% or at 100k.

**D2 — Teacher-forced per-step audit of the official 100k AP-MDM checkpoint
(decides transition-operator vs composition/termination for cause #3).** No
training. Replay held-out solver trajectories through the frozen checkpoint
teacher-forced: at each recorded state, measure unmask-token accuracy and
remask precision/recall; separately score the model's halt behavior on
terminal states. Bounded: one GPU-day worst case on a 1,000-puzzle seeded
subsample. Predeclared: per-step unmask accuracy >= 99% with halt accuracy
< 50% on terminal states => composition/termination is the binding failure
(rank cause #3 above #1 for AP-MDM and prioritize supervised-halting data);
per-step accuracy < 95% => the transition operator itself does not
generalize (cause #1/#2 confirmed for AP-MDM; a 1M-step continuation is
*not* warranted). Stop after the subsample; do not iterate thresholds.

**D3 — 9x9 data-scale control (decides cause #2 at the paper's scale).** One
monotone `random_insertion` arm, identical to VC-1a except 10,000 fresh
training puzzles generated by the checked-in upstream generator under a new
declared seed, with the existing leakage gate proving zero overlap with the
10,000-puzzle verdict array. Matched 78,100 updates, seed 42, same
evaluation. Predeclared outcomes, all three exclusive: **scale causal** =
train interpolated (digit accuracy >= 0.99) *and* held-out exact >= 5% at
78,100 => data scale is causal at 9x9 (cause #2 promoted over #1 for the
monotone family); **mechanism gap** = train interpolated *and* held-out
exact 0/256 => the mechanism gap persists at 100x data and #1 is confirmed
independent of scale; **inconclusive** = train not yet interpolated at
78,100 (as the 10k-update R1 arm was not, at 0.734) => neither binary
reading is licensed, and the only authorized continuation is a single
matched extension to 234,300 updates (3x) with identical thresholds — if
still not interpolated there, the gate scores inconclusive regardless and
stops. No other extension, no threshold edits.

**D4 — Geometry control (decides cause #4), only if D1 fails.** One VC-1c
keyed arm with added row/column/box coordinate embeddings on the 4-token
cells (the geometry helpers already exist as scalar functions,
`SUDOKU_REPRO_IMPLEMENTATION_REPORT_20260808.md` §11), matched 78,100
updates. Predeclared: held-out keyed exact > 0 or counterfactual payload
> 50% => representation was binding for grid addressing; no movement =>
retrieval inability is objective-level, not positional. Skip D4 entirely if
D1 passes (the task design, not position, was the issue).

Stopping rules for the sequence: run D1 first; D2 is independent and may run
in parallel; D3 runs only if D1 fails (i.e., if retrieval itself is
suspect — otherwise the memorization account is confirmed and D3's answer is
already priced in); D4 only after a D1 failure. No gate authorizes a follow-on
sweep; each produces a single predeclared bit.

---

## 12. Bottom line

The leading hypothesis for the star/Sudoku divergence is a task-mechanism
gap: the star task's answers are visible, keyed, short components and its
training set (50,000) is too large to memorize, while the AP-MDM/IP/S4
Sudoku arms faced the conjunction of invisible answers and a memorizable
training set (100 or 64 instances). The hypothesis is supported by the
controlled VC-1 interventions and by counterfactual diagnostics that show
memorized solutions beating supplied payloads on every completed arm — but
it is not an isolated, proved cause: visibility, dataset scale, task
interface, and fixed-canvas versus variable-length process all co-vary in
the star-versus-Sudoku contrast, and only VC-1 varies visibility alone.
On the small-sample Sudoku lines, nearly every arm fit its training set and
then failed held-out — post-interpolation memorization. The exceptions that
bound the claim: the S4 AP-MDM arms completed 1M updates without
interpolating at all; the S4 IRED arms' post-300k interpolation is remotely
observed but not durably archived; and the released-IRED 9x9 arm — with
9,000 training puzzles, not a tiny set — still under-reproduced its paper
badly on the harder OOD split (0.0447 vs 62.1%), so neither scale nor the
mechanism account alone explains every Sudoku failure. The AP-MDM 99.28%
claim additionally rests on a sampler/evaluator the authors did not release
and a parameter count its own architecture contradicts; our run kept the
authors' training stack faithful and still scored 0/256 through the
reconstructed evaluator at 100k steps. The next decisive evidence is not
more Sudoku training — it is the D1 random-payload gate, which costs under
an hour and cleanly separates "memorization shortcut" from "retrieval
unlearnable".
