# AP-MDM Sudoku: Paper / Code Audit

Date: 2026-08-08
Auditor: independent Claude Opus 5 reproduction implementer
Scope: everything needed to reproduce the Sudoku result of arXiv:2510.06190v2
("Beyond Next-Token Prediction: Any-Process Masked Diffusion Models" / AP-MDM)
from the official repository at upstream commit `836b79b3`.

## 0. Authenticated inputs

All four authoritative inputs were re-hashed locally and match the
specification exactly.

| Artifact | SHA-256 | Status |
| --- | --- | --- |
| `2510.06190v2.pdf` | `efec4ac4…4eba9c` | matches |
| `2510.06190v2-source.tar` | `891ef81f…ec085c` | matches |
| `dataset/sudoku/sudoku-100.npy` | `a3e26287…b676e0` | matches, shape `(100, 325)`, dtype `uint64` |
| `dataset/sudoku/sudoku-test-10k.npy` | `36f57185…9eeb7c` | matches, shape `(10000, 325)`, dtype `int64` |

Paper sources read in full: `iclr2026_conference.tex` (main text and results
table), `method.tex` (architecture, both training regimes, Algorithm 1),
`exp.tex` (implementation details), `examples.tex` (Sudoku state
representation, atomic operations, data statistics), `tf_arc.tex` (encoder
definition). Official code read in full: `dataset/sudoku/{sudoku_generator,
sudoku_solver, sudoku_loader, sudoku_verifier}.py`, `train/apmdm_dataloader.py`,
`train/diffusion.py`, `train/models/dit.py`, `train/dataloader.py`,
`train/main.py`, `train/configs/sudoku.yaml`, `train/scripts/sudoku.sh`.

### 0.1 Leakage

Measured directly, three independent ways, over the full arrays:

| Overlap level | Count |
| --- | --- |
| raw 325-int rows (dtype-normalised) | **0** |
| decoded 9x9 puzzle grids | **0** |
| decoded 9x9 solution grids | **0** |
| (puzzle, solution) pairs | **0** |

All 100 training rows are distinct; all 10,000 test rows are distinct. The
specification's independent measurement is confirmed and is now re-checked in
code on every evaluation (`repro.evaluator.check_evaluation_preconditions`
refuses to score anything if overlap is non-zero).

Difficulty, measured from the givens count:

| Split | givens min | max | mean |
| --- | --- | --- | --- |
| train (100) | 21 | 25 | 22.99 |
| test (10,000) | 20 | 28 | 24.22 |

The training instances are on average harder (fewer givens) than the held-out
instances, consistent with the appendix's "100 hard Sudoku puzzles". Note the
main text calls the same 100 instances "moderately hard" while `examples.tex`
calls them "hard"; this is cosmetic but is recorded for completeness.

---

## 1. Discrepancies the specification requires to be surfaced

### 1.1 Architecture and parameter count (spec item 1) — **unresolved in the paper itself**

`exp.tex` states: *"For Sudoku, we use 6 layers, 4 attention heads, hidden
dimension d=256, feed-forward dimension 4d=1024, maximum sequence length 400,
vocabulary size 31."* The results table in `iclr2026_conference.tex` states
`AP-MDM | 100 | 1.2M | 99.28%`, i.e. approximately 1.2M parameters.

The checked-in `train/configs/sudoku.yaml` instead declares 4 blocks, 2 heads,
hidden size 128, `cond_dim` 64.

Exact parameter counts, computed from the upstream module structure
(`train/models/dit.py`: `EmbeddingLayer` + `TimestepEmbedder` + N x `DDiTBlock`
with adaLN + `APMDMQuadrupleHead`), and cross-checked against a closed-form
formula (`repro.model.analytic_parameter_count`):

| Configuration | blocks | heads | d | ffn | cond_dim | exact parameters |
| --- | --- | --- | --- | --- | --- | --- |
| Paper-faithful (`train/configs/sudoku_paper.yaml`) | 6 | 4 | 256 | 1024 | 128 | **6,050,594** |
| Paper-faithful, if `cond_dim=64` were assumed instead | 6 | 4 | 256 | 1024 | 64 | 5,399,202 |
| Historical (`train/configs/sudoku.yaml`, unchanged) | 4 | 2 | 128 | 512 | 64 | **1,035,426** |

**Finding.** The paper's stated Sudoku architecture cannot have ~1.2M
parameters; at 6x4x256 it has ~5.4M–6.1M depending on the unstated `cond_dim`,
i.e. 4.5x–5x the reported figure. The reported ~1.2M is within ~14% of the
*historical* 4x2x128 configuration (1,035,426). The paper's own comparison
sentence ("outperforming ARM and any-order MDM with 5x parameters") multiplies
the reported 1.2M by 5 to characterise the baselines, so the 1.2M figure is
load-bearing in the paper's argument, not incidental.

This is an internal inconsistency in the paper that the code does not resolve.
Per the specification, this replication (a) leaves `train/configs/sudoku.yaml`
byte-identical, (b) adds `train/configs/sudoku_paper.yaml` implementing the
architecture the paper *describes* (6/4/256/1024) as the fixed primary arm, and
(c) reports both exact counts here and in the sealed manifest. No attempt is
made to reconcile the two by editing either config. `cond_dim` is not stated by
the paper; 128 is taken because every other checked-in model config
(`tiny`/`small`/`medium`) uses 128, and the choice is recorded in the manifest.

Deciding *which* of the two architectures the 99.28% number came from is not
possible from the released artifacts. If the primary arm fails to replicate,
the historical 4x2x128 architecture is the first declared follow-up arm.

### 1.2 Transitions per puzzle (spec item 2) — **25,022.6 confirmed exactly**

`examples.tex` states both *"each Sudoku puzzle produces 1421.3 state-transition
tuples … on average"* and *"we generated training data from 100 hard Sudoku
puzzles, each yielding 25,022.6 training state-transition tuples on average"*.

Measured by running the checked-in generator unmodified over all 100
authenticated source puzzles:

| Quantity | Measured |
| --- | --- |
| source puzzles | 100 |
| total solver-derived transitions | 2,502,258 |
| transitions per source puzzle, mean | **25,022.58** |
| median | 24,175 |
| standard deviation | 4,017.46 |
| min / max | 19,572 / 38,180 |

The mean reproduces the paper's 25,022.6 to the published precision (difference
-0.02). The second figure, 1421.3, is ~17.6x smaller and is not reproduced by
these 100 puzzles under any reading — but it is explained. Running the same
generator over a seeded random sample of 1,000 *held-out* puzzles gives:

| Quantity | Held-out sample (n=1000, seed 20260808) |
| --- | --- |
| transitions per puzzle, mean | 1,063.7 |
| median | 433 |
| 90th / 99th percentile | 2,417 / 9,033 |
| min / max | 108 / 22,082 |

So the two published statistics describe two different puzzle populations,
exactly as the appendix's wording implies: ~10^3 transitions for the general
(evaluation-grade) population and ~2.5 x 10^4 for the 100 hard training
instances. The general figure is the same order of magnitude as the measured
1,063.7 (the paper's 1421.3 presumably comes from a different sample of the
same population); the training figure is exact. **The claim that 100 source
puzzles expand into ~2.5M supervised transitions is confirmed.**

This measurement is used only to justify the frozen inference ceiling and to
project evaluation cost. It selects no threshold, and it is a property of the
symbolic solver, never of a model.

Per-operation totals across all 2,502,258 transitions:

| Operation | Count |
| --- | --- |
| `assign_remask` / `assign_unmask` | 900,238 each |
| `branch_remask` / `branch_unmask` | 88,073 each |
| `contradiction_remask` / `contradiction_unmask` | 87,606 each |
| `backtrack_modified_remask` / `backtrack_modified_unmask` | 87,606 each |
| `skull_to_normal_step1` / `step2` | 87,605 each |
| `skull_to_branch_step1` / `step2` | 1 each |

Every atomic operation described in `examples.tex` is exercised. Each
remask step is paired one-to-one with its unmask step, as the appendix
describes ("we generate a 2-step transition"). Note how rare
`skull_to_branch` is: exactly one occurrence in 2.5M transitions, from a single
puzzle. Totals: 6,788,523 remask labels set and 6,788,523 unmask target
positions, and **zero** insert or delete labels (see 2.2).

**Terminology.** Throughout this replication, the 100 instances are *source
puzzles*; everything the solver emits is a *solver-derived transition*.
Transitions are deterministic expansions of their source puzzle and are never
described as independent puzzles or independent samples. The accounting API
(`repro.trajectories.accounting_from_counts`) reports
`n_source_puzzles` separately from
`total_solver_derived_transitions` and keeps the per-puzzle vector, and a test
asserts the disclaimer is present.

### 1.3 Vocabulary 31 vs 32 (spec item 3) — **resolved, no change made**

`examples.tex` says *"The vocabulary consists of 32 tokens"*; `exp.tex`, the
results table and `train/configs/sudoku.yaml` all say 31.

Authenticated against the generator: `APMDMSampleGenerator` declares
`VOCAB_SIZE = 32` with ids 0..31, where id 31 is `[EOS]`. The current
generator's `on_end` is annotated *"Modified: Remove final expand+eos
operation"* and emits nothing, so `[EOS]` never appears in any generated
sequence. The maximum token id reachable in data is therefore 30 (`[SEP]`), and
31 embedding rows suffice. Measured over a full trajectory: token ids observed
are a subset of 0..30, and `[EOS]` never occurs.

So both numbers are right about different things: 32 is the declared token
inventory, 31 is the effective vocabulary of the data. The replication uses 31
and records both numbers in the manifest. The vocabulary is not changed to force
agreement.

### 1.4 Stale absolute paths (spec item 4)

`train/configs/sudoku.yaml` contains `/workspace/data`,
`/workspace/projects/APMDM/data/apmdm_training_dataset.pkl.gz` and
`/workspace/projects/APMDM/data/vocab_cache.pkl`; `train/scripts/sudoku.sh`
hard-codes `/workspace/env/bin/python` and three more `/workspace/...` paths;
`train/dataloader.py` and `train/apmdm_dataloader.py` carry `/workspace/...`
defaults. None of these exist on the target machine.

The historical config keeps its stale paths (a test asserts they are still
there, so they cannot be silently rewritten). The replication introduces a
portable data root (`$APMDM_REPRO_DATA_ROOT`, default `~/apmdm-repro-data`) and
a guard that refuses to write bulk artifacts inside the Git work tree. No
scientific setting is changed by this.

### 1.5 Baselines are imported, not trained (spec item 5)

`iclr2026_conference.tex` explicitly states *"Results of ARM and AO-MDM are
taken from Kim et al."*. Nothing in this repository trains ARM or any-order MDM
for Sudoku. This replication therefore tests **only** the AP-MDM 99.28% claim
and makes no controlled comparison against 87.18% / 89.49%. The manifest and
report state this.

---

## 2. Additional discrepancies found during the audit

These are not in the specification's list but are material to reproducing the
result, and several block a naive reproduction outright.

### 2.1 The Sudoku state is 4 tokens per cell, not 3

`examples.tex`: *"represented as a sequence of 324 tokens, where each cell is
encoded using 3 consecutive tokens: (value, color, marker)"*. 81 cells x 3
tokens is 243, not 324. The generator
(`_encode_current_state_from_grid`, `_get_cell_index`) uses **four** tokens per
cell — `(value, color, marker, separator)` — and 81 x 4 = 324. The prose
under-counts by omitting the `[SEP]` delimiter it separately lists in the
vocabulary. The 324 figure, the code, and this replication agree; the "3
consecutive tokens" phrase does not.

### 2.2 The Sudoku task never exercises insert or delete

Measured over every transition generated from the source puzzles: the insert
(`e_star`) and delete (`c_star`) label arrays are **identically zero**, for every
transition of every puzzle. Every Sudoku atomic operation is expressed with
remask + unmask only. Consequences:

* the sequence length is invariant at 324 throughout training and inference;
* the insert and delete heads are trained exclusively on negative targets, so
  the paper's "four operations" are, for Sudoku, two supervised operations plus
  two heads pinned off;
* the mandated test *"each loss/head receives finite nonzero gradients on a
  fixture containing the relevant operation"* cannot be satisfied from Sudoku
  data. This replication satisfies it with a deterministic synthetic fixture
  (`repro.smoke.four_operation_fixture`) and states so;
* any insert/delete firing at inference time is off-distribution behaviour. The
  sampler records it as an anomaly rather than absorbing it silently.

This does not contradict the paper — the paper's Sudoku illustration is about
`remask` — but a reader of `exp.tex` would reasonably expect all four heads to
be exercised, and the released Sudoku pipeline does not exercise two of them.

### 2.3 The official sampler cannot solve Sudoku puzzles

`train/diffusion.py::_sample_apmdm` is *unconditional*: it starts from
`self._sample_prior(batch, config.model.length)`, i.e. an all-`[MASK]` sequence
of length 400, and never sees the puzzle. The paper's Algorithm 1 is
prompt-conditioned (`Require: input prompt x`; `x_0 <- x`). Three further
problems compound this:

1. its stopping criterion is *"generation of an `[EOS]` token"*, and the Sudoku
   generator never emits `[EOS]` (see 1.3), so the criterion can never fire;
2. `sampling.max_steps: 50` in `train/configs/sudoku.yaml` is orders of
   magnitude below the number of transitions a hard puzzle needs
   (tens of thousands, see 1.2);
3. it materialises a `.item()` per token per step in a Python double loop.

There is no checked-in Sudoku evaluation harness of any kind: nothing decodes a
generated state back to a grid, nothing compares against the held-out array,
nothing computes accuracy. **The 99.28% number cannot be recomputed from the
released code as-is.** Reconstructing the evaluator from Algorithm 1 is the
single largest inference this replication has to make, and it is the largest
threat to a faithful comparison.

`repro.sampler` implements Algorithm 1 directly: start from the encoded puzzle,
one forward pass per step, apply `g` with thresholds `(tau_r, tau_i, tau_d)`,
terminate at a fixed point or at a frozen step ceiling. It is validated by
replaying a real solver trajectory as an oracle policy and checking the sampler
reaches the ground-truth solution.

### 2.4 Three inconsistent implementations of the transition function `g`

| Source | delete condition | remask priority |
| --- | --- | --- |
| paper Algorithm 1 (`method.tex`) | `ctrl[3]==1 **and** x_i == MASK` | remask > unmask > keep |
| `train/diffusion.py::_sample_apmdm` | `c_star[i] and x[i] == mask_index` | remask > unmask > keep |
| `dataset/sudoku/sudoku_verifier.py::simulate_apmdm_inference` | `c_star[i]==1` (no mask check) | remask > unmask > keep |

The official verifier deletes *any* position whose delete bit is set, including
non-mask positions, contradicting Algorithm 1 and the sampler. For Sudoku this
is inert (all delete bits are zero), but the verifier is the artifact a reader
would use to check generated data, so the divergence matters. This replication
implements the paper's condition (`repro.transition.apply_transition`) and uses
that one function for generator validation, for inference, and in tests.

**Positive result:** with the paper's `g`, every transition the official
generator produces satisfies `g(x_k, y*, ctrl*) == x_{k+1}` exactly, and each
trajectory is a continuous chain (`x_{k+1}` of step *t* equals `x_k` of step
*t+1*), starting from the encoded source puzzle and ending at the ground-truth
solution. The generator and Algorithm 1 agree.

### 2.5 The configured `model.vocab_size` is decorative

`train/diffusion.py` sets `self.vocab_size = self.tokenizer.vocab_size` and
passes *that* to `DIT(...)`; `config.model.vocab_size` is never read by the
backbone. The Sudoku tokenizer
(`train/apmdm_dataloader.py::APMDMTokenizer._ensure_special_tokens`) starts from
the generator's 32-entry vocabulary, appends `[PAD]` and `[BOS]`, and then pads
*"to match checkpoint"* to a hard-coded `target_vocab_size = 34`. So a run of
the historical stack builds a **34-row** embedding while the config, the paper
and `train/scripts/sudoku.sh` all say 31. The three extra rows are unreachable
in the data; the discrepancy is harmless numerically but means the released
config does not describe the model that upstream actually trained, and the
`model.vocab_size=31` override in `train/scripts/sudoku.sh` has no effect at
all. The replication builds exactly 31 rows and binds the number into the
manifest and every checkpoint.

### 2.6 Halting is never supervised

The last transition of a trajectory *produces* the solved grid; no transition
ever *takes* the solved grid as its input `x_k`. The model is therefore never
shown the terminal state and never trained to emit "no operation" on it.
Termination at inference relies entirely on generalisation (the model must
predict all-zero remask on an unseen state). Algorithm 1 leaves the criterion
open ("e.g. sequence convergence, generation of special tokens, or reaching a
maximum iteration limit") and the `[EOS]` option was removed from the generator,
so **fixed-point convergence is the only criterion the released data supports**.
This replication predeclares it, records the termination reason per puzzle
(`fixed_point` / `max_steps` / `length_overflow` / `empty_sequence`), and scores
the terminal state either way. A model that solves the puzzle and then keeps
churning will be scored on the churned state; that is the honest reading of
"the output of the generation process", and the per-puzzle rows retain the step
at which the grid first became complete so the alternative reading can be
audited after the fact.

### 2.7 The historical validation split is not a random sample

`train/apmdm_dataloader.py` splits the flat transition list at
`int(total * train_ratio)` **without shuffling** (`train_ratio: 0.99`).
Transitions are appended puzzle by puzzle, so the resulting "validation" set is
the tail end of the last one or two puzzles' trajectories, not a sample of the
distribution. In the traditional (non-streaming) path the training half is then
shuffled with a hard-coded `random.seed(42)`, independent of `config.seed`.

The replication uses a seeded random split of transitions and labels it exactly
what it is: an **in-distribution transition monitor** over the same 100 source
puzzles, used to watch optimisation, never reported as generalisation. The only
held-out generalisation set is the official 10,000-puzzle array.

### 2.8 Silent data-repair paths in the dataloader

`APMDMDataset._process_item*` truncates `x_k`, `r_star`, `e_star`, `c_star` to
their common minimum length, allows `y_star` to differ by up to 10 and otherwise
truncates it to `min_len + 3`, and returns `None` (dropping the sample) if the
length exceeds `max_length`. Any real length inconsistency would be repaired
into silently wrong supervision rather than raised. Sudoku never triggers this
(all lengths are exactly 324), but the replication validates lengths strictly
and raises instead.

### 2.9 Config fields that do not apply to the supervised Sudoku path

For completeness, the following historical-config fields have no effect on
supervised AP-MDM Sudoku training and should not be read as scientific
settings: `noise.type`, `training.antithetic_sampling`, `training.sampling_eps`,
`sampling.steps`, `sampling.noise_removal`, `model.scale_by_sigma`,
`model.tie_word_embeddings`, `apmdm.remasking_strategy` /
`expansion_strategy` / `contraction_strategy` (the sampler hard-codes
thresholding at 0.5), and `checkpointing.monitor: val/loss` (which monitors the
non-random split of 2.7). `training.ema: 0` disables EMA; the paper does not
mention EMA either way.

### 2.10 FlashAttention is not available on the target stack

`train/models/dit.py` imports `flash_attn` at module scope and uses
`flash_attn_varlen_qkvpacked_func` plus `flash_attn.layers.rotary.
apply_rotary_emb_qkv_`. The available stack is torch 2.7.0 / CUDA 12.8 on an
A10 (compute capability 8.6); `flash_attn` is not installed and there is no
matching prebuilt wheel. Per the specification's item 7 this replication
implements a PyTorch SDPA fallback
(`torch.nn.functional.scaled_dot_product_attention`, `is_causal=False`) plus an
explicit rotate-half RoPE reproducing the exact convention upstream gets from
`apply_rotary_emb_qkv_` when handed half-width cos/sin.

The module structure, parameter tensors and initialisation are otherwise
identical to upstream, so parameter counts and state-dict shapes match. **The
kernels are not claimed to be numerically identical.** A test pins the rotary
convention and checks SDPA against an explicit softmax attention in float64; a
second test, skipped on this machine, compares FlashAttention against SDPA
within a declared tolerance (max |delta| <= 2e-2 in half precision) if
`flash_attn` ever becomes importable.

---

## 3. Ambiguities that remain unresolved

These are inferences this replication had to make. Each is recorded in the
sealed manifest so that a reviewer can overrule it.

1. **Which architecture produced 99.28%** — the described 6x4x256 (~6.05M) or
   the checked-in 4x2x128 (~1.04M, closest to the reported 1.2M). Primary arm:
   the described one. (1.1)
2. **`cond_dim` and `dropout`** for the paper architecture: unstated. Taken as
   128 (repository convention) and 0.1 (historical Sudoku config).
3. **Termination criterion**: fixed-point convergence, since `[EOS]` was removed
   from the generator. (2.6)
4. **Maximum generation steps**: not stated anywhere; the historical config's
   `sampling.max_steps: 50` is unusable (2.3). Frozen at **65,536** forward
   passes per puzzle, i.e. 1.72x the longest symbolic trajectory among the 100
   *training* puzzles (38,180 transitions) and 2.6x their mean. Derived from the
   training split only, never from the verdict set. This ceiling dominates the
   worst-case evaluation cost; the production budget must be frozen against
   measured throughput before any full run.
5. **Accuracy definition**: exact equality with the supplied ground-truth grid.
   The paper says only "accuracy". Validity, givens-consistency and completeness
   are recorded separately for every puzzle so any other definition can be
   recomputed from the immutable rows without re-running the model.
6. **Whether the 10,000-puzzle array is the array the paper scored**: the paper
   says it follows Kim et al.'s setup; the checked-in array is what is available
   and is what is used.
7. **Optimizer-step budget actually used**: the paper says "up to 1M steps or
   until convergence" without naming the convergence criterion or the step count
   at which 99.28% was measured.

---

## 4. What this replication changes, and what it does not

**Unchanged**: `train/configs/sudoku.yaml` (byte-identical, pinned by hash in a
test), `dataset/sudoku/*.py`, `train/diffusion.py`, `train/models/dit.py`,
`train/apmdm_dataloader.py`, `train/main.py`. The official generator is used
unmodified to produce every transition.

**Added**: `train/configs/sudoku_paper.yaml` (paper-faithful architecture,
portable paths, frozen inference ceiling) and the `repro/` package (portable
data pipeline, paper-faithful model with SDPA, supervised objective,
prompt-conditioned Algorithm-1 sampler, exact evaluator, sealed manifest,
telemetry, resumable trainer, fail-closed GPU supervisor, CLI) plus `tests/`.

**Not done in this unit**: no production training run, no GPU smoke, no tuning
of any kind, no baseline reproduction, no claim about ARM or AO-MDM.
