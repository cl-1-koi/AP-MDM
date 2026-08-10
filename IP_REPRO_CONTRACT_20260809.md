# Paper-native Insertion Process reproduction contract

Date: 2026-08-09

## Question

Can the permutation-variational Insertion Process (IP) from Zhang et al.,
*Variational Learning for Insertion-based Generation* (arXiv:2606.02133v3),
reproduce its claimed learned-order advantage when the released equations are
implemented without the fixed-canvas Sudoku specialization?

This arm tests the paper's actual variable-length process: the model chooses
where to insert, what token to insert, and when to terminate. It does not use
remasking, replacement, deletion, MCTS, or solver transitions.

## Upstream availability audit

- No public implementation or checkpoint for Zhang et al.'s IP method was
  linked from arXiv v1, v2, or v3, and none was found in the authors' public
  repositories on 2026-08-09.
- The cited predecessor task repository was subsequently located at
  `https://github.com/dhruvdcoder/ILM` (audited commit
  `6cc27f104fd926c8256aff28682c3fe66050ce77`). It releases the exact hard
  star-graph generator and a `vstar_medium_v2` configuration whose 50,000 /
  100 / 5,000 train/validation/test counts and graph dimensions match the new
  paper. We adapt and attribute that generator, while reconstructing the new
  IP model and objective from Zhang et al.'s equations.
- The standard GuacaMol dataset and evaluator are public, but the paper omits
  the decoder width, attention-head count, batch size, tokenizer details,
  maximum sequence length, training steps/epochs, random seeds, and hardware.
  It only specifies 18 decoder layers, three posterior layers, optimizer/LRs,
  weight decay, EMA, cosine scheduling, and that training ran "to convergence."
- Neither paper nor predecessor code freezes generated split seeds or hashes,
  so this reproduction declares and hashes its own deterministic splits before
  training.

Consequently, results must be called a reconstruction, not an exact upstream
reproduction, unless the authors provide the missing configuration.

## Fidelity gates

### G0: exact finite-state objective

Use tiny variable-length token sequences for which every permutation can be
enumerated. Before training a Transformer, verify:

1. permutation-prefix to insertion-slot mapping is bijective;
2. the exact sum over all insertion trajectories matches the permutation sum;
3. the closed-form expectation over the next index matches enumeration;
4. the two-sample RLOO automatic-differentiation surrogate matches an
   independently enumerated expected gradient within a declared tolerance;
5. the policy-based termination distribution used by G1 is normalized and its
   search-free decoder halts correctly. Classifier-based termination is a
   separate paper variant and is deferred unless G1 requires it.

No GPU training arm may pass G0 on loss reduction alone.

### G1: paper-native star-graph planning

Implement the released hard star-graph task: degree 5, arm lengths uniformly
sampled in [6, 12], randomized junction position, and node vocabulary size 56.
Use the paper-declared planning architecture and optimizer:

- generative Transformer: 12 layers, eight heads, width 128, dropout 0.1,
  SwiGLU and RoPE base 10,000;
- Plackett--Luce posterior: six layers, four heads, width 64;
- two posterior permutations and RLOO;
- AdamW at 1e-4, cosine decay with 1,000 warm-up steps;
- EMA 0.999 after step 200; batch 64; 100 epochs.

Use the predecessor's released split sizes and freeze our generated split
seeds and hashes before observing results. Compare matched FO-ARM and learned
IP arms; the paper reports 24.0% and 83.0% sequence accuracy respectively, so
this is a high-signal mechanistic gate. The table's 26.5% result is a
fixed-canvas AO-ARM, not the variable-length random-policy AO-IP ablation used
later for GuacaMol. Reconstruct AO-ARM separately before comparing to 26.5%; an
AO-IP star-graph arm may still isolate the learned-order mechanism but has no
paper-reported star-graph target. Report exact-match, token accuracy, Hamming
distance, Levenshtein distance, ELBO, RLOO variance, termination errors,
examples/s, and GPU-hours.

### G2: GuacaMol/ChEMBL SMILES

Only enter G2 if G0 passes and G1 shows a replicated learned-order advantage.
Use the official GuacaMol split and hashes. First run a capacity-matched short
curve for FO-ARM, AO-IP, and learned IP; scale to convergence only if the
learned arm emits a stable ordering signal and improves a preregistered metric.

Primary outcomes are validity, valid+unique, valid+unique+novel, normalized KL,
and normalized FCD on the standard evaluation sample counts. Also measure the
paper's qualitative claim that the schedule generates paired parentheses/ring
markers before atom tokens, without using this observation as a training
label. The paper reports, for classifier-terminated IP, 97.4% validity, 97.3%
valid+unique, 95.6% valid+unique+novel, 97.2 normalized KL, and 89.2 normalized
FCD.

Execution amendment (2026-08-10): G1 learned IP reached 80.7% exact versus the
paper's 83.0%, so G2 is open. The official GuacaMol v1 files are authenticated
against BenevolentAI's published MD5 values. G2 begins with capacity-matched
FO-ARM, classifier-terminated AO-IP, and classifier-terminated learned-IP
curves. Because the paper omits model widths/heads, batch size, tokenizer,
maximum length, training duration, seeds, and hardware, these arms are labeled
reconstructions. The shared declared capacity is an 18-layer, width-256,
8-head decoder and, for learned IP, a 3-layer, width-128, 4-head posterior.
The first gate tests validity, valid+unique, valid+unique+novel, termination,
and emergence of the reported scaffold-before-atoms insertion schedule before
authorizing convergence-scale training and the official 10,000-sample KL/FCD
panel.

Continuation amendment (2026-08-10): the 20,000-update curves were only about
one GuacaMol epoch. Their sealed 1,000-sample panels reached 19.1% validity for
FO-ARM, 43.4% for AO-IP, and 41.8% for learned IP. Although learned IP did not
beat AO-IP at this early gate, the owner prioritizes approximate recovery of
the paper's converged performance over a clean cold-start comparison. Continue
all three exact 20k parents for 80,000 additional updates (about four epochs)
after one declared Ash--Adams shrink-and-perturb restart: initialize the decoder
from its EMA weights, initialize the learned posterior from its trained raw
weights, apply lambda=0.95 and Gaussian sigma=0.005 to all trainable parameters,
discard AdamW moments, reset decoder EMA from the transformed weights, and run
a new 1,000-update warmup plus cosine schedule over the 80k continuation. Save
absolute-step checkpoints at 40k, 60k, 80k, and 100k; evaluate 512 samples every
10k and 2,000 samples at completion. This is deliberately a warm-restart
performance-seeking continuation, not an unbiased comparison with training the
same schedule from scratch.

## Stop conditions

- Stop and repair on any G0 gradient/marginal mismatch.
- Stop an arm on non-finite losses, non-termination, collapsed insertion-slot
  entropy without improving likelihood, or invalid telemetry.
- Do not interpret a Sudoku fixed-canvas result as evidence for or against the
  paper-native variable-length model.
- Preserve checkpoints, manifests, per-sample outputs, and hashes before
  releasing compute.
