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

- No public implementation or checkpoint was linked from arXiv v1, v2, or v3.
- No implementation was found in the public repositories of the first author
  or coauthors, or by GitHub repository/code search on 2026-08-09.
- The standard GuacaMol dataset and evaluator are public, but the paper omits
  the decoder width, attention-head count, batch size, tokenizer details,
  maximum sequence length, training steps/epochs, random seeds, and hardware.
  It only specifies 18 decoder layers, three posterior layers, optimizer/LRs,
  weight decay, EMA, cosine scheduling, and that training ran "to convergence."
- The planning appendix specifies more of the model and optimizer but omits
  total dataset sizes and seeds.

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
5. policy-based and classifier-based termination produce normalized sampling
   distributions and halt correctly.

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

Because upstream dataset counts and seeds are missing, freeze our generator,
split sizes, and seeds before observing results. Compare matched FO-ARM,
random-order AO-IP, and learned IP arms. The paper reports 24.0%, 26.5%, and
83.0% sequence accuracy respectively, so this is a high-signal mechanistic
gate. Report exact-match, token accuracy, Hamming distance, Levenshtein
distance, ELBO, RLOO variance, termination errors, examples/s, and GPU-hours.

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

## Stop conditions

- Stop and repair on any G0 gradient/marginal mismatch.
- Stop an arm on non-finite losses, non-termination, collapsed insertion-slot
  entropy without improving likelihood, or invalid telemetry.
- Do not interpret a Sudoku fixed-canvas result as evidence for or against the
  paper-native variable-length model.
- Preserve checkpoints, manifests, per-sample outputs, and hashes before
  releasing compute.
