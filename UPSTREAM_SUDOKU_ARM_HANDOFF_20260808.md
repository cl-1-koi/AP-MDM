# Released AP-MDM Sudoku Arm

This branch keeps the paper authors' generator, `Diffusion` loss, DIT module
layout, optimiser, and Lightning training path as the upstream control.  Bulk
data and checkpoints remain outside Git under
`/home/ubuntu/apmdm-official-data`.

## Provenance

- Upstream: `https://github.com/chr26195/AP-MDM`
- Frozen upstream commit: `836b79b30286301ee1cca975db7e8468bc2a653e`
- Official generator output: 2,502,258 transitions from 100 source puzzles
- Official generated vocabulary: 32 tokens
- Effective released tokenizer vocabulary: 34 tokens after adding BOS and PAD

Run the machine-readable report with:

```bash
PY=/home/ubuntu/apmdm-repro-env/venv/bin/python
$PY -m repro.cli upstream-report \
  --vocab-cache /home/ubuntu/apmdm-official-data/vocab_cache.pkl \
  --out /home/ubuntu/apmdm-official-data/upstream_architectures.json
```

The effective released architectures are:

| Arm | Configuration | Effective parameters |
| --- | --- | ---: |
| Released/check-in | 4 blocks, 2 heads, d=128, V=34 | 1,036,197 |
| Paper-declared | 6 blocks, 4 heads, d=256, V=34 | 6,052,133 |

Neither equals the paper's reported approximately 1.2M parameters.  They are
separate arms and must never share a label or checkpoint lineage.

## Minimal compatibility changes

The scientific training semantics are unchanged.  The branch changes only:

1. FlashAttention falls back to PyTorch SDPA on the target Torch 2.7/CUDA 12.8
   stack.  A tensor-equivalence test compares it to the independently audited
   portable backbone after loading identical weights.
2. The unused autoregressive backend is not imported when FlashAttention is
   absent.
3. The dataset factory forwards the configured vocabulary path instead of
   silently constructing a 14-token fallback vocabulary.
4. Streaming chunk names include `train` or `validation`; otherwise validation
   overwrites the first training chunks in the released loader.

All four changes are regression-tested in `tests/test_upstream_adapter.py`.

## Bounded released-stack smoke

The smoke uses the released `train/main.py`, `Diffusion`, precomputed losses,
Lightning optimiser path, and generated pickle.  It is machinery validation,
not a Sudoku result.

```bash
mkdir -p /home/ubuntu/apmdm-official-data/runs/released-smoke
cd /home/ubuntu/AP-MDM-official-sudoku-arm-20260808
PY=/home/ubuntu/apmdm-repro-env/venv/bin/python
$PY train/main.py --config-name=sudoku \
  data.dataset_path=/home/ubuntu/apmdm-official-data/sudoku-100.pkl.gz \
  data.vocab_cache_path=/home/ubuntu/apmdm-official-data/vocab_cache.pkl \
  data.cache_dir=/home/ubuntu/apmdm-official-data/cache \
  data.streaming=true data.chunk_size=10000 \
  trainer.devices=1 trainer.max_steps=200 trainer.num_sanity_val_steps=0 \
  trainer.val_check_interval=100 trainer.limit_val_batches=2 \
  trainer.log_every_n_steps=10 loader.num_workers=4 wandb=null \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=100 \
  checkpointing.save_dir=/home/ubuntu/apmdm-official-data/runs/released-smoke \
  hydra.run.dir=/home/ubuntu/apmdm-official-data/runs/released-smoke/hydra
```

The paper-declared smoke is the same command with
`--config-name=sudoku_paper` and a distinct output directory.

Measured on the local A10, using the released Lightning stack and batch 256:

| Arm | 200-step wall time | Steady steps/s | Sequences/s | Validation NLL at step 200 |
| --- | ---: | ---: | ---: | ---: |
| Released/check-in | ~35 s | 8.14 | 2,085 | 2.10145 |
| Paper-declared | ~62 s | 3.66 | 938 | 1.86449 |

The steady rates include the short step-100 validation.  Linear projections
are approximately 3.4/7.6 A10-hours for 100k steps and 34/76 A10-hours for 1M
steps, respectively.  These are capacity estimates, not evidence that either
model will solve held-out puzzles.

Both step-200 checkpoints loaded strictly through the conditional adapter and
wrote 16 held-out per-puzzle rows.  As expected at this tiny step count, both
immediately selected a fixed point without operations and scored 0/16; this is
a plumbing check, not a replication verdict.

## Conditional evaluation

The official sampler is unconditional and cannot initialize from a Sudoku
puzzle.  `repro.upstream_adapter.OfficialBackboneAdapter` exposes the released
DIT to the audited prompt-conditioned sampler without replacing the model or
training stack.  A Lightning checkpoint is loaded strictly from its
`backbone.*` tensors:

```bash
$PY -m repro.cli upstream-evaluate \
  --checkpoint /path/to/last.ckpt \
  --config-name sudoku \
  --vocab-cache /home/ubuntu/apmdm-official-data/vocab_cache.pkl \
  --limit 256 --batch-size 256 --max-steps 65536 --device cuda --bf16
```

Every evaluation records the checkpoint hash, vocabulary hash, upstream
commit, effective architecture signature, immutable per-puzzle rows, sampler
settings, and aggregate validity/correctness statistics.
