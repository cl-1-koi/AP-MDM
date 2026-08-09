"""Portable command-line entry points for the AP-MDM Sudoku replication.

All subcommands accept ``--data-root`` (default ``$APMDM_REPRO_DATA_ROOT`` or
``~/apmdm-repro-data``) and never write bulk artefacts into the Git work tree::

    python -m repro.cli authenticate
    python -m repro.cli generate
    python -m repro.cli manifest --run-id r1
    python -m repro.cli train     --run-id r1 --max-steps 200
    python -m repro.cli evaluate  --run-id r1 --limit 256
    python -m repro.cli preflight
    python -m repro.cli smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from repro import data as data_mod
from repro import vocab
from repro.config import load_historical_config, load_paper_config
from repro.evaluator import (
    aggregate,
    check_evaluation_preconditions,
    evaluate_states,
    write_rows,
)
from repro.hashing import sha256_array, sha256_json
from repro.manifest import build_manifest, load_manifest, write_manifest
from repro.model import analytic_parameter_count, build_model, count_parameters
from repro.paths import data_root, eval_dir, runs_dir, transitions_dir
from repro.sampler import SamplerConfig, generate as sample_generate
from repro.trainer import Trainer, TrainerSettings
from repro.trajectories import (
    TransitionDataset,
    build_transition_store,
    deterministic_split,
)

DEFAULT_TAG = "sudoku-100"


def _emit(obj: dict, out: Optional[str] = None) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, default=str)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)


def _set_data_root(value: Optional[str]) -> Path:
    if value:
        os.environ["APMDM_REPRO_DATA_ROOT"] = str(Path(value).expanduser().resolve())
    return data_root(create=True)


# --------------------------------------------------------------------------
# authenticate
# --------------------------------------------------------------------------


def cmd_authenticate(args: argparse.Namespace) -> int:
    _set_data_root(args.data_root)
    train = data_mod.load_split("train")
    test = data_mod.load_split("test")
    overlap = data_mod.measure_overlap(train, test)
    data_mod.assert_no_leakage(overlap)
    vocab_report = vocab.authenticate_vocabulary()

    paper_cfg = load_paper_config()
    hist_cfg = load_historical_config()
    report = {
        "data": {"train": train.provenance(), "test": test.provenance(), "overlap": overlap.as_dict()},
        "vocabulary": vocab_report.as_dict(),
        "configs": {
            "paper_faithful": {
                **paper_cfg.as_dict(),
                "parameter_count": count_parameters(paper_cfg.model).as_dict(),
                "analytic_parameter_count": analytic_parameter_count(paper_cfg.model),
            },
            "historical": {
                **hist_cfg.as_dict(),
                "parameter_count": count_parameters(hist_cfg.model).as_dict(),
                "analytic_parameter_count": analytic_parameter_count(hist_cfg.model),
            },
        },
        "paper_reported_parameter_count": 1_200_000,
    }
    _emit(report, args.out)
    return 0


def cmd_upstream_report(args: argparse.Namespace) -> int:
    """Authenticate the released backbone/configuration as a separate arm."""
    from repro.upstream_adapter import architecture_report

    report = architecture_report(args.vocab_cache)
    _emit(report, args.out)
    return 0


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    _set_data_root(args.data_root)
    train = data_mod.load_split("train")
    limit = args.limit or train.n
    puzzles = np.asarray(train.puzzles[:limit])
    initial = data_mod.encode_puzzle_states(puzzles)
    out_dir = transitions_dir(create=True) / args.tag
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"transition store already exists: {out_dir} (use --force to regenerate)")
        index = json.loads((out_dir / "index.json").read_text())
    else:
        index = build_transition_store(
            puzzles,
            out_dir=out_dir,
            tag=args.tag,
            validate=not args.no_validate,
            initial_states=initial,
            progress=args.progress,
        )
    _emit(
        {
            "tag": args.tag,
            "path": str(out_dir),
            "accounting": index["accounting"],
            "validation_summary": index.get("validation_summary", {}),
            "index_sha256": index.get("index_sha256"),
            "paper_reported_transitions_per_puzzle": 25022.6,
            "paper_reported_transitions_per_puzzle_generic": 1421.3,
        },
        args.out,
    )
    return 0


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def _run_dir(run_id: str) -> Path:
    return runs_dir(create=True) / run_id


def _split_for(dataset: TransitionDataset, cfg) -> tuple[np.ndarray, np.ndarray]:
    return deterministic_split(len(dataset), cfg.train_ratio, cfg.seed)


def cmd_manifest(args: argparse.Namespace) -> int:
    _set_data_root(args.data_root)
    cfg = load_paper_config()
    train = data_mod.load_split("train")
    test = data_mod.load_split("test")
    overlap = data_mod.measure_overlap(train, test)
    data_mod.assert_no_leakage(overlap)

    store = transitions_dir() / args.tag
    dataset = TransitionDataset(store)
    train_idx, monitor_idx = _split_for(dataset, cfg)

    run_dir = _run_dir(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    sampler_cfg = SamplerConfig(
        tau_remask=cfg.remasking_threshold,
        tau_insert=cfg.expansion_threshold,
        tau_delete=cfg.contraction_threshold,
        max_steps=args.sampler_max_steps or cfg.sampling_max_steps,
    )

    manifest = build_manifest(
        config=cfg,
        parameter_count=count_parameters(cfg.model, cfg.time_conditioning),
        vocabulary_report=vocab.authenticate_vocabulary(),
        train_provenance=train.provenance(),
        test_provenance=test.provenance(),
        overlap=overlap.as_dict(),
        transition_index=dataset.index,
        split_hashes={
            "train_indices_sha256": sha256_array(train_idx),
            "monitor_indices_sha256": sha256_array(monitor_idx),
            "n_train_transitions": int(train_idx.size),
            "n_monitor_transitions": int(monitor_idx.size),
            "train_ratio": cfg.train_ratio,
            "split_semantics": (
                "monitor split is in-distribution (same 100 source puzzles); the "
                "only held-out generalisation set is the official 10,000-puzzle array"
            ),
        },
        sampler_config=sampler_cfg.as_dict(),
        run_paths={
            "data_root": str(data_root()),
            "transition_store": str(store),
            "run_dir": str(run_dir),
            "checkpoints": str(run_dir / "checkpoints"),
            "telemetry": str(run_dir / "telemetry.jsonl"),
            "eval_dir": str(eval_dir(create=True)),
        },
        compute_ceilings={
            "smoke_ceiling_combined_a10_minutes": 15,
            "production_ceiling": "TO BE FROZEN BY INDEPENDENT REVIEW from measured smoke throughput",
            "max_optimizer_steps": cfg.max_steps,
            "paid_capacity": "forbidden (no RunPod or other rented capacity)",
        },
        checkpoint_every_n_steps=args.checkpoint_every,
        eval_every_n_steps=args.monitor_every,
        seeds={
            "seed": cfg.seed,
            "python_random": cfg.seed,
            "numpy": cfg.seed,
            "torch": cfg.seed,
            "torch_cuda": cfg.seed,
            "dataloader": cfg.seed,
            "sampling": cfg.seed,
            "note": "seed 42 is the paper/config lineage; a second seed is a follow-up, not a substitute",
        },
        notes=list(args.note or []),
    )
    path = run_dir / "manifest.json"
    digest = write_manifest(path, manifest, allow_overwrite=args.force)
    _emit({"manifest": str(path), "manifest_sha256": digest}, args.out)
    return 0


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    _set_data_root(args.data_root)
    cfg = load_paper_config()
    run_dir = _run_dir(args.run_id)
    manifest = load_manifest(run_dir / "manifest.json")
    dataset = TransitionDataset(transitions_dir() / args.tag)
    train_idx, monitor_idx = _split_for(dataset, cfg)

    settings = TrainerSettings(
        max_steps=args.max_steps if args.max_steps is not None else cfg.max_steps,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        monitor_every=args.monitor_every,
        device=args.device,
        time_budget_seconds=args.time_budget,
    )
    trainer = Trainer(
        config=cfg,
        dataset=dataset,
        train_indices=train_idx,
        monitor_indices=monitor_idx,
        run_dir=run_dir,
        settings=settings,
        manifest_sha256=manifest["seal"]["manifest_sha256"],
        run_id=args.run_id,
    )
    resumed = trainer.maybe_resume() if not args.no_resume else False
    try:
        summary = trainer.train(max_steps=settings.max_steps)
    finally:
        trainer.close()
    _emit({"resumed": resumed, **{k: str(v) for k, v in summary.items()}}, args.out)
    return 0


# --------------------------------------------------------------------------
# monotone insertion ladder
# --------------------------------------------------------------------------


def _insertion_settings_from_args(args: argparse.Namespace):
    from repro.insertion_trainer import InsertionTrainerSettings

    return InsertionTrainerSettings(
        arm=args.arm,
        run_id=args.run_id,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_limit=args.eval_limit,
        device=args.device,
        seed=args.seed,
        q_lr_ratio=args.q_lr_ratio,
        bf16=args.bf16,
        time_budget_seconds=args.time_budget,
        condition_mode=args.condition_mode,
        keep_last_checkpoints=args.keep_last_checkpoints,
    )


def cmd_insertion_manifest(args: argparse.Namespace) -> int:
    import torch

    from repro.insertion import (
        SudokuInsertionPolicy,
        SudokuOrderPosterior,
        default_posterior_spec,
        effective_model_spec,
    )
    from repro.insertion_trainer import build_insertion_manifest

    _set_data_root(args.data_root)
    cfg = load_paper_config()
    settings = _insertion_settings_from_args(args)
    train = data_mod.load_split("train")
    test = data_mod.load_split("test")
    overlap = data_mod.measure_overlap(train, test)
    data_mod.assert_no_leakage(overlap)
    torch.manual_seed(settings.seed)
    policy = SudokuInsertionPolicy(effective_model_spec(cfg.model))
    posterior = (
        SudokuOrderPosterior(default_posterior_spec(policy.spec))
        if settings.arm == "learned_insertion"
        else None
    )
    run_dir = _run_dir(settings.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_insertion_manifest(
        config=cfg,
        settings=settings,
        train=train,
        test=test,
        overlap=overlap.as_dict(),
        policy=policy,
        posterior=posterior,
        run_dir=run_dir,
    )
    path = run_dir / "manifest.json"
    if path.exists() and not args.force:
        raise RuntimeError(f"refusing to overwrite existing manifest: {path}")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    _emit(
        {
            "manifest": str(path),
            "manifest_sha256": manifest["seal"]["manifest_sha256"],
            "policy_parameters": manifest["architecture"]["policy_parameters"],
            "posterior_parameters": manifest["architecture"]["posterior_parameters"],
        },
        args.out,
    )
    return 0


def _settings_from_insertion_manifest(manifest: dict):
    from repro.insertion_trainer import InsertionTrainerSettings

    fields = dict(manifest["training"])
    return InsertionTrainerSettings(**fields)


def cmd_insertion_train(args: argparse.Namespace) -> int:
    from repro.insertion_trainer import (
        InsertionTrainer,
        verify_insertion_manifest,
    )

    _set_data_root(args.data_root)
    cfg = load_paper_config()
    run_dir = _run_dir(args.run_id)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    digest = verify_insertion_manifest(manifest)
    settings = _settings_from_insertion_manifest(manifest)
    train = data_mod.load_split("train")
    test = data_mod.load_split("test")
    overlap = data_mod.measure_overlap(train, test)
    data_mod.assert_no_leakage(overlap)
    trainer = InsertionTrainer(cfg, settings, train, test, run_dir, digest)
    resumed = trainer.maybe_resume() if not args.no_resume else False
    try:
        summary = trainer.train()
    finally:
        trainer.close()
    _emit({"resumed": resumed, **summary}, args.out)
    return 0


def cmd_insertion_evaluate(args: argparse.Namespace) -> int:
    from repro.insertion_trainer import (
        InsertionTrainer,
        verify_insertion_manifest,
    )

    _set_data_root(args.data_root)
    cfg = load_paper_config()
    run_dir = _run_dir(args.run_id)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    digest = verify_insertion_manifest(manifest)
    settings = _settings_from_insertion_manifest(manifest)
    train = data_mod.load_split("train")
    test = data_mod.load_split("test")
    trainer = InsertionTrainer(cfg, settings, train, test, run_dir, digest)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "latest.pt"
    try:
        trainer.load_checkpoint(checkpoint)
        result = trainer.evaluate(limit=args.limit)
    finally:
        trainer.close()
    _emit(result, args.out)
    return 0


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    import torch

    _set_data_root(args.data_root)
    cfg = load_paper_config()
    run_dir = _run_dir(args.run_id)
    manifest = load_manifest(run_dir / "manifest.json")

    train = data_mod.load_split("train")
    test = data_mod.load_split("test")
    overlap = data_mod.measure_overlap(train, test)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "latest.pt"
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check_evaluation_preconditions(manifest, payload["binding"], overlap)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = build_model(cfg.model, time_conditioning=cfg.time_conditioning)
    model.load_state_dict(payload["model"])
    model.to(device).eval()

    sampler_cfg = SamplerConfig(
        tau_remask=manifest["inference"]["tau_remask"],
        tau_insert=manifest["inference"]["tau_insert"],
        tau_delete=manifest["inference"]["tau_delete"],
        max_steps=args.max_steps or manifest["inference"]["max_steps"],
    )

    n = args.limit or test.n
    indices = list(range(n))
    rows = []
    started = time.time()
    autocast = torch.bfloat16 if (device.type == "cuda" and args.bf16) else None
    for start in range(0, n, args.batch_size):
        chunk = indices[start : start + args.batch_size]
        states0 = data_mod.encode_puzzle_states(np.asarray(test.puzzles[chunk]))
        result = sample_generate(
            model, states0, sampler_cfg, device=device, autocast_dtype=autocast,
            progress_every=args.progress_every,
        )
        rows.extend(
            evaluate_states(chunk, test.puzzles, test.solutions, result.states, result.traces)
        )
        print(
            f"  evaluated {len(rows)}/{n} puzzles "
            f"({result.total_forward_passes} forwards, {result.batch_seconds:.1f}s)",
            flush=True,
        )

    tag = args.eval_tag or f"{args.run_id}_step{payload['binding']['step']}"
    rows_path = eval_dir(create=True) / f"rows_{tag}.jsonl"
    rows_sha = write_rows(rows_path, rows)
    summary = aggregate(rows)
    summary_doc = {
        "run_id": args.run_id,
        "checkpoint": str(ckpt_path),
        "checkpoint_step": payload["binding"]["step"],
        "manifest_sha256": manifest["seal"]["manifest_sha256"],
        "rows_path": str(rows_path),
        "rows_file_sha256": rows_sha,
        "n_evaluated": len(rows),
        "n_available": test.n,
        "wall_seconds": time.time() - started,
        "sampler": sampler_cfg.as_dict(),
        "aggregate": summary,
    }
    summary_doc["summary_sha256"] = sha256_json(summary_doc)
    (eval_dir(create=True) / f"summary_{tag}.json").write_text(
        json.dumps(summary_doc, indent=2, sort_keys=True)
    )
    _emit(summary_doc, args.out)
    return 0


def cmd_upstream_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a released-Lightning checkpoint with conditional Sudoku input."""
    import torch

    from repro.hashing import sha256_file
    from repro.upstream_adapter import (
        UPSTREAM_COMMIT,
        build_upstream_adapter,
        load_lightning_checkpoint,
    )

    _set_data_root(args.data_root)
    cfg = (
        load_historical_config()
        if args.config_name == "sudoku"
        else load_paper_config()
    )
    train = data_mod.load_split("train")
    test = data_mod.load_split("test")
    overlap = data_mod.measure_overlap(train, test)
    data_mod.assert_no_leakage(overlap)

    device = torch.device(
        args.device
        if (args.device != "cuda" or torch.cuda.is_available())
        else "cpu"
    )
    model = build_upstream_adapter(
        args.config_name,
        args.vocab_cache,
        seed=cfg.seed,
        device=device,
    )
    payload = load_lightning_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(device).eval()

    sampler_cfg = SamplerConfig(
        tau_remask=cfg.remasking_threshold,
        tau_insert=cfg.expansion_threshold,
        tau_delete=cfg.contraction_threshold,
        max_steps=args.max_steps,
    )
    n = min(args.limit or test.n, test.n)
    rows = []
    started = time.time()
    autocast = torch.bfloat16 if (device.type == "cuda" and args.bf16) else None
    for start in range(0, n, args.batch_size):
        indices = list(range(start, min(start + args.batch_size, n)))
        states0 = data_mod.encode_puzzle_states(np.asarray(test.puzzles[indices]))
        result = sample_generate(
            model,
            states0,
            sampler_cfg,
            device=device,
            autocast_dtype=autocast,
            progress_every=args.progress_every,
        )
        rows.extend(
            evaluate_states(
                indices, test.puzzles, test.solutions, result.states, result.traces
            )
        )
        print(f"  evaluated {len(rows)}/{n} puzzles", flush=True)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    step = int(payload.get("global_step", -1))
    tag = args.eval_tag or f"upstream-{args.config_name}-step{step}"
    rows_path = eval_dir(create=True) / f"rows_{tag}.jsonl"
    rows_sha = write_rows(rows_path, rows)
    summary_doc = {
        "backend": "released AP-MDM training stack + conditional adapter",
        "upstream_commit": UPSTREAM_COMMIT,
        "config_name": args.config_name,
        "architecture_signature": model.architecture_signature(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": step,
        "vocab_cache": str(Path(args.vocab_cache).expanduser().resolve()),
        "vocab_cache_sha256": sha256_file(Path(args.vocab_cache).expanduser().resolve()),
        "rows_path": str(rows_path),
        "rows_file_sha256": rows_sha,
        "n_evaluated": len(rows),
        "n_available": test.n,
        "wall_seconds": time.time() - started,
        "sampler": sampler_cfg.as_dict(),
        "aggregate": aggregate(rows),
    }
    summary_doc["summary_sha256"] = sha256_json(summary_doc)
    (eval_dir(create=True) / f"summary_{tag}.json").write_text(
        json.dumps(summary_doc, indent=2, sort_keys=True)
    )
    _emit(summary_doc, args.out)
    return 0


# --------------------------------------------------------------------------
# preflight / smoke
# --------------------------------------------------------------------------


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Measure forward-pass and optimizer-step throughput for cost projection."""
    import torch

    from repro.losses import supervised_loss
    from repro.smoke import four_operation_fixture

    _set_data_root(args.data_root)
    cfg = load_paper_config()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = build_model(cfg.model, time_conditioning=cfg.time_conditioning, seed=cfg.seed).to(device)
    autocast = torch.bfloat16 if (device.type == "cuda" and args.bf16) else None

    batch = {
        k: v.to(device)
        for k, v in four_operation_fixture(batch=args.batch_size, length=vocab.SEQUENCE_LENGTH).items()
    }
    params = sum(p.numel() for p in model.parameters())
    tokens = args.batch_size * vocab.SEQUENCE_LENGTH

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # ---- inference forward passes ----
    model.eval()
    with torch.no_grad():
        for _ in range(args.warmup):
            model(batch["x_k"]) if autocast is None else _autocast_forward(model, batch["x_k"], autocast)
        sync()
        start = time.perf_counter()
        for _ in range(args.iters):
            model(batch["x_k"]) if autocast is None else _autocast_forward(model, batch["x_k"], autocast)
        sync()
        forward_seconds = (time.perf_counter() - start) / args.iters

    # ---- training steps ----
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps, weight_decay=cfg.optim.weight_decay,
    )

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x_k"]) if autocast is None else _autocast_forward(model, batch["x_k"], autocast)
        loss = supervised_loss(
            outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"],
            batch["c_star"], attention_mask=batch["attention_mask"],
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
        optimizer.step()

    for _ in range(args.warmup):
        one_step()
    sync()
    start = time.perf_counter()
    for _ in range(args.iters):
        one_step()
    sync()
    step_seconds = (time.perf_counter() - start) / args.iters

    report = {
        "device": str(device),
        "bf16_autocast": autocast is not None,
        "batch_size": args.batch_size,
        "sequence_length": vocab.SEQUENCE_LENGTH,
        "parameters": params,
        "iters": args.iters,
        "forward_seconds_per_pass": forward_seconds,
        "forward_sequences_per_second": args.batch_size / forward_seconds,
        "forward_tflops_effective": 2 * params * tokens / forward_seconds / 1e12,
        "train_seconds_per_step": step_seconds,
        "train_steps_per_second": 1.0 / step_seconds,
        "train_tokens_per_second": tokens / step_seconds,
        "train_tflops_effective": 6 * params * tokens / step_seconds / 1e12,
        "projected_hours_1M_steps": step_seconds * 1_000_000 / 3600,
        "note": (
            "Effective TFLOP/s uses the 2ND (inference) and 6ND (training) "
            "convention and excludes attention FLOPs, so it is a lower bound."
        ),
        **{f"gpu_{k}": v for k, v in _gpu_snapshot().items()},
    }
    _emit(report, args.out)
    return 0


def _autocast_forward(model, x, dtype):
    import torch

    with torch.autocast(device_type="cuda", dtype=dtype):
        return model(x)


def _gpu_snapshot() -> dict:
    from repro.telemetry import gpu_snapshot

    return gpu_snapshot()


def cmd_preflight(args: argparse.Namespace) -> int:
    from repro.supervisor import preflight_report

    _emit(preflight_report(ceiling_seconds=args.ceiling_seconds), args.out)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """CPU smoke: one optimizer step plus a tiny four-operation overfit."""
    import torch

    from repro.smoke import cpu_overfit_smoke, one_batch_smoke

    _set_data_root(args.data_root)
    cfg = load_paper_config()
    result = {
        "one_batch": one_batch_smoke(cfg, device=args.device),
        "tiny_overfit": cpu_overfit_smoke(device=args.device, steps=args.overfit_steps),
    }
    _emit(result, args.out)
    ok = result["one_batch"]["all_heads_received_gradient"] and result["tiny_overfit"]["passed"]
    return 0 if ok else 1


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repro.cli", description=__doc__)
    parser.add_argument("--data-root", default=None, help="out-of-tree data root")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("authenticate", help="authenticate data, vocabulary and architectures")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_authenticate)

    p = sub.add_parser(
        "upstream-report",
        help="report the released architectures with their effective vocabulary",
    )
    p.add_argument("--vocab-cache", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_upstream_report)

    p = sub.add_parser("generate", help="generate and validate solver transitions")
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("manifest", help="build and seal the experiment manifest")
    p.add_argument("--run-id", required=True)
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--monitor-every", type=int, default=3_000)
    p.add_argument("--sampler-max-steps", type=int, default=None)
    p.add_argument("--note", action="append", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("train", help="train (resumable)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--monitor-every", type=int, default=3_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser(
        "insertion-manifest",
        help="seal one fixed-AR/random-insertion/learned-insertion Sudoku arm",
    )
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--arm",
        required=True,
        choices=("fixed_ar", "random_insertion", "learned_insertion"),
    )
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-limit", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--q-lr-ratio", type=float, default=0.01)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--keep-last-checkpoints", type=int, default=3)
    p.add_argument(
        "--condition-mode",
        choices=(
            "puzzle_only",
            "aligned_solution_hint",
            "transposed_solution_hint",
        ),
        default="puzzle_only",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_insertion_manifest)

    p = sub.add_parser("insertion-train", help="train a sealed monotone Sudoku arm")
    p.add_argument("--run-id", required=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_insertion_train)

    p = sub.add_parser("insertion-evaluate", help="evaluate a monotone Sudoku checkpoint")
    p.add_argument("--run-id", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_insertion_evaluate)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint on held-out puzzles")
    p.add_argument("--run-id", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--eval-tag", default=None)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser(
        "upstream-evaluate",
        help="conditionally evaluate a checkpoint produced by the released trainer",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config-name", choices=("sudoku", "sudoku_paper"), required=True)
    p.add_argument("--vocab-cache", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=65536)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--eval-tag", default=None)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_upstream_evaluate)

    p = sub.add_parser("benchmark", help="measure throughput for the cost projection")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("preflight", help="fail-closed GPU supervisor preflight")
    p.add_argument("--ceiling-seconds", type=float, default=900.0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("smoke", help="CPU one-batch and tiny-overfit smoke")
    p.add_argument("--device", default="cpu")
    p.add_argument("--overfit-steps", type=int, default=400)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_smoke)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
