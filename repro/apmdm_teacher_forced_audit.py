"""Held-out teacher-forced transition audit for the official AP-MDM checkpoint.

This is gate D2 from the cross-paper Sudoku failure review.  It does not train
or free-run the model.  The official solver generates its declared transition
supervision on a frozen held-out puzzle subset, and the released backbone is
scored one transition at a time.  Terminal solutions are evaluated separately
for fixed-point stability because the released generator emits no explicit
halt target.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from repro import data as data_mod
from repro.hashing import sha256_array, sha256_file, sha256_json
from repro.losses import subs_argmax
from repro.official import data_manager
from repro.paths import assert_outside_repo, repo_root
from repro.telemetry import gpu_snapshot
from repro.upstream_adapter import (
    UPSTREAM_COMMIT,
    build_upstream_adapter,
    load_lightning_checkpoint,
)
from repro.vocab import MASK


HEADS = {
    "remask": "remasking_logits",
    "insert": "expansion_logits",
    "delete": "contraction_logits",
}


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root()), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def selected_indices(n_available: int, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or limit > n_available:
        raise ValueError("limit must lie in [1,n_available]")
    generator = np.random.default_rng(seed)
    return generator.permutation(n_available)[:limit].astype(np.int64)


def _empty_counts() -> Counter:
    counts = Counter()
    for head in HEADS:
        for suffix in ("target_positive", "predicted_positive", "tp", "fp", "fn"):
            counts[f"{head}_{suffix}"] = 0
    return counts


def evaluate_transition_batch(
    model,
    samples: list[dict[str, Any]],
    *,
    device: torch.device,
    threshold: float,
    autocast_dtype: torch.dtype | None,
) -> Counter:
    """Return additive teacher-forced counts for one fixed-length batch."""
    if not samples:
        return _empty_counts()
    keys = ("x_k", "y_star", "r_star", "e_star", "c_star", "x_k_plus_1")
    arrays = {key: np.stack([np.asarray(row[key]) for row in samples]) for key in keys}
    lengths = {value.shape[1] for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent transition lengths: {sorted(lengths)}")
    tensors = {
        key: torch.as_tensor(value, dtype=torch.long, device=device)
        for key, value in arrays.items()
    }
    x = tensors["x_k"]
    if autocast_dtype is not None and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            outputs = model(x)
    else:
        outputs = model(x)

    predicted_content = subs_argmax(outputs["unmasking_logits"], x, MASK)
    unmask_positions = (x == MASK) & (tensors["y_star"] != MASK)
    predicted_heads: dict[str, torch.Tensor] = {}
    labels = {
        "remask": tensors["r_star"].bool(),
        "insert": tensors["e_star"].bool(),
        "delete": tensors["c_star"].bool(),
    }
    counts = _empty_counts()
    counts["transitions"] = len(samples)
    counts["unmask_total"] = int(unmask_positions.sum())
    counts["unmask_correct"] = int(
        ((predicted_content == tensors["y_star"]) & unmask_positions).sum()
    )
    for name, output_key in HEADS.items():
        predicted = torch.sigmoid(outputs[output_key].squeeze(-1).float()) > threshold
        predicted_heads[name] = predicted
        target = labels[name]
        counts[f"{name}_target_positive"] = int(target.sum())
        counts[f"{name}_predicted_positive"] = int(predicted.sum())
        counts[f"{name}_tp"] = int((predicted & target).sum())
        counts[f"{name}_fp"] = int((predicted & ~target).sum())
        counts[f"{name}_fn"] = int((~predicted & target).sum())

    kept = torch.where(x == MASK, predicted_content, x)
    predicted_next = torch.where(
        predicted_heads["remask"], torch.full_like(kept, MASK), kept
    )
    structural = predicted_heads["insert"].any(-1) | (
        predicted_heads["delete"] & (x == MASK)
    ).any(-1)
    exact = (~structural) & (predicted_next == tensors["x_k_plus_1"]).all(-1)
    counts["transition_exact"] = int(exact.sum())

    operations = [str(row.get("solver_metadata", {}).get("operation", "unknown")) for row in samples]
    for operation in set(operations):
        selected = torch.as_tensor(
            [value == operation for value in operations], device=device, dtype=torch.bool
        )
        counts[f"operation::{operation}::transitions"] = int(selected.sum())
        counts[f"operation::{operation}::transition_exact"] = int(exact[selected].sum())
        positions = unmask_positions[selected]
        counts[f"operation::{operation}::unmask_total"] = int(positions.sum())
        counts[f"operation::{operation}::unmask_correct"] = int(
            ((predicted_content[selected] == tensors["y_star"][selected]) & positions).sum()
        )
    return counts


@torch.no_grad()
def evaluate_terminal_stability(
    model,
    terminal_states: np.ndarray,
    *,
    device: torch.device,
    threshold: float,
    autocast_dtype: torch.dtype | None,
) -> dict[str, int]:
    states = torch.as_tensor(np.asarray(terminal_states), dtype=torch.long, device=device)
    if states.ndim != 2:
        raise ValueError("terminal_states must be (batch, sequence_length)")
    if autocast_dtype is not None and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            outputs = model(states)
    else:
        outputs = model(states)
    fired = {
        name: torch.sigmoid(outputs[key].squeeze(-1).float()) > threshold
        for name, key in HEADS.items()
    }
    stable = ~(fired["remask"].any(-1) | fired["insert"].any(-1) | fired["delete"].any(-1))
    return {
        "terminal_states": int(len(states)),
        "terminal_stable": int(stable.sum()),
        "terminal_remask_positive": int(fired["remask"].sum()),
        "terminal_insert_positive": int(fired["insert"].sum()),
        "terminal_delete_positive": int(fired["delete"].sum()),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(counts: Counter) -> dict[str, Any]:
    result: dict[str, Any] = {
        "transitions": counts["transitions"],
        "transition_exact": counts["transition_exact"],
        "transition_exact_rate": _ratio(counts["transition_exact"], counts["transitions"]),
        "unmask_positions": counts["unmask_total"],
        "unmask_correct": counts["unmask_correct"],
        "unmask_accuracy": _ratio(counts["unmask_correct"], counts["unmask_total"]),
    }
    for head in HEADS:
        tp, fp, fn = counts[f"{head}_tp"], counts[f"{head}_fp"], counts[f"{head}_fn"]
        result[head] = {
            "target_positive": counts[f"{head}_target_positive"],
            "predicted_positive": counts[f"{head}_predicted_positive"],
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
    operations = {}
    for key in counts:
        if not key.startswith("operation::") or not key.endswith("::transitions"):
            continue
        operation = key.split("::")[1]
        total = counts[key]
        operations[operation] = {
            "transitions": total,
            "transition_exact_rate": _ratio(
                counts[f"operation::{operation}::transition_exact"], total
            ),
            "unmask_accuracy": _ratio(
                counts[f"operation::{operation}::unmask_correct"],
                counts[f"operation::{operation}::unmask_total"],
            ),
        }
    result["operations"] = operations
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = assert_outside_repo(Path(args.output))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if _git("status", "--porcelain"):
        raise RuntimeError("refusing to run from a dirty worktree")
    output.mkdir(parents=True)
    split = data_mod.load_split("test")
    indices = selected_indices(split.n, args.limit, args.seed)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    vocab_cache = Path(args.vocab_cache).expanduser().resolve()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = build_upstream_adapter(
        args.config_name, vocab_cache, seed=args.seed, device=device
    )
    checkpoint_payload = load_lightning_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device).eval()
    autocast_dtype = torch.bfloat16 if device.type == "cuda" and args.bf16 else None
    manifest_content = {
        "schema": "apmdm/teacher-forced-heldout-audit-v1",
        "experiment": "D2 held-out teacher-forced transition and terminal-stability audit",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_dirty": False,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_global_step": int(checkpoint_payload.get("global_step", -1)),
        "vocab_cache": str(vocab_cache),
        "vocab_cache_sha256": sha256_file(vocab_cache),
        "test": split.provenance(),
        "selected_indices": indices.tolist(),
        "selected_indices_sha256": sha256_array(indices),
        "limit": args.limit,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "config_name": args.config_name,
        "bf16": args.bf16,
        "architecture_signature": model.architecture_signature(),
        "hardware": {"device": str(device), **gpu_snapshot(device)},
        "semantics": {
            "transition_source": "official held-out Sudoku solver trajectories",
            "termination_proxy": "fixed-point stability on each official solver trajectory's final encoded state; no explicit halt label exists",
            "insert_delete_targets": "official Sudoku generator emits no positive insert/delete labels",
        },
    }
    manifest = {**manifest_content, "seal": {"manifest_sha256": sha256_json(manifest_content)}}
    _write_json(output / "manifest.json", manifest)
    rows_path = output / "puzzle_rows.jsonl"
    counts = _empty_counts()
    terminal_counts = Counter()
    manager = data_manager()
    started = time.time()
    with rows_path.open("w") as rows_file:
        for ordinal, index in enumerate(indices.tolist(), 1):
            samples = manager.generate_from_puzzle(split.puzzles[index], instance_id=index)
            puzzle_counts = _empty_counts()
            for start in range(0, len(samples), args.batch_size):
                puzzle_counts.update(
                    evaluate_transition_batch(
                        model,
                        samples[start : start + args.batch_size],
                        device=device,
                        threshold=args.threshold,
                        autocast_dtype=autocast_dtype,
                    )
                )
            terminal = evaluate_terminal_stability(
                model,
                np.asarray(samples[-1]["x_k_plus_1"])[None],
                device=device,
                threshold=args.threshold,
                autocast_dtype=autocast_dtype,
            )
            counts.update(puzzle_counts)
            terminal_counts.update(terminal)
            row = {
                "ordinal": ordinal,
                "index": index,
                "transition": summarize(puzzle_counts),
                "terminal": terminal,
            }
            rows_file.write(json.dumps(row, sort_keys=True) + "\n")
            rows_file.flush()
            if ordinal % args.progress_every == 0 or ordinal == len(indices):
                print(
                    json.dumps(
                        {
                            "puzzles": ordinal,
                            "transitions": counts["transitions"],
                            "unmask_accuracy": _ratio(counts["unmask_correct"], counts["unmask_total"]),
                            "transition_exact_rate": _ratio(counts["transition_exact"], counts["transitions"]),
                            "terminal_stable_rate": _ratio(
                                terminal_counts["terminal_stable"],
                                terminal_counts["terminal_states"],
                            ),
                            "elapsed_s": time.time() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    summary = {
        "schema": "apmdm/teacher-forced-heldout-audit-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest["seal"]["manifest_sha256"],
        "puzzles": len(indices),
        "transition": summarize(counts),
        "terminal": {
            **dict(terminal_counts),
            "terminal_stable_rate": _ratio(
                terminal_counts["terminal_stable"], terminal_counts["terminal_states"]
            ),
        },
        "rows_path": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "wall_seconds": time.time() - started,
    }
    summary["summary_sha256"] = sha256_json(summary)
    _write_json(output / "summary.json", summary)
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "completion.json"
    ]
    completion = {
        "schema": "apmdm/teacher-forced-heldout-audit-completion-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["seal"]["manifest_sha256"],
        "puzzles": len(indices),
        "files": files,
    }
    _write_json(output / "completion.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--config-name", choices=("sudoku", "sudoku_paper"), default="sudoku_paper")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
