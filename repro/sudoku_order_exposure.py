"""Frozen-policy order and rollout-exposure diagnostics for S4 Sudoku."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from repro.hashing import sha256_file, sha256_json
from repro.paths import repo_root
from repro.small_grokking import (
    BOX_COLS,
    BOX_ROWS,
    CELLS,
    SIZE,
    S4Policy,
    S4Split,
    count_solutions,
    encode_state,
    frozen_splits,
    is_valid,
    model_spec,
)


Order = Literal["fixed", "random", "learned", "oracle_mrv"]
ORDERS: tuple[Order, ...] = ("fixed", "random", "learned", "oracle_mrv")
DEFAULT_HORIZONS = (1, 2, 4, 8, 10)


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


def legal_candidate_counts(grids: torch.Tensor) -> torch.Tensor:
    """Return legal digit counts for every cell in a batch of partial S4 boards."""
    if grids.ndim != 3 or grids.shape[1:] != (SIZE, SIZE):
        raise ValueError("expected grids with shape (B,4,4)")
    counts = torch.zeros((len(grids), CELLS), dtype=torch.long, device=grids.device)
    for cell in range(CELLS):
        row, col = divmod(cell, SIZE)
        r0 = row - row % BOX_ROWS
        c0 = col - col % BOX_COLS
        used = torch.cat(
            (
                grids[:, row, :],
                grids[:, :, col],
                grids[:, r0 : r0 + BOX_ROWS, c0 : c0 + BOX_COLS].reshape(
                    len(grids), -1
                ),
            ),
            dim=1,
        )
        legal = torch.zeros(len(grids), dtype=torch.long, device=grids.device)
        for digit in range(1, SIZE + 1):
            legal += ~(used == digit).any(-1)
        counts[:, cell] = legal
    return counts


def choose_cell(
    grids: torch.Tensor,
    empty: torch.Tensor,
    order: Order,
    output: dict[str, torch.Tensor],
    generator: torch.Generator,
) -> torch.Tensor:
    if order == "fixed":
        scores = -torch.arange(CELLS, device=grids.device).float()[None]
    elif order == "random":
        scores = torch.rand(empty.shape, device=grids.device, generator=generator)
    elif order == "learned":
        scores = output["cell_logits"].float()
    elif order == "oracle_mrv":
        # Fewer candidates wins; subtracting the cell index makes the tie-break
        # row-major.  No target/solution information enters this calculation.
        counts = legal_candidate_counts(grids)
        indices = torch.arange(CELLS, device=grids.device)[None]
        scores = -(counts * (CELLS + 1) + indices).float()
    else:
        raise ValueError(f"unknown order {order}")
    return scores.expand_as(empty).masked_fill(~empty, -torch.inf).argmax(-1)


def partial_consistent(grid: np.ndarray) -> bool:
    """Whether every nonzero row, column, and box contains no duplicate."""
    def unique_nonzero(values: np.ndarray) -> bool:
        nonzero = values[values != 0]
        return len(nonzero) == len(set(nonzero.tolist()))

    for index in range(SIZE):
        if not unique_nonzero(grid[index]) or not unique_nonzero(grid[:, index]):
            return False
    for r0 in range(0, SIZE, BOX_ROWS):
        for c0 in range(0, SIZE, BOX_COLS):
            if not unique_nonzero(
                grid[r0 : r0 + BOX_ROWS, c0 : c0 + BOX_COLS].ravel()
            ):
                return False
    return True


def partial_continuable(grid: np.ndarray) -> bool:
    """Whether a consistent partial board has at least one exact completion."""
    return partial_consistent(grid) and count_solutions(grid, limit=1) > 0


def _snapshot(
    grids: torch.Tensor,
    split: S4Split,
    surviving: torch.Tensor,
    correct_decisions: torch.Tensor,
    horizon: int,
) -> dict[str, Any]:
    array = grids.detach().cpu().numpy().astype(np.uint8)
    consistent = np.asarray([partial_consistent(grid) for grid in array])
    # ``count_solutions`` accepts only a consistent starting puzzle; it treats
    # any filled board as a leaf.  Guard that precondition explicitly because
    # policy rollouts can create inconsistent full boards.
    continuable = np.asarray([partial_continuable(grid) for grid in array])
    complete = (array != 0).all(axis=(1, 2))
    exact = complete & (array == split.solutions).all(axis=(1, 2))
    valid = complete & np.asarray([is_valid(grid) for grid in array])
    return {
        "horizon": horizon,
        "trajectories": split.n,
        "target_prefix_survival": int(surviving.sum()),
        "target_prefix_survival_rate": float(surviving.float().mean()),
        "correct_decisions": int(correct_decisions.sum()),
        "decision_accuracy": float(correct_decisions.sum() / (split.n * horizon)),
        "partial_consistent": int(consistent.sum()),
        "partial_consistent_rate": float(consistent.mean()),
        "continuable": int(continuable.sum()),
        "continuable_rate": float(continuable.mean()),
        "exact": int(exact.sum()),
        "exact_rate": float(exact.mean()),
        "valid": int(valid.sum()),
        "valid_rate": float(valid.mean()),
    }


@torch.no_grad()
def rollout_horizons(
    policy: S4Policy,
    split: S4Split,
    *,
    order: Order,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    device: torch.device,
    seed: int,
    bf16: bool,
) -> list[dict[str, Any]]:
    maximum = int((split.puzzles == 0).sum(axis=(1, 2)).max())
    if not horizons or any(value <= 0 or value > maximum for value in horizons):
        raise ValueError(f"horizons must lie in [1,{maximum}]")
    puzzles = torch.as_tensor(
        np.array(split.puzzles, copy=True), dtype=torch.long, device=device
    )
    solutions = torch.as_tensor(
        np.array(split.solutions, copy=True), dtype=torch.long, device=device
    )
    grids = puzzles.clone()
    generator = torch.Generator(device=device).manual_seed(seed)
    hint_generator = torch.Generator(device=device).manual_seed(seed + 1_000_003)
    hints = torch.rand(
        (split.n, CELLS), device=device, generator=hint_generator
    ).argsort(-1)
    surviving = torch.ones(split.n, dtype=torch.bool, device=device)
    correct_decisions = torch.zeros(split.n, dtype=torch.long, device=device)
    policy.eval()
    rows: list[dict[str, Any]] = []
    for step in range(1, max(horizons) + 1):
        empty = grids.reshape(split.n, CELLS) == 0
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and bf16
            else nullcontext()
        )
        with context:
            output = policy(encode_state(grids, solutions, hints))
        cell = choose_cell(grids, empty, order, output, generator)
        batch = torch.arange(split.n, device=device)
        digit = output["digit_logits"][batch, cell].argmax(-1) + 1
        target = solutions.reshape(split.n, CELLS)[batch, cell]
        correct = digit == target
        surviving &= correct
        correct_decisions += correct.long()
        grids.reshape(split.n, CELLS)[batch, cell] = digit
        if step in horizons:
            rows.append(_snapshot(grids, split, surviving, correct_decisions, step))
    return rows


def load_policy(run_dir: Path, checkpoint: Path, device: torch.device) -> tuple[S4Policy, dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    settings = manifest["settings"]
    expected = run_dir / checkpoint.name
    if checkpoint.resolve() != expected.resolve():
        raise ValueError("checkpoint must belong to its run directory")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload["manifest_sha256"] != manifest["seal"]["manifest_sha256"]:
        raise ValueError("checkpoint/manifest seal mismatch")
    spec = model_spec(settings["width"], settings["blocks"], settings["heads"])
    policy = S4Policy(spec).to(device)
    policy.load_state_dict(payload["policy"])
    return policy, {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(payload["step"]),
        "trained_arm": payload["arm"],
        "weight_decay": float(settings["weight_decay"]),
        "manifest_sha256": payload["manifest_sha256"],
    }


def run_panel(
    run_dirs: list[Path],
    output: Path,
    *,
    repeats: int,
    seed: int,
    device_name: str,
    bf16: bool,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if _git("status", "--porcelain"):
        raise RuntimeError("refusing to evaluate from a dirty worktree")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _, heldout, overlap = frozen_splits(seed)
    output.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        checkpoint = run_dir / "checkpoint_step_001000000.pt"
        policy, source = load_policy(run_dir, checkpoint, device)
        sources.append(source)
        for repeat in range(repeats):
            paired_seed = seed + 100_000 + repeat
            for order in ORDERS:
                metrics = rollout_horizons(
                    policy,
                    heldout,
                    order=order,
                    horizons=horizons,
                    device=device,
                    seed=paired_seed,
                    bf16=bf16,
                )
                for row in metrics:
                    results.append(
                        {
                            "trained_arm": source["trained_arm"],
                            "weight_decay": source["weight_decay"],
                            "checkpoint_sha256": source["checkpoint_sha256"],
                            "order": order,
                            "repeat": repeat,
                            "seed": paired_seed,
                            **row,
                        }
                    )
    result_path = output / "results.jsonl"
    temporary = result_path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, result_path)
    manifest_content = {
        "schema": "apmdm/s4-order-exposure-panel-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "sources": sources,
        "data": {"heldout": heldout.provenance(), "overlap": overlap},
        "contract": {
            "orders": list(ORDERS),
            "horizons": list(horizons),
            "repeats": repeats,
            "base_seed": seed,
            "keyed_target_hints": True,
            "oracle_uses_target": False,
        },
        "hardware": {"device": str(device), "bf16": bf16},
        "results": {
            "path": result_path.name,
            "rows": len(results),
            "sha256": sha256_file(result_path),
        },
        "code": {
            "path": "repro/sudoku_order_exposure.py",
            "sha256": sha256_file(Path(__file__)),
        },
    }
    manifest = {
        **manifest_content,
        "seal": {"manifest_sha256": sha256_json(manifest_content)},
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def parse_horizons(text: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in text.split(",") if value.strip()}))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--horizons", default="1,2,4,8,10")
    args = parser.parse_args()
    manifest = run_panel(
        [Path(value).resolve() for value in args.run_dir],
        Path(args.output).resolve(),
        repeats=args.repeats,
        seed=args.seed,
        device_name=args.device,
        bf16=args.bf16,
        horizons=parse_horizons(args.horizons),
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
