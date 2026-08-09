"""Counterfactual key-usage diagnostics for Sudoku VC-1c checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from repro import data as data_mod
from repro import vocab
from repro.config import ModelSpec
from repro.hashing import sha256_file
from repro.insertion import SudokuInsertionPolicy, encode_partial_grids, solve_monotone
from repro.paths import repo_root


SCHEMA = "apmdm/keyed-diagnostics-v1"


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root()), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _fresh_orders(
    batch: int, *, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    return torch.rand(
        (batch, vocab.NUM_CELLS), device=device, generator=generator
    ).argsort(dim=-1)


@torch.no_grad()
def payload_metrics(
    policy: SudokuInsertionPolicy,
    puzzles: np.ndarray,
    solutions: np.ndarray,
    payloads: np.ndarray,
    *,
    device: torch.device | str,
    seed: int,
    batch_size: int = 256,
    keyed: bool = True,
    include_hints: bool = True,
    bf16: bool = True,
) -> dict[str, Any]:
    """Measure one-forward blank-cell following of visible keyed payloads."""
    device = torch.device(device)
    if puzzles.shape != solutions.shape or puzzles.shape != payloads.shape:
        raise ValueError("puzzles, solutions, and payloads must have equal shapes")
    if puzzles.ndim != 3 or tuple(puzzles.shape[1:]) != (9, 9):
        raise ValueError("keyed diagnostics require (N,9,9) arrays")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    policy.eval()
    target_correct = solution_correct = blank_total = 0
    target_exact = solution_exact = 0
    n = int(puzzles.shape[0])
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        puzzle = torch.as_tensor(
            np.array(puzzles[start:stop], copy=True), dtype=torch.long, device=device
        )
        solution = torch.as_tensor(
            np.array(solutions[start:stop], copy=True), dtype=torch.long, device=device
        )
        payload = torch.as_tensor(
            np.array(payloads[start:stop], copy=True), dtype=torch.long, device=device
        )
        order = _fresh_orders(stop - start, device=device, generator=generator)
        shuffled = payload.reshape(stop - start, -1).gather(1, order).reshape(-1, 9, 9)
        if include_hints:
            tokens = encode_partial_grids(
                puzzle,
                shuffled,
                order.reshape(-1, 9, 9) if keyed else None,
            )
        else:
            tokens = encode_partial_grids(puzzle)
        with _autocast(device, bf16):
            prediction = policy(tokens)["digit_logits"].argmax(dim=-1) + 1
        flat_prediction = prediction.reshape(stop - start, -1)
        flat_solution = solution.reshape(stop - start, -1)
        flat_payload = payload.reshape(stop - start, -1)
        blank = puzzle.reshape(stop - start, -1) == 0
        target_match = (flat_prediction == flat_payload) & blank
        solution_match = (flat_prediction == flat_solution) & blank
        counts = blank.sum(dim=-1)
        target_counts = target_match.sum(dim=-1)
        solution_counts = solution_match.sum(dim=-1)
        target_correct += int(target_counts.sum())
        solution_correct += int(solution_counts.sum())
        blank_total += int(counts.sum())
        target_exact += int((target_counts == counts).sum())
        solution_exact += int((solution_counts == counts).sum())
    return {
        "n": n,
        "blank_cells": blank_total,
        "payload_accuracy": target_correct / blank_total if blank_total else 0.0,
        "solution_accuracy": solution_correct / blank_total if blank_total else 0.0,
        "payload_exact": target_exact,
        "payload_exact_rate": target_exact / n if n else 0.0,
        "solution_exact": solution_exact,
        "solution_exact_rate": solution_exact / n if n else 0.0,
        "keyed": keyed,
        "include_hints": include_hints,
        "shuffle_seed": int(seed),
    }


def counterfactual_payloads(solutions: np.ndarray, seed: int) -> np.ndarray:
    """Change every solution digit by a seeded nonzero offset modulo nine."""
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, 9, size=solutions.shape, dtype=np.uint8)
    payloads = ((solutions.astype(np.uint16) - 1 + offsets) % 9 + 1).astype(np.uint8)
    if bool((payloads == solutions).any()):
        raise AssertionError("counterfactual construction left an unchanged digit")
    return payloads


def _load_policy(
    manifest_path: Path, checkpoint_path: Path, device: torch.device
) -> tuple[SudokuInsertionPolicy, dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("training", {}).get("condition_mode") != "keyed_shuffled_solution_hint":
        raise ValueError("manifest is not a keyed shuffled-answer run")
    spec = ModelSpec(**manifest["architecture"]["policy_spec"])
    policy = SudokuInsertionPolicy(spec).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("manifest_sha256") != manifest.get("seal", {}).get("manifest_sha256"):
        raise ValueError("checkpoint/manifest seal mismatch")
    if checkpoint.get("step") is None:
        raise ValueError("checkpoint step missing")
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    return policy, manifest, checkpoint


def run_diagnostics(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: str,
    heldout_limit: int,
    seed: int,
    bf16: bool,
) -> dict[str, Any]:
    resolved_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    policy, manifest, checkpoint = _load_policy(
        manifest_path, checkpoint_path, resolved_device
    )
    train = data_mod.load_split("train")
    test = data_mod.load_split("test")
    heldout_n = min(int(heldout_limit), test.n)
    test_puzzles = np.asarray(test.puzzles[:heldout_n])
    test_solutions = np.asarray(test.solutions[:heldout_n])
    train_puzzles = np.asarray(train.puzzles)
    train_solutions = np.asarray(train.solutions)
    test_counterfactual = counterfactual_payloads(test_solutions, seed + 301)
    train_counterfactual = counterfactual_payloads(train_solutions, seed + 211)

    metrics = {
        "train_keyed_solution": payload_metrics(
            policy, train_puzzles, train_solutions, train_solutions,
            device=resolved_device, seed=seed + 1, bf16=bf16,
        ),
        "heldout_keyed_solution": payload_metrics(
            policy, test_puzzles, test_solutions, test_solutions,
            device=resolved_device, seed=seed + 2, bf16=bf16,
        ),
        "heldout_unkeyed_shuffled_solution": payload_metrics(
            policy, test_puzzles, test_solutions, test_solutions,
            device=resolved_device, seed=seed + 2, keyed=False, bf16=bf16,
        ),
        "heldout_no_hint": payload_metrics(
            policy, test_puzzles, test_solutions, test_solutions,
            device=resolved_device, seed=seed + 2, include_hints=False, bf16=bf16,
        ),
        "train_keyed_counterfactual": payload_metrics(
            policy, train_puzzles, train_solutions, train_counterfactual,
            device=resolved_device, seed=seed + 3, bf16=bf16,
        ),
        "heldout_keyed_counterfactual": payload_metrics(
            policy, test_puzzles, test_solutions, test_counterfactual,
            device=resolved_device, seed=seed + 4, bf16=bf16,
        ),
    }
    train_rollout = solve_monotone(
        policy,
        train_puzzles,
        "random_insertion",
        device=resolved_device,
        seed=seed + 5,
        condition_mode="keyed_shuffled_solution_hint",
        solutions=train_solutions,
    ).aggregate()
    heldout_rollout = solve_monotone(
        policy,
        test_puzzles,
        "random_insertion",
        device=resolved_device,
        seed=seed + 6,
        condition_mode="keyed_shuffled_solution_hint",
        solutions=test_solutions,
    ).aggregate()
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_code": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain") or ""),
            "source_sha256": sha256_file(Path(__file__)),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "step": int(checkpoint["step"]),
            "source_commit": manifest["code"]["commit"],
            "manifest_sha256": manifest["seal"]["manifest_sha256"],
        },
        "evaluation": {
            "seed": int(seed),
            "heldout_limit": heldout_n,
            "bf16": bool(bf16),
            "device": str(resolved_device),
            "counterfactual": "every target digit shifted by a seeded nonzero offset modulo nine",
        },
        "metrics": metrics,
        "rollout": {"train": train_rollout, "heldout": heldout_rollout},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heldout-limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = run_diagnostics(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        heldout_limit=args.heldout_limit,
        seed=args.seed,
        bf16=args.bf16,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
