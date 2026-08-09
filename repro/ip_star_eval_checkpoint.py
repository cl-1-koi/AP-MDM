"""Evaluate a sealed star-graph checkpoint on the deterministic test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from repro.ip_star_data import SPLIT_SHA256, TEST_COUNT, TEST_SEED, generate_graphs, graphs_sha256
from repro.ip_star_model import InsertionDecoder, TransformerSpec
from repro.ip_star_train import evaluate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    output: str | Path,
    device: str = "cuda",
    batch_size: int = 64,
    test_count: int = TEST_COUNT,
) -> dict:
    checkpoint_path = Path(checkpoint).resolve()
    output_path = Path(output).resolve()
    resolved_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    state = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    settings = state["settings"]
    arm = settings["arm"]
    spec = TransformerSpec(**settings["decoder_spec"])
    decoder = InsertionDecoder(spec).to(resolved_device)
    graphs = generate_graphs(test_count, TEST_SEED)
    split_hash = graphs_sha256(graphs)
    if test_count == TEST_COUNT and split_hash != SPLIT_SHA256["test"]:
        raise RuntimeError("frozen test split hash mismatch")

    started = time.perf_counter()
    decoder.load_state_dict(state["decoder"])
    decoder.eval()
    raw = evaluate(
        decoder, graphs, arm=arm, device=resolved_device, batch_size=batch_size
    )
    weights = {"raw": raw}

    ema_state = state.get("ema")
    if ema_state and ema_state.get("initialized"):
        decoder.load_state_dict(ema_state["shadow"])
        decoder.eval()
        weights["ema"] = evaluate(
            decoder, graphs, arm=arm, device=resolved_device, batch_size=batch_size
        )

    result = {
        "schema_version": 1,
        "kind": "star_graph_checkpoint_test_evaluation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": int(state["step"]),
        "arm": arm,
        "test_count": test_count,
        "test_seed": TEST_SEED,
        "test_split_sha256": split_hash,
        "decoding": "search-free greedy",
        "weights": weights,
        "elapsed_s": time.perf_counter() - started,
        "source_commit": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain")),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=TEST_COUNT)
    arguments = parser.parse_args()
    result = evaluate_checkpoint(
        arguments.checkpoint,
        output=arguments.output,
        device=arguments.device,
        batch_size=arguments.batch_size,
        test_count=arguments.test_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
