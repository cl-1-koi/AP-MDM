"""Four-operation AP-MDM on the frozen S4 keyed-payload benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from repro.config import ModelSpec
from repro.hashing import sha256_file, sha256_json
from repro.losses import supervised_loss
from repro.model import APMDMEncoder
from repro.paths import repo_root
from repro.small_grokking import (
    CELLS,
    SIZE,
    S4Split,
    counterfactual_payloads,
    encode_state,
    frozen_splits,
    is_valid,
)
from repro.telemetry import TelemetryWriter, gpu_snapshot
from repro.trainer import constant_schedule_with_warmup, seed_everything


CONDITION_LENGTH = 4 * CELLS
BASE_VOCAB = 25
MASK = BASE_VOCAB
BOS = BASE_VOCAB + 1
PAD = BASE_VOCAB + 2
VOCAB_SIZE = BASE_VOCAB + 3
# A corruption contains BOS, every non-omitted target, and independently up to
# one inserted extra after each target.  The random process does not cap extras
# at eight, so the model bound must cover its true (rare) maximum.
MAX_OUTPUT_LENGTH = 1 + 2 * CELLS
MAX_PACKED_LENGTH = CONDITION_LENGTH + MAX_OUTPUT_LENGTH


@dataclass(frozen=True)
class CorruptionBatch:
    indices: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    y_star: torch.Tensor
    remask: torch.Tensor
    insert: torch.Tensor
    delete: torch.Tensor
    sigma: torch.Tensor


def model_spec(width: int = 64, blocks: int = 2, heads: int = 4) -> ModelSpec:
    return ModelSpec(
        vocab_size=VOCAB_SIZE,
        length=MAX_PACKED_LENGTH,
        hidden_size=width,
        n_heads=heads,
        n_blocks=blocks,
        cond_dim=max(32, width // 2),
        dropout=0.1,
        mlp_ratio=4,
    )


def _hint_order(
    batch: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    return torch.rand((batch, CELLS), device=device, generator=generator).argsort(-1)


def conditions(
    puzzles: torch.Tensor,
    payloads: torch.Tensor,
    generator: torch.Generator,
    *,
    keyed: bool = True,
    include_hints: bool = True,
) -> torch.Tensor:
    return encode_state(
        puzzles,
        payloads,
        _hint_order(len(puzzles), puzzles.device, generator),
        keyed=keyed,
        include_hints=include_hints,
    )


def _compress_expanded(
    value: torch.Tensor, valid: torch.Tensor, fill: int | bool
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = valid.sum(-1)
    width = int(lengths.max())
    output = torch.full(
        (len(value), width), fill,
        dtype=value.dtype, device=value.device,
    )
    positions = valid.long().cumsum(-1) - 1
    rows = torch.arange(len(value), device=value.device)[:, None].expand_as(value)
    output[rows[valid], positions[valid]] = value[valid]
    mask = torch.arange(width, device=value.device)[None] < lengths[:, None]
    return output, mask


def corrupt_targets(
    condition: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
) -> CorruptionBatch:
    """Create aligned variable-length supervision for all four AP operations."""
    batch = len(targets)
    random = torch.rand(
        (4, batch, CELLS), device=targets.device, generator=generator
    )
    raw_omit = random[0] < 0.14
    omit = raw_omit & torch.cat(
        (torch.ones_like(raw_omit[:, :1]), ~raw_omit[:, :-1]), dim=1
    )
    masked = (random[1] < 0.22) & ~omit
    wrong = (random[1] >= 0.22) & (random[1] < 0.38) & ~omit
    extra = (random[2] < 0.12) & ~omit
    wrong_offset = (torch.floor(random[3] * (SIZE - 1)).long() + 1)
    wrong_digit = (targets - 1 + wrong_offset) % SIZE + 1

    expanded_length = 1 + 2 * CELLS
    x = torch.full(
        (batch, expanded_length), PAD, dtype=torch.long, device=targets.device
    )
    y = torch.full_like(x, PAD)
    valid = torch.zeros_like(x, dtype=torch.bool)
    remask = torch.zeros_like(x, dtype=torch.bool)
    insert = torch.zeros_like(x, dtype=torch.bool)
    delete = torch.zeros_like(x, dtype=torch.bool)
    x[:, 0] = BOS
    y[:, 0] = BOS
    valid[:, 0] = True
    for cell in range(CELLS):
        main = 1 + 2 * cell
        extra_position = main + 1
        present = ~omit[:, cell]
        value = torch.where(
            masked[:, cell],
            torch.full_like(targets[:, cell], MASK),
            torch.where(wrong[:, cell], wrong_digit[:, cell], targets[:, cell]),
        )
        x[:, main] = value
        y[:, main] = targets[:, cell]
        valid[:, main] = present
        remask[:, main] = wrong[:, cell]
        x[:, extra_position] = MASK
        y[:, extra_position] = MASK
        valid[:, extra_position] = extra[:, cell]
        delete[:, extra_position] = extra[:, cell]
        anchor = 0 if cell == 0 else 1 + 2 * (cell - 1)
        insert[:, anchor] |= omit[:, cell]

    packed_x, output_mask = _compress_expanded(x, valid, PAD)
    packed_y, _ = _compress_expanded(y, valid, PAD)
    packed_r, _ = _compress_expanded(remask, valid, False)
    packed_i, _ = _compress_expanded(insert, valid, False)
    packed_d, _ = _compress_expanded(delete, valid, False)
    width = CONDITION_LENGTH + packed_x.shape[1]
    indices = torch.full(
        (batch, width), PAD, dtype=torch.long, device=targets.device
    )
    y_star = torch.full_like(indices, PAD)
    indices[:, :CONDITION_LENGTH] = condition
    indices[:, CONDITION_LENGTH:] = packed_x
    y_star[:, CONDITION_LENGTH:] = packed_y
    attention_mask = torch.zeros_like(indices, dtype=torch.bool)
    attention_mask[:, :CONDITION_LENGTH] = True
    attention_mask[:, CONDITION_LENGTH:] = output_mask
    loss_mask = torch.zeros_like(attention_mask)
    loss_mask[:, CONDITION_LENGTH:] = output_mask
    def controls(value: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(indices, dtype=torch.bool)
        result[:, CONDITION_LENGTH:] = value
        return result
    corruption_count = (
        omit.sum(-1) + masked.sum(-1) + wrong.sum(-1) + extra.sum(-1)
    )
    return CorruptionBatch(
        indices=indices,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        y_star=y_star,
        remask=controls(packed_r),
        insert=controls(packed_i),
        delete=controls(packed_d),
        sigma=corruption_count.float() / CELLS,
    )


def apmdm_loss(
    model: APMDMEncoder, batch: CorruptionBatch
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs = model(
        batch.indices, sigma=batch.sigma, attention_mask=batch.attention_mask
    )
    losses = supervised_loss(
        outputs,
        batch.indices,
        batch.y_star,
        batch.remask,
        batch.insert,
        batch.delete,
        attention_mask=batch.loss_mask,
        mask_index=MASK,
    )
    metrics: dict[str, torch.Tensor] = {
        "loss_unmask": losses.unmask.detach(),
        "loss_remask": losses.remask.detach(),
        "loss_insert": losses.insert.detach(),
        "loss_delete": losses.delete.detach(),
        "unmask_positions": torch.as_tensor(
            losses.n_unmask_positions, device=batch.indices.device
        ),
        "mean_sigma": batch.sigma.mean().detach(),
    }
    valid = batch.loss_mask
    valid_count = valid.sum().clamp_min(1)
    for name, key, target in (
        ("remask", "remasking_logits", batch.remask),
        ("insert", "expansion_logits", batch.insert),
        ("delete", "contraction_logits", batch.delete),
    ):
        probability = torch.sigmoid(outputs[key].squeeze(-1).float())
        predicted = probability > 0.5
        positives = target & valid
        metrics[f"{name}_target_rate"] = positives.sum().float() / valid_count
        metrics[f"{name}_predicted_rate"] = (predicted & valid).sum().float() / valid_count
        metrics[f"{name}_probability_mean"] = (
            probability * valid
        ).sum() / valid_count
        metrics[f"{name}_recall"] = (
            (predicted & positives).sum().float() / positives.sum().clamp_min(1)
        )
    return losses.total, metrics


def _pack_generation(
    condition: torch.Tensor, states: list[list[int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(state) for state in states)
    indices = torch.full(
        (len(states), CONDITION_LENGTH + width),
        PAD, dtype=torch.long, device=condition.device,
    )
    attention = torch.zeros_like(indices, dtype=torch.bool)
    indices[:, :CONDITION_LENGTH] = condition
    attention[:, :CONDITION_LENGTH] = True
    for row, state in enumerate(states):
        indices[row, CONDITION_LENGTH : CONDITION_LENGTH + len(state)] = torch.as_tensor(
            state, dtype=torch.long, device=condition.device
        )
        attention[row, CONDITION_LENGTH : CONDITION_LENGTH + len(state)] = True
    return indices, attention


@torch.no_grad()
def generate(
    model: APMDMEncoder,
    condition: torch.Tensor,
    *,
    bf16: bool,
    max_steps: int = 32,
    threshold: float = 0.5,
) -> tuple[list[list[int]], list[bool], list[dict[str, int]]]:
    states = [[BOS] + [MASK] * CELLS for _ in range(len(condition))]
    active = [True] * len(states)
    terminated = [False] * len(states)
    traces = [
        {"steps": 0, "unmask": 0, "remask": 0, "insert": 0,
         "delete": 0, "length_changes": 0}
        for _ in states
    ]
    model.eval()
    for step in range(1, max_steps + 1):
        indices, attention = _pack_generation(condition, states)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if indices.device.type == "cuda" and bf16 else nullcontext()
        )
        sigma = torch.as_tensor(
            [state.count(MASK) / CELLS for state in states],
            dtype=torch.float32, device=indices.device,
        )
        with context:
            outputs = model(indices, sigma=sigma, attention_mask=attention)
        for row, state in enumerate(states):
            if not active[row]:
                continue
            length = len(state)
            start = CONDITION_LENGTH
            content = outputs["unmasking_logits"][row, start : start + length, 1 : SIZE + 1]
            predicted_digit = content.argmax(-1) + 1
            remask = torch.sigmoid(
                outputs["remasking_logits"][row, start : start + length, 0].float()
            ) > threshold
            insert = torch.sigmoid(
                outputs["expansion_logits"][row, start : start + length, 0].float()
            ) > threshold
            delete = torch.sigmoid(
                outputs["contraction_logits"][row, start : start + length, 0].float()
            ) > threshold
            successor: list[int] = []
            for position, token in enumerate(state):
                if position == 0:
                    successor.append(BOS)
                    if bool(insert[position]):
                        successor.append(MASK)
                        traces[row]["insert"] += 1
                    continue
                if bool(delete[position]) and token == MASK:
                    traces[row]["delete"] += 1
                    continue
                if bool(remask[position]):
                    successor.append(MASK)
                    traces[row]["remask"] += 1
                elif token == MASK:
                    successor.append(int(predicted_digit[position]))
                    traces[row]["unmask"] += 1
                else:
                    successor.append(token)
                if bool(insert[position]):
                    successor.append(MASK)
                    traces[row]["insert"] += 1
            traces[row]["steps"] = step
            if len(successor) != len(state):
                traces[row]["length_changes"] += 1
            if successor == state:
                active[row] = False
                terminated[row] = True
            elif len(successor) > MAX_OUTPUT_LENGTH or not successor:
                active[row] = False
            else:
                states[row] = successor
        if not any(active):
            break
    return states, terminated, traces


def _score(
    states: list[list[int]],
    terminated: list[bool],
    traces: list[dict[str, int]],
    split: S4Split,
    payloads: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = split.solutions.reshape(-1, CELLS)
    payload = payloads.reshape(-1, CELLS)
    blank = split.puzzles.reshape(-1, CELLS) == 0
    predicted = np.zeros_like(target)
    lengths = []
    for row, state in enumerate(states):
        sequence = state[1:] if state and state[0] == BOS else state
        lengths.append(len(sequence))
        predicted[row, : min(CELLS, len(sequence))] = sequence[:CELLS]
    complete = np.asarray(lengths) == CELLS
    solution_match = (predicted == target) & blank
    payload_match = (predicted == payload) & blank
    blank_counts = blank.sum(-1)
    solution_counts = solution_match.sum(-1)
    payload_counts = payload_match.sum(-1)
    exact = complete & (predicted == target).all(-1)
    payload_exact = complete & ((predicted == payload) | ~blank).all(-1)
    valid = complete & np.asarray(
        [is_valid(grid.reshape(SIZE, SIZE)) for grid in predicted]
    )
    rows = [
        {
            "index": row, "state": states[row], "trace": traces[row],
            "predicted_length": lengths[row], "terminated": bool(terminated[row]),
            "exact": bool(exact[row]), "valid": bool(valid[row]),
            "blank_correct": int(solution_counts[row]),
            "blank_total": int(blank_counts[row]),
        }
        for row in range(split.n)
    ]
    blank_total = int(blank_counts.sum())
    aggregate = {
        "n": split.n, "exact": int(exact.sum()), "exact_rate": float(exact.mean()),
        "valid": int(valid.sum()), "valid_rate": float(valid.mean()),
        "terminated": int(sum(terminated)), "terminated_rate": float(np.mean(terminated)),
        "complete_length": int(complete.sum()),
        "mean_predicted_length": float(np.mean(lengths)),
        "blank_accuracy": float(solution_counts.sum() / blank_total),
        "payload_accuracy": float(payload_counts.sum() / blank_total),
        "payload_exact": int(payload_exact.sum()),
        "payload_exact_rate": float(payload_exact.mean()),
    }
    for name in ("steps", "unmask", "remask", "insert", "delete", "length_changes"):
        aggregate[f"mean_{name}"] = float(np.mean([trace[name] for trace in traces]))
    return aggregate, rows


@torch.no_grad()
def evaluate_condition(
    model: APMDMEncoder,
    split: S4Split,
    payloads: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    bf16: bool,
    keyed: bool = True,
    include_hints: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    puzzles = torch.as_tensor(
        np.array(split.puzzles, copy=True), dtype=torch.long, device=device
    )
    payload = torch.as_tensor(
        np.array(payloads, copy=True), dtype=torch.long, device=device
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    condition = conditions(
        puzzles, payload, generator, keyed=keyed, include_hints=include_hints
    )
    states, terminated, traces = generate(model, condition, bf16=bf16)
    aggregate, rows = _score(states, terminated, traces, split, payloads)
    aggregate.update(
        {"shuffle_seed": seed, "keyed": keyed, "include_hints": include_hints}
    )
    return aggregate, rows


def diagnostic_panel(
    model: APMDMEncoder,
    split: S4Split,
    *,
    device: torch.device,
    seed: int,
    bf16: bool,
) -> dict[str, Any]:
    normal, _ = evaluate_condition(
        model, split, split.solutions, device=device, seed=seed, bf16=bf16
    )
    unkeyed, _ = evaluate_condition(
        model, split, split.solutions, device=device, seed=seed,
        bf16=bf16, keyed=False,
    )
    no_hint, _ = evaluate_condition(
        model, split, split.solutions, device=device, seed=seed,
        bf16=bf16, include_hints=False,
    )
    counterfactual, _ = evaluate_condition(
        model, split, counterfactual_payloads(split.solutions, seed + 1),
        device=device, seed=seed, bf16=bf16,
    )
    return {
        "keyed_solution": normal,
        "unkeyed_shuffled_solution": unkeyed,
        "no_hint": no_hint,
        "keyed_counterfactual": counterfactual,
        "key_ablation_delta": normal["blank_accuracy"] - unkeyed["blank_accuracy"],
        "hint_ablation_delta": normal["blank_accuracy"] - no_hint["blank_accuracy"],
    }


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root()), *args], capture_output=True,
        text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class Settings:
    output: str
    max_steps: int
    batch_size: int
    width: int
    blocks: int
    heads: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    seed: int
    device: str
    bf16: bool
    log_every: int
    eval_steps: tuple[int, ...]


def run(settings: Settings) -> dict[str, Any]:
    output = Path(settings.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if _git("status", "--porcelain"):
        raise RuntimeError("refusing to run from a dirty worktree")
    output.mkdir(parents=True)
    train, test, overlap = frozen_splits(settings.seed)
    device = torch.device(
        settings.device if settings.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    seed_everything(settings.seed)
    generator = torch.Generator(device=device).manual_seed(settings.seed + 17)
    spec = model_spec(settings.width, settings.blocks, settings.heads)
    model = APMDMEncoder(spec, time_conditioning=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=settings.weight_decay,
    )
    scheduler = constant_schedule_with_warmup(optimizer, settings.warmup_steps)
    manifest_content = {
        "schema": "apmdm/s4-apmdm-manifest-v1",
        "experiment": "S4 four-operation AP-MDM grokking phase diagram",
        "settings": asdict(settings),
        "mechanism": {
            "arm": "ap_mdm", "fixed_canvas": False, "variable_length": True,
            "operations": ["unmask", "remask", "insert", "delete"],
            "time_conditioning": True,
            "trajectory_source": "on-the-fly aligned corruptions of terminal grids",
        },
        "data": {"train": train.provenance(), "test": test.provenance(), "overlap": overlap},
        "architecture": {
            "signature": model.architecture_signature(), "spec": spec.as_dict(),
            "parameters": model.parameter_count().as_dict(),
        },
        "optimizer": {
            "name": "AdamW", "lr": settings.learning_rate,
            "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": settings.weight_decay, "gradient_clip": 1.0,
            "warmup_steps": settings.warmup_steps,
            "schedule": "constant after linear warmup",
        },
        "code": {
            "commit": _git("rev-parse", "HEAD"), "dirty": False,
            "sources": {
                name: sha256_file(repo_root() / name)
                for name in (
                    "repro/s4_apmdm.py", "repro/model.py", "repro/losses.py",
                    "repro/small_grokking.py",
                    "SMALL_SUDOKU_GROKKING_LADDER_20260809.md",
                )
            },
        },
        "hardware": {"device": str(device), **gpu_snapshot(device)},
    }
    manifest = {
        **manifest_content,
        "seal": {"manifest_sha256": sha256_json(manifest_content)},
    }
    _write_json(output / "manifest.json", manifest)
    telemetry = TelemetryWriter(output / "telemetry.jsonl", run_id=output.name)
    train_puzzles = torch.as_tensor(
        np.array(train.puzzles, copy=True), dtype=torch.long, device=device
    )
    train_solutions = torch.as_tensor(
        np.array(train.solutions, copy=True), dtype=torch.long, device=device
    )
    autocast_enabled = device.type == "cuda" and settings.bf16
    started = window_started = time.time()
    window_examples = 0
    evaluations: list[dict[str, Any]] = []
    try:
        for step in range(1, settings.max_steps + 1):
            model.train()
            indices = torch.randint(
                0, train.n, (settings.batch_size,), device=device, generator=generator
            )
            puzzles = train_puzzles[indices]
            solutions = train_solutions[indices]
            condition = conditions(puzzles, solutions, generator)
            batch = corrupt_targets(condition, solutions.reshape(-1, CELLS), generator)
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if autocast_enabled else nullcontext()
            )
            with context:
                loss, metrics = apmdm_loss(model, batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            window_examples += settings.batch_size
            if step % settings.log_every == 0 or step == settings.max_steps:
                now = time.time()
                elapsed = max(now - window_started, 1e-9)
                telemetry.write(
                    "train", step=step, loss=float(loss.detach()),
                    grad_norm=float(grad_norm),
                    steps_per_second=settings.log_every / elapsed,
                    examples_per_second=window_examples / elapsed,
                    lr=[float(value) for value in scheduler.get_last_lr()],
                    **{name: float(value) for name, value in metrics.items()},
                    **{f"gpu_{name}": value for name, value in gpu_snapshot(device).items()},
                )
                window_started = now
                window_examples = 0
            if step in settings.eval_steps or step == settings.max_steps:
                for split_name, split in (("train", train), ("heldout", test)):
                    eval_seed = settings.seed + step + (
                        0 if split_name == "train" else 100_000
                    )
                    aggregate, rows = evaluate_condition(
                        model, split, split.solutions,
                        device=device, seed=eval_seed, bf16=settings.bf16,
                    )
                    diagnostics = diagnostic_panel(
                        model, split, device=device, seed=eval_seed, bf16=settings.bf16
                    )
                    rows_path = output / f"eval_{split_name}_rows_step_{step:09d}.jsonl"
                    diagnostics_path = output / f"diagnostics_{split_name}_step_{step:09d}.json"
                    _write_rows(rows_path, rows)
                    _write_json(diagnostics_path, diagnostics)
                    event = {
                        "step": step, "split": split_name, **aggregate,
                        "rows_path": str(rows_path), "rows_sha256": sha256_file(rows_path),
                        "diagnostics_path": str(diagnostics_path),
                        "diagnostics_sha256": sha256_file(diagnostics_path),
                        "counterfactual_payload_accuracy": diagnostics["keyed_counterfactual"]["payload_accuracy"],
                        "counterfactual_solution_accuracy": diagnostics["keyed_counterfactual"]["blank_accuracy"],
                        "key_ablation_delta": diagnostics["key_ablation_delta"],
                        "hint_ablation_delta": diagnostics["hint_ablation_delta"],
                    }
                    telemetry.write("eval", **event)
                    evaluations.append(event)
                checkpoint = {
                    "version": 1, "step": step, "arm": "ap_mdm",
                    "manifest_sha256": manifest["seal"]["manifest_sha256"],
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "generator_state": generator.get_state(),
                }
                checkpoint_path = output / f"checkpoint_step_{step:09d}.pt"
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save(checkpoint, temporary)
                os.replace(temporary, checkpoint_path)
                telemetry.write(
                    "checkpoint", step=step, path=str(checkpoint_path),
                    sha256=sha256_file(checkpoint_path),
                )
        summary = {
            "completed": True, "end_step": settings.max_steps,
            "wall_seconds": time.time() - started, "evaluations": evaluations,
        }
        _write_json(output / "train_summary.json", summary)
        telemetry.write("train_summary", **summary)
    finally:
        telemetry.close()
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "completion.json"
    ]
    completion = {
        "schema": "apmdm/s4-apmdm-completion-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": manifest["code"]["commit"],
        "manifest_sha256": manifest["seal"]["manifest_sha256"],
        "arm": "ap_mdm", "step": settings.max_steps, "files": files,
    }
    _write_json(output / "completion.json", completion)
    return completion


def parse_eval_steps(text: str, max_steps: int) -> tuple[int, ...]:
    values = sorted({int(value) for value in text.split(",") if value.strip()})
    if any(value <= 0 or value > max_steps for value in values):
        raise ValueError("evaluation steps must lie in [1,max_steps]")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, choices=(0.0, 0.01), required=True)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--eval-steps", default="1000,3000,10000,30000,100000,300000,1000000"
    )
    args = parser.parse_args()
    print(json.dumps(run(Settings(
        output=args.output, max_steps=args.max_steps, batch_size=args.batch_size,
        width=args.width, blocks=args.blocks, heads=args.heads,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, seed=args.seed, device=args.device,
        bf16=args.bf16, log_every=args.log_every,
        eval_steps=parse_eval_steps(args.eval_steps, args.max_steps),
    )), sort_keys=True))


if __name__ == "__main__":
    main()
