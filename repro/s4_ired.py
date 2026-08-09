"""Transformer IRED comparator on the frozen S4 keyed-payload benchmark."""

from __future__ import annotations

import argparse
import json
import math
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
import torch.nn as nn
import torch.nn.functional as F

from repro.hashing import sha256_file, sha256_json
from repro.model import DDiTBlock, EmbeddingLayer, LayerNorm, Rotary, TimestepEmbedder
from repro.paths import repo_root
from repro.small_grokking import (
    CELLS,
    SIZE,
    S4Split,
    VOCAB_SIZE as CONDITION_VOCAB,
    counterfactual_payloads,
    encode_state,
    frozen_splits,
    is_valid,
)
from repro.telemetry import TelemetryWriter, gpu_snapshot
from repro.trainer import constant_schedule_with_warmup, seed_everything


CONDITION_LENGTH = 4 * CELLS
LANDSCAPES = 10


def cosine_schedule(steps: int = LANDSCAPES) -> dict[str, torch.Tensor]:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((x / steps) + 0.008) / 1.008 * math.pi * 0.5).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0, 0.999)
    alphas = 1 - betas
    cumulative = torch.cumprod(alphas, 0).float()
    previous = F.pad(cumulative[:-1], (1, 0), value=1.0)
    return {
        "sqrt_alpha": cumulative.sqrt(),
        "sqrt_one_minus": (1 - cumulative).sqrt(),
        "sqrt_recip": (1 / cumulative).sqrt(),
        "sqrt_recipm1": (1 / cumulative - 1).sqrt(),
        "posterior_variance": (betas.float() * (1 - previous) / (1 - cumulative)).clamp_min(1e-20),
        "posterior_mean_coef1": betas.float() * previous.sqrt() / (1 - cumulative),
        "posterior_mean_coef2": (1 - previous) * alphas.float().sqrt() / (1 - cumulative),
        "opt_step_size": betas.float() * (1 / (1 - cumulative)).sqrt(),
    }


class S4Energy(nn.Module):
    def __init__(self, width: int = 64, blocks: int = 2, heads: int = 4):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.n_blocks = blocks
        self.n_heads = heads
        self.condition_embed = EmbeddingLayer(width, CONDITION_VOCAB)
        self.output_embed = nn.Linear(SIZE, width)
        self.type_embed = nn.Parameter(torch.zeros(2, width))
        nn.init.normal_(self.type_embed, std=0.02)
        self.sigma_map = TimestepEmbedder(max(32, width // 2))
        self.rotary = Rotary(width // heads)
        self.blocks = nn.ModuleList(
            DDiTBlock(width, heads, max(32, width // 2), dropout=0.1)
            for _ in range(blocks)
        )
        self.norm = LayerNorm(width)
        self.residual = nn.Linear(width, SIZE)
        nn.init.normal_(self.residual.weight, std=0.02)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self, condition: torch.Tensor, output: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        if condition.shape[1:] != (CONDITION_LENGTH,):
            raise ValueError("condition must have 64 tokens")
        if output.shape[1:] != (CELLS, SIZE):
            raise ValueError("output must have shape (B,16,4)")
        batch = len(condition)
        sigma = timestep.float().reshape(batch) / max(LANDSCAPES - 1, 1)
        conditioning = F.silu(self.sigma_map(sigma))
        condition_hidden = self.condition_embed(condition) + self.type_embed[0]
        output_hidden = self.output_embed(output) + self.type_embed[1]
        hidden = torch.cat((condition_hidden, output_hidden), dim=1)
        rotary = self.rotary(hidden.shape[1], hidden.device, hidden.dtype)
        for block in self.blocks:
            hidden = block(
                hidden, rotary, conditioning, manual_attention=True
            )
        residual = self.residual(self.norm(hidden[:, CONDITION_LENGTH:]))
        return residual.square().sum(dim=(1, 2), keepdim=True)

    def signature(self) -> str:
        return (
            f"s4-ired-transformer|L={self.n_blocks}|H={self.n_heads}"
            f"|d={self.width}|condition={CONDITION_LENGTH}|output={CELLS}x{SIZE}"
        )


def _one_hot(solutions: torch.Tensor) -> torch.Tensor:
    return (F.one_hot(solutions.reshape(-1, CELLS) - 1, SIZE).float() - 0.5) * 2


def _q_sample(
    target: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    shape = (len(target), 1, 1)
    a = schedule["sqrt_alpha"].to(target.device)[timestep].reshape(shape)
    b = schedule["sqrt_one_minus"].to(target.device)[timestep].reshape(shape)
    return a * target + b * noise


def energy_gradient(
    model: S4Energy,
    condition: torch.Tensor,
    output: torch.Tensor,
    timestep: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = output.requires_grad_(True)
    energy = model(condition, output, timestep)
    gradient = torch.autograd.grad(
        energy.sum(), output, create_graph=create_graph
    )[0]
    return energy, gradient


def ired_loss(
    model: S4Energy,
    condition: torch.Tensor,
    solutions: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    schedule = cosine_schedule()
    target = _one_hot(solutions)
    timestep = torch.randint(
        0, LANDSCAPES, (len(target),),
        device=target.device, generator=generator,
    )
    noise = torch.randn(
        target.shape, device=target.device, generator=generator
    )
    noisy = _q_sample(target, timestep, noise, schedule)
    _, predicted_noise = energy_gradient(
        model, condition, noisy, timestep, create_graph=True
    )
    denoise = F.mse_loss(predicted_noise, noise)

    digits = solutions.reshape(-1, CELLS) - 1
    corrupt_mask = torch.rand(
        digits.shape, device=digits.device, generator=generator
    ) < 0.05
    random_digit = torch.randint(
        0, SIZE, digits.shape, device=digits.device, generator=generator
    )
    negative_digit = torch.where(corrupt_mask, random_digit, digits)
    negative = (F.one_hot(negative_digit, SIZE).float() - 0.5) * 2
    real_noisy = _q_sample(target, timestep, noise, schedule)
    fake_noisy = _q_sample(negative, timestep, noise, schedule)
    real_energy = model(condition, real_noisy, timestep)
    fake_energy = model(condition, fake_noisy, timestep)
    logits = -torch.cat((real_energy, fake_energy), dim=-1).reshape(len(target), 2)
    energy_nce = F.cross_entropy(
        logits, torch.zeros(len(target), dtype=torch.long, device=target.device)
    )
    total = denoise + 0.05 * energy_nce
    return total, {
        "loss_denoise": denoise.detach(),
        "loss_energy_nce": energy_nce.detach(),
        "real_energy": real_energy.mean().detach(),
        "fake_energy": fake_energy.mean().detach(),
        "gradient_norm_output": predicted_noise.detach().norm(dim=(1, 2)).mean(),
        "mean_timestep": timestep.float().mean().detach(),
        "negative_changed_rate": corrupt_mask.float().mean().detach(),
    }


def _given_condition(
    puzzles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = puzzles.reshape(-1, CELLS)
    mask = (flat != 0)[..., None]
    safe = (flat - 1).clamp_min(0)
    value = (F.one_hot(safe, SIZE).float() - 0.5) * 2
    return value, mask


def _inner_optimize(
    model: S4Energy,
    condition: torch.Tensor,
    output: torch.Tensor,
    timestep: torch.Tensor,
    step_size: float,
    steps: int,
    clamp: float,
    given_value: torch.Tensor,
    given_mask: torch.Tensor,
) -> torch.Tensor:
    for _ in range(steps):
        with torch.enable_grad():
            energy, gradient = energy_gradient(
                model, condition, output.detach(), timestep, create_graph=False
            )
        candidate = (output - step_size * gradient).clamp(-clamp, clamp).detach()
        candidate = torch.where(given_mask, clamp * given_value, candidate)
        with torch.no_grad():
            candidate_energy = model(condition, candidate, timestep)
            rejected = (candidate_energy > energy.detach()).reshape(len(output))
            candidate[rejected] = output[rejected]
        output = candidate
    return output


@torch.no_grad()
def generate(
    model: S4Energy,
    condition: torch.Tensor,
    puzzles: torch.Tensor,
    *,
    seed: int,
    inner_steps: int = 20,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = condition.device
    generator = torch.Generator(device=device).manual_seed(seed)
    schedule = {name: value.to(device) for name, value in cosine_schedule().items()}
    output = torch.randn(
        (len(condition), CELLS, SIZE), device=device, generator=generator
    )
    given_value, given_mask = _given_condition(puzzles)
    gradient_norms = []
    model.eval()
    for time_index in reversed(range(LANDSCAPES)):
        timestep = torch.full(
            (len(condition),), time_index, dtype=torch.long, device=device
        )
        with torch.enable_grad():
            _, predicted_noise = energy_gradient(
                model, condition, output.detach(), timestep, create_graph=False
            )
        gradient_norms.append(float(predicted_noise.norm(dim=(1, 2)).mean()))
        x_start = (
            schedule["sqrt_recip"][time_index] * output
            - schedule["sqrt_recipm1"][time_index] * predicted_noise
        ).clamp(-1, 1)
        mean = (
            schedule["posterior_mean_coef1"][time_index] * x_start
            + schedule["posterior_mean_coef2"][time_index] * output
        )
        if time_index:
            noise = torch.randn(
                output.shape, device=device, generator=generator
            )
            output = mean + schedule["posterior_variance"][time_index].sqrt() * noise
        else:
            output = mean
        clamp = float(schedule["sqrt_alpha"][time_index])
        output = _inner_optimize(
            model,
            condition,
            output,
            timestep,
            float(schedule["opt_step_size"][time_index]),
            inner_steps,
            clamp,
            given_value,
            given_mask,
        )
    digits = output.argmax(-1) + 1
    return digits, {
        "landscapes": LANDSCAPES,
        "inner_steps_per_landscape": inner_steps,
        "energy_gradient_norm_mean": float(np.mean(gradient_norms)),
    }


def conditions(
    puzzles: torch.Tensor,
    payloads: torch.Tensor,
    generator: torch.Generator,
    *,
    keyed: bool = True,
    include_hints: bool = True,
) -> torch.Tensor:
    order = torch.rand(
        (len(puzzles), CELLS), device=puzzles.device, generator=generator
    ).argsort(-1)
    return encode_state(
        puzzles, payloads, order, keyed=keyed, include_hints=include_hints
    )


def _score(
    predicted: np.ndarray, split: S4Split, payloads: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = split.solutions.reshape(-1, CELLS)
    payload = payloads.reshape(-1, CELLS)
    blank = split.puzzles.reshape(-1, CELLS) == 0
    flat = predicted.reshape(-1, CELLS)
    solution_match = (flat == target) & blank
    payload_match = (flat == payload) & blank
    blank_counts = blank.sum(-1)
    solution_counts = solution_match.sum(-1)
    payload_counts = payload_match.sum(-1)
    exact = (flat == target).all(-1)
    payload_exact = ((flat == payload) | ~blank).all(-1)
    valid = np.asarray([is_valid(grid) for grid in predicted])
    rows = [
        {
            "index": row, "grid": predicted[row].tolist(),
            "exact": bool(exact[row]), "valid": bool(valid[row]),
            "blank_correct": int(solution_counts[row]),
            "blank_total": int(blank_counts[row]),
        }
        for row in range(split.n)
    ]
    blank_total = int(blank_counts.sum())
    return {
        "n": split.n, "exact": int(exact.sum()), "exact_rate": float(exact.mean()),
        "valid": int(valid.sum()), "valid_rate": float(valid.mean()),
        "blank_accuracy": float(solution_counts.sum() / blank_total),
        "payload_accuracy": float(payload_counts.sum() / blank_total),
        "payload_exact": int(payload_exact.sum()),
        "payload_exact_rate": float(payload_exact.mean()),
    }, rows


@torch.no_grad()
def evaluate_condition(
    model: S4Energy,
    split: S4Split,
    payloads: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    inner_steps: int,
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
    predicted, reasoning = generate(
        model, condition, puzzles, seed=seed + 17, inner_steps=inner_steps
    )
    aggregate, rows = _score(
        predicted.cpu().numpy().reshape(-1, SIZE, SIZE), split, payloads
    )
    aggregate.update(
        {**reasoning, "shuffle_seed": seed, "keyed": keyed,
         "include_hints": include_hints}
    )
    return aggregate, rows


def diagnostic_panel(
    model: S4Energy,
    split: S4Split,
    *,
    device: torch.device,
    seed: int,
    inner_steps: int,
) -> dict[str, Any]:
    normal, _ = evaluate_condition(
        model, split, split.solutions, device=device, seed=seed,
        inner_steps=inner_steps,
    )
    unkeyed, _ = evaluate_condition(
        model, split, split.solutions, device=device, seed=seed,
        inner_steps=inner_steps, keyed=False,
    )
    no_hint, _ = evaluate_condition(
        model, split, split.solutions, device=device, seed=seed,
        inner_steps=inner_steps, include_hints=False,
    )
    counterfactual, _ = evaluate_condition(
        model, split, counterfactual_payloads(split.solutions, seed + 1),
        device=device, seed=seed, inner_steps=inner_steps,
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
    inner_steps: int
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
    model = S4Energy(settings.width, settings.blocks, settings.heads).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=settings.weight_decay,
    )
    scheduler = constant_schedule_with_warmup(optimizer, settings.warmup_steps)
    manifest_content = {
        "schema": "apmdm/s4-ired-manifest-v1",
        "experiment": "S4 Iterative Reasoning through Energy Diffusion comparator",
        "settings": asdict(settings),
        "mechanism": {
            "arm": "ired", "energy_landscapes": LANDSCAPES,
            "inner_gradient_steps": settings.inner_steps,
            "loss": "energy-gradient noise prediction plus 0.05-weight NCE",
            "negative_corruption_probability": 0.05,
            "sampler": "reverse diffusion with per-landscape energy minimization",
        },
        "architecture": {
            "name": "Transformer energy network (shared token condition ABI)",
            "signature": model.signature(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "released_comparator_note": "mechanism follows released IRED objective; replaces the hard-coded 9x9 CNN with the common S4 Transformer condition encoder",
        },
        "data": {"train": train.provenance(), "test": test.provenance(), "overlap": overlap},
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
                    "repro/s4_ired.py", "repro/model.py", "repro/small_grokking.py",
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
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if autocast_enabled else nullcontext()
            )
            with context:
                loss, metrics = ired_loss(model, condition, solutions, generator)
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
                        model, split, split.solutions, device=device,
                        seed=eval_seed, inner_steps=settings.inner_steps,
                    )
                    diagnostics = diagnostic_panel(
                        model, split, device=device, seed=eval_seed,
                        inner_steps=settings.inner_steps,
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
                    "version": 1, "step": step, "arm": "ired",
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
        "schema": "apmdm/s4-ired-completion-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": manifest["code"]["commit"],
        "manifest_sha256": manifest["seal"]["manifest_sha256"],
        "arm": "ired", "step": settings.max_steps, "files": files,
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
    parser.add_argument("--inner-steps", type=int, default=20)
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
        warmup_steps=args.warmup_steps, inner_steps=args.inner_steps,
        seed=args.seed, device=args.device, bf16=args.bf16,
        log_every=args.log_every,
        eval_steps=parse_eval_steps(args.eval_steps, args.max_steps),
    )), sort_keys=True))


if __name__ == "__main__":
    main()
