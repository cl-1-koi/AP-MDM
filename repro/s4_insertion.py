"""Variable-length AO-IP and learned IP on the frozen S4 Sudoku task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from repro.hashing import sha256_file, sha256_json
from repro.ip_star_model import SequenceTransformer, TransformerSpec
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


Arm = Literal["ao_ip", "ip"]
ARMS = frozenset(("ao_ip", "ip"))
BASE_VOCAB = 25
EOS = BASE_VOCAB
PAD = BASE_VOCAB + 1
VOCAB_SIZE = BASE_VOCAB + 2
CONDITION_LENGTH = 4 * CELLS


@dataclass(frozen=True)
class InsertionBatch:
    conditions: torch.Tensor
    condition_mask: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    base_mask: torch.Tensor
    lengths: torch.Tensor


@dataclass(frozen=True)
class DecoderOutput:
    location_logits: torch.Tensor
    content_logits: torch.Tensor
    slot_mask: torch.Tensor


def _hint_order(
    batch: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    return torch.rand((batch, CELLS), device=device, generator=generator).argsort(-1)


def make_batch(
    puzzles: torch.Tensor,
    solutions: torch.Tensor,
    generator: torch.Generator,
    *,
    payloads: torch.Tensor | None = None,
    keyed: bool = True,
    include_hints: bool = True,
) -> InsertionBatch:
    if puzzles.shape != solutions.shape or puzzles.shape[1:] != (SIZE, SIZE):
        raise ValueError("expected matching (B,4,4) puzzles and solutions")
    payloads = solutions if payloads is None else payloads
    if payloads.shape != solutions.shape:
        raise ValueError("payloads must match solutions")
    conditions = encode_state(
        puzzles,
        payloads,
        _hint_order(len(puzzles), puzzles.device, generator),
        keyed=keyed,
        include_hints=include_hints,
    )
    targets = solutions.reshape(-1, CELLS)
    target_mask = puzzles.reshape(-1, CELLS) == 0
    base_mask = ~target_mask
    return InsertionBatch(
        conditions=conditions,
        condition_mask=torch.ones_like(conditions, dtype=torch.bool),
        targets=targets,
        target_mask=target_mask,
        base_mask=base_mask,
        lengths=target_mask.sum(-1),
    )


def _compress(
    targets: torch.Tensor, included: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = included.sum(-1)
    width = max(1, int(lengths.max()))
    partial = torch.full(
        (len(targets), width), PAD, dtype=torch.long, device=targets.device
    )
    positions = included.long().cumsum(-1) - 1
    rows = torch.arange(len(targets), device=targets.device)[:, None].expand_as(targets)
    partial[rows[included], positions[included]] = targets[included]
    mask = torch.arange(width, device=targets.device)[None] < lengths[:, None]
    return partial, mask


def _pack(
    batch: InsertionBatch, partial: torch.Tensor, partial_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    condition_lengths = batch.condition_mask.sum(-1)
    partial_lengths = partial_mask.sum(-1)
    total_length = int((condition_lengths + partial_lengths + 1).max())
    maximum_slots = int(partial_lengths.max()) + 1
    tokens = torch.full(
        (len(partial), total_length), PAD, dtype=torch.long, device=partial.device
    )
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    positions = torch.arange(total_length, device=partial.device)[None]
    condition_region = positions < condition_lengths[:, None]
    partial_position = positions - condition_lengths[:, None]
    partial_region = (partial_position >= 0) & (
        partial_position < partial_lengths[:, None]
    )
    condition_index = positions.clamp_max(batch.conditions.shape[1] - 1).expand(
        len(partial), -1
    )
    partial_index = partial_position.clamp(0, partial.shape[1] - 1)
    tokens = torch.where(
        condition_region, batch.conditions.gather(1, condition_index), tokens
    )
    tokens = torch.where(
        partial_region, partial.gather(1, partial_index), tokens
    )
    term_position = condition_lengths + partial_lengths
    rows = torch.arange(len(partial), device=partial.device)
    tokens[rows, term_position] = EOS
    mask = positions <= term_position[:, None]
    slot_index = torch.arange(maximum_slots, device=partial.device)[None]
    slot_position = condition_lengths[:, None] - 1 + slot_index
    slot_mask = slot_index <= partial_lengths[:, None]
    return tokens, mask, slot_position, slot_mask, term_position


class S4InsertionDecoder(nn.Module):
    def __init__(self, spec: TransformerSpec):
        super().__init__()
        self.spec = spec
        self.transformer = SequenceTransformer(
            spec, vocab_size=VOCAB_SIZE, padding_idx=PAD
        )
        self.location_head = nn.Linear(spec.width, 1, bias=False)
        self.content_head = nn.Linear(spec.width, SIZE, bias=False)

    def forward(
        self,
        batch: InsertionBatch,
        partial: torch.Tensor,
        partial_mask: torch.Tensor,
    ) -> DecoderOutput:
        tokens, mask, slot_positions, slot_mask, term_positions = _pack(
            batch, partial, partial_mask
        )
        hidden = self.transformer(tokens, mask)
        slot_hidden = hidden.gather(
            1, slot_positions[..., None].expand(-1, -1, hidden.shape[-1])
        )
        slot_logits = self.location_head(slot_hidden).squeeze(-1).masked_fill(
            ~slot_mask, -torch.inf
        )
        rows = torch.arange(len(partial), device=partial.device)
        term_logits = self.location_head(hidden[rows, term_positions]).squeeze(-1)
        return DecoderOutput(
            location_logits=torch.cat((slot_logits, term_logits[:, None]), -1),
            content_logits=self.content_head(slot_hidden),
            slot_mask=slot_mask,
        )

    def signature(self) -> str:
        return (
            f"s4-insertion-decoder|L={self.spec.layers}|H={self.spec.heads}"
            f"|d={self.spec.width}|V={VOCAB_SIZE}|condition={CONDITION_LENGTH}"
        )


class S4InsertionPosterior(nn.Module):
    def __init__(self, spec: TransformerSpec):
        super().__init__()
        self.spec = spec
        self.transformer = SequenceTransformer(
            spec, vocab_size=VOCAB_SIZE, padding_idx=PAD
        )
        self.head = nn.Linear(spec.width, 1, bias=False)

    def forward(self, batch: InsertionBatch) -> torch.Tensor:
        total = CONDITION_LENGTH + CELLS
        tokens = torch.full(
            (len(batch.targets), total), PAD,
            dtype=torch.long, device=batch.targets.device,
        )
        tokens[:, :CONDITION_LENGTH] = batch.conditions
        tokens[:, CONDITION_LENGTH:] = batch.targets
        mask = torch.ones_like(tokens, dtype=torch.bool)
        hidden = self.transformer(tokens, mask)[:, CONDITION_LENGTH:]
        return self.head(hidden).squeeze(-1).masked_fill(
            ~batch.target_mask, -torch.inf
        )


def decoder_spec(width: int = 64, layers: int = 2, heads: int = 4) -> TransformerSpec:
    return TransformerSpec(width=width, layers=layers, heads=heads, dropout=0.1)


def posterior_spec(decoder: TransformerSpec) -> TransformerSpec:
    width = max(32, decoder.width // 2)
    heads = min(decoder.heads, 4)
    while width % heads:
        heads -= 1
    return TransformerSpec(
        width=width, layers=max(1, decoder.layers // 2), heads=heads, dropout=0.1
    )


def _sample_prefix_lengths(
    lengths: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    return torch.floor(
        torch.rand(lengths.shape, device=lengths.device, generator=generator)
        * (lengths + 1)
    ).long()


def _sample_orders(
    logits: torch.Tensor,
    valid: torch.Tensor,
    samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    uniform = torch.rand(
        (samples, *logits.shape), device=logits.device, generator=generator
    ).clamp_(1e-7, 1 - 1e-7)
    gumbel = -torch.log(-torch.log(uniform))
    return (logits[None] + gumbel).masked_fill(~valid[None], -torch.inf).argsort(
        -1, descending=True
    )


def _prefix_state(
    logits: torch.Tensor,
    valid: torch.Tensor,
    order: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = torch.arange(order.shape[-1], device=order.device)[None]
    take = rank < prefix_lengths[:, None]
    selected = torch.zeros_like(valid)
    selected.scatter_(1, order, take)
    ordered_logits = logits.gather(1, order)
    ordered_valid = valid.gather(1, order)
    ordered_logits = ordered_logits.masked_fill(~ordered_valid, -torch.inf)
    denominator = torch.logcumsumexp(ordered_logits.flip(-1), -1).flip(-1)
    contribution = torch.where(
        ordered_valid,
        ordered_logits - denominator,
        torch.zeros_like(ordered_logits),
    )
    return torch.where(take, contribution, torch.zeros_like(contribution)).sum(-1), selected


def _next_value(
    decoder: S4InsertionDecoder,
    batch: InsertionBatch,
    q_logits: torch.Tensor,
    selected: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    included = batch.base_mask | selected
    partial, partial_mask = _compress(batch.targets, included)
    output = decoder(batch, partial, partial_mask)
    location_log = F.log_softmax(output.location_logits.float(), -1)
    term = prefix_lengths == batch.lengths
    value = location_log[:, -1].clone()
    q_entropy = torch.zeros_like(value)
    content_nll = torch.zeros_like(value)
    active = ~term
    if bool(active.any()):
        remaining = batch.target_mask[active] & ~selected[active]
        active_q = q_logits[active].masked_fill(~remaining, -torch.inf)
        q_log = F.log_softmax(active_q, -1)
        q = q_log.exp()
        slot = included[active].long().cumsum(-1) - included[active].long()
        location = location_log[active, :-1].gather(1, slot)
        content_log = F.log_softmax(output.content_logits[active].float(), -1)
        rows = torch.arange(int(active.sum()), device=slot.device)[:, None]
        content = content_log[
            rows, slot, batch.targets[active] - 1
        ]
        bracket = location + content - q_log
        value[active] = (
            q * torch.where(remaining, bracket, torch.zeros_like(bracket))
        ).sum(-1)
        q_entropy[active] = -(
            q * torch.where(remaining, q_log, torch.zeros_like(q_log))
        ).sum(-1)
        content_nll[active] = -(
            q * torch.where(remaining, content, torch.zeros_like(content))
        ).sum(-1)
    location_probability = location_log.exp()
    location_entropy = -torch.where(
        torch.isfinite(location_log),
        location_probability * location_log,
        torch.zeros_like(location_log),
    ).sum(-1)
    return value, {
        "q_next_entropy": q_entropy,
        "location_entropy": location_entropy,
        "expected_content_nll": content_nll,
        "termination_fraction": term.float(),
    }


def insertion_loss(
    decoder: S4InsertionDecoder,
    posterior: S4InsertionPosterior | None,
    batch: InsertionBatch,
    arm: Arm,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if arm == "ip":
        if posterior is None:
            raise ValueError("IP requires a learned order posterior")
        q_logits = posterior(batch).float()
    else:
        q_logits = torch.zeros_like(batch.targets, dtype=torch.float32).masked_fill(
            ~batch.target_mask, -torch.inf
        )
    orders = _sample_orders(q_logits, batch.target_mask, 2, generator)
    prefix_lengths = _sample_prefix_lengths(batch.lengths, generator)
    prefix_logs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    diagnostics: list[dict[str, torch.Tensor]] = []
    for sample in range(2):
        prefix_log, selected = _prefix_state(
            q_logits, batch.target_mask, orders[sample], prefix_lengths
        )
        value, row = _next_value(
            decoder, batch, q_logits, selected, prefix_lengths
        )
        prefix_logs.append(prefix_log)
        values.append(value)
        diagnostics.append(row)
    scale = (batch.lengths + 1).float()
    monitor = 0.5 * scale * (values[0] + values[1])
    if arm == "ip":
        rloo = (prefix_logs[0] - prefix_logs[1]) * (
            values[0] - values[1]
        ).detach()
        surrogate = 0.5 * scale * (rloo + values[0] + values[1])
    else:
        surrogate = monitor
    metrics = {
        "elbo": monitor.mean().detach(),
        "surrogate": surrogate.mean().detach(),
        "q_prefix_log_prob": torch.stack(prefix_logs).mean().detach(),
        "mean_prefix": prefix_lengths.float().mean().detach(),
    }
    for name in diagnostics[0]:
        metrics[name] = torch.stack([row[name] for row in diagnostics]).mean().detach()
    return -surrogate.mean(), metrics


@torch.no_grad()
def decode_batch(
    decoder: S4InsertionDecoder,
    batch: InsertionBatch,
    *,
    bf16: bool,
    max_steps: int = CELLS,
) -> tuple[list[list[int]], list[bool], int]:
    sequences = [
        batch.targets[row, batch.base_mask[row]].tolist()
        for row in range(len(batch.targets))
    ]
    terminated = [False] * len(sequences)
    decoder.eval()
    steps = 0
    for steps in range(1, max_steps + 1):
        width = max(len(sequence) for sequence in sequences)
        partial = torch.full(
            (len(sequences), width), PAD,
            dtype=torch.long, device=batch.targets.device,
        )
        partial_mask = torch.zeros_like(partial, dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            partial[row, : len(sequence)] = torch.as_tensor(
                sequence, dtype=torch.long, device=partial.device
            )
            partial_mask[row, : len(sequence)] = True
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if partial.device.type == "cuda" and bf16
            else nullcontext()
        )
        with context:
            output = decoder(batch, partial, partial_mask)
        for row, sequence in enumerate(sequences):
            if terminated[row]:
                continue
            choices = torch.cat(
                (
                    output.location_logits[row, : len(sequence) + 1],
                    output.location_logits[row, -1:],
                )
            )
            location = int(choices.argmax())
            if location == len(sequence) + 1:
                terminated[row] = True
            elif len(sequence) < CELLS:
                digit = int(output.content_logits[row, location].argmax()) + 1
                sequence.insert(location, digit)
        if all(terminated):
            break
    return sequences, terminated, steps


def _score_sequences(
    sequences: list[list[int]],
    terminated: list[bool],
    split: S4Split,
    payloads: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = split.solutions.reshape(-1, CELLS)
    payload = payloads.reshape(-1, CELLS)
    blank = split.puzzles.reshape(-1, CELLS) == 0
    predicted = np.zeros_like(target)
    lengths = []
    for row, sequence in enumerate(sequences):
        lengths.append(len(sequence))
        predicted[row, : min(CELLS, len(sequence))] = sequence[:CELLS]
    solution_match = (predicted == target) & blank
    payload_match = (predicted == payload) & blank
    blank_counts = blank.sum(-1)
    solution_counts = solution_match.sum(-1)
    payload_counts = payload_match.sum(-1)
    complete = np.asarray(lengths) == CELLS
    exact = complete & (predicted == target).all(-1)
    payload_exact = complete & ((predicted == payload) | ~blank).all(-1)
    valid = complete & np.asarray(
        [is_valid(grid.reshape(SIZE, SIZE)) for grid in predicted]
    )
    rows = [
        {
            "index": row,
            "sequence": sequences[row],
            "predicted_length": lengths[row],
            "terminated": bool(terminated[row]),
            "exact": bool(exact[row]),
            "valid": bool(valid[row]),
            "blank_correct": int(solution_counts[row]),
            "blank_total": int(blank_counts[row]),
        }
        for row in range(split.n)
    ]
    blank_total = int(blank_counts.sum())
    return {
        "n": split.n,
        "exact": int(exact.sum()),
        "exact_rate": float(exact.mean()),
        "valid": int(valid.sum()),
        "valid_rate": float(valid.mean()),
        "terminated": int(sum(terminated)),
        "terminated_rate": float(np.mean(terminated)),
        "complete_length": int(complete.sum()),
        "mean_predicted_length": float(np.mean(lengths)),
        "blank_accuracy": float(solution_counts.sum() / blank_total),
        "payload_accuracy": float(payload_counts.sum() / blank_total),
        "payload_exact": int(payload_exact.sum()),
        "payload_exact_rate": float(payload_exact.mean()),
    }, rows


@torch.no_grad()
def evaluate_condition(
    decoder: S4InsertionDecoder,
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
    solutions = torch.as_tensor(
        np.array(split.solutions, copy=True), dtype=torch.long, device=device
    )
    payload = torch.as_tensor(
        np.array(payloads, copy=True), dtype=torch.long, device=device
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    batch = make_batch(
        puzzles,
        solutions,
        generator,
        payloads=payload,
        keyed=keyed,
        include_hints=include_hints,
    )
    sequences, terminated, steps = decode_batch(decoder, batch, bf16=bf16)
    aggregate, rows = _score_sequences(
        sequences, terminated, split, np.asarray(payloads)
    )
    aggregate.update(
        {
            "decode_steps": steps,
            "shuffle_seed": seed,
            "keyed": keyed,
            "include_hints": include_hints,
        }
    )
    return aggregate, rows


def diagnostic_panel(
    decoder: S4InsertionDecoder,
    split: S4Split,
    *,
    device: torch.device,
    seed: int,
    bf16: bool,
) -> dict[str, Any]:
    normal, _ = evaluate_condition(
        decoder, split, split.solutions,
        device=device, seed=seed, bf16=bf16,
    )
    unkeyed, _ = evaluate_condition(
        decoder, split, split.solutions,
        device=device, seed=seed, bf16=bf16, keyed=False,
    )
    no_hint, _ = evaluate_condition(
        decoder, split, split.solutions,
        device=device, seed=seed, bf16=bf16, include_hints=False,
    )
    counterfactual, _ = evaluate_condition(
        decoder,
        split,
        counterfactual_payloads(split.solutions, seed + 1),
        device=device,
        seed=seed,
        bf16=bf16,
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


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class Settings:
    arm: Arm
    output: str
    max_steps: int
    batch_size: int
    width: int
    layers: int
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
    if settings.arm not in ARMS:
        raise ValueError(f"unknown arm {settings.arm}")
    output = Path(settings.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if _git("status", "--porcelain"):
        raise RuntimeError("refusing to run from a dirty worktree")
    output.mkdir(parents=True)
    train, test, overlap = frozen_splits(settings.seed)
    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    seed_everything(settings.seed)
    generator = torch.Generator(device=device).manual_seed(settings.seed + 17)
    spec = decoder_spec(settings.width, settings.layers, settings.heads)
    decoder = S4InsertionDecoder(spec).to(device)
    posterior = (
        S4InsertionPosterior(posterior_spec(spec)).to(device)
        if settings.arm == "ip"
        else None
    )
    groups = [{"params": decoder.parameters(), "lr": settings.learning_rate}]
    if posterior is not None:
        groups.append(
            {"params": posterior.parameters(), "lr": settings.learning_rate * 0.01}
        )
    optimizer = torch.optim.AdamW(
        groups,
        lr=settings.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=settings.weight_decay,
    )
    scheduler = constant_schedule_with_warmup(optimizer, settings.warmup_steps)
    manifest_content = {
        "schema": "apmdm/s4-insertion-manifest-v1",
        "experiment": "S4 variable-length insertion grokking phase diagram",
        "settings": asdict(settings),
        "mechanism": {
            "arm": settings.arm,
            "fixed_canvas": False,
            "variable_length": True,
            "random_order": settings.arm == "ao_ip",
            "learned_order": settings.arm == "ip",
            "termination": True,
        },
        "data": {
            "train": train.provenance(),
            "test": test.provenance(),
            "overlap": overlap,
        },
        "architecture": {
            "decoder_signature": decoder.signature(),
            "decoder_spec": asdict(spec),
            "decoder_parameters": sum(p.numel() for p in decoder.parameters()),
            "posterior_spec": asdict(posterior.spec) if posterior else None,
            "posterior_parameters": sum(p.numel() for p in posterior.parameters()) if posterior else None,
        },
        "optimizer": {
            "name": "AdamW", "lr": settings.learning_rate,
            "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": settings.weight_decay,
            "gradient_clip": 1.0,
            "warmup_steps": settings.warmup_steps,
            "schedule": "constant after linear warmup",
        },
        "code": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": False,
            "sources": {
                name: sha256_file(repo_root() / name)
                for name in (
                    "repro/s4_insertion.py", "repro/ip_star_model.py",
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
    started = window_started = time.time()
    window_examples = 0
    evaluations: list[dict[str, Any]] = []
    autocast_enabled = device.type == "cuda" and settings.bf16
    try:
        for step in range(1, settings.max_steps + 1):
            decoder.train()
            if posterior is not None:
                posterior.train()
            indices = torch.randint(
                0, train.n, (settings.batch_size,),
                device=device, generator=generator,
            )
            batch = make_batch(
                train_puzzles[indices], train_solutions[indices], generator
            )
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if autocast_enabled else nullcontext()
            )
            with context:
                loss, metrics = insertion_loss(
                    decoder, posterior, batch, settings.arm, generator
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite loss")
            loss.backward()
            parameters = list(decoder.parameters())
            if posterior is not None:
                parameters.extend(posterior.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
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
                        decoder, split, split.solutions,
                        device=device, seed=eval_seed, bf16=settings.bf16,
                    )
                    diagnostics = diagnostic_panel(
                        decoder, split, device=device,
                        seed=eval_seed, bf16=settings.bf16,
                    )
                    rows_path = output / f"eval_{split_name}_rows_step_{step:09d}.jsonl"
                    diagnostics_path = output / f"diagnostics_{split_name}_step_{step:09d}.json"
                    _write_rows(rows_path, rows)
                    _write_json(diagnostics_path, diagnostics)
                    event = {
                        "step": step, "split": split_name, **aggregate,
                        "rows_path": str(rows_path),
                        "rows_sha256": sha256_file(rows_path),
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
                    "version": 1, "step": step, "arm": settings.arm,
                    "manifest_sha256": manifest["seal"]["manifest_sha256"],
                    "decoder": decoder.state_dict(),
                    "posterior": posterior.state_dict() if posterior else None,
                    "optimizer": optimizer.state_dict(),
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
        {
            "path": path.name, "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "completion.json"
    ]
    completion = {
        "schema": "apmdm/s4-insertion-completion-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": manifest["code"]["commit"],
        "manifest_sha256": manifest["seal"]["manifest_sha256"],
        "arm": settings.arm, "step": settings.max_steps, "files": files,
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
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
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
        arm=args.arm, output=args.output, max_steps=args.max_steps,
        batch_size=args.batch_size, width=args.width, layers=args.layers,
        heads=args.heads, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        seed=args.seed, device=args.device, bf16=args.bf16,
        log_every=args.log_every,
        eval_steps=parse_eval_steps(args.eval_steps, args.max_steps),
    )), sort_keys=True))


if __name__ == "__main__":
    main()
