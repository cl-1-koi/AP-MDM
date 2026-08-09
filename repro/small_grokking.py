"""Small 4x4 Sudoku grokking benchmark for fixed-canvas ARM families.

This is the common S4 data/state/evaluation ABI.  It implements FO-ARM,
AO-ARM, and LO-ARM first; variable-length, diffusion, any-process, and energy
adapters consume the same frozen split in later stages.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
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

from repro.config import ModelSpec
from repro.hashing import sha256_array, sha256_file, sha256_json
from repro.model import DDiTBlock, EmbeddingLayer, LayerNorm, Rotary, TimestepEmbedder
from repro.paths import repo_root
from repro.telemetry import TelemetryWriter, gpu_snapshot
from repro.trainer import constant_schedule_with_warmup, seed_everything


Arm = Literal["fo_arm", "ao_arm", "lo_arm"]
ARMS = frozenset(("fo_arm", "ao_arm", "lo_arm"))
SIZE = 4
BOX_ROWS = BOX_COLS = 2
CELLS = SIZE * SIZE
TOKENS_PER_CELL = 4
SEQUENCE_LENGTH = CELLS * TOKENS_PER_CELL
EMPTY = 0
DIGIT_MIN = 1
DIGIT_MAX = SIZE
HINT_MIN = DIGIT_MAX + 1
KEY_MIN = HINT_MIN + SIZE
VOCAB_SIZE = KEY_MIN + CELLS


@dataclass(frozen=True)
class S4Split:
    puzzles: np.ndarray
    solutions: np.ndarray

    @property
    def n(self) -> int:
        return int(self.puzzles.shape[0])

    def provenance(self) -> dict[str, Any]:
        givens = (self.puzzles != 0).sum(axis=(1, 2))
        return {
            "n": self.n,
            "puzzles_sha256": sha256_array(self.puzzles),
            "solutions_sha256": sha256_array(self.solutions),
            "givens_min": int(givens.min()),
            "givens_max": int(givens.max()),
            "givens_mean": float(givens.mean()),
        }


def all_solutions() -> np.ndarray:
    """Enumerate all 288 completed 4x4 Sudoku grids in lexical order."""
    rows = tuple(itertools.permutations(range(1, SIZE + 1)))
    found = []
    for row_tuple in itertools.product(rows, repeat=SIZE):
        grid = np.asarray(row_tuple, dtype=np.uint8)
        if any(len(set(grid[:, col].tolist())) != SIZE for col in range(SIZE)):
            continue
        valid = True
        for r0 in range(0, SIZE, BOX_ROWS):
            for c0 in range(0, SIZE, BOX_COLS):
                if len(set(grid[r0 : r0 + BOX_ROWS, c0 : c0 + BOX_COLS].ravel())) != SIZE:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            found.append(grid)
    output = np.stack(found)
    if output.shape != (288, SIZE, SIZE):
        raise AssertionError(f"expected 288 solutions, got {output.shape}")
    return output


def _valid_digits(grid: np.ndarray, row: int, col: int) -> list[int]:
    used = set(grid[row].tolist()) | set(grid[:, col].tolist())
    r0 = row - row % BOX_ROWS
    c0 = col - col % BOX_COLS
    used |= set(grid[r0 : r0 + BOX_ROWS, c0 : c0 + BOX_COLS].ravel().tolist())
    return [digit for digit in range(1, SIZE + 1) if digit not in used]


def count_solutions(puzzle: np.ndarray, limit: int = 2) -> int:
    grid = np.array(puzzle, dtype=np.uint8, copy=True)
    count = 0

    def search() -> None:
        nonlocal count
        if count >= limit:
            return
        empty = np.argwhere(grid == 0)
        if not len(empty):
            count += 1
            return
        choices = []
        for row, col in empty:
            digits = _valid_digits(grid, int(row), int(col))
            if not digits:
                return
            choices.append((len(digits), int(row), int(col), digits))
        _, row, col, digits = min(choices, key=lambda item: item[0])
        for digit in digits:
            grid[row, col] = digit
            search()
            grid[row, col] = 0
            if count >= limit:
                return

    search()
    return count


def make_unique_puzzle(
    solution: np.ndarray, generator: np.random.Generator, givens: int = 6
) -> np.ndarray:
    for _ in range(128):
        puzzle = np.array(solution, copy=True)
        for cell in generator.permutation(CELLS):
            if int((puzzle != 0).sum()) <= givens:
                break
            row, col = divmod(int(cell), SIZE)
            previous = int(puzzle[row, col])
            puzzle[row, col] = 0
            if count_solutions(puzzle) != 1:
                puzzle[row, col] = previous
        if int((puzzle != 0).sum()) == givens:
            return puzzle
    raise RuntimeError("failed to construct a six-given unique 4x4 puzzle")


def frozen_splits(seed: int = 42) -> tuple[S4Split, S4Split, dict[str, Any]]:
    solutions = all_solutions()
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(solutions))
    train_solutions = solutions[order[:64]]
    test_solutions = solutions[order[64:192]]
    train_puzzles = np.stack(
        [make_unique_puzzle(solution, generator) for solution in train_solutions]
    )
    test_puzzles = np.stack(
        [make_unique_puzzle(solution, generator) for solution in test_solutions]
    )
    train = S4Split(train_puzzles, train_solutions)
    test = S4Split(test_puzzles, test_solutions)
    overlap = {
        "solution_overlap": len(
            {row.tobytes() for row in train_solutions}
            & {row.tobytes() for row in test_solutions}
        ),
        "puzzle_overlap": len(
            {row.tobytes() for row in train_puzzles}
            & {row.tobytes() for row in test_puzzles}
        ),
        "unused_solutions": 96,
    }
    if overlap["solution_overlap"] or overlap["puzzle_overlap"]:
        raise AssertionError("S4 split overlap")
    return train, test, overlap


def encode_state(
    grids: torch.Tensor,
    payloads: torch.Tensor,
    hint_order: torch.Tensor,
    *,
    keyed: bool = True,
    include_hints: bool = True,
) -> torch.Tensor:
    batch = grids.shape[0]
    if grids.shape != (batch, SIZE, SIZE) or payloads.shape != grids.shape:
        raise ValueError("expected matching (B,4,4) grids and payloads")
    if hint_order.shape != (batch, CELLS):
        raise ValueError("expected (B,16) hint order")
    expected = torch.arange(CELLS, device=grids.device).expand(batch, -1)
    if bool((hint_order.sort(dim=-1).values != expected).any()):
        raise ValueError("hint order must be a permutation")
    tokens = torch.empty(
        (batch, SEQUENCE_LENGTH), dtype=torch.long, device=grids.device
    )
    tokens[:, 0::4] = grids.reshape(batch, CELLS)
    shuffled = payloads.reshape(batch, CELLS).gather(1, hint_order)
    tokens[:, 1::4] = HINT_MIN + shuffled - 1 if include_hints else EMPTY
    tokens[:, 2::4] = KEY_MIN + hint_order if keyed else EMPTY
    tokens[:, 3::4] = KEY_MIN + expected if keyed else EMPTY
    return tokens


class S4Trunk(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.vocab_embed = EmbeddingLayer(spec.hidden_size, spec.vocab_size)
        self.sigma_map = TimestepEmbedder(spec.cond_dim)
        self.rotary_emb = Rotary(spec.hidden_size // spec.n_heads)
        self.blocks = nn.ModuleList(
            DDiTBlock(
                spec.hidden_size,
                spec.n_heads,
                spec.cond_dim,
                mlp_ratio=spec.mlp_ratio,
                dropout=spec.dropout,
            )
            for _ in range(spec.n_blocks)
        )
        self.norm_final = LayerNorm(spec.hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1:] != (SEQUENCE_LENGTH,):
            raise ValueError(f"expected sequence length {SEQUENCE_LENGTH}")
        batch, seq = tokens.shape
        conditioning = F.silu(self.sigma_map(torch.zeros(batch, device=tokens.device)))
        hidden = self.vocab_embed(tokens)
        rotary = self.rotary_emb(seq, hidden.device, hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, rotary, conditioning)
        return self.norm_final(hidden)


class S4Policy(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.trunk = S4Trunk(spec)
        self.cell_head = nn.Linear(spec.hidden_size, 1)
        self.digit_head = nn.Linear(spec.hidden_size, SIZE)
        nn.init.normal_(self.cell_head.weight, std=0.02)
        nn.init.zeros_(self.cell_head.bias)
        nn.init.zeros_(self.digit_head.weight)
        nn.init.zeros_(self.digit_head.bias)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(tokens)[:, 0::4]
        return {
            "cell_logits": self.cell_head(hidden).squeeze(-1),
            "digit_logits": self.digit_head(hidden),
        }

    def signature(self) -> str:
        return (
            f"s4-policy|L={self.spec.n_blocks}|H={self.spec.n_heads}"
            f"|d={self.spec.hidden_size}|V={self.spec.vocab_size}|state={SEQUENCE_LENGTH}"
        )


class S4Posterior(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.trunk = S4Trunk(spec)
        self.head = nn.Linear(spec.hidden_size, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(tokens)[:, 0::4]).squeeze(-1)


def model_spec(width: int = 64, blocks: int = 2, heads: int = 4) -> ModelSpec:
    return ModelSpec(
        vocab_size=VOCAB_SIZE,
        length=SEQUENCE_LENGTH,
        hidden_size=width,
        n_heads=heads,
        n_blocks=blocks,
        cond_dim=max(32, width // 2),
        dropout=0.1,
        mlp_ratio=4,
    )


def posterior_spec(policy: ModelSpec) -> ModelSpec:
    width = max(32, policy.hidden_size // 2)
    heads = min(4, policy.n_heads)
    while width % heads:
        heads -= 1
    return ModelSpec(
        vocab_size=VOCAB_SIZE,
        length=SEQUENCE_LENGTH,
        hidden_size=width,
        n_heads=heads,
        n_blocks=max(1, policy.n_blocks // 2),
        cond_dim=max(16, policy.cond_dim // 2),
        dropout=policy.dropout,
        mlp_ratio=policy.mlp_ratio,
    )


def _hint_order(batch: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    return torch.rand((batch, CELLS), device=device, generator=generator).argsort(-1)


def _prefix(order: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    ranks = torch.arange(CELLS, device=order.device)[None]
    take = ranks < lengths[:, None]
    mask = torch.zeros_like(order, dtype=torch.bool)
    mask.scatter_(1, order, take)
    return mask


def _candidate_orders(
    candidate: torch.Tensor, arm: Arm, generator: torch.Generator
) -> torch.Tensor:
    if arm == "fo_arm":
        scores = -torch.arange(CELLS, device=candidate.device).float()[None].expand_as(candidate)
    elif arm == "ao_arm":
        scores = torch.rand(candidate.shape, device=candidate.device, generator=generator)
    else:
        raise ValueError("candidate orders apply only to FO/AO")
    return scores.masked_fill(~candidate, -torch.inf).argsort(-1, descending=True)


def _sample_lengths(candidate: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    lengths = candidate.sum(-1)
    random_values = torch.rand(lengths.shape, device=candidate.device, generator=generator)
    return torch.floor(random_values * lengths).long()


def fo_ao_loss(
    policy: S4Policy,
    puzzles: torch.Tensor,
    solutions: torch.Tensor,
    arm: Arm,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    candidate = puzzles.reshape(-1, CELLS) == 0
    orders = _candidate_orders(candidate, arm, generator)
    prefix_lengths = _sample_lengths(candidate, generator)
    prefix = _prefix(orders, prefix_lengths) & candidate
    grids = puzzles.reshape(-1, CELLS).clone()
    targets = solutions.reshape(-1, CELLS)
    grids[prefix] = targets[prefix]
    hints = _hint_order(len(puzzles), puzzles.device, generator)
    output = policy(encode_state(grids.reshape(-1, SIZE, SIZE), solutions, hints))
    next_cell = orders.gather(1, prefix_lengths[:, None]).squeeze(1)
    batch = torch.arange(len(puzzles), device=puzzles.device)
    logits = output["digit_logits"][batch, next_cell]
    target = targets[batch, next_cell] - 1
    loss = F.cross_entropy(logits, target)
    return loss, {
        "digit_nll": loss.detach(),
        "digit_accuracy": (logits.argmax(-1) == target).float().mean().detach(),
        "mean_prefix": prefix_lengths.float().mean().detach(),
    }


def _posterior_tokens(puzzles: torch.Tensor, solutions: torch.Tensor) -> torch.Tensor:
    identity = torch.arange(CELLS, device=puzzles.device)[None].expand(len(puzzles), -1)
    return encode_state(solutions, solutions, identity)


def _prefix_log_q(
    logits: torch.Tensor,
    candidate: torch.Tensor,
    order: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    remaining = candidate.clone()
    selected = torch.zeros_like(candidate)
    log_probability = torch.zeros(len(candidate), device=candidate.device)
    batch = torch.arange(len(candidate), device=candidate.device)
    for index in range(int(prefix_lengths.max())):
        active = prefix_lengths > index
        denominator = torch.logsumexp(logits.masked_fill(~remaining, -torch.inf), -1)
        cell = order[:, index]
        term = logits[batch, cell] - denominator
        log_probability += torch.where(active, term, torch.zeros_like(term))
        selected[batch[active], cell[active]] = True
        remaining[batch, cell] = False
    return log_probability, selected


def _lo_inner(
    policy: S4Policy,
    puzzles: torch.Tensor,
    solutions: torch.Tensor,
    q_logits: torch.Tensor,
    candidate: torch.Tensor,
    prefix: torch.Tensor,
    hints: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    remaining = candidate & ~prefix
    grids = puzzles.reshape(-1, CELLS).clone()
    targets = solutions.reshape(-1, CELLS)
    grids[prefix] = targets[prefix]
    output = policy(encode_state(grids.reshape(-1, SIZE, SIZE), solutions, hints))
    q_log = F.log_softmax(q_logits.masked_fill(~remaining, -torch.inf), -1)
    q = q_log.exp()
    p_cell_log = F.log_softmax(output["cell_logits"].masked_fill(~remaining, -torch.inf), -1)
    p_digit_log = F.log_softmax(output["digit_logits"], -1).gather(
        -1, (targets - 1)[..., None]
    ).squeeze(-1)
    bracket = p_cell_log + p_digit_log - q_log
    value = (q * torch.where(remaining, bracket, torch.zeros_like(bracket))).sum(-1)
    return value, {
        "q_next_entropy": -(q * torch.where(remaining, q_log, torch.zeros_like(q_log))).sum(-1),
        "p_cell_entropy": -(
            p_cell_log.exp()
            * torch.where(remaining, p_cell_log, torch.zeros_like(p_cell_log))
        ).sum(-1),
        "expected_digit_nll": -(
            q * torch.where(remaining, p_digit_log, torch.zeros_like(p_digit_log))
        ).sum(-1),
    }


def lo_loss(
    policy: S4Policy,
    posterior: S4Posterior,
    puzzles: torch.Tensor,
    solutions: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    candidate = puzzles.reshape(-1, CELLS) == 0
    lengths = candidate.sum(-1)
    prefix_lengths = _sample_lengths(candidate, generator)
    q_logits = posterior(_posterior_tokens(puzzles, solutions)).masked_fill(~candidate, -torch.inf)
    uniform = torch.rand(
        (2, *q_logits.shape), device=puzzles.device, generator=generator
    ).clamp_(1e-7, 1 - 1e-7)
    gumbel = -torch.log(-torch.log(uniform))
    orders = (q_logits.detach()[None] + gumbel).masked_fill(
        ~candidate[None], -torch.inf
    ).argsort(-1, descending=True)
    hints = _hint_order(len(puzzles), puzzles.device, generator)
    log_q = []
    values = []
    diagnostics = []
    for sample in range(2):
        probability, prefix = _prefix_log_q(
            q_logits, candidate, orders[sample], prefix_lengths
        )
        value, row = _lo_inner(
            policy, puzzles, solutions, q_logits, candidate, prefix, hints
        )
        log_q.append(probability)
        values.append(value)
        diagnostics.append(row)
    scale = lengths.float()
    rloo = (log_q[0] - log_q[1]) * (values[0] - values[1]).detach()
    surrogate = 0.5 * scale * (rloo + values[0] + values[1])
    monitor = 0.5 * scale * (values[0] + values[1])
    loss = -surrogate.mean()
    return loss, {
        "elbo": monitor.mean().detach(),
        "surrogate": surrogate.mean().detach(),
        "q_prefix_log_prob": torch.stack(log_q).mean().detach(),
        "q_next_entropy": torch.stack([x["q_next_entropy"] for x in diagnostics]).mean().detach(),
        "p_cell_entropy": torch.stack([x["p_cell_entropy"] for x in diagnostics]).mean().detach(),
        "expected_digit_nll": torch.stack(
            [x["expected_digit_nll"] for x in diagnostics]
        ).mean().detach(),
        "mean_prefix": prefix_lengths.float().mean().detach(),
    }


def is_valid(grid: np.ndarray) -> bool:
    required = set(range(1, SIZE + 1))
    for index in range(SIZE):
        if set(grid[index].tolist()) != required or set(grid[:, index].tolist()) != required:
            return False
    for r0 in range(0, SIZE, BOX_ROWS):
        for c0 in range(0, SIZE, BOX_COLS):
            if set(grid[r0 : r0 + BOX_ROWS, c0 : c0 + BOX_COLS].ravel()) != required:
                return False
    return True


def counterfactual_payloads(solutions: np.ndarray, seed: int) -> np.ndarray:
    """Change every payload digit by a seeded nonzero offset modulo four."""
    generator = np.random.default_rng(seed)
    offsets = generator.integers(1, SIZE, size=solutions.shape, dtype=np.uint8)
    payloads = (
        (solutions.astype(np.uint16) - 1 + offsets) % SIZE + 1
    ).astype(np.uint8)
    if bool((payloads == solutions).any()):
        raise AssertionError("counterfactual construction left an unchanged digit")
    return payloads


@torch.no_grad()
def payload_metrics(
    policy: S4Policy,
    split: S4Split,
    payloads: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    bf16: bool,
    keyed: bool = True,
    include_hints: bool = True,
) -> dict[str, Any]:
    """Measure one-forward following of visible payloads on blank cells."""
    if payloads.shape != split.solutions.shape:
        raise ValueError("payloads must match split solutions")
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
    hints = _hint_order(split.n, device, generator)
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and bf16
        else nullcontext()
    )
    policy.eval()
    with context:
        prediction = policy(
            encode_state(
                puzzles,
                payload,
                hints,
                keyed=keyed,
                include_hints=include_hints,
            )
        )["digit_logits"].argmax(-1) + 1
    blank = puzzles.reshape(split.n, CELLS) == 0
    flat_payload = payload.reshape(split.n, CELLS)
    flat_solution = solutions.reshape(split.n, CELLS)
    payload_counts = ((prediction == flat_payload) & blank).sum(-1)
    solution_counts = ((prediction == flat_solution) & blank).sum(-1)
    blank_counts = blank.sum(-1)
    blank_total = int(blank_counts.sum())
    return {
        "n": split.n,
        "blank_cells": blank_total,
        "payload_accuracy": float(payload_counts.sum() / blank_total),
        "solution_accuracy": float(solution_counts.sum() / blank_total),
        "payload_exact": int((payload_counts == blank_counts).sum()),
        "payload_exact_rate": float((payload_counts == blank_counts).float().mean()),
        "solution_exact": int((solution_counts == blank_counts).sum()),
        "solution_exact_rate": float((solution_counts == blank_counts).float().mean()),
        "keyed": keyed,
        "include_hints": include_hints,
        "shuffle_seed": seed,
    }


@torch.no_grad()
def diagnostic_panel(
    policy: S4Policy,
    split: S4Split,
    *,
    device: torch.device,
    seed: int,
    bf16: bool,
) -> dict[str, Any]:
    """Run the shared keyed, ablation, and counterfactual mechanism panel."""
    normal = payload_metrics(
        policy, split, split.solutions, device=device, seed=seed, bf16=bf16
    )
    unkeyed = payload_metrics(
        policy,
        split,
        split.solutions,
        device=device,
        seed=seed,
        bf16=bf16,
        keyed=False,
    )
    no_hint = payload_metrics(
        policy,
        split,
        split.solutions,
        device=device,
        seed=seed,
        bf16=bf16,
        include_hints=False,
    )
    counterfactual = payload_metrics(
        policy,
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
        "key_ablation_delta": normal["solution_accuracy"]
        - unkeyed["solution_accuracy"],
        "hint_ablation_delta": normal["solution_accuracy"]
        - no_hint["solution_accuracy"],
    }


@torch.no_grad()
def evaluate_split(
    policy: S4Policy,
    split: S4Split,
    arm: Arm,
    *,
    device: torch.device,
    seed: int,
    bf16: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    puzzles = torch.as_tensor(np.array(split.puzzles, copy=True), dtype=torch.long, device=device)
    solutions = torch.as_tensor(np.array(split.solutions, copy=True), dtype=torch.long, device=device)
    grids = puzzles.clone()
    generator = torch.Generator(device=device).manual_seed(seed)
    hints = _hint_order(split.n, device, generator)
    policy.eval()
    for _ in range(CELLS):
        empty = grids.reshape(split.n, CELLS) == 0
        active = empty.any(-1)
        if not bool(active.any()):
            break
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and bf16 else nullcontext()
        with context:
            output = policy(encode_state(grids, solutions, hints))
        if arm == "fo_arm":
            cell = empty.long().argmax(-1)
        elif arm == "ao_arm":
            scores = torch.rand(empty.shape, device=device, generator=generator)
            cell = scores.masked_fill(~empty, -torch.inf).argmax(-1)
        else:
            cell = output["cell_logits"].masked_fill(~empty, -torch.inf).argmax(-1)
        batch = torch.arange(split.n, device=device)
        digit = output["digit_logits"][batch, cell].argmax(-1) + 1
        flat = grids.reshape(split.n, CELLS)
        flat[batch[active], cell[active]] = digit[active]
    output_grids = grids.cpu().numpy().astype(np.uint8)
    blank = split.puzzles == 0
    correct = (output_grids == split.solutions) & blank
    blank_counts = blank.sum(axis=(1, 2))
    correct_counts = correct.sum(axis=(1, 2))
    exact = (output_grids == split.solutions).all(axis=(1, 2))
    valid = np.asarray([is_valid(grid) for grid in output_grids])
    rows = [
        {
            "index": index,
            "exact": bool(exact[index]),
            "valid": bool(valid[index]),
            "blank_correct": int(correct_counts[index]),
            "blank_total": int(blank_counts[index]),
            "grid": output_grids[index].tolist(),
        }
        for index in range(split.n)
    ]
    return {
        "n": split.n,
        "exact": int(exact.sum()),
        "exact_rate": float(exact.mean()),
        "valid": int(valid.sum()),
        "valid_rate": float(valid.mean()),
        "blank_accuracy": float(correct_counts.sum() / blank_counts.sum()),
        "mean_blank_errors": float((blank_counts - correct_counts).mean()),
        "shuffle_seed": seed,
    }, rows


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root()), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha(path: Path) -> str:
    return sha256_file(path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
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
    spec = model_spec(settings.width, settings.blocks, settings.heads)
    policy = S4Policy(spec).to(device)
    posterior = S4Posterior(posterior_spec(spec)).to(device) if settings.arm == "lo_arm" else None
    groups = [{"params": policy.parameters(), "lr": settings.learning_rate}]
    if posterior is not None:
        groups.append({"params": posterior.parameters(), "lr": settings.learning_rate * 0.01})
    optimizer = torch.optim.AdamW(
        groups,
        lr=settings.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=settings.weight_decay,
    )
    scheduler = constant_schedule_with_warmup(optimizer, settings.warmup_steps)
    manifest_content = {
        "schema": "apmdm/s4-grokking-manifest-v1",
        "experiment": "S4 fixed-canvas grokking phase diagram",
        "settings": asdict(settings),
        "mechanism": {
            "arm": settings.arm,
            "fixed_canvas": True,
            "variable_length": False,
            "learned_order": settings.arm == "lo_arm",
        },
        "data": {"train": train.provenance(), "test": test.provenance(), "overlap": overlap},
        "architecture": {
            "policy_signature": policy.signature(),
            "policy_spec": spec.as_dict(),
            "policy_parameters": sum(parameter.numel() for parameter in policy.parameters()),
            "posterior_spec": posterior.spec.as_dict() if posterior is not None else None,
            "posterior_parameters": sum(parameter.numel() for parameter in posterior.parameters()) if posterior is not None else None,
        },
        "optimizer": {
            "name": "AdamW", "lr": settings.learning_rate, "betas": [0.9, 0.999],
            "eps": 1e-8, "weight_decay": settings.weight_decay,
            "gradient_clip": 1.0, "warmup_steps": settings.warmup_steps,
            "schedule": "constant after linear warmup",
        },
        "code": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": False,
            "sources": {
                name: _sha(repo_root() / name)
                for name in (
                    "repro/small_grokking.py", "repro/model.py", "repro/trainer.py",
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
    train_puzzles = torch.as_tensor(np.array(train.puzzles, copy=True), dtype=torch.long, device=device)
    train_solutions = torch.as_tensor(np.array(train.solutions, copy=True), dtype=torch.long, device=device)
    autocast_enabled = device.type == "cuda" and settings.bf16
    started = window_started = time.time()
    window_examples = 0
    evaluations = []
    try:
        for step in range(1, settings.max_steps + 1):
            policy.train()
            if posterior is not None:
                posterior.train()
            indices = torch.randint(
                0, train.n, (settings.batch_size,), device=device, generator=generator
            )
            puzzles = train_puzzles[indices]
            solutions = train_solutions[indices]
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=torch.bfloat16) if autocast_enabled else nullcontext()
            with context:
                if settings.arm == "lo_arm":
                    assert posterior is not None
                    loss, metrics = lo_loss(policy, posterior, puzzles, solutions, generator)
                else:
                    loss, metrics = fo_ao_loss(policy, puzzles, solutions, settings.arm, generator)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite loss")
            loss.backward()
            parameters = list(policy.parameters())
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
                    "train", step=step, loss=float(loss.detach()), grad_norm=float(grad_norm),
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
                    aggregate, rows = evaluate_split(
                        policy, split, settings.arm, device=device,
                        seed=settings.seed + step + (0 if split_name == "train" else 100_000),
                        bf16=settings.bf16,
                    )
                    diagnostic_seed = settings.seed + step + (
                        0 if split_name == "train" else 100_000
                    )
                    diagnostics = diagnostic_panel(
                        policy,
                        split,
                        device=device,
                        seed=diagnostic_seed,
                        bf16=settings.bf16,
                    )
                    rows_path = output / f"eval_{split_name}_rows_step_{step:09d}.jsonl"
                    _write_rows(rows_path, rows)
                    diagnostics_path = (
                        output / f"diagnostics_{split_name}_step_{step:09d}.json"
                    )
                    _write_json(diagnostics_path, diagnostics)
                    event = {
                        "step": step, "split": split_name, **aggregate,
                        "rows_path": str(rows_path), "rows_sha256": _sha(rows_path),
                        "diagnostics_path": str(diagnostics_path),
                        "diagnostics_sha256": _sha(diagnostics_path),
                        "payload_accuracy": diagnostics["keyed_solution"]["payload_accuracy"],
                        "counterfactual_payload_accuracy": diagnostics["keyed_counterfactual"]["payload_accuracy"],
                        "counterfactual_solution_accuracy": diagnostics["keyed_counterfactual"]["solution_accuracy"],
                        "key_ablation_delta": diagnostics["key_ablation_delta"],
                        "hint_ablation_delta": diagnostics["hint_ablation_delta"],
                    }
                    telemetry.write("eval", **event)
                    evaluations.append(event)
                checkpoint = {
                    "version": 1, "step": step, "arm": settings.arm,
                    "manifest_sha256": manifest["seal"]["manifest_sha256"],
                    "policy": policy.state_dict(),
                    "posterior": posterior.state_dict() if posterior is not None else None,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "generator_state": generator.get_state(),
                }
                checkpoint_path = output / f"checkpoint_step_{step:09d}.pt"
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save(checkpoint, temporary)
                os.replace(temporary, checkpoint_path)
                telemetry.write(
                    "checkpoint", step=step, path=str(checkpoint_path), sha256=_sha(checkpoint_path)
                )
        summary = {
            "completed": True, "end_step": settings.max_steps,
            "wall_seconds": time.time() - started, "evaluations": evaluations,
        }
        _write_json(output / "train_summary.json", summary)
        telemetry.write("train_summary", **summary)
    finally:
        telemetry.close()
    files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "completion.json":
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)})
    completion = {
        "schema": "apmdm/s4-grokking-completion-v1",
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
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, choices=(0.0, 0.01), required=True)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-steps", default="1000,3000,10000,30000,100000,300000,1000000")
    args = parser.parse_args()
    settings = Settings(
        arm=args.arm, output=args.output, max_steps=args.max_steps,
        batch_size=args.batch_size, width=args.width, blocks=args.blocks,
        heads=args.heads, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        seed=args.seed, device=args.device, bf16=args.bf16,
        log_every=args.log_every,
        eval_steps=parse_eval_steps(args.eval_steps, args.max_steps),
    )
    print(json.dumps(run(settings), sort_keys=True))


if __name__ == "__main__":
    main()
