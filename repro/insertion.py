"""Monotone Sudoku generation controls for the AP-MDM reproduction.

The three arms share one spatial Transformer policy and the same authenticated
100-puzzle training split:

``fixed_ar``
    Fill the first empty cell in row-major order and predict only its digit.
``random_insertion``
    Fill a uniformly selected empty cell and predict only its digit.
``learned_insertion``
    Learn both the next empty cell and its digit.  A training-only
    Plackett--Luce posterior over solution-cell permutations is optimized with
    the two-sample RLOO estimator from Zhang et al. (2026), *Variational
    Learning for Insertion-based Generation*.

Sudoku has a known fixed canvas, so completion is exact when no empty cells
remain; there is no learned length/termination head.  The learned arm is the
fixed-canvas specialization of the paper's permutation ELBO.  It retains the
scientific mechanism of interest--learned non-monotone construction order--but
does not claim the paper's variable-length termination result.

Every generated action is monotone: once a digit is written it cannot be
remasked, replaced, or deleted.  This is the intended bridge between a
left-to-right policy and full AP-MDM backtracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from repro import data as data_mod
from repro import vocab
from repro.config import ModelSpec
from repro.model import DDiTBlock, EmbeddingLayer, LayerNorm, Rotary, TimestepEmbedder

InsertionArm = Literal["fixed_ar", "random_insertion", "learned_insertion"]
VALID_ARMS = frozenset(("fixed_ar", "random_insertion", "learned_insertion"))


def validate_arm(arm: str) -> InsertionArm:
    if arm not in VALID_ARMS:
        raise ValueError(f"unknown insertion arm {arm!r}; expected one of {sorted(VALID_ARMS)}")
    return arm  # type: ignore[return-value]


def effective_model_spec(spec: ModelSpec, vocab_size: int = 34) -> ModelSpec:
    """Return the released runtime shape (the tokenizer expands 31 to 34 ids)."""
    return replace(spec, vocab_size=int(vocab_size))


def encode_partial_grids(grids: torch.Tensor) -> torch.Tensor:
    """Encode ``(B,9,9)`` partial grids as the released 324-token state."""
    if grids.ndim != 3 or tuple(grids.shape[1:]) != (9, 9):
        raise ValueError(f"expected (B,9,9), got {tuple(grids.shape)}")
    if grids.dtype != torch.long:
        grids = grids.long()
    if bool(((grids < 0) | (grids > 9)).any()):
        raise ValueError("grid values must be in 0..9")
    batch = grids.shape[0]
    tokens = torch.empty(
        (batch, vocab.SEQUENCE_LENGTH), dtype=torch.long, device=grids.device
    )
    flat = grids.reshape(batch, vocab.NUM_CELLS)
    tokens[:, 0::4] = flat
    tokens[:, 1::4] = vocab.WHITE
    tokens[:, 2::4] = vocab.NORMAL
    tokens[:, 3::4] = vocab.SEPARATOR
    return tokens


def encode_posterior_targets(puzzles: torch.Tensor, solutions: torch.Tensor) -> torch.Tensor:
    """Encode complete targets while exposing which values were original clues.

    Value tokens contain the solution.  The color slot is WHITE for clues and
    COLOR_1 for generated cells.  No Sudoku rule or solver signal is encoded.
    """
    _validate_puzzle_solution_tensors(puzzles, solutions)
    tokens = encode_partial_grids(solutions)
    given = puzzles.reshape(puzzles.shape[0], vocab.NUM_CELLS) != 0
    colors = torch.where(
        given,
        torch.full_like(given, vocab.WHITE, dtype=torch.long),
        torch.full_like(given, vocab.COLOR_MIN, dtype=torch.long),
    )
    tokens[:, 1::4] = colors
    return tokens


def _validate_puzzle_solution_tensors(
    puzzles: torch.Tensor, solutions: torch.Tensor
) -> None:
    if puzzles.shape != solutions.shape or puzzles.ndim != 3 or tuple(puzzles.shape[1:]) != (9, 9):
        raise ValueError(
            f"expected matching (B,9,9) tensors, got {tuple(puzzles.shape)} and "
            f"{tuple(solutions.shape)}"
        )
    if bool(((puzzles < 0) | (puzzles > 9)).any()):
        raise ValueError("puzzle values must be in 0..9")
    if bool(((solutions < 1) | (solutions > 9)).any()):
        raise ValueError("solution values must be in 1..9")
    given = puzzles != 0
    if not bool((puzzles[given] == solutions[given]).all()):
        raise ValueError("solution violates a given")


class SudokuSpatialTrunk(nn.Module):
    """The released DIT trunk with task-specific heads removed."""

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
        if tokens.ndim != 2:
            raise ValueError(f"expected (B,S) tokens, got {tuple(tokens.shape)}")
        if tokens.shape[1] != vocab.SEQUENCE_LENGTH:
            raise ValueError(
                f"expected {vocab.SEQUENCE_LENGTH} tokens, got {tokens.shape[1]}"
            )
        if int(tokens.min()) < 0 or int(tokens.max()) >= self.spec.vocab_size:
            raise ValueError("token id outside model vocabulary")
        batch, seq = tokens.shape
        sigma = torch.zeros(batch, device=tokens.device)
        c = F.silu(self.sigma_map(sigma))
        x = self.vocab_embed(tokens)
        rotary = self.rotary_emb(seq, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, rotary, c)
        return self.norm_final(x)


class SudokuInsertionPolicy(nn.Module):
    """Shared policy for every monotone arm.

    ``cell_logits`` chooses one of the 81 board cells for learned insertion;
    ``digit_logits`` predicts 1..9 at every cell.  Fixed and random-order arms
    use only the digit head.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.trunk = SudokuSpatialTrunk(spec)
        self.cell_head = nn.Linear(spec.hidden_size, 1)
        self.digit_head = nn.Linear(spec.hidden_size, 9)
        nn.init.normal_(self.cell_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.cell_head.bias)
        nn.init.zeros_(self.digit_head.weight)
        nn.init.zeros_(self.digit_head.bias)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(tokens)[:, 0::4]
        return {
            "cell_logits": self.cell_head(hidden).squeeze(-1),
            "digit_logits": self.digit_head(hidden),
        }

    def signature(self) -> str:
        return (
            f"sudoku-monotone-policy|L={self.spec.n_blocks}|H={self.spec.n_heads}"
            f"|d={self.spec.hidden_size}|V={self.spec.vocab_size}|state=324"
            "|heads=cell+digit9"
        )


class SudokuOrderPosterior(nn.Module):
    """Training-only Plackett--Luce posterior over non-given cell orders."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.trunk = SudokuSpatialTrunk(spec)
        self.order_head = nn.Linear(spec.hidden_size, 1)
        nn.init.normal_(self.order_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.order_head.bias)

    def forward(self, target_tokens: torch.Tensor) -> torch.Tensor:
        return self.order_head(self.trunk(target_tokens)[:, 0::4]).squeeze(-1)


def default_posterior_spec(policy_spec: ModelSpec) -> ModelSpec:
    """Paper-like smaller posterior: half width and half depth, at least one block."""
    hidden = max(64, policy_spec.hidden_size // 2)
    heads = min(policy_spec.n_heads, 4)
    while hidden % heads:
        heads -= 1
    cond = max(32, policy_spec.cond_dim // 2)
    return replace(
        policy_spec,
        hidden_size=hidden,
        cond_dim=cond,
        n_heads=heads,
        n_blocks=max(1, policy_spec.n_blocks // 2),
    )


def _orders(candidate: torch.Tensor, arm: InsertionArm, generator: torch.Generator) -> torch.Tensor:
    batch, cells = candidate.shape
    if arm == "fixed_ar":
        scores = -torch.arange(cells, device=candidate.device).expand(batch, -1)
        scores = scores.to(torch.float32)
    elif arm == "random_insertion":
        scores = torch.rand((batch, cells), device=candidate.device, generator=generator)
    else:
        raise ValueError("_orders only supports fixed_ar and random_insertion")
    return scores.masked_fill(~candidate, -torch.inf).argsort(dim=-1, descending=True)


def _sample_prefix_lengths(lengths: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Uniformly sample prefix lengths in ``[0, L-1]`` for each example."""
    if bool((lengths <= 0).any()):
        raise ValueError("every training puzzle must have at least one empty cell")
    u = torch.rand(lengths.shape, device=lengths.device, generator=generator)
    return torch.floor(u * lengths).long()


def _prefix_mask(order: torch.Tensor, prefix_lengths: torch.Tensor) -> torch.Tensor:
    batch, cells = order.shape
    ranks = torch.arange(cells, device=order.device)[None, :]
    take = ranks < prefix_lengths[:, None]
    out = torch.zeros((batch, cells), dtype=torch.bool, device=order.device)
    out.scatter_(1, order, take)
    return out


def _partial_states(
    puzzles: torch.Tensor, solutions: torch.Tensor, prefix: torch.Tensor
) -> torch.Tensor:
    batch = puzzles.shape[0]
    values = puzzles.reshape(batch, vocab.NUM_CELLS).clone()
    solved = solutions.reshape(batch, vocab.NUM_CELLS)
    values[prefix] = solved[prefix]
    return encode_partial_grids(values.reshape(batch, 9, 9))


@dataclass(frozen=True)
class MonotoneLoss:
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


def fixed_or_random_loss(
    policy: SudokuInsertionPolicy,
    puzzles: torch.Tensor,
    solutions: torch.Tensor,
    arm: InsertionArm,
    generator: torch.Generator,
) -> MonotoneLoss:
    """Teacher-forced digit loss under fixed or uniformly random fill order."""
    if arm not in ("fixed_ar", "random_insertion"):
        raise ValueError("fixed_or_random_loss requires a non-learned arm")
    _validate_puzzle_solution_tensors(puzzles, solutions)
    candidate = puzzles.reshape(puzzles.shape[0], -1) == 0
    lengths = candidate.sum(dim=-1)
    order = _orders(candidate, arm, generator)
    prefix_lengths = _sample_prefix_lengths(lengths, generator)
    prefix = _prefix_mask(order, prefix_lengths) & candidate
    next_cell = order.gather(1, prefix_lengths[:, None]).squeeze(1)
    output = policy(_partial_states(puzzles, solutions, prefix))
    batch_index = torch.arange(puzzles.shape[0], device=puzzles.device)
    next_logits = output["digit_logits"][batch_index, next_cell]
    target = solutions.reshape(puzzles.shape[0], -1)[batch_index, next_cell] - 1
    loss = F.cross_entropy(next_logits, target)
    accuracy = (next_logits.argmax(dim=-1) == target).float().mean()
    return MonotoneLoss(
        loss=loss,
        metrics={
            "digit_nll": loss.detach(),
            "digit_accuracy": accuracy.detach(),
            "mean_prefix": prefix_lengths.float().mean().detach(),
        },
    )


def _gumbel(shape: torch.Size, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    u = torch.rand(shape, device=device, generator=generator).clamp_(1e-7, 1 - 1e-7)
    return -torch.log(-torch.log(u))


def _sample_plackett_luce_orders(
    q_logits: torch.Tensor,
    candidate: torch.Tensor,
    samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    scores = q_logits.detach()[None] + _gumbel(
        torch.Size((samples, *q_logits.shape)), q_logits.device, generator
    )
    return scores.masked_fill(~candidate[None], -torch.inf).argsort(
        dim=-1, descending=True
    )


def _prefix_log_probability(
    q_logits: torch.Tensor,
    candidate: torch.Tensor,
    order: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``log q(prefix)`` and its selected-cell mask."""
    batch, cells = candidate.shape
    remaining = candidate.clone()
    selected_mask = torch.zeros_like(candidate)
    log_probability = torch.zeros(batch, device=q_logits.device)
    batch_index = torch.arange(batch, device=q_logits.device)
    max_prefix = int(prefix_lengths.max().item())
    for k in range(max_prefix):
        active = prefix_lengths > k
        denom = torch.logsumexp(q_logits.masked_fill(~remaining, -torch.inf), dim=-1)
        selected = order[:, k]
        term = q_logits[batch_index, selected] - denom
        log_probability = log_probability + torch.where(active, term, torch.zeros_like(term))
        active_index = batch_index[active]
        active_selected = selected[active]
        selected_mask[active_index, active_selected] = True
        remaining[batch_index, selected] = False
    return log_probability, selected_mask


def _exact_next_elbo(
    policy: SudokuInsertionPolicy,
    puzzles: torch.Tensor,
    solutions: torch.Tensor,
    q_logits: torch.Tensor,
    candidate: torch.Tensor,
    prefix: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the exact inner expectation over the next unfilled cell."""
    remaining = candidate & ~prefix
    if bool((remaining.sum(dim=-1) <= 0).any()):
        raise ValueError("prefix must leave at least one cell for the exact next-step sum")
    output = policy(_partial_states(puzzles, solutions, prefix))
    q_log_prob = F.log_softmax(q_logits.masked_fill(~remaining, -torch.inf), dim=-1)
    q_prob = q_log_prob.exp()
    p_cell_log_prob = F.log_softmax(
        output["cell_logits"].masked_fill(~remaining, -torch.inf), dim=-1
    )
    targets = solutions.reshape(solutions.shape[0], -1).long() - 1
    p_digit_log_prob = F.log_softmax(output["digit_logits"], dim=-1).gather(
        -1, targets[..., None]
    ).squeeze(-1)
    bracket = p_cell_log_prob + p_digit_log_prob - q_log_prob
    f_value = (q_prob * torch.where(remaining, bracket, torch.zeros_like(bracket))).sum(-1)
    q_entropy = -(q_prob * torch.where(remaining, q_log_prob, torch.zeros_like(q_log_prob))).sum(-1)
    p_entropy = -(
        p_cell_log_prob.exp()
        * torch.where(remaining, p_cell_log_prob, torch.zeros_like(p_cell_log_prob))
    ).sum(-1)
    expected_digit_nll = -(
        q_prob * torch.where(remaining, p_digit_log_prob, torch.zeros_like(p_digit_log_prob))
    ).sum(-1)
    return f_value, {
        "q_next_entropy": q_entropy,
        "p_cell_entropy": p_entropy,
        "expected_digit_nll": expected_digit_nll,
    }


def learned_insertion_loss(
    policy: SudokuInsertionPolicy,
    posterior: SudokuOrderPosterior,
    puzzles: torch.Tensor,
    solutions: torch.Tensor,
    generator: torch.Generator,
    rloo_samples: int = 2,
) -> MonotoneLoss:
    """Fixed-canvas permutation ELBO with the paper's two-sample RLOO surrogate."""
    if rloo_samples != 2:
        raise ValueError("the audited RLOO implementation requires exactly two samples")
    _validate_puzzle_solution_tensors(puzzles, solutions)
    candidate = puzzles.reshape(puzzles.shape[0], -1) == 0
    lengths = candidate.sum(dim=-1)
    prefix_lengths = _sample_prefix_lengths(lengths, generator)
    target_tokens = encode_posterior_targets(puzzles, solutions)
    q_logits = posterior(target_tokens).masked_fill(~candidate, -torch.inf)
    orders = _sample_plackett_luce_orders(
        q_logits, candidate, rloo_samples, generator
    )

    prefix_log_q = []
    f_values = []
    diagnostic_rows = []
    for sample_index in range(rloo_samples):
        log_q, prefix = _prefix_log_probability(
            q_logits, candidate, orders[sample_index], prefix_lengths
        )
        f_value, diagnostics = _exact_next_elbo(
            policy, puzzles, solutions, q_logits, candidate, prefix
        )
        prefix_log_q.append(log_q)
        f_values.append(f_value)
        diagnostic_rows.append(diagnostics)

    log_q_1, log_q_2 = prefix_log_q
    f_1, f_2 = f_values
    scale = lengths.float()
    rloo = (log_q_1 - log_q_2) * (f_1 - f_2).detach()
    surrogate = 0.5 * scale * (rloo + f_1 + f_2)
    monitor_elbo = 0.5 * scale * (f_1 + f_2)
    loss = -surrogate.mean()

    def mean_diag(name: str) -> torch.Tensor:
        return torch.stack([row[name] for row in diagnostic_rows]).mean().detach()

    return MonotoneLoss(
        loss=loss,
        metrics={
            "elbo": monitor_elbo.mean().detach(),
            "surrogate": surrogate.mean().detach(),
            "q_prefix_log_prob": torch.stack(prefix_log_q).mean().detach(),
            "q_next_entropy": mean_diag("q_next_entropy"),
            "p_cell_entropy": mean_diag("p_cell_entropy"),
            "expected_digit_nll": mean_diag("expected_digit_nll"),
            "mean_prefix": prefix_lengths.float().mean().detach(),
            "mean_blanks": lengths.float().mean().detach(),
        },
    )


@dataclass(frozen=True)
class SolveResult:
    grids: np.ndarray
    solved: np.ndarray
    valid: np.ndarray
    consistent: np.ndarray
    steps: np.ndarray

    def aggregate(self) -> dict:
        n = int(self.solved.size)
        return {
            "n": n,
            "solved": int(self.solved.sum()),
            "solve_rate": float(self.solved.mean()) if n else 0.0,
            "valid": int(self.valid.sum()),
            "valid_rate": float(self.valid.mean()) if n else 0.0,
            "consistent": int(self.consistent.sum()),
            "consistent_rate": float(self.consistent.mean()) if n else 0.0,
            "mean_steps": float(self.steps.mean()) if n else 0.0,
        }


@torch.no_grad()
def solve_monotone(
    policy: SudokuInsertionPolicy,
    puzzles: np.ndarray,
    arm: InsertionArm,
    *,
    device: torch.device | str,
    seed: int = 0,
    sample_digits: bool = False,
) -> SolveResult:
    """Solve each puzzle without revising any selected digit."""
    arm = validate_arm(arm)
    device = torch.device(device)
    arr = np.asarray(puzzles, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[1:] != (9, 9):
        raise ValueError(f"expected (N,9,9), got {arr.shape}")
    grids = torch.as_tensor(arr, dtype=torch.long, device=device).clone()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    steps = torch.zeros(grids.shape[0], dtype=torch.long, device=device)
    policy.eval()

    for _ in range(vocab.NUM_CELLS):
        empty = grids.reshape(grids.shape[0], -1) == 0
        active = empty.any(dim=-1)
        if not bool(active.any()):
            break
        output = policy(encode_partial_grids(grids))
        if arm == "fixed_ar":
            cell = empty.to(torch.int64).argmax(dim=-1)
        elif arm == "random_insertion":
            random_scores = torch.rand(empty.shape, device=device, generator=generator)
            cell = random_scores.masked_fill(~empty, -torch.inf).argmax(dim=-1)
        else:
            cell = output["cell_logits"].masked_fill(~empty, -torch.inf).argmax(dim=-1)
        batch_index = torch.arange(grids.shape[0], device=device)
        digit_logits = output["digit_logits"][batch_index, cell]
        if sample_digits:
            digit = torch.multinomial(
                F.softmax(digit_logits, dim=-1), 1, generator=generator
            ).squeeze(-1) + 1
        else:
            digit = digit_logits.argmax(dim=-1) + 1
        flat = grids.reshape(grids.shape[0], -1)
        flat[batch_index[active], cell[active]] = digit[active]
        steps[active] += 1

    out = grids.cpu().numpy().astype(np.uint8)
    valid = np.asarray([data_mod.is_valid_grid(g) for g in out], dtype=bool)
    consistent = np.asarray(
        [data_mod.is_consistent_with_givens(p, g) for p, g in zip(arr, out)],
        dtype=bool,
    )
    solved = valid & consistent
    return SolveResult(
        grids=out,
        solved=solved,
        valid=valid,
        consistent=consistent,
        steps=steps.cpu().numpy(),
    )
