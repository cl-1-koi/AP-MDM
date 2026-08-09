"""Tests for the monotone Sudoku generation ladder."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from repro import vocab
from repro.config import ModelSpec
from repro.insertion import (
    SudokuInsertionPolicy,
    SudokuOrderPosterior,
    default_posterior_spec,
    effective_model_spec,
    encode_partial_grids,
    encode_posterior_targets,
    fixed_or_random_loss,
    learned_insertion_loss,
    solve_monotone,
)


def solved_grid() -> np.ndarray:
    return np.asarray(
        [[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)],
        dtype=np.uint8,
    )


def tiny_spec() -> ModelSpec:
    return ModelSpec(
        vocab_size=34,
        length=400,
        hidden_size=32,
        n_heads=4,
        n_blocks=1,
        cond_dim=16,
        dropout=0.0,
        mlp_ratio=2,
    )


def batch_pair(batch: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    solution = solved_grid()
    puzzle = solution.copy()
    puzzle[0, 0] = 0
    puzzle[1, 4] = 0
    puzzle[7, 8] = 0
    puzzles = torch.as_tensor(np.repeat(puzzle[None], batch, axis=0)).long()
    solutions = torch.as_tensor(np.repeat(solution[None], batch, axis=0)).long()
    return puzzles, solutions


def test_partial_and_posterior_encodings_preserve_spatial_contract():
    puzzles, solutions = batch_pair(1)
    partial = encode_partial_grids(puzzles)
    target = encode_posterior_targets(puzzles, solutions)
    assert partial.shape == (1, 324)
    assert torch.equal(partial[:, 0::4], puzzles.reshape(1, 81))
    assert torch.all(partial[:, 1::4] == vocab.WHITE)
    assert torch.all(partial[:, 2::4] == vocab.NORMAL)
    assert torch.all(partial[:, 3::4] == vocab.SEPARATOR)
    assert torch.equal(target[:, 0::4], solutions.reshape(1, 81))
    missing = puzzles.reshape(1, 81) == 0
    assert torch.all(target[:, 1::4][missing] == vocab.COLOR_MIN)
    assert torch.all(target[:, 1::4][~missing] == vocab.WHITE)


def test_transposed_solution_hint_exposes_relabelled_digits_at_other_cells():
    puzzles, solutions = batch_pair(1)
    hints = solutions.transpose(-2, -1)
    encoded = encode_partial_grids(puzzles, hints)
    colors = encoded[:, 1::4].reshape(1, 9, 9)
    expected = vocab.COLOR_MIN + hints - 1
    assert torch.equal(colors, expected)
    # The hint needed at (row, col) is deliberately stored at (col, row).
    assert int(colors[0, 4, 1]) == vocab.COLOR_MIN + int(solutions[0, 1, 4]) - 1


@pytest.mark.parametrize("arm", ["fixed_ar", "random_insertion"])
def test_nonlearned_order_loss_is_finite_and_trains_digit_head(arm):
    puzzles, solutions = batch_pair()
    model = SudokuInsertionPolicy(tiny_spec())
    generator = torch.Generator().manual_seed(11)
    result = fixed_or_random_loss(model, puzzles, solutions, arm, generator)
    assert torch.isfinite(result.loss)
    assert 0 <= float(result.metrics["digit_accuracy"]) <= 1
    result.loss.backward()
    grad = model.digit_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0


def test_oracle_hint_random_insertion_loss_is_finite():
    puzzles, solutions = batch_pair()
    model = SudokuInsertionPolicy(tiny_spec())
    generator = torch.Generator().manual_seed(17)
    result = fixed_or_random_loss(
        model,
        puzzles,
        solutions,
        "random_insertion",
        generator,
        condition_mode="transposed_solution_hint",
    )
    assert torch.isfinite(result.loss)


def test_learned_permutation_elbo_reaches_policy_and_posterior():
    puzzles, solutions = batch_pair(4)
    policy = SudokuInsertionPolicy(tiny_spec())
    posterior = SudokuOrderPosterior(default_posterior_spec(tiny_spec()))
    generator = torch.Generator().manual_seed(23)
    result = learned_insertion_loss(
        policy, posterior, puzzles, solutions, generator, rloo_samples=2
    )
    assert torch.isfinite(result.loss)
    for value in result.metrics.values():
        assert torch.isfinite(value)
    result.loss.backward()
    p_grad = policy.digit_head.weight.grad
    q_grad = posterior.order_head.weight.grad
    assert p_grad is not None and float(p_grad.abs().sum()) > 0
    assert q_grad is not None and torch.isfinite(q_grad).all() and float(q_grad.abs().sum()) > 0


def test_runtime_spec_records_tokenizer_expansion():
    spec = effective_model_spec(tiny_spec(), vocab_size=34)
    assert spec.vocab_size == 34


class PerfectScriptedPolicy(nn.Module):
    def __init__(self, solution: np.ndarray):
        super().__init__()
        self.register_buffer("solution", torch.as_tensor(solution).reshape(81).long())

    def forward(self, tokens: torch.Tensor):
        batch = tokens.shape[0]
        digit_logits = torch.full((batch, 81, 9), -100.0, device=tokens.device)
        targets = self.solution[None, :, None].expand(batch, -1, -1) - 1
        digit_logits.scatter_(2, targets, 100.0)
        cell_logits = -torch.arange(81, device=tokens.device).float()[None].expand(batch, -1)
        return {"cell_logits": cell_logits, "digit_logits": digit_logits}


@pytest.mark.parametrize("arm", ["fixed_ar", "random_insertion", "learned_insertion"])
def test_monotone_solver_can_complete_without_revision(arm):
    solution = solved_grid()
    puzzle = solution.copy()
    puzzle[0, 0] = 0
    puzzle[1, 4] = 0
    puzzle[7, 8] = 0
    model = PerfectScriptedPolicy(solution)
    result = solve_monotone(model, puzzle[None], arm, device="cpu", seed=7)
    assert result.aggregate()["solve_rate"] == 1.0
    assert result.steps.tolist() == [3]
    assert np.array_equal(result.grids[0], solution)
