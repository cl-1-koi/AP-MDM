"""Tests for the bounded D2 teacher-forced AP-MDM audit."""

from __future__ import annotations

import numpy as np
import torch

from repro.apmdm_teacher_forced_audit import (
    evaluate_terminal_stability,
    evaluate_transition_batch,
    selected_indices,
    summarize,
)
from repro.vocab import MASK


class StableTokenTwoModel:
    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, length = x.shape
        logits = torch.zeros((batch, length, 34), device=x.device)
        logits[..., 2] = 10.0
        inactive = torch.full((batch, length, 1), -10.0, device=x.device)
        return {
            "unmasking_logits": logits,
            "remasking_logits": inactive,
            "expansion_logits": inactive,
            "contraction_logits": inactive,
        }


def test_selected_indices_are_seeded_unique_and_bounded():
    first = selected_indices(100, 16, 42)
    second = selected_indices(100, 16, 42)
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 16
    assert int(first.min()) >= 0 and int(first.max()) < 100


def test_perfect_unmask_transition_is_counted_exactly():
    sample = {
        "x_k": np.asarray([MASK, 1], dtype=np.uint8),
        "y_star": np.asarray([2, 1], dtype=np.uint8),
        "r_star": np.zeros(2, dtype=np.uint8),
        "e_star": np.zeros(2, dtype=np.uint8),
        "c_star": np.zeros(2, dtype=np.uint8),
        "x_k_plus_1": np.asarray([2, 1], dtype=np.uint8),
        "solver_metadata": {"operation": "assign_unmask"},
    }
    counts = evaluate_transition_batch(
        StableTokenTwoModel(),
        [sample],
        device=torch.device("cpu"),
        threshold=0.5,
        autocast_dtype=None,
    )
    result = summarize(counts)
    assert result["transitions"] == 1
    assert result["transition_exact_rate"] == 1.0
    assert result["unmask_accuracy"] == 1.0
    assert result["operations"]["assign_unmask"]["transition_exact_rate"] == 1.0
    assert result["remask"]["predicted_positive"] == 0


def test_terminal_stability_requires_all_operation_heads_to_stay_off():
    terminal_state = np.ones((1, 324), dtype=np.uint8)
    result = evaluate_terminal_stability(
        StableTokenTwoModel(),
        terminal_state,
        device=torch.device("cpu"),
        threshold=0.5,
        autocast_dtype=None,
    )
    assert result["terminal_states"] == 1
    assert result["terminal_stable"] == 1
    assert result["terminal_remask_positive"] == 0
