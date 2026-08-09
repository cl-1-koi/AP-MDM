"""Tests for counterfactual key-usage diagnostics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from repro import vocab
from repro.keyed_diagnostics import counterfactual_payloads, payload_metrics
from test_insertion import solved_grid


class KeyCopyPolicy(nn.Module):
    """Perfectly copy the color paired with each repeated cell-key token."""

    def forward(self, tokens: torch.Tensor):
        colors = tokens[:, 1::4]
        hint_keys = tokens[:, 2::4]
        query_keys = tokens[:, 3::4]
        matches = query_keys[:, :, None] == hint_keys[:, None, :]
        selected = torch.einsum("bqh,bh->bq", matches.long(), colors)
        digits = selected - vocab.COLOR_MIN
        logits = torch.full(
            (*digits.shape, 9), -100.0, dtype=torch.float32, device=tokens.device
        )
        logits.scatter_(2, digits[..., None], 100.0)
        return {
            "digit_logits": logits,
            "cell_logits": torch.zeros_like(digits, dtype=torch.float32),
        }


def test_counterfactual_payloads_change_every_digit_deterministically():
    solutions = np.repeat(solved_grid()[None], 3, axis=0)
    first = counterfactual_payloads(solutions, 17)
    second = counterfactual_payloads(solutions, 17)
    assert np.array_equal(first, second)
    assert not bool((first == solutions).any())
    assert int(first.min()) == 1 and int(first.max()) == 9


def test_perfect_key_copy_policy_follows_normal_and_counterfactual_payloads():
    solutions = np.repeat(solved_grid()[None], 4, axis=0)
    puzzles = solutions.copy()
    puzzles[:, ::2, ::2] = 0
    counterfactual = counterfactual_payloads(solutions, 23)
    policy = KeyCopyPolicy()
    normal = payload_metrics(
        policy, puzzles, solutions, solutions,
        device="cpu", seed=29, batch_size=2, bf16=False,
    )
    changed = payload_metrics(
        policy, puzzles, solutions, counterfactual,
        device="cpu", seed=31, batch_size=2, bf16=False,
    )
    assert normal["payload_accuracy"] == 1.0
    assert normal["payload_exact_rate"] == 1.0
    assert changed["payload_accuracy"] == 1.0
    assert changed["payload_exact_rate"] == 1.0
    assert changed["solution_accuracy"] == 0.0

