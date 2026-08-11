"""Tests for the S4 order/exposure diagnostic gate."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from repro.small_grokking import CELLS, HINT_MIN, KEY_MIN, S4Split, frozen_splits
from repro.sudoku_order_exposure import (
    choose_cell,
    legal_candidate_counts,
    partial_consistent,
    partial_continuable,
    rollout_horizons,
)


class HintCopyPolicy(nn.Module):
    """Perfectly recover the keyed target payload for rollout tests."""

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = len(tokens)
        hint_digits = tokens[:, 1::4] - HINT_MIN
        hint_cells = tokens[:, 2::4] - KEY_MIN
        target = torch.zeros((batch, CELLS), dtype=torch.long, device=tokens.device)
        target.scatter_(1, hint_cells, hint_digits)
        digit_logits = torch.full(
            (batch, CELLS, 4), -10.0, dtype=torch.float32, device=tokens.device
        )
        digit_logits.scatter_(2, target[..., None], 10.0)
        return {
            "cell_logits": torch.zeros((batch, CELLS), device=tokens.device),
            "digit_logits": digit_logits,
        }


def test_mrv_uses_fewest_candidates_and_row_major_ties():
    grids = torch.tensor(
        [[[1, 2, 0, 4], [3, 4, 0, 0], [2, 1, 4, 3], [4, 3, 2, 1]]]
    )
    empty = grids.reshape(1, CELLS) == 0
    counts = legal_candidate_counts(grids)
    output = {
        "cell_logits": torch.zeros((1, CELLS)),
        "digit_logits": torch.zeros((1, CELLS, 4)),
    }
    selected = choose_cell(
        grids, empty, "oracle_mrv", output, torch.Generator().manual_seed(1)
    )
    assert int(counts[0, 2]) == 1
    assert int(selected[0]) == 2


def test_partial_consistency_distinguishes_duplicates():
    good = np.asarray([[1, 0, 0, 4], [0, 4, 1, 0], [0, 1, 4, 0], [4, 0, 0, 1]])
    bad = good.copy()
    bad[0, 1] = 1
    assert partial_consistent(good)
    assert not partial_consistent(bad)
    invalid_full = np.asarray(
        [[1, 1, 3, 4], [3, 4, 1, 2], [2, 3, 4, 1], [4, 2, 2, 3]]
    )
    assert not partial_continuable(invalid_full)


def test_perfect_keyed_policy_survives_every_horizon_for_all_orders():
    _, heldout, _ = frozen_splits(42)
    split = S4Split(heldout.puzzles[:4], heldout.solutions[:4])
    policy = HintCopyPolicy()
    for order in ("fixed", "random", "learned", "oracle_mrv"):
        rows = rollout_horizons(
            policy,
            split,
            order=order,
            horizons=(1, 2, 4, 8, 10),
            device=torch.device("cpu"),
            seed=7,
            bf16=False,
        )
        assert all(row["target_prefix_survival_rate"] == 1.0 for row in rows)
        assert rows[-1]["exact_rate"] == 1.0
        assert rows[-1]["valid_rate"] == 1.0
