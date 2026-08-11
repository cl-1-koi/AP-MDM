"""Tests for the common S4 grokking data and fixed-canvas arms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from repro.small_grokking import (
    CELLS,
    KEY_MIN,
    SEQUENCE_LENGTH,
    S4Policy,
    S4Posterior,
    apply_target_mode,
    all_solutions,
    count_solutions,
    counterfactual_payloads,
    diagnostic_panel,
    encode_state,
    evaluate_split,
    fo_ao_loss,
    frozen_splits,
    lo_loss,
    oracle_mrv_orders,
    mdm_loss,
    model_spec,
    posterior_spec,
)


def test_enumerates_all_288_solutions_and_disjoint_frozen_splits():
    assert all_solutions().shape == (288, 4, 4)
    train, test, overlap = frozen_splits(42)
    assert train.n == 64 and test.n == 128
    assert overlap == {"solution_overlap": 0, "puzzle_overlap": 0, "unused_solutions": 96}
    assert all(count_solutions(puzzle) == 1 for puzzle in train.puzzles)
    assert all(count_solutions(puzzle) == 1 for puzzle in test.puzzles)
    assert np.all((train.puzzles != 0).sum((1, 2)) == 6)


def test_keyed_state_repeats_each_content_key():
    train, _, _ = frozen_splits(42)
    puzzles = torch.as_tensor(train.puzzles[:2]).long()
    solutions = torch.as_tensor(train.solutions[:2]).long()
    order = torch.stack((torch.arange(CELLS - 1, -1, -1), torch.randperm(CELLS)))
    tokens = encode_state(puzzles, solutions, order)
    assert tokens.shape == (2, SEQUENCE_LENGTH)
    assert torch.equal(tokens[:, 2::4], KEY_MIN + order)
    assert torch.equal(tokens[:, 3::4], KEY_MIN + torch.arange(CELLS).expand(2, -1))


def test_random_payload_mode_is_deterministic_and_preserves_givens():
    train, test, _ = frozen_splits(42)
    random_train, random_test, contract = apply_target_mode(
        train, test, mode="random_payload", seed=42
    )
    repeated_train, repeated_test, repeated_contract = apply_target_mode(
        train, test, mode="random_payload", seed=42
    )
    assert contract == repeated_contract
    assert np.array_equal(random_train.solutions, repeated_train.solutions)
    assert np.array_equal(random_test.solutions, repeated_test.solutions)
    for base, random_split in ((train, random_train), (test, random_test)):
        given = base.puzzles != 0
        blank = ~given
        assert np.array_equal(random_split.solutions[given], base.puzzles[given])
        assert 1 <= int(random_split.solutions[blank].min())
        assert int(random_split.solutions[blank].max()) <= 4
        # The random blank targets are not merely the Sudoku completion.
        assert float((random_split.solutions[blank] == base.solutions[blank]).mean()) < 0.35


def test_solution_target_mode_is_identity():
    train, test, _ = frozen_splits(42)
    selected_train, selected_test, contract = apply_target_mode(
        train, test, mode="sudoku_solution", seed=42
    )
    assert selected_train is train
    assert selected_test is test
    assert contract["mode"] == "sudoku_solution"


@pytest.mark.parametrize("arm", ["fo_arm", "ao_arm", "oracle_arm"])
def test_fixed_canvas_loss_is_finite_and_backpropagates(arm):
    train, _, _ = frozen_splits(42)
    puzzles = torch.as_tensor(train.puzzles[:4]).long()
    solutions = torch.as_tensor(train.solutions[:4]).long()
    policy = S4Policy(model_spec(width=32, blocks=1, heads=4))
    loss, metrics = fo_ao_loss(
        policy, puzzles, solutions, arm, torch.Generator().manual_seed(7)
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert policy.digit_head.weight.grad is not None


def test_oracle_mrv_orders_cover_every_blank_without_target_lookahead():
    train, _, _ = frozen_splits(42)
    puzzles = torch.as_tensor(train.puzzles[:4]).long()
    solutions = torch.as_tensor(train.solutions[:4]).long()
    orders = oracle_mrv_orders(puzzles, solutions)
    assert orders.shape == (4, 10)
    blank = puzzles.reshape(4, CELLS) == 0
    selected = torch.zeros_like(blank)
    selected.scatter_(1, orders, True)
    assert torch.equal(selected, blank)


def test_learning_order_loss_reaches_policy_and_posterior():
    train, _, _ = frozen_splits(42)
    puzzles = torch.as_tensor(train.puzzles[:4]).long()
    solutions = torch.as_tensor(train.solutions[:4]).long()
    spec = model_spec(width=32, blocks=1, heads=4)
    policy = S4Policy(spec)
    posterior = S4Posterior(posterior_spec(spec))
    loss, metrics = lo_loss(
        policy, posterior, puzzles, solutions, torch.Generator().manual_seed(11)
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert policy.digit_head.weight.grad is not None
    assert posterior.head.weight.grad is not None


def test_counterfactual_payloads_change_every_digit():
    train, _, _ = frozen_splits(42)
    payloads = counterfactual_payloads(train.solutions, 19)
    assert payloads.shape == train.solutions.shape
    assert np.all(payloads != train.solutions)
    assert int(payloads.min()) == 1
    assert int(payloads.max()) == 4


def test_diagnostic_panel_emits_all_declared_mechanism_metrics():
    train, _, _ = frozen_splits(42)
    split = type(train)(train.puzzles[:4], train.solutions[:4])
    policy = S4Policy(model_spec(width=32, blocks=1, heads=4))
    panel = diagnostic_panel(
        policy,
        split,
        arm="ao_arm",
        device=torch.device("cpu"),
        seed=23,
        bf16=False,
    )
    assert set(panel) == {
        "keyed_solution",
        "unkeyed_shuffled_solution",
        "no_hint",
        "keyed_counterfactual",
        "key_ablation_delta",
        "hint_ablation_delta",
    }
    assert panel["keyed_counterfactual"]["blank_cells"] > 0


def test_masked_diffusion_loss_and_rollout_path_are_finite():
    train, _, _ = frozen_splits(42)
    puzzles = torch.as_tensor(train.puzzles[:4]).long()
    solutions = torch.as_tensor(train.solutions[:4]).long()
    policy = S4Policy(model_spec(width=32, blocks=1, heads=4))
    loss, metrics = mdm_loss(
        policy, puzzles, solutions, torch.Generator().manual_seed(29)
    )
    assert torch.isfinite(loss)
    assert metrics["mean_masked"] >= 1
    loss.backward()
    assert policy.digit_head.weight.grad is not None
    panel = diagnostic_panel(
        policy,
        type(train)(train.puzzles[:4], train.solutions[:4]),
        arm="mdm",
        device=torch.device("cpu"),
        seed=31,
        bf16=False,
    )
    assert panel["keyed_solution"]["blank_cells"] == 40
    aggregate, rows = evaluate_split(
        policy,
        type(train)(train.puzzles[:4], train.solutions[:4]),
        "mdm",
        device=torch.device("cpu"),
        seed=37,
        bf16=False,
    )
    assert aggregate["n"] == 4
    assert len(rows) == 4
