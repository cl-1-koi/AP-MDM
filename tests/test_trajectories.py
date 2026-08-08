"""Trajectory generation, atomic operations, accounting and storage tests."""

from __future__ import annotations

import numpy as np
import pytest

from repro import data as data_mod
from repro import vocab
from repro.hashing import sha256_array
from repro.smoke import sequence_roundtrip_fixture
from repro.transition import apply_transition, apply_transition_length_preserving
from repro.trajectories import (
    OPERATION_NAMES,
    TransitionDataset,
    accounting_from_counts,
    build_transition_store,
    deterministic_split,
    generate_puzzle_transitions,
    validate_puzzle_transitions,
)


# --------------------------------------------------------------------------
# The transition function g
# --------------------------------------------------------------------------


def test_transition_function_matches_hand_worked_fixture():
    x, y, r, e, d, expected = sequence_roundtrip_fixture()
    assert np.array_equal(apply_transition(x, y, r, e, d), expected)


def test_remask_beats_unmask_and_delete_only_applies_to_masks():
    M = vocab.MASK
    x = np.array([M, 7], dtype=np.int64)
    y = np.array([4, 4], dtype=np.int64)
    # position 0 is MASK with remask=1 -> stays MASK, does NOT take y
    assert np.array_equal(
        apply_transition(x, y, [1, 0], [0, 0], [0, 0]), np.array([M, 7])
    )
    # delete on a non-MASK position is a no-op
    assert np.array_equal(
        apply_transition(x, y, [0, 0], [0, 0], [0, 1]), np.array([4, 7])
    )
    # delete on a MASK position removes it
    assert np.array_equal(apply_transition(x, y, [0, 0], [0, 0], [1, 0]), np.array([7]))


def test_length_preserving_fast_path_agrees_with_the_general_form():
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(4, 20))
        x = rng.integers(0, 31, size=n)
        y = rng.integers(0, 10, size=n)
        r = rng.integers(0, 2, size=n)
        zeros = np.zeros(n, dtype=np.int64)
        assert np.array_equal(
            apply_transition(x, y, r, zeros, zeros),
            apply_transition_length_preserving(x, y, r),
        )


def test_transition_rejects_mismatched_control_lengths():
    with pytest.raises(ValueError):
        apply_transition([1, 2, 3], [1, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0])


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def test_generation_is_deterministic(train_split):
    a = generate_puzzle_transitions(np.asarray(train_split.puzzles[1]), 1)
    b = generate_puzzle_transitions(np.asarray(train_split.puzzles[1]), 1)
    assert a.count == b.count
    for name in ("x_k", "y_star", "r_star", "e_star", "c_star", "x_next", "op_id"):
        assert sha256_array(getattr(a, name)) == sha256_array(getattr(b, name))


def test_generated_transitions_validate(puzzle_transitions, train_split):
    initial = data_mod.encode_puzzle_state(train_split.puzzles[0])
    report = validate_puzzle_transitions(puzzle_transitions, initial_state=initial)
    assert report.ok, report.errors[:4]
    assert report.g_consistent == report.n_transitions
    assert report.chain_continuous == report.n_transitions - 1
    assert report.control_binary
    assert report.tokens_legal


def test_atomic_operations_are_all_exercised(puzzle_transitions):
    counts = puzzle_transitions.operation_counts()
    for operation in (
        "assign_remask",
        "assign_unmask",
        "branch_remask",
        "branch_unmask",
        "contradiction_remask",
        "contradiction_unmask",
        "backtrack_modified_remask",
        "backtrack_modified_unmask",
        "skull_to_normal_step1",
        "skull_to_normal_step2",
    ):
        assert counts[operation] > 0, operation
    assert set(counts) == set(OPERATION_NAMES)


def test_remask_unmask_steps_pair_up(puzzle_transitions):
    counts = puzzle_transitions.operation_counts()
    for a, b in (
        ("assign_remask", "assign_unmask"),
        ("branch_remask", "branch_unmask"),
        ("contradiction_remask", "contradiction_unmask"),
        ("backtrack_modified_remask", "backtrack_modified_unmask"),
        ("skull_to_normal_step1", "skull_to_normal_step2"),
        ("skull_to_branch_step1", "skull_to_branch_step2"),
    ):
        assert counts[a] == counts[b], (a, b, counts[a], counts[b])


def test_trajectory_ends_at_the_ground_truth_solution(puzzle_transitions, train_split):
    final = data_mod.decode_state(puzzle_transitions.x_next[-1])
    assert not final.malformed
    assert final.complete
    assert np.array_equal(final.grid, train_split.solutions[0])


def test_sudoku_supervision_never_sets_insert_or_delete_labels(puzzle_transitions):
    """Measured fact, recorded in the audit: Sudoku exercises only 2 of 4 ops."""
    assert int(puzzle_transitions.e_star.sum()) == 0
    assert int(puzzle_transitions.c_star.sum()) == 0
    assert int(puzzle_transitions.r_star.sum()) > 0


def test_backtrack_stores_the_failed_digit_in_the_marker_slot(puzzle_transitions):
    markers = puzzle_transitions.x_next[:, vocab.SLOT_MARKER :: vocab.TOKENS_PER_CELL]
    digits_in_marker = ((markers >= vocab.DIGIT_MIN) & (markers <= vocab.DIGIT_MAX)).sum()
    assert digits_in_marker > 0


def test_validation_detects_a_corrupted_transition(puzzle_transitions):
    import copy

    corrupted = copy.deepcopy(puzzle_transitions)
    corrupted.x_next = np.array(corrupted.x_next, copy=True)
    corrupted.x_next[5, 0] = (corrupted.x_next[5, 0] + 1) % 10
    report = validate_puzzle_transitions(corrupted)
    assert not report.ok
    assert report.g_consistent < report.n_transitions


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------


def test_accounting_reports_per_source_puzzle_not_independent_samples():
    counts = [10, 20, 30]
    acct = accounting_from_counts(counts, {name: 0 for name in OPERATION_NAMES})
    assert acct["n_source_puzzles"] == 3
    assert acct["total_solver_derived_transitions"] == 60
    assert acct["transitions_per_source_puzzle_mean"] == 20.0
    assert acct["transitions_per_source_puzzle"] == counts
    assert "not independent puzzles" in acct["note"]
    # No key claims transitions are samples or puzzles.
    for key in acct:
        assert "independent_samples" not in key


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_store(tmp_path_factory, train_split):
    out = tmp_path_factory.mktemp("store") / "tiny"
    puzzles = np.asarray(train_split.puzzles[:2])
    initial = data_mod.encode_puzzle_states(puzzles)
    index = build_transition_store(
        puzzles, out_dir=out, tag="tiny", validate=True, initial_states=initial
    )
    return out, index


def test_store_roundtrip_preserves_every_array(small_store, train_split):
    out, index = small_store
    dataset = TransitionDataset(out)
    assert len(dataset) == index["total_transitions"]
    assert dataset.n_source_puzzles == 2

    reference = generate_puzzle_transitions(np.asarray(train_split.puzzles[0]), 0)
    idx = np.arange(min(64, reference.count))
    batch = dataset.batch(idx)
    assert np.array_equal(batch["x_k"], reference.x_k[idx])
    assert np.array_equal(batch["y_star"], reference.y_star[idx])
    assert np.array_equal(batch["r_star"], reference.r_star[idx])
    assert np.array_equal(batch["e_star"], reference.e_star[idx])
    assert np.array_equal(batch["c_star"], reference.c_star[idx])
    assert (batch["instance_id"] == 0).all()


def test_store_index_records_accounting_and_hashes(small_store):
    _, index = small_store
    assert index["accounting"]["n_source_puzzles"] == 2
    assert index["validation_summary"]["all_g_consistent"]
    assert index["validation_summary"]["all_chains_continuous"]
    assert index["validation_summary"]["total_insert_labels_set"] == 0
    assert index["validation_summary"]["total_delete_labels_set"] == 0
    assert set(index["file_sha256"]) >= {"x_k.u8", "y_star.u8", "r_star.bits"}
    assert len(index["index_sha256"]) == 64


def test_store_refuses_to_write_inside_the_repository(train_split):
    from repro.paths import repo_root

    with pytest.raises(ValueError):
        build_transition_store(
            np.asarray(train_split.puzzles[:1]), out_dir=repo_root() / "should_not_exist"
        )
    assert not (repo_root() / "should_not_exist").exists()


def test_deterministic_split_is_reproducible_and_disjoint():
    a_train, a_monitor = deterministic_split(1000, 0.99, 42)
    b_train, b_monitor = deterministic_split(1000, 0.99, 42)
    assert np.array_equal(a_train, b_train)
    assert np.array_equal(a_monitor, b_monitor)
    assert a_train.size == 990 and a_monitor.size == 10
    assert set(a_train.tolist()).isdisjoint(a_monitor.tolist())
    assert sorted(a_train.tolist() + a_monitor.tolist()) == list(range(1000))

    c_train, _ = deterministic_split(1000, 0.99, 43)
    assert not np.array_equal(a_train, c_train)
