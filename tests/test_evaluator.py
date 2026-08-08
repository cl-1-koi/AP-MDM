"""Evaluator tests: exact scoring, malformed rejection, aggregation, intervals, gates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from repro import data as data_mod
from repro import vocab
from repro.evaluator import (
    EvaluationGateError,
    ImmutableArtifactError,
    aggregate,
    check_evaluation_preconditions,
    clopper_pearson_interval,
    evaluate_state,
    evaluate_states,
    proportion_report,
    read_rows,
    regularized_incomplete_beta,
    verdict_label,
    wilson_interval,
    write_rows,
)
from repro.sampler import TERMINATED_FIXED_POINT, TERMINATED_MAX_STEPS, RowTrace


def _solved_state(solution: np.ndarray) -> np.ndarray:
    state = np.empty(vocab.SEQUENCE_LENGTH, dtype=np.uint8)
    state[0::4] = np.asarray(solution).reshape(-1)
    state[1::4] = vocab.WHITE
    state[2::4] = vocab.NORMAL
    state[3::4] = vocab.SEPARATOR
    return state


def _trace(status=TERMINATED_FIXED_POINT, steps=10):
    return RowTrace(
        steps=steps, forward_passes=steps, status=status, n_remask=3, n_unmask=4,
        final_length=vocab.SEQUENCE_LENGTH,
    )


def test_exact_scoring_of_a_correct_solution(test_split):
    state = _solved_state(test_split.solutions[0])
    row = evaluate_state(0, test_split.puzzles[0], test_split.solutions[0], state, _trace())
    assert row.exact_match and row.valid and row.consistent_with_givens
    assert row.complete and row.solved and not row.malformed and not row.timeout


def test_a_valid_but_different_grid_is_not_an_exact_match(test_split):
    other = None
    for i in range(1, test_split.n):
        if not np.array_equal(test_split.solutions[i], test_split.solutions[0]):
            other = test_split.solutions[i]
            break
    row = evaluate_state(0, test_split.puzzles[0], test_split.solutions[0], _solved_state(other), _trace())
    assert not row.exact_match
    assert row.valid  # a different completed grid can still satisfy the rules


def test_rule_violation_is_detected(test_split):
    broken = np.array(test_split.solutions[0], copy=True)
    broken[0, 0], broken[0, 1] = broken[0, 1], broken[0, 0]
    row = evaluate_state(0, test_split.puzzles[0], test_split.solutions[0], _solved_state(broken), _trace())
    assert not row.valid and not row.exact_match


def test_incomplete_state_is_not_exact(test_split):
    state = _solved_state(test_split.solutions[0])
    state[0] = vocab.MASK
    row = evaluate_state(0, test_split.puzzles[0], test_split.solutions[0], state, _trace())
    assert not row.complete and not row.exact_match and not row.valid
    assert row.n_masked_values == 1


def test_malformed_output_is_rejected(test_split):
    state = _solved_state(test_split.solutions[0])
    state[3] = vocab.NORMAL  # separator slot corrupted
    row = evaluate_state(0, test_split.puzzles[0], test_split.solutions[0], state, _trace())
    assert row.malformed and not row.exact_match and not row.valid
    assert "slot 3" in row.malformed_reason


def test_wrong_length_state_is_rejected(test_split):
    state = _solved_state(test_split.solutions[0])[:-4]
    row = evaluate_state(0, test_split.puzzles[0], test_split.solutions[0], state, _trace())
    assert row.malformed and not row.exact_match


def test_timeout_is_recorded(test_split):
    state = _solved_state(test_split.solutions[0])
    row = evaluate_state(0, test_split.puzzles[0], test_split.solutions[0], state, _trace(TERMINATED_MAX_STEPS))
    assert row.timeout
    assert not row.solved  # terminated by ceiling, not by reaching a fixed point
    assert row.exact_match  # the state itself is still scored honestly


def test_aggregation_is_order_independent(test_split):
    states = [_solved_state(test_split.solutions[i]) for i in range(8)]
    states[3][0] = vocab.MASK
    traces = [_trace() for _ in range(8)]
    rows = evaluate_states(list(range(8)), test_split.puzzles, test_split.solutions, states, traces)
    forward = aggregate(rows)
    reverse = aggregate(list(reversed(rows)))
    assert forward == reverse
    assert forward["exact_accuracy"]["successes"] == 7
    assert forward["n_puzzles"] == 8


def test_aggregation_rejects_duplicate_puzzle_rows(test_split):
    states = [_solved_state(test_split.solutions[0])] * 2
    rows = evaluate_states([0, 0], test_split.puzzles, test_split.solutions, states, [_trace(), _trace()])
    with pytest.raises(ValueError):
        aggregate(rows)


def test_binomial_intervals_are_sane():
    lo, hi = wilson_interval(9928, 10000)
    assert lo < 0.9928 < hi
    assert hi - lo < 0.005
    clo, chi = clopper_pearson_interval(9928, 10000)
    assert clo < 0.9928 < chi
    assert clo <= lo + 1e-3 and chi >= hi - 1e-3

    assert wilson_interval(0, 100)[0] == 0.0
    assert wilson_interval(100, 100)[1] == 1.0
    assert clopper_pearson_interval(0, 100)[0] == 0.0
    assert clopper_pearson_interval(100, 100)[1] == 1.0
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_regularized_incomplete_beta_matches_known_values():
    assert math.isclose(regularized_incomplete_beta(1.0, 1.0, 0.25), 0.25, abs_tol=1e-9)
    assert math.isclose(regularized_incomplete_beta(2.0, 2.0, 0.5), 0.5, abs_tol=1e-9)
    assert math.isclose(regularized_incomplete_beta(0.5, 0.5, 0.5), 0.5, abs_tol=1e-9)
    assert regularized_incomplete_beta(3.0, 5.0, 0.0) == 0.0
    assert regularized_incomplete_beta(3.0, 5.0, 1.0) == 1.0


def test_proportion_report_shape():
    report = proportion_report(97, 100)
    assert report["estimate"] == 0.97
    assert report["wilson_95_lower"] < 0.97 < report["wilson_95_upper"]
    assert report["clopper_pearson_95_lower"] < 0.97 < report["clopper_pearson_95_upper"]


def test_predeclared_verdict_labels():
    assert verdict_label(0.9928, False) == "strong replication"
    assert verdict_label(0.9878, False) == "strong replication"
    assert verdict_label(0.9877, False) == "partial replication"
    assert verdict_label(0.9700, False) == "partial replication"
    assert verdict_label(0.9699, False) == "failure to replicate"
    assert verdict_label(0.9999, True) == "invalid/inconclusive"


def test_rows_are_immutable_on_disk(tmp_path, test_split):
    states = [_solved_state(test_split.solutions[i]) for i in range(3)]
    rows = evaluate_states([0, 1, 2], test_split.puzzles, test_split.solutions, states, [_trace()] * 3)
    path = tmp_path / "rows.jsonl"
    digest = write_rows(path, rows)
    assert len(digest) == 64
    with pytest.raises(ImmutableArtifactError):
        write_rows(path, rows)
    assert [r.as_dict() for r in read_rows(path)] == [r.as_dict() for r in sorted(rows, key=lambda r: r.puzzle_index)]


def _manifest_stub():
    return {
        "architecture": {"signature": "sig-A"},
        "vocabulary": {"effective_size": 31},
        "seal": {"manifest_sha256": "deadbeef"},
    }


def _binding_stub(**overrides):
    binding = {
        "architecture_signature": "sig-A",
        "vocabulary_size": 31,
        "manifest_sha256": "deadbeef",
        "step": 10,
    }
    binding.update(overrides)
    return binding


def test_evaluation_gate_accepts_a_matching_pair(train_split, test_split):
    overlap = data_mod.measure_overlap(train_split, test_split)
    check_evaluation_preconditions(_manifest_stub(), _binding_stub(), overlap)


def test_evaluation_gate_rejects_leakage(train_split):
    overlap = data_mod.measure_overlap(
        train_split, train_split, train_raw=data_mod.load_raw("train"), test_raw=data_mod.load_raw("train")
    )
    with pytest.raises(EvaluationGateError):
        check_evaluation_preconditions(_manifest_stub(), _binding_stub(), overlap)


@pytest.mark.parametrize(
    "override",
    [
        {"architecture_signature": "sig-B"},
        {"vocabulary_size": 34},
        {"manifest_sha256": "cafebabe"},
    ],
)
def test_evaluation_gate_rejects_mismatched_checkpoints(train_split, test_split, override):
    overlap = data_mod.measure_overlap(train_split, test_split)
    with pytest.raises(EvaluationGateError):
        check_evaluation_preconditions(_manifest_stub(), _binding_stub(**override), overlap)


def test_evaluation_gate_requires_binding_fields(train_split, test_split):
    overlap = data_mod.measure_overlap(train_split, test_split)
    with pytest.raises(EvaluationGateError):
        check_evaluation_preconditions(_manifest_stub(), {"step": 1}, overlap)
