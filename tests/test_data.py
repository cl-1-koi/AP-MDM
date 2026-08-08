"""Data provenance, structure, uniqueness, leakage and encoding tests."""

from __future__ import annotations

import numpy as np
import pytest

from repro import data as data_mod
from repro import vocab


def test_file_hashes_and_shapes(train_split, test_split):
    assert train_split.file_sha256 == data_mod.EXPECTED_FILE_SHA256[data_mod.TRAIN_FILENAME]
    assert test_split.file_sha256 == data_mod.EXPECTED_FILE_SHA256[data_mod.TEST_FILENAME]
    assert train_split.puzzles.shape == (100, 9, 9)
    assert train_split.solutions.shape == (100, 9, 9)
    assert test_split.puzzles.shape == (10000, 9, 9)
    assert test_split.solutions.shape == (10000, 9, 9)


def test_hash_mismatch_is_rejected(tmp_path):
    raw = data_mod.load_raw("train")
    tampered = np.array(raw, copy=True)
    tampered[0, 0] = tampered[0, 0] + 1
    path = tmp_path / "tampered.npy"
    np.save(path, tampered)
    with pytest.raises(data_mod.ProvenanceError):
        data_mod.load_split("train", path=path)


def test_every_solution_is_a_valid_grid(train_split, test_split):
    for i in range(train_split.n):
        assert data_mod.is_valid_grid(train_split.solutions[i])
    rng = np.random.default_rng(0)
    for i in rng.choice(test_split.n, size=400, replace=False):
        assert data_mod.is_valid_grid(test_split.solutions[int(i)])


def test_puzzles_are_consistent_with_their_solutions(train_split, test_split):
    for split in (train_split, test_split):
        given = split.puzzles != 0
        assert np.array_equal(split.puzzles[given], split.solutions[given])


def test_rows_are_unique_within_each_split(train_split, test_split):
    for split in (train_split, test_split):
        flat = np.ascontiguousarray(split.puzzles).reshape(split.n, -1)
        assert len({row.tobytes() for row in flat}) == split.n


def test_zero_train_test_overlap(train_split, test_split):
    report = data_mod.measure_overlap(train_split, test_split)
    assert report.raw_row_overlap == 0
    assert report.puzzle_grid_overlap == 0
    assert report.solution_grid_overlap == 0
    assert report.puzzle_solution_pair_overlap == 0
    assert report.clean
    data_mod.assert_no_leakage(report)


def test_leakage_is_detected_when_present(train_split):
    injected = data_mod.SudokuSplit(
        name="test",
        path=train_split.path,
        file_sha256=train_split.file_sha256,
        raw_sha256=train_split.raw_sha256,
        puzzles=train_split.puzzles,
        solutions=train_split.solutions,
        raw_dtype=train_split.raw_dtype,
        n=train_split.n,
    )
    report = data_mod.measure_overlap(train_split, injected, train_raw=data_mod.load_raw("train"), test_raw=data_mod.load_raw("train"))
    assert not report.clean
    with pytest.raises(data_mod.LeakageError):
        data_mod.assert_no_leakage(report)


def test_decoding_matches_the_official_loader(train_split):
    from repro.official import _ensure_path

    _ensure_path()
    from sudoku_loader import SudokuLoader  # type: ignore[import-not-found]

    for i in (0, 7, 42, 99):
        puzzle, solution = SudokuLoader.read_sudoku(str(train_split.path), i)
        assert np.array_equal(np.asarray(puzzle), train_split.puzzles[i])
        assert np.array_equal(np.asarray(solution), train_split.solutions[i])


def test_encode_decode_roundtrip(train_split):
    state = data_mod.encode_puzzle_state(train_split.puzzles[3])
    assert state.shape == (vocab.SEQUENCE_LENGTH,)
    assert (state[vocab.SLOT_SEPARATOR :: 4] == vocab.SEPARATOR).all()
    assert (state[vocab.SLOT_COLOR :: 4] == vocab.WHITE).all()
    assert (state[vocab.SLOT_MARKER :: 4] == vocab.NORMAL).all()
    decoded = data_mod.decode_state(state)
    assert not decoded.malformed
    assert np.array_equal(decoded.grid, train_split.puzzles[3])
    assert not decoded.complete


def test_batched_encoding_matches_single(train_split):
    batched = data_mod.encode_puzzle_states(train_split.puzzles[:5])
    for i in range(5):
        assert np.array_equal(batched[i], data_mod.encode_puzzle_state(train_split.puzzles[i]))


def test_decode_state_rejects_malformed_states(train_split):
    state = np.array(data_mod.encode_puzzle_state(train_split.puzzles[0]), copy=True)
    state[vocab.SLOT_SEPARATOR] = vocab.NORMAL
    decoded = data_mod.decode_state(state)
    assert decoded.malformed
    assert "slot 3" in decoded.malformed_reason

    short = data_mod.decode_state(state[:100])
    assert short.malformed
    assert "length" in short.malformed_reason

    bad_colour = np.array(data_mod.encode_puzzle_state(train_split.puzzles[0]), copy=True)
    bad_colour[vocab.SLOT_COLOR] = vocab.SEPARATOR
    assert data_mod.decode_state(bad_colour).malformed


def test_grid_predicates(train_split):
    solution = train_split.solutions[0]
    assert data_mod.is_valid_grid(solution)
    assert data_mod.is_consistent_with_givens(train_split.puzzles[0], solution)

    broken = np.array(solution, copy=True)
    broken[0, 0], broken[0, 1] = broken[0, 1], broken[0, 0]
    assert not data_mod.is_valid_grid(broken)

    incomplete = np.array(solution, copy=True)
    incomplete[4, 4] = 0
    assert not data_mod.is_valid_grid(incomplete)

    wrong_given = np.array(solution, copy=True)
    given = np.argwhere(train_split.puzzles[0] != 0)[0]
    wrong_given[given[0], given[1]] = 1 + (wrong_given[given[0], given[1]] % 9)
    assert not data_mod.is_consistent_with_givens(train_split.puzzles[0], wrong_given)


def test_structural_validation_rejects_bad_rows():
    raw = np.array(data_mod.load_raw("train"), dtype=np.int64)
    broken = raw.copy()
    broken[0, 1] = 9  # row index out of range
    with pytest.raises(data_mod.ProvenanceError):
        data_mod.decode_rows(broken)

    broken2 = raw.copy()
    broken2[0, 0] = 0  # declared givens no longer matches
    with pytest.raises(data_mod.ProvenanceError):
        data_mod.decode_rows(broken2)
