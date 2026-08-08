"""Authenticated access to the official Sudoku arrays, plus leakage checks.

The two checked-in arrays have the layout ``(N, 325)``:

``row[0]``      number of givens
``row[1 + 4k]`` ``(grid_row, grid_col, value, strategy)`` for ``k`` in 0..80

with ``strategy == 0`` marking a *given* cell.  Decoding matches
``dataset/sudoku/sudoku_loader.py::SudokuLoader.read_sudoku`` exactly; the
implementation here is vectorised and returns immutable results, but the test
suite pins it against the official loader row by row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from repro import vocab
from repro.hashing import sha256_array, sha256_file
from repro.paths import dataset_dir

TRAIN_FILENAME = "sudoku-100.npy"
TEST_FILENAME = "sudoku-test-10k.npy"

#: Authoritative SHA-256 digests declared by the replication specification.
EXPECTED_FILE_SHA256: Dict[str, str] = {
    TRAIN_FILENAME: "a3e262870715fe4d3371a2682fd0656baf5af76e88ccf5f31ff83d5b2ab676e0",
    TEST_FILENAME: "36f57185fbdd7cac097c8e365845b917c0e7f9bcb7b8de212b7424d8b99eeb7c",
}

EXPECTED_SHAPE: Dict[str, Tuple[int, int]] = {
    TRAIN_FILENAME: (100, 325),
    TEST_FILENAME: (10000, 325),
}

ROW_WIDTH = 325


class ProvenanceError(RuntimeError):
    """Raised when a data file fails hash, shape or structural authentication."""


class LeakageError(RuntimeError):
    """Raised when train/test overlap is detected."""


@dataclass(frozen=True)
class SudokuSplit:
    """A decoded Sudoku split."""

    name: str
    path: Path
    file_sha256: str
    raw_sha256: str
    puzzles: np.ndarray  # (N, 9, 9) uint8, 0 = empty
    solutions: np.ndarray  # (N, 9, 9) uint8, 1..9
    raw_dtype: str
    n: int

    @property
    def givens_per_puzzle(self) -> np.ndarray:
        return (self.puzzles != 0).sum(axis=(1, 2))

    def provenance(self) -> dict:
        return {
            "name": self.name,
            "filename": self.path.name,
            "file_sha256": self.file_sha256,
            "raw_array_sha256": self.raw_sha256,
            "raw_dtype": self.raw_dtype,
            "n": int(self.n),
            "puzzles_sha256": sha256_array(self.puzzles),
            "solutions_sha256": sha256_array(self.solutions),
            "givens_min": int(self.givens_per_puzzle.min()),
            "givens_max": int(self.givens_per_puzzle.max()),
            "givens_mean": float(self.givens_per_puzzle.mean()),
        }


def decode_rows(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode ``(N, 325)`` rows into ``(puzzles, solutions)`` as ``(N, 9, 9)``.

    Raises:
        ProvenanceError: on structurally invalid rows.
    """
    if raw.ndim != 2 or raw.shape[1] != ROW_WIDTH:
        raise ProvenanceError(f"expected (N, {ROW_WIDTH}) array, got {raw.shape}")

    n = raw.shape[0]
    body = np.asarray(raw[:, 1:], dtype=np.int64).reshape(n, 81, 4)
    rows, cols, values, strategies = (body[:, :, i] for i in range(4))

    if not ((rows >= 0) & (rows <= 8)).all():
        raise ProvenanceError("row index outside 0..8")
    if not ((cols >= 0) & (cols <= 8)).all():
        raise ProvenanceError("column index outside 0..8")
    if not ((values >= 1) & (values <= 9)).all():
        raise ProvenanceError("cell value outside 1..9")

    flat = rows * 9 + cols
    order = np.argsort(flat, axis=1, kind="stable")
    if not (np.take_along_axis(flat, order, axis=1) == np.arange(81)[None, :]).all():
        raise ProvenanceError("rows do not enumerate each of the 81 cells exactly once")

    solutions = np.zeros((n, 9, 9), dtype=np.uint8)
    puzzles = np.zeros((n, 9, 9), dtype=np.uint8)
    idx = np.arange(n)[:, None]
    solutions[idx, rows, cols] = values.astype(np.uint8)
    given = strategies == 0
    puzzles[idx, rows, cols] = np.where(given, values, 0).astype(np.uint8)

    declared = np.asarray(raw[:, 0], dtype=np.int64)
    actual = given.sum(axis=1)
    if not (declared == actual).all():
        bad = int(np.flatnonzero(declared != actual)[0])
        raise ProvenanceError(
            f"row {bad}: declared givens {declared[bad]} != counted givens {actual[bad]}"
        )

    puzzles.flags.writeable = False
    solutions.flags.writeable = False
    return puzzles, solutions


def load_split(
    name: str,
    path: str | Path | None = None,
    verify_hash: bool = True,
) -> SudokuSplit:
    """Load and authenticate one of the official splits.

    Args:
        name: ``"train"`` or ``"test"``.
        path: optional explicit path, defaulting to the checked-in file.
        verify_hash: enforce the specification's SHA-256 digest.
    """
    if name not in {"train", "test"}:
        raise ValueError(f"unknown split {name!r}")
    filename = TRAIN_FILENAME if name == "train" else TEST_FILENAME
    resolved = Path(path) if path is not None else dataset_dir() / filename
    if not resolved.exists():
        raise ProvenanceError(f"missing data file: {resolved}")

    file_hash = sha256_file(resolved)
    if verify_hash and file_hash != EXPECTED_FILE_SHA256[filename]:
        raise ProvenanceError(
            f"{resolved.name}: sha256 {file_hash} != expected "
            f"{EXPECTED_FILE_SHA256[filename]}"
        )

    raw = np.load(resolved)
    if raw.shape != EXPECTED_SHAPE[filename]:
        raise ProvenanceError(
            f"{resolved.name}: shape {raw.shape} != expected {EXPECTED_SHAPE[filename]}"
        )

    puzzles, solutions = decode_rows(raw)
    return SudokuSplit(
        name=name,
        path=resolved,
        file_sha256=file_hash,
        raw_sha256=sha256_array(raw),
        puzzles=puzzles,
        solutions=solutions,
        raw_dtype=str(raw.dtype),
        n=int(raw.shape[0]),
    )


def load_raw(name: str, path: str | Path | None = None) -> np.ndarray:
    """Load the raw ``(N, 325)`` array for a split without decoding."""
    filename = TRAIN_FILENAME if name == "train" else TEST_FILENAME
    resolved = Path(path) if path is not None else dataset_dir() / filename
    return np.load(resolved)


# --------------------------------------------------------------------------
# Grid predicates
# --------------------------------------------------------------------------


def is_valid_grid(grid: np.ndarray) -> bool:
    """True iff ``grid`` is a complete, rule-satisfying 9x9 Sudoku solution."""
    arr = np.asarray(grid)
    if arr.shape != (9, 9):
        return False
    if not ((arr >= 1) & (arr <= 9)).all():
        return False
    target = set(range(1, 10))
    for i in range(9):
        if set(arr[i].tolist()) != target:
            return False
        if set(arr[:, i].tolist()) != target:
            return False
    for br in range(3):
        for bc in range(3):
            box = arr[br * 3 : br * 3 + 3, bc * 3 : bc * 3 + 3]
            if set(box.reshape(-1).tolist()) != target:
                return False
    return True


def is_consistent_with_givens(puzzle: np.ndarray, grid: np.ndarray) -> bool:
    """True iff ``grid`` preserves every non-empty cell of ``puzzle``."""
    p = np.asarray(puzzle)
    g = np.asarray(grid)
    if p.shape != (9, 9) or g.shape != (9, 9):
        return False
    given = p != 0
    return bool((g[given] == p[given]).all())


# --------------------------------------------------------------------------
# State encoding
# --------------------------------------------------------------------------


def encode_puzzle_state(puzzle: np.ndarray) -> np.ndarray:
    """Encode a 9x9 puzzle into the 324-token initial AP-MDM state.

    Matches ``APMDMSampleGenerator._encode_current_state_from_grid`` for a
    freshly started solve: every cell is ``(value|EMPTY, WHITE, NORMAL, SEP)``.
    """
    p = np.asarray(puzzle)
    if p.shape != (9, 9):
        raise ValueError(f"expected (9, 9) puzzle, got {p.shape}")
    state = np.empty((vocab.SEQUENCE_LENGTH,), dtype=np.uint8)
    flat = p.reshape(-1).astype(np.uint8)
    state[0::4] = np.where(flat == 0, vocab.EMPTY, flat)
    state[1::4] = vocab.WHITE
    state[2::4] = vocab.NORMAL
    state[3::4] = vocab.SEPARATOR
    return state


def encode_puzzle_states(puzzles: np.ndarray) -> np.ndarray:
    """Vectorised :func:`encode_puzzle_state` over ``(N, 9, 9)``."""
    arr = np.asarray(puzzles)
    if arr.ndim != 3 or arr.shape[1:] != (9, 9):
        raise ValueError(f"expected (N, 9, 9) puzzles, got {arr.shape}")
    n = arr.shape[0]
    state = np.empty((n, vocab.SEQUENCE_LENGTH), dtype=np.uint8)
    flat = arr.reshape(n, 81).astype(np.uint8)
    state[:, 0::4] = np.where(flat == 0, vocab.EMPTY, flat)
    state[:, 1::4] = vocab.WHITE
    state[:, 2::4] = vocab.NORMAL
    state[:, 3::4] = vocab.SEPARATOR
    return state


@dataclass(frozen=True)
class DecodedState:
    """Result of decoding a generated token state back to a grid."""

    grid: np.ndarray  # (9, 9) uint8, 0 for empty/masked
    complete: bool
    n_masked_values: int
    n_empty_values: int
    malformed: bool
    malformed_reason: str


def decode_state(state: np.ndarray) -> DecodedState:
    """Decode a 324-token AP-MDM state into a grid plus well-formedness flags."""
    arr = np.asarray(state).reshape(-1)
    if arr.size != vocab.SEQUENCE_LENGTH:
        return DecodedState(
            grid=np.zeros((9, 9), dtype=np.uint8),
            complete=False,
            n_masked_values=0,
            n_empty_values=0,
            malformed=True,
            malformed_reason=f"length {arr.size} != {vocab.SEQUENCE_LENGTH}",
        )

    reasons = []
    for slot, legal in vocab.LEGAL_TOKENS_BY_SLOT.items():
        slice_ = arr[slot :: vocab.TOKENS_PER_CELL]
        illegal = np.array([int(t) not in legal for t in slice_], dtype=bool)
        if illegal.any():
            first = int(np.flatnonzero(illegal)[0])
            reasons.append(
                f"slot {slot} cell {first} holds illegal token {int(slice_[first])}"
            )

    values = arr[vocab.SLOT_VALUE :: vocab.TOKENS_PER_CELL]
    digits = (values >= vocab.DIGIT_MIN) & (values <= vocab.DIGIT_MAX)
    grid = np.where(digits, values, 0).astype(np.uint8).reshape(9, 9)
    return DecodedState(
        grid=grid,
        complete=bool(digits.all()),
        n_masked_values=int((values == vocab.MASK).sum()),
        n_empty_values=int((values == vocab.EMPTY).sum()),
        malformed=bool(reasons),
        malformed_reason="; ".join(reasons),
    )


# --------------------------------------------------------------------------
# Leakage / overlap
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlapReport:
    """Measured overlap between the train and test splits."""

    raw_row_overlap: int
    puzzle_grid_overlap: int
    solution_grid_overlap: int
    puzzle_solution_pair_overlap: int
    train_unique_rows: int
    test_unique_rows: int

    @property
    def clean(self) -> bool:
        return (
            self.raw_row_overlap == 0
            and self.puzzle_grid_overlap == 0
            and self.solution_grid_overlap == 0
            and self.puzzle_solution_pair_overlap == 0
        )

    def as_dict(self) -> dict:
        return {
            "raw_row_overlap": self.raw_row_overlap,
            "puzzle_grid_overlap": self.puzzle_grid_overlap,
            "solution_grid_overlap": self.solution_grid_overlap,
            "puzzle_solution_pair_overlap": self.puzzle_solution_pair_overlap,
            "train_unique_rows": self.train_unique_rows,
            "test_unique_rows": self.test_unique_rows,
            "clean": self.clean,
        }


def _byte_rows(array: np.ndarray) -> set:
    contiguous = np.ascontiguousarray(array)
    flat = contiguous.reshape(contiguous.shape[0], -1)
    return {row.tobytes() for row in flat}


def measure_overlap(
    train: SudokuSplit,
    test: SudokuSplit,
    train_raw: np.ndarray | None = None,
    test_raw: np.ndarray | None = None,
) -> OverlapReport:
    """Measure train/test overlap at raw-row, puzzle, solution and pair level.

    Raw rows are normalised to a common dtype first: the checked-in train array
    is ``uint64`` while the test array is ``int64``, so a naive byte comparison
    would be dtype-dependent rather than value-dependent.
    """
    tr_raw = load_raw("train", train.path) if train_raw is None else train_raw
    te_raw = load_raw("test", test.path) if test_raw is None else test_raw
    tr_norm = np.asarray(tr_raw, dtype=np.int64)
    te_norm = np.asarray(te_raw, dtype=np.int64)

    tr_rows, te_rows = _byte_rows(tr_norm), _byte_rows(te_norm)
    tr_p, te_p = _byte_rows(train.puzzles), _byte_rows(test.puzzles)
    tr_s, te_s = _byte_rows(train.solutions), _byte_rows(test.solutions)
    tr_pair = {
        train.puzzles[i].tobytes() + train.solutions[i].tobytes()
        for i in range(train.n)
    }
    te_pair = {
        test.puzzles[i].tobytes() + test.solutions[i].tobytes() for i in range(test.n)
    }

    return OverlapReport(
        raw_row_overlap=len(tr_rows & te_rows),
        puzzle_grid_overlap=len(tr_p & te_p),
        solution_grid_overlap=len(tr_s & te_s),
        puzzle_solution_pair_overlap=len(tr_pair & te_pair),
        train_unique_rows=len(tr_rows),
        test_unique_rows=len(te_rows),
    )


def assert_no_leakage(report: OverlapReport) -> None:
    """Raise :class:`LeakageError` unless the overlap report is clean."""
    if not report.clean:
        raise LeakageError(f"train/test overlap detected: {report.as_dict()}")
