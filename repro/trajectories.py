"""Deterministic solver-trajectory generation, validation, accounting and storage.

Terminology is deliberate and load-bearing:

* **source puzzle** - one of the 100 authenticated training Sudoku instances.
* **transition** (equivalently *state-transition tuple*, *training tuple*) - one
  ``(x_k, y*, ctrl*) -> x_{k+1}`` record emitted by the symbolic backtracking
  solver while solving a source puzzle.

Transitions are **not** independent puzzles and are **not** independent
samples: every transition in a trajectory is a deterministic function of its
source puzzle.  Accounting below is always reported per source puzzle.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from repro import vocab
from repro.hashing import sha256_file, sha256_json
from repro.official import data_manager
from repro.paths import assert_outside_repo, transitions_dir
from repro.transition import apply_transition, apply_transition_length_preserving

CONTROL_BYTES = (vocab.SEQUENCE_LENGTH + 7) // 8  # 41

#: Operation labels emitted by the official generator, in a fixed order so that
#: the integer op ids stored on disk are stable across runs and machines.
OPERATION_NAMES: Tuple[str, ...] = (
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
    "skull_to_branch_step1",
    "skull_to_branch_step2",
)
OPERATION_IDS: Dict[str, int] = {name: i for i, name in enumerate(OPERATION_NAMES)}


class TrajectoryError(RuntimeError):
    """Raised when generated transitions fail structural validation."""


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


@dataclass
class PuzzleTransitions:
    """All transitions derived from a single source puzzle."""

    instance_id: int
    x_k: np.ndarray  # (T, 324) uint8
    y_star: np.ndarray  # (T, 324) uint8
    r_star: np.ndarray  # (T, 324) uint8 in {0, 1}
    e_star: np.ndarray  # (T, 324) uint8 in {0, 1}
    c_star: np.ndarray  # (T, 324) uint8 in {0, 1}
    x_next: np.ndarray  # (T, 324) uint8
    op_id: np.ndarray  # (T,) uint16
    seconds: float = 0.0

    @property
    def count(self) -> int:
        return int(self.x_k.shape[0])

    def operation_counts(self) -> Dict[str, int]:
        counts = np.bincount(self.op_id, minlength=len(OPERATION_NAMES))
        return {OPERATION_NAMES[i]: int(counts[i]) for i in range(len(OPERATION_NAMES))}


def generate_puzzle_transitions(puzzle: np.ndarray, instance_id: int) -> PuzzleTransitions:
    """Run the official generator on one source puzzle and collect transitions.

    The official generator is used unmodified; this function only marshals its
    output into dense arrays and rejects anything unexpected.
    """
    manager = data_manager()
    start = time.perf_counter()
    samples = manager.generate_from_puzzle(np.asarray(puzzle, dtype=np.int64), instance_id)
    seconds = time.perf_counter() - start
    if not samples:
        raise TrajectoryError(f"puzzle {instance_id}: generator produced no transitions")

    count = len(samples)
    shape = (count, vocab.SEQUENCE_LENGTH)
    x_k = np.empty(shape, dtype=np.uint8)
    y_star = np.empty(shape, dtype=np.uint8)
    r_star = np.empty(shape, dtype=np.uint8)
    e_star = np.empty(shape, dtype=np.uint8)
    c_star = np.empty(shape, dtype=np.uint8)
    x_next = np.empty(shape, dtype=np.uint8)
    op_id = np.empty((count,), dtype=np.uint16)

    for i, sample in enumerate(samples):
        for name, dest in (
            ("x_k", x_k),
            ("y_star", y_star),
            ("r_star", r_star),
            ("e_star", e_star),
            ("c_star", c_star),
            ("x_k_plus_1", x_next),
        ):
            arr = np.asarray(sample[name], dtype=np.uint8).reshape(-1)
            if arr.size != vocab.SEQUENCE_LENGTH:
                raise TrajectoryError(
                    f"puzzle {instance_id} transition {i}: {name} has length "
                    f"{arr.size}, expected {vocab.SEQUENCE_LENGTH}"
                )
            dest[i] = arr
        operation = str(sample["solver_metadata"]["operation"])
        if operation not in OPERATION_IDS:
            raise TrajectoryError(
                f"puzzle {instance_id} transition {i}: unknown operation {operation!r}"
            )
        op_id[i] = OPERATION_IDS[operation]
        recorded = int(sample["solver_metadata"]["instance_id"])
        if recorded != instance_id:
            raise TrajectoryError(
                f"puzzle {instance_id} transition {i}: instance_id {recorded} mismatch"
            )

    return PuzzleTransitions(
        instance_id=instance_id,
        x_k=x_k,
        y_star=y_star,
        r_star=r_star,
        e_star=e_star,
        c_star=c_star,
        x_next=x_next,
        op_id=op_id,
        seconds=seconds,
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Structural validation of one puzzle's transitions."""

    instance_id: int
    n_transitions: int
    g_consistent: int
    chain_continuous: int
    control_binary: bool
    tokens_legal: bool
    insert_labels_nonzero: int
    delete_labels_nonzero: int
    remask_labels_nonzero: int
    unmask_target_positions: int
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_puzzle_transitions(
    tr: PuzzleTransitions, initial_state: np.ndarray | None = None
) -> ValidationReport:
    """Validate one puzzle's transitions against the paper's Algorithm 1.

    Checks, per transition:

    * control labels are binary,
    * every token occupies a legal role for its cell slot,
    * ``g(x_k, y*, ctrl*) == x_{k+1}`` exactly,
    * the chain is continuous: ``x_{k+1}`` of step ``t`` equals ``x_k`` of
      step ``t + 1``,
    * (optionally) the first ``x_k`` equals the encoded source puzzle.
    """
    errors: List[str] = []
    n = tr.count

    control_binary = True
    for name, arr in (("r_star", tr.r_star), ("e_star", tr.e_star), ("c_star", tr.c_star)):
        if not np.isin(arr, (0, 1)).all():
            control_binary = False
            errors.append(f"{name} contains values outside {{0, 1}}")

    tokens_legal = True
    for slot, legal in vocab.LEGAL_TOKENS_BY_SLOT.items():
        legal_arr = np.zeros(256, dtype=bool)
        legal_arr[list(legal)] = True
        for name, arr in (("x_k", tr.x_k), ("y_star", tr.y_star), ("x_next", tr.x_next)):
            column = arr[:, slot :: vocab.TOKENS_PER_CELL]
            if not legal_arr[column].all():
                bad = int(column[~legal_arr[column]][0])
                tokens_legal = False
                errors.append(f"{name} slot {slot} holds illegal token {bad}")

    # g-consistency.  Transitions with no insert/delete control are checked with
    # the vectorised length-preserving form of g; the rest fall back to the
    # general per-position implementation.  Both come from repro.transition, so
    # the two paths cannot drift apart (a test pins their agreement).
    structural = (tr.e_star.any(axis=1)) | ((tr.c_star & (tr.x_k == vocab.MASK)).any(axis=1))
    plain = ~structural
    g_ok = np.zeros(n, dtype=bool)
    if plain.any():
        produced = apply_transition_length_preserving(
            tr.x_k[plain], tr.y_star[plain], tr.r_star[plain]
        )
        g_ok[plain] = (produced == tr.x_next[plain]).all(axis=1)
    for i in np.flatnonzero(structural):
        produced = apply_transition(
            tr.x_k[i], tr.y_star[i], tr.r_star[i], tr.e_star[i], tr.c_star[i]
        )
        g_ok[i] = produced.size == tr.x_next[i].size and np.array_equal(
            produced, tr.x_next[i]
        )
    g_consistent = int(g_ok.sum())
    for i in np.flatnonzero(~g_ok)[:8]:
        errors.append(
            f"transition {int(i)} ({OPERATION_NAMES[tr.op_id[i]]}): "
            "g(x_k, y*, ctrl*) != x_{k+1}"
        )

    if n > 1:
        continuous = (tr.x_next[:-1] == tr.x_k[1:]).all(axis=1)
        chain_continuous = int(continuous.sum())
        for i in np.flatnonzero(~continuous)[:4]:
            errors.append(f"chain break between transition {int(i)} and {int(i) + 1}")
    else:
        chain_continuous = 0

    if initial_state is not None and not np.array_equal(tr.x_k[0], initial_state):
        errors.append("first x_k does not equal the encoded source puzzle state")

    unmask_positions = int(
        ((tr.x_k == vocab.MASK) & (tr.y_star != vocab.MASK)).sum()
    )
    return ValidationReport(
        instance_id=tr.instance_id,
        n_transitions=n,
        g_consistent=g_consistent,
        chain_continuous=chain_continuous,
        control_binary=control_binary,
        tokens_legal=tokens_legal,
        insert_labels_nonzero=int(tr.e_star.sum()),
        delete_labels_nonzero=int(tr.c_star.sum()),
        remask_labels_nonzero=int(tr.r_star.sum()),
        unmask_target_positions=unmask_positions,
        errors=errors,
    )


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

_ARRAY_FILES = {
    "x_k": ("x_k.u8", np.uint8, vocab.SEQUENCE_LENGTH),
    "y_star": ("y_star.u8", np.uint8, vocab.SEQUENCE_LENGTH),
    "r_star": ("r_star.bits", np.uint8, CONTROL_BYTES),
    "e_star": ("e_star.bits", np.uint8, CONTROL_BYTES),
    "c_star": ("c_star.bits", np.uint8, CONTROL_BYTES),
}


class TransitionWriter:
    """Streams generated transitions into out-of-tree memory-mapped files."""

    def __init__(self, root: Path, total: int):
        self.root = assert_outside_repo(Path(root))
        self.root.mkdir(parents=True, exist_ok=True)
        self.total = int(total)
        self._offset = 0
        self._maps = {}
        for key, (filename, dtype, width) in _ARRAY_FILES.items():
            self._maps[key] = np.memmap(
                self.root / filename, dtype=dtype, mode="w+", shape=(self.total, width)
            )
        self._instance = np.memmap(
            self.root / "instance_id.u16", dtype=np.uint16, mode="w+", shape=(self.total,)
        )
        self._op = np.memmap(
            self.root / "op_id.u16", dtype=np.uint16, mode="w+", shape=(self.total,)
        )

    def append(self, tr: PuzzleTransitions) -> Tuple[int, int]:
        start = self._offset
        end = start + tr.count
        if end > self.total:
            raise TrajectoryError(
                f"transition capacity exceeded: {end} > {self.total}"
            )
        self._maps["x_k"][start:end] = tr.x_k
        self._maps["y_star"][start:end] = tr.y_star
        self._maps["r_star"][start:end] = np.packbits(tr.r_star, axis=1)
        self._maps["e_star"][start:end] = np.packbits(tr.e_star, axis=1)
        self._maps["c_star"][start:end] = np.packbits(tr.c_star, axis=1)
        self._instance[start:end] = tr.instance_id
        self._op[start:end] = tr.op_id
        self._offset = end
        return start, end

    def close(self) -> int:
        written = self._offset
        for key, mmap in list(self._maps.items()):
            mmap.flush()
            del mmap
        self._maps.clear()
        self._instance.flush()
        self._op.flush()
        del self._instance
        del self._op
        if written != self.total:
            raise TrajectoryError(
                f"wrote {written} transitions but reserved {self.total}"
            )
        return written


class TransitionDataset:
    """Read-only view over a generated transition store."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        index_path = self.root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"missing transition index: {index_path}")
        self.index = json.loads(index_path.read_text())
        self.total = int(self.index["total_transitions"])
        self._maps = {}
        for key, (filename, dtype, width) in _ARRAY_FILES.items():
            self._maps[key] = np.memmap(
                self.root / filename, dtype=dtype, mode="r", shape=(self.total, width)
            )
        self.instance_id = np.memmap(
            self.root / "instance_id.u16", dtype=np.uint16, mode="r", shape=(self.total,)
        )
        self.op_id = np.memmap(
            self.root / "op_id.u16", dtype=np.uint16, mode="r", shape=(self.total,)
        )

    def __len__(self) -> int:
        return self.total

    @property
    def n_source_puzzles(self) -> int:
        return int(self.index["n_source_puzzles"])

    def batch(self, indices: Sequence[int] | np.ndarray) -> Dict[str, np.ndarray]:
        """Gather a batch of transitions with control labels unpacked to 0/1."""
        idx = np.asarray(indices, dtype=np.int64)
        out = {
            "x_k": np.asarray(self._maps["x_k"][idx], dtype=np.int64),
            "y_star": np.asarray(self._maps["y_star"][idx], dtype=np.int64),
        }
        for key in ("r_star", "e_star", "c_star"):
            packed = np.asarray(self._maps[key][idx])
            out[key] = np.unpackbits(packed, axis=1)[:, : vocab.SEQUENCE_LENGTH].astype(
                np.int64
            )
        out["instance_id"] = np.asarray(self.instance_id[idx], dtype=np.int64)
        out["op_id"] = np.asarray(self.op_id[idx], dtype=np.int64)
        return out

    def file_hashes(self) -> Dict[str, str]:
        names = [f for f, _, _ in _ARRAY_FILES.values()] + [
            "instance_id.u16",
            "op_id.u16",
        ]
        return {name: sha256_file(self.root / name) for name in sorted(names)}


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------


def accounting_from_counts(
    per_puzzle: Sequence[int], operation_totals: Dict[str, int]
) -> dict:
    """Summarise transition counts per source puzzle.

    The returned dictionary deliberately labels every aggregate as
    *solver-derived transitions*; the only "sample" count that refers to
    independent instances is ``n_source_puzzles``.
    """
    counts = np.asarray(per_puzzle, dtype=np.int64)
    return {
        "n_source_puzzles": int(counts.size),
        "total_solver_derived_transitions": int(counts.sum()),
        "transitions_per_source_puzzle_mean": float(counts.mean()),
        "transitions_per_source_puzzle_std": float(counts.std(ddof=0)),
        "transitions_per_source_puzzle_min": int(counts.min()),
        "transitions_per_source_puzzle_max": int(counts.max()),
        "transitions_per_source_puzzle_median": float(np.median(counts)),
        "transitions_per_source_puzzle": counts.tolist(),
        "operation_totals": dict(sorted(operation_totals.items())),
        "note": (
            "Transitions are deterministic expansions of the source puzzles. "
            "They are not independent puzzles and not independent samples."
        ),
    }


def build_transition_store(
    puzzles: np.ndarray,
    out_dir: str | Path | None = None,
    tag: str = "sudoku-100",
    validate: bool = True,
    initial_states: np.ndarray | None = None,
    progress: bool = False,
) -> dict:
    """Generate, validate, account and store transitions for all source puzzles.

    Returns the index dictionary that is also written to ``index.json``.
    """
    root = Path(out_dir) if out_dir is not None else transitions_dir(create=True) / tag
    root = assert_outside_repo(root)

    puzzle_arr = np.asarray(puzzles)
    n_puzzles = int(puzzle_arr.shape[0])

    generated: List[PuzzleTransitions] = []
    reports: List[ValidationReport] = []
    started = time.time()
    for i in range(n_puzzles):
        tr = generate_puzzle_transitions(puzzle_arr[i], i)
        if validate:
            initial = None if initial_states is None else initial_states[i]
            report = validate_puzzle_transitions(tr, initial_state=initial)
            if not report.ok:
                raise TrajectoryError(
                    f"puzzle {i} failed validation: {report.errors[:4]}"
                )
            reports.append(report)
        # x_next is only needed for validation; drop it so that generating all
        # 100 trajectories stays within a sane memory footprint.
        tr.x_next = np.empty((0, vocab.SEQUENCE_LENGTH), dtype=np.uint8)
        generated.append(tr)
        if progress:
            print(f"  puzzle {i:3d}: {tr.count:6d} transitions ({tr.seconds:.2f}s)", flush=True)

    total = sum(tr.count for tr in generated)
    writer = TransitionWriter(root, total)
    spans = []
    for tr in generated:
        spans.append(writer.append(tr))
    writer.close()

    operation_totals: Dict[str, int] = {name: 0 for name in OPERATION_NAMES}
    for tr in generated:
        for name, value in tr.operation_counts().items():
            operation_totals[name] += value

    per_puzzle = [tr.count for tr in generated]
    accounting = accounting_from_counts(per_puzzle, operation_totals)

    index = {
        "tag": tag,
        "generator": "dataset/sudoku/sudoku_generator.py (unmodified)",
        "sequence_length": vocab.SEQUENCE_LENGTH,
        "control_bytes": CONTROL_BYTES,
        "operation_names": list(OPERATION_NAMES),
        "total_transitions": total,
        "n_source_puzzles": n_puzzles,
        "spans": [[int(a), int(b)] for a, b in spans],
        "accounting": accounting,
        "validated": bool(validate),
        "validation_summary": (
            {
                "all_g_consistent": all(
                    r.g_consistent == r.n_transitions for r in reports
                ),
                "all_chains_continuous": all(
                    r.chain_continuous == r.n_transitions - 1 for r in reports
                ),
                "all_tokens_legal": all(r.tokens_legal for r in reports),
                "all_controls_binary": all(r.control_binary for r in reports),
                "total_insert_labels_set": sum(r.insert_labels_nonzero for r in reports),
                "total_delete_labels_set": sum(r.delete_labels_nonzero for r in reports),
                "total_remask_labels_set": sum(r.remask_labels_nonzero for r in reports),
                "total_unmask_target_positions": sum(
                    r.unmask_target_positions for r in reports
                ),
            }
            if validate
            else {}
        ),
        "generation_seconds": round(time.time() - started, 3),
    }
    (root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))

    dataset = TransitionDataset(root)
    index["file_sha256"] = dataset.file_hashes()
    index["index_sha256"] = sha256_json(
        {k: v for k, v in index.items() if k != "index_sha256"}
    )
    (root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def deterministic_split(
    total: int, train_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic transition-level split.

    The held-out part is an *in-distribution monitor* drawn from the same 100
    source puzzles.  It measures optimisation progress only; the sole held-out
    generalisation set in this replication is the official 10,000-puzzle array.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    cut = int(total * train_ratio)
    return np.sort(perm[:cut]), np.sort(perm[cut:])
