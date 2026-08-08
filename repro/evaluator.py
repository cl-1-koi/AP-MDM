"""Exact Sudoku evaluator: immutable per-puzzle rows and deterministic aggregates.

Evaluation semantics required by the replication contract, recorded for every
held-out puzzle:

* exact equality with the supplied ground-truth grid (primary metric),
* validity under Sudoku row/column/sub-grid constraints,
* consistency with the givens,
* solved/terminated, timeout, invalid token/shape, maximum generation steps,
* number and type of remask/insert/delete/unmask operations and forward passes.

The symbolic backtracking data generator is *never* scored as model accuracy;
this module only ever consumes states produced by a model checkpoint.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from repro import data as data_mod
from repro import vocab
from repro.hashing import sha256_file, sha256_json
from repro.sampler import (
    TERMINATED_FIXED_POINT,
    TERMINATED_MAX_STEPS,
    RowTrace,
)

Z_95 = 1.959963984540054


# --------------------------------------------------------------------------
# Binomial intervals
# --------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == total else min(1.0, centre + half)
    return (lower, upper)


def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps, max_iter = 1e-300, 3e-16, 400
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``; deterministic, no SciPy."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def _beta_quantile(p: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial interval."""
    if total <= 0:
        return (float("nan"), float("nan"))
    lower = 0.0 if successes == 0 else _beta_quantile(alpha / 2, successes, total - successes + 1)
    upper = 1.0 if successes == total else _beta_quantile(1 - alpha / 2, successes + 1, total - successes)
    return (lower, upper)


def proportion_report(successes: int, total: int) -> Dict[str, float]:
    """Point estimate plus both 95% binomial intervals."""
    wl, wu = wilson_interval(successes, total)
    cl, cu = clopper_pearson_interval(successes, total)
    return {
        "successes": int(successes),
        "total": int(total),
        "estimate": successes / total if total else float("nan"),
        "wilson_95_lower": wl,
        "wilson_95_upper": wu,
        "clopper_pearson_95_lower": cl,
        "clopper_pearson_95_upper": cu,
    }


# --------------------------------------------------------------------------
# Per-puzzle rows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PuzzleRow:
    """Immutable evaluation record for one held-out puzzle."""

    puzzle_index: int
    n_givens: int
    exact_match: bool
    valid: bool
    consistent_with_givens: bool
    complete: bool
    solved: bool
    malformed: bool
    malformed_reason: str
    timeout: bool
    status: str
    steps: int
    forward_passes: int
    n_remask_ops: int
    n_unmask_ops: int
    n_insert_ops: int
    n_delete_ops: int
    length_changes: int
    final_length: int
    first_complete_step: int
    n_masked_values: int
    n_empty_values: int
    predicted_grid_sha256: str

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_state(
    puzzle_index: int,
    puzzle: np.ndarray,
    solution: np.ndarray,
    state: np.ndarray,
    trace: RowTrace,
) -> PuzzleRow:
    """Score one generated state against its ground truth."""
    from repro.hashing import sha256_array

    decoded = data_mod.decode_state(state)
    valid = (not decoded.malformed) and data_mod.is_valid_grid(decoded.grid)
    consistent = (not decoded.malformed) and data_mod.is_consistent_with_givens(puzzle, decoded.grid)
    exact = (
        (not decoded.malformed)
        and decoded.complete
        and bool(np.array_equal(decoded.grid, np.asarray(solution)))
    )
    timeout = trace.status == TERMINATED_MAX_STEPS
    return PuzzleRow(
        puzzle_index=int(puzzle_index),
        n_givens=int((np.asarray(puzzle) != 0).sum()),
        exact_match=bool(exact),
        valid=bool(valid),
        consistent_with_givens=bool(consistent),
        complete=bool(decoded.complete),
        solved=bool(trace.status == TERMINATED_FIXED_POINT and decoded.complete),
        malformed=bool(decoded.malformed),
        malformed_reason=decoded.malformed_reason,
        timeout=bool(timeout),
        status=trace.status,
        steps=int(trace.steps),
        forward_passes=int(trace.forward_passes),
        n_remask_ops=int(trace.n_remask),
        n_unmask_ops=int(trace.n_unmask),
        n_insert_ops=int(trace.n_insert),
        n_delete_ops=int(trace.n_delete),
        length_changes=int(trace.length_changes),
        final_length=int(trace.final_length),
        first_complete_step=int(trace.first_complete_step),
        n_masked_values=int(decoded.n_masked_values),
        n_empty_values=int(decoded.n_empty_values),
        predicted_grid_sha256=sha256_array(decoded.grid),
    )


def evaluate_states(
    puzzle_indices: Sequence[int],
    puzzles: np.ndarray,
    solutions: np.ndarray,
    states: Sequence[np.ndarray],
    traces: Sequence[RowTrace],
) -> List[PuzzleRow]:
    """Score a batch of generated states."""
    if not (len(puzzle_indices) == len(states) == len(traces)):
        raise ValueError("puzzle_indices, states and traces must have equal length")
    return [
        evaluate_state(idx, puzzles[idx], solutions[idx], states[i], traces[i])
        for i, idx in enumerate(puzzle_indices)
    ]


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

#: Predeclared replication verdict thresholds (exact accuracy on 10,000 puzzles).
PAPER_TARGET = 0.9928
STRONG_REPLICATION_MIN = 0.9878
PARTIAL_REPLICATION_MIN = 0.9700


def verdict_label(exact_accuracy: float, invalidating_failure: bool) -> str:
    """Map a measured exact accuracy to the predeclared verdict label."""
    if invalidating_failure:
        return "invalid/inconclusive"
    if exact_accuracy >= STRONG_REPLICATION_MIN:
        return "strong replication"
    if exact_accuracy >= PARTIAL_REPLICATION_MIN:
        return "partial replication"
    return "failure to replicate"


def aggregate(rows: Sequence[PuzzleRow], invalidating_failure: bool = False) -> dict:
    """Deterministically aggregate per-puzzle rows.

    Rows are sorted by ``puzzle_index`` first, so the result does not depend on
    batch order or on how many shards the evaluation was split into.
    """
    ordered = sorted(rows, key=lambda r: r.puzzle_index)
    indices = [r.puzzle_index for r in ordered]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate puzzle_index values in evaluation rows")

    total = len(ordered)
    exact = sum(r.exact_match for r in ordered)
    valid = sum(r.valid for r in ordered)
    consistent = sum(r.consistent_with_givens for r in ordered)
    complete = sum(r.complete for r in ordered)
    solved = sum(r.solved for r in ordered)
    malformed = sum(r.malformed for r in ordered)
    timeouts = sum(r.timeout for r in ordered)

    steps = np.array([r.steps for r in ordered], dtype=np.int64)
    forwards = np.array([r.forward_passes for r in ordered], dtype=np.int64)
    exact_report = proportion_report(exact, total)
    return {
        "n_puzzles": total,
        "exact_accuracy": exact_report,
        "validity": proportion_report(valid, total),
        "givens_consistency": proportion_report(consistent, total),
        "completeness": proportion_report(complete, total),
        "terminated_solved": proportion_report(solved, total),
        "malformed": int(malformed),
        "timeouts": int(timeouts),
        "status_counts": _counter([r.status for r in ordered]),
        "operations": {
            "total_remask": int(sum(r.n_remask_ops for r in ordered)),
            "total_unmask": int(sum(r.n_unmask_ops for r in ordered)),
            "total_insert": int(sum(r.n_insert_ops for r in ordered)),
            "total_delete": int(sum(r.n_delete_ops for r in ordered)),
            "length_changes": int(sum(r.length_changes for r in ordered)),
        },
        "steps": _int_stats(steps),
        "forward_passes": _int_stats(forwards),
        "paper_target": PAPER_TARGET,
        "verdict": verdict_label(exact_report["estimate"], invalidating_failure),
        "verdict_thresholds": {
            "strong_replication_min": STRONG_REPLICATION_MIN,
            "partial_replication_min": PARTIAL_REPLICATION_MIN,
        },
        "rows_sha256": sha256_json([r.as_dict() for r in ordered]),
    }


def _counter(values: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _int_stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "total": 0}
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "total": int(arr.sum()),
    }


# --------------------------------------------------------------------------
# Immutable persistence
# --------------------------------------------------------------------------


class ImmutableArtifactError(RuntimeError):
    """Raised when an attempt is made to overwrite an evaluation artefact."""


def write_rows(path: str | Path, rows: Sequence[PuzzleRow]) -> str:
    """Write per-puzzle rows as JSONL; refuses to overwrite. Returns the digest."""
    target = Path(path)
    if target.exists():
        raise ImmutableArtifactError(f"refusing to overwrite existing rows file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: r.puzzle_index)
    with open(target, "w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row.as_dict(), sort_keys=True) + "\n")
    return sha256_file(target)


def read_rows(path: str | Path) -> List[PuzzleRow]:
    """Read per-puzzle rows back from JSONL."""
    rows: List[PuzzleRow] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(PuzzleRow(**json.loads(line)))
    return rows


# --------------------------------------------------------------------------
# Pre-flight gates
# --------------------------------------------------------------------------


class EvaluationGateError(RuntimeError):
    """Raised when the evaluator's pre-flight checks fail."""


def check_evaluation_preconditions(
    manifest: dict,
    checkpoint_meta: dict,
    overlap: data_mod.OverlapReport,
) -> None:
    """Reject leakage, mismatched manifests and mismatched checkpoints."""
    if not overlap.clean:
        raise EvaluationGateError(f"train/test overlap detected: {overlap.as_dict()}")

    for field_name in ("architecture_signature", "vocabulary_size", "manifest_sha256"):
        if field_name not in checkpoint_meta:
            raise EvaluationGateError(f"checkpoint metadata missing {field_name!r}")

    expected_signature = manifest["architecture"]["signature"]
    if checkpoint_meta["architecture_signature"] != expected_signature:
        raise EvaluationGateError(
            "checkpoint architecture does not match the manifest: "
            f"{checkpoint_meta['architecture_signature']!r} != {expected_signature!r}"
        )
    if int(checkpoint_meta["vocabulary_size"]) != int(manifest["vocabulary"]["effective_size"]):
        raise EvaluationGateError("checkpoint vocabulary size does not match the manifest")
    if checkpoint_meta["manifest_sha256"] != manifest["seal"]["manifest_sha256"]:
        raise EvaluationGateError(
            "checkpoint was produced under a different sealed manifest"
        )
