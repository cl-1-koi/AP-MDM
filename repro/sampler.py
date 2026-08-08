"""Prompt-conditioned AP-MDM iterative generation (paper Algorithm 1).

The upstream sampler ``train/diffusion.py::_sample_apmdm`` starts from the
all-``[MASK]`` prior of length ``model.length`` and stops on an ``[EOS]`` token.
Neither is usable for Sudoku: Algorithm 1 in the paper takes an *input prompt*
(``x_0 <- x``), and the Sudoku generator never emits ``[EOS]``.  This module
reconstructs Algorithm 1 faithfully:

* generation starts from the encoded puzzle state,
* each step performs one forward pass and applies ``g`` (see
  :mod:`repro.transition`) with thresholds ``tau_r``, ``tau_i``, ``tau_d``,
* a sequence terminates when it reaches a **fixed point** (``x_{t+1} == x_t``)
  or when the frozen maximum-step ceiling is hit.

Rows are grouped by current length so that insert/delete operations - which the
Sudoku supervision never exercises - are handled exactly rather than silently
padded away.  Any length change is recorded as an anomaly.  The
length-preserving case (the only one Sudoku produces) is fully vectorised,
because inference cost dominates the production budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from repro import vocab
from repro.losses import subs_argmax
from repro.transition import apply_transition

TERMINATED_FIXED_POINT = "fixed_point"
TERMINATED_MAX_STEPS = "max_steps"
TERMINATED_LENGTH_OVERFLOW = "length_overflow"
TERMINATED_EMPTY = "empty_sequence"

_STATUS_BY_CODE = {
    0: TERMINATED_MAX_STEPS,
    1: TERMINATED_FIXED_POINT,
    2: TERMINATED_LENGTH_OVERFLOW,
    3: TERMINATED_EMPTY,
}


@dataclass(frozen=True)
class SamplerConfig:
    """Frozen inference settings.

    All thresholds must be fixed before production evaluation and are recorded
    in the sealed manifest.  They are never selected on the 10,000-puzzle
    verdict set.
    """

    tau_remask: float = 0.5
    tau_insert: float = 0.5
    tau_delete: float = 0.5
    max_steps: int = 65536
    mask_index: int = vocab.MASK

    def as_dict(self) -> dict:
        return {
            "tau_remask": self.tau_remask,
            "tau_insert": self.tau_insert,
            "tau_delete": self.tau_delete,
            "max_steps": self.max_steps,
            "mask_index": self.mask_index,
            "termination": "fixed point (x_{t+1} == x_t) or max_steps",
        }


@dataclass
class RowTrace:
    """Per-sequence generation telemetry."""

    steps: int = 0
    forward_passes: int = 0
    status: str = TERMINATED_MAX_STEPS
    n_remask: int = 0
    n_unmask: int = 0
    n_insert: int = 0
    n_delete: int = 0
    length_changes: int = 0
    final_length: int = 0
    first_complete_step: int = -1

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "forward_passes": self.forward_passes,
            "status": self.status,
            "n_remask_ops": self.n_remask,
            "n_unmask_ops": self.n_unmask,
            "n_insert_ops": self.n_insert,
            "n_delete_ops": self.n_delete,
            "length_changes": self.length_changes,
            "final_length": self.final_length,
            "first_complete_step": self.first_complete_step,
        }


@dataclass
class GenerationResult:
    """Result of generating for a batch of prompts."""

    states: List[np.ndarray]
    traces: List[RowTrace]
    total_forward_passes: int = 0
    total_steps: int = 0
    batch_seconds: float = 0.0
    peak_memory_bytes: int = 0
    extra: Dict[str, float] = field(default_factory=dict)


def _values_complete(state: torch.Tensor) -> torch.Tensor:
    """Per-row flag: every cell's value slot holds a digit 1..9."""
    values = state[:, vocab.SLOT_VALUE :: vocab.TOKENS_PER_CELL]
    return ((values >= vocab.DIGIT_MIN) & (values <= vocab.DIGIT_MAX)).all(dim=1)


@torch.no_grad()
def generate(
    model,
    initial_states: np.ndarray | torch.Tensor,
    config: SamplerConfig,
    device: Optional[torch.device] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    progress_every: int = 0,
) -> GenerationResult:
    """Run Algorithm 1 to a fixed point for a batch of prompt states."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    initial = torch.as_tensor(np.asarray(initial_states), dtype=torch.long)
    if initial.dim() != 2:
        raise ValueError("initial_states must be (batch, length)")
    batch, width = initial.shape

    buffer = torch.full((batch, model.max_length), config.mask_index, dtype=torch.long, device=device)
    buffer[:, :width] = initial.to(device)

    zeros = lambda: torch.zeros(batch, dtype=torch.long, device=device)  # noqa: E731
    lengths = torch.full((batch,), width, dtype=torch.long, device=device)
    active = torch.ones(batch, dtype=torch.bool, device=device)
    steps, forwards = zeros(), zeros()
    n_remask, n_unmask, n_insert, n_delete = zeros(), zeros(), zeros(), zeros()
    length_changes = zeros()
    status_code = zeros()  # 0 == max_steps
    first_complete = torch.full((batch,), -1, dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    total_forward = 0

    initial_complete = _values_complete(buffer[:, :width])
    first_complete[initial_complete] = 0

    for step in range(1, config.max_steps + 1):
        idx_active = torch.nonzero(active, as_tuple=False).reshape(-1)
        if idx_active.numel() == 0:
            break

        active_lengths = lengths[idx_active]
        for length in torch.unique(active_lengths).tolist():
            rows = idx_active[active_lengths == length]
            x = buffer[rows, :length]

            if autocast_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    outputs = model(x)
            else:
                outputs = model(x)
            total_forward += 1
            forwards[rows] += 1
            steps[rows] = step

            y = subs_argmax(outputs["unmasking_logits"], x, config.mask_index)
            r = torch.sigmoid(outputs["remasking_logits"].squeeze(-1).float()) > config.tau_remask
            e = torch.sigmoid(outputs["expansion_logits"].squeeze(-1).float()) > config.tau_insert
            d = torch.sigmoid(outputs["contraction_logits"].squeeze(-1).float()) > config.tau_delete

            is_mask = x == config.mask_index
            deletes = d & is_mask
            structural = e.any(dim=1) | deletes.any(dim=1)

            # ---- fast path: no insert/delete fires -> pure vectorised g ----
            plain = ~structural
            if bool(plain.any()):
                sel = torch.nonzero(plain, as_tuple=False).reshape(-1)
                global_rows = rows[sel]
                xs, rs, ys = x[sel], r[sel], y[sel]
                kept = torch.where(xs == config.mask_index, ys, xs)
                z = torch.where(rs, torch.full_like(kept, config.mask_index), kept)
                buffer[global_rows, :length] = z
                n_remask[global_rows] += rs.sum(dim=1)
                n_unmask[global_rows] += ((xs == config.mask_index) & ~rs).sum(dim=1)
                unchanged = (z == xs).all(dim=1)
                done = global_rows[unchanged]
                active[done] = False
                status_code[done] = 1  # fixed point

            # ---- exact path: insert and/or delete fired (off-distribution) ----
            for local in torch.nonzero(structural, as_tuple=False).reshape(-1).tolist():
                row = int(rows[local])
                x_np = x[local].detach().cpu().numpy()
                z_np = apply_transition(
                    x_np,
                    y[local].detach().cpu().numpy(),
                    r[local].detach().cpu().numpy().astype(np.int64),
                    e[local].detach().cpu().numpy().astype(np.int64),
                    d[local].detach().cpu().numpy().astype(np.int64),
                    mask_token=config.mask_index,
                )
                n_remask[row] += int(r[local].sum())
                n_unmask[row] += int(((x[local] == config.mask_index) & ~r[local]).sum())
                n_insert[row] += int(e[local].sum())
                n_delete[row] += int(deletes[local].sum())
                new_length = int(z_np.size)
                if new_length != length:
                    length_changes[row] += 1
                if new_length == 0:
                    status_code[row] = 3
                    active[row] = False
                    lengths[row] = 0
                    continue
                if new_length > model.max_length:
                    status_code[row] = 2
                    active[row] = False
                    continue
                changed = new_length != length or not np.array_equal(z_np, x_np)
                buffer[row, :new_length] = torch.as_tensor(z_np, dtype=torch.long, device=device)
                if new_length < length:
                    buffer[row, new_length:length] = config.mask_index
                lengths[row] = new_length
                if not changed:
                    status_code[row] = 1
                    active[row] = False

        # First step at which a standard-width row's grid became value-complete.
        pending = (first_complete < 0) & (lengths == width)
        if bool(pending.any()):
            rows = torch.nonzero(pending, as_tuple=False).reshape(-1)
            complete = _values_complete(buffer[rows, :width])
            first_complete[rows[complete]] = step

        if progress_every and step % progress_every == 0:
            print(
                f"    step {step}: {int(active.sum())}/{batch} sequences still active",
                flush=True,
            )

    seconds = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

    lengths_cpu = lengths.tolist()
    traces = [
        RowTrace(
            steps=int(s),
            forward_passes=int(f),
            status=_STATUS_BY_CODE[int(code)],
            n_remask=int(rm),
            n_unmask=int(um),
            n_insert=int(ins),
            n_delete=int(dele),
            length_changes=int(lc),
            final_length=int(ln),
            first_complete_step=int(fc),
        )
        for s, f, code, rm, um, ins, dele, lc, ln, fc in zip(
            steps.tolist(),
            forwards.tolist(),
            status_code.tolist(),
            n_remask.tolist(),
            n_unmask.tolist(),
            n_insert.tolist(),
            n_delete.tolist(),
            length_changes.tolist(),
            lengths_cpu,
            first_complete.tolist(),
        )
    ]
    states = [
        buffer[row, : lengths_cpu[row]].detach().cpu().numpy().astype(np.uint8)
        for row in range(batch)
    ]

    if was_training:
        model.train()

    return GenerationResult(
        states=states,
        traces=traces,
        total_forward_passes=total_forward,
        total_steps=int(steps.max()) if batch else 0,
        batch_seconds=seconds,
        peak_memory_bytes=peak,
    )


def symbolic_trajectory_steps(puzzles: Sequence[np.ndarray]) -> List[int]:
    """Number of solver transitions the symbolic generator needs per puzzle.

    Used only to justify the frozen ``max_steps`` ceiling from the **training**
    puzzles and to project evaluation cost - never to tune anything on the
    verdict set.
    """
    from repro.trajectories import generate_puzzle_transitions

    return [generate_puzzle_transitions(p, i).count for i, p in enumerate(puzzles)]
