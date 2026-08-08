"""The AP-MDM state-transition function ``g`` (paper Algorithm 1).

Reference (``method.tex``, Algorithm 1 "Any-Process Generation")::

    z <- []
    for i in 1..|x|:
        if ctrl[i][3] == 1 and x[i] == MASK:   # delete / contraction
            continue
        if ctrl[i][1] == 1:                    # remask   (highest priority)
            append MASK
        elif x[i] == MASK:                     # unmask
            append y[i]
        else:                                  # keep
            append x[i]
        if ctrl[i][2] == 1:                    # insert / expansion
            append MASK

The same function is used in three places, which is the point of factoring it
out: to validate that the official generator's ``x_{k+1}`` really is
``g(x_k, y*, ctrl*)``, to drive inference, and to test the sampler.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from repro.vocab import MASK


def apply_transition(
    x: Sequence[int] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    remask: Sequence[int] | np.ndarray,
    insert: Sequence[int] | np.ndarray,
    delete: Sequence[int] | np.ndarray,
    mask_token: int = MASK,
) -> np.ndarray:
    """Apply ``g`` to a single sequence; returns the successor state.

    Raises:
        ValueError: if the control vectors do not all match ``len(x)``.
    """
    x_arr = np.asarray(x).reshape(-1)
    n = x_arr.size
    y_arr = np.asarray(y).reshape(-1)
    r_arr = np.asarray(remask).reshape(-1)
    e_arr = np.asarray(insert).reshape(-1)
    d_arr = np.asarray(delete).reshape(-1)
    for name, arr in (("y", y_arr), ("remask", r_arr), ("insert", e_arr), ("delete", d_arr)):
        if arr.size != n:
            raise ValueError(f"{name} has length {arr.size}, expected {n}")

    out: list[int] = []
    for i in range(n):
        if d_arr[i] and x_arr[i] == mask_token:
            continue
        if r_arr[i]:
            out.append(mask_token)
        elif x_arr[i] == mask_token:
            out.append(int(y_arr[i]))
        else:
            out.append(int(x_arr[i]))
        if e_arr[i]:
            out.append(mask_token)
    return np.asarray(out, dtype=x_arr.dtype)


def apply_transition_length_preserving(
    x: np.ndarray,
    y: np.ndarray,
    remask: np.ndarray,
    mask_token: int = MASK,
) -> np.ndarray:
    """Vectorised ``g`` for the case where no insert/delete control fires.

    Equivalent to :func:`apply_transition` with all-zero ``insert``/``delete``
    vectors, and used as the sampler's fast path.  Works on ``(..., L)`` arrays
    (NumPy or Torch tensors both satisfy the operations used here).
    """
    kept = np.where(x == mask_token, y, x)
    return np.where(remask.astype(bool), mask_token, kept)


def transition_changes_length(
    insert: np.ndarray, delete: np.ndarray, x: np.ndarray, mask_token: int = MASK
) -> bool:
    """True iff applying ``g`` would change the sequence length."""
    inserts = np.asarray(insert).astype(bool).sum()
    deletes = (np.asarray(delete).astype(bool) & (np.asarray(x) == mask_token)).sum()
    return bool(inserts != 0 or deletes != 0)
