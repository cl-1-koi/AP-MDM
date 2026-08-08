"""Deterministic hashing helpers used for provenance and sealing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """SHA-256 of an array's dtype, shape and C-contiguous buffer.

    Includes dtype and shape so that arrays which differ only in declared dtype
    (for example the checked-in train array is ``uint64`` while the test array
    is ``int64``) never collide.
    """
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(repr(contiguous.shape).encode("utf-8"))
    digest.update(b"|")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def sha256_arrays(arrays: Iterable[np.ndarray]) -> str:
    """SHA-256 over a sequence of arrays, order-sensitive."""
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(sha256_array(array).encode("ascii"))
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Canonical JSON encoding: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(obj: Any) -> str:
    """SHA-256 of the canonical JSON encoding of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
