"""Append-only JSONL telemetry.

Telemetry is written outside the Git work tree and is adequate to diagnose:

* operation-head collapse (per-head predicted vs. target positive rates,
  precision/recall, probability mean/max),
* non-emission (zero predicted positives while targets are positive),
* looping and non-termination (sampler step/forward-pass distributions),
* invalid sequences (malformed-state and length-change counters),
* loss components, throughput, memory, checkpoint progress, accuracy curves.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


class TelemetryWriter:
    """Append-only JSONL writer with flush-on-write semantics."""

    def __init__(self, path: str | Path, run_id: str, flush_every: int = 1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.flush_every = max(1, int(flush_every))
        self._handle = open(self.path, "a", encoding="utf-8")
        self._written = 0
        self._t0 = time.time()

    def write(self, kind: str, **fields: Any) -> Dict[str, Any]:
        record = {
            "ts": time.time(),
            "elapsed_s": round(time.time() - self._t0, 6),
            "run_id": self.run_id,
            "kind": kind,
            **fields,
        }
        self._handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        self._written += 1
        if self._written % self.flush_every == 0:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        return record

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(obj)


def read_telemetry(path: str | Path, kind: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Iterate telemetry records, optionally filtered by ``kind``."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if kind is None or record.get("kind") == kind:
                yield record


def gpu_snapshot(device=None) -> Dict[str, Any]:
    """Current CUDA memory / utilisation snapshot (safe on CPU-only hosts)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        dev = torch.device(device) if device is not None else torch.device(
            "cuda", torch.cuda.current_device()
        )
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        free, total = torch.cuda.mem_get_info(dev)
        return {
            "cuda": True,
            "device": str(dev),
            "allocated_bytes": int(torch.cuda.memory_allocated(dev)),
            "reserved_bytes": int(torch.cuda.memory_reserved(dev)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
            "free_bytes": int(free),
            "total_bytes": int(total),
        }
    except Exception as exc:  # pragma: no cover
        return {"cuda": False, "error": str(exc)}
