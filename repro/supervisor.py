"""Fail-closed GPU supervisor.

Hard rules, enforced by refusing to run rather than by warning:

* **Never** create, resume or use RunPod or any other paid capacity.  Any
  RunPod marker in the environment aborts immediately.
* Only the existing local A10 and the optional existing ``a10-220`` host are
  permitted.
* The GPU must be authenticated as idle before a run starts and returned to
  idle afterwards.
* A wall-clock ceiling is enforced; the initial smoke ceiling is 15 combined
  A10 minutes.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: Hostnames permitted to run GPU work.
ALLOWED_HOSTS = {"a10-220"}

#: GPU model substrings permitted to run GPU work.
ALLOWED_GPU_SUBSTRINGS = ("A10",)

#: Environment markers that indicate paid/rented capacity.
FORBIDDEN_ENV_MARKERS = (
    "RUNPOD_POD_ID",
    "RUNPOD_API_KEY",
    "RUNPOD_POD_HOSTNAME",
    "RUNPOD_PUBLIC_IP",
    "RUNPOD_ENDPOINT_ID",
    "RUNPOD_GPU_COUNT",
    "VAST_CONTAINERLABEL",
    "LAMBDA_INSTANCE_ID",
    "COREWEAVE_NAMESPACE",
)

FORBIDDEN_PATHS = ("/runpod-volume", "/runpod")

#: Smoke ceiling in seconds (15 combined A10 minutes).
SMOKE_CEILING_SECONDS = 15 * 60

IDLE_UTILIZATION_PCT = 5
IDLE_MEMORY_MIB = 512


class SupervisorError(RuntimeError):
    """Raised whenever a fail-closed precondition is not met."""


@dataclass
class GpuState:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_pct: int
    compute_processes: List[str] = field(default_factory=list)

    @property
    def idle(self) -> bool:
        return (
            not self.compute_processes
            and self.memory_used_mib <= IDLE_MEMORY_MIB
            and self.utilization_pct <= IDLE_UTILIZATION_PCT
        )

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "utilization_pct": self.utilization_pct,
            "compute_processes": self.compute_processes,
            "idle": self.idle,
        }


def _nvidia_smi(args: List[str]) -> str:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise SupervisorError("nvidia-smi not found; refusing to run GPU work")
    result = subprocess.run([binary, *args], capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0:
        raise SupervisorError(f"nvidia-smi failed: {result.stderr.strip()}")
    return result.stdout


def query_gpus() -> List[GpuState]:
    """Query every visible GPU's identity, memory, utilisation and processes."""
    raw = _nvidia_smi(
        [
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    procs = _nvidia_smi(
        ["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]
    )
    uuids = _nvidia_smi(["--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    uuid_to_index = {}
    for line in uuids.strip().splitlines():
        if not line.strip():
            continue
        idx, uuid = [part.strip() for part in line.split(",")]
        uuid_to_index[uuid] = int(idx)

    by_index: Dict[int, List[str]] = {}
    for line in procs.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        idx = uuid_to_index.get(parts[0])
        if idx is not None:
            by_index.setdefault(idx, []).append(f"pid={parts[1]} mem={parts[2]}MiB")

    states: List[GpuState] = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        idx, name, total, used, util = [part.strip() for part in line.split(",")]
        index = int(idx)
        states.append(
            GpuState(
                index=index,
                name=name,
                memory_total_mib=int(total),
                memory_used_mib=int(used),
                utilization_pct=int(util),
                compute_processes=by_index.get(index, []),
            )
        )
    return states


def assert_no_paid_capacity() -> None:
    """Abort if any marker of rented/paid capacity is present."""
    present = [key for key in FORBIDDEN_ENV_MARKERS if os.environ.get(key)]
    if present:
        raise SupervisorError(
            f"refusing to run: paid-capacity environment markers present: {present}"
        )
    for path in FORBIDDEN_PATHS:
        if os.path.exists(path):
            raise SupervisorError(f"refusing to run: paid-capacity path present: {path}")
    host = platform.node().lower()
    if "runpod" in host or "vast" in host:
        raise SupervisorError(f"refusing to run: hostname {host!r} indicates rented capacity")


def assert_allowed_host(gpus: List[GpuState]) -> str:
    """Allow only the local A10 machine or the existing ``a10-220`` host."""
    host = platform.node()
    if host in ALLOWED_HOSTS:
        return host
    if not gpus:
        raise SupervisorError(f"no GPU visible on host {host!r}; refusing to run GPU work")
    for gpu in gpus:
        if not any(sub in gpu.name for sub in ALLOWED_GPU_SUBSTRINGS):
            raise SupervisorError(
                f"GPU {gpu.index} is {gpu.name!r}; only {ALLOWED_GPU_SUBSTRINGS} are permitted"
            )
    return host


def assert_gpus_idle(gpus: List[GpuState], indices: Optional[List[int]] = None) -> None:
    """Require the target GPUs to be idle."""
    targets = [g for g in gpus if indices is None or g.index in indices]
    if not targets:
        raise SupervisorError("no target GPU found")
    busy = [g.as_dict() for g in targets if not g.idle]
    if busy:
        raise SupervisorError(f"refusing to run: target GPUs are not idle: {busy}")


@dataclass
class SupervisorSession:
    """Record of one supervised GPU session."""

    host: str
    ceiling_seconds: float
    started: float
    gpus_before: List[dict]
    gpus_after: List[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    returned_to_idle: bool = False
    exceeded_ceiling: bool = False

    def as_dict(self) -> dict:
        return {
            "host": self.host,
            "ceiling_seconds": self.ceiling_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "gpus_before": self.gpus_before,
            "gpus_after": self.gpus_after,
            "returned_to_idle": self.returned_to_idle,
            "exceeded_ceiling": self.exceeded_ceiling,
        }


class GpuSession:
    """Context manager enforcing every fail-closed precondition and the ceiling."""

    def __init__(
        self,
        ceiling_seconds: float = SMOKE_CEILING_SECONDS,
        gpu_indices: Optional[List[int]] = None,
        require_idle: bool = True,
    ):
        self.ceiling_seconds = float(ceiling_seconds)
        self.gpu_indices = gpu_indices
        self.require_idle = require_idle
        self.session: Optional[SupervisorSession] = None

    def __enter__(self) -> SupervisorSession:
        assert_no_paid_capacity()
        gpus = query_gpus()
        host = assert_allowed_host(gpus)
        if self.require_idle:
            assert_gpus_idle(gpus, self.gpu_indices)
        self.session = SupervisorSession(
            host=host,
            ceiling_seconds=self.ceiling_seconds,
            started=time.time(),
            gpus_before=[g.as_dict() for g in gpus],
        )
        return self.session

    def check(self) -> None:
        """Raise if the wall-clock ceiling has been exceeded."""
        assert self.session is not None
        elapsed = time.time() - self.session.started
        if elapsed > self.ceiling_seconds:
            self.session.exceeded_ceiling = True
            raise SupervisorError(
                f"GPU ceiling exceeded: {elapsed:.1f}s > {self.ceiling_seconds:.1f}s"
            )

    def remaining(self) -> float:
        assert self.session is not None
        return max(0.0, self.ceiling_seconds - (time.time() - self.session.started))

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.session is not None
        self.session.elapsed_seconds = time.time() - self.session.started
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass
        for _ in range(10):
            gpus = query_gpus()
            targets = [
                g for g in gpus if self.gpu_indices is None or g.index in self.gpu_indices
            ]
            if all(g.idle for g in targets):
                self.session.returned_to_idle = True
                break
            time.sleep(1.0)
        self.session.gpus_after = [g.as_dict() for g in query_gpus()]


def preflight_report(ceiling_seconds: float = SMOKE_CEILING_SECONDS) -> dict:
    """Non-mutating check of every supervisor precondition."""
    report: dict = {"ceiling_seconds": ceiling_seconds}
    try:
        assert_no_paid_capacity()
        report["paid_capacity_markers"] = "absent"
    except SupervisorError as exc:
        report["paid_capacity_markers"] = str(exc)
        report["ok"] = False
        return report
    try:
        gpus = query_gpus()
        report["gpus"] = [g.as_dict() for g in gpus]
        report["host"] = assert_allowed_host(gpus)
        assert_gpus_idle(gpus)
        report["idle"] = True
        report["ok"] = True
    except SupervisorError as exc:
        report["error"] = str(exc)
        report["ok"] = False
    return report
