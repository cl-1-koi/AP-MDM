#!/usr/bin/env python3
"""Token-free supervision and artifact syncing for Sudoku VC-1."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SSH_KEY = "/home/ubuntu/.ssh/id_ed25519"
STATUS_ROOT = Path("/home/ubuntu/apmdm-official-data/sudoku-vc1-supervisor")


@dataclass(frozen=True)
class Workload:
    condition_mode: str
    pod_id: str
    host: str
    port: int
    hourly_cost_usd: float
    tmux_session: str
    remote_output: str
    run_id: str
    local_parent: str
    queued_next_experiment: str
    retain_rationale: str

    @property
    def remote_run(self) -> str:
        return f"{self.remote_output}/data/runs/{self.run_id}"

    @property
    def local_output(self) -> Path:
        return Path(self.local_parent) / Path(self.remote_output).name

    @property
    def local_run(self) -> Path:
        return self.local_output / "data" / "runs" / self.run_id


WORKLOADS = (
    Workload(
        condition_mode="aligned_solution_hint",
        pod_id="82i4fwoch34p40",
        host="64.247.206.218",
        port=11960,
        hourly_cost_usd=0.99,
        tmux_session="vc1_aligned_full",
        remote_output="/workspace/artifacts/sudoku-vc1/full-aligned-c1c458e-s42",
        run_id="vc1-aligned_solution_hint-s42-u78100",
        local_parent="/home/ubuntu/apmdm-official-data/sudoku-vc1/aligned",
        queued_next_experiment="keyed shuffled-answer control, then visible-answer star length scaling",
        retain_rationale="retain high-value 32-vCPU L40S for declared keyed-reordering follow-up",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(arguments: list[str], timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments, check=False, capture_output=True, text=True, timeout=timeout
    )


def ssh(workload: Workload, remote: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return command(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-i", SSH_KEY, "-p", str(workload.port),
            f"root@{workload.host}", remote,
        ],
        timeout=timeout,
    )


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def active_inventory() -> dict[str, Any]:
    result = command(["runpod", "pod", "list"], timeout=30)
    active = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "RUNNING" in line:
                match = re.search(r"\|\s*([a-z0-9]{14})\s*\|", line)
                if match:
                    active.append(match.group(1))
    return {
        "ok": result.returncode == 0,
        "active_pod_ids": sorted(active),
        "error": result.stderr.strip() if result.returncode else None,
    }


def parse_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = record.get("kind")
        if isinstance(kind, str):
            key = f"eval:{record.get('step')}" if kind == "eval" else kind
            latest[key] = record
    return latest


def sync(workload: Workload) -> dict[str, Any]:
    parent = Path(workload.local_parent)
    parent.mkdir(parents=True, exist_ok=True)
    result = command(
        [
            "rsync", "-a", "--partial", "--timeout=60",
            "-e", f"ssh -o BatchMode=yes -o ConnectTimeout=10 -i {SSH_KEY} -p {workload.port}",
            f"root@{workload.host}:{workload.remote_output}", f"{parent}/",
        ],
        timeout=240,
    )
    runner_remote = f"{workload.remote_output}.runner.log"
    runner = command(
        [
            "rsync", "-a", "--partial", "--timeout=60",
            "-e", f"ssh -o BatchMode=yes -o ConnectTimeout=10 -i {SSH_KEY} -p {workload.port}",
            f"root@{workload.host}:{runner_remote}", f"{parent}/",
        ],
        timeout=120,
    )
    return {
        "ok": result.returncode == 0 and runner.returncode == 0,
        "output_error": result.stderr.strip()[-1000:] if result.returncode else None,
        "runner_error": runner.stderr.strip()[-1000:] if runner.returncode else None,
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_hashes(workload: Workload) -> dict[str, str]:
    parent = Path(workload.local_parent)
    paths = []
    if workload.local_output.exists():
        paths.extend(path for path in workload.local_output.rglob("*") if path.is_file())
    runner = parent / f"{Path(workload.remote_output).name}.runner.log"
    if runner.is_file():
        paths.append(runner)
    return {str(path.relative_to(parent)): file_hash(path) for path in sorted(paths)}


def remote_hashes(workload: Workload) -> tuple[dict[str, str], str | None]:
    parent = str(Path(workload.remote_output).parent)
    output = shlex.quote(Path(workload.remote_output).name)
    runner = shlex.quote(f"{Path(workload.remote_output).name}.runner.log")
    result = ssh(
        workload,
        f"cd {shlex.quote(parent)} && find {output} {runner} -type f -print0 | sort -z | xargs -0 sha256sum",
        timeout=240,
    )
    if result.returncode:
        return {}, result.stderr.strip()
    rows = {}
    for line in result.stdout.splitlines():
        digest, separator, path = line.partition("  ")
        if separator and len(digest) == 64:
            rows[path] = digest
    return rows, None


def expected_steps() -> list[int]:
    return list(range(5_000, 78_100, 5_000)) + [78_100]


def poll(workload: Workload) -> dict[str, Any]:
    control = ssh(
        workload,
        f"tmux has-session -t {shlex.quote(workload.tmux_session)} 2>/dev/null; status=$?; pgrep -af 'run_sudoku_vc1_runpod|repro.cli insertion-train' || true; exit $status",
    )
    sync_result = sync(workload)
    latest = parse_latest(workload.local_run / "telemetry.jsonl")
    present = []
    for path in workload.local_run.glob("checkpoints/checkpoint_step_*.pt"):
        match = re.search(r"checkpoint_step_(\d+)\.pt$", path.name)
        if match:
            present.append(int(match.group(1)))
    present.sort()
    completion = workload.local_output / "completion.json"
    eligible = completion.is_file() and not (set(expected_steps()) - set(present))
    closure: dict[str, Any] = {"eligible": eligible, "verified": False}
    if eligible and sync_result["ok"]:
        remote, error = remote_hashes(workload)
        local = local_hashes(workload)
        closure.update(
            {
                "verified": bool(remote and remote == local),
                "remote_error": error,
                "remote_file_count": len(remote),
                "local_file_count": len(local),
                "missing_local": sorted(set(remote) - set(local)),
                "extra_local": sorted(set(local) - set(remote)),
                "hash_mismatches": sorted(
                    name for name in set(remote) & set(local) if remote[name] != local[name]
                ),
            }
        )
    return {
        "pod_id": workload.pod_id,
        "condition_mode": workload.condition_mode,
        "checked_utc": utc_now(),
        "tmux_alive": control.returncode == 0,
        "processes": control.stdout.splitlines(),
        "latest": latest,
        "present_checkpoint_steps": present,
        "missing_checkpoint_steps": sorted(set(expected_steps()) - set(present)),
        "sync": sync_result,
        "artifact_closure": closure,
        "queued_next_experiment": workload.queued_next_experiment,
        "retain_rationale": workload.retain_rationale,
    }


def poll_once() -> dict[str, Any]:
    snapshot = {
        "schema": "apmdm/sudoku-vc1-supervisor-v1",
        "checked_utc": utc_now(),
        "hourly_burn_usd": sum(workload.hourly_cost_usd for workload in WORKLOADS),
        "inventory": active_inventory(),
        "workloads": [poll(workload) for workload in WORKLOADS],
    }
    atomic_json(STATUS_ROOT / "status.json", snapshot)
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    with (STATUS_ROOT / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return snapshot


def main() -> None:
    while True:
        print(json.dumps(poll_once(), sort_keys=True), flush=True)
        time.sleep(300)


if __name__ == "__main__":
    main()
