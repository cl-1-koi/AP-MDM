#!/usr/bin/env python3
"""Durable, token-free supervision and artifact syncing for the G1 panel.

This watcher never launches or terminates pods. It records account-level
inventory, semantic telemetry progress, remote process/session state, and
durable-copy status. Completed runs receive a byte-for-byte remote/local hash
comparison before they can be considered artifact-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SSH_KEY = "/home/ubuntu/.ssh/id_ed25519"
STATUS_ROOT = Path("/home/ubuntu/apmdm-official-data/ip-star-supervisor")


@dataclass(frozen=True)
class Workload:
    arm: str
    pod_id: str
    host: str
    port: int
    tmux_session: str
    remote_parent: str
    run_name: str
    local_parent: str
    expected_final_step: int
    expected_checkpoint_interval: int
    hourly_cost_usd: float
    queued_next_experiment: str
    retain_rationale: str

    @property
    def remote_run(self) -> str:
        return f"{self.remote_parent}/{self.run_name}"

    @property
    def remote_console(self) -> str:
        return f"{self.remote_run}.console.log"

    @property
    def local_run(self) -> Path:
        return Path(self.local_parent) / self.run_name


WORKLOADS = (
    Workload(
        arm="learned_ip",
        pod_id="82i4fwoch34p40",
        host="64.247.206.218",
        port=11960,
        tmux_session="ip_learned_full",
        remote_parent="/workspace/artifacts/ip-star",
        run_name="health-learned-ac97943-s42-1000",
        local_parent="/home/ubuntu/apmdm-official-data/ip-star-runpod-82i4fwoch34p40",
        expected_final_step=78_100,
        expected_checkpoint_interval=5_000,
        hourly_cost_usd=0.99,
        queued_next_experiment="G1 verdict, then GuacaMol capacity-matched smoke only if learned order wins",
        retain_rationale="retain for declared G1-to-G2 handoff; 32-vCPU L40S is high-value",
    ),
    Workload(
        arm="random_ao_ip",
        pod_id="2fz6v1qstmkiez",
        host="69.30.85.29",
        port=22104,
        tmux_session="ip_aoip_full",
        remote_parent="/workspace/artifacts/ip-star",
        run_name="aoip-9a5fc7c-s42",
        local_parent="/home/ubuntu/apmdm-official-data/ip-star-runpod-2fz6v1qstmkiez",
        expected_final_step=78_100,
        expected_checkpoint_interval=5_000,
        hourly_cost_usd=0.44,
        queued_next_experiment="official/predecessor hard-star XLNet audit or fixed-canvas AO-ARM reconstruction",
        retain_rationale="retain briefly for declared follow-up; ordinary A40 at $0.44/hr",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(arguments: list[str], timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ssh(workload: Workload, remote_command: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-i",
            SSH_KEY,
            "-p",
            str(workload.port),
            f"root@{workload.host}",
            remote_command,
        ],
        timeout=timeout,
    )


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_telemetry(text: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for raw_line in text.splitlines():
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("kind"), str):
            key = record["kind"]
            if key == "evaluation":
                key = f"evaluation:{record.get('split')}:{record.get('weights')}"
            latest[key] = record
    return latest


def active_inventory() -> dict[str, Any]:
    result = command(["runpod", "pod", "list"], timeout=30)
    active_ids: list[str] = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "RUNNING" in line:
                match = re.search(r"\|\s*([a-z0-9]{14})\s*\|", line)
                if match:
                    active_ids.append(match.group(1))
    return {
        "ok": result.returncode == 0,
        "active_pod_ids": sorted(active_ids),
        "error": result.stderr.strip() if result.returncode else None,
    }


def sync_artifacts(workload: Workload) -> dict[str, Any]:
    local_parent = Path(workload.local_parent)
    local_parent.mkdir(parents=True, exist_ok=True)
    sources = (workload.remote_run, workload.remote_console)
    results = []
    for source in sources:
        result = command(
            [
                "rsync",
                "-a",
                "--partial",
                "--timeout=60",
                "-e",
                f"ssh -o BatchMode=yes -o ConnectTimeout=10 -i {SSH_KEY} -p {workload.port}",
                f"root@{workload.host}:{source}",
                f"{local_parent}/",
            ],
            timeout=180,
        )
        results.append(
            {
                "source": source,
                "ok": result.returncode == 0,
                "error": result.stderr.strip()[-1_000:] if result.returncode else None,
            }
        )
    return {"ok": all(row["ok"] for row in results), "filesets": results}


def local_hashes(workload: Workload) -> dict[str, str]:
    parent = Path(workload.local_parent)
    paths = []
    if workload.local_run.exists():
        paths.extend(path for path in workload.local_run.rglob("*") if path.is_file())
    console = parent / f"{workload.run_name}.console.log"
    if console.is_file():
        paths.append(console)
    hashes: dict[str, str] = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(path.relative_to(parent))] = digest.hexdigest()
    return hashes


def remote_hashes(workload: Workload) -> tuple[dict[str, str], str | None]:
    run_name = shlex.quote(workload.run_name)
    console_name = shlex.quote(f"{workload.run_name}.console.log")
    remote = (
        f"cd {shlex.quote(workload.remote_parent)} && "
        f"find {run_name} {console_name} -type f -print0 | sort -z | xargs -0 sha256sum"
    )
    result = ssh(workload, remote, timeout=180)
    if result.returncode:
        return {}, result.stderr.strip()
    hashes = {}
    for line in result.stdout.splitlines():
        digest, separator, relative = line.partition("  ")
        if separator and len(digest) == 64:
            hashes[relative] = digest
    return hashes, None


def expected_checkpoint_steps(workload: Workload) -> list[int]:
    values = list(
        range(
            workload.expected_checkpoint_interval,
            workload.expected_final_step,
            workload.expected_checkpoint_interval,
        )
    )
    values.append(workload.expected_final_step)
    return values


def poll_workload(workload: Workload) -> dict[str, Any]:
    telemetry_path = f"{workload.remote_run}/telemetry.jsonl"
    telemetry = ssh(workload, f"tail -n 300 -- {shlex.quote(telemetry_path)}")
    control = ssh(
        workload,
        (
            f"tmux has-session -t {shlex.quote(workload.tmux_session)} 2>/dev/null; "
            "tmux_status=$?; "
            "pgrep -af 'python -m repro.ip_star_train' || true; "
            "exit $tmux_status"
        ),
    )
    latest = parse_telemetry(telemetry.stdout) if telemetry.returncode == 0 else {}
    sync = sync_artifacts(workload) if telemetry.returncode == 0 else {"ok": False}
    complete = latest.get("complete")
    checkpoint_files = sorted(workload.local_run.glob("checkpoints/step_*.pt"))
    present_steps = []
    for path in checkpoint_files:
        match = re.search(r"step_(\d+)\.pt$", path.name)
        if match:
            present_steps.append(int(match.group(1)))
    required = expected_checkpoint_steps(workload)
    missing_required = [step for step in required if step not in present_steps]

    closure: dict[str, Any] = {
        "eligible": bool(
            complete
            and int(complete.get("step", -1)) == workload.expected_final_step
            and not missing_required
            and sync.get("ok")
        ),
        "verified": False,
    }
    if closure["eligible"]:
        remote, remote_error = remote_hashes(workload)
        local = local_hashes(workload)
        closure.update(
            {
                "remote_error": remote_error,
                "remote_file_count": len(remote),
                "local_file_count": len(local),
                "verified": bool(remote and remote == local),
                "missing_local": sorted(set(remote) - set(local)),
                "extra_local": sorted(set(local) - set(remote)),
                "hash_mismatches": sorted(
                    path for path in set(remote) & set(local) if remote[path] != local[path]
                ),
            }
        )

    return {
        "arm": workload.arm,
        "pod_id": workload.pod_id,
        "checked_utc": utc_now(),
        "ssh_ok": telemetry.returncode == 0,
        "ssh_error": telemetry.stderr.strip()[-1_000:] if telemetry.returncode else None,
        "tmux_alive": control.returncode == 0,
        "training_processes": control.stdout.splitlines(),
        "latest": latest,
        "sync": sync,
        "present_checkpoint_steps": present_steps,
        "missing_required_checkpoint_steps": missing_required,
        "artifact_closure": closure,
        "queued_next_experiment": workload.queued_next_experiment,
        "decision": "retain",
        "retain_rationale": workload.retain_rationale,
    }


def poll_once() -> dict[str, Any]:
    snapshot = {
        "schema": "ip-star-supervisor-v1",
        "checked_utc": utc_now(),
        "hourly_burn_usd": sum(workload.hourly_cost_usd for workload in WORKLOADS),
        "inventory": active_inventory(),
        "workloads": [poll_workload(workload) for workload in WORKLOADS],
    }
    atomic_json(STATUS_ROOT / "status.json", snapshot)
    with (STATUS_ROOT / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    while True:
        snapshot = poll_once()
        print(json.dumps(snapshot, sort_keys=True), flush=True)
        if arguments.once:
            return
        time.sleep(max(arguments.interval, 30))


if __name__ == "__main__":
    main()
