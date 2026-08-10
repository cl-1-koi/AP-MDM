#!/usr/bin/env python3
"""Token-free RunPod inventory, progress, sync, and closure for the S4 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import runpod


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout="",
            stderr=f"command timed out after {timeout} seconds; will retry next cycle",
        )


def ssh_args(pod: dict[str, Any]) -> list[str]:
    return [
        "ssh", "-i", pod["ssh_key"], "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10", "-p", str(pod["port"]),
        f"root@{pod['host']}",
    ]


def remote_probe(pod: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    remote = workload["remote_output"]
    shell = (
        f"if test -s {remote}/completion.json; then echo complete; "
        f"elif test -s {remote}/failure.json; then echo failed; "
        "else echo running; fi; "
        f"tail -n 3 {remote}/telemetry.jsonl 2>/dev/null || true"
    )
    result = command([*ssh_args(pod), shell], timeout=25)
    lines = result.stdout.splitlines()
    return {
        "ssh_returncode": result.returncode,
        "state": lines[0] if result.returncode == 0 and lines else "unreachable",
        "telemetry_tail": lines[1:] if result.returncode == 0 else [],
        "error": result.stderr.strip()[-500:] if result.returncode else "",
    }


def sync_and_verify(pod: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    local = Path(workload["local_output"])
    local.mkdir(parents=True, exist_ok=True)
    source = (
        f"root@{pod['host']}:{workload['remote_output'].rstrip('/')}/"
    )
    result = command(
        [
            "rsync", "-a", "-e",
            f"ssh -i {pod['ssh_key']} -p {pod['port']} -o BatchMode=yes",
            source, str(local) + "/",
        ],
        timeout=1800,
    )
    if result.returncode:
        return {"verified": False, "error": result.stderr.strip()[-1000:]}
    completion_path = local / "completion.json"
    if not completion_path.is_file():
        return {"verified": False, "error": "completion.json missing after sync"}
    completion = json.loads(completion_path.read_text())
    checked = 0
    for item in completion.get("files", []):
        path = local / item["path"]
        if not path.is_file():
            return {"verified": False, "error": f"missing {item['path']}"}
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            return {"verified": False, "error": f"hash mismatch {item['path']}"}
        checked += 1
    closure = {
        "verified": True,
        "verified_utc": utc_now(),
        "files": checked,
        "completion_sha256": sha256(completion_path),
        "source_commit": completion.get("source_commit"),
        "step": completion.get("step"),
    }
    atomic_json(local / "artifact_closure.json", closure)
    return closure


def cycle(config: dict[str, Any]) -> dict[str, Any]:
    account = {pod["id"]: pod for pod in runpod.get_pods()}
    report: dict[str, Any] = {
        "schema": "apmdm/s4-panel-supervisor-v1",
        "updated_utc": utc_now(),
        "account_active_pods": sorted(
            pod_id for pod_id, pod in account.items()
            if pod.get("desiredStatus") == "RUNNING"
        ),
        "pods": [],
    }
    for pod in config["pods"]:
        account_pod = account.get(pod["pod_id"])
        pod_row: dict[str, Any] = {
            "pod_id": pod["pod_id"],
            "name": pod["name"],
            "account_state": account_pod.get("desiredStatus") if account_pod else "absent",
            "hourly_cost_usd": pod["hourly_cost_usd"],
            "queued_next_experiment": pod["queued_next_experiment"],
            "retain_or_terminate_rationale": pod["retain_or_terminate_rationale"],
            "workloads": [],
        }
        for workload in pod["workloads"]:
            probe = remote_probe(pod, workload)
            row = {"name": workload["name"], **probe}
            if probe["state"] == "complete":
                row["artifact_closure"] = sync_and_verify(pod, workload)
            pod_row["workloads"].append(row)
        report["pods"].append(pod_row)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        config = json.loads(args.config.read_text())
        report = cycle(config)
        atomic_json(args.status, report)
        if args.once:
            break
        time.sleep(max(30, args.interval))


if __name__ == "__main__":
    main()
