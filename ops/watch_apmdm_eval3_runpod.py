"""Sync both co-located closures, then terminate the ordinary RunPod A40."""

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
import tomli


POD_ID = "bk223asss2nj7a"
ENDPOINT = "root@69.30.85.104"
PORT = "22008"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def ssh(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh", "-i", "/home/ubuntu/.ssh/id_ed25519", "-p", PORT,
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", ENDPOINT, command,
        ],
        text=True,
        capture_output=True,
    )


def sync(remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_stderr = local_dir / "sync_remote.stderr"
    with remote_stderr.open("wb") as stderr_handle:
        remote = subprocess.Popen(
            [
                "ssh", "-i", "/home/ubuntu/.ssh/id_ed25519", "-p", PORT,
                "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", ENDPOINT,
                "tar", "-C", remote_dir.rstrip("/"), "-cf", "-", ".",
            ],
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
        )
        assert remote.stdout is not None
        extracted = subprocess.run(
            ["tar", "-C", str(local_dir), "-xf", "-"], stdin=remote.stdout
        )
        remote.stdout.close()
        remote_status = remote.wait()
    if extracted.returncode != 0 or remote_status != 0:
        raise RuntimeError(
            f"artifact sync failed: remote={remote_status}, local={extracted.returncode}"
        )


def verify_success(local_dir: Path, expected_commit: str) -> dict[str, Any]:
    completion_path = local_dir / "completion.json"
    completion = json.loads(completion_path.read_text())
    if completion["source_commit"] != expected_commit:
        raise RuntimeError("source commit mismatch")
    checked = []
    for entry in completion["files"]:
        path = local_dir / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"missing/wrong-size artifact: {entry['path']}")
        if digest(path) != entry["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {entry['path']}")
        checked.append(entry["path"])
    durable_checkpoint = Path(completion["input_checkpoint"]["durable_source"])
    durable_vocab = Path(completion["input_vocab"]["durable_source"])
    if digest(durable_checkpoint) != completion["input_checkpoint"]["sha256"]:
        raise RuntimeError("durable parent checkpoint hash mismatch")
    if digest(durable_vocab) != completion["input_vocab"]["sha256"]:
        raise RuntimeError("durable vocabulary hash mismatch")
    return {
        "kind": "successful_completion",
        "completion_sha256": digest(completion_path),
        "verified_files": checked,
        "source_commit": expected_commit,
    }


def verify_failure(local_dir: Path, expected_commit: str) -> dict[str, Any]:
    failure = json.loads((local_dir / "failure.json").read_text())
    if failure["source_commit"] != expected_commit:
        raise RuntimeError("failure source commit mismatch")
    files = []
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(local_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    if not files:
        raise RuntimeError("failed run has no synced diagnostics")
    return {"kind": "failed_run_diagnostics", "failure": failure, "verified_files": files}


def configure_runpod() -> None:
    config = tomli.load(Path("/home/ubuntu/.runpod/config.toml").open("rb"))
    runpod.api_key = config["default"]["api_key"]


def terminate_and_verify() -> dict[str, Any]:
    configure_runpod()
    runpod.terminate_pod(POD_ID)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if not any(pod.get("id") == POD_ID for pod in runpod.get_pods()):
            return {"terminated_utc": utc_now(), "inventory_absent_verified": True}
        time.sleep(10)
    raise RuntimeError("pod remained in inventory after termination")


def main() -> None:
    global POD_ID, ENDPOINT, PORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default=POD_ID)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--go7-closure", required=True)
    parser.add_argument("--max-seconds", type=int, default=12000)
    args = parser.parse_args()
    POD_ID = args.pod_id
    ENDPOINT = args.endpoint
    PORT = str(args.port)
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    log_path = local_dir / "watcher.jsonl"
    started = time.monotonic()
    while time.monotonic() - started <= args.max_seconds:
        probe = ssh(
            f"if test -s {args.remote_dir}/completion.json; then echo complete; "
            f"elif test -s {args.remote_dir}/failure.json; then echo failed; "
            "else echo queued_or_running; fi; "
            f"tail -n 1 {args.remote_dir}/eval3.log 2>/dev/null || true"
        )
        state = probe.stdout.splitlines()[0] if probe.returncode == 0 and probe.stdout else "unreachable"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "utc": utc_now(), "state": state,
                        "ssh_status": probe.returncode, "remote": probe.stdout[-1000:],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if state in ("complete", "failed"):
            sync(args.remote_dir, local_dir)
            closure = (
                verify_success(local_dir, args.source_commit)
                if state == "complete"
                else verify_failure(local_dir, args.source_commit)
            )
            go7_path = Path(args.go7_closure)
            go7_deadline = time.monotonic() + 600
            while time.monotonic() < go7_deadline:
                if go7_path.is_file():
                    go7 = json.loads(go7_path.read_text())
                    if go7.get("pod_retained") and go7.get("kind") in {
                        "successful_completion",
                        "failed_run_diagnostics",
                    }:
                        closure["go7_artifact_closure_sha256"] = digest(go7_path)
                        break
                time.sleep(15)
            else:
                raise RuntimeError("Go7 artifact closure did not pass; refusing pod termination")
            closure["artifact_sync_verified_utc"] = utc_now()
            closure["pod_id"] = POD_ID
            atomic_json(local_dir / "artifact_closure.json", closure)
            closure.update(terminate_and_verify())
            atomic_json(local_dir / "artifact_closure.json", closure)
            print(json.dumps(closure, sort_keys=True), flush=True)
            return
        time.sleep(60)
    raise RuntimeError("AP-MDM RunPod watcher exceeded its declared wall clock")


if __name__ == "__main__":
    main()
