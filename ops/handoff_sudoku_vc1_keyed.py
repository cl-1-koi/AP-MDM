#!/usr/bin/env python3
"""Verified aligned-to-keyed Sudoku VC-1 handoff on the retained L40S."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


POD_ID = "82i4fwoch34p40"
HOST = "64.247.206.218"
PORT = 11960
SSH_KEY = "/home/ubuntu/.ssh/id_ed25519"
SOURCE = "/workspace/ip-repro-keyed-f992c50"
COMMIT = "f992c50ed07b78a0da9eb41c1a0a0edcb50f7f6f"
MODE = "keyed_shuffled_solution_hint"
REMOTE_PARENT = "/workspace/artifacts/sudoku-vc1"
SMOKE_NAME = "smoke-keyed-f992c50-s42"
FULL_NAME = "full-keyed-f992c50-s42"
LOCAL_PARENT = Path("/home/ubuntu/apmdm-official-data/sudoku-vc1/keyed")
STATUS_PATH = Path("/home/ubuntu/apmdm-official-data/sudoku-vc1-supervisor/status.json")
HANDOFF_PATH = Path("/home/ubuntu/apmdm-official-data/sudoku-vc1-supervisor/keyed-handoff.json")
LOCAL_REPO = "/home/ubuntu/IP-repro-paper-native-20260809"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout)


def ssh(remote: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return command(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-i", SSH_KEY, "-p", str(PORT), f"root@{HOST}", remote,
        ],
        timeout=timeout,
    )


def write_status(state: str, **fields: object) -> None:
    payload = {
        "schema": "apmdm/sudoku-vc1-keyed-handoff-v1",
        "checked_utc": now(),
        "pod_id": POD_ID,
        "state": state,
        "source_commit": COMMIT,
        "condition_mode": MODE,
        **fields,
    }
    HANDOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = HANDOFF_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, HANDOFF_PATH)


def aligned_closure_verified() -> bool:
    try:
        payload = json.loads(STATUS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    rows = payload.get("workloads", [])
    return any(
        row.get("condition_mode") == "aligned_solution_hint"
        and row.get("artifact_closure", {}).get("verified") is True
        and row.get("tmux_alive") is False
        for row in rows
    )


def ensure_remote_source() -> None:
    check = ssh(
        f"test $(git -C {shlex.quote(SOURCE)} rev-parse HEAD) = {COMMIT} && "
        f"test -z \"$(git -C {shlex.quote(SOURCE)} status --porcelain)\""
    )
    if check.returncode:
        raise RuntimeError(f"remote source preflight failed: {check.stderr.strip()}")


def remote_output_absent(name: str) -> None:
    check = ssh(
        f"test ! -e {shlex.quote(REMOTE_PARENT + '/' + name)} && "
        f"test ! -e {shlex.quote(REMOTE_PARENT + '/' + name + '.runner.log')}"
    )
    if check.returncode:
        raise RuntimeError(f"refusing to overwrite existing remote output {name}")


def launch(name: str, tmux_name: str, steps: int) -> None:
    remote_output = f"{REMOTE_PARENT}/{name}"
    inner = (
        f"cd {shlex.quote(SOURCE)} && "
        f"bash ops/run_sudoku_vc1_runpod.sh {MODE} {shlex.quote(remote_output)} "
        f"{COMMIT} {steps} > {shlex.quote(remote_output + '.runner.log')} 2>&1"
    )
    result = ssh(
        f"tmux new-session -d -s {shlex.quote(tmux_name)} "
        f"{shlex.quote('bash -lc ' + shlex.quote(inner))}"
    )
    if result.returncode:
        raise RuntimeError(f"launch failed: {result.stderr.strip()}")


def wait_for_completion(name: str, tmux_name: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = ssh(
            f"if test -s {shlex.quote(REMOTE_PARENT + '/' + name + '/completion.json')}; "
            "then echo COMPLETE; "
            f"elif test -s {shlex.quote(REMOTE_PARENT + '/' + name + '/failure.json')}; "
            "then echo FAILED; "
            f"elif tmux has-session -t {shlex.quote(tmux_name)} 2>/dev/null; "
            "then echo RUNNING; else echo VANISHED; fi"
        )
        state = result.stdout.strip()
        write_status(f"smoke_{state.lower()}")
        if state == "COMPLETE":
            return
        if state in {"FAILED", "VANISHED"}:
            raise RuntimeError(f"smoke terminated as {state}")
        time.sleep(10)
    raise TimeoutError("keyed smoke exceeded time budget")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_and_verify(name: str) -> Path:
    LOCAL_PARENT.mkdir(parents=True, exist_ok=True)
    remote = f"root@{HOST}:{REMOTE_PARENT}/{name}"
    transport = f"ssh -o BatchMode=yes -o ConnectTimeout=10 -i {SSH_KEY} -p {PORT}"
    for source in (remote, remote + ".runner.log"):
        result = command(
            ["rsync", "-a", "--partial", "--timeout=60", "-e", transport, source, str(LOCAL_PARENT) + "/"],
            timeout=240,
        )
        if result.returncode:
            raise RuntimeError(f"smoke sync failed: {result.stderr.strip()}")
    remote_hash = ssh(
        f"cd {shlex.quote(REMOTE_PARENT)} && find {shlex.quote(name)} "
        f"{shlex.quote(name + '.runner.log')} -type f -print0 | sort -z | xargs -0 sha256sum",
        timeout=240,
    )
    if remote_hash.returncode:
        raise RuntimeError(f"remote hash failed: {remote_hash.stderr.strip()}")
    expected = {}
    for line in remote_hash.stdout.splitlines():
        digest, separator, relative = line.partition("  ")
        if separator:
            expected[relative] = digest
    local = {}
    output = LOCAL_PARENT / name
    for path in sorted(output.rglob("*")):
        if path.is_file():
            local[str(path.relative_to(LOCAL_PARENT))] = sha256(path)
    runner = LOCAL_PARENT / f"{name}.runner.log"
    local[runner.name] = sha256(runner)
    if not expected or expected != local:
        raise RuntimeError("remote/local smoke hashes differ")
    return output


def validate_smoke(output: Path) -> dict:
    completion = json.loads((output / "completion.json").read_text())
    if completion.get("source_commit") != COMMIT or completion.get("step") != 100:
        raise RuntimeError("smoke completion provenance mismatch")
    if completion.get("condition_mode") != MODE:
        raise RuntimeError("smoke condition-mode mismatch")
    run = output / "data" / "runs" / f"vc1-{MODE}-s42-u100"
    events = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
    train = [row for row in events if row.get("kind") == "train"]
    evaluation = [row for row in events if row.get("kind") == "eval"]
    if not train or train[-1].get("step") != 100 or train[-1].get("gpu_cuda") is not True:
        raise RuntimeError("smoke training telemetry incomplete")
    for key in ("loss", "digit_nll", "grad_norm", "steps_per_second"):
        if not math.isfinite(float(train[-1][key])):
            raise RuntimeError(f"smoke emitted non-finite {key}")
    if not evaluation or evaluation[-1].get("step") != 100:
        raise RuntimeError("smoke evaluation telemetry incomplete")
    checkpoint = run / "checkpoints" / "checkpoint_step_000000100.pt"
    if not checkpoint.is_file():
        raise RuntimeError("smoke checkpoint missing")
    return {"train": train[-1], "evaluation": evaluation[-1]}


def wait_for_full_health(timeout_s: int = 300) -> dict:
    telemetry = (
        f"{REMOTE_PARENT}/{FULL_NAME}/data/runs/"
        f"vc1-{MODE}-s42-u78100/telemetry.jsonl"
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = ssh(f"test -s {shlex.quote(telemetry)} && tail -n 1 {shlex.quote(telemetry)}")
        if result.returncode == 0 and result.stdout.strip():
            row = json.loads(result.stdout.strip())
            if row.get("kind") == "train" and int(row.get("step", 0)) >= 100:
                if row.get("gpu_cuda") is not True or not math.isfinite(float(row["loss"])):
                    raise RuntimeError("full run failed its initial health telemetry")
                return row
        time.sleep(10)
    raise TimeoutError("full keyed run did not emit timely health telemetry")


def switch_supervisor() -> None:
    command(["tmux", "kill-session", "-t", "sudoku_vc1_supervisor"], timeout=20)
    result = command(
        [
            "tmux", "new-session", "-d", "-s", "sudoku_vc1_supervisor",
            "bash", "-lc",
            f"cd {shlex.quote(LOCAL_REPO)} && exec python ops/watch_sudoku_vc1_panel.py --phase keyed",
        ],
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError(f"keyed supervisor launch failed: {result.stderr.strip()}")


def main() -> None:
    write_status("waiting_for_aligned_closure")
    while not aligned_closure_verified():
        time.sleep(30)
    write_status("aligned_closure_verified")
    ensure_remote_source()
    remote_output_absent(SMOKE_NAME)
    launch(SMOKE_NAME, "vc1_keyed_smoke", 100)
    wait_for_completion(SMOKE_NAME, "vc1_keyed_smoke", 900)
    smoke = validate_smoke(sync_and_verify(SMOKE_NAME))
    write_status("smoke_verified", smoke=smoke)
    remote_output_absent(FULL_NAME)
    launch(FULL_NAME, "vc1_keyed_full", 78_100)
    health = wait_for_full_health()
    switch_supervisor()
    write_status("full_healthy", full_health=health, smoke=smoke)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("blocked", error=f"{type(error).__name__}: {error}")
        raise
