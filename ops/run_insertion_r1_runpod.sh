#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 ARM OUTPUT_DIR SOURCE_COMMIT" >&2
  exit 2
fi

arm="$1"
output_dir="$2"
source_commit="$3"
case "$arm" in fixed_ar|random_insertion|learned_insertion) ;; *) exit 3 ;; esac
run_id="mono-r1-${arm}-s42"
mkdir -p "$output_dir"
stage="preflight"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ] && [ ! -s "$output_dir/completion.json" ]; then
    python - "$output_dir/failure.json" "$stage" "$status" "$started_utc" "$source_commit" "$arm" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, stage, status, started, commit, arm = sys.argv[1:]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump({
        "schema": "apmdm/sudoku-monotone-r1-failure-v1",
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started, "stage": stage, "exit_status": int(status),
        "source_commit": commit, "arm": arm,
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
PY
  fi
  exit "$status"
}
trap on_exit EXIT

test "$(git rev-parse HEAD)" = "$source_commit"
test -z "$(git status --porcelain)"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("CUDA_PREFLIGHT", torch.__version__, torch.cuda.get_device_name(0))
PY

export APMDM_REPRO_DATA_ROOT="$output_dir/data"
export APMDM_INSERTION_PAID_CAPACITY_AUTHORIZATION="owner_authorized_2026-08-08_runpod_parallel_r1"

stage="manifest"
python -m repro.cli insertion-manifest \
  --run-id "$run_id" --arm "$arm" --max-steps 10000 --batch-size 64 \
  --checkpoint-every 10000 --log-every 100 --eval-every 1000 --eval-limit 256 \
  --device cuda --seed 42 --q-lr-ratio 0.01 --bf16 --time-budget 2700 \
  --out "$output_dir/manifest_command.json" \
  > "$output_dir/manifest.log" 2>&1

stage="train"
timeout --signal=INT --kill-after=60s 2760s \
  python -m repro.cli insertion-train --run-id "$run_id" --no-resume \
  --out "$output_dir/train_command.json" \
  > "$output_dir/train.log" 2>&1

stage="validate"
python - "$output_dir" "$run_id" "$source_commit" "$arm" <<'PY'
import hashlib, json, math, os, statistics, sys
from datetime import datetime, timezone
from pathlib import Path
import torch

out = Path(sys.argv[1]).resolve(); run_id, commit, arm = sys.argv[2:]
run = out / "data" / "runs" / run_id

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = json.loads((run / "manifest.json").read_text())
summary = json.loads((run / "train_summary.json").read_text())
if manifest["code"]["commit"] != commit or manifest["code"]["dirty"]:
    raise SystemExit("source manifest drift")
if manifest["compute"]["paid_capacity"] != "owner_authorized_2026-08-08_runpod_parallel_r1":
    raise SystemExit("paid-capacity authorization absent")
if not summary["completed"] or summary["end_step"] != 10000 or summary["target_step"] != 10000:
    raise SystemExit("panel did not reach the fixed endpoint")

checkpoint = run / "checkpoints" / "checkpoint_step_000010000.pt"
latest = run / "checkpoints" / "latest.pt"
if not checkpoint.is_file() or not latest.is_file():
    raise SystemExit("checkpoint missing")
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
required = {"policy", "optimizer", "scheduler", "generator_state", "rng"}
if payload.get("step") != 10000 or payload.get("arm") != arm or not required.issubset(payload):
    raise SystemExit("checkpoint payload incomplete")
if payload.get("manifest_sha256") != manifest["seal"]["manifest_sha256"]:
    raise SystemExit("checkpoint/manifest seal mismatch")

events = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
train = [event for event in events if event.get("kind") == "train"]
evals = [event for event in events if event.get("kind") == "eval"]
if len(train) != 100 or [event.get("step") for event in evals] != list(range(1000, 10001, 1000)):
    raise SystemExit("telemetry cadence incomplete")
for event in train:
    if event.get("gpu_cuda") is not True:
        raise SystemExit("CUDA memory telemetry absent")
    for key in ("gpu_allocated_bytes", "gpu_reserved_bytes", "gpu_free_bytes", "gpu_total_bytes"):
        if not isinstance(event.get(key), int):
            raise SystemExit(f"GPU memory telemetry absent: {key}")
    for key, value in event.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise SystemExit(f"nonfinite {key}")

common_metrics = {"loss", "steps_per_second", "examples_per_second"}
if not common_metrics.issubset(set().union(*(event.keys() for event in train))):
    raise SystemExit("common training metrics absent")
if arm == "learned_insertion":
    required_metrics = {
        "elbo", "surrogate", "q_prefix_log_prob", "q_next_entropy",
        "p_cell_entropy", "expected_digit_nll",
    }
else:
    required_metrics = {"digit_nll", "digit_accuracy", "mean_prefix"}
if not required_metrics.issubset(set().union(*(event.keys() for event in train))):
    raise SystemExit("arm-specific training metrics absent")

rows = []
for event in evals:
    rows_path = Path(event["rows_path"])
    if not rows_path.is_file() or sha(rows_path) != event["rows_sha256"]:
        raise SystemExit("evaluation rows hash mismatch")
    if sum(1 for _ in rows_path.open()) != 256:
        raise SystemExit("evaluation row count mismatch")
    rows.append({
        "step": event["step"], "solve_rate": event["solve_rate"],
        "valid_rate": event["valid_rate"], "rows_sha256": event["rows_sha256"],
    })

tracked = ["loss"] + (["elbo", "expected_digit_nll", "q_next_entropy", "p_cell_entropy"]
                       if arm == "learned_insertion" else ["digit_nll", "digit_accuracy"])
curve = {
    "schema": "apmdm/sudoku-monotone-r1-curve-v1", "arm": arm,
    "source_commit": commit, "train_records": len(train),
    "wall_seconds": summary["wall_seconds"],
    "median_steps_per_second": statistics.median(event["steps_per_second"] for event in train),
    "median_examples_per_second": statistics.median(event["examples_per_second"] for event in train),
    "max_gpu_allocated_bytes": max(event["gpu_max_allocated_bytes"] for event in train),
    "metrics": {key: {"first": train[0][key], "last": train[-1][key],
                       "delta": train[-1][key] - train[0][key]} for key in tracked},
    "evaluation": rows,
}
(out / "curve_summary.json").write_text(json.dumps(curve, indent=2, sort_keys=True) + "\n")

files = []
for path in sorted(out.rglob("*")):
    if path.is_file() and path.name not in {"completion.json", "runner.log"}:
        files.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size,
                      "sha256": sha(path)})
completion = {
    "schema": "apmdm/sudoku-monotone-r1-completion-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": commit, "arm": arm, "run_id": run_id, "step": 10000,
    "manifest_sha256": manifest["seal"]["manifest_sha256"],
    "checkpoint_sha256": sha(checkpoint), "curve_summary_sha256": sha(out / "curve_summary.json"),
    "files": files,
}
tmp = out / "completion.json.tmp"
tmp.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out / "completion.json")
PY
stage="complete"
echo "INSERTION_R1_COMPLETE arm=$arm"
