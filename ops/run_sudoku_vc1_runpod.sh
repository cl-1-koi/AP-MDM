#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 CONDITION_MODE OUTPUT_DIR SOURCE_COMMIT MAX_STEPS" >&2
  exit 2
fi

condition_mode="$1"
output_dir="$2"
source_commit="$3"
max_steps="$4"
case "$condition_mode" in
  puzzle_only|aligned_solution_hint|transposed_solution_hint|keyed_shuffled_solution_hint) ;;
  *) exit 3 ;;
esac
case "$max_steps" in
  ''|*[!0-9]*) exit 4 ;;
esac

arm="random_insertion"
run_id="vc1-${condition_mode}-s42-u${max_steps}"
mkdir -p "$output_dir"
stage="preflight"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ] && [ ! -s "$output_dir/completion.json" ]; then
    python - "$output_dir/failure.json" "$stage" "$status" "$started_utc" \
      "$source_commit" "$condition_mode" "$max_steps" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, stage, status, started, commit, mode, steps = sys.argv[1:]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump({
        "schema": "apmdm/sudoku-vc1-failure-v1",
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started,
        "stage": stage,
        "exit_status": int(status),
        "source_commit": commit,
        "condition_mode": mode,
        "target_step": int(steps),
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
export APMDM_INSERTION_PAID_CAPACITY_AUTHORIZATION="owner_authorized_2026-08-09_sudoku_vc1"

checkpoint_every=5000
eval_every=5000
keep_checkpoints=$(( (max_steps + checkpoint_every - 1) / checkpoint_every + 1 ))

stage="manifest"
python -m repro.cli insertion-manifest \
  --run-id "$run_id" --arm "$arm" --condition-mode "$condition_mode" \
  --max-steps "$max_steps" --batch-size 64 \
  --checkpoint-every "$checkpoint_every" --keep-last-checkpoints "$keep_checkpoints" \
  --log-every 100 --eval-every "$eval_every" --eval-limit 256 \
  --device cuda --seed 42 --q-lr-ratio 0.01 --bf16 \
  --out "$output_dir/manifest_command.json" \
  > "$output_dir/manifest.log" 2>&1

stage="train"
python -m repro.cli insertion-train --run-id "$run_id" --no-resume \
  --out "$output_dir/train_command.json" \
  > "$output_dir/train.log" 2>&1

stage="validate"
python - "$output_dir" "$run_id" "$source_commit" "$condition_mode" "$max_steps" <<'PY'
import hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
import torch

out = Path(sys.argv[1]).resolve()
run_id, commit, mode, target_text = sys.argv[2:]
target = int(target_text)
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
if manifest["training"]["condition_mode"] != mode:
    raise SystemExit("condition-mode drift")
if manifest["compute"]["paid_capacity"] != "owner_authorized_2026-08-09_sudoku_vc1":
    raise SystemExit("paid-capacity authorization absent")
if not summary["completed"] or summary["end_step"] != target:
    raise SystemExit("run did not reach the sealed endpoint")

checkpoint_steps = list(range(5000, target, 5000)) + [target]
checkpoint_paths = [run / "checkpoints" / f"checkpoint_step_{step:09d}.pt" for step in checkpoint_steps]
if any(not path.is_file() for path in checkpoint_paths):
    raise SystemExit("checkpoint schedule incomplete")
for step, path in zip(checkpoint_steps, checkpoint_paths):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("step") != step or payload.get("arm") != "random_insertion":
        raise SystemExit(f"checkpoint payload mismatch at {step}")
    if payload.get("manifest_sha256") != manifest["seal"]["manifest_sha256"]:
        raise SystemExit(f"checkpoint/manifest mismatch at {step}")

events = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
train = [event for event in events if event.get("kind") == "train"]
evals = [event for event in events if event.get("kind") == "eval"]
expected_evals = list(range(5000, target, 5000)) + [target]
if [event.get("step") for event in evals] != expected_evals:
    raise SystemExit("evaluation cadence incomplete")
if not train or train[-1].get("step") != target:
    raise SystemExit("training telemetry endpoint missing")
for event in train:
    if event.get("gpu_cuda") is not True:
        raise SystemExit("CUDA telemetry absent")
    for key, value in event.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise SystemExit(f"nonfinite {key}")
for event in evals:
    rows_path = Path(event["rows_path"])
    if not rows_path.is_file() or sha(rows_path) != event["rows_sha256"]:
        raise SystemExit("evaluation rows hash mismatch")
    if sum(1 for _ in rows_path.open()) != 256:
        raise SystemExit("evaluation row count mismatch")

curve = {
    "schema": "apmdm/sudoku-vc1-curve-v1",
    "condition_mode": mode,
    "source_commit": commit,
    "target_step": target,
    "training": [
        {key: event[key] for key in ("step", "loss", "digit_nll", "digit_accuracy", "steps_per_second")}
        for event in train
    ],
    "evaluation": [
        {key: event[key] for key in ("step", "solve_rate", "valid_rate", "consistent_rate", "rows_sha256")}
        for event in evals
    ],
}
(out / "curve_summary.json").write_text(json.dumps(curve, indent=2, sort_keys=True) + "\n")

files = []
for path in sorted(out.rglob("*")):
    if path.is_file() and path.name not in {"completion.json", "runner.log"}:
        files.append({
            "path": str(path.relative_to(out)),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        })
completion = {
    "schema": "apmdm/sudoku-vc1-completion-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": commit,
    "condition_mode": mode,
    "run_id": run_id,
    "step": target,
    "manifest_sha256": manifest["seal"]["manifest_sha256"],
    "files": files,
}
tmp = out / "completion.json.tmp"
tmp.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out / "completion.json")
PY

stage="complete"
echo "SUDOKU_VC1_COMPLETE mode=$condition_mode step=$max_steps"
