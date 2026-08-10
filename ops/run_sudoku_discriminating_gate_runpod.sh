#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 SOURCE_COMMIT OUTPUT_DIR GATE GPU_INDEX" >&2
  exit 2
fi

source_commit="$1"
output_dir="$2"
gate="$3"
gpu_index="$4"
checkpoint="/workspace/inputs/d2/10-100000.ckpt"
vocab_cache="/workspace/inputs/d2/vocab_cache.pkl"
console_log="${output_dir}.console.log"
stage="preflight"

case "$gate" in
  d1|d2) ;;
  *) echo "unsupported gate: $gate" >&2; exit 2 ;;
esac

test "$(git rev-parse HEAD)" = "$source_commit"
test -z "$(git status --porcelain)"
test ! -e "$output_dir"
test ! -e "$console_log"
mkdir -p "$(dirname "$output_dir")"
export CUDA_VISIBLE_DEVICES="$gpu_index"

write_failure() {
  status="$?"
  if [ "$status" -ne 0 ] && [ ! -s "$output_dir/completion.json" ]; then
    mkdir -p "$output_dir"
    if [ -s "$console_log" ]; then
      mv "$console_log" "$output_dir/console.log"
    fi
    python - "$output_dir" "$source_commit" "$gate" "$stage" "$status" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
failure = {
    "schema": "apmdm/sudoku-discriminating-gate-failure-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": sys.argv[2],
    "gate": sys.argv[3],
    "stage": sys.argv[4],
    "exit_status": int(sys.argv[5]),
}
tmp = out / "failure.json.tmp"
tmp.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out / "failure.json")
PY
  fi
  exit "$status"
}
trap write_failure EXIT

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("CUDA_PREFLIGHT", torch.__version__, torch.cuda.get_device_name(0))
PY

if [ "$gate" = "d1" ]; then
  stage="d1_train"
  timeout --signal=INT --kill-after=120s 7200s \
    python -m repro.small_grokking \
      --arm fo_arm \
      --output "$output_dir" \
      --max-steps 100000 \
      --batch-size 256 \
      --width 64 \
      --blocks 2 \
      --heads 4 \
      --learning-rate 0.0001 \
      --weight-decay 0.01 \
      --warmup-steps 250 \
      --seed 42 \
      --device cuda \
      --bf16 \
      --log-every 100 \
      --eval-steps 1000,3000,10000,30000,100000 \
      --target-mode random_payload \
      --early-stop-payload-exact-rate 0.5 \
      >"$console_log" 2>&1
else
  stage="d2_audit"
  test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = \
    "52fc05c7f85fd4ae62a7058f277423631cbee4685595c21ee19f4abe91ab47fe"
  test "$(sha256sum "$vocab_cache" | cut -d' ' -f1)" = \
    "1df27c69fd08e443d905fdb0ecc2beccc3f72ed968fcaf073316450972c0ee6e"
  timeout --signal=INT --kill-after=120s 14400s \
    python -m repro.apmdm_teacher_forced_audit \
      --checkpoint "$checkpoint" \
      --vocab-cache "$vocab_cache" \
      --output "$output_dir" \
      --limit 1000 \
      --seed 42 \
      --batch-size 512 \
      --threshold 0.5 \
      --config-name sudoku_paper \
      --device cuda \
      --bf16 \
      --progress-every 10 \
      >"$console_log" 2>&1
fi

stage="closure"
test -s "$output_dir/completion.json"
mv "$console_log" "$output_dir/console.log"
python - "$output_dir" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

out = Path(sys.argv[1])
completion_path = out / "completion.json"
completion = json.loads(completion_path.read_text())
log = out / "console.log"
digest = hashlib.sha256(log.read_bytes()).hexdigest()
completion["files"] = [
    row for row in completion.get("files", []) if row.get("path") != log.name
]
completion["files"].append({
    "path": log.name,
    "bytes": log.stat().st_size,
    "sha256": digest,
})
temporary = completion_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
os.replace(temporary, completion_path)
PY

echo "SUDOKU_DISCRIMINATING_GATE_COMPLETE gate=$gate source=$source_commit"
