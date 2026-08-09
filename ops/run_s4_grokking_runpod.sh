#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 SOURCE_COMMIT OUTPUT_DIR ARM WEIGHT_DECAY GPU_INDEX" >&2
  exit 2
fi

source_commit="$1"
output_dir="$2"
arm="$3"
weight_decay="$4"
gpu_index="$5"

case "$arm" in
  fo_arm|ao_arm|lo_arm|mdm) ;;
  *) echo "unsupported S4 arm: $arm" >&2; exit 2 ;;
esac
case "$weight_decay" in
  0|0.0|0.01) ;;
  *) echo "unsupported weight decay: $weight_decay" >&2; exit 2 ;;
esac

test "$(git rev-parse HEAD)" = "$source_commit"
test -z "$(git status --porcelain)"
test ! -e "$output_dir"
mkdir -p "$(dirname "$output_dir")"
export CUDA_VISIBLE_DEVICES="$gpu_index"

write_failure() {
  status="$?"
  if [ "$status" -ne 0 ] && [ ! -s "$output_dir/completion.json" ]; then
    mkdir -p "$output_dir"
    python - "$output_dir" "$source_commit" "$arm" "$weight_decay" "$status" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
failure = {
    "schema": "apmdm/s4-grokking-failure-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": sys.argv[2],
    "arm": sys.argv[3],
    "weight_decay": float(sys.argv[4]),
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

python -m repro.small_grokking \
  --arm "$arm" \
  --output "$output_dir" \
  --max-steps 1000000 \
  --batch-size 256 \
  --width 64 \
  --blocks 2 \
  --heads 4 \
  --learning-rate 0.0001 \
  --weight-decay "$weight_decay" \
  --warmup-steps 250 \
  --seed 42 \
  --device cuda \
  --bf16 \
  --log-every 1000 \
  --eval-steps 1000,3000,10000,30000,100000,300000,1000000

test -s "$output_dir/completion.json"
echo "S4_GROKKING_COMPLETE arm=$arm weight_decay=$weight_decay source=$source_commit"
