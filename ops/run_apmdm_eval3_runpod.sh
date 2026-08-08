#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 CHECKPOINT VOCAB OUTPUT_DIR SOURCE_COMMIT" >&2
  exit 2
fi

checkpoint="$1"
vocab="$2"
output_dir="$3"
source_commit="$4"
expected_checkpoint_sha="52fc05c7f85fd4ae62a7058f277423631cbee4685595c21ee19f4abe91ab47fe"
expected_vocab_sha="1df27c69fd08e443d905fdb0ecc2beccc3f72ed968fcaf073316450972c0ee6e"
mkdir -p "$output_dir"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stage="preflight"

on_exit() {
  status=$?
  if [ "$status" -ne 0 ] && [ ! -s "$output_dir/completion.json" ]; then
    python - "$output_dir/failure.json" "$stage" "$status" "$started_utc" "$source_commit" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, stage, status, started, commit = sys.argv[1:]
temporary = path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "schema": "apmdm/paper100k-eval3-runpod-failure-v1",
            "started_utc": started,
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "exit_status": int(status),
            "source_commit": commit,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
os.replace(temporary, path)
PY
  fi
  exit "$status"
}
trap on_exit EXIT

test "$(git rev-parse HEAD)" = "$source_commit"
test -z "$(git status --porcelain)"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint_sha"
test "$(sha256sum "$vocab" | awk '{print $1}')" = "$expected_vocab_sha"

export PYTHONUNBUFFERED=1
export APMDM_REPRO_DATA_ROOT="$output_dir/data"

CUDA_VISIBLE_DEVICES="" python - "$checkpoint" "$vocab" "$output_dir/preflight.json" "$source_commit" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from repro.upstream_adapter import build_upstream_adapter, load_lightning_checkpoint

checkpoint, vocab, output, commit = sys.argv[1:]
model = build_upstream_adapter("sudoku_paper", vocab, seed=42, device="cpu")
payload = load_lightning_checkpoint(model, checkpoint, map_location="cpu")
if int(payload.get("global_step", -1)) != 100000:
    raise SystemExit("checkpoint is not global step 100000")
with torch.no_grad():
    probe = model(torch.zeros((1, 325), dtype=torch.long))
if not probe or not all(torch.isfinite(value).all() for value in probe.values()):
    raise SystemExit("non-finite CPU model preflight")
report = {
    "schema": "apmdm/paper100k-eval3-runpod-preflight-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": commit,
    "checkpoint_global_step": int(payload["global_step"]),
    "checkpoint_sha256": "52fc05c7f85fd4ae62a7058f277423631cbee4685595c21ee19f4abe91ab47fe",
    "vocab_sha256": "1df27c69fd08e443d905fdb0ecc2beccc3f72ed968fcaf073316450972c0ee6e",
    "architecture_signature": model.architecture_signature(),
    "torch_version": torch.__version__,
    "cuda_available_for_runner": bool(torch.cuda.is_available()),
}
temporary = output + ".tmp"
Path(temporary).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
PY

stage="evaluate"
started_epoch="$(date -u +%s)"
timeout --signal=INT --kill-after=120s 7200s \
  python -m repro.cli upstream-evaluate \
  --checkpoint "$checkpoint" \
  --config-name sudoku_paper \
  --vocab-cache "$vocab" \
  --limit 256 --batch-size 256 --max-steps 65536 --device cuda --bf16 \
  --eval-tag upstream-paper-100k-seed42 \
  --out "$output_dir/eval3_summary.json" \
  > "$output_dir/eval3.log" 2>&1
ended_epoch="$(date -u +%s)"
printf 'started_utc=%s\nended_utc=%s\nwall_seconds=%s\nexit_code=0\n' \
  "$started_utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$((ended_epoch - started_epoch))" \
  > "$output_dir/eval3_exit_status.txt"

stage="validate"
python - "$output_dir" "$checkpoint" "$vocab" "$source_commit" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

output_dir = Path(sys.argv[1]).resolve()
checkpoint = Path(sys.argv[2]).resolve()
vocab = Path(sys.argv[3]).resolve()
source_commit = sys.argv[4]

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

summary_path = output_dir / "eval3_summary.json"
summary = json.loads(summary_path.read_text())
if summary["n_evaluated"] != 256 or summary["checkpoint_step"] != 100000:
    raise SystemExit("semantic verdict count/step validation failed")
if summary["checkpoint_sha256"] != "52fc05c7f85fd4ae62a7058f277423631cbee4685595c21ee19f4abe91ab47fe":
    raise SystemExit("summary checkpoint hash mismatch")
if summary["vocab_cache_sha256"] != "1df27c69fd08e443d905fdb0ecc2beccc3f72ed968fcaf073316450972c0ee6e":
    raise SystemExit("summary vocab hash mismatch")
rows_path = Path(summary["rows_path"])
if not rows_path.is_file() or digest(rows_path) != summary["rows_file_sha256"]:
    raise SystemExit("raw verdict rows are missing or changed")
if sum(1 for _ in rows_path.open("r", encoding="utf-8")) != 256:
    raise SystemExit("raw verdict row count mismatch")

files = []
for path in sorted(output_dir.rglob("*")):
    if not path.is_file() or path.name in {"completion.json", "runner.log"}:
        continue
    files.append(
        {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    )
completion = {
    "schema": "apmdm/paper100k-eval3-runpod-completion-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": source_commit,
    "input_checkpoint": {
        "remote_path": str(checkpoint),
        "sha256": digest(checkpoint),
        "durable_source": "/home/ubuntu/apmdm-official-data/runs/paper-declared-100k/checkpoints/last.ckpt",
    },
    "input_vocab": {
        "remote_path": str(vocab),
        "sha256": digest(vocab),
        "durable_source": "/home/ubuntu/apmdm-official-data/vocab_cache.pkl",
    },
    "n_evaluated": 256,
    "checkpoint_step": 100000,
    "summary_sha256": digest(summary_path),
    "rows_sha256": digest(rows_path),
    "files": files,
}
temporary = output_dir / "completion.json.tmp"
temporary.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output_dir / "completion.json")
PY

stage="complete"
echo "APMDM_EVAL3_RUNPOD_COMPLETE"
