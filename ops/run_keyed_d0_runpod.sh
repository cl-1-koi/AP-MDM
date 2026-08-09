#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 SOURCE_COMMIT OUTPUT_DIR" >&2
  exit 2
fi

source_commit="$1"
output_dir="$2"
checkpoint_root="/workspace/artifacts/sudoku-vc1/full-keyed-f992c50-s42"
run_root="$checkpoint_root/data/runs/vc1-keyed_shuffled_solution_hint-s42-u78100"
manifest="$run_root/manifest.json"
checkpoint="$run_root/checkpoints/checkpoint_step_000078100.pt"

test "$(git rev-parse HEAD)" = "$source_commit"
test -z "$(git status --porcelain)"
test -s "$checkpoint_root/completion.json"
test -s "$manifest"
test -s "$checkpoint"
test ! -e "$output_dir"
mkdir -p "$output_dir"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("CUDA_PREFLIGHT", torch.__version__, torch.cuda.get_device_name(0))
PY

python -m repro.keyed_diagnostics \
  --manifest "$manifest" \
  --checkpoint "$checkpoint" \
  --output "$output_dir/diagnostics.json" \
  --device cuda --heldout-limit 256 --seed 42 --bf16 \
  > "$output_dir/diagnostics.log" 2>&1

python - "$output_dir" "$source_commit" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
result = json.loads((out / "diagnostics.json").read_text())
if result.get("schema") != "apmdm/keyed-diagnostics-v1":
    raise SystemExit("diagnostic schema mismatch")
if result.get("diagnostic_code", {}).get("commit") != commit:
    raise SystemExit("diagnostic source commit mismatch")
if result.get("diagnostic_code", {}).get("dirty") is not False:
    raise SystemExit("diagnostic ran from dirty source")
if result.get("checkpoint", {}).get("step") != 78100:
    raise SystemExit("wrong checkpoint step")
required = {
    "train_keyed_solution", "heldout_keyed_solution",
    "heldout_unkeyed_shuffled_solution", "heldout_no_hint",
    "train_keyed_counterfactual", "heldout_keyed_counterfactual",
}
if set(result.get("metrics", {})) != required:
    raise SystemExit("diagnostic metric set mismatch")

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = []
for path in sorted(out.iterdir()):
    if path.is_file() and path.name != "completion.json":
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
completion = {
    "schema": "apmdm/keyed-diagnostics-completion-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": commit,
    "checkpoint_sha256": result["checkpoint"]["sha256"],
    "files": files,
}
tmp = out / "completion.json.tmp"
tmp.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out / "completion.json")
PY

echo "KEYED_D0_COMPLETE source=$source_commit"
