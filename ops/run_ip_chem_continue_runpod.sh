#!/usr/bin/env bash
set -euo pipefail

arm="${1:?arm: fixed, random, or learned}"
output_dir="${2:?absolute output directory}"
commit="${3:?full source commit}"
parent_checkpoint="${4:?absolute parent checkpoint}"
parent_sha256="${5:?parent checkpoint sha256}"
max_steps="${6:-100000}"
seed="${7:-42}"
data_dir="${8:-/workspace/data/guacamol-v1}"

test "$(git rev-parse HEAD)" = "$commit"
test -z "$(git status --porcelain)"
test "$(sha256sum "$parent_checkpoint" | awk '{print $1}')" = "$parent_sha256"
test ! -e "$output_dir"

python -m repro.ip_chem_train "$arm" \
  --data-dir "$data_dir" \
  --output-dir "$output_dir" \
  --max-steps "$max_steps" \
  --seed "$seed" \
  --resume "$parent_checkpoint" \
  --shrink-lambda 0.95 \
  --perturb-sigma 0.005 \
  --reset-optimizer-on-resume \
  --restart-schedule-on-resume \
  --resume-decoder-from-ema \
  --eval-every 10000 \
  --checkpoint-every 20000 \
  --eval-samples 512 \
  --final-eval-samples 2000
