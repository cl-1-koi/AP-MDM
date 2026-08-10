#!/usr/bin/env bash
set -euo pipefail

arm="${1:?arm: fixed, random, or learned}"
output_dir="${2:?absolute output directory}"
commit="${3:?full source commit}"
max_steps="${4:-20000}"
seed="${5:-42}"
data_dir="${6:-/workspace/data/guacamol-v1}"

test "$(git rev-parse HEAD)" = "$commit"
test -z "$(git status --porcelain)"
python -m repro.ip_chem_train "$arm" \
  --data-dir "$data_dir" \
  --output-dir "$output_dir" \
  --max-steps "$max_steps" \
  --seed "$seed"
