#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/pipeline_common.sh"

run_logged "11_test_baselines_all" \
  "${PYTHON_BIN}" testing.py \
  --dataset all \
  --model all \
  --batch-size "${BASELINE_BATCH_SIZE}" \
  --device "${DEVICE}"
