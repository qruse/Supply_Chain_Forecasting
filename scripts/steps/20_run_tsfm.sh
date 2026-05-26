#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/pipeline_common.sh"

run_logged "20_run_tsfm_${TSFM_DATASET}_${TSFM_MODEL}_${TSFM_SPLIT}" \
  "${PYTHON_BIN}" foundation_experiments.py \
  --dataset "${TSFM_DATASET}" \
  --model "${TSFM_MODEL}" \
  --split "${TSFM_SPLIT}" \
  --batch-size "${TSFM_BATCH_SIZE}" \
  --num-steps "${TSFM_STEPS}" \
  --finetune-mode lora \
  --device "${DEVICE}"
