#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/pipeline_common.sh"

run_logged "10_train_baselines_all" \
  "${PYTHON_BIN}" training.py \
  --dataset all \
  --model all \
  --epochs "${BASELINE_EPOCHS}" \
  --batch-size "${BASELINE_BATCH_SIZE}" \
  --lr "${BASELINE_LR}" \
  --hidden-size "${BASELINE_HIDDEN_SIZE}" \
  --num-layers "${BASELINE_NUM_LAYERS}" \
  --dropout "${BASELINE_DROPOUT}" \
  --patience "${BASELINE_PATIENCE}" \
  --device "${DEVICE}"
