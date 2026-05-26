#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/pipeline_common.sh"

skip_or_run "03_inventory_dataset_build" \
  "${ROOT_DIR}/data/inventory/processed/inventory_daily.csv" \
  "${ROOT_DIR}/data/inventory/processed/inventory_daily_train.csv" \
  "${ROOT_DIR}/data/inventory/processed/inventory_daily_val.csv" \
  "${ROOT_DIR}/data/inventory/processed/inventory_daily_test.csv" \
  "${ROOT_DIR}/artifacts/dataset_quality/inventory_quality.json" \
  -- "${PYTHON_BIN}" build_inventory_dataset.py
