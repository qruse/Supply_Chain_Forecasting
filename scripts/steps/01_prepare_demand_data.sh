#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/pipeline_common.sh"

skip_or_run "01_dataco_preprocessing" \
  "${ROOT_DIR}/data/df_cleaned.csv" \
  "${ROOT_DIR}/data/daily_demand.csv" \
  "${ROOT_DIR}/artifacts/dataset_quality/dataco_preprocessing_quality.json" \
  -- "${PYTHON_BIN}" preprocessing.py

skip_or_run "02_dataco_sku_panel_build" \
  "${ROOT_DIR}/data/sku_daily.csv" \
  "${ROOT_DIR}/data/sku_daily_train.csv" \
  "${ROOT_DIR}/data/sku_daily_val.csv" \
  "${ROOT_DIR}/data/sku_daily_test.csv" \
  "${ROOT_DIR}/data/sku_xy_30_7_train.npz" \
  "${ROOT_DIR}/data/sku_xy_30_7_val.npz" \
  "${ROOT_DIR}/data/sku_xy_30_7_test.npz" \
  "${ROOT_DIR}/artifacts/dataset_quality/dataco_sku_panel_quality.json" \
  -- "${PYTHON_BIN}" data_build.py
