#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/pipeline_common.sh"

skip_or_run "04_leadtime_dataset_build" \
  "${ROOT_DIR}/data/leadtime/processed/bpi_po_goods_receipt_leadtime_cases.csv" \
  "${ROOT_DIR}/data/leadtime/processed/bpi_procurement_leadtime_daily.csv" \
  "${ROOT_DIR}/data/leadtime/processed/bpi_procurement_leadtime_daily_spend_area.csv" \
  "${ROOT_DIR}/data/leadtime/processed/bpi_procurement_leadtime_daily_train.csv" \
  "${ROOT_DIR}/data/leadtime/processed/bpi_procurement_leadtime_daily_val.csv" \
  "${ROOT_DIR}/data/leadtime/processed/bpi_procurement_leadtime_daily_test.csv" \
  "${ROOT_DIR}/artifacts/dataset_quality/leadtime_quality.json" \
  -- "${PYTHON_BIN}" build_leadtime_dataset.py
