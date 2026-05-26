#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/pipeline_common.sh"

run_logged_shell "05_data_quality_final_check" "'${PYTHON_BIN}' - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path('${ROOT_DIR}')
reports = [
    root / 'artifacts/dataset_quality/dataco_preprocessing_quality.json',
    root / 'artifacts/dataset_quality/dataco_sku_panel_quality.json',
    root / 'artifacts/dataset_quality/inventory_quality.json',
    root / 'artifacts/dataset_quality/leadtime_quality.json',
]
for path in reports:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    print(f'## {path.name}')
    for key in [
        'exact_duplicate_rows_removed',
        'inventory_exact_duplicate_rows_removed',
        'sales_exact_duplicate_rows_removed',
        'exact_duplicate_event_rows_removed',
        'critical_missing_rows_removed',
        'inventory_critical_missing_rows_removed',
        'sales_critical_missing_rows_removed',
        'missing_case_rows_removed',
        'missing_timestamp_rows_removed',
        'panel_duplicate_sku_date_rows',
        'panel_duplicate_product_date_rows',
        'daily_duplicate_dates',
        'daily_spend_duplicate_group_dates',
        'negative_leadtime_rows_removed',
    ]:
        if key in data:
            print(f'{key}={data[key]}')

checks = [
    ('dataco_sku_daily', root / 'data/sku_daily.csv', ['sku_id', 'order_date']),
    ('inventory_daily', root / 'data/inventory/processed/inventory_daily.csv', ['Product No', 'date']),
    ('leadtime_daily', root / 'data/leadtime/processed/bpi_procurement_leadtime_daily.csv', ['po_created_date']),
    ('leadtime_daily_spend', root / 'data/leadtime/processed/bpi_procurement_leadtime_daily_spend_area.csv', ['po_created_date', 'case Spend area text']),
]
for name, path, keys in checks:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    duplicate_keys = int(df.duplicated(keys).sum())
    missing = {col: int(v) for col, v in df.isna().sum().items() if int(v) > 0}
    print(f'{name}: rows={len(df)} duplicate_keys={duplicate_keys} missing_cols={missing}')
    if duplicate_keys:
        raise SystemExit(f'{name} has duplicate key rows: {duplicate_keys}')
    if missing:
        raise SystemExit(f'{name} has missing values: {missing}')
PY"
