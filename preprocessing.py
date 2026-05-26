from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "DataCoSupplyChainDataset.csv"
OUTPUT_PATH = ROOT / "data" / "df_cleaned.csv"
DAILY_OUTPUT_PATH = ROOT / "data" / "daily_demand.csv"
QUALITY_DIR = ROOT / "artifacts" / "dataset_quality"
QUALITY_REPORT_PATH = QUALITY_DIR / "dataco_preprocessing_quality.json"


DROP_COLUMNS = [
    # PII / non-useful identifiers
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Password",
    "Customer Zipcode",
    # Unusable or almost-unusable fields
    "Product Description",
    "Product Image",
    "Order Zipcode",
    # Constant / near-constant operational metadata
    "Product Status",
    # Leakage-prone or post-order fulfillment fields for demand forecasting
    "shipping date (DateOrders)",
    "Days for shipping (real)",
    "Late_delivery_risk",
    "Delivery Status",
]

CRITICAL_COLUMNS = [
    "order date (DateOrders)",
    "Product Card Id",
    "Order Item Quantity",
    "Order Item Id",
    "Order Id",
]

NUMERIC_COLUMNS = [
    "Order Item Quantity",
    "Order Item Id",
    "Order Id",
    "Order Customer Id",
    "Product Card Id",
    "Category Id",
    "Department Id",
    "Sales",
    "Order Item Discount",
]

CATEGORICAL_COLUMNS = [
    "Category Name",
    "Department Name",
    "Product Name",
]


def _missing_summary(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(count) for col, count in df.isna().sum().items() if int(count) > 0}


def clean_dataco_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {
        "input_rows": int(len(df)),
        "input_columns": int(len(df.columns)),
        "input_missing": _missing_summary(df),
    }

    exact_duplicate_rows = int(df.duplicated().sum())
    if exact_duplicate_rows:
        df = df.drop_duplicates().copy()
    report["exact_duplicate_rows_removed"] = exact_duplicate_rows

    existing_drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=existing_drop_cols)
    report["dropped_columns"] = existing_drop_cols

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    order_dt = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df = df.assign(order_dt_quality_check=order_dt)
    existing_critical = [col for col in CRITICAL_COLUMNS if col in df.columns]
    before_critical_drop = len(df)
    df = df.dropna(subset=[*existing_critical, "order_dt_quality_check"]).copy()
    report["critical_missing_rows_removed"] = int(before_critical_drop - len(df))
    df = df.drop(columns=["order_dt_quality_check"])

    nonnegative_cols = ["Order Item Quantity", "Sales", "Order Item Discount"]
    clipped: dict[str, int] = {}
    for col in nonnegative_cols:
        if col in df.columns:
            neg_count = int((df[col] < 0).sum())
            if neg_count:
                df[col] = df[col].clip(lower=0)
            clipped[col] = neg_count
    report["negative_values_clipped_to_zero"] = clipped

    fill_zero_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
    numeric_missing_after_critical = {col: int(df[col].isna().sum()) for col in fill_zero_cols if int(df[col].isna().sum()) > 0}
    if fill_zero_cols:
        df[fill_zero_cols] = df[fill_zero_cols].fillna(0)
    report["numeric_missing_filled_with_zero"] = numeric_missing_after_critical

    report["output_rows"] = int(len(df))
    report["output_columns"] = int(len(df.columns))
    report["output_missing"] = _missing_summary(df)
    return df, report


def main() -> None:
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH, encoding="latin1", low_memory=False)
    df_cleaned, quality_report = clean_dataco_transactions(df)
    order_dt = pd.to_datetime(df_cleaned["order date (DateOrders)"], errors="coerce")
    daily_source = df_cleaned.assign(order_dt=order_dt).dropna(subset=["order_dt"])

    daily_demand = (
        daily_source.groupby(daily_source["order_dt"].dt.date)
        .agg(
            demand_qty=("Order Item Quantity", "sum"),
            order_rows=("Order Item Id", "count"),
            unique_orders=("Order Id", "nunique"),
            unique_customers=("Order Customer Id", "nunique"),
            unique_products=("Product Card Id", "nunique"),
            total_sales=("Sales", "sum"),
            total_discount=("Order Item Discount", "sum"),
        )
        .reset_index()
        .rename(columns={"order_dt": "order_date"})
    )
    daily_demand["order_date"] = pd.to_datetime(daily_demand["order_date"])
    daily_demand = daily_demand.sort_values("order_date").reset_index(drop=True)

    df_cleaned.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    daily_demand.to_csv(DAILY_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    QUALITY_REPORT_PATH.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"input shape: {df.shape}")
    print(f"dropped columns ({len(quality_report['dropped_columns'])}): {quality_report['dropped_columns']}")
    print(f"exact duplicates removed: {quality_report['exact_duplicate_rows_removed']}")
    print(f"critical missing rows removed: {quality_report['critical_missing_rows_removed']}")
    print(f"output shape: {df_cleaned.shape}")
    print(f"saved to: {OUTPUT_PATH}")
    print(f"daily demand shape: {daily_demand.shape}")
    print(f"saved to: {DAILY_OUTPUT_PATH}")
    print(f"quality report: {QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
