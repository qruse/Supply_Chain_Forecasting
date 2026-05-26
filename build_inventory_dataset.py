from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "inventory" / "raw"
PROCESSED_DIR = ROOT / "data" / "inventory" / "processed"

INVENTORY_RAW = RAW_DIR / "retail_inventory_ml_apl.csv"
SALES_RAW = RAW_DIR / "retail_sales_ml_apl.csv"

INVENTORY_DAILY = PROCESSED_DIR / "inventory_daily.csv"
TRAIN_DAILY = PROCESSED_DIR / "inventory_daily_train.csv"
VAL_DAILY = PROCESSED_DIR / "inventory_daily_val.csv"
TEST_DAILY = PROCESSED_DIR / "inventory_daily_test.csv"
SUMMARY_PATH = PROCESSED_DIR / "inventory_dataset_summary.txt"
QUALITY_DIR = ROOT / "artifacts" / "dataset_quality"
QUALITY_REPORT_PATH = QUALITY_DIR / "inventory_quality.json"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def _missing_summary(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(count) for col, count in df.isna().sum().items() if int(count) > 0}


def clean_inventory_raw(inventory: pd.DataFrame, sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    report = {
        "inventory_input_rows": int(len(inventory)),
        "sales_input_rows": int(len(sales)),
        "inventory_input_missing": _missing_summary(inventory),
        "sales_input_missing": _missing_summary(sales),
    }

    inv_dupes = int(inventory.duplicated().sum())
    sales_dupes = int(sales.duplicated().sum())
    if inv_dupes:
        inventory = inventory.drop_duplicates().copy()
    if sales_dupes:
        sales = sales.drop_duplicates().copy()
    report["inventory_exact_duplicate_rows_removed"] = inv_dupes
    report["sales_exact_duplicate_rows_removed"] = sales_dupes

    inventory["Start Date"] = pd.to_datetime(inventory["Start Date"], errors="coerce")
    inventory["End Date"] = pd.to_datetime(inventory["End Date"], errors="coerce")
    sales["Transaction Date"] = pd.to_datetime(sales["Transaction Date"], errors="coerce")

    inv_numeric = ["Qty on hand", "Stocks Selling Amount", "Cost of Stocks", "Stock Unit Selling Price", "Stock Unit Cost Price"]
    sales_numeric = ["Qty Sold", "Sales Amount", "Cogs", "Number of Transactions"]
    for col in inv_numeric:
        inventory[col] = pd.to_numeric(inventory[col], errors="coerce")
    for col in sales_numeric:
        sales[col] = pd.to_numeric(sales[col], errors="coerce")
    sales["Is Return"] = pd.to_numeric(sales["Is Return"], errors="coerce").fillna(0)

    before_inv = len(inventory)
    inventory = inventory.dropna(subset=["Product No", "Start Date"]).copy()
    before_sales = len(sales)
    sales = sales.dropna(subset=["Product No", "Transaction Date"]).copy()
    report["inventory_critical_missing_rows_removed"] = int(before_inv - len(inventory))
    report["sales_critical_missing_rows_removed"] = int(before_sales - len(sales))

    inv_numeric_missing = {col: int(inventory[col].isna().sum()) for col in inv_numeric if int(inventory[col].isna().sum()) > 0}
    sales_numeric_missing = {col: int(sales[col].isna().sum()) for col in sales_numeric if int(sales[col].isna().sum()) > 0}
    inventory[inv_numeric] = inventory[inv_numeric].fillna(0.0)
    sales[sales_numeric] = sales[sales_numeric].fillna(0.0)
    report["inventory_numeric_missing_filled_with_zero"] = inv_numeric_missing
    report["sales_numeric_missing_filled_with_zero"] = sales_numeric_missing

    categorical_cols = ["Supplier", "Product Description", "Product Division", "Product Category", "Product Subcategory", "Product Segment", "Store"]
    for col in categorical_cols:
        if col in inventory.columns:
            inventory[col] = inventory[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        if col in sales.columns:
            sales[col] = sales[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    report["inventory_output_rows"] = int(len(inventory))
    report["sales_output_rows"] = int(len(sales))
    return inventory, sales, report


def split_by_date(dates: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = pd.Index(pd.to_datetime(dates).drop_duplicates().sort_values())
    n_dates = len(unique_dates)
    train_end_idx = round(n_dates * TRAIN_RATIO) - 1
    val_end_idx = round(n_dates * (TRAIN_RATIO + VAL_RATIO)) - 1
    return pd.Timestamp(unique_dates[train_end_idx]), pd.Timestamp(unique_dates[val_end_idx])


def build_daily_inventory() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    inventory_cols = [
        "Start Date",
        "End Date",
        "Stock Status",
        "Supplier",
        "Product No",
        "Product Description",
        "Product Division",
        "Product Category",
        "Product Subcategory",
        "Product Segment",
        "Store",
        "Qty on hand",
        "Stocks Selling Amount",
        "Cost of Stocks",
        "Stock Unit Selling Price",
        "Stock Unit Cost Price",
    ]
    sales_cols = [
        "Transaction Date",
        "Is Return",
        "Supplier",
        "Product No",
        "Product Division",
        "Product Category",
        "Product Subcategory",
        "Product Segment",
        "Store",
        "Qty Sold",
        "Sales Amount",
        "Cogs",
        "Number of Transactions",
    ]

    inventory = pd.read_csv(INVENTORY_RAW, usecols=inventory_cols)
    sales = pd.read_csv(SALES_RAW, usecols=sales_cols)
    inventory, sales, quality_report = clean_inventory_raw(inventory, sales)

    min_date = min(inventory["Start Date"].min(), sales["Transaction Date"].min())
    max_sales_date = sales["Transaction Date"].max()
    finite_end = inventory.loc[inventory["End Date"].dt.year < 2100, "End Date"].max()
    max_date = max_sales_date if pd.notna(max_sales_date) else finite_end

    inventory["End Date"] = inventory["End Date"].where(inventory["End Date"].dt.year < 2100, max_date)
    inventory["End Date"] = inventory["End Date"].clip(upper=max_date)
    inventory = inventory[inventory["Start Date"] <= inventory["End Date"]].copy()

    products = sorted(set(inventory["Product No"]).union(set(sales["Product No"])))
    dates = pd.date_range(min_date, max_date, freq="D")

    meta_cols = [
        "Product No",
        "Supplier",
        "Product Description",
        "Product Division",
        "Product Category",
        "Product Subcategory",
        "Product Segment",
    ]
    meta = (
        pd.concat(
            [
                inventory[meta_cols],
                sales[[c for c in meta_cols if c in sales.columns]],
            ],
            ignore_index=True,
        )
        .dropna(subset=["Product No"])
        .drop_duplicates("Product No")
    )

    start_delta = (
        inventory.groupby(["Product No", "Start Date"], as_index=False)
        .agg(
            qty_delta=("Qty on hand", "sum"),
            stock_value_delta=("Stocks Selling Amount", "sum"),
            stock_cost_delta=("Cost of Stocks", "sum"),
        )
        .rename(columns={"Start Date": "date"})
    )
    end_delta = inventory.copy()
    end_delta["date"] = end_delta["End Date"] + pd.Timedelta(days=1)
    end_delta = end_delta[end_delta["date"] <= max_date + pd.Timedelta(days=1)]
    end_delta = (
        end_delta.groupby(["Product No", "date"], as_index=False)
        .agg(
            qty_delta=("Qty on hand", "sum"),
            stock_value_delta=("Stocks Selling Amount", "sum"),
            stock_cost_delta=("Cost of Stocks", "sum"),
        )
    )
    end_delta[["qty_delta", "stock_value_delta", "stock_cost_delta"]] *= -1

    deltas = pd.concat([start_delta, end_delta], ignore_index=True)
    deltas = (
        deltas.groupby(["Product No", "date"], as_index=False)[
            ["qty_delta", "stock_value_delta", "stock_cost_delta"]
        ]
        .sum()
        .sort_values(["Product No", "date"])
    )

    base = pd.MultiIndex.from_product([products, dates], names=["Product No", "date"]).to_frame(index=False)
    panel = base.merge(deltas, on=["Product No", "date"], how="left")
    panel[["qty_delta", "stock_value_delta", "stock_cost_delta"]] = panel[
        ["qty_delta", "stock_value_delta", "stock_cost_delta"]
    ].fillna(0.0)
    panel[["qty_on_hand", "stock_value", "stock_cost"]] = panel.groupby("Product No", sort=False)[
        ["qty_delta", "stock_value_delta", "stock_cost_delta"]
    ].cumsum()
    panel = panel.drop(columns=["qty_delta", "stock_value_delta", "stock_cost_delta"])

    sales_daily = (
        sales.groupby(["Product No", "Transaction Date"], as_index=False)
        .agg(
            qty_sold=("Qty Sold", "sum"),
            sales_amount=("Sales Amount", "sum"),
            cogs=("Cogs", "sum"),
            transaction_count=("Number of Transactions", "sum"),
            return_count=("Is Return", "sum"),
            active_stores=("Store", "nunique"),
        )
        .rename(columns={"Transaction Date": "date"})
    )
    panel = panel.merge(sales_daily, on=["Product No", "date"], how="left")
    fill_cols = ["qty_sold", "sales_amount", "cogs", "transaction_count", "return_count", "active_stores"]
    panel[fill_cols] = panel[fill_cols].fillna(0.0)
    panel = panel.merge(meta, on="Product No", how="left")
    for col in ["Supplier", "Product Description", "Product Division", "Product Category", "Product Subcategory", "Product Segment"]:
        if col in panel.columns:
            panel[col] = panel[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    dt = panel["date"]
    iso = dt.dt.isocalendar()
    panel["day_of_week"] = dt.dt.dayofweek.astype(int)
    panel["day_of_month"] = dt.dt.day.astype(int)
    panel["month"] = dt.dt.month.astype(int)
    panel["quarter"] = dt.dt.quarter.astype(int)
    panel["year"] = dt.dt.year.astype(int)
    panel["week_of_year"] = iso.week.astype(int)
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype(int)
    panel["stockout_flag"] = (panel["qty_on_hand"] <= 0).astype(int)
    panel["stock_unit_margin"] = np.where(panel["qty_on_hand"] != 0, (panel["stock_value"] - panel["stock_cost"]) / panel["qty_on_hand"], 0.0)
    panel = panel.sort_values(["Product No", "date"]).reset_index(drop=True)
    quality_report["panel_rows"] = int(len(panel))
    quality_report["product_count"] = int(panel["Product No"].nunique())
    quality_report["panel_duplicate_product_date_rows"] = int(panel.duplicated(["Product No", "date"]).sum())
    quality_report["panel_missing_after_fill"] = _missing_summary(panel)
    QUALITY_REPORT_PATH.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return panel


def main() -> None:
    panel = build_daily_inventory()
    train_end, val_end = split_by_date(panel["date"])
    train = panel[panel["date"] <= train_end].copy()
    val = panel[(panel["date"] > train_end) & (panel["date"] <= val_end)].copy()
    test = panel[panel["date"] > val_end].copy()

    panel.to_csv(INVENTORY_DAILY, index=False, encoding="utf-8-sig")
    train.to_csv(TRAIN_DAILY, index=False, encoding="utf-8-sig")
    val.to_csv(VAL_DAILY, index=False, encoding="utf-8-sig")
    test.to_csv(TEST_DAILY, index=False, encoding="utf-8-sig")

    summary = [
        f"panel_shape={panel.shape}",
        f"date_range={panel['date'].min().date()}->{panel['date'].max().date()}",
        f"products={panel['Product No'].nunique()}",
        f"train_rows={len(train)} val_rows={len(val)} test_rows={len(test)}",
        f"split_end_dates=train:{train_end.date()} val:{val_end.date()}",
        f"target=qty_on_hand",
        f"auxiliary_targets=stockout_flag, stock_value, stock_cost",
    ]
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"saved: {INVENTORY_DAILY}")
    print(f"saved: {TRAIN_DAILY}")
    print(f"saved: {VAL_DAILY}")
    print(f"saved: {TEST_DAILY}")
    print(f"quality report: {QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
