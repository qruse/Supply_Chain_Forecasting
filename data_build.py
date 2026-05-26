from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CLEANED_PATH = ROOT / "data" / "df_cleaned.csv"

SKU_DAILY_PATH = ROOT / "data" / "sku_daily.csv"
SKU_TRAIN_DAILY_PATH = ROOT / "data" / "sku_daily_train.csv"
SKU_VAL_DAILY_PATH = ROOT / "data" / "sku_daily_val.csv"
SKU_TEST_DAILY_PATH = ROOT / "data" / "sku_daily_test.csv"

TRAIN_TENSOR_PATH = ROOT / "data" / "sku_xy_30_7_train.npz"
VAL_TENSOR_PATH = ROOT / "data" / "sku_xy_30_7_val.npz"
TEST_TENSOR_PATH = ROOT / "data" / "sku_xy_30_7_test.npz"
QUALITY_DIR = ROOT / "artifacts" / "dataset_quality"
QUALITY_REPORT_PATH = QUALITY_DIR / "dataco_sku_panel_quality.json"

LOOKBACK_DAYS = 30
HORIZON_DAYS = 7
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

NUMERIC_FEATURE_COLUMNS = [
    "demand_qty",
    "order_rows",
    "unique_orders",
    "unique_customers",
    "total_sales",
    "total_discount",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "year",
    "week_of_year",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

STATIC_ID_COLUMNS = ["sku_id", "category_id", "department_id"]
STATIC_CATEGORICAL_COLUMNS = ["product_name", "category_name", "department_name"]

QUALITY_REPORT: dict = {}


def _missing_summary(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(count) for col, count in df.isna().sum().items() if int(count) > 0}


def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_PATH, encoding="utf-8-sig", low_memory=False)
    QUALITY_REPORT["input_rows"] = int(len(df))
    QUALITY_REPORT["input_missing"] = _missing_summary(df)
    exact_dupes = int(df.duplicated().sum())
    if exact_dupes:
        df = df.drop_duplicates().copy()
    QUALITY_REPORT["exact_duplicate_rows_removed"] = exact_dupes

    df["order_dt"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    numeric_cols = [
        "Product Card Id",
        "Category Id",
        "Department Id",
        "Order Item Quantity",
        "Order Item Id",
        "Order Id",
        "Order Customer Id",
        "Sales",
        "Order Item Discount",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    before_drop = len(df)
    df = df.dropna(subset=["order_dt", "Product Card Id", "Order Item Quantity"]).copy()
    QUALITY_REPORT["critical_missing_rows_removed"] = int(before_drop - len(df))

    for col in ["Category Name", "Department Name", "Product Name"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    for col in ["Sales", "Order Item Discount", "Order Item Id", "Order Id", "Order Customer Id", "Category Id", "Department Id"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if "Order Item Quantity" in df.columns:
        neg_qty = int((df["Order Item Quantity"] < 0).sum())
        df["Order Item Quantity"] = df["Order Item Quantity"].clip(lower=0)
        QUALITY_REPORT["negative_order_item_quantity_clipped"] = neg_qty
    df["order_date"] = df["order_dt"].dt.normalize()
    return df


def build_sku_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the cleaned transaction table to SKU-by-day demand panel."""
    daily_agg = (
        df.groupby(["Product Card Id", "order_date"], as_index=False)
        .agg(
            demand_qty=("Order Item Quantity", "sum"),
            order_rows=("Order Item Id", "count"),
            unique_orders=("Order Id", "nunique"),
            unique_customers=("Order Customer Id", "nunique"),
            total_sales=("Sales", "sum"),
            total_discount=("Order Item Discount", "sum"),
        )
        .rename(columns={"Product Card Id": "sku_id"})
    )
    QUALITY_REPORT["duplicate_sku_date_rows_after_aggregation"] = int(daily_agg.duplicated(["sku_id", "order_date"]).sum())

    meta = (
        df.groupby("Product Card Id", as_index=False)
        .agg(
            category_id=("Category Id", "first"),
            department_id=("Department Id", "first"),
            category_name=("Category Name", "first"),
            department_name=("Department Name", "first"),
            product_name=("Product Name", "first"),
        )
        .rename(columns={"Product Card Id": "sku_id"})
    )
    for col in ["category_name", "department_name", "product_name"]:
        meta[col] = meta[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    full_dates = pd.date_range(df["order_date"].min(), df["order_date"].max(), freq="D")
    rows = []
    for _, meta_row in meta.iterrows():
        sku_id = meta_row["sku_id"]
        sku_daily = daily_agg[daily_agg["sku_id"] == sku_id]
        base = pd.DataFrame({"order_date": full_dates})
        base["sku_id"] = sku_id
        base["category_id"] = meta_row["category_id"]
        base["department_id"] = meta_row["department_id"]
        base["category_name"] = meta_row["category_name"]
        base["department_name"] = meta_row["department_name"]
        base["product_name"] = meta_row["product_name"]
        merged = base.merge(sku_daily, on=["sku_id", "order_date"], how="left")
        fill_cols = [
            "demand_qty",
            "order_rows",
            "unique_orders",
            "unique_customers",
            "total_sales",
            "total_discount",
        ]
        merged[fill_cols] = merged[fill_cols].fillna(0)
        rows.append(merged)

    panel = pd.concat(rows, ignore_index=True)
    panel[STATIC_CATEGORICAL_COLUMNS] = panel[STATIC_CATEGORICAL_COLUMNS].fillna("Unknown")
    panel[STATIC_ID_COLUMNS] = panel[STATIC_ID_COLUMNS].fillna(0)
    panel = panel.sort_values(["sku_id", "order_date"]).reset_index(drop=True)
    QUALITY_REPORT["panel_duplicate_sku_date_rows"] = int(panel.duplicated(["sku_id", "order_date"]).sum())
    QUALITY_REPORT["panel_missing_after_fill"] = _missing_summary(panel)
    return panel


def build_static_one_hot(meta: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot encoded static categorical features at the SKU level."""
    static_ohe = pd.get_dummies(
        meta[["sku_id", *STATIC_CATEGORICAL_COLUMNS]].copy(),
        columns=STATIC_CATEGORICAL_COLUMNS,
        prefix=STATIC_CATEGORICAL_COLUMNS,
        prefix_sep="__",
        dtype=np.float32,
    )
    static_ohe = static_ohe.sort_values("sku_id").reset_index(drop=True)
    return static_ohe


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(panel["order_date"])
    iso = dt.dt.isocalendar()

    panel = panel.copy()
    panel["day_of_week"] = dt.dt.dayofweek.astype(int)
    panel["day_of_month"] = dt.dt.day.astype(int)
    panel["month"] = dt.dt.month.astype(int)
    panel["quarter"] = dt.dt.quarter.astype(int)
    panel["year"] = dt.dt.year.astype(int)
    panel["week_of_year"] = iso.week.astype(int)
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype(int)
    panel["dow_sin"] = np.sin(2 * np.pi * panel["day_of_week"] / 7)
    panel["dow_cos"] = np.cos(2 * np.pi * panel["day_of_week"] / 7)
    panel["month_sin"] = np.sin(2 * np.pi * (panel["month"] - 1) / 12)
    panel["month_cos"] = np.cos(2 * np.pi * (panel["month"] - 1) / 12)
    return panel


def split_by_date(panel: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = pd.Index(panel["order_date"].drop_duplicates().sort_values())
    n_dates = len(unique_dates)
    train_end_idx = round(n_dates * TRAIN_RATIO) - 1
    val_end_idx = round(n_dates * (TRAIN_RATIO + VAL_RATIO)) - 1
    train_end_date = pd.Timestamp(unique_dates[train_end_idx])
    val_end_date = pd.Timestamp(unique_dates[val_end_idx])
    return train_end_date, val_end_date


def save_date_splits(panel: pd.DataFrame, train_end_date: pd.Timestamp, val_end_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = panel[panel["order_date"] <= train_end_date].copy()
    val = panel[(panel["order_date"] > train_end_date) & (panel["order_date"] <= val_end_date)].copy()
    test = panel[panel["order_date"] > val_end_date].copy()
    return train, val, test


def make_tensor_windows(
    panel: pd.DataFrame,
    train_end_date: pd.Timestamp,
    val_end_date: pd.Timestamp,
    static_lookup: pd.DataFrame,
) -> tuple[dict, dict, dict]:
    """Create X/y tensors for 30-day lookback -> 7-day horizon per SKU."""
    static_feature_names = list(static_lookup.columns)
    x_list: list[np.ndarray] = []
    x_static_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    sku_ids: list[int] = []
    category_ids: list[int] = []
    department_ids: list[int] = []
    origin_dates: list[np.datetime64] = []
    target_starts: list[np.datetime64] = []
    target_ends: list[np.datetime64] = []

    for sku_id, grp in panel.groupby("sku_id", sort=False):
        grp = grp.sort_values("order_date").reset_index(drop=True)
        x_feat = grp[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y_feat = grp["demand_qty"].to_numpy(dtype=np.float32)
        dates = grp["order_date"].to_numpy(dtype="datetime64[D]")
        category_id = int(grp["category_id"].iloc[0])
        department_id = int(grp["department_id"].iloc[0])
        static_vector = static_lookup.loc[sku_id].to_numpy(dtype=np.float32)

        max_start = len(grp) - HORIZON_DAYS
        for end_idx in range(LOOKBACK_DAYS - 1, max_start):
            x_list.append(x_feat[end_idx - LOOKBACK_DAYS + 1 : end_idx + 1])
            x_static_list.append(static_vector)
            y_list.append(y_feat[end_idx + 1 : end_idx + 1 + HORIZON_DAYS])
            sku_ids.append(int(sku_id))
            category_ids.append(category_id)
            department_ids.append(department_id)
            origin_dates.append(dates[end_idx])
            target_starts.append(dates[end_idx + 1])
            target_ends.append(dates[end_idx + HORIZON_DAYS])

    X_num = np.stack(x_list).astype(np.float32)
    X_static = np.stack(x_static_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    sku_ids_arr = np.asarray(sku_ids, dtype=np.int64)
    category_ids_arr = np.asarray(category_ids, dtype=np.int64)
    department_ids_arr = np.asarray(department_ids, dtype=np.int64)
    origin_dates_arr = np.asarray(origin_dates)
    target_starts_arr = np.asarray(target_starts)
    target_ends_arr = np.asarray(target_ends)

    train_mask = target_ends_arr <= np.datetime64(train_end_date.date())
    val_mask = (target_ends_arr > np.datetime64(train_end_date.date())) & (target_ends_arr <= np.datetime64(val_end_date.date()))
    test_mask = target_ends_arr > np.datetime64(val_end_date.date())

    def pack(mask: np.ndarray) -> dict:
        return {
            "X_num": X_num[mask],
            "X_static": X_static[mask],
            "y": y[mask],
            "sku_id": sku_ids_arr[mask],
            "category_id": category_ids_arr[mask],
            "department_id": department_ids_arr[mask],
            "origin_date": origin_dates_arr[mask],
            "target_start": target_starts_arr[mask],
            "target_end": target_ends_arr[mask],
            "feature_names": np.array(NUMERIC_FEATURE_COLUMNS, dtype="U"),
            "static_feature_names": np.array(static_feature_names, dtype="U"),
        }

    return pack(train_mask), pack(val_mask), pack(test_mask)


def main() -> None:
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    df = load_transactions()
    panel = build_sku_daily_panel(df)
    panel = add_calendar_features(panel)
    QUALITY_REPORT["panel_rows"] = int(len(panel))
    QUALITY_REPORT["sku_count"] = int(panel["sku_id"].nunique())
    QUALITY_REPORT["date_range"] = {
        "min": str(panel["order_date"].min().date()),
        "max": str(panel["order_date"].max().date()),
    }

    meta = (
        panel.groupby("sku_id", as_index=False)
        .agg(
            category_id=("category_id", "first"),
            department_id=("department_id", "first"),
            category_name=("category_name", "first"),
            department_name=("department_name", "first"),
            product_name=("product_name", "first"),
        )
        .sort_values("sku_id")
        .reset_index(drop=True)
    )
    static_one_hot = build_static_one_hot(meta)
    static_feature_names = [col for col in static_one_hot.columns if col != "sku_id"]
    static_lookup = static_one_hot.set_index("sku_id")[static_feature_names]

    train_end_date, val_end_date = split_by_date(panel)
    train_panel, val_panel, test_panel = save_date_splits(panel, train_end_date, val_end_date)
    train_tensors, val_tensors, test_tensors = make_tensor_windows(
        panel,
        train_end_date,
        val_end_date,
        static_lookup,
    )

    panel.to_csv(SKU_DAILY_PATH, index=False, encoding="utf-8-sig")
    train_panel.to_csv(SKU_TRAIN_DAILY_PATH, index=False, encoding="utf-8-sig")
    val_panel.to_csv(SKU_VAL_DAILY_PATH, index=False, encoding="utf-8-sig")
    test_panel.to_csv(SKU_TEST_DAILY_PATH, index=False, encoding="utf-8-sig")

    np.savez_compressed(TRAIN_TENSOR_PATH, **train_tensors)
    np.savez_compressed(VAL_TENSOR_PATH, **val_tensors)
    np.savez_compressed(TEST_TENSOR_PATH, **test_tensors)
    QUALITY_REPORT["tensor_shapes"] = {
        "train_X_num": list(train_tensors["X_num"].shape),
        "train_y": list(train_tensors["y"].shape),
        "val_X_num": list(val_tensors["X_num"].shape),
        "val_y": list(val_tensors["y"].shape),
        "test_X_num": list(test_tensors["X_num"].shape),
        "test_y": list(test_tensors["y"].shape),
    }
    QUALITY_REPORT_PATH.write_text(json.dumps(QUALITY_REPORT, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"panel shape: {panel.shape}")
    print(f"panel date range: {panel['order_date'].min()} -> {panel['order_date'].max()}")
    print(f"split sizes (rows): train={len(train_panel)}, val={len(val_panel)}, test={len(test_panel)}")
    print(f"split end dates: train={train_end_date.date()}, val={val_end_date.date()}, test={test_panel['order_date'].max().date()}")
    print(
        "tensor shapes: "
        f"train={train_tensors['X_num'].shape}/{train_tensors['y'].shape}, "
        f"val={val_tensors['X_num'].shape}/{val_tensors['y'].shape}, "
        f"test={test_tensors['X_num'].shape}/{test_tensors['y'].shape}"
    )
    print(f"static one-hot dim: {train_tensors['X_static'].shape[-1]}")
    print(f"saved: {SKU_DAILY_PATH}")
    print(f"saved: {SKU_TRAIN_DAILY_PATH}")
    print(f"saved: {SKU_VAL_DAILY_PATH}")
    print(f"saved: {SKU_TEST_DAILY_PATH}")
    print(f"saved: {TRAIN_TENSOR_PATH}")
    print(f"saved: {VAL_TENSOR_PATH}")
    print(f"saved: {TEST_TENSOR_PATH}")
    print(f"quality report: {QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
