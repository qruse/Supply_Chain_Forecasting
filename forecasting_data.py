from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOOKBACK_DAYS = 30
HORIZON_DAYS = 7
DATASET_NAMES = ("demand", "inventory", "leadtime")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    full_path: Path
    train_path: Path
    val_path: Path
    test_path: Path
    date_col: str
    series_col: str
    target_col: str
    static_categorical_cols: tuple[str, ...] = ()
    max_static_cardinality: int = 120
    window_stride: int = 1
    arima_max_series: int | None = None


DATASET_CONFIGS = {
    "demand": DatasetConfig(
        name="demand",
        full_path=DATA_DIR / "sku_daily.csv",
        train_path=DATA_DIR / "sku_daily_train.csv",
        val_path=DATA_DIR / "sku_daily_val.csv",
        test_path=DATA_DIR / "sku_daily_test.csv",
        date_col="order_date",
        series_col="sku_id",
        target_col="demand_qty",
        static_categorical_cols=("product_name", "category_name", "department_name"),
        window_stride=1,
        arima_max_series=None,
    ),
    "inventory": DatasetConfig(
        name="inventory",
        full_path=DATA_DIR / "inventory" / "processed" / "inventory_daily.csv",
        train_path=DATA_DIR / "inventory" / "processed" / "inventory_daily_train.csv",
        val_path=DATA_DIR / "inventory" / "processed" / "inventory_daily_val.csv",
        test_path=DATA_DIR / "inventory" / "processed" / "inventory_daily_test.csv",
        date_col="date",
        series_col="Product No",
        target_col="qty_on_hand",
        static_categorical_cols=(
            "Supplier",
            "Product Division",
            "Product Category",
            "Product Subcategory",
            "Product Segment",
        ),
        max_static_cardinality=80,
        window_stride=7,
        arima_max_series=200,
    ),
    "leadtime": DatasetConfig(
        name="leadtime",
        full_path=DATA_DIR / "leadtime" / "processed" / "bpi_procurement_leadtime_daily.csv",
        train_path=DATA_DIR / "leadtime" / "processed" / "bpi_procurement_leadtime_daily_train.csv",
        val_path=DATA_DIR / "leadtime" / "processed" / "bpi_procurement_leadtime_daily_val.csv",
        test_path=DATA_DIR / "leadtime" / "processed" / "bpi_procurement_leadtime_daily_test.csv",
        date_col="po_created_date",
        series_col="__series_id",
        target_col="median_lead_time_days",
        window_stride=1,
        arima_max_series=None,
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Expected one of {sorted(DATASET_CONFIGS)}")
    return DATASET_CONFIGS[name]


def resolve_dataset_names(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(DATASET_NAMES)
    return [dataset_arg]


def load_dataset_frame(config: DatasetConfig, split: str = "full") -> pd.DataFrame:
    path = {
        "full": config.full_path,
        "train": config.train_path,
        "val": config.val_path,
        "test": config.test_path,
    }[split]
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.target_col]).copy()
    if config.series_col not in df.columns:
        df[config.series_col] = "overall"
    df[config.series_col] = df[config.series_col].astype(str)
    df[config.target_col] = pd.to_numeric(df[config.target_col], errors="coerce").fillna(0.0)
    return df.sort_values([config.series_col, config.date_col]).reset_index(drop=True)


def split_end_dates(config: DatasetConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    train = load_dataset_frame(config, "train")
    val = load_dataset_frame(config, "val")
    return pd.Timestamp(train[config.date_col].max()), pd.Timestamp(val[config.date_col].max())


def numeric_feature_columns(df: pd.DataFrame, config: DatasetConfig) -> list[str]:
    excluded = {config.date_col, config.series_col}
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if config.target_col not in cols:
        cols.insert(0, config.target_col)
    return cols


def static_feature_lookup(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    candidates = []
    for col in config.static_categorical_cols:
        if col in df.columns and df[col].nunique(dropna=True) <= config.max_static_cardinality:
            candidates.append(col)
    series_ids = sorted(df[config.series_col].astype(str).unique())
    if not candidates:
        return pd.DataFrame(index=pd.Index(series_ids, name=config.series_col))
    meta = (
        df[[config.series_col, *candidates]]
        .drop_duplicates(config.series_col)
        .set_index(config.series_col)
        .reindex(series_ids)
    )
    for col in candidates:
        meta[col] = meta[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    ohe = pd.get_dummies(meta, columns=candidates, prefix=candidates, prefix_sep="__", dtype=np.float32)
    return ohe.astype(np.float32)


def build_window_arrays(
    config: DatasetConfig,
    *,
    split: str,
    lookback: int = LOOKBACK_DAYS,
    horizon: int = HORIZON_DAYS,
) -> dict[str, np.ndarray]:
    full = load_dataset_frame(config, "full")
    train_end, val_end = split_end_dates(config)
    features = numeric_feature_columns(full, config)
    full[features] = full[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    static_lookup = static_feature_lookup(full, config)
    series_values = sorted(full[config.series_col].astype(str).unique())
    series_to_code = {series_id: idx for idx, series_id in enumerate(series_values)}

    x_num_list: list[np.ndarray] = []
    x_static_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    series_code_list: list[int] = []
    origin_dates: list[np.datetime64] = []
    target_starts: list[np.datetime64] = []
    target_ends: list[np.datetime64] = []

    for series_id, group in full.groupby(config.series_col, sort=True):
        group = group.sort_values(config.date_col).reset_index(drop=True)
        x_feat = group[features].to_numpy(dtype=np.float32)
        y_feat = group[config.target_col].to_numpy(dtype=np.float32)
        dates = group[config.date_col].to_numpy(dtype="datetime64[D]")
        if len(group) < lookback + horizon:
            continue
        if static_lookup.shape[1] == 0:
            static_vec = np.zeros((0,), dtype=np.float32)
        else:
            static_vec = static_lookup.loc[str(series_id)].to_numpy(dtype=np.float32)
        max_start = len(group) - horizon
        for end_idx in range(lookback - 1, max_start, config.window_stride):
            target_end = pd.Timestamp(dates[end_idx + horizon])
            if split == "train" and target_end > train_end:
                continue
            if split == "val" and not (train_end < target_end <= val_end):
                continue
            if split == "test" and target_end <= val_end:
                continue
            x_num_list.append(x_feat[end_idx - lookback + 1 : end_idx + 1])
            x_static_list.append(static_vec)
            y_list.append(y_feat[end_idx + 1 : end_idx + 1 + horizon])
            series_code_list.append(series_to_code[str(series_id)])
            origin_dates.append(dates[end_idx])
            target_starts.append(dates[end_idx + 1])
            target_ends.append(dates[end_idx + horizon])

    if not x_num_list:
        raise ValueError(f"No windows built for dataset={config.name}, split={split}")
    x_static = (
        np.stack(x_static_list).astype(np.float32)
        if static_lookup.shape[1] > 0
        else np.zeros((len(x_num_list), 0), dtype=np.float32)
    )
    return {
        "X_num": np.stack(x_num_list).astype(np.float32),
        "X_static": x_static,
        "y": np.stack(y_list).astype(np.float32),
        "series_code": np.asarray(series_code_list, dtype=np.int64),
        "series_id_values": np.asarray(series_values, dtype="U"),
        "origin_date": np.asarray(origin_dates),
        "target_start": np.asarray(target_starts),
        "target_end": np.asarray(target_ends),
        "feature_names": np.asarray(features, dtype="U"),
        "static_feature_names": np.asarray(list(static_lookup.columns), dtype="U"),
    }
