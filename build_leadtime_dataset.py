from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "leadtime" / "raw" / "BPI_Challenge_2019.csv"
PROCESSED_DIR = ROOT / "data" / "leadtime" / "processed"

CASE_LEADTIME_PATH = PROCESSED_DIR / "bpi_po_goods_receipt_leadtime_cases.csv"
DAILY_PATH = PROCESSED_DIR / "bpi_procurement_leadtime_daily.csv"
DAILY_SPEND_PATH = PROCESSED_DIR / "bpi_procurement_leadtime_daily_spend_area.csv"
DAILY_TRAIN_PATH = PROCESSED_DIR / "bpi_procurement_leadtime_daily_train.csv"
DAILY_VAL_PATH = PROCESSED_DIR / "bpi_procurement_leadtime_daily_val.csv"
DAILY_TEST_PATH = PROCESSED_DIR / "bpi_procurement_leadtime_daily_test.csv"
DAILY_SPEND_TRAIN_PATH = PROCESSED_DIR / "bpi_procurement_leadtime_daily_spend_area_train.csv"
DAILY_SPEND_VAL_PATH = PROCESSED_DIR / "bpi_procurement_leadtime_daily_spend_area_val.csv"
DAILY_SPEND_TEST_PATH = PROCESSED_DIR / "bpi_procurement_leadtime_daily_spend_area_test.csv"
WEEKLY_PATH = PROCESSED_DIR / "bpi_procurement_leadtime_weekly.csv"
WEEKLY_SPEND_PATH = PROCESSED_DIR / "bpi_procurement_leadtime_weekly_spend_area.csv"
SUMMARY_PATH = PROCESSED_DIR / "leadtime_dataset_summary.txt"
QUALITY_DIR = ROOT / "artifacts" / "dataset_quality"
QUALITY_REPORT_PATH = QUALITY_DIR / "leadtime_quality.json"

ACT_CREATE_PO = "Create Purchase Order Item"
ACT_GOODS_RECEIPT = "Record Goods Receipt"
TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S.%f"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
QUALITY_REPORT: dict = {}


def _missing_summary(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(count) for col, count in df.isna().sum().items() if int(count) > 0}


def split_by_date(df: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    unique_dates = pd.Index(pd.to_datetime(df[date_col]).drop_duplicates().sort_values())
    n_dates = len(unique_dates)
    train_end = pd.Timestamp(unique_dates[round(n_dates * TRAIN_RATIO) - 1])
    val_end = pd.Timestamp(unique_dates[round(n_dates * (TRAIN_RATIO + VAL_RATIO)) - 1])
    train = df[df[date_col] <= train_end].copy()
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test = df[df[date_col] > val_end].copy()
    return train, val, test, train_end, val_end


def load_relevant_events() -> pd.DataFrame:
    usecols = [
        "case Spend area text",
        "case Company",
        "case Sub spend area text",
        "case Purchasing Document",
        "case Vendor",
        "case Item Type",
        "case Item Category",
        "case Spend classification text",
        "case Source",
        "case Item",
        "case concept:name",
        "case Goods Receipt",
        "event concept:name",
        "event Cumulative net worth (EUR)",
        "event time:timestamp",
    ]
    chunks = []
    input_rows = 0
    exact_dupes_removed = 0
    missing_ts_removed = 0
    missing_case_removed = 0
    for chunk in pd.read_csv(RAW_PATH, usecols=usecols, chunksize=300_000, encoding="latin1", low_memory=False):
        input_rows += len(chunk)
        chunk = chunk[chunk["event concept:name"].isin([ACT_CREATE_PO, ACT_GOODS_RECEIPT])].copy()
        dupe_count = int(chunk.duplicated().sum())
        exact_dupes_removed += dupe_count
        if dupe_count:
            chunk = chunk.drop_duplicates().copy()
        chunk["event_ts"] = pd.to_datetime(chunk["event time:timestamp"], format=TIMESTAMP_FORMAT, errors="coerce")
        chunk["event Cumulative net worth (EUR)"] = pd.to_numeric(chunk["event Cumulative net worth (EUR)"], errors="coerce")
        before_case = len(chunk)
        chunk = chunk.dropna(subset=["case concept:name"]).copy()
        missing_case_removed += before_case - len(chunk)
        before_ts = len(chunk)
        chunk = chunk.dropna(subset=["event_ts"]).copy()
        missing_ts_removed += before_ts - len(chunk)
        for col in [
            "case Spend area text",
            "case Company",
            "case Sub spend area text",
            "case Purchasing Document",
            "case Vendor",
            "case Item Type",
            "case Item Category",
            "case Spend classification text",
            "case Source",
            "case Item",
            "case Goods Receipt",
        ]:
            chunk[col] = chunk[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        chunk["event Cumulative net worth (EUR)"] = chunk["event Cumulative net worth (EUR)"].fillna(0.0)
        chunks.append(chunk)
    events = pd.concat(chunks, ignore_index=True)
    QUALITY_REPORT["raw_rows_scanned"] = int(input_rows)
    QUALITY_REPORT["relevant_event_rows"] = int(len(events))
    QUALITY_REPORT["exact_duplicate_event_rows_removed"] = int(exact_dupes_removed)
    QUALITY_REPORT["missing_case_rows_removed"] = int(missing_case_removed)
    QUALITY_REPORT["missing_timestamp_rows_removed"] = int(missing_ts_removed)
    QUALITY_REPORT["event_missing_after_fill"] = _missing_summary(events)
    return events


def build_case_leadtimes(events: pd.DataFrame) -> pd.DataFrame:
    QUALITY_REPORT["duplicate_create_events_collapsed"] = int(events[events["event concept:name"] == ACT_CREATE_PO].duplicated("case concept:name").sum())
    QUALITY_REPORT["duplicate_receipt_events_collapsed"] = int(events[events["event concept:name"] == ACT_GOODS_RECEIPT].duplicated("case concept:name").sum())
    create = (
        events[events["event concept:name"] == ACT_CREATE_PO]
        .sort_values("event_ts")
        .drop_duplicates("case concept:name")
        .rename(columns={"event_ts": "po_created_ts", "event Cumulative net worth (EUR)": "po_value_eur"})
    )
    receipt = (
        events[events["event concept:name"] == ACT_GOODS_RECEIPT]
        .sort_values("event_ts")
        .drop_duplicates("case concept:name")
        .loc[:, ["case concept:name", "event_ts"]]
        .rename(columns={"event_ts": "goods_receipt_ts"})
    )

    meta_cols = [
        "case concept:name",
        "case Spend area text",
        "case Company",
        "case Sub spend area text",
        "case Purchasing Document",
        "case Vendor",
        "case Item Type",
        "case Item Category",
        "case Spend classification text",
        "case Source",
        "case Item",
        "case Goods Receipt",
        "po_created_ts",
        "po_value_eur",
    ]
    case_df = create[meta_cols].merge(receipt, on="case concept:name", how="left")
    case_df["lead_time_days"] = (case_df["goods_receipt_ts"] - case_df["po_created_ts"]).dt.total_seconds() / 86400
    case_df["has_goods_receipt"] = case_df["goods_receipt_ts"].notna().astype(int)
    negative_leadtime_rows = int((case_df["lead_time_days"] < 0).sum())
    case_df = case_df[case_df["lead_time_days"].isna() | (case_df["lead_time_days"] >= 0)].copy()
    case_df["po_created_date"] = case_df["po_created_ts"].dt.normalize()
    case_df["po_created_week"] = case_df["po_created_ts"].dt.to_period("W-MON").dt.start_time
    case_df = case_df.sort_values(["po_created_ts", "case concept:name"]).reset_index(drop=True)
    QUALITY_REPORT["negative_leadtime_rows_removed"] = negative_leadtime_rows
    QUALITY_REPORT["case_rows"] = int(len(case_df))
    QUALITY_REPORT["case_missing_after_processing"] = _missing_summary(case_df)
    return case_df


def build_periodic(case_df: pd.DataFrame, date_col: str, group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = []
    received = case_df[case_df["has_goods_receipt"] == 1].copy()
    grouped = received.groupby([date_col, *group_cols], dropna=False)
    periodic = grouped.agg(
        po_item_count=("case concept:name", "nunique"),
        total_po_value_eur=("po_value_eur", "sum"),
        mean_lead_time_days=("lead_time_days", "mean"),
        median_lead_time_days=("lead_time_days", "median"),
        p90_lead_time_days=("lead_time_days", lambda s: float(s.quantile(0.9))),
        min_lead_time_days=("lead_time_days", "min"),
        max_lead_time_days=("lead_time_days", "max"),
        vendor_count=("case Vendor", "nunique"),
    ).reset_index()
    return periodic.sort_values([date_col, *group_cols]).reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    events = load_relevant_events()
    case_df = build_case_leadtimes(events)
    daily = build_periodic(case_df, "po_created_date")
    daily_spend = build_periodic(case_df, "po_created_date", ["case Spend area text"])
    weekly = build_periodic(case_df, "po_created_week")
    weekly_spend = build_periodic(case_df, "po_created_week", ["case Spend area text"])
    QUALITY_REPORT["daily_duplicate_dates"] = int(daily.duplicated(["po_created_date"]).sum())
    QUALITY_REPORT["daily_spend_duplicate_group_dates"] = int(daily_spend.duplicated(["po_created_date", "case Spend area text"]).sum())

    case_df.to_csv(CASE_LEADTIME_PATH, index=False, encoding="utf-8-sig")
    daily.to_csv(DAILY_PATH, index=False, encoding="utf-8-sig")
    daily_spend.to_csv(DAILY_SPEND_PATH, index=False, encoding="utf-8-sig")
    weekly.to_csv(WEEKLY_PATH, index=False, encoding="utf-8-sig")
    weekly_spend.to_csv(WEEKLY_SPEND_PATH, index=False, encoding="utf-8-sig")
    daily_train, daily_val, daily_test, daily_train_end, daily_val_end = split_by_date(daily, "po_created_date")
    spend_train, spend_val, spend_test, _, _ = split_by_date(daily_spend, "po_created_date")
    daily_train.to_csv(DAILY_TRAIN_PATH, index=False, encoding="utf-8-sig")
    daily_val.to_csv(DAILY_VAL_PATH, index=False, encoding="utf-8-sig")
    daily_test.to_csv(DAILY_TEST_PATH, index=False, encoding="utf-8-sig")
    spend_train.to_csv(DAILY_SPEND_TRAIN_PATH, index=False, encoding="utf-8-sig")
    spend_val.to_csv(DAILY_SPEND_VAL_PATH, index=False, encoding="utf-8-sig")
    spend_test.to_csv(DAILY_SPEND_TEST_PATH, index=False, encoding="utf-8-sig")

    received = case_df[case_df["has_goods_receipt"] == 1]
    summary = [
        f"events_relevant_shape={events.shape}",
        f"case_shape={case_df.shape}",
        f"cases_with_goods_receipt={len(received)}",
        f"po_created_range={case_df['po_created_ts'].min()}->{case_df['po_created_ts'].max()}",
        f"goods_receipt_range={received['goods_receipt_ts'].min()}->{received['goods_receipt_ts'].max()}",
        f"daily_shape={daily.shape}",
        f"daily_spend_shape={daily_spend.shape}",
        f"daily_split_rows=train:{len(daily_train)} val:{len(daily_val)} test:{len(daily_test)}",
        f"daily_spend_split_rows=train:{len(spend_train)} val:{len(spend_val)} test:{len(spend_test)}",
        f"daily_split_end_dates=train:{daily_train_end.date()} val:{daily_val_end.date()}",
        f"weekly_shape={weekly.shape}",
        f"weekly_spend_shape={weekly_spend.shape}",
        f"target=median_lead_time_days or mean_lead_time_days",
        f"case_target=lead_time_days",
    ]
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    QUALITY_REPORT_PATH.write_text(json.dumps(QUALITY_REPORT, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"saved: {CASE_LEADTIME_PATH}")
    print(f"saved: {DAILY_PATH}")
    print(f"saved: {DAILY_SPEND_PATH}")
    print(f"saved: {DAILY_TRAIN_PATH}")
    print(f"saved: {DAILY_VAL_PATH}")
    print(f"saved: {DAILY_TEST_PATH}")
    print(f"saved: {DAILY_SPEND_TRAIN_PATH}")
    print(f"saved: {DAILY_SPEND_VAL_PATH}")
    print(f"saved: {DAILY_SPEND_TEST_PATH}")
    print(f"saved: {WEEKLY_PATH}")
    print(f"saved: {WEEKLY_SPEND_PATH}")
    print(f"quality report: {QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
