from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from chronos import Chronos2Pipeline, ChronosBoltPipeline, ChronosPipeline
from forecasting_data import DatasetConfig, get_dataset_config, load_dataset_frame, resolve_dataset_names


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
RESULT_ROOT = ARTIFACT_DIR / "results"
CHRONOS_RESULTS_DIR = RESULT_ROOT / "chronos"
MODEL_DIR = ARTIFACT_DIR / "foundation_models"

CHRONOS1_MODEL_ID = "amazon/chronos-t5-small"
CHRONOS_BOLT_MODEL_ID = "amazon/chronos-bolt-small"
CHRONOS2_MODEL_ID = "amazon/chronos-2"
TTM_MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-transformers"

LOOKBACK_DAYS = 30
HORIZON_DAYS = 7
TTM_CONTEXT_LENGTH = 512
TTM_OUTPUT_LENGTH = 96
TIMESFM_CONTEXT_LENGTH = 128
TIMESFM_OUTPUT_LENGTH = 128
WINDOW_STRIDE = 7
MEDIAN_QUANTILE_INDEX = 4
MAX_BATCH_SIZE = 16
STANDARD_SERIES_COL = "sku_id"
STANDARD_DATE_COL = "order_date"
STANDARD_TARGET_COL = "demand_qty"

KNOWN_FUTURE_NUMERIC_COVARIATES = [
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


def resolve_device_map(device_arg: str = "auto") -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device_arg


def _format_mib(num_bytes: int) -> float:
    return num_bytes / 1024 / 1024


def log_cuda_memory(stage: str) -> None:
    if not torch.cuda.is_available():
        print(f"[cuda-memory][{stage}] cuda unavailable")
        return
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    print(
        f"[cuda-memory][{stage}] "
        f"allocated={_format_mib(allocated):.1f}MiB "
        f"reserved={_format_mib(reserved):.1f}MiB "
        f"peak_allocated={_format_mib(peak_allocated):.1f}MiB "
        f"peak_reserved={_format_mib(peak_reserved):.1f}MiB"
    )


def reset_cuda_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log_cuda_memory("after-reset")


def cleanup_cuda_memory(stage: str = "cleanup") -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        log_cuda_memory(stage)


@dataclass
class SeriesSplit:
    sku_id: str
    history: pd.DataFrame
    future: pd.DataFrame
    numeric_past_covariates: tuple[str, ...]
    known_future_numeric_covariates: tuple[str, ...]
    categorical_covariates: tuple[str, ...]


@dataclass(frozen=True)
class TsfmFeatureSpec:
    numeric_past_covariates: tuple[str, ...]
    known_future_numeric_covariates: tuple[str, ...]
    categorical_covariates: tuple[str, ...]


class TargetWindowDataset(Dataset):
    def __init__(
        self,
        daily_df: pd.DataFrame,
        *,
        context_length: int,
        prediction_length: int,
        stride: int = WINDOW_STRIDE,
    ) -> None:
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        for _, group in daily_df.groupby(STANDARD_SERIES_COL):
            values = group.sort_values(STANDARD_DATE_COL)[STANDARD_TARGET_COL].to_numpy(dtype=np.float32, copy=True)
            limit = len(values) - context_length - prediction_length + 1
            if limit <= 0:
                continue
            for start in range(0, limit, stride):
                context = values[start : start + context_length]
                future = values[start + context_length : start + context_length + prediction_length]
                self.windows.append((context, future))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        context, future = self.windows[index]
        return torch.tensor(context, dtype=torch.float32), torch.tensor(future, dtype=torch.float32)


def ensure_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    CHRONOS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def chronos_model_dir(dataset_name: str, model_name: str) -> Path:
    path = CHRONOS_RESULTS_DIR / dataset_name / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df[STANDARD_DATE_COL], errors="coerce")
    df["day_of_week"] = dates.dt.dayofweek.astype("int16")
    df["day_of_month"] = dates.dt.day.astype("int16")
    df["month"] = dates.dt.month.astype("int16")
    df["quarter"] = dates.dt.quarter.astype("int16")
    df["year"] = dates.dt.year.astype("int16")
    df["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12).astype(np.float32)
    return df


def to_tsfm_frame(config: DatasetConfig, split: str) -> pd.DataFrame:
    df = load_dataset_frame(config, split).copy()
    rename_map = {
        config.series_col: STANDARD_SERIES_COL,
        config.date_col: STANDARD_DATE_COL,
        config.target_col: STANDARD_TARGET_COL,
    }
    df = df.rename(columns=rename_map)
    df[STANDARD_SERIES_COL] = df[STANDARD_SERIES_COL].astype(str)
    df[STANDARD_DATE_COL] = pd.to_datetime(df[STANDARD_DATE_COL], errors="coerce")
    df[STANDARD_TARGET_COL] = pd.to_numeric(df[STANDARD_TARGET_COL], errors="coerce").fillna(0.0).astype(np.float32)
    df = df.dropna(subset=[STANDARD_DATE_COL]).copy()
    df = add_calendar_features(df)

    for col in df.columns:
        if col in {STANDARD_SERIES_COL, STANDARD_DATE_COL}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    return df.sort_values([STANDARD_SERIES_COL, STANDARD_DATE_COL]).reset_index(drop=True)


def load_daily_frames(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return to_tsfm_frame(config, "train"), to_tsfm_frame(config, "val"), to_tsfm_frame(config, "test")


def build_feature_spec(config: DatasetConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> TsfmFeatureSpec:
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    excluded = {STANDARD_SERIES_COL, STANDARD_DATE_COL, STANDARD_TARGET_COL}
    calendar_cols = [col for col in KNOWN_FUTURE_NUMERIC_COVARIATES if col in combined.columns]
    static_cols = tuple(col for col in config.static_categorical_cols if col in combined.columns)
    numeric_past = []
    for col in combined.columns:
        if col in excluded or col in calendar_cols:
            continue
        if pd.api.types.is_numeric_dtype(combined[col]):
            numeric_past.append(col)
    return TsfmFeatureSpec(
        numeric_past_covariates=tuple(numeric_past),
        known_future_numeric_covariates=tuple(calendar_cols),
        categorical_covariates=static_cols,
    )


def min_history_length(jobs: list[tuple[str, pd.DataFrame, pd.DataFrame]]) -> int:
    lengths = []
    for _, history_df, future_df in jobs:
        future_series = set(future_df[STANDARD_SERIES_COL].astype(str).unique())
        for series_id, group in history_df.groupby(STANDARD_SERIES_COL):
            if str(series_id) in future_series:
                lengths.append(len(group))
    return min(lengths) if lengths else 0


def effective_context_length(requested: int, jobs: list[tuple[str, pd.DataFrame, pd.DataFrame]], *, minimum: int = LOOKBACK_DAYS) -> int:
    available = min_history_length(jobs)
    if available <= 0:
        return requested
    return max(minimum, min(requested, available))


def effective_training_context_length(
    requested: int,
    jobs: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    *,
    output_length: int,
    minimum: int = LOOKBACK_DAYS,
) -> int:
    available = min_history_length(jobs)
    if available <= output_length:
        return minimum
    return max(minimum, min(requested, available - output_length))


def build_split(history_df: pd.DataFrame, future_df: pd.DataFrame, sku_id: str, feature_spec: TsfmFeatureSpec) -> SeriesSplit:
    history = history_df.loc[history_df[STANDARD_SERIES_COL] == sku_id].sort_values(STANDARD_DATE_COL).reset_index(drop=True)
    future = future_df.loc[future_df[STANDARD_SERIES_COL] == sku_id].sort_values(STANDARD_DATE_COL).reset_index(drop=True)
    if history.empty or future.empty:
        raise ValueError(f"Empty history/future for sku_id={sku_id}")
    return SeriesSplit(
        sku_id=sku_id,
        history=history,
        future=future,
        numeric_past_covariates=feature_spec.numeric_past_covariates,
        known_future_numeric_covariates=feature_spec.known_future_numeric_covariates,
        categorical_covariates=feature_spec.categorical_covariates,
    )


def _to_numpy(values: pd.Series, *, dtype: np.dtype | type = np.float32) -> np.ndarray:
    return values.to_numpy(dtype=dtype, copy=True)


def build_chronos1_input(split: SeriesSplit) -> list[torch.Tensor]:
    return [torch.tensor(_to_numpy(split.history[STANDARD_TARGET_COL]), dtype=torch.float32)]


def build_chronosbolt_input(split: SeriesSplit) -> list[torch.Tensor]:
    return [torch.tensor(_to_numpy(split.history[STANDARD_TARGET_COL]), dtype=torch.float32)]


def build_chronos2_input(split: SeriesSplit, future_slice: pd.DataFrame | None = None) -> list[dict]:
    history = split.history
    future = split.future if future_slice is None else future_slice
    past_covariates: dict[str, np.ndarray] = {}
    future_covariates: dict[str, np.ndarray] = {}

    for col in split.numeric_past_covariates:
        past_covariates[col] = _to_numpy(history[col])
    for col in split.known_future_numeric_covariates:
        past_covariates[col] = _to_numpy(history[col])
    for col in split.categorical_covariates:
        past_covariates[col] = history[col].astype(str).to_numpy(copy=True)

    for col in split.known_future_numeric_covariates:
        future_covariates[col] = _to_numpy(future[col])
    for col in split.categorical_covariates:
        future_covariates[col] = future[col].astype(str).to_numpy(copy=True)

    return [
        {
            "target": _to_numpy(history[STANDARD_TARGET_COL]),
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
        }
    ]


def build_chronos2_finetune_inputs(history_df: pd.DataFrame, feature_spec: TsfmFeatureSpec) -> list[dict]:
    tasks = []
    for sku_id in sorted(history_df[STANDARD_SERIES_COL].unique()):
        history = history_df.loc[history_df[STANDARD_SERIES_COL] == sku_id].sort_values(STANDARD_DATE_COL).reset_index(drop=True)
        if len(history) < LOOKBACK_DAYS + HORIZON_DAYS:
            continue

        past_covariates: dict[str, np.ndarray] = {}
        for col in feature_spec.numeric_past_covariates + feature_spec.known_future_numeric_covariates:
            past_covariates[col] = _to_numpy(history[col])
        for col in feature_spec.categorical_covariates:
            past_covariates[col] = history[col].astype(str).to_numpy(copy=True)

        # During fit(), Chronos-2 uses the presence of keys in future_covariates
        # to infer which covariates are known into the future. Keep operational
        # demand-derived features past-only, and mark only calendar/static fields
        # as known-future covariates.
        future_covariates = {key: None for key in feature_spec.known_future_numeric_covariates + feature_spec.categorical_covariates}
        tasks.append(
            {
                "target": _to_numpy(history[STANDARD_TARGET_COL]),
                "past_covariates": past_covariates,
                "future_covariates": future_covariates,
            }
        )
    return tasks


def build_eval_splits(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    feature_spec: TsfmFeatureSpec,
) -> list[SeriesSplit]:
    skus = sorted(future_df[STANDARD_SERIES_COL].astype(str).unique())
    return [build_split(history_df, future_df, sku_id, feature_spec) for sku_id in skus]


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    nonzero_mask = np.abs(flat_true) > 1e-8
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs((flat_true[nonzero_mask] - flat_pred[nonzero_mask]) / flat_true[nonzero_mask])) * 100.0)
    else:
        mape = float("nan")
    ss_res = float(np.sum((flat_true - flat_pred) ** 2))
    ss_tot = float(np.sum((flat_true - flat_true.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": r2}


def save_sku_metrics(path_csv: Path, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty and "model_name" in df.columns:
        ordered = ["model_name"] + [col for col in df.columns if col != "model_name"]
        df = df.loc[:, ordered]
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")
    return df


def save_sku_mae(path_csv: Path, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["model_name", "sku_id", "mae"])
    else:
        df = df.loc[:, [col for col in ["model_name", "sku_id", "mae"] if col in df.columns]]
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")
    return df


def save_chronos_outputs(
    dataset_name: str,
    model_name: str,
    split_name: str,
    forecast_df: pd.DataFrame,
    metrics: dict,
    sku_df: pd.DataFrame,
) -> None:
    model_dir = chronos_model_dir(dataset_name, model_name)
    forecast_path = model_dir / f"{model_name}_{split_name}_forecasts.csv"
    metrics_path = model_dir / f"{model_name}_{split_name}_metrics.csv"
    sku_metrics_path = model_dir / f"{model_name}_{split_name}_sku_metrics.csv"
    sku_mae_path = model_dir / f"{model_name}_{split_name}_sku_mae.csv"

    forecast_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    metrics_df = pd.DataFrame([{"dataset_name": dataset_name, "model_name": model_name, **metrics}])
    if not metrics_df.empty and "model_name" in metrics_df.columns:
        ordered = ["model_name"] + [col for col in metrics_df.columns if col != "model_name"]
        metrics_df = metrics_df.loc[:, ordered]
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    save_sku_metrics(sku_metrics_path, sku_df.to_dict(orient="records"))
    save_sku_mae(sku_mae_path, sku_df.to_dict(orient="records"))


def _predict_chronos1(pipeline: ChronosPipeline, split: SeriesSplit, context_length: int, batch_size: int) -> np.ndarray:
    inputs = build_chronos1_input(split)
    preds = pipeline.predict(
        inputs,
        prediction_length=HORIZON_DAYS,
        limit_prediction_length=False,
    )
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    if isinstance(pred, torch.Tensor):
        if pred.dim() == 3:
            return pred[0].mean(dim=0).detach().cpu().numpy().astype(np.float32)
        if pred.dim() == 2:
            return pred.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    pred = np.asarray(pred)
    if pred.ndim == 3:
        return pred[0].mean(axis=0).astype(np.float32)
    if pred.ndim == 2:
        return pred.mean(axis=0).astype(np.float32)
    raise RuntimeError("Unexpected Chronos-1 prediction shape")


def _predict_chronosbolt(pipeline: ChronosBoltPipeline, split: SeriesSplit, context_length: int, batch_size: int) -> np.ndarray:
    inputs = build_chronosbolt_input(split)
    preds = pipeline.predict(
        inputs,
        prediction_length=HORIZON_DAYS,
        limit_prediction_length=False,
    )
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    if isinstance(pred, torch.Tensor):
        if pred.dim() == 3:
            return pred[0, MEDIAN_QUANTILE_INDEX].detach().cpu().numpy().astype(np.float32)
        if pred.dim() == 2:
            return pred[MEDIAN_QUANTILE_INDEX].detach().cpu().numpy().astype(np.float32)
    pred = np.asarray(pred)
    if pred.ndim == 3:
        return pred[0, MEDIAN_QUANTILE_INDEX].astype(np.float32)
    if pred.ndim == 2:
        return pred[MEDIAN_QUANTILE_INDEX].astype(np.float32)
    raise RuntimeError("Unexpected Chronos-Bolt prediction shape")


def _predict_chronos2(pipeline: Chronos2Pipeline, split: SeriesSplit, context_length: int, batch_size: int) -> np.ndarray:
    inputs = build_chronos2_input(split, future_slice=split.future.iloc[:HORIZON_DAYS].copy())
    preds = pipeline.predict(
        inputs,
        prediction_length=HORIZON_DAYS,
        batch_size=batch_size,
        context_length=context_length,
        cross_learning=False,
        limit_prediction_length=False,
    )
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    if isinstance(pred, torch.Tensor):
        if pred.dim() == 3:
            return pred[0, MEDIAN_QUANTILE_INDEX].detach().cpu().numpy().astype(np.float32)
        if pred.dim() == 2:
            return pred[MEDIAN_QUANTILE_INDEX].detach().cpu().numpy().astype(np.float32)
    pred = np.asarray(pred)
    if pred.ndim == 3:
        return pred[0, MEDIAN_QUANTILE_INDEX].astype(np.float32)
    if pred.ndim == 2:
        return pred[MEDIAN_QUANTILE_INDEX].astype(np.float32)
    raise RuntimeError("Unexpected Chronos-2 prediction shape")


def _require_ttm_class():
    try:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    except ImportError as exc:
        raise RuntimeError("Granite TTM requires `granite-tsfm`. Install it with: pip install granite-tsfm") from exc
    return TinyTimeMixerForPrediction


def _require_timesfm_official_package():
    try:
        import timesfm
    except ImportError as exc:
        raise RuntimeError(
            "TimesFM 2.5 zero-shot requires the official package. Install it with: "
            "pip install 'timesfm[torch] @ git+https://github.com/google-research/timesfm.git'"
        ) from exc
    return timesfm


def _require_timesfm_transformers_class():
    try:
        from transformers import TimesFm2_5ModelForPrediction
        return TimesFm2_5ModelForPrediction
    except ImportError as exc:
        raise RuntimeError(
            "TimesFM 2.5 fine-tuning requires a Transformers build exposing "
            "`TimesFm2_5ModelForPrediction`. In this shared environment, "
            "Transformers 4.57 is kept for Chronos/Granite compatibility; use a "
            "separate fine-tuning venv for TimesFM 2.5 LoRA/full fine-tuning."
        ) from exc


def _predict_ttm(model: torch.nn.Module, split: SeriesSplit, device: torch.device, context_length: int) -> np.ndarray:
    values = _to_numpy(split.history[STANDARD_TARGET_COL]).astype(np.float32)
    if len(values) < context_length:
        raise ValueError(f"TTM needs at least {context_length} history points, got {len(values)} for sku_id={split.sku_id}")
    context = torch.tensor(values[-context_length:], dtype=torch.float32, device=device).view(1, context_length, 1)
    with torch.no_grad():
        output = model(past_values=context, return_loss=False)
    pred = output.prediction_outputs[0, :HORIZON_DAYS, 0].detach().cpu().numpy().astype(np.float32)
    return pred


def _predict_timesfm(model, split: SeriesSplit, context_length: int) -> np.ndarray:
    values = _to_numpy(split.history[STANDARD_TARGET_COL]).astype(np.float32)
    if len(values) < context_length:
        raise ValueError(f"TimesFM needs at least {context_length} history points, got {len(values)} for sku_id={split.sku_id}")
    point_forecast, _ = model.forecast(horizon=HORIZON_DAYS, inputs=[values[-context_length:]])
    pred = point_forecast[0, :HORIZON_DAYS].astype(np.float32)
    return np.clip(pred, a_min=0.0, a_max=None)


def _predict_timesfm_transformers(model: torch.nn.Module, split: SeriesSplit, device: torch.device, context_length: int) -> np.ndarray:
    values = _to_numpy(split.history[STANDARD_TARGET_COL]).astype(np.float32)
    if len(values) < context_length:
        raise ValueError(f"TimesFM needs at least {context_length} history points, got {len(values)} for sku_id={split.sku_id}")
    context = torch.tensor(values[-context_length:], dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(
            past_values=[context],
            freq=[0],
            forecast_context_len=context_length,
            truncate_negative=False,
        )
    pred = output.mean_predictions[0, :HORIZON_DAYS].detach().cpu().numpy().astype(np.float32)
    return np.clip(pred, a_min=0.0, a_max=None)


def evaluate_torch_foundation_model(
    model_name: str,
    splits: list[SeriesSplit],
    predictor,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows = []
    all_true = []
    all_pred = []
    sku_rows = []
    for split in splits:
        future_slice = split.future.iloc[:HORIZON_DAYS].copy()
        y_true = _to_numpy(future_slice[STANDARD_TARGET_COL]).astype(np.float32)
        y_pred = predictor(split)
        all_true.append(y_true)
        all_pred.append(y_pred)
        sku_metrics = compute_metrics(y_true, y_pred)
        sku_rows.append({"model_name": model_name, "sku_id": split.sku_id, **sku_metrics})
        rows.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "sku_id": split.sku_id,
                    "order_date": future_slice[STANDARD_DATE_COL].to_numpy(),
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        )
    forecast_df = pd.concat(rows, ignore_index=True)
    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)
    metrics = {"model_name": model_name, **compute_metrics(y_true_all, y_pred_all)}
    return forecast_df, metrics, pd.DataFrame(sku_rows)


def _run_finetune_loop(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
    model_name: str,
    timesfm_mode: bool = False,
    context_length: int,
) -> None:
    if len(loader.dataset) == 0:
        raise ValueError(f"{model_name} has no train windows. Increase available history or reduce context length.")
    model.train()
    iterator = iter(loader)
    for step in range(1, num_steps + 1):
        try:
            context, future = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            context, future = next(iterator)
        context = context.to(device)
        future = future.to(device)
        optimizer.zero_grad(set_to_none=True)
        if timesfm_mode:
            output = model(
                past_values=[row for row in context],
                freq=[0] * context.size(0),
                future_values=future,
                forecast_context_len=context_length,
                truncate_negative=False,
            )
            loss = output.loss
        else:
            output = model(
                past_values=context.unsqueeze(-1),
                future_values=future.unsqueeze(-1),
                return_loss=True,
            )
            loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step == num_steps or step % 20 == 0:
            print(f"[{model_name}] step={step}/{num_steps} loss={float(loss.detach().cpu()):.6f}")


def finetune_ttm(
    train_df: pd.DataFrame,
    *,
    dataset_name: str,
    context_length: int,
    output_length: int,
    learning_rate: float,
    num_steps: int,
    batch_size: int,
    device: torch.device,
) -> torch.nn.Module:
    TinyTimeMixerForPrediction = _require_ttm_class()
    if device.type == "cuda":
        reset_cuda_memory_stats()
    model = TinyTimeMixerForPrediction.from_pretrained(TTM_MODEL_ID).to(device)
    dataset = TargetWindowDataset(train_df, context_length=context_length, prediction_length=output_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"[ttm_ft] fine-tuning windows={len(dataset)}, batch_size={batch_size}, context={context_length}, output={output_length}")
    log_cuda_memory("before-ttm-finetune")
    _run_finetune_loop(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        num_steps=num_steps,
        model_name="ttm_ft",
        context_length=context_length,
    )
    log_cuda_memory("after-ttm-finetune")
    save_dir = MODEL_DIR / dataset_name / "ttm_ft"
    model.save_pretrained(save_dir)
    return model.eval()


def load_ttm(device: torch.device) -> torch.nn.Module:
    TinyTimeMixerForPrediction = _require_ttm_class()
    return TinyTimeMixerForPrediction.from_pretrained(TTM_MODEL_ID).to(device).eval()


def _freeze_timesfm_for_head_tuning(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    trainable = 0
    for name, param in model.named_parameters():
        if "horizon_ff_layer" in name:
            param.requires_grad = True
            trainable += param.numel()
    if trainable == 0:
        for param in model.parameters():
            param.requires_grad = True
        print("[timesfm_ft] horizon head not found; falling back to full fine-tuning")
    else:
        print(f"[timesfm_ft] trainable horizon-head parameters={trainable}")


def load_timesfm(context_length: int):
    timesfm = _require_timesfm_official_package()
    model = timesfm.TimesFM_2p5_200M_torch._from_pretrained(
        model_id="google/timesfm-2.5-200m-pytorch",
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        token=None,
        torch_compile=False,
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=max(context_length, TIMESFM_CONTEXT_LENGTH),
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def load_timesfm_transformers(device: torch.device) -> torch.nn.Module:
    TimesFm2_5ModelForPrediction = _require_timesfm_transformers_class()
    return TimesFm2_5ModelForPrediction.from_pretrained(TIMESFM_MODEL_ID).to(device).eval()


def finetune_timesfm(
    train_df: pd.DataFrame,
    *,
    dataset_name: str,
    context_length: int,
    output_length: int,
    learning_rate: float,
    num_steps: int,
    batch_size: int,
    device: torch.device,
    train_mode: Literal["head", "full"],
) -> torch.nn.Module:
    if device.type == "cuda":
        reset_cuda_memory_stats()
    model = load_timesfm_transformers(device)
    if train_mode == "head":
        _freeze_timesfm_for_head_tuning(model)
    dataset = TargetWindowDataset(train_df, context_length=context_length, prediction_length=output_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    print(
        f"[timesfm_ft] fine-tuning windows={len(dataset)}, batch_size={batch_size}, "
        f"context={context_length}, output={output_length}, mode={train_mode}"
    )
    log_cuda_memory("before-timesfm-finetune")
    _run_finetune_loop(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        num_steps=num_steps,
        model_name="timesfm_ft",
        timesfm_mode=True,
        context_length=context_length,
    )
    log_cuda_memory("after-timesfm-finetune")
    save_dir = MODEL_DIR / dataset_name / f"timesfm_ft_{train_mode}"
    model.save_pretrained(save_dir)
    return model.eval()


def evaluate_zero_shot(
    model_name: Literal["chronos1", "chronosbolt", "chronos2"],
    splits: list[SeriesSplit],
    context_length: int,
    device_map: str,
    batch_size: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    if model_name == "chronos1":
        pipeline = ChronosPipeline.from_pretrained(CHRONOS1_MODEL_ID, device_map=device_map)
        predictor = lambda split: _predict_chronos1(pipeline, split, context_length, batch_size)
    elif model_name == "chronosbolt":
        pipeline = ChronosBoltPipeline.from_pretrained(CHRONOS_BOLT_MODEL_ID, device_map=device_map)
        predictor = lambda split: _predict_chronosbolt(pipeline, split, context_length, batch_size)
    elif model_name == "chronos2":
        pipeline = Chronos2Pipeline.from_pretrained(CHRONOS2_MODEL_ID, device_map=device_map)
        predictor = lambda split: _predict_chronos2(pipeline, split, context_length, batch_size)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    rows = []
    all_true = []
    all_pred = []
    sku_rows = []
    for split in splits:
        future_slice = split.future.iloc[:HORIZON_DAYS].copy()
        y_true = _to_numpy(future_slice[STANDARD_TARGET_COL]).astype(np.float32)
        y_pred = predictor(split)
        all_true.append(y_true)
        all_pred.append(y_pred)
        sku_metrics = compute_metrics(y_true, y_pred)
        sku_rows.append({"model_name": model_name, "sku_id": split.sku_id, **sku_metrics})
        rows.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "sku_id": split.sku_id,
                    "order_date": future_slice[STANDARD_DATE_COL].to_numpy(),
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        )

    forecast_df = pd.concat(rows, ignore_index=True)
    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)
    overall = compute_metrics(y_true_all, y_pred_all)
    metrics = {
        "model_name": model_name,
        **overall,
    }
    return forecast_df, metrics, pd.DataFrame(sku_rows)


def finetune_chronos2(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    dataset_name: str,
    feature_spec: TsfmFeatureSpec,
    context_length: int,
    learning_rate: float,
    num_steps: int,
    batch_size: int,
    finetune_mode: Literal["full", "lora"],
    device_map: str,
) -> Chronos2Pipeline:
    if device_map == "cuda":
        reset_cuda_memory_stats()
    base = Chronos2Pipeline.from_pretrained(CHRONOS2_MODEL_ID, device_map=device_map)
    train_inputs = build_chronos2_finetune_inputs(train_df, feature_spec)
    val_inputs = build_chronos2_finetune_inputs(val_df, feature_spec)
    print(
        f"[chronos2_ft] fine-tuning series: train={len(train_inputs)}, "
        f"val={len(val_inputs)}, batch_size={batch_size}, mode={finetune_mode}"
    )
    log_cuda_memory("before-finetune")
    try:
        tuned = base.fit(
            train_inputs,
            prediction_length=HORIZON_DAYS,
            validation_inputs=val_inputs,
            finetune_mode=finetune_mode,
            learning_rate=learning_rate,
            num_steps=num_steps,
            batch_size=batch_size,
            context_length=context_length,
            output_dir=MODEL_DIR / dataset_name / f"chronos2_{finetune_mode}",
            min_past=LOOKBACK_DAYS,
            finetuned_ckpt_name=f"{dataset_name}_chronos2_{finetune_mode}_ckpt",
            disable_data_parallel=True,
        )
    finally:
        log_cuda_memory("after-finetune")
    save_dir = MODEL_DIR / dataset_name / f"chronos2_{finetune_mode}" / "saved_pipeline"
    tuned.save_pretrained(save_dir)
    return tuned


def predict_with_finetuned_chronos2(
    pipeline: Chronos2Pipeline,
    splits: list[SeriesSplit],
    *,
    context_length: int,
    batch_size: int,
    model_name: str,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows = []
    all_true = []
    all_pred = []
    sku_rows = []
    for split in splits:
        future_slice = split.future.iloc[:HORIZON_DAYS].copy()
        y_true = _to_numpy(future_slice[STANDARD_TARGET_COL]).astype(np.float32)
        y_pred = _predict_chronos2(pipeline, split, context_length, batch_size)
        all_true.append(y_true)
        all_pred.append(y_pred)
        sku_metrics = compute_metrics(y_true, y_pred)
        sku_rows.append({"model_name": model_name, "sku_id": split.sku_id, **sku_metrics})
        rows.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "sku_id": split.sku_id,
                    "order_date": future_slice[STANDARD_DATE_COL].to_numpy(),
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        )
    forecast_df = pd.concat(rows, ignore_index=True)
    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)
    overall = compute_metrics(y_true_all, y_pred_all)
    metrics = {
        "model_name": model_name,
        **overall,
    }
    return forecast_df, metrics, pd.DataFrame(sku_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-series foundation model zero-shot and fine-tuning experiments.")
    parser.add_argument("--dataset", choices=["demand", "inventory", "leadtime", "all"], default="demand")
    parser.add_argument(
        "--model",
        choices=["chronos1", "chronosbolt", "chronos2", "chronos2_ft", "ttm", "ttm_ft", "timesfm", "timesfm_ft", "all"],
        default="all",
    )
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--context-length", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--ttm-context-length", type=int, default=TTM_CONTEXT_LENGTH)
    parser.add_argument("--timesfm-context-length", type=int, default=TIMESFM_CONTEXT_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--finetune-mode", choices=["full", "lora"], default="lora")
    parser.add_argument("--timesfm-train-mode", choices=["head", "full"], default="head")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if args.batch_size > MAX_BATCH_SIZE:
        print(f"batch size capped from {args.batch_size} to {MAX_BATCH_SIZE}")
        args.batch_size = MAX_BATCH_SIZE
    device_map = resolve_device_map(args.device)
    torch_device = torch.device(device_map)
    print(f"using foundation model device: {device_map}")
    for dataset_name in resolve_dataset_names(args.dataset):
        config = get_dataset_config(dataset_name)
        print(f"foundation dataset: {dataset_name}")
        train_df, val_df, test_df = load_daily_frames(config)
        feature_spec = build_feature_spec(config, train_df, val_df, test_df)
        print(
            f"[{dataset_name}:features] past_numeric={len(feature_spec.numeric_past_covariates)} "
            f"future_numeric={len(feature_spec.known_future_numeric_covariates)} "
            f"categorical={len(feature_spec.categorical_covariates)}"
        )

        zero_shot_jobs: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
        if args.split in {"val", "both"}:
            zero_shot_jobs.append(("val", train_df, val_df))
        if args.split in {"test", "both"}:
            zero_shot_jobs.append(("test", pd.concat([train_df, val_df], ignore_index=True), test_df))

        min_history = min_history_length(zero_shot_jobs)
        ttm_context = args.ttm_context_length
        can_run_ttm = min_history >= args.ttm_context_length
        can_finetune_ttm = min_history >= args.ttm_context_length + TTM_OUTPUT_LENGTH
        timesfm_context = effective_context_length(args.timesfm_context_length, zero_shot_jobs)
        timesfm_ft_context = effective_training_context_length(args.timesfm_context_length, zero_shot_jobs, output_length=TIMESFM_OUTPUT_LENGTH)
        if args.model in {"ttm", "ttm_ft"} and not can_run_ttm:
            print(
                f"[{dataset_name}:ttm] skipped: min history {min_history} is shorter than "
                f"pretrained TTM context {args.ttm_context_length}"
            )
        elif args.model == "ttm_ft" and not can_finetune_ttm:
            print(
                f"[{dataset_name}:ttm_ft] skipped: min history {min_history} is shorter than "
                f"context+output {args.ttm_context_length + TTM_OUTPUT_LENGTH}"
            )
        if timesfm_context != args.timesfm_context_length:
            print(f"[{dataset_name}:timesfm] context capped from {args.timesfm_context_length} to {timesfm_context}")

        if args.model in {"chronos1", "all"}:
            for split_name, history_df, future_df in zero_shot_jobs:
                splits = build_eval_splits(history_df, future_df, feature_spec)
                forecast_df, metrics, sku_df = evaluate_zero_shot("chronos1", splits, args.context_length, device_map, args.batch_size)
                save_chronos_outputs(dataset_name, "chronos1", split_name, forecast_df, metrics, sku_df)
                print(f"[{dataset_name}:chronos1][{split_name}] mae={metrics['mae']:.6f} mse={metrics['mse']:.6f} rmse={metrics['rmse']:.6f} mape={metrics['mape']:.6f} r2={metrics['r2']:.6f}")

        if args.model in {"chronosbolt", "all"}:
            for split_name, history_df, future_df in zero_shot_jobs:
                splits = build_eval_splits(history_df, future_df, feature_spec)
                forecast_df, metrics, sku_df = evaluate_zero_shot("chronosbolt", splits, args.context_length, device_map, args.batch_size)
                save_chronos_outputs(dataset_name, "chronosbolt", split_name, forecast_df, metrics, sku_df)
                print(f"[{dataset_name}:chronosbolt][{split_name}] mae={metrics['mae']:.6f} mse={metrics['mse']:.6f} rmse={metrics['rmse']:.6f} mape={metrics['mape']:.6f} r2={metrics['r2']:.6f}")

        if args.model in {"chronos2", "all"}:
            for split_name, history_df, future_df in zero_shot_jobs:
                splits = build_eval_splits(history_df, future_df, feature_spec)
                forecast_df, metrics, sku_df = evaluate_zero_shot("chronos2", splits, args.context_length, device_map, args.batch_size)
                save_chronos_outputs(dataset_name, "chronos2", split_name, forecast_df, metrics, sku_df)
                print(f"[{dataset_name}:chronos2][{split_name}] mae={metrics['mae']:.6f} mse={metrics['mse']:.6f} rmse={metrics['rmse']:.6f} mape={metrics['mape']:.6f} r2={metrics['r2']:.6f}")

        if args.model in {"chronos2_ft", "all"}:
            tuned = None
            try:
                tuned = finetune_chronos2(
                    train_df,
                    val_df,
                    dataset_name=dataset_name,
                    feature_spec=feature_spec,
                    context_length=args.context_length,
                    learning_rate=args.learning_rate,
                    num_steps=args.num_steps,
                    batch_size=args.batch_size,
                    finetune_mode=args.finetune_mode,
                    device_map=device_map,
                )
                model_slug = f"chronos2_ft_{args.finetune_mode}"

                for split_name, history_df, future_df in zero_shot_jobs:
                    splits = build_eval_splits(history_df, future_df, feature_spec)
                    forecast_df, metrics, sku_df = predict_with_finetuned_chronos2(
                        tuned,
                        splits,
                        context_length=args.context_length,
                        batch_size=args.batch_size,
                        model_name=model_slug,
                    )
                    save_chronos_outputs(dataset_name, model_slug, split_name, forecast_df, metrics, sku_df)
                    print(f"[{dataset_name}:chronos2_ft:{args.finetune_mode}][{split_name}] mae={metrics['mae']:.6f} mse={metrics['mse']:.6f} rmse={metrics['rmse']:.6f} mape={metrics['mape']:.6f} r2={metrics['r2']:.6f}")
                print(f"[{dataset_name}:chronos2_ft:{args.finetune_mode}] saved: {chronos_model_dir(dataset_name, model_slug)}")
            finally:
                del tuned
                cleanup_cuda_memory(f"after-{dataset_name}-chronos2-ft")

        if args.model == "ttm" and can_run_ttm:
            model = None
            try:
                model = load_ttm(torch_device)
                for split_name, history_df, future_df in zero_shot_jobs:
                    splits = build_eval_splits(history_df, future_df, feature_spec)
                    forecast_df, metrics, sku_df = evaluate_torch_foundation_model(
                        "ttm",
                        splits,
                        lambda split: _predict_ttm(model, split, torch_device, ttm_context),
                    )
                    save_chronos_outputs(dataset_name, "ttm", split_name, forecast_df, metrics, sku_df)
                    print(f"[{dataset_name}:ttm][{split_name}] mae={metrics['mae']:.6f} mse={metrics['mse']:.6f} rmse={metrics['rmse']:.6f} mape={metrics['mape']:.6f} r2={metrics['r2']:.6f}")
            finally:
                del model
                cleanup_cuda_memory(f"after-{dataset_name}-ttm")

        if args.model == "ttm_ft" and can_finetune_ttm:
            model = None
            try:
                model = finetune_ttm(
                    train_df,
                    dataset_name=dataset_name,
                    context_length=ttm_context,
                    output_length=TTM_OUTPUT_LENGTH,
                    learning_rate=args.learning_rate,
                    num_steps=args.num_steps,
                    batch_size=args.batch_size,
                    device=torch_device,
                )
                for split_name, history_df, future_df in zero_shot_jobs:
                    splits = build_eval_splits(history_df, future_df, feature_spec)
                    forecast_df, metrics, sku_df = evaluate_torch_foundation_model(
                        "ttm_ft",
                        splits,
                        lambda split: _predict_ttm(model, split, torch_device, ttm_context),
                    )
                    save_chronos_outputs(dataset_name, "ttm_ft", split_name, forecast_df, metrics, sku_df)
                    print(f"[{dataset_name}:ttm_ft][{split_name}] mae={metrics['mae']:.6f} mse={metrics['mse']:.6f} rmse={metrics['rmse']:.6f} mape={metrics['mape']:.6f} r2={metrics['r2']:.6f}")
                print(f"[{dataset_name}:ttm_ft] saved: {MODEL_DIR / dataset_name / 'ttm_ft'}")
            finally:
                del model
                cleanup_cuda_memory(f"after-{dataset_name}-ttm-ft")

        if args.model in {"timesfm", "all"}:
            model = None
            try:
                model = load_timesfm(timesfm_context)
                for split_name, history_df, future_df in zero_shot_jobs:
                    splits = build_eval_splits(history_df, future_df, feature_spec)
                    forecast_df, metrics, sku_df = evaluate_torch_foundation_model(
                        "timesfm",
                        splits,
                        lambda split: _predict_timesfm(model, split, timesfm_context),
                    )
                    save_chronos_outputs(dataset_name, "timesfm", split_name, forecast_df, metrics, sku_df)
                    print(f"[{dataset_name}:timesfm][{split_name}] mae={metrics['mae']:.6f} mse={metrics['mse']:.6f} rmse={metrics['rmse']:.6f} mape={metrics['mape']:.6f} r2={metrics['r2']:.6f}")
            finally:
                del model
                cleanup_cuda_memory(f"after-{dataset_name}-timesfm")

        if args.model == "timesfm_ft":
            model = None
            try:
                model = finetune_timesfm(
                    train_df,
                    dataset_name=dataset_name,
                    context_length=timesfm_ft_context,
                    output_length=TIMESFM_OUTPUT_LENGTH,
                    learning_rate=args.learning_rate,
                    num_steps=args.num_steps,
                    batch_size=args.batch_size,
                    device=torch_device,
                    train_mode=args.timesfm_train_mode,
                )
                model_slug = f"timesfm_ft_{args.timesfm_train_mode}"
                for split_name, history_df, future_df in zero_shot_jobs:
                    splits = build_eval_splits(history_df, future_df, feature_spec)
                    forecast_df, metrics, sku_df = evaluate_torch_foundation_model(
                        model_slug,
                        splits,
                        lambda split: _predict_timesfm_transformers(model, split, torch_device, timesfm_ft_context),
                    )
                    save_chronos_outputs(dataset_name, model_slug, split_name, forecast_df, metrics, sku_df)
                    print(f"[{dataset_name}:{model_slug}][{split_name}] mae={metrics['mae']:.6f} mse={metrics['mse']:.6f} rmse={metrics['rmse']:.6f} mape={metrics['mape']:.6f} r2={metrics['r2']:.6f}")
                print(f"[{dataset_name}:{model_slug}] saved: {MODEL_DIR / dataset_name / model_slug}")
            finally:
                del model
                cleanup_cuda_memory(f"after-{dataset_name}-timesfm-ft")


if __name__ == "__main__":
    main()
