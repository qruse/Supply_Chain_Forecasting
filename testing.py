from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

from forecasting_data import DatasetConfig, build_window_arrays, get_dataset_config, load_dataset_frame, resolve_dataset_names
from model import HORIZON, build_recurrent_model, safe_arima_forecast


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
RESULT_ROOT = ARTIFACT_DIR / "results"
BASELINE_RESULTS_DIR = RESULT_ROOT / "baseline"
MAX_BATCH_SIZE = 32

RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def model_results_dir(dataset_name: str, model_name: str) -> Path:
    path = BASELINE_RESULTS_DIR / dataset_name / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def standardize_x(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)


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
    try:
        r2 = float(r2_score(flat_true, flat_pred))
    except Exception:
        r2 = float("nan")
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": r2}


def evaluate_recurrent_checkpoint(
    ckpt_path: Path,
    x_num: np.ndarray,
    x_static: np.ndarray,
    y: np.ndarray,
    series_codes: np.ndarray,
    series_id_values: np.ndarray,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> tuple[dict, pd.DataFrame]:
    if device is None:
        device = resolve_device("auto")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_recurrent_model(
        model_name=ckpt["model_name"],
        numeric_size=int(ckpt["numeric_size"]),
        static_size=int(ckpt["static_size"]),
        hidden_size=int(ckpt["hidden_size"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
        horizon=int(ckpt["horizon"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if "feature_min" in ckpt and "feature_range" in ckpt:
        feature_min = ckpt["feature_min"].astype(np.float32)
        feature_range = ckpt["feature_range"].astype(np.float32)
    else:
        feature_min = ckpt["feature_mean"].astype(np.float32)
        feature_range = ckpt["feature_scale"].astype(np.float32)
    x_num = standardize_x(x_num, feature_min, feature_range)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_num), torch.from_numpy(x_static), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    preds = []
    targets = []
    with torch.no_grad():
        for xb_num, xb_static, yb in loader:
            xb_num = xb_num.to(device)
            xb_static = xb_static.to(device)
            pred = model(xb_num, xb_static)
            preds.append(pred.cpu().numpy())
            targets.append(yb.numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    per_horizon_mae = np.abs(y_true - y_pred).mean(axis=0)
    per_horizon_mse = ((y_true - y_pred) ** 2).mean(axis=0)
    per_horizon_rmse = np.sqrt(per_horizon_mse)
    per_horizon_mape = []
    per_horizon_r2 = []
    for horizon_idx in range(y_true.shape[1]):
        horizon_metrics = compute_metrics(y_true[:, horizon_idx], y_pred[:, horizon_idx])
        per_horizon_mape.append(horizon_metrics["mape"])
        per_horizon_r2.append(horizon_metrics["r2"])
    overall = compute_metrics(y_true, y_pred)
    row = {
        "model_name": ckpt["model_name"],
        "dataset_name": ckpt.get("dataset_name", "demand"),
        "checkpoint": ckpt_path.name,
        "epoch": int(ckpt["epoch"]),
        "is_best": ckpt_path.name.startswith("best_"),
        "test_mae": overall["mae"],
        "test_mse": overall["mse"],
        "test_rmse": overall["rmse"],
        "test_mape": overall["mape"],
        "test_r2": overall["r2"],
    }
    for i, value in enumerate(per_horizon_mae, start=1):
        row[f"mae_t_plus_{i}"] = float(value)
    for i, value in enumerate(per_horizon_mse, start=1):
        row[f"mse_t_plus_{i}"] = float(value)
    for i, value in enumerate(per_horizon_rmse, start=1):
        row[f"rmse_t_plus_{i}"] = float(value)
    for i, value in enumerate(per_horizon_mape, start=1):
        row[f"mape_t_plus_{i}"] = float(value)
    for i, value in enumerate(per_horizon_r2, start=1):
        row[f"r2_t_plus_{i}"] = float(value)

    flat_codes = np.repeat(series_codes, y_true.shape[1])
    series_id_values = np.asarray(series_id_values).astype(str)
    flat_series_ids = series_id_values[flat_codes]
    sku_df = pd.DataFrame(
        {
            "series_id": flat_series_ids,
            "y_true": y_true.reshape(-1),
            "y_pred": y_pred.reshape(-1),
        }
    )
    sku_rows = []
    for series_id, grp in sku_df.groupby("series_id", sort=True):
        metrics = compute_metrics(grp["y_true"].to_numpy(), grp["y_pred"].to_numpy())
        sku_rows.append({"model_name": ckpt["model_name"], "series_id": series_id, **metrics})
    return row, pd.DataFrame(sku_rows)


def evaluate_arima_test(config: DatasetConfig) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    train_df = load_dataset_frame(config, "train")
    val_df = load_dataset_frame(config, "val")
    test_df = load_dataset_frame(config, "test")

    history_df = pd.concat([train_df, val_df], ignore_index=True)
    series_ids = sorted(test_df[config.series_col].astype(str).unique())
    if config.arima_max_series is not None:
        series_ids = series_ids[: config.arima_max_series]
    rows = []
    per_sku_rows = []
    all_true = []
    all_pred = []

    for series_id in series_ids:
        hist_series = (
            history_df.loc[history_df[config.series_col].astype(str) == str(series_id)]
            .sort_values(config.date_col)[config.target_col]
            .to_numpy(dtype=np.float64)
        )
        test_slice = (
            test_df.loc[test_df[config.series_col].astype(str) == str(series_id)]
            .sort_values(config.date_col)[[config.date_col, config.target_col]]
        )
        if len(test_slice) == 0:
            continue

        pred = safe_arima_forecast(hist_series, len(test_slice))
        true = test_slice[config.target_col].to_numpy(dtype=np.float64)
        all_true.append(true)
        all_pred.append(pred)
        metrics = compute_metrics(true, pred)
        per_sku_rows.append({"model_name": "arima", "series_id": series_id, **metrics})
        rows.append(
            pd.DataFrame(
                {
                    "dataset_name": config.name,
                    "series_id": series_id,
                    "date": test_slice[config.date_col].to_numpy(),
                    "y_true": true,
                    "y_pred": pred,
                }
            )
        )

    if not rows:
        raise ValueError(f"No ARIMA test forecasts for dataset={config.name}")
    forecast_df = pd.concat(rows, ignore_index=True)
    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)
    overall = compute_metrics(y_true_all, y_pred_all)
    print(
        f"[{config.name}:ARIMA-TEST] mae={overall['mae']:.6f} mse={overall['mse']:.6f} "
        f"rmse={overall['rmse']:.6f} mape={overall['mape']:.6f} r2={overall['r2']:.6f}"
    )
    return (
        {
            "model_name": "arima",
            "dataset_name": config.name,
            "checkpoint": "arima_test",
            "epoch": None,
            "is_best": True,
            "test_mae": overall["mae"],
            "test_mse": overall["mse"],
            "test_rmse": overall["rmse"],
            "test_mape": overall["mape"],
            "test_r2": overall["r2"],
        },
        forecast_df,
        pd.DataFrame(per_sku_rows),
    )


def save_model_metrics(dataset_name: str, model_name: str, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    if not df.empty and "model_name" in df.columns:
        ordered = ["model_name"] + [col for col in df.columns if col != "model_name"]
        df = df.loc[:, ordered]
    path = model_results_dir(dataset_name, model_name) / f"{model_name}_test_metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_sku_metrics(dataset_name: str, model_name: str, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    if not df.empty and "model_name" in df.columns:
        ordered = ["model_name"] + [col for col in df.columns if col != "model_name"]
        df = df.loc[:, ordered]
    path = model_results_dir(dataset_name, model_name) / f"{model_name}_test_sku_metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_sku_mae_metrics(dataset_name: str, model_name: str, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["model_name", "series_id", "mae"])
    else:
        df = df.loc[:, [col for col in ["model_name", "series_id", "mae"] if col in df.columns]]
    path = model_results_dir(dataset_name, model_name) / f"{model_name}_test_sku_mae.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_forecasts(dataset_name: str, model_name: str, filename: str, df: pd.DataFrame) -> Path:
    path = model_results_dir(dataset_name, model_name) / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def list_recurrent_checkpoints(dataset_name: str, model_name: Literal["rnn", "lstm", "gru", "tcn"]) -> list[Path]:
    model_dir = CHECKPOINT_DIR / dataset_name / model_name
    if not model_dir.exists():
        return []
    epoch_files = sorted(model_dir.glob("epoch_*.pt"))
    best_files = sorted(model_dir.glob("best_*.pt"))
    return epoch_files + best_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints on the test split.")
    parser.add_argument("--dataset", choices=["demand", "inventory", "leadtime", "all"], default="demand")
    parser.add_argument("--model", choices=["arima", "rnn", "lstm", "gru", "tcn", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size > MAX_BATCH_SIZE:
        print(f"batch size capped from {args.batch_size} to {MAX_BATCH_SIZE}")
        args.batch_size = MAX_BATCH_SIZE
    device = resolve_device(args.device)
    print(f"using torch device: {device}")
    for dataset_name in resolve_dataset_names(args.dataset):
        config = get_dataset_config(dataset_name)
        print(f"testing dataset: {dataset_name}")
        test_arrays = build_window_arrays(config, split="test")
        x_test = test_arrays["X_num"]
        x_static_test = test_arrays["X_static"]
        y_test = test_arrays["y"]
        series_codes = test_arrays["series_code"]
        series_id_values = test_arrays["series_id_values"]

        if args.model in {"arima", "all"}:
            arima_row, arima_forecast_df, arima_sku_df = evaluate_arima_test(config)
            save_model_metrics(dataset_name, "arima", [arima_row])
            save_sku_metrics(dataset_name, "arima", arima_sku_df.to_dict(orient="records"))
            save_sku_mae_metrics(dataset_name, "arima", arima_sku_df.to_dict(orient="records"))
            save_forecasts(dataset_name, "arima", "arima_test_forecasts.csv", arima_forecast_df)
            print(f"[{dataset_name}:ARIMA] saved: {model_results_dir(dataset_name, 'arima')}")

        for model_name in ["rnn", "lstm", "gru", "tcn"]:
            if args.model not in {model_name, "all"}:
                continue
            rows = []
            sku_rows = []
            for ckpt_path in list_recurrent_checkpoints(dataset_name, model_name):
                row, sku_df = evaluate_recurrent_checkpoint(
                    ckpt_path,
                    x_test,
                    x_static_test,
                    y_test,
                    series_codes,
                    series_id_values,
                    batch_size=args.batch_size,
                    device=device,
                )
                rows.append(row)
                sku_rows.extend(sku_df.to_dict(orient="records"))
                print(
                    f"[{dataset_name}:{model_name.upper()}][{ckpt_path.name}] "
                    f"mae={row['test_mae']:.6f} mse={row['test_mse']:.6f} "
                    f"rmse={row['test_rmse']:.6f} mape={row['test_mape']:.6f} r2={row['test_r2']:.6f}"
                )
            if rows:
                save_model_metrics(dataset_name, model_name, rows)
                save_sku_metrics(dataset_name, model_name, sku_rows)
                save_sku_mae_metrics(dataset_name, model_name, sku_rows)
                print(f"[{dataset_name}:{model_name.upper()}] saved: {model_results_dir(dataset_name, model_name)}")
            else:
                print(f"[{dataset_name}:{model_name.upper()}] no checkpoints found")


if __name__ == "__main__":
    main()
