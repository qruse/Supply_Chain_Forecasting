from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from forecasting_data import DatasetConfig, build_window_arrays, get_dataset_config, load_dataset_frame, resolve_dataset_names
from model import HORIZON, build_recurrent_model, safe_arima_forecast


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
RESULT_DIR = ARTIFACT_DIR / "results"

METRICS_PATH = ARTIFACT_DIR / "training_metrics.json"
MAX_BATCH_SIZE = 32
CHECKPOINT_PATTERNS = ("epoch_*.pt", "best_*.pt", "*_history.csv")


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_checkpoint_root(checkpoint_root: Path) -> int:
    removed = 0
    if not checkpoint_root.exists():
        return removed
    for pattern in CHECKPOINT_PATTERNS:
        for path in checkpoint_root.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


@dataclass
class TrainResult:
    model_name: str
    best_val_mse: float
    best_epoch: int | None
    history_path: Path
    checkpoint_dir: Path | None = None
    best_checkpoint_path: Path | None = None


class WindowDataset(Dataset):
    def __init__(
        self,
        x_num: np.ndarray,
        x_static: np.ndarray,
        y: np.ndarray,
        *,
        x_mean: np.ndarray | None = None,
        x_scale: np.ndarray | None = None,
    ):
        self.x_num = x_num.astype(np.float32)
        self.x_static = x_static.astype(np.float32)
        self.y = y.astype(np.float32)
        if x_mean is not None and x_scale is not None:
            self.x_num = (self.x_num - x_mean.reshape(1, 1, -1)) / x_scale.reshape(1, 1, -1)

    def __len__(self) -> int:
        return len(self.x_num)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x_num[idx]),
            torch.from_numpy(self.x_static[idx]),
            torch.from_numpy(self.y[idx]),
        )

def fit_feature_scaler(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = train_x.reshape(-1, train_x.shape[-1])
    data_min = np.nanmin(flat, axis=0).astype(np.float32)
    data_max = np.nanmax(flat, axis=0).astype(np.float32)
    data_range = (data_max - data_min).astype(np.float32)
    data_range = np.where(data_range == 0, 1.0, data_range)
    return data_min, data_range


def scale_x(x: np.ndarray, data_min: np.ndarray, data_range: np.ndarray) -> np.ndarray:
    return (x - data_min.reshape(1, 1, -1)) / data_range.reshape(1, 1, -1)


def mse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def evaluate_recurrent(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x_num, x_static, y in loader:
            x_num = x_num.to(device)
            x_static = x_static.to(device)
            y = y.to(device)
            pred = model(x_num, x_static)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    return mse_np(y_true, y_pred)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    epoch: int,
    val_mse: float,
    train_mse: float,
    model_name: str,
    numeric_size: int,
    static_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    x_min: np.ndarray,
    x_range: np.ndarray,
    dataset_name: str,
    feature_names: np.ndarray,
    static_feature_names: np.ndarray,
) -> None:
    torch.save(
        {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "epoch": epoch,
            "val_mse": val_mse,
            "train_mse": train_mse,
            "model_state_dict": model.state_dict(),
            "numeric_size": numeric_size,
            "static_size": static_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "horizon": HORIZON,
            "feature_min": x_min,
            "feature_range": x_range,
            "feature_names": feature_names,
            "static_feature_names": static_feature_names,
        },
        path,
    )


def train_recurrent_model(
    config: DatasetConfig,
    model_name: Literal["rnn", "lstm", "gru", "tcn"],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    patience: int,
    seed: int,
    device: torch.device,
    clean_checkpoints: bool,
) -> TrainResult:
    set_seed(seed)

    train_arrays = build_window_arrays(config, split="train")
    val_arrays = build_window_arrays(config, split="val")
    train_x_num = train_arrays["X_num"]
    train_x_static = train_arrays["X_static"]
    val_x_num = val_arrays["X_num"]
    val_x_static = val_arrays["X_static"]
    x_min, x_range = fit_feature_scaler(train_x_num)
    train_x_num = scale_x(train_x_num, x_min, x_range)
    val_x_num = scale_x(val_x_num, x_min, x_range)

    train_loader = DataLoader(
        WindowDataset(train_arrays["X_num"], train_arrays["X_static"], train_arrays["y"], x_mean=x_min, x_scale=x_range),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        WindowDataset(val_arrays["X_num"], val_arrays["X_static"], val_arrays["y"], x_mean=x_min, x_scale=x_range),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    numeric_size = train_x_num.shape[-1]
    static_size = train_x_static.shape[-1]
    model = build_recurrent_model(
        model_name=model_name,
        numeric_size=numeric_size,
        static_size=static_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=HORIZON,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    checkpoint_root = CHECKPOINT_DIR / config.name / model_name
    if clean_checkpoints:
        removed = cleanup_checkpoint_root(checkpoint_root)
        if removed:
            print(f"[{config.name}:{model_name.upper()}] cleaned stale checkpoint files: {removed}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    best_epoch = -1
    best_state = None
    best_train_mse = math.inf
    patience_left = patience
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x_num_batch, x_static_batch, y_batch in train_loader:
            x_num_batch = x_num_batch.to(device)
            x_static_batch = x_static_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_num_batch, x_static_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_mse = float(np.mean(train_losses))
        val_mse = evaluate_recurrent(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        epoch_path = checkpoint_root / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            epoch_path,
            model=model,
            epoch=epoch,
            val_mse=val_mse,
            train_mse=train_mse,
            model_name=model_name,
            numeric_size=numeric_size,
            static_size=static_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            x_min=x_min,
            x_range=x_range,
            dataset_name=config.name,
            feature_names=train_arrays["feature_names"],
            static_feature_names=train_arrays["static_feature_names"],
        )

        print(f"[{config.name}:{model_name.upper()}] epoch={epoch:03d} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_train_mse = train_mse
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{config.name}:{model_name.upper()}] early stopping at epoch {epoch:03d}")
                break

    if best_state is None:
        raise RuntimeError(f"No best state recorded for {model_name}")

    model.load_state_dict(best_state)
    best_path = checkpoint_root / f"best_{model_name}_epoch_{best_epoch:03d}.pt"
    save_checkpoint(
        best_path,
        model=model,
        epoch=best_epoch,
        val_mse=best_val,
        train_mse=best_train_mse,
        model_name=model_name,
        numeric_size=numeric_size,
        static_size=static_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        x_min=x_min,
        x_range=x_range,
        dataset_name=config.name,
        feature_names=train_arrays["feature_names"],
        static_feature_names=train_arrays["static_feature_names"],
    )

    history_path = checkpoint_root / f"{model_name}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

    return TrainResult(
        model_name=model_name,
        best_val_mse=best_val,
        best_epoch=best_epoch,
        history_path=history_path,
        checkpoint_dir=checkpoint_root,
        best_checkpoint_path=best_path,
    )


def train_arima(config: DatasetConfig) -> TrainResult:
    train_df = load_dataset_frame(config, "train")
    val_df = load_dataset_frame(config, "val")

    series_ids = sorted(train_df[config.series_col].astype(str).unique())
    if config.arima_max_series is not None:
        series_ids = series_ids[: config.arima_max_series]
    forecasts = []
    val_errors = []

    for series_id in series_ids:
        train_series = (
            train_df.loc[train_df[config.series_col].astype(str) == str(series_id)]
            .sort_values(config.date_col)[config.target_col]
            .to_numpy(dtype=np.float64)
        )
        val_slice = (
            val_df.loc[val_df[config.series_col].astype(str) == str(series_id)]
            .sort_values(config.date_col)[[config.date_col, config.target_col]]
        )
        if len(val_slice) == 0:
            continue

        pred = safe_arima_forecast(train_series, len(val_slice))
        true = val_slice[config.target_col].to_numpy(dtype=np.float64)
        val_errors.append(np.mean((true - pred) ** 2))

        forecasts.append(
            pd.DataFrame(
                {
                    "dataset_name": config.name,
                    "series_id": series_id,
                    "date": val_slice[config.date_col].to_numpy(),
                    "y_true": true,
                    "y_pred": pred,
                }
            )
        )

    if not forecasts:
        raise ValueError(f"No ARIMA validation forecasts for dataset={config.name}")
    forecast_df = pd.concat(forecasts, ignore_index=True)
    result_dir = RESULT_DIR / "baseline" / config.name / "arima"
    result_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = result_dir / "arima_val_forecasts.csv"
    forecast_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    best_val = float(np.mean(val_errors))
    print(f"[{config.name}:ARIMA] val_mse={best_val:.6f}")
    return TrainResult(
        model_name="arima",
        best_val_mse=best_val,
        best_epoch=None,
        history_path=forecast_path,
        checkpoint_dir=None,
        best_checkpoint_path=None,
    )


def save_metrics(dataset_name: str, results: list[TrainResult]) -> dict:
    payload = {
        r.model_name: {
            "best_val_mse": r.best_val_mse,
            "best_epoch": r.best_epoch,
            "history_path": str(r.history_path),
            "checkpoint_dir": str(r.checkpoint_dir) if r.checkpoint_dir else None,
            "best_checkpoint_path": str(r.best_checkpoint_path) if r.best_checkpoint_path else None,
        }
        for r in results
    }
    path = ARTIFACT_DIR / f"training_metrics_{dataset_name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved metrics: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline forecasting models.")
    parser.add_argument("--dataset", choices=["demand", "inventory", "leadtime", "all"], default="demand")
    parser.add_argument("--model", choices=["arima", "rnn", "lstm", "gru", "tcn", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--keep-old-checkpoints",
        action="store_true",
        default=os.environ.get("KEEP_OLD_CHECKPOINTS") == "1",
        help="Do not remove existing epoch/best checkpoints before training each dataset/model.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if args.batch_size > MAX_BATCH_SIZE:
        print(f"batch size capped from {args.batch_size} to {MAX_BATCH_SIZE}")
        args.batch_size = MAX_BATCH_SIZE
    device = resolve_device(args.device)
    print(f"using torch device: {device}")

    all_metrics: dict[str, dict] = {}
    for dataset_name in resolve_dataset_names(args.dataset):
        config = get_dataset_config(dataset_name)
        print(f"training dataset: {dataset_name}")
        results: list[TrainResult] = []
        if args.model in {"arima", "all"}:
            results.append(train_arima(config))
        for model_name in ["rnn", "lstm", "gru", "tcn"]:
            if args.model in {model_name, "all"}:
                results.append(
                    train_recurrent_model(
                        config,
                        model_name=model_name,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        patience=args.patience,
                        seed=args.seed,
                        device=device,
                        clean_checkpoints=not args.keep_old_checkpoints,
                    )
                )
        all_metrics[dataset_name] = save_metrics(dataset_name, results)
        for result in results:
            print(
                f"{dataset_name}:{result.model_name}: val_mse={result.best_val_mse:.6f}"
                + (f", best_epoch={result.best_epoch}" if result.best_epoch is not None else "")
            )
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"saved combined metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
