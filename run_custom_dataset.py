from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Import existing model architecture and helpers
from model import build_recurrent_model, safe_arima_forecast, HORIZON

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "artifacts" / "results" / "custom"
CHRONOS_MODEL_ID = "amazon/chronos-bolt-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast on custom CSV time-series data.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to your custom CSV file.")
    parser.add_argument("--date-col", type=str, required=True, help="Column name for timestamps/dates.")
    parser.add_argument("--target-col", type=str, required=True, help="Column name for the forecasting target value.")
    parser.add_argument("--series-col", type=str, default=None, help="Column name for the series identifier (optional).")
    
    parser.add_argument("--model", choices=["arima", "lstm", "tcn", "chronosbolt"], default="lstm", help="Forecasting model to use.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs for deep learning models.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training/inference.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for neural networks.")
    parser.add_argument("--hidden-size", type=int, default=48, help="Recurrent hidden size or TCN channels.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of layers in neural networks.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    
    parser.add_argument("--lookback", type=int, default=30, help="Lookback window size (days/periods).")
    parser.add_argument("--horizon", type=int, default=7, help="Prediction horizon size (days/periods).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Compute device to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class CustomWindowDataset(Dataset):
    def __init__(self, x_num: np.ndarray, y: np.ndarray):
        self.x_num = x_num.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.x_num)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x_num[idx]), torch.from_numpy(self.y[idx])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    
    # Calculate R2 score manually
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["day_of_week"] = dt.dt.dayofweek.astype(np.float32)
    df["day_of_month"] = dt.dt.day.astype(np.float32)
    df["month"] = dt.dt.month.astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0).astype(np.float32)
    return df


def build_windows(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    series_col: str,
    lookback: int,
    horizon: int,
    features: list[str]
) -> dict[str, np.ndarray]:
    series_ids = sorted(df[series_col].astype(str).unique())
    series_to_code = {sid: idx for idx, sid in enumerate(series_ids)}
    
    x_list, y_list, code_list, origin_dates, target_starts, target_ends = [], [], [], [], [], []
    
    for sid, group in df.groupby(series_col):
        group = group.sort_values(date_col).reset_index(drop=True)
        if len(group) < lookback + horizon:
            continue
        
        x_feat = group[features].to_numpy(dtype=np.float32)
        y_feat = group[target_col].to_numpy(dtype=np.float32)
        dates = group[date_col].to_numpy(dtype="datetime64[D]")
        
        max_start = len(group) - horizon
        for end_idx in range(lookback - 1, max_start):
            x_list.append(x_feat[end_idx - lookback + 1 : end_idx + 1])
            y_list.append(y_feat[end_idx + 1 : end_idx + 1 + horizon])
            code_list.append(series_to_code[str(sid)])
            origin_dates.append(dates[end_idx])
            target_starts.append(dates[end_idx + 1])
            target_ends.append(dates[end_idx + horizon])
            
    if not x_list:
        raise ValueError("Insufficient data to build lookback/horizon windows. Make sure some series have length >= lookback + horizon.")
        
    return {
        "X_num": np.stack(x_list),
        "y": np.stack(y_list),
        "series_code": np.array(code_list, dtype=np.int64),
        "series_id_values": np.array(series_ids, dtype="U"),
        "origin_date": np.array(origin_dates),
        "target_start": np.array(target_starts),
        "target_end": np.array(target_ends),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = resolve_device(args.device)
    print(f"Loading custom dataset: {args.csv_path}")
    
    # Load and preprocess
    df = pd.read_csv(args.csv_path, low_memory=False)
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.dropna(subset=[args.date_col]).copy()
    
    if args.series_col is None:
        df["__series_id"] = "overall"
        series_col = "__series_id"
    else:
        series_col = args.series_col
        df[series_col] = df[series_col].astype(str)
        
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce").fillna(0.0)
    df = df.sort_values([series_col, args.date_col]).reset_index(drop=True)
    
    # Add simple calendar features
    df = add_calendar_features(df, args.date_col)
    
    features = [args.target_col, "dow_sin", "dow_cos", "day_of_month", "month"]
    print(f"Extracted features: {features}")
    
    # Train / Val / Test chronological split based on unique dates
    unique_dates = pd.Index(df[args.date_col].drop_duplicates().sort_values())
    n_dates = len(unique_dates)
    if n_dates < args.lookback + args.horizon + 3:
        raise ValueError(f"Too few unique date points ({n_dates}) for lookback={args.lookback} and horizon={args.horizon}")
        
    train_end_idx = round(n_dates * 0.70) - 1
    val_end_idx = round(n_dates * 0.85) - 1
    train_end = unique_dates[train_end_idx]
    val_end = unique_dates[val_end_idx]
    
    print(f"Data splits: Train <= {train_end.date()}, Val <= {val_end.date()}, Test > {val_end.date()}")
    
    # Build window arrays
    w = build_windows(df, args.date_col, args.target_col, series_col, args.lookback, args.horizon, features)
    
    # Split index masks
    target_ends = w["target_end"]
    train_mask = target_ends <= np.datetime64(train_end.date())
    val_mask = (target_ends > np.datetime64(train_end.date())) & (target_ends <= np.datetime64(val_end.date()))
    test_mask = target_ends > np.datetime64(val_end.date())
    
    X_num_train, y_train = w["X_num"][train_mask], w["y"][train_mask]
    X_num_val, y_val = w["X_num"][val_mask], w["y"][val_mask]
    X_num_test, y_test = w["X_num"][test_mask], w["y"][test_mask]
    
    test_series_codes = w["series_code"][test_mask]
    test_series_ids = w["series_id_values"]
    test_origin_dates = w["origin_date"][test_mask]
    
    print(f"Window shapes: Train X={X_num_train.shape}, Val X={X_num_val.shape}, Test X={X_num_test.shape}")
    
    if len(X_num_train) == 0 or len(X_num_test) == 0:
        raise ValueError("Train or Test split contains 0 windows. Please ensure you have sufficient historical dates.")
        
    # Scale features using Train statistics
    flat_train = X_num_train.reshape(-1, X_num_train.shape[-1])
    x_min = np.nanmin(flat_train, axis=0)
    x_max = np.nanmax(flat_train, axis=0)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0  # Avoid division by zero
    
    def scale(x: np.ndarray) -> np.ndarray:
        return (x - x_min.reshape(1, 1, -1)) / x_range.reshape(1, 1, -1)
        
    scaled_train_X = scale(X_num_train)
    scaled_val_X = scale(X_num_val)
    scaled_test_X = scale(X_num_test)
    
    predictions = None
    
    if args.model == "arima":
        print("Running ARIMA forecast...")
        arima_preds = []
        series_ids_val = w["series_id_values"]
        
        # Build historic frame for statsmodels ARIMA lookup
        hist_df = df[df[args.date_col] <= val_end]
        test_df = df[df[args.date_col] > val_end]
        
        # For each test window, predict the horizon
        for idx in range(len(X_num_test)):
            scode = test_series_codes[idx]
            sid = test_series_ids[scode]
            origin_date = test_origin_dates[idx]
            
            # Extract history up to the origin date
            sub_hist = df[(df[series_col] == sid) & (df[args.date_col] <= pd.Timestamp(origin_date))]
            series_history = sub_hist.sort_values(args.date_col)[args.target_col].to_numpy(dtype=np.float64)
            
            pred = safe_arima_forecast(series_history, args.horizon)
            arima_preds.append(pred)
            
        predictions = np.stack(arima_preds)
        
    elif args.model in ["lstm", "tcn"]:
        print(f"Training {args.model.upper()} model on {device}...")
        train_loader = DataLoader(CustomWindowDataset(scaled_train_X, y_train), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(CustomWindowDataset(scaled_val_X, y_val), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(CustomWindowDataset(scaled_test_X, y_test), batch_size=args.batch_size, shuffle=False)
        
        numeric_size = scaled_train_X.shape[-1]
        model = build_recurrent_model(
            model_name=args.model,
            numeric_size=numeric_size,
            static_size=0,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            horizon=args.horizon
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        
        best_val_loss = math.inf
        best_state = None
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(bx)
            train_loss /= len(scaled_train_X)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = model(bx)
                    loss = criterion(pred, by)
                    val_loss += loss.item() * len(bx)
            val_loss /= max(len(scaled_val_X), 1)
            
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
        if best_state is not None:
            model.load_state_dict(best_state)
            
        # Test evaluation
        model.eval()
        test_preds = []
        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(device)
                pred = model(bx)
                test_preds.append(pred.cpu().numpy())
        predictions = np.concatenate(test_preds, axis=0)
        
    elif args.model == "chronosbolt":
        print("Running Chronos-Bolt zero-shot inference...")
        try:
            from chronos import ChronosBoltPipeline
        except ImportError:
            raise RuntimeError("Chronos library not found. Ensure chronos-forecasting is installed.")
            
        pipeline = ChronosBoltPipeline.from_pretrained(CHRONOS_MODEL_ID, device_map=args.device if args.device != "auto" else "auto")
        
        # Prepare list of target histories
        inputs = []
        for idx in range(len(X_num_test)):
            # Use target_col (index 0) from the lookback feature window
            inputs.append(torch.tensor(X_num_test[idx, :, 0], dtype=torch.float32))
            
        # Batched inference
        preds_list = []
        for i in range(0, len(inputs), args.batch_size):
            batch = inputs[i : i + args.batch_size]
            forecast = pipeline.predict(batch, prediction_length=args.horizon, limit_prediction_length=False)
            # Take the median quantile (index 4 out of 9 quantiles)
            if isinstance(forecast, torch.Tensor):
                forecast = forecast.cpu().numpy()
            preds_list.append(forecast[:, 4])
            
        predictions = np.concatenate(preds_list, axis=0)
        
    # Evaluate and print results
    metrics = compute_metrics(y_test, predictions)
    print("\n" + "="*40)
    print(f"Evaluation results for {args.model.upper()} on Test split:")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print("="*40)
    
    # Save outputs
    # Save predictions
    out_df_list = []
    for idx in range(len(X_num_test)):
        scode = test_series_codes[idx]
        sid = test_series_ids[scode]
        origin = test_origin_dates[idx]
        
        for h in range(args.horizon):
            out_df_list.append({
                "series_id": sid,
                "origin_date": origin,
                "horizon_step": h + 1,
                "y_true": float(y_test[idx, h]),
                "y_pred": float(predictions[idx, h]),
            })
            
    out_df = pd.DataFrame(out_df_list)
    predictions_path = RESULT_DIR / f"{args.model}_predictions.csv"
    metrics_path = RESULT_DIR / f"{args.model}_metrics.json"
    
    out_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    print(f"\nSaved predictions to: {predictions_path}")
    print(f"Saved metrics to: {metrics_path}\n")


if __name__ == "__main__":
    main()
