from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA
from torch import nn


HORIZON = 7


class RecurrentForecaster(nn.Module):
    def __init__(
        self,
        numeric_size: int,
        static_size: int = 0,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
        cell: Literal["rnn", "lstm", "gru"] = "lstm",
        horizon: int = HORIZON,
    ):
        super().__init__()
        self.cell = cell
        self.numeric_size = numeric_size
        self.static_size = static_size
        input_size = numeric_size + static_size
        if cell == "rnn":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="tanh",
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif cell == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x_num: torch.Tensor, x_static: torch.Tensor | None = None) -> torch.Tensor:
        if x_static is not None:
            if x_static.dim() == 2:
                x_static = x_static.unsqueeze(1).expand(-1, x_num.size(1), -1)
            x = torch.cat([x_num, x_static], dim=-1)
        else:
            x = x_num
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        numeric_size: int,
        static_size: int = 0,
        hidden_size: int = 48,
        num_layers: int = 1,
        dropout: float = 0.10,
        kernel_size: int = 3,
        horizon: int = HORIZON,
    ):
        super().__init__()
        self.numeric_size = numeric_size
        self.static_size = static_size
        input_size = numeric_size + static_size
        channels = []
        in_channels = input_size
        for layer_idx in range(num_layers):
            out_channels = hidden_size
            channels.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**layer_idx,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*channels)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x_num: torch.Tensor, x_static: torch.Tensor | None = None) -> torch.Tensor:
        if x_static is not None:
            if x_static.dim() == 2:
                x_static = x_static.unsqueeze(1).expand(-1, x_num.size(1), -1)
            x = torch.cat([x_num, x_static], dim=-1)
        else:
            x = x_num
        x = x.transpose(1, 2)
        out = self.network(x)
        last = out[:, :, -1]
        return self.head(last)


def build_recurrent_model(
    model_name: Literal["rnn", "lstm", "gru", "tcn"],
    numeric_size: int,
    static_size: int = 0,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.15,
    horizon: int = HORIZON,
) -> nn.Module:
    if model_name == "tcn":
        return TCNForecaster(
            numeric_size=numeric_size,
            static_size=static_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
        )
    return RecurrentForecaster(
        numeric_size=numeric_size,
        static_size=static_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        cell=model_name,
        horizon=horizon,
    )


def safe_arima_forecast(series: np.ndarray, steps: int) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    if len(series) == 0:
        return np.zeros(steps, dtype=np.float64)
    if np.allclose(series, series[0]):
        return np.full(steps, series[-1], dtype=np.float64)

    transformed = np.log1p(np.clip(series, 0, None))
    candidates = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 1, 0)]
    best_result = None
    best_aic = math.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in candidates:
            try:
                result = ARIMA(
                    transformed,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
                if np.isfinite(result.aic) and result.aic < best_aic:
                    best_aic = result.aic
                    best_result = result
            except Exception:
                continue

    if best_result is None:
        return np.full(steps, series[-1], dtype=np.float64)

    forecast = best_result.forecast(steps=steps)
    forecast = np.expm1(np.asarray(forecast, dtype=np.float64))
    return np.clip(forecast, 0.0, None)
