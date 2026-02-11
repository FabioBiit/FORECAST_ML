from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for time-based train/val/test split."""
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15


def time_split(df: pl.DataFrame, cfg: SplitConfig) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split dataframe into train/val/test based on temporal order."""
    n = df.height
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)

    train = df.slice(0, n_train)
    val = df.slice(n_train, n_val)
    test = df.slice(n_train + n_val, n - (n_train + n_val))
    return train, val, test


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE and RMSE regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}
