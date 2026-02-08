from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT/"data"/"processed"/"power_hourly.parquet"


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15


def time_split(df: pl.DataFrame, cfg: SplitConfig) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    n = df.height
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    
    train = df.slice(0, n_train)
    val = df.slice(n_train, n_val)
    test = df.slice(n_train + n_val, n - (n_train + n_val))
    return train, val, test


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}


def main(horizon: int = 24) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Run features/build_dataset.py first.")

    df = pl.read_parquet(DATA_PATH).sort("dt")
    # Baseline: y_hat(t) = y(t-horizon)
    df = df.with_columns(pl.col("y").shift(horizon).alias("y_hat_naive_24h"))

    # Per valutare baseline su val/test, dobbiamo togliere righe con pred null
    df_eval = df.drop_nulls(["y", "y_hat_naive_24h"])

    train, val, test = time_split(df_eval, SplitConfig())

    y_val = val["y"].to_numpy()
    yhat_val = val["y_hat_naive_24h"].to_numpy()

    y_test = test["y"].to_numpy()
    yhat_test = test["y_hat_naive_24h"].to_numpy()

    m_val = metrics(y_val, yhat_val)
    m_test = metrics(y_test, yhat_test)

    print("[baseline] Naive 24h results")
    print(f"  val  | MAE={m_val['mae']:.4f} RMSE={m_val['rmse']:.4f} | rows={val.height}")
    print(f"  test | MAE={m_test['mae']:.4f} RMSE={m_test['rmse']:.4f} | rows={test.height}")


if __name__ == "__main__":
    main(horizon=24)