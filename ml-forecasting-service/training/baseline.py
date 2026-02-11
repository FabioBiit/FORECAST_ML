from __future__ import annotations

from pathlib import Path
import polars as pl

from .utils import SplitConfig, metrics, time_split


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT/"data"/"processed"/"power_hourly.parquet"


def main(horizon: int = 24) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Run features/build_dataset.py first.")

    df = pl.read_parquet(DATA_PATH).sort("dt")

    # Naive baseline: predict t+horizon using value at t
    df = df.with_columns(pl.col("y").shift(horizon).alias("y_hat_naive_24h"))

    # Drop rows with null predictions for evaluation
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