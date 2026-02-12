from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import polars as pl
import mlflow
from xgboost import XGBRegressor

from .utils import SplitConfig, metrics, time_split


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT/"data"/"processed"/"power_hourly.parquet"
MODEL_DIR = REPO_ROOT/"models"
MODEL_PATH = MODEL_DIR/"xgb_power_h24.joblib"


@dataclass(frozen=True)
class FeatConfig:
    horizon: int = 24
    lags: tuple[int, ...] = (24, 48, 72, 168)  # 1d,2d,3d,1w
    rolling_windows: tuple[int, ...] = (24, 168)  # 1d,1w


def build_features(df: pl.DataFrame, fc: FeatConfig) -> pl.DataFrame:
    """Build lag, rolling, and calendar features for time series forecasting."""
    # Target: y(t + horizon)
    df = df.with_columns(pl.col("y").shift(-fc.horizon).alias("y_target"))

    # Lag features: capture autocorrelation and seasonality
    for lag in fc.lags:
        df = df.with_columns(pl.col("y").shift(lag).alias(f"lag_{lag}"))

    # Rolling statistics: use shift(1) to prevent data leakage
    for w in fc.rolling_windows:
        y_shifted = pl.col("y").shift(1)
        df = df.with_columns([
            y_shifted.rolling_mean(window_size=w, min_periods=w).alias(f"roll_mean_{w}"),
            y_shifted.rolling_std(window_size=w, min_periods=w).alias(f"roll_std_{w}"),
        ])

    # Calendar features
    df = df.with_columns([
        pl.col("dt").dt.hour().alias("hour"),
        pl.col("dt").dt.weekday().alias("weekday"),  # 1..7
        pl.col("dt").dt.month().alias("month"),
    ])

    # Drop rows with nulls caused by shifts/rolling/target
    feature_cols = [c for c in df.columns if c not in ("dt", "y", "y_target")]
    df = df.drop_nulls(feature_cols + ["y_target"])
    return df


def to_xy(df: pl.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(feature_cols).to_numpy()
    y = df["y_target"].to_numpy()
    return X, y


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Run features/build_dataset.py first.")

    fc = FeatConfig(horizon=24)
    split_cfg = SplitConfig()

    df = pl.read_parquet(DATA_PATH).sort("dt")
    df_feat = build_features(df, fc)

    feature_cols = [c for c in df_feat.columns if c not in ("dt", "y", "y_target")]

    train_df, val_df, test_df = time_split(df_feat, split_cfg)

    X_train, y_train = to_xy(train_df, feature_cols)
    X_val, y_val = to_xy(val_df, feature_cols)
    X_test, y_test = to_xy(test_df, feature_cols)

    # Naive baseline for comparison: y_hat(t+24) = y(t)
    yhat_val_naive = val_df["y"].to_numpy()
    yhat_test_naive = test_df["y"].to_numpy()

    naive_val = metrics(y_val, yhat_val_naive)
    naive_test = metrics(y_test, yhat_test_naive)

    model = XGBRegressor(
        n_estimators=800,               # Numero di alberi (boosting rounds)
        max_depth=6,                    # Profondità massima di ogni albero
        learning_rate=0.05,             # Tasso di apprendimento (shrinkage per round)
        subsample=0.9,                  # Frazione di righe campionate per ogni albero
        colsample_bytree=0.9,           # Frazione di feature campionate per ogni albero
        objective="reg:squarederror",   # Funzione di loss: errore quadratico medio
        random_state=42,                # Seed per riproducibilità
        n_jobs=-1                       # Usa tutti i core CPU disponibili
    )

    mlflow.set_experiment("Power_Forecasting_h24")

    with mlflow.start_run(run_name="xgb_h24_v1"):
        mlflow.log_params({
            "horizon": fc.horizon,
            "lags": list(fc.lags),
            "rolling_windows": list(fc.rolling_windows),
            "train_frac": split_cfg.train_frac,
            "val_frac": split_cfg.val_frac,
            "model": "XGBRegressor",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
        })

        mlflow.log_metrics({
            "naive_val_mae": naive_val["mae"],
            "naive_val_rmse": naive_val["rmse"],
            "naive_test_mae": naive_test["mae"],
            "naive_test_rmse": naive_test["rmse"],
        })

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        yhat_val = model.predict(X_val)
        yhat_test = model.predict(X_test)

        m_val = metrics(y_val, yhat_val)
        m_test = metrics(y_test, yhat_test)

        mlflow.log_metrics({
            "val_mae": m_val["mae"],
            "val_rmse": m_val["rmse"],
            "test_mae": m_test["mae"],
            "test_rmse": m_test["rmse"],
        })

        # Salvataggio modello + metadata
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": model,
            "feature_cols": feature_cols,
            "horizon": fc.horizon,
        }
        joblib.dump(bundle, MODEL_PATH)

        mlflow.log_artifact(str(MODEL_PATH))

        print("[train] Done.")
        print(f"naive(val) MAE={naive_val['mae']:.4f} RMSE={naive_val['rmse']:.4f}")
        print(f"model(val) MAE={m_val['mae']:.4f} RMSE={m_val['rmse']:.4f}")
        print(f"naive(test) MAE={naive_test['mae']:.4f} RMSE={naive_test['rmse']:.4f}")
        print(f"model(test) MAE={m_test['mae']:.4f} RMSE={m_test['rmse']:.4f}")
        print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()