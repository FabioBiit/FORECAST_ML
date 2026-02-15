from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT/"models"/"xgb_power_h24.joblib"


@dataclass(frozen=True)
class InferenceConfig:
    # coerente con training/train.py
    horizon: int = 24
    lags: tuple[int, ...] = (24, 48, 72, 168)
    rolling_windows: tuple[int, ...] = (24, 168)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_bundle(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}. Run training/train.py first.")
    bundle = joblib.load(model_path)
    # atteso: {"model": ..., "feature_cols": [...], "horizon": 24}
    if "model" not in bundle or "feature_cols" not in bundle:
        raise ValueError("Invalid model bundle format: expected keys 'model' and 'feature_cols'.")
    return bundle


def _build_feature_row(history: pl.DataFrame, last_dt: datetime, cfg: InferenceConfig) -> dict[str, float]:
    """
    Crea UNA riga di feature per il tempo t = last_dt, coerente col training.
    - Lag: y(t-k)
    - Rolling: su passato escluso il valore corrente (come shift(1) in training)
    - Calendar: hour, weekday (1..7), month
    """
    h = (
        history
        .with_columns(pl.col("dt").cast(pl.Datetime("us", "UTC")))
        .sort("dt")
        .filter(pl.col("dt") <= pl.lit(last_dt))
    )

    if h.is_empty():
        raise ValueError("History is empty after filtering to last_dt.")

    y_series = h["y"].to_list()  # include current y(t) come ultimo

    def lag(k: int) -> float:
        # lag_k(t) = y(t-k) => posizione -1-k
        idx = -1 - k
        if abs(idx) > len(y_series):
            raise ValueError(f"Not enough history for lag_{k}. Need at least {k+1} points before t.")
        val = y_series[idx]
        if val is None:
            raise ValueError(f"Null encountered for lag_{k}. Provide complete y history.")
        return float(val)

    # rolling su passato escluso current (equivalente a shift(1) prima del rolling)
    y_past = [v for v in y_series[:-1] if v is not None]

    def roll_mean(w: int) -> float:
        if len(y_past) < w:
            raise ValueError(f"Not enough history for rolling_mean_{w} (excluding current).")
        arr = np.array(y_past[-w:], dtype=float)
        return float(arr.mean())

    def roll_std(w: int) -> float:
        if len(y_past) < w:
            raise ValueError(f"Not enough history for rolling_std_{w} (excluding current).")
        arr = np.array(y_past[-w:], dtype=float)
        return float(arr.std(ddof=0))

    feats: dict[str, float] = {}

    for k in cfg.lags:
        feats[f"lag_{k}"] = lag(k)

    for w in cfg.rolling_windows:
        feats[f"roll_mean_{w}"] = roll_mean(w)
        feats[f"roll_std_{w}"] = roll_std(w)

    # Calendar features (coerenti col training)
    feats["hour"] = float(last_dt.hour)
    feats["weekday"] = float(last_dt.weekday() + 1)  # Monday=1 ... Sunday=7
    feats["month"] = float(last_dt.month)

    return feats


def predict_24h(
    observations: Iterable[dict[str, Any]],
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """
    observations: iterable di dict con chiavi:
      - dt: ISO string (es. "2026-02-08T10:00:00Z") o datetime
      - y: float
    ritorna:
      {"used_last_dt", "forecast_dt", "yhat", "horizon_hours"}
    """
    cfg = InferenceConfig()
    bundle = load_bundle(model_path)

    model = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]
    horizon = int(bundle.get("horizon", cfg.horizon))

    if horizon != cfg.horizon:
        raise ValueError(f"Bundle horizon={horizon} differs from inference horizon={cfg.horizon}.")

    parsed_rows = []
    for obs in observations:
        dt_raw = obs.get("dt")
        y_raw = obs.get("y")
        if dt_raw is None or y_raw is None:
            raise ValueError("Each observation must contain 'dt' and 'y'.")

        if isinstance(dt_raw, str):
            dt = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
        elif isinstance(dt_raw, datetime):
            dt = dt_raw
        else:
            raise ValueError("Field 'dt' must be ISO string or datetime.")

        dt = _ensure_utc(dt)

        try:
            y = float(y_raw)
        except Exception as e:
            raise ValueError(f"Invalid y value: {y_raw}") from e

        parsed_rows.append({"dt": dt, "y": y})

    if len(parsed_rows) < 200:
        # non è un vincolo matematico, ma evita richieste troppo corte
        # (lag_168 + rolling_168 richiedono comunque molta storia)
        pass

    history = pl.DataFrame(parsed_rows)

    last_dt = max(r["dt"] for r in parsed_rows)
    last_dt = _ensure_utc(last_dt)

    feats = _build_feature_row(history, last_dt, cfg)

    missing = [c for c in feature_cols if c not in feats]
    if missing:
        raise ValueError(f"Missing features for inference: {missing}")

    X = np.array([[feats[c] for c in feature_cols]], dtype=float)
    yhat = float(model.predict(X)[0])

    forecast_dt = last_dt + timedelta(hours=cfg.horizon)

    return {
        "used_last_dt": last_dt.isoformat().replace("+00:00", "Z"),
        "forecast_dt": forecast_dt.isoformat().replace("+00:00", "Z"),
        "yhat": yhat,
        "horizon_hours": cfg.horizon,
    }