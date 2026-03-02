from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class StoreConfig:
    db_path: Path


class MonitorStore:
    def __init__(self, cfg: StoreConfig):
        self.cfg = cfg
        self.cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.cfg.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,

                    series_id TEXT NOT NULL,
                    used_last_dt TEXT NOT NULL,
                    forecast_dt TEXT NOT NULL,
                    horizon_hours INTEGER NOT NULL,

                    yhat REAL NOT NULL,

                    obs_count INTEGER NOT NULL,
                    model_path TEXT NOT NULL,

                    actual_y REAL,
                    actual_received_at TEXT,

                    abs_error REAL,
                    sq_error REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_forecast_dt ON predictions(forecast_dt)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_series_id ON predictions(series_id)")
            conn.commit()

    def log_prediction(
        self,
        *,
        series_id: str,
        used_last_dt: str,
        forecast_dt: str,
        horizon_hours: int,
        yhat: float,
        obs_count: int,
        model_path: str,
    ) -> str:
        pred_id = str(uuid4())
        created_at = utc_now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    id, created_at,
                    series_id, used_last_dt, forecast_dt, horizon_hours,
                    yhat, obs_count, model_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pred_id, created_at,
                    series_id, used_last_dt, forecast_dt, int(horizon_hours),
                    float(yhat), int(obs_count), model_path
                ),
            )
            conn.commit()

        return pred_id

    def attach_actual_by_id(self, pred_id: str, actual_y: float) -> dict[str, Any]:
        actual_received_at = utc_now_iso()

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
            if row is None:
                raise KeyError(f"prediction id not found: {pred_id}")

            yhat = float(row["yhat"])
            abs_error = abs(float(actual_y) - yhat)
            sq_error = (float(actual_y) - yhat) ** 2

            conn.execute(
                """
                UPDATE predictions
                SET actual_y = ?, actual_received_at = ?, abs_error = ?, sq_error = ?
                WHERE id = ?
                """,
                (float(actual_y), actual_received_at, float(abs_error), float(sq_error), pred_id),
            )
            conn.commit()

        return {
            "id": pred_id,
            "actual_y": float(actual_y),
            "abs_error": float(abs_error),
            "sq_error": float(sq_error),
            "actual_received_at": actual_received_at,
        }

    def attach_actual_by_forecast_dt(self, forecast_dt: str, actual_y: float) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM predictions WHERE forecast_dt = ? ORDER BY created_at DESC LIMIT 1",
                (forecast_dt,),
            ).fetchone()

        if row is None:
            raise KeyError(f"no prediction found for forecast_dt={forecast_dt}")

        return self.attach_actual_by_id(str(row["id"]), actual_y)

    def metrics(self, series_id: Optional[str] = None) -> dict[str, Any]:
        where = "WHERE actual_y IS NOT NULL"
        params: tuple[Any, ...] = ()
        if series_id:
            where += " AND series_id = ?"
            params = (series_id,)

        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) AS c FROM predictions" + ("" if not series_id else " WHERE series_id = ?"),
                (() if not series_id else (series_id,)),
            ).fetchone()["c"]

            with_actual = conn.execute(
                "SELECT COUNT(*) AS c FROM predictions " + where,
                params,
            ).fetchone()["c"]

            row = conn.execute(
                f"""
                SELECT
                  AVG(abs_error) AS mae,
                  AVG(sq_error) AS mse
                FROM predictions
                {where}
                """,
                params,
            ).fetchone()

        mae = row["mae"]
        mse = row["mse"]
        rmse = (mse ** 0.5) if mse is not None else None

        return {
            "series_id": series_id,
            "total_predictions": int(total),
            "predictions_with_actual": int(with_actual),
            "mae": float(mae) if mae is not None else None,
            "rmse": float(rmse) if rmse is not None else None,
        }