from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from monitoring.store import MonitorStore, StoreConfig
from inference.predict import predict_24h, DEFAULT_MODEL_PATH  # adatta al tuo progetto


app = FastAPI(title="Forecasting API", version="0.3.0")

STATE_DIR = Path(os.getenv("STATE_DIR", "state"))
DB_PATH = Path(os.getenv("DB_PATH", str(STATE_DIR / "app.db")))

store = MonitorStore(StoreConfig(db_path=DB_PATH))


class Observation(BaseModel):
    dt: str
    y: float


class PredictRequest(BaseModel):
    series_id: str = Field(..., description="e.g. uci_household, opsd_load_IT, opsd_load_DE, ...")
    observations: List[Observation]
    model_path: Optional[str] = None  # opzionale


class PredictResponse(BaseModel):
    request_id: str
    used_last_dt: str
    forecast_dt: str
    yhat: float
    horizon_hours: int


class LogActualRequest(BaseModel):
    request_id: Optional[str] = None
    forecast_dt: Optional[str] = None
    actual_y: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_bundle_found": Path(DEFAULT_MODEL_PATH).exists(),
        "model_path": str(DEFAULT_MODEL_PATH),
        "db_path": str(DB_PATH),
        "db_found": DB_PATH.exists(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model_path = DEFAULT_MODEL_PATH if (req.model_path in (None, "", "string")) else Path(req.model_path)

    try:
        result = predict_24h(
            observations=[o.model_dump() for o in req.observations],
            model_path=model_path,
        )

        request_id = store.log_prediction(
            series_id=req.series_id,
            used_last_dt=result["used_last_dt"],
            forecast_dt=result["forecast_dt"],
            horizon_hours=int(result["horizon_hours"]),
            yhat=float(result["yhat"]),
            obs_count=len(req.observations),
            model_path=str(model_path),
        )

        return {"request_id": request_id, **result}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.post("/log_actual")
def log_actual(req: LogActualRequest):
    # Exactly one selector
    if (req.request_id is None) == (req.forecast_dt is None):
        raise HTTPException(status_code=422, detail="Provide exactly one of: request_id OR forecast_dt")

    try:
        if req.request_id is not None:
            out = store.attach_actual_by_id(req.request_id, req.actual_y)
        else:
            out = store.attach_actual_by_forecast_dt(req.forecast_dt, req.actual_y)
        return {"status": "ok", **out}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/metrics")
def metrics(series_id: Optional[str] = None):
    try:
        return {"status": "ok", **store.metrics(series_id=series_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")