from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference.predict import predict_24h, DEFAULT_MODEL_PATH

app = FastAPI(title="Power Forecasting API", version="0.1.0")


class Observation(BaseModel):
    dt: str = Field(..., description="ISO datetime string, e.g. 2026-02-08T10:00:00Z")
    y: float = Field(..., description="Observed value (target) at dt")


class PredictRequest(BaseModel):
    observations: List[Observation] = Field(
        ...,
        description="Hourly observations. Must contain enough history (>= 169 points + rolling windows).",
        min_items=50,
    )
    model_path: Optional[str] = Field(None, description="Optional override path to model bundle")


class PredictResponse(BaseModel):
    used_last_dt: str
    forecast_dt: str
    yhat: float
    horizon_hours: int


@app.get("/health")
def health():
    exists = Path(DEFAULT_MODEL_PATH).exists()
    return {"status": "ok", "model_bundle_found": exists, "model_path": str(DEFAULT_MODEL_PATH)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model_path = Path(req.model_path) if req.model_path else DEFAULT_MODEL_PATH

    try:
        return predict_24h(
            observations=[o.model_dump() for o in req.observations],
            model_path=model_path,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        # tipicamente: storia insufficiente, feature mancanti, parsing errato, etc.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")