from __future__ import annotations

from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_rejects_short_history_or_missing_model():
    # 10 ore => insufficiente per lag_168 / rolling_168
    start = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    observations = []
    for i in range(10):
        observations.append(
            {
                "dt": (start + timedelta(hours=i)).isoformat().replace("+00:00", "Z"),
                "y": 1.0 + i * 0.01,
            }
        )

    r = client.post("/predict", json={"observations": observations})

    # 404 se non hai ancora creato models/xgb_power_h24.joblib
    # 422 se il modello esiste ma la history è troppo corta
    assert r.status_code in (404, 422)
    print("Response:", r.status_code, r.json())