import json
from pathlib import Path
import polars as pl
import requests

API_URL = "http://localhost:8000/predict"
DATA_PATH = Path("data/processed/power_hourly.parquet")

N_HOURS = 500  # >= 200 consigliato

df = pl.read_parquet(DATA_PATH).sort("dt").tail(N_HOURS)

payload = {
    "observations": [
        {
            "dt": dt.replace(tzinfo=None).isoformat() + "Z" if hasattr(dt, "isoformat") else str(dt),
            "y": float(y),
        }
        for dt, y in zip(df["dt"].to_list(), df["y"].to_list())
    ]
}

r = requests.post(API_URL, json=payload, timeout=30)
print("status:", r.status_code)
print(json.dumps(r.json(), indent=2))