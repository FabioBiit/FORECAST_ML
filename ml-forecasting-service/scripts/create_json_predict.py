import json
from pathlib import Path
import polars as pl

DATA_PATH = Path("data/processed/power_hourly.parquet")
OUTPUT_JSON = Path("scripts/json/payload_500.json")

N_HOURS = 500  # >= 200

df = pl.read_parquet(DATA_PATH).sort("dt").tail(N_HOURS)

payload = {
    "observations": [
        {
            "dt": dt.isoformat() + "Z",
            "y": float(y),
        }
        for dt, y in zip(df["dt"].to_list(), df["y"].to_list())
    ]
}

# Scrittura su file
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"Creato file JSON: {OUTPUT_JSON}")
print(f"Numero osservazioni: {len(payload['observations'])}")