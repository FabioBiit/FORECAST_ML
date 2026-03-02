from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
import requests

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def extract_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unexpected payload shape")


def fetch_outturn_recent(days: int) -> Any:
    # Proviamo con from/to; se l’API lo ignora, comunque ci torna latest window
    now = datetime.now(timezone.utc)
    req_from = now - timedelta(days=days)

    url = f"{BASE_URL}/demand/outturn"
    params = {
        "from": req_from.isoformat().replace("+00:00", "Z"),
        "to": now.isoformat().replace("+00:00", "Z"),
        "format": "json",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def normalize(rows: List[Dict[str, Any]], y_field: str) -> pl.DataFrame:
    dt = [parse_dt(r["startTime"]) for r in rows]
    y = [float(r[y_field]) for r in rows]
    return (
        pl.DataFrame({"dt": dt, "y": y})
        .with_columns(
            pl.col("dt").cast(pl.Datetime("ms", "UTC")),
            pl.col("y").cast(pl.Float64),
            pl.lit("elexon_indo_recent").alias("series_id"),
        )
        .sort("dt")
    )


def to_hourly(df_halfhour: pl.DataFrame) -> pl.DataFrame:
    return (
        df_halfhour
        .with_columns(pl.col("dt").dt.truncate("1h").alias("hour"))
        .group_by("hour")
        .agg(pl.col("y").mean().alias("y"))
        .rename({"hour": "dt"})
        .with_columns(pl.lit("elexon_indo_recent").alias("series_id"))
        .sort("dt")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent_days", type=int, default=30)
    ap.add_argument("--y_field", default="initialTransmissionSystemDemandOutturn")
    ap.add_argument("--out_processed", default="data/processed")
    args = ap.parse_args()

    payload = fetch_outturn_recent(args.recent_days)
    rows = extract_rows(payload)

    df_raw = normalize(rows, args.y_field)
    df_hourly = to_hourly(df_raw)

    out_dir = Path(args.out_processed); out_dir.mkdir(parents=True, exist_ok=True)
    df_hourly.write_parquet(out_dir / "power_hourly_elexon_recent.parquet")

    print(df_hourly.select(
        pl.len().alias("rows"),
        pl.min("dt").alias("min_dt"),
        pl.max("dt").alias("max_dt"),
    ))


if __name__ == "__main__":
    main()