from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import polars as pl
import requests

# OPSD time series data package (hourly singleindex)
# Versione indicata come "latest" nella pagina OPSD (2020-10-06). 
OPSD_CSV_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

UTC_FMT = "%Y-%m-%dT%H:%M:%SZ"

RAW_DEFAULT = Path("data/raw")
PROCESSED_DEFAULT = Path("data/processed")


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def ingest_one_country(raw_csv_path: Path, country: str, field_suffix: str) -> pl.DataFrame:
    """
    Estrae una sola colonna di load da OPSD:
      - utc_timestamp
      - <COUNTRY>_<field_suffix>  (default: load_actual_entsoe_transparency)
    """
    col_name = f"{country}_{field_suffix}"

    df = pl.read_csv(
        raw_csv_path,
        columns=["utc_timestamp", col_name],
        null_values=["", "NA", "null"],
        ignore_errors=True,
    ).rename({col_name: "y"})

    df = (
        df.with_columns(
            pl.col("utc_timestamp")
            .str.strptime(pl.Datetime, format=UTC_FMT, strict=False)
            .dt.replace_time_zone("UTC")
            .alias("dt"),
            pl.col("y").cast(pl.Float64),
            pl.lit(f"opsd_load_{country}").alias("series_id"),
        )
        .select(["dt", "y", "series_id"])
        .drop_nulls(["dt", "y"])
        .sort("dt")
    )

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--countries",
        default="IT,DE,FR,ES",
        help="Comma-separated country codes, e.g. IT,DE,FR,ES",
    )
    ap.add_argument(
        "--field_suffix",
        default="load_actual_entsoe_transparency",
        help="OPSD column suffix (default is a good load series).",
    )
    ap.add_argument("--csv_url", default=OPSD_CSV_URL)
    ap.add_argument("--raw_dir", default=str(RAW_DEFAULT))
    ap.add_argument("--processed_dir", default=str(PROCESSED_DEFAULT))
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = raw_dir / "opsd_time_series_60min_singleindex_2020-10-06.csv"

    if not raw_csv_path.exists():
        print(f"Downloading OPSD CSV → {raw_csv_path}")
        download(args.csv_url, raw_csv_path)
    else:
        print(f"Using cached OPSD CSV → {raw_csv_path}")

    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    print("Countries:", countries)

    for c in countries:
        df = ingest_one_country(raw_csv_path, c, args.field_suffix)
        out_path = processed_dir / f"power_hourly_opsd_{c}.parquet"
        df.write_parquet(out_path)

        stats = df.select(
            pl.len().alias("rows"),
            pl.min("dt").alias("min_dt"),
            pl.max("dt").alias("max_dt"),
        ).to_dict(as_series=False)

        print(f"{c} saved → {out_path} | stats={stats}")


if __name__ == "__main__":
    main()