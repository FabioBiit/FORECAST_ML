from __future__ import annotations

from pathlib import Path
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "household_power_consumption.txt"
OUT_PATH = REPO_ROOT / "data" / "processed" / "power_hourly.parquet"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}. Run data_ingestion/fetch_data.py first.")

    # Lazy scan: veloce e memory-friendly
    df_energy = pl.scan_csv(
        RAW_PATH,
        separator=";",
        null_values=["?", ""],
        try_parse_dates=False,
    )

    # Costruisco timestamp (Date + Time) e cast numerici
    df_energy = (
        df_energy.with_columns(
            pl.concat_str([pl.col("Date"), pl.lit(" "), pl.col("Time")]).alias("dt_str")
        )
        .with_columns(
            pl.col("dt_str").str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S", strict=False).alias("dt")
        )
        .drop(["dt_str"])
    )

    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    df_energy = df_energy.with_columns([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in numeric_cols])

    # Target: Global_active_power (kW) -> media oraria
    df_hourly = (
        df_energy.select(["dt", "Global_active_power"])
        .with_columns(pl.col("Global_active_power").fill_null(pl.col("Global_active_power").median()).alias("Global_active_power"))
        .sort("dt")
        .group_by_dynamic("dt", every="1h", closed="left", label="left")
        .agg(pl.col("Global_active_power").mean().alias("y"))
        .drop_nulls(["y"])
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final = df_hourly.collect() # engine="streaming" SE I DATASET SONO MOLTO GRANDI, ALTRIMENTI COLLECT () E CARICA TUTTO IN MEMORIA
    df_final.write_parquet(OUT_PATH)
    print(f"[features] Saved hourly dataset: {OUT_PATH} | rows={df_final.height} | from={df_final['dt'][0]} to={df_final['dt'][-1]}")


if __name__ == "__main__":
    main()