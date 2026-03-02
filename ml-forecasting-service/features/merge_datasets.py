from __future__ import annotations

from pathlib import Path
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT/"data"
PROCESSED_DIR = DATA_DIR/"processed"

UCI_PATH = PROCESSED_DIR/"power_hourly.parquet"
OPSD_PATH = PROCESSED_DIR/"power_hourly_opsd_*.parquet"
OUT_PATH = DATA_DIR/"marge_df_processed"/"power_hourly_all.parquet"


def _ensure_schema(df: pl.DataFrame, *, series_id: str | None = None) -> pl.DataFrame:
    """
    Normalizza schema a: dt (UTC), y (float), series_id (string)
    """
    cols = set(df.columns)
    if "dt" not in cols or "y" not in cols:
        raise ValueError(f"Expected columns ['dt','y'] but got {df.columns}")

    if "series_id" in cols:
        out = df.select(["dt", "y", "series_id"]).with_columns(
            pl.col("dt").cast(pl.Datetime("ms", "UTC")),
            pl.col("y").cast(pl.Float64),
            pl.col("series_id").cast(pl.Utf8),
        )
    else:
        if series_id is None:
            raise ValueError("series_id is missing and no series_id was provided")
        out = df.select(["dt", "y"]).with_columns(
            pl.col("dt").cast(pl.Datetime("ms", "UTC")),
            pl.col("y").cast(pl.Float64),
            pl.lit(series_id).alias("series_id"),
        )

    return out.select(["dt", "y", "series_id"])


def load_uci() -> pl.DataFrame:
    if not UCI_PATH.exists():
        raise FileNotFoundError(f"Missing UCI dataset: {UCI_PATH}")

    df = pl.read_parquet(UCI_PATH)
    # UCI non ha series_id, lo aggiungiamo
    return _ensure_schema(df, series_id="uci_household")


def load_opsd_many() -> list[pl.DataFrame]:
    opsd_files = sorted(OPSD_PATH.parent.glob(OPSD_PATH.name))
    print("OPSD files found:", [p.name for p in opsd_files])

    if not opsd_files:
        raise FileNotFoundError(
            f"No OPSD datasets found in {OPSD_PATH.parent}. "
            f"Expected pattern: {OPSD_PATH.name}. "
            "Run: python data_ingestion/ingest_opsd_load.py --countries IT,DE,FR,ES"
        )

    dfs: list[pl.DataFrame] = []
    for p in opsd_files:
        df = pl.read_parquet(p)

        # Se manca series_id lo ricaviamo dal filename (ultimo token: IT/DE/FR/ES)
        if "series_id" not in df.columns:
            country = p.stem.split("_")[-1]
            df = _ensure_schema(df, series_id=f"opsd_load_{country}")
        else:
            df = _ensure_schema(df)

        dfs.append(df)

    return dfs


def main() -> None:
    dfs = []
    dfs.append(load_uci())
    dfs.extend(load_opsd_many())

    uci = load_uci()
    opsd_list = load_opsd_many()

    print("UCI:", uci.height, uci.select(pl.col("dt").min().alias("min_dt"), pl.col("dt").max().alias("max_dt")).to_dict(as_series=False))
    for i, d in enumerate(opsd_list):
        print("OPSD", i, "rows:", d.height, "series:", d.select(pl.col("series_id").unique()).to_series().to_list()[:3])

    all_df = (
        pl.concat(dfs, how="vertical")
        .drop_nulls(["dt", "y", "series_id"])
        .unique(subset=["series_id", "dt"])  # evita duplicati
        .sort(["series_id", "dt"])
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_df.write_parquet(OUT_PATH)

    print(f"Merged → {OUT_PATH}")
    print(
        all_df.group_by("series_id")
        .agg(
            pl.len().alias("rows"),
            pl.min("dt").alias("min_dt"),
            pl.max("dt").alias("max_dt"),
            pl.mean("y").alias("mean_y"),
        )
        .sort("series_id")
    )

    # --- OPTIONAL (ELEXON) ---
    #
    # ELEXON_PATH = PROCESSED_DIR / "power_hourly_elexon_recent.parquet"
    # if ELEXON_PATH.exists():
    #     elexon = _ensure_schema(pl.read_parquet(ELEXON_PATH))
    #     all_df = (
    #         pl.concat([all_df, elexon], how="vertical")
    #         .unique(subset=["series_id", "dt"])
    #         .sort(["series_id", "dt"])
    #     )
    #     all_df.write_parquet(OUT_PATH)
    #     print("(optional) added elexon recent window")


if __name__ == "__main__":
    main()