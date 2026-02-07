from pathlib import Path
import polars as pl

# Test per verificare il corretto calcolo del percorso del repository
REPO_ROOT = Path(__file__).resolve().parents[0]
# Mettendo 1 prende il repo padre e permette di spostarsi agevolmente tra cartelle e sottocartelle
# REPO_ROOT_P = Path(__file__).resolve().parents[1]

# print(f"\nRoot Repo Completa: {REPO_ROOT}\n")
# print(f"\nRoot Repo Padre: {REPO_ROOT_P}\n")

df_parquet = pl.read_parquet(REPO_ROOT/"data"/"processed"/"power_hourly.parquet")
df_parquet.show(100)