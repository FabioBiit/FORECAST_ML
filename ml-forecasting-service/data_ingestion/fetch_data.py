from __future__ import annotations

import zipfile
from pathlib import Path
import requests


UCI_ZIP_URL = ("https://archive.ics.uci.edu/static/public/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip")

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)


def unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def main() -> None:
    zip_path = RAW_DIR / "uci_household_power.zip"
    print(f"[ingestion] Downloading -> {zip_path}")
    download_file(UCI_ZIP_URL, zip_path)

    print(f"[ingestion] Unzipping -> {RAW_DIR}")
    unzip(zip_path, RAW_DIR)

    txt_path = RAW_DIR / "household_power_consumption.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Expected file not found after unzip: {txt_path}")

    print(f"[ingestion] OK. Raw file ready: {txt_path} ({txt_path.stat().st_size / (1024**2):.1f} MB)")


if __name__ == "__main__":
    main()