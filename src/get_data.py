"""
get_data.py
-----------
Fetch all macroeconomic indicators (FRED) and NASDAQ (^IXIC) index data.
Save everything into data/raw/ as CSV files.
"""

import os
import pandas as pd
from utils.fred_api import FREDClient
from utils.yahoo_api import fetch_nasdaq_history
from utils.helpers import ensure_directories, save_dataframe, log


# ------------------------
# 1. Define FRED Indicators
# ------------------------
FRED_SERIES = {
    "GDP": "GDP",
    "CPI": "CPIAUCSL",
    "UNRATE": "UNRATE",
    "FEDFUNDS": "FEDFUNDS",
    "INDPRO": "INDPRO",
    "RSAFS": "RSAFS",
    "HOUST": "HOUST",
    "DGS3MO": "DGS3MO",
    "DGS10": "DGS10",
    "T10Y2Y": "T10Y2Y",
    "VIX": "VIXCLS"
}


def main():
    log("Starting data download pipeline ...")

    # Step 1: Ensure /data/raw exists
    raw_dir, _ = ensure_directories()

    # Step 2: Initialize FRED client (YOU MUST PUT YOUR FRED API KEY BELOW)
    FRED_API_KEY = "dca474e4ebb8d4b9092e387e067c07cc"      # <- replace with your own key
    fred = FREDClient(api_key=FRED_API_KEY)

    # Step 3: Download macro data
    for name, series_id in FRED_SERIES.items():
        log(f"Fetching {name} ({series_id}) from FRED ...")
        try:
            df = fred.fetch_series(series_id)
            save_path = os.path.join(raw_dir, f"{name}.csv")
            save_dataframe(df, save_path)
        except Exception as e:
            log(f"[ERROR] Failed fetching {series_id}: {e}")

    # Step 4: Download NASDAQ (^IXIC)
    log("Fetching NASDAQ (^IXIC) index history ...")
    try:
        df_nasdaq = fetch_nasdaq_history(start="2000-01-01")
        save_path = os.path.join(raw_dir, "NASDAQ.csv")
        save_dataframe(df_nasdaq, save_path)
    except Exception as e:
        log(f"[ERROR] Failed fetching NASDAQ: {e}")

    log("Data download complete! All files saved to /data/raw/.")


if __name__ == "__main__":
    main()
