import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')


class FREDDataCollector:
    """
    A simple FRED API client for fetching macroeconomic time series.
    """
    def __init__(self, api_key: str):
        self.base_url = "https://api.stlouisfed.org/fred"
        self.api_key = api_key

    def get_series_data(self, series_id: str) -> pd.DataFrame:
        """
        Fetch a time series from FRED API starting from 2014-12-31.
        """
        url = f"{self.base_url}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": "2014-12-31"
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data["observations"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["value"]]
            return df.rename(columns={"value": series_id})
        else:
            print(f"Error fetching data for {series_id}: {response.status_code}")
            return None


# ===== Your original FRED series list =====
FRED_SERIES = {
    "GDP": "GDP",
    "CPI": "CPIAUCSL",
    "UNRATE": "UNRATE",
    "FEDFUNDS": "FEDFUNDS",
    "INDUSTRIAL_PRODUCTION": "INDPRO",
    "RETAIL_SALES": "RSAFS",
    "HOUSING_STARTS": "HOUST",
    "DGS3MO": "DGS3MO",
    "VIX": "VIXCLS",
    "STLFSI": "STLFSI",
    "TEDRATE": "TEDRATE",
}


def main():
    # Ensure raw folder exists
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("===== Fetching macro-financial data from FRED API =====")
    api_key = "dca474e4ebb8d4b9092e387e067c07cc"
    collector = FREDDataCollector(api_key)

    dfs = []
    for name, series_id in FRED_SERIES.items():
        print(f"Fetching {name} ({series_id}) ...")
        df = collector.get_series_data(series_id)
        if df is not None:
            dfs.append(df)
        else:
            print(f"Skipping {series_id} due to fetch error.")

    macro_finance_data = pd.concat(dfs, axis=1)
    macro_output = os.path.join(raw_dir, "raw_US_macro_finance_fixed.xlsx")
    macro_finance_data.to_excel(macro_output)
    print(f"[Saved] {macro_output}")

    # Fetch Wilshire 5000 (^W5000)
    print("===== Fetching WILSHIRE 5000 index (^W5000) =====")
    w5000 = yf.download("^W5000", start="2014-12-31")
    w5000.columns = ["Close", "High", "Low", "Open", "Volume"]
    w5000 = w5000[["Close"]].rename(columns={"Close": "WILSHIRE5000"})
    w5000.index = pd.to_datetime(w5000.index)
    w5000_output = os.path.join(raw_dir, "raw_w5000.xlsx")
    w5000.to_excel(w5000_output)
    print(f"[Saved] {w5000_output}")

    # Fetch NASDAQ (^IXIC)
    print("===== Fetching NASDAQ index (^IXIC) =====")
    nasdaq = yf.download("^IXIC", start="2014-12-31", end=None)
    nasdaq_close = nasdaq[["Close"]].rename(columns={"Close": "nasdaq_close"})
    nasdaq_close.index = pd.to_datetime(nasdaq_close.index)
    nasdaq_output = os.path.join(raw_dir, "raw_nasdaq_close.xlsx")
    nasdaq_close.to_excel(nasdaq_output)
    print(f"[Saved] {nasdaq_output}")

    print("===== All raw datasets downloaded successfully =====")


if __name__ == "__main__":
    main()


