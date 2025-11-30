import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta
import warnings
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
        Returns:
            A pandas DataFrame indexed by date with the series values.
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


# ===== Corrected FRED Series List (same as your notebook) =====
FRED_SERIES = {
    # Macro
    "GDP": "GDP",
    "CPI": "CPIAUCSL",
    "UNRATE": "UNRATE",
    "FEDFUNDS": "FEDFUNDS",
    "INDUSTRIAL_PRODUCTION": "INDPRO",
    "RETAIL_SALES": "RSAFS",
    "HOUSING_STARTS": "HOUST",

    # Yield curve
    "DGS3MO": "DGS3MO",

    # VIX
    "VIX": "VIXCLS",

    # Liquidity / financial stress proxies
    "STLFSI": "STLFSI",
    "TEDRATE": "TEDRATE",
}


def main():
    """
    Main pipeline for fetching macro-finance data (FRED + Yahoo Finance)
    and saving all raw datasets to Excel files.
    """
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
    macro_finance_data.to_excel("raw_US_macro_finance_fixed.xlsx")
    print("[Saved] raw_US_macro_finance_fixed.xlsx")

    # ===== Fetch Wilshire 5000 (^W5000) =====
    print("===== Fetching WILSHIRE 5000 index (^W5000) =====")
    w5000 = yf.download("^W5000", start="2014-12-31")
    w5000.columns = ["Close", "High", "Low", "Open", "Volume"]
    w5000 = w5000[["Close"]].rename(columns={"Close": "WILSHIRE5000"})
    w5000.index = pd.to_datetime(w5000.index)
    w5000.to_excel("raw_w5000.xlsx")
    print("[Saved] raw_w5000.xlsx")

    # ===== Fetch NASDAQ (^IXIC) daily close =====
    print("===== Fetching NASDAQ index (^IXIC) =====")
    ticker = "^IXIC"

    nasdaq = yf.download(ticker, start="2014-12-31", end=None)
    nasdaq_close = nasdaq[["Close"]].rename(columns={"Close": "nasdaq_close"})
    nasdaq_close.index = pd.to_datetime(nasdaq_close.index)
    nasdaq_close.to_excel("raw_nasdaq_close.xlsx")
    print("[Saved] raw_nasdaq_close.xlsx")

    print("===== All raw datasets downloaded successfully =====")


if __name__ == "__main__":
    main()
