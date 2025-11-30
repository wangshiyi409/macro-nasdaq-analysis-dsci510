"""
clean_data.py
-------------
Merge all macroeconomic indicators and NASDAQ index into a single dataset.
Save cleaned dataset into data/processed/.
"""

import os
import pandas as pd
from utils.helpers import ensure_directories, save_dataframe, log


RAW_FILES = {
    "GDP": "GDP.csv",
    "CPI": "CPI.csv",
    "UNRATE": "UNRATE.csv",
    "FEDFUNDS": "FEDFUNDS.csv",
    "INDPRO": "INDPRO.csv",
    "RSAFS": "RSAFS.csv",
    "HOUST": "HOUST.csv",
    "DGS3MO": "DGS3MO.csv",
    "DGS10": "DGS10.csv",
    "T10Y2Y": "T10Y2Y.csv",
    "VIX": "VIX.csv",
    "NASDAQ": "NASDAQ.csv"
}



def load_csv(path):
    """Utility to load a CSV cleanly and standardize the Date column."""
    df = pd.read_csv(path)

    # Normalize column names: date -> Date, value -> Value, etc.
    df.columns = [col.capitalize() for col in df.columns]

    # 统一把 Date 解析成 datetime，并去掉时区信息
    # utc=True 可以处理带时区的字符串，tz_localize(None) 去掉时区，只保留日期时间
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)

    return df



def main():
    log("Starting data cleaning & merging ...")

    raw_dir, processed_dir = ensure_directories()

    merged_df = None

    # ----------------------------
    # Merge all macroeconomic data
    # ----------------------------
    for name, filename in RAW_FILES.items():
        file_path = os.path.join(raw_dir, filename)

        if not os.path.exists(file_path):
            log(f"[WARNING] Missing raw file: {filename}, skipping.")
            continue

        log(f"Loading {filename} ...")
        df = load_csv(file_path)

        if merged_df is None:
            merged_df = df.rename(columns={"Value": name})
        else:
            merged_df = merged_df.merge(
                df.rename(columns={"Value": name}),
                on="Date",
                how="outer"
            )

    # ----------------------------
    # Sort by date
    # ----------------------------
    merged_df = merged_df.sort_values("Date")

    # ----------------------------
    # Fill missing values (forward-fill then back-fill)
    # ----------------------------
    merged_df = merged_df.ffill().bfill()

    # ----------------------------
    # Save final cleaned dataset
    # ----------------------------
    save_path = os.path.join(processed_dir, "merged_data.csv")
    save_dataframe(merged_df, save_path)

    log("Cleaned dataset created!")
    log(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
