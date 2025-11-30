import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def main():

    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Load raw Excel files
    macro_finance_data = pd.read_excel(r"data/raw/raw_US_macro_finance_fixed.xlsx", index_col=0)
    w5000 = pd.read_excel("data/raw/raw_w5000.xlsx", index_col=0)
    nasdaq_close = pd.read_excel("data/raw/raw_nasdaq_close.xlsx", index_col=0)

    # Merge macro + Wilshire
    macro_with_wilshire = macro_finance_data.join(w5000, how="outer")
    macro_with_wilshire["WILSHIRE5000"] = macro_with_wilshire["WILSHIRE5000"].ffill()

    # Buffett Indicator
    macro_with_wilshire["BUFFETT_INDICATOR"] = (
        macro_with_wilshire["WILSHIRE5000"] / macro_with_wilshire["GDP"]
    )

    # Merge NASDAQ
    market_data = macro_with_wilshire.join(nasdaq_close, how="outer")
    market_data["nasdaq_close"] = market_data["nasdaq_close"].ffill()

    # Drop WILSHIRE5000 intermediate column
    market_data = market_data.drop(columns=["WILSHIRE5000"])

    # Drop columns with no data after 2025-01-01
    df_after = market_data.loc["2025-01-01":]
    valid_columns = df_after.notna().any(axis=0)
    features_with_values = valid_columns[valid_columns].index.tolist()

    # Final processed dataset
    market_data_processed = market_data.loc[:, features_with_values]

    # Save to processed folder
    output_path = os.path.join(processed_dir, "processed_market_data.xlsx")
    market_data_processed.to_excel(output_path)

    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()

