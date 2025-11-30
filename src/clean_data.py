import pandas as pd
import numpy as np
from datetime import datetime, timedelta

"""
Clean and merge macro-finance data, Wilshire 5000, and NASDAQ data.
This script directly converts your Jupyter Notebook logic into a .py file
with NO modifications to logic, formulas, column names, or behavior.
"""

# ---------------------------------------------------------
# 1. Load raw data 
# ---------------------------------------------------------
macro_finance_data = pd.read_excel(r"raw_US_macro_finance_fixed.xlsx", index_col=0)
w5000 = pd.read_excel("raw_w5000.xlsx", index_col=0)
nasdaq_close = pd.read_excel("raw_nasdaq_close.xlsx", index_col=0)

# ---------------------------------------------------------
# 2. Merge macro-finance with Wilshire 5000
# ---------------------------------------------------------
macro_with_wilshire = macro_finance_data.join(w5000, how="outer")

# Forward-fill missing WILSHIRE5000 values
macro_with_wilshire["WILSHIRE5000"] = macro_with_wilshire["WILSHIRE5000"].ffill()

# Compute the Buffett Indicator
macro_with_wilshire["BUFFETT_INDICATOR"] = (
    macro_with_wilshire["WILSHIRE5000"] / macro_with_wilshire["GDP"]
)

# ---------------------------------------------------------
# 3. Merge NASDAQ close
# ---------------------------------------------------------
market_data = macro_with_wilshire.join(nasdaq_close, how="outer")

# Forward-fill NASDAQ values
market_data.loc[:, "nasdaq_close"] = market_data["nasdaq_close"].ffill()

# Remove temporary WILSHIRE5000 column 
market_data = market_data.drop(columns=["WILSHIRE5000"])

# ---------------------------------------------------------
# 4. Keep only features with values after 2025-01-01
# ---------------------------------------------------------
df_after = market_data.loc["2025-01-01":]
valid_columns = df_after.notna().any(axis=0)
features_with_values = valid_columns[valid_columns].index.tolist()

market_data_processed = market_data.loc[:, features_with_values]

# ---------------------------------------------------------
# 5. Save processed data
# ---------------------------------------------------------
market_data_processed.to_excel(r"processed market data.xlsx")

print("Saved: processed market data.xlsx")

