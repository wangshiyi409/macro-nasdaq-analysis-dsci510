import yfinance as yf
import pandas as pd


def fetch_nasdaq_history(start="2000-01-01", end=None):
    """
    Fetch NASDAQ (^IXIC) daily historical prices using yfinance.

    Returns:
        DataFrame with columns: ['Date', 'Close']
    """

    ticker = yf.Ticker("^IXIC")   # NASDAQ Composite Index
    df = ticker.history(start=start, end=end)

    if df.empty:
        raise ValueError("Failed to download NASDAQ price data")

    df = df.reset_index()[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"])

    return df
