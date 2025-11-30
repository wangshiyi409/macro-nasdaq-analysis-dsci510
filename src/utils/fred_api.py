import requests
import pandas as pd

class FREDClient:
    """
    Simple wrapper for the FRED API to fetch macroeconomic indicators.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_series(self, series_id: str) -> pd.DataFrame:
        """
        Fetch a single FRED series and return it as a pandas DataFrame.
        """

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }

        response = requests.get(self.base_url, params=params)
        response.raise_for_status()

        json_data = response.json()
        observations = json_data.get("observations", [])

        df = pd.DataFrame(observations)
        df = df[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df
