import os
import pandas as pd


# ---------------------------
#  Project Path Utilities
# ---------------------------

def get_project_root():
    """
    Returns the absolute path to the project root directory.
    (This file is inside src/utils/, so root is two levels up.)
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def get_data_dir():
    """
    Returns the data directory path.
    """
    root = get_project_root()
    return os.path.join(root, "data")


def ensure_directories():
    """
    Create data/raw and data/processed folders if they do not exist.
    """
    base = get_data_dir()
    raw_dir = os.path.join(base, "raw")
    processed_dir = os.path.join(base, "processed")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    return raw_dir, processed_dir


# ---------------------------
#  File Save Utilities
# ---------------------------

def save_dataframe(df: pd.DataFrame, path: str):
    """
    Save a DataFrame to CSV safely, ensuring directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Saved] {path}")


# ---------------------------
#  Logging Utility
# ---------------------------

def log(msg: str):
    """
    Simple logger for consistent printing.
    """
    print(f"[INFO] {msg}")
