"""
run_analysis.py
---------------
Perform statistical analysis + logistic regression + ROC curve + confusion matrix.
Save all intermediate results to /results/.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
from utils.helpers import get_project_root, log, save_dataframe


def compute_max_drawdown(prices, window=60):
    """
    Compute rolling max drawdown for a price series.
    """
    roll_max = prices.rolling(window).max()
    drawdown = (prices - roll_max) / roll_max
    return drawdown


def main():
    log("Starting analysis pipeline...")

    root = get_project_root()
    processed_path = os.path.join(root, "data/processed/merged_data.csv")
    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ----------------------------
    # Load processed dataset
    # ----------------------------
    log("Loading merged_data.csv ...")
    df = pd.read_csv(processed_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # ----------------------------
    # Compute NASDAQ drawdown features
    # ----------------------------
    df["NASDAQ_Drawdown60"] = compute_max_drawdown(df["Nasdaq"], window=60)

    # Tail-Risk Label: 60-day drop > -2% (= -0.02)
    df["Risk_Label"] = (df["NASDAQ_Drawdown60"] < -0.02).astype(int)

    # Save the dataset with labels
    save_dataframe(df, os.path.join(results_dir, "analysis_dataset.csv"))

    # ----------------------------
    # Select Features for ML
    # ----------------------------
    FEATURES = [
        "Gdp", "Cpi", "Unrate", "Fedfunds",
        "Indpro", "Rsafs", "Houst",
        "Dgs3mo", "Dgs10", "T10y2y", "Vix"
    ]

    X = df[FEATURES].fillna(method="ffill").fillna(method="bfill")
    y = df["Risk_Label"]

    # ----------------------------
    # Logistic Regression Model
    # ----------------------------
    log("Training Logistic Regression model...")

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    preds = model.predict_proba(X)[:, 1]

    # ----------------------------
    # ROC Curve
    # ----------------------------
    fpr, tpr, _ = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr
    })
    save_dataframe(roc_
