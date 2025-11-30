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
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # 把 Close 列重命名成 NASDAQ，方便解释
    if "Close" in df.columns:
        df = df.rename(columns={"Close": "NASDAQ"})

    # ----------------------------
    # Compute NASDAQ drawdown features
    # ----------------------------
    df["NASDAQ_Drawdown60"] = compute_max_drawdown(df["NASDAQ"], window=60)

    # Tail-Risk Label: 60-day drop > -2% (= -0.02)
    df["Risk_Label"] = (df["NASDAQ_Drawdown60"] < -0.02).astype(int)

    # Save the dataset with labels
    save_dataframe(df, os.path.join(results_dir, "analysis_dataset.csv"))

    # ----------------------------
    # Select Features for ML
    # ----------------------------
    FEATURES = [
        "GDP", "CPI", "UNRATE", "FEDFUNDS",
        "INDPRO", "RSAFS", "HOUST",
        "DGS3MO", "DGS10", "T10Y2Y", "VIX"
    ]

    X = df[FEATURES].copy()
    X = X.ffill().bfill()   # double fill just in case
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
    save_dataframe(roc_df, os.path.join(results_dir, "roc_curve_data.csv"))

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
    y_pred_label = (preds > 0.5).astype(int)
    cm = confusion_matrix(y, y_pred_label)
    cm_df = pd.DataFrame(cm, columns=["Pred_0", "Pred_1"], index=["Actual_0", "Actual_1"])
    save_dataframe(cm_df, os.path.join(results_dir, "confusion_matrix.csv"))

    # ----------------------------
    # Save Metrics
    # ----------------------------
    metrics_df = pd.DataFrame({
        "roc_auc": [roc_auc]
    })
    save_dataframe(metrics_df, os.path.join(results_dir, "metrics.csv"))

    # ----------------------------
    # Correlation Matrix (for heatmap)
    # ----------------------------
    corr_cols = ["NASDAQ"] + FEATURES
    corr = df[corr_cols].corr()
    save_dataframe(corr, os.path.join(results_dir, "correlation_matrix.csv"))

    log(f"AUC = {roc_auc:.4f}")
    log("Analysis complete! Results saved in /results/.")


if __name__ == "__main__":
    main()
