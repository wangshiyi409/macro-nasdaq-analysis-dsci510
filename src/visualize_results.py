"""
visualize_results.py
--------------------
Create visualizations:
- Macro indicators + NASDAQ time series
- Correlation heatmap
- ROC curve
- Confusion matrix

All figures are saved into /results/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.helpers import get_project_root, log


def plot_macro_timeseries(df, save_path):
    """
    Plot NASDAQ and one or two macro indicators over time.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 主轴：NASDAQ
    ax1.plot(df["Date"], df["NASDAQ"], label="NASDAQ", linewidth=1.2)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("NASDAQ Index Level")
    ax1.tick_params(axis="y")

    # 次轴：选 GDP（可以换成其他宏观指标）
    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["GDP"], label="GDP", linestyle="--", alpha=0.7)
    ax2.set_ylabel("GDP (Level or Scaled)")

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.title("NASDAQ vs GDP Over Time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log(f"[Saved] {save_path}")


def plot_correlation_heatmap(corr_df, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap: NASDAQ and Macro Indicators")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log(f"[Saved] {save_path}")


def plot_roc_curve(roc_df, metrics_df, save_path):
    plt.figure(figsize=(6, 6))
    plt.plot(roc_df["fpr"], roc_df["tpr"], label=f"ROC curve (AUC = {metrics_df['roc_auc'][0]:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Tail-Risk Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log(f"[Saved] {save_path}")


def plot_confusion_matrix(cm_df, save_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Threshold = 0.5)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log(f"[Saved] {save_path}")


def main():
    log("Starting visualization pipeline...")

    root = get_project_root()
    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. 加载 analysis_dataset，用于时间序列图
    analysis_path = os.path.join(results_dir, "analysis_dataset.csv")
    df = pd.read_csv(analysis_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # 2. 时间序列图：NASDAQ + GDP
    plot_macro_timeseries(df, os.path.join(results_dir, "macro_timeseries.png"))

    # 3. 相关性热力图
    corr_path = os.path.join(results_dir, "correlation_matrix.csv")
    corr_df = pd.read_csv(corr_path, index_col=0)
    plot_correlation_heatmap(corr_df, os.path.join(results_dir, "correlation_heatmap.png"))

    # 4. ROC 曲线
    roc_df = pd.read_csv(os.path.join(results_dir, "roc_curve_data.csv"))
    metrics_df = pd.read_csv(os.path.join(results_dir, "metrics.csv"))
    plot_roc_curve(roc_df, metrics_df, os.path.join(results_dir, "roc_curve.png"))

    # 5. 混淆矩阵
    cm_df = pd.read_csv(os.path.join(results_dir, "confusion_matrix.csv"), index_col=0)
    plot_confusion_matrix(cm_df, os.path.join(results_dir, "confusion_matrix.png"))

    log("All visualizations created in /results/.")


if __name__ == "__main__":
    main()
