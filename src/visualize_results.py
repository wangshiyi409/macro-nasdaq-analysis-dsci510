"""
visualize_results.py
--------------------
Create all visualizations used in the final project:

1. Correlation heatmap (from processed macro correlation coefficient)
2. Macro indicators vs NASDAQ (4x2 subplot)
3. ROC curve (loaded from Excel)
4. Confusion Matrix (loaded from Excel)

All generated images will be saved in the /results/ folder.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # ============================================================
    # 0. Prepare paths
    # ============================================================
    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)

    # ============================================================
    # 1. Load Data
    # ============================================================
    df_raw = pd.read_excel("data/processed/processed macro correlation coefficient.xlsx", index_col=0)
    market_data = pd.read_excel("data/processed/processed market data.xlsx", index_col=0)

    # The strong macro indicators you manually selected
    market_data_strong = market_data.loc[:, [
        'BUFFETT_INDICATOR', 'RSAFS', 'GDP', 'CPIAUCSL',
        'HOUST', 'DGS3MO', 'FEDFUNDS', 'nasdaq_close'
    ]]

    # ============================================================
    # 2. Correlation Heatmap
    # ============================================================
    df_corr = df_raw.set_index("Indicator").T

    plt.figure(figsize=(14, 3))
    sns.heatmap(
        df_corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Correlation / Value', 'shrink': 0.8},
        linewidths=0.5,
        linecolor="white"
    )

    plt.title("Economic Indicators Correlation Heatmap", fontsize=16, pad=25)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    heatmap_path = os.path.join(results_dir, "heatmap_correlation.png")
    plt.savefig(heatmap_path, dpi=300)
    print(f"[Saved] {heatmap_path}")
    plt.close()

    # ============================================================
    # 3. Statistics printout
    # ============================================================
    print("Data Statistics:")
    print(f"Range: {min(df_raw['Value']):.3f} → {max(df_raw['Value']):.3f}")
    print(f"Mean: {np.mean(df_raw['Value']):.3f}")
    print(f"Positive count: {sum(x > 0 for x in df_raw['Value'])}")
    print(f"Negative count: {sum(x < 0 for x in df_raw['Value'])}")

    # ============================================================
    # 4. Macro Indicators vs NASDAQ (4x2 subplot)
    # ============================================================
    df_plot = market_data_strong.dropna(how="all")
    features = [
        'BUFFETT_INDICATOR', 'RSAFS', 'GDP', 'CPIAUCSL',
        'HOUST', 'DGS3MO', 'FEDFUNDS'
    ]

    sns.set_style("whitegrid")

    color_main = sns.color_palette("Blues", 5)[3]
    color_ref = sns.color_palette("Reds", 5)[3]

    plt.rcParams['axes.edgecolor'] = '#aaaaaa'
    plt.rcParams['axes.linewidth'] = 0.7
    plt.rcParams['font.size'] = 10

    fig, axes = plt.subplots(4, 2, figsize=(17, 18))
    axes = axes.flatten()

    for ax, feature in zip(axes, features):

        df_plot[feature].dropna().plot(
            ax=ax,
            label=feature,
            color=color_main,
            linewidth=2
        )

        ax2 = ax.twinx()
        df_plot["nasdaq_close"].dropna().plot(
            ax=ax2,
            label="NASDAQ",
            color=color_ref,
            alpha=0.65,
            linewidth=2
        )

        ax.set_title(f"{feature} vs NASDAQ Composite", fontsize=13, fontweight="bold", pad=8)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        ax2.legend(loc="upper right", fontsize=9, frameon=False)

        ax.grid(axis='x', linestyle="--", linewidth=0.6, alpha=0.5)
        ax2.grid(axis='y', linestyle="--", linewidth=0.6, alpha=0.4)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax2.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    # Hide empty subplots
    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle(
        "Macro Indicators vs NASDAQ Composite Index",
        fontsize=18,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()

    macro_plot_path = os.path.join(results_dir, "macro_indicators_vs_nasdaq.png")
    plt.savefig(macro_plot_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {macro_plot_path}")
    plt.close()

    # ============================================================
    # 5. ROC Curve
    # ============================================================
    roc_df = pd.read_excel("data/processed/processed results_tail_risk.xlsx", sheet_name="ROC_Data")
    auc_df = pd.read_excel("data/processed/processed results_tail_risk.xlsx", sheet_name="AUC")
    auc_value = auc_df["AUC"].iloc[0]

    plt.figure(figsize=(8, 6))
    plt.plot(roc_df["fpr"], roc_df["tpr"], label=f"ROC Curve (AUC = {auc_value:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")

    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    roc_path = os.path.join(results_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    print(f"[Saved] {roc_path}")
    plt.close()

    # ============================================================
    # 6. Confusion Matrix
    # ============================================================
    cm_df = pd.read_excel("data/processed/processed results_tail_risk.xlsx", sheet_name="Confusion_Matrix", index_col=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    confusion_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(confusion_path, dpi=300)
    print(f"[Saved] {confusion_path}")
    plt.close()

    # ============================================================
    # Done
    # ============================================================
    print("All visualizations saved in /results/")


if __name__ == "__main__":
    main()

