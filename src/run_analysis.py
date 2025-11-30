import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression


def main():

    processed_dir = "data/processed/"

    market_data = pd.read_excel(processed_dir + "processed_market_data.xlsx", index_col=0)

    macro_features = [
        "GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS", "INDPRO",
        "RSAFS", "HOUST", "DGS3MO", "VIXCLS", "BUFFETT_INDICATOR"
    ]

    # === Compute correlations ===
    correlations = {}

    for feature in macro_features:
        temp = market_data[[feature, "nasdaq_close"]].dropna()
        correlations[feature] = temp[feature].corr(temp["nasdaq_close"])

    corr_df = pd.DataFrame.from_dict(correlations, orient="index", columns=["correlation_with_nasdaq"])
    corr_df.to_excel(processed_dir + "processed_macro_correlation.xlsx")

    # === Select strong features ===
    strong_features = corr_df[abs(corr_df["correlation_with_nasdaq"]) > 0.5].index.tolist()

    market_data_strong = market_data[strong_features + ["nasdaq_close"]].copy()
    market_data_strong.index = pd.to_datetime(market_data_strong.index)

    # === Risk Label (future 60-day drawdown) ===
    horizon = 60

    market_data_strong["min_price_60d"] = (
        market_data_strong["nasdaq_close"].rolling(window=horizon, min_periods=1)
        .min()
        .shift(-horizon + 1)
    ).bfill()

    market_data_strong["max_drawdown_60d"] = (
        market_data_strong["min_price_60d"] / market_data_strong["nasdaq_close"] - 1
    ).bfill()

    market_data_strong["tail_risk_60d"] = (market_data_strong["max_drawdown_60d"] <= -0.02).astype(int)

    # === Build model ===
    df_model = market_data_strong.dropna(subset=["tail_risk_60d"])

    X = df_model[strong_features]
    y = df_model["tail_risk_60d"]

    X = X.ffill().bfill()

    # === Train-test split by date ===
    split_date = "2021-01-01"
    X_train = X[X.index < split_date]
    y_train = y[y.index < split_date]
    X_test = X[X.index >= split_date]
    y_test = y[y.index >= split_date]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    auc_score = auc(*roc_curve(y_test, proba_test)[:2])

    # === Save ROC, AUC, Confusion Matrix ===
    fpr, tpr, thresholds = roc_curve(y_test, proba_test)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    auc_df = pd.DataFrame({"AUC": [auc_score]})
    cm = confusion_matrix(y_test, (proba_test > 0.5).astype(int))
    cm_df = pd.DataFrame(cm, columns=["Pred_0", "Pred_1"], index=["Actual_0", "Actual_1"])

    with pd.ExcelWriter(processed_dir + "results_tail_risk.xlsx") as writer:
        roc_df.to_excel(writer, sheet_name="ROC_Data", index=False)
        auc_df.to_excel(writer, sheet_name="AUC", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")

    print("[Saved] results_tail_risk.xlsx")


if __name__ == "__main__":
    main()

