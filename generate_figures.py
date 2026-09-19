"""
generate_figures.py — regenerates every chart used in README.md and the
notebook, plus metrics.json. Run after train_model.py.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from train_model import (
    BOOLEAN_COLS, CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES,
    EXPLAINER_PATH, load_and_clean,
)

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)
sns.set_style("whitegrid")


def main():
    df = load_and_clean()
    metrics = {"n_rows": len(df), "positive_rate": round(float(df["diabetes_status"].mean()), 4)}

    # --- class balance (corrected direction) --------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df["diabetes_status"].value_counts().rename({1: "Diabetic", 0: "Not diabetic"})
    counts.plot(kind="bar", ax=ax, color=["#C44E52", "#4C72B0"])
    ax.set_ylabel("Patients")
    ax.set_title("Class balance")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ASSETS / "class_balance.png", dpi=140)
    plt.close(fig)

    # --- glucose/HbA1c vs label — the leakage check --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, thresh in zip(axes, ["fasting_glucose_mg_dl", "hba1c_percent"], [126, 6.5]):
        sns.histplot(df, x=col, hue="diabetes_status", kde=True, ax=ax,
                     palette={0: "#4C72B0", 1: "#C44E52"}, legend=(col == "hba1c_percent"))
        ax.axvline(thresh, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(col)
    plt.suptitle("The two diagnostic labs vs. the label they help define")
    plt.tight_layout()
    plt.savefig(ASSETS / "leakage_check.png", dpi=140)
    plt.close(fig)

    rule_pred = ((df["fasting_glucose_mg_dl"] >= 126) | (df["hba1c_percent"] >= 6.5)).astype(int)
    metrics["threshold_rule_agreement"] = round(float((rule_pred == df["diabetes_status"]).mean()), 4)

    # --- missing-data heatmap (structural, not random) -----------------------
    raw = pd.read_csv(HERE / "data" / "diabetes_t2dm_africa_extra_large_10000.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(raw[["education", "parity"]].isna().T, cbar=False, ax=ax,
                cmap=["#4C72B0", "#C44E52"])
    ax.set_title("Missing values: education (survey non-response) vs. parity (not applicable)")
    plt.tight_layout()
    plt.savefig(ASSETS / "missingness.png", dpi=140)
    plt.close(fig)
    metrics["parity_pct_missing"] = round(float(raw["parity"].isna().mean() * 100), 1)
    metrics["parity_recorded_for_pregnant_only"] = bool(
        raw.loc[raw["parity"].notna(), "is_pregnant"].all()
    )

    # --- correlation heatmap, continuous features -----------------------------
    continuous = ["age", "bmi", "fasting_glucose_mg_dl", "hba1c_percent",
                  "total_cholesterol_mg_dl", "ldl_mg_dl", "hdl_mg_dl",
                  "triglycerides_mg_dl", "creatinine_mg_dl"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    corr = df[continuous].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation, continuous clinical features")
    plt.tight_layout()
    plt.savefig(ASSETS / "correlation_heatmap.png", dpi=140)
    plt.close(fig)

    # --- model comparison ------------------------------------------------------
    artifact = joblib.load(MODEL_PATH)
    cv_results = pd.DataFrame(artifact["cv_results"]).T.sort_values("roc_auc", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(cv_results.index, cv_results["roc_auc"], color="#4C72B0")
    ax.set_xlim(0.99, 1.001)
    ax.set_xlabel("5-fold CV ROC-AUC")
    ax.set_title("Model comparison (all within 0.0005 of each other)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(ASSETS / "model_comparison.png", dpi=140)
    plt.close(fig)
    metrics["cv_results"] = cv_results.round(4).to_dict("index")
    metrics["model_name"] = artifact["model_name"]

    # --- logistic regression coefficients --------------------------------------
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names
    coefs = classifier.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    top15 = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index).head(15).sort_values("coef")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = ["#C44E52" if c > 0 else "#4C72B0" for c in top15["coef"]]
    ax.barh(top15["feature"], top15["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (standardized numeric / one-hot categorical)")
    ax.set_title("Top 15 drivers — Logistic Regression")
    plt.tight_layout()
    plt.savefig(ASSETS / "coefficients.png", dpi=140)
    plt.close(fig)
    metrics["top_positive_coefs"] = (
        coef_df[coef_df["coef"] > 0].sort_values("coef", ascending=False)
        [["feature", "coef"]].round(3).head(8).to_dict("records")
    )
    metrics["top_negative_coefs"] = (
        coef_df[coef_df["coef"] < 0].sort_values("coef")
        [["feature", "coef"]].round(3).head(8).to_dict("records")
    )

    # --- SHAP summary plot on the actual deployed model -------------------------
    explainer = joblib.load(EXPLAINER_PATH)
    X = df.drop(columns=["diabetes_status"])
    X_t = preprocessor.transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    sample_idx = np.random.RandomState(42).choice(len(X_t), size=min(300, len(X_t)), replace=False)
    sv = explainer(X_t[sample_idx])
    values = sv.values[..., 1] if sv.values.ndim == 3 else sv.values

    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(values, X_t[sample_idx], feature_names=feature_names, show=False)
    # (LinearExplainer's output is already 2D -- (n, features) -- so the
    # ndim==3 branch above simply doesn't trigger for it; kept generic in
    # case the shipped model type changes later.)
    plt.tight_layout()
    plt.savefig(ASSETS / "shap_summary.png", dpi=140)
    plt.close(fig)

    with open(HERE / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Wrote {ASSETS} charts and metrics.json")


if __name__ == "__main__":
    main()
