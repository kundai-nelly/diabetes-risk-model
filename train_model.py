"""
train_model.py — Type 2 diabetes screening risk model.

Cleans diabetes_t2dm_africa_extra_large_10000.csv, engineers features,
compares six classifiers under 5-fold stratified CV with in-fold SMOTE,
tunes the winner with Optuna, and saves the final pipeline + a SHAP
explainer fit on the actual deployed model.

Script form of notebooks/Diabetes_Risk_Analysis.ipynb.

Run with:  python train_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "diabetes_t2dm_africa_extra_large_10000.csv"
MODEL_PATH = HERE / "diabetes_production_model.pkl"
EXPLAINER_PATH = HERE / "shap_explainer.pkl"

LEAKAGE_COLS = [
    "diagnosed", "years_since_diagnosis", "on_medication", "medication_type",
    "adherence", "microalbuminuria", "retinopathy", "nephropathy", "neuropathy",
    "cardiovascular_disease", "diabetic_foot_ulcer", "preeclampsia",
    "cesarean_delivery", "macrosomic_baby", "neonatal_hypoglycemia", "nicu_admission",
]
ZERO_IMPUTE_COLS = [
    "fasting_glucose_mg_dl", "hba1c_percent", "total_cholesterol_mg_dl",
    "ldl_mg_dl", "hdl_mg_dl", "triglycerides_mg_dl", "creatinine_mg_dl",
]
# These arrived as pandas bool dtype. bool is neither np.number nor
# object/category, so the original script's
# X.select_dtypes(include=np.number) / .select_dtypes(include=["object"])
# split silently dropped every one of them -- 8 of the model's clinical
# risk factors (family history, hypertension, PCOS, HIV status, physical
# activity, previous GDM, previous macrosomia, current pregnancy) were
# never seen by any of the six models compared below. Confirmed by
# re-running the original's exact selection code and diffing the columns
# it picked against df.columns. Cast to int and included explicitly here.
BOOLEAN_COLS = [
    "is_pregnant", "family_history_diabetes", "previous_gdm", "physically_active",
    "has_hypertension", "previous_macrosomia", "pcos", "hiv_positive",
]
CATEGORICAL_FEATURES = ["sex", "residence", "education", "bmi_category"]
ENGINEERED_FEATURES = ["glucose_hba1c_ratio", "bmi_age_interaction", "cholesterol_ratio", "renal_risk"]
NUMERIC_FEATURES = (
    ["age", "bmi", "parity"] + ZERO_IMPUTE_COLS + BOOLEAN_COLS + ENGINEERED_FEATURES
)


def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "patient_id" in df.columns:
        df = df.drop(columns=["patient_id"])

    existing_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=existing_leakage, errors="ignore")

    target_map = {"Diabetic": 1, "Normal": 0, "Prediabetes": 0}
    df["diabetes_status"] = df["diabetes_status"].map(target_map)
    df = df.dropna(subset=["diabetes_status"])
    df["diabetes_status"] = df["diabetes_status"].astype(int)

    for col in ZERO_IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    for col in BOOLEAN_COLS:
        df[col] = df[col].astype(int)

    # `parity` (number of prior pregnancies) is only recorded for the ~1.4%
    # of rows where is_pregnant == True -- everyone else (all men, all
    # non-pregnant women) is NaN because the field doesn't apply to them,
    # not because it's missing data. SimpleImputer(strategy="median") in
    # the original script filled ALL of those NaNs with the median of the
    # 142 pregnant rows (~2), fabricating "2 prior pregnancies" for every
    # man in the dataset. 0 is the honest default for "not applicable".
    df["parity"] = df["parity"].fillna(0)

    df["glucose_hba1c_ratio"] = df["fasting_glucose_mg_dl"] / (df["hba1c_percent"] + 1e-5)
    df["bmi_age_interaction"] = df["bmi"] * df["age"]
    df["cholesterol_ratio"] = df["total_cholesterol_mg_dl"] / (df["hdl_mg_dl"] + 1e-5)
    df["renal_risk"] = df["creatinine_mg_dl"] * (df["has_hypertension"] + 1)

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def create_pipeline(classifier, preprocessor) -> ImbPipeline:
    # SMOTE runs inside the pipeline, so it only ever sees a training fold
    # (never the held-out fold) -- correct, matches the original.
    return ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier),
    ])


def main():
    df = load_and_clean()
    y = df["diabetes_status"]
    X = df.drop(columns=["diabetes_status"])
    print(f"Rows: {len(df)}. Diabetic (positive class): {y.mean():.1%}.")

    preprocessor = build_preprocessor()

    # SMOTE already rebalances every training fold to ~1:1 before each
    # classifier sees it, and each model's class_weight="balanced" then
    # recomputes from that already-balanced fold, so it self-corrects to
    # ~1.0 automatically. XGBoost has no class_weight; the original passed
    # a fixed scale_pos_weight computed from the *global* imbalance on top
    # of the *already SMOTE-balanced* fold, double-correcting in a way the
    # other five models don't. Dropped here for consistency -- SMOTE alone
    # handles balance for every model in this comparison.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(class_weight="balanced", verbose=-1, random_state=42),
        "CatBoost": CatBoostClassifier(silent=True, auto_class_weights="Balanced", random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = {}
    for name, clf in models.items():
        print(f"Evaluating {name}...")
        pipe = create_pipeline(clf, preprocessor)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {metric: float(np.mean(scores[f"test_{metric}"])) for metric in scoring}
        print(f"  ROC AUC: {results[name]['roc_auc']:.3f}  F1: {results[name]['f1']:.3f}")

    results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    print(f"\nBest by raw CV ROC-AUC: {results_df.index[0]} ({results_df.iloc[0]['roc_auc']:.4f})")

    # --- Optuna tuning of XGBoost, to demonstrate the fix (see below) with
    # real numbers, regardless of whether XGBoost tops the raw comparison ---
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        }
        clf = XGBClassifier(**params, eval_metric="logloss", random_state=42)
        pipe = create_pipeline(clf, preprocessor)
        # BUG FIXED: the original called cross_validate(..., scoring="roc_auc")
        # -- a single string -- then read scores["test_roc_auc"]. For a
        # single-metric scoring string, cross_validate's key is "test_score",
        # not "test_<name>"; every trial raised KeyError, was caught by a
        # blanket except, and returned 0.0. All 30 trials tied at 0.0, so
        # Optuna's TPE sampler had no signal to optimize against and
        # study.best_params was arbitrary. Passing scoring as a dict keeps
        # the metric name in the result key.
        scores = cross_validate(pipe, X, y, cv=3, scoring={"roc_auc": "roc_auc"}, n_jobs=-1)
        return float(np.mean(scores["test_roc_auc"]))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=25, show_progress_bar=False)
    print(f"Optuna-tuned XGBoost CV ROC-AUC: {study.best_value:.4f} (trial {study.best_trial.number}, "
          f"vs {results['XGBoost']['roc_auc']:.4f} untuned)")

    # --- Final model choice ---
    # Every model above lands within 0.005 ROC-AUC of every other (0.9995-
    # 0.9998) -- see the README for why: fasting_glucose_mg_dl and
    # hba1c_percent are the diagnostic criteria the label was built from, so
    # this is closer to "recover a threshold rule" than "learn a genuinely
    # hard pattern," and a 300-tree gradient booster has no real edge over a
    # linear model on a problem like that. With performance differences this
    # far into noise, and given SHAP explainability is the app's whole
    # second act, Logistic Regression ships: its coefficients are directly
    # auditable by a clinician (SHAP is more trustworthy layered on a model
    # that's already simple than on a black box it's approximating), at a
    # real but tiny cost (0.9934 vs Random Forest's 0.9968 F1).
    final_name = "Logistic Regression"
    final_classifier = models[final_name]
    final_pipe = create_pipeline(final_classifier, preprocessor)
    final_pipe.fit(X, y)
    joblib.dump({
        "pipeline": final_pipe,
        "model_name": final_name,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "boolean_features": BOOLEAN_COLS,
        "cv_results": results_df.to_dict("index"),
    }, MODEL_PATH)
    print(f"Saved production model to {MODEL_PATH}")

    # --- SHAP explainer on the ACTUAL deployed model, on real (non-SMOTE) data ---
    # The original refit a brand-new XGBClassifier on preprocessor.fit_transform(X)
    # (a second, separate fit, not SMOTE-resampled) purely to get something for
    # SHAP to explain -- close to the deployed model in hyperparameters, but not
    # literally the same fitted object. This reuses final_pipe's own already-fitted
    # preprocessor and classifier directly, so the explanations are faithful to
    # what's actually shipped.
    fitted_preprocessor = final_pipe.named_steps["preprocessor"]
    fitted_classifier = final_pipe.named_steps["classifier"]
    X_transformed = fitted_preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    if hasattr(fitted_classifier, "coef_"):
        # LinearExplainer computes exact SHAP values analytically for a
        # linear model -- no sampling, no repeated predict_proba calls. The
        # generic shap.Explainer(model.predict_proba, background) tried
        # first here took ~7 seconds per row (it perturbs and re-predicts
        # many times), which is a real problem for a "click to predict"
        # button. LinearExplainer is the right tool once the shipped model
        # is linear, and it's ~4 orders of magnitude faster.
        explainer = shap.LinearExplainer(fitted_classifier, X_transformed[:500])
    elif hasattr(fitted_classifier, "get_booster") or type(fitted_classifier).__name__ in (
        "RandomForestClassifier", "LGBMClassifier", "CatBoostClassifier",
    ):
        explainer = shap.TreeExplainer(fitted_classifier)
    else:
        background = X_transformed[np.random.RandomState(42).choice(len(X_transformed), 200, replace=False)]
        explainer = shap.Explainer(fitted_classifier.predict_proba, background)
    joblib.dump(explainer, EXPLAINER_PATH)
    print(f"Saved SHAP explainer to {EXPLAINER_PATH}")

    return results_df


if __name__ == "__main__":
    main()
