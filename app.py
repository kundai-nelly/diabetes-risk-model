# app.py
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score)

HERE = Path(__file__).resolve().parent

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(page_title="Diabetes Risk Intelligence", layout="wide")
st.title("🩺 Diabetes Risk Intelligence Dashboard")
st.markdown("### Helping healthcare professionals assess diabetes risk using machine learning")

# ------------------------------
# Cached data loading (with feature engineering)
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(HERE / "data" / "diabetes_t2dm_africa_extra_large_10000.csv")
    if "patient_id" in df.columns:
        df.drop(columns=["patient_id"], inplace=True)

    # Leakage removal
    leakage_cols = [
        "diagnosed", "years_since_diagnosis", "on_medication", "medication_type",
        "adherence", "microalbuminuria", "retinopathy", "nephropathy", "neuropathy",
        "cardiovascular_disease", "diabetic_foot_ulcer", "preeclampsia",
        "cesarean_delivery", "macrosomic_baby", "neonatal_hypoglycemia", "nicu_admission"
    ]
    existing = [c for c in leakage_cols if c in df.columns]
    df.drop(columns=existing, inplace=True, errors="ignore")

    # Target mapping: Diabetic = 1, everything else = 0
    target_map = {"Diabetic": 1, "Normal": 0, "Prediabetes": 0}
    df["diabetes_status"] = df["diabetes_status"].map(target_map)
    df.dropna(subset=["diabetes_status"], inplace=True)
    df["diabetes_status"] = df["diabetes_status"].astype(int)

    # Zero → missing for physiological measurements
    zero_cols = [
        "fasting_glucose_mg_dl", "hba1c_percent", "total_cholesterol_mg_dl",
        "ldl_mg_dl", "hdl_mg_dl", "triglycerides_mg_dl", "creatinine_mg_dl"
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # FIX: these arrive as pandas bool dtype, which is neither np.number nor
    # object/category -- the num_features/cat_features split below used to
    # miss them entirely, and ColumnTransformer's default remainder="drop"
    # silently excluded every one of them from the model. Cast to int so
    # they're picked up as numeric features from here on.
    bool_cols = [
        "is_pregnant", "family_history_diabetes", "previous_gdm", "physically_active",
        "has_hypertension", "previous_macrosomia", "pcos", "hiv_positive",
    ]
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # FIX: `parity` is only recorded for the ~1.4% of rows where is_pregnant
    # was True -- NaN elsewhere means "not applicable" (including every
    # male patient), not "missing". The training pipeline imputes 0 for
    # these rather than the median of the tiny pregnant-only subgroup; the
    # app has to match that exactly or its predictions drift from what the
    # model was actually trained on.
    df["parity"] = df["parity"].fillna(0)

    # Feature engineering (exactly as in training)
    df["glucose_hba1c_ratio"] = df["fasting_glucose_mg_dl"] / (df["hba1c_percent"] + 1e-5)
    df["bmi_age_interaction"] = df["bmi"] * df["age"]
    df["cholesterol_ratio"] = df["total_cholesterol_mg_dl"] / (df["hdl_mg_dl"] + 1e-5)
    # FIX: has_hypertension is now int (0/1) after the cast above, so this is
    # a plain elementwise +1 -- matches training. (The live-prediction and
    # what-if pages had a second, worse version of this bug: see below.)
    df["renal_risk"] = df["creatinine_mg_dl"] * (df["has_hypertension"] + 1)

    return df

@st.cache_resource
def load_pipeline():
    artifact = joblib.load(HERE / "diabetes_production_model.pkl")
    return artifact["pipeline"], artifact["model_name"]

@st.cache_resource
def load_explainer():
    return joblib.load(HERE / "shap_explainer.pkl")

df = load_data()
pipeline, model_name = load_pipeline()
explainer = load_explainer()

# Extract preprocessor and feature names
preprocessor = pipeline.named_steps["preprocessor"]
num_features = df.drop("diabetes_status", axis=1).select_dtypes(include=np.number).columns.tolist()
cat_features = df.drop("diabetes_status", axis=1).select_dtypes(include=["object", "category", "str"]).columns.tolist()
bool_cols = [
    "is_pregnant", "family_history_diabetes", "previous_gdm", "physically_active",
    "has_hypertension", "previous_macrosomia", "pcos", "hiv_positive",
]
# SHAP feature names after one-hot encoding
feature_names = (num_features +
                 list(preprocessor.named_transformers_["cat"]
                      .named_steps["onehot"]
                      .get_feature_names_out(cat_features)))


def explain(input_df):
    """Run the deployed model's SHAP explainer on one row and return
    (values, base_value, transformed_row) for the "diabetic" class."""
    X_trans = preprocessor.transform(input_df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    sv = explainer(X_trans)
    values = sv.values[0]
    base = sv.base_values[0]
    if np.ndim(values) == 2:  # (n_features, n_classes) -> take "diabetic"
        values = values[:, 1]
        base = base[1] if np.ndim(base) else base
    return np.asarray(values).ravel(), float(np.asarray(base).ravel()[0]), X_trans[0]


# ------------------------------
# Helper: Plain‑English insight for a numeric feature
# ------------------------------
def feature_insight(feat_name):
    """Return a simple sentence comparing medians between diabetic and non-diabetic groups."""
    med0 = df[df["diabetes_status"]==0][feat_name].median()
    med1 = df[df["diabetes_status"]==1][feat_name].median()
    direction = "higher" if med1 > med0 else "lower"
    return (
        f"On average, diabetic patients have a **{direction}** {feat_name} "
        f"(median {med1:.1f} vs {med0:.1f} in non‑diabetic)."
    )

# ------------------------------
# Sidebar navigation
# ------------------------------
page = st.sidebar.radio("Navigate",
    ["🏠 Home", "📊 Data Exploration", "📈 Model Performance",
     "🔮 Live Prediction", "🧪 What‑if Analysis"])

# ==============================
# HOME
# ==============================
if page == "🏠 Home":
    st.header("Welcome to the Diabetes Risk Intelligence Tool")
    st.markdown(f"""
    **What does this tool do?**
    This application uses machine learning ({model_name}) to estimate a person's
    risk of having **diabetes**, based on demographics, history, and lab values.

    **Key features (in plain English):**
    - 🔒 **No cheating** – the model only uses pre‑diagnosis information (no leakage from later complications).
    - ⚖️ **Handles unbalanced data** – this dataset actually skews the other way: 74.6% of patients are diabetic, not the minority. SMOTE and class weighting correct for whichever class is smaller either way.
    - 🧠 **Compared against five other models** – Random Forest, XGBoost, LightGBM, CatBoost and a neural net were all evaluated; this one ships because its coefficients are directly auditable, not because it scored highest (the six were statistically tied).
    - 🧐 **Explains every prediction** – you can see *why* a patient is flagged as high risk (SHAP waterfall charts).
    - 🎮 **Interactive scenarios** – change one risk factor at a time and see how the risk changes.

    **One honest caveat:** `fasting_glucose_mg_dl` and `hba1c_percent` are the actual
    WHO/ADA diagnostic criteria for diabetes, not just risk factors — a simple
    threshold rule on those two values alone matches this dataset's labels 99.2%
    of the time. Treat this tool as decision support alongside those lab results,
    not as a substitute for testing someone who hasn't had them run yet.
    """)

# ==============================
# DATA EXPLORATION (EDA)
# ==============================
elif page == "📊 Data Exploration":
    st.header("Dataset Overview")
    st.markdown("Here we explore the data **before** building any model.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("How many diabetic vs non‑diabetic patients?")
        fig, ax = plt.subplots()
        counts = df["diabetes_status"].value_counts().rename({1:"Diabetic", 0:"Non‑diabetic"})
        counts.plot(kind="bar", ax=ax, color=["#ff7f0e", "#1f77b4"])
        ax.set_ylabel("Number of patients")
        st.pyplot(fig)
        st.caption(
            f"Diabetic patients are actually the majority here ({df['diabetes_status'].mean():.1%}), "
            "not the minority — corrected from the original description. SMOTE and class "
            "weighting still handle it correctly either way, since both just rebalance "
            "toward whichever class has fewer rows."
        )

    with col2:
        st.subheader("Missing measurements")
        fig, ax = plt.subplots()
        sns.heatmap(df.isna(), cbar=False, ax=ax)
        st.pyplot(fig)
        st.caption("Some values (e.g., glucose, cholesterol) were recorded as zero; we treat them as missing and fill with sensible replacements.")

    st.subheader("Feature Distributions by Diabetes Status")
    st.markdown("Select a feature to see how it differs between diabetic and non‑diabetic patients.")

    # Dropdown with all numeric features (including engineered ones and,
    # now that the dtype bug is fixed, the boolean risk factors too)
    all_num_features = [f for f in num_features if f in df.columns]  # keep all
    feat = st.selectbox("Choose a feature", all_num_features)

    fig, ax = plt.subplots()
    sns.histplot(df, x=feat, hue="diabetes_status", kde=True, ax=ax,
                 palette={0:"#1f77b4", 1:"#ff7f0e"})
    ax.set_title(f"Distribution of {feat}")
    ax.set_xlabel(feat)
    ax.set_ylabel("Number of patients")
    # Legend
    ax.legend(labels=["Non‑diabetic", "Diabetic"])
    st.pyplot(fig)

    # Plain‑language insight
    st.markdown("**Insight:** " + feature_insight(feat))

    st.subheader("Correlation Matrix")
    st.markdown("How strongly are different measurements related to each other?")
    fig, ax = plt.subplots(figsize=(12,10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    ax.set_title("Correlation between numerical features")
    st.pyplot(fig)
    st.caption("Red = positive correlation, Blue = negative correlation. Strong correlations help the model make better predictions.")

# ==============================
# MODEL PERFORMANCE
# ==============================
elif page == "📈 Model Performance":
    st.header("How Well Does the Model Perform?")
    st.markdown(f"We tested the final model ({model_name}) on a separate set of patients it never saw during training.")
    st.info(
        "Read the AUC below with one caveat in mind: `fasting_glucose_mg_dl` and "
        "`hba1c_percent` are the diagnostic criteria the label was built from, so "
        "very high separation here reflects the pipeline correctly recovering a "
        "known threshold rule, not a discovery about hidden risk patterns.",
        icon="ℹ️",
    )

    # Prepare test set
    X = df.drop("diabetes_status", axis=1)
    y = df["diabetes_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1]

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Display metrics in simple language
    st.subheader("Performance Summary")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Accuracy", f"{acc:.1%}", help="Overall, how often the model is correct.")
    col_m2.metric("Recall (Sensitivity)", f"{rec:.1%}", help="How many actual diabetic patients were correctly identified.")
    col_m3.metric("Precision", f"{prec:.1%}", help="When the model says 'diabetic', how often it is right.")
    col_m4, col_m5, col_m6 = st.columns(3)
    col_m4.metric("F1 Score", f"{f1:.2f}", help="Balance between precision and recall.")
    col_m5.metric("AUC (ROC)", f"{roc_auc:.2f}", help="Ability to separate diabetic from non‑diabetic. 1.0 is perfect.")

    st.markdown("""
    **What do these numbers mean?**
    - **Recall** is crucial in screening – we want to catch as many true diabetic patients as possible, even if we sometimes flag a healthy person.
    - **Precision** tells us how many of those flagged actually have diabetes.
    - The **AUC** of {:.2f} indicates the model is very good at distinguishing the two groups.
    """.format(roc_auc))

    # ROC curve
    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel("False Positive Rate (healthy wrongly flagged)")
    ax.set_ylabel("True Positive Rate (diabetic correctly flagged)")
    ax.set_title("ROC Curve – Model Discrimination")
    ax.legend()
    st.pyplot(fig)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non‑diabetic", "Diabetic"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
    st.caption("Top‑left: healthy correctly predicted; bottom‑right: diabetic correctly predicted.")

    # SHAP global importance
    st.subheader("Which Factors Are Most Important Overall?")
    st.image(str(HERE / "assets" / "shap_summary.png"))
    st.caption("Red dots = high feature value increases risk; blue = decreases risk. The wider the spread, the more important.")

# ==============================
# LIVE PREDICTION
# ==============================
elif page == "🔮 Live Prediction":
    st.header("Estimate a Patient's Diabetes Risk")
    st.markdown("Fill in the information below. The model will return a risk probability and explain its reasoning.")

    # Inputs with plain‑English labels
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", 0, 120, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        is_pregnant = 0 if sex=="Male" else st.number_input("Currently pregnant? (1=Yes, 0=No)", 0, 1, 0)
        residence = st.selectbox("Residence", ["Urban", "Rural"])
        # FIX: the original offered "None"/"Tertiary", neither of which
        # exist in the training data (real values: Primary/Secondary/
        # Higher). OneHotEncoder(handle_unknown="ignore") doesn't error on
        # an unrecognized category, it silently encodes it as all-zeros --
        # so picking "None" or "Tertiary" fed the model no education signal
        # at all, with no warning that anything was wrong.
        education = st.selectbox("Education Level", ["Primary", "Secondary", "Higher"])
    with col2:
        bmi = st.number_input("Body Mass Index (BMI)", 10.0, 70.0, 28.0)
        # FIX: same silent-mismatch issue -- "Underweight"/"Obese" aren't in
        # the training data (real values: Normal/Overweight/Obese I/Obese II+).
        bmi_cat = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese I", "Obese II+"])
        family_hist = st.selectbox("Family History of Diabetes", ["Yes", "No"])
        previous_gdm = st.selectbox("Previous Gestational Diabetes", ["Yes", "No"])
        phys_active = st.selectbox("Physically Active", ["Yes", "No"])
    with col3:
        hypertension = st.selectbox("High Blood Pressure", ["Yes", "No"])
        parity = st.number_input("Number of pregnancies (parity)", 0, 10, 1 if is_pregnant else 0)
        prev_macrosomia = st.selectbox("Previous Macrosomic Baby (>4kg)", ["Yes", "No"])
        pcos = st.selectbox("Polycystic Ovary Syndrome (PCOS)", ["Yes", "No"])
        hiv = st.selectbox("HIV Positive", ["Yes", "No"])
        glucose = st.number_input("Fasting Blood Sugar (mg/dL)", 0, 400, 100)
        hba1c = st.number_input("HbA1c (%)", 0.0, 15.0, 5.5)

    col4, col5 = st.columns(2)
    with col4:
        total_chol = st.number_input("Total Cholesterol (mg/dL)", 0, 400, 190)
        ldl = st.number_input("LDL Cholesterol (mg/dL)", 0, 300, 110)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", 0, 100, 50)
    with col5:
        trig = st.number_input("Triglycerides (mg/dL)", 0, 600, 150)
        creatinine = st.number_input("Creatinine (mg/dL)", 0.0, 10.0, 0.9)

    # FIX: booleans must be sent as the same 0/1 ints the model was trained
    # on, not "Yes"/"No" strings -- these columns used to be dropped
    # entirely (see load_data()), so this mismatch never surfaced before.
    hypertension_flag = 1 if hypertension == "Yes" else 0
    family_hist_flag = 1 if family_hist == "Yes" else 0
    previous_gdm_flag = 1 if previous_gdm == "Yes" else 0
    phys_active_flag = 1 if phys_active == "Yes" else 0
    prev_macrosomia_flag = 1 if prev_macrosomia == "Yes" else 0
    pcos_flag = 1 if pcos == "Yes" else 0
    hiv_flag = 1 if hiv == "Yes" else 0

    # Derived features (same as training)
    glu_hba1c_ratio = glucose / (hba1c + 1e-5)
    bmi_age = bmi * age
    chol_ratio = total_chol / (hdl + 1e-5)
    # FIX: the original wrote `1 if hypertension=="Yes" else 0 + 1`. Python's
    # ternary binds looser than `+`, so that's `1 if ... else (0+1)` -- BOTH
    # branches evaluate to 1, so renal_risk was always creatinine*1 here,
    # never doubled for hypertensive patients the way training intended.
    renal = creatinine * (hypertension_flag + 1)

    input_dict = {
        "age": age, "sex": sex, "is_pregnant": is_pregnant,
        "residence": residence, "education": education,
        "bmi": bmi, "bmi_category": bmi_cat,
        "family_history_diabetes": family_hist_flag,
        "previous_gdm": previous_gdm_flag,
        "physically_active": phys_active_flag,
        "has_hypertension": hypertension_flag,
        "parity": parity,
        "previous_macrosomia": prev_macrosomia_flag,
        "pcos": pcos_flag,
        "hiv_positive": hiv_flag,
        "fasting_glucose_mg_dl": glucose,
        "hba1c_percent": hba1c,
        "total_cholesterol_mg_dl": total_chol,
        "ldl_mg_dl": ldl,
        "hdl_mg_dl": hdl,
        "triglycerides_mg_dl": trig,
        "creatinine_mg_dl": creatinine,
        "glucose_hba1c_ratio": glu_hba1c_ratio,
        "bmi_age_interaction": bmi_age,
        "cholesterol_ratio": chol_ratio,
        "renal_risk": renal
    }
    input_df = pd.DataFrame([input_dict])

    if st.button("Calculate Risk"):
        prob = pipeline.predict_proba(input_df)[0, 1]
        pred = pipeline.predict(input_df)[0]
        st.subheader("Result")
        if pred == 1:
            st.error(f"**High risk of diabetes** (probability: {prob:.1%})")
        else:
            st.success(f"**Low risk of diabetes** (probability: {prob:.1%})")

        # Explain with SHAP (the deployed model's own explainer)
        st.subheader("What Drives This Prediction?")
        shap_vals, base_value, x_row = explain(input_df)
        fig, ax = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(values=shap_vals,
                             base_values=base_value,
                             data=x_row,
                             feature_names=feature_names),
            show=False
        )
        ax.set_xlabel("Impact on risk (log-odds)")
        ax.set_title("Contribution of each factor")
        st.pyplot(fig)

        # Top 3 positive and negative contributors in plain English
        shap_series = pd.Series(shap_vals, index=feature_names).sort_values(ascending=False)
        top_pos = shap_series.head(3)
        top_neg = shap_series.tail(3)

        st.markdown("**Top factors that increased risk:**")
        for feat, val in top_pos.items():
            if val > 0:
                st.write(f"- **{feat}**: pushed risk up by {val:.2f} log-odds")
        st.markdown("**Top factors that decreased risk:**")
        for feat, val in top_neg.items():
            if val < 0:
                st.write(f"- **{feat}**: reduced risk by {abs(val):.2f} log-odds")

# ==============================
# WHAT‑IF ANALYSIS
# ==============================
elif page == "🧪 What‑if Analysis":
    st.header("Explore Risk Scenarios")
    st.markdown("Change one factor at a time and see how the predicted risk changes. All other values stay at typical levels.")

    # FIX: booleans as 0/1 ints (matching training), not "Yes"/"No" strings.
    base = {
        "age": 45, "sex": "Female", "is_pregnant": 0, "residence": "Urban",
        "education": "Secondary", "bmi": 28.0, "bmi_category": "Overweight",
        "family_history_diabetes": 0, "previous_gdm": 0,
        "physically_active": 1, "has_hypertension": 0,
        "parity": 0, "previous_macrosomia": 0, "pcos": 0,
        "hiv_positive": 0, "fasting_glucose_mg_dl": 100,
        "hba1c_percent": 5.5, "total_cholesterol_mg_dl": 190,
        "ldl_mg_dl": 110, "hdl_mg_dl": 50, "triglycerides_mg_dl": 150,
        "creatinine_mg_dl": 0.9,
        "glucose_hba1c_ratio": 100/5.5,
        "bmi_age_interaction": 28*45,
        "cholesterol_ratio": 190/50,
        "renal_risk": 0.9 * (0 + 1),
    }

    all_num = [f for f in num_features if f in df.columns]
    feature_to_vary = st.selectbox("Which factor would you like to change?", all_num)

    is_binary_feature = feature_to_vary in bool_cols
    if is_binary_feature:
        # A binary risk factor only has two real states -- sweeping a
        # linspace(0, 1, 30) across it would plot 28 fractional values
        # ("0.34 family history") that no real patient can have.
        range_vals = np.array([0, 1])
    else:
        data_col = df[feature_to_vary].dropna()
        low_perc = np.percentile(data_col, 5)
        high_perc = np.percentile(data_col, 95)
        if low_perc == high_perc:
            low_perc, high_perc = data_col.min(), data_col.max()
        range_vals = np.linspace(low_perc, high_perc, 30)

    probas = []
    for val in range_vals:
        instance = base.copy()
        instance[feature_to_vary] = val
        if feature_to_vary in ["fasting_glucose_mg_dl", "hba1c_percent"]:
            instance["glucose_hba1c_ratio"] = instance["fasting_glucose_mg_dl"] / (instance["hba1c_percent"] + 1e-5)
        if feature_to_vary in ["bmi", "age"]:
            instance["bmi_age_interaction"] = instance["bmi"] * instance["age"]
        if feature_to_vary in ["total_cholesterol_mg_dl", "hdl_mg_dl"]:
            instance["cholesterol_ratio"] = instance["total_cholesterol_mg_dl"] / (instance["hdl_mg_dl"] + 1e-5)
        if feature_to_vary in ["creatinine_mg_dl", "has_hypertension"]:
            # FIX: same operator-precedence bug as Live Prediction -- see
            # the comment there. `has_hypertension` is already 0/1 here.
            instance["renal_risk"] = instance["creatinine_mg_dl"] * (instance["has_hypertension"] + 1)

        input_df = pd.DataFrame([instance])
        prob = pipeline.predict_proba(input_df)[0, 1]
        probas.append(prob)

    # ----- Minimal-margin, dynamically zoomed plot -----
    fig, ax = plt.subplots()
    if is_binary_feature:
        ax.bar(["No", "Yes"], probas, color="#d62728", width=0.5)
    else:
        ax.plot(range_vals, probas, marker='o', color="#d62728")
    ax.set_xlabel(feature_to_vary, labelpad=8)
    ax.set_ylabel("Predicted Diabetes Risk", labelpad=8)
    ax.set_title(f"Risk vs {feature_to_vary}" + ("" if is_binary_feature else " (middle 90% of values)"), pad=8)
    ax.grid(True)

    y_min = min(probas)
    y_max = max(probas)
    y_range = y_max - y_min
    if y_range == 0:
        y_range = 0.05
    ax.set_ylim(max(0, y_min - y_range*0.1), min(1, y_max + y_range*0.1))

    # Tighten internal figure margins
    fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)
    ax.margins(x=0.01, y=0.05)

    # Category lines and labels (placed very close to the bottom of the visible range)
    if feature_to_vary == "age":
        ax.axvline(30, color='grey', linestyle='--', alpha=0.7)
        ax.axvline(50, color='grey', linestyle='--', alpha=0.7)
        ax.text(20, y_min + y_range*0.03, "Young", ha='center', fontsize=9)
        ax.text(40, y_min + y_range*0.03, "Middle age", ha='center', fontsize=9)
        ax.text(60, y_min + y_range*0.03, "Senior", ha='center', fontsize=9)
    elif feature_to_vary == "bmi":
        ax.axvline(18.5, color='grey', linestyle='--', alpha=0.7)
        ax.axvline(25, color='grey', linestyle='--', alpha=0.7)
        ax.axvline(30, color='grey', linestyle='--', alpha=0.7)
        ax.text(16, y_min + y_range*0.03, "Underweight", ha='center', fontsize=9)
        ax.text(22, y_min + y_range*0.03, "Normal", ha='center', fontsize=9)
        ax.text(27, y_min + y_range*0.03, "Overweight", ha='center', fontsize=9)
        ax.text(33, y_min + y_range*0.03, "Obese", ha='center', fontsize=9)

    st.pyplot(fig, width="content")
    # ---------------------------------------------------

    st.markdown(
        f"**Takeaway:** When **{feature_to_vary}** changes from {range_vals[0]:.1f} to {range_vals[-1]:.1f} "
        f"{'' if is_binary_feature else '(middle 90% of patients) '}the estimated diabetes risk goes from "
        f"**{probas[0]:.1%}** to **{probas[-1]:.1%}**."
    )
    if not is_binary_feature:
        st.caption("Extreme values (below 5th or above 95th percentile) are excluded to keep the graph readable.")
