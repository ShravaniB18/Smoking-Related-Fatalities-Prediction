import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import statsmodels.api as sm

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(page_title="Hybrid Smoking-Related Fatalities Predictor", page_icon="⚕️", layout="wide")
st.title("⚕️ Hybrid Smoking-Related Fatalities Predictor")
st.markdown("""
This app automatically selects the best model depending on the ICD10 Diagnosis type:
- 🧠 **Poisson GLM Model** → used for `All respiratory diseases`, `All cancers`, `All circulatory diseases`.
- ⚡ **XGBoost Regression Model** → used for all other diagnosis categories.
""")

# =====================================================
# Poisson GLM Loader
# =====================================================
@st.cache_resource
def load_poisson_model():
    artifact_dir = "artifacts"
    model_path = os.path.join(artifact_dir, "best_model_glm.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Poisson model not found in artifacts folder.")
    model = joblib.load(model_path)
    return model

@st.cache_data
def load_poisson_data():
    data_path = os.path.join("artifacts", "model_results.xls")
    if not os.path.exists(data_path):
        st.warning("⚠️ Poisson dataset not found. Prediction only will work.")
        return pd.DataFrame()
    try:
        df = pd.read_excel(data_path, engine="openpyxl")
    except Exception:
        df = pd.read_csv(data_path)
    return df

# =====================================================
#  XGBoost Loader and Trainer
# =====================================================
@st.cache_data
def load_xgb_data(data_glob="data/*.csv"):
    csv_files = glob.glob(data_glob)
    if not csv_files:
        st.warning("⚠️ No CSV files found in data folder.")
        return pd.DataFrame()
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df

def ensure_features(df):
    df = df.copy()
    if 'Fatalities' not in df.columns and 'Value' in df.columns:
        df.rename(columns={'Value': 'Fatalities'}, inplace=True)

    if 'admissions_count' not in df.columns:
        adm_cols = [c for c in df.columns if 'admission' in c.lower()]
        df['admissions_count'] = df[adm_cols].select_dtypes(include=[np.number]).sum(axis=1) if adm_cols else 0

    if 'prescriptions_total' not in df.columns:
        pres_cols = [c for c in df.columns if 'prescription' in c.lower()]
        df['prescriptions_total'] = df[pres_cols].select_dtypes(include=[np.number]).sum(axis=1) if pres_cols else 0

    if 'Tobacco Price Index' not in df.columns:
        price_cols = [c for c in df.columns if 'price' in c.lower()]
        df['Tobacco Price Index'] = df[price_cols[0]] if price_cols else np.nan

    if 'smok_prev' not in df.columns:
        smok_cols = [c for c in df.columns if 'smok' in c.lower()]
        df['smok_prev'] = df[smok_cols[0]] if smok_cols else np.nan

    if 'Sex' not in df.columns:
        df['Sex'] = 'Unknown'

    if 'ICD10 Diagnosis' not in df.columns:
        df['ICD10 Diagnosis'] = 'All cancers'

    for col in ['admissions_count', 'prescriptions_total', 'smok_prev', 'Tobacco Price Index', 'Fatalities']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Fatalities'], inplace=True)
    return df

@st.cache_resource
def train_xgb_model(df):
    num_features = ["admissions_count", "prescriptions_total", "smok_prev", "Tobacco Price Index"]
    cat_features = ["Sex", "ICD10 Diagnosis"]

    X = df[num_features + cat_features].copy()
    y = df["Fatalities"].astype(float)

    for c in num_features:
        X[c].fillna(X[c].median(), inplace=True)
    for c in cat_features:
        X[c].fillna("Unknown", inplace=True)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    calib = LinearRegression().fit(pipeline.predict(X_train).reshape(-1, 1), y_train)
    rmse = np.sqrt(np.mean((pipeline.predict(X_test) - y_test) ** 2))
    return pipeline, calib, num_features, cat_features, rmse

def calibrated_predict(model, calib, X):
    raw_pred = model.predict(X)
    return calib.predict(raw_pred.reshape(-1, 1))

# =====================================================
# 🔄 Load Both Models and Data
# =====================================================
st.sidebar.header("Model Status")

try:
    poisson_model = load_poisson_model()
    df_poisson = load_poisson_data()
    st.sidebar.success("Poisson GLM Loaded")
except Exception as e:
    st.sidebar.error(f"Poisson Model Error: {e}")
    df_poisson = pd.DataFrame()

df_xgb = load_xgb_data()
if not df_xgb.empty:
    df_xgb = ensure_features(df_xgb)
    with st.spinner("Training XGBoost model..."):
        pipeline, calib, num_features, cat_features, rmse = train_xgb_model(df_xgb)
    st.sidebar.success("XGBoost Loaded ")
else:
    st.sidebar.error("❌ No data for XGBoost model.")
    st.stop()

# =====================================================
# 🎛️ Input Controls
# =====================================================
st.sidebar.header("🔧 Enter Input Parameters")

if not df_poisson.empty and 'Year' in df_poisson:
    available_years = sorted(df_poisson['Year'].dropna().unique().tolist())
else:
    available_years = [2019, 2020, 2021, 2022]

# ICD10 diagnosis combined list from both datasets
# --- Safe conversion helper ---
def safe_tolist(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)

# --- Safely collect and convert all values to strings before sorting ---
sexes = sorted(set(map(str, safe_tolist(df_poisson.get('Sex', [])))) |
               set(map(str, safe_tolist(df_xgb.get('Sex', [])))))

diagnoses = sorted(set(map(str, safe_tolist(df_poisson.get('ICD10 Diagnosis', [])))) |
                   set(map(str, safe_tolist(df_xgb.get('ICD10 Diagnosis', [])))))


year = st.sidebar.selectbox("🗓️ Year", available_years)
sex = st.sidebar.selectbox("🚻 Sex", sexes)
diagnosis = st.sidebar.selectbox("🧬 ICD10 Diagnosis", diagnoses)

admissions = st.sidebar.number_input("🏥 Admissions Count", min_value=0, step=100, value=5000)
prescriptions = st.sidebar.number_input("💊 Total Prescriptions", min_value=0, step=100, value=3000)
smok_prev = st.sidebar.slider("🚬 Smoking Prevalence (%)", 0.0, 100.0, 25.0)
tob_price = st.sidebar.number_input("💰 Tobacco Price Index", min_value=0.0, step=1.0, value=120.0)

# =====================================================
# 🔮 Prediction Logic
# =====================================================
input_data = pd.DataFrame([{
    'Year': year,
    'Sex': sex,
    'ICD10 Diagnosis': diagnosis,
    'admissions_count': admissions,
    'prescriptions_total': prescriptions,
    'smok_prev': smok_prev,
    'Tobacco Price Index': tob_price
}])

st.markdown("### 📋 Input Summary")
st.dataframe(input_data)

high_confidence_diseases = [
    "All respiratory diseases",
    "All cancers",
    "All circulatory diseases"
]

if st.button("🚀 Predict Fatalities"):
    try:
        if diagnosis in high_confidence_diseases:
            # --- Use Poisson GLM ---
            pred = poisson_model.predict(input_data)[0]
            model_used = "Poisson GLM"
        else:
            # --- Use XGBoost ---
            pred = calibrated_predict(pipeline, calib, input_data)[0]
            model_used = "XGBoost Regression"

        st.success(f"### 🧩 Predicted Fatalities: **{pred:,.0f}** ({model_used})")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# =====================================================
# 📊 Visualization Section
# =====================================================

st.markdown("---")
st.subheader("📈 Model-Based Visualization")

if diagnosis in high_confidence_diseases:
    # -------------------------------------------------
    # 🔥 Poisson GLM Sensitivity Visualization (Heatmap)
    # -------------------------------------------------
    st.markdown("#### 🔍 Sensitivity of Poisson GLM to Smoking & Admissions")

    input_row = input_data.iloc[0]
    admissions_count = input_row["admissions_count"]
    prescriptions_total = input_row["prescriptions_total"]
    smok_prev = input_row["smok_prev"]
    tobacco_price_index = input_row["Tobacco Price Index"]

    # Create small ranges (±20%)
    admission_range = np.linspace(admissions_count * 0.8, admissions_count * 1.2, 10)
    smoking_range = np.linspace(smok_prev * 0.8, smok_prev * 1.2, 10)

    A, S = np.meshgrid(admission_range, smoking_range)
    preds = np.zeros_like(A)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            sample = pd.DataFrame({
                'admissions_count': [A[i, j]],
                'prescriptions_total': [prescriptions_total],
                'smok_prev': [S[i, j]],
                'Tobacco Price Index': [tobacco_price_index],
                'Sex': [sex],
                'ICD10 Diagnosis': [diagnosis]
            })
            preds[i, j] = poisson_model.predict(sample)[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(preds, origin='lower', cmap='viridis',
                   extent=[admission_range.min(), admission_range.max(),
                           smoking_range.min(), smoking_range.max()],
                   aspect='auto')
    plt.colorbar(im, ax=ax, label="Predicted Fatalities")
    ax.set_xlabel("Admissions Count")
    ax.set_ylabel("Smoking Prevalence (%)")
    ax.set_title(f"Impact of Smoking & Admissions on Predicted Fatalities ({diagnosis})")
    ax.scatter(admissions_count, smok_prev, color='red', s=100, edgecolor='black', label='User Input')
    ax.legend()
    st.pyplot(fig)

else:
    # -------------------------------------------------
    # ⚡ XGBoost Visualization — Feature Sensitivity
    # -------------------------------------------------
    st.markdown("#### ⚡ Impact of Smoking Prevalence on Predicted Fatalities (XGBoost)")

    input_row = input_data.iloc[0]
    smok_prev = input_row["smok_prev"]

    # Range ±30% of current smoking prevalence
    smoking_range = np.linspace(smok_prev * 0.7, smok_prev * 1.3, 20)

    predictions = []
    for s in smoking_range:
        temp = input_data.copy()
        temp["smok_prev"] = s
        pred = calibrated_predict(pipeline, calib, temp)[0]
        predictions.append(pred)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(smoking_range, predictions, marker='o', color='teal')
    ax.axvline(smok_prev, color='red', linestyle='--', label='User Input')
    ax.set_title("Effect of Smoking Prevalence on Predicted Fatalities (XGBoost)")
    ax.set_xlabel("Smoking Prevalence (%)")
    ax.set_ylabel("Predicted Fatalities")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)



# st.markdown("---")
# st.subheader("📈 Historical Trend (if available)")

# df_ref = df_poisson if diagnosis in high_confidence_diseases else df_xgb
# model_used = "Poisson GLM" if diagnosis in high_confidence_diseases else "XGBoost Regression"

# if not df_ref.empty and 'Year' in df_ref and 'Fatalities' in df_ref:
#     diag_data = df_ref[df_ref['ICD10 Diagnosis'] == diagnosis]

#     if not diag_data.empty:
#         diag_data = diag_data.copy()

#         # ✅ Convert year to numeric safely
#         diag_data['Year'] = pd.to_numeric(diag_data['Year'], errors='coerce')
#         diag_data = diag_data.dropna(subset=['Year'])
#         diag_data = diag_data.sort_values('Year')

#         fig, ax = plt.subplots(figsize=(8, 4))
#         ax.plot(diag_data['Year'], diag_data['Fatalities'], marker='o', label='Actual')

#         if 'Predicted' in diag_data.columns:
#             ax.plot(diag_data['Year'], diag_data['Predicted'], marker='x', linestyle='--', label='Predicted')

#         ax.set_xlabel("Year")
#         ax.set_ylabel("Fatalities")
#         ax.set_title(f"Fatalities Trend for {diagnosis} ({model_used})")
#         ax.legend()
#         ax.grid(True, linestyle='--', alpha=0.6)

#         # ✅ Rotate x-ticks for better readability
#         plt.xticks(rotation=45)
#         plt.tight_layout()

#         st.pyplot(fig)
#     else:
#         st.info("No data available for the selected diagnosis.")
# else:
#     st.warning("Year or Fatalities column not found in the dataset.")

