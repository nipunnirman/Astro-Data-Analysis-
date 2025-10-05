import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from streamlit_lottie import st_lottie
import json

# ================================================================
# 1. Page Configuration
# ================================================================
st.set_page_config(
    page_title="🪐 Kepler Exoplanet Classifier Dashboard",
    layout="wide",
    page_icon="🌠",
)

# Custom styling (your original perfect colors)
st.markdown("""
    <style>
    body { background-color: #0E1117; color: #FAFAFA; }
    .stApp { background-color: #0E1117; }
    h1, h2, h3, h4 { color: #6CCAFF; }
    .css-18e3th9 { padding: 2rem 1rem 2rem 1rem; }
    .stButton>button {
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        color: white;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        padding: 0.6em 1.2em;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

# ================================================================
# 2. Load Lottie Animation
# ================================================================
def load_lottie(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception:
        return None

planet_anim = load_lottie("planet.json")  
# ================================================================
# 3. Load Model and Preprocessing
# ================================================================
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    return model, scaler, imputer

features = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
    'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag',
    'koi_fpflag_nt','koi_fpflag_ss','koi_fpflag_co','koi_fpflag_ec'
]

try:
    model, scaler, imputer = load_model()
except Exception as e:
    st.error("⚠️ Could not load model files. Please ensure model, scaler, and imputer exist.")
    st.stop()

# ================================================================
# 4. Header
# ================================================================
st.title("🌌 Kepler Exoplanet Discovery Dashboard")
st.markdown("A modern machine learning interface for analyzing and predicting exoplanet candidates using Kepler telescope data.")

tab1, tab2 = st.tabs(["🔍 Search by Planet Name", "🧩 Manual Input Mode"])

# ================================================================
# 5. Tab 1 — Planet Name Search
# ================================================================
with tab1:
    st.subheader("🔎 Search for a Planet by Kepler Name")

    uploaded_file = st.file_uploader("📂 Upload Kepler Dataset (CSV with 'kepler_name')", type=["csv"])
    planet_name = st.text_input("Enter Kepler planet name (e.g., Kepler-22 b):")

    if st.button("Search Planet"):
        if uploaded_file is None:
            st.warning("⚠️ Please upload a dataset first.")
        else:
            df = pd.read_csv(uploaded_file, comment="#")
            matches = df[df['kepler_name'].astype(str).str.strip().str.lower() == planet_name.strip().lower()]

            if matches.empty:
                st.error(f"❌ No planet found named '{planet_name}'")
            else:
                st.success(f"✅ Found planet: {matches.iloc[0]['kepler_name']}")
                planet_features = matches[features]
                X_imp = imputer.transform(planet_features)
                X_scaled = scaler.transform(X_imp)
                prob = model.predict_proba(X_scaled)[0, 1]
                pred = "🪐 Likely Exoplanet" if prob > 0.5 else "❌ Not an Exoplanet Candidate"

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Prediction", pred)
                    st.metric("Probability", f"{prob:.3f}")

                    # Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={'text': "Exoplanet Likelihood (%)"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "#00c6ff"},
                               'steps': [
                                   {'range': [0, 50], 'color': "#332f2f"},
                                   {'range': [50, 100], 'color': "#0f3c5c"}]}
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    # 🌠 Show animation when exoplanet is detected
                    if prob > 0.5 and planet_anim:
                        st_lottie(planet_anim, height=250, key="anim1")

                with col2:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_scaled)
                    shap.initjs()
                    st.subheader("🧠 SHAP Force Plot")
                    force_plot_html = shap.force_plot(
                        explainer.expected_value,
                        shap_values[0, :],
                        pd.DataFrame(X_scaled, columns=features).iloc[0, :],
                        matplotlib=False
                    )
                    st.components.v1.html(shap.getjs() + force_plot_html.html(), height=400)

                # Top 5 positive/negative features
                shap_df = pd.DataFrame({
                    "Feature": features,
                    "SHAP Value": shap_values[0, :],
                    "Feature Value": planet_features.iloc[0].values
                })
                shap_df["|SHAP|"] = shap_df["SHAP Value"].abs()
                top_pos = shap_df.sort_values("SHAP Value", ascending=False).head(5)
                top_neg = shap_df.sort_values("SHAP Value", ascending=True).head(5)

                c1, c2 = st.columns(2)
                with c1:
                    st.write("📈 **Top 5 Positive Features (Support Exoplanet)**")
                    st.dataframe(top_pos.style.background_gradient(cmap="Greens"))
                with c2:
                    st.write("📉 **Top 5 Negative Features (Oppose Exoplanet)**")
                    st.dataframe(top_neg.style.background_gradient(cmap="Reds"))

# ================================================================
# 6. Tab 2 — Manual Input Mode
# ================================================================
with tab2:
    st.subheader("🧩 Manual Feature Input")

    default_values = {
        'koi_period': 10.0, 'koi_duration': 2.0, 'koi_depth': 500.0,
        'koi_prad': 1.0, 'koi_teq': 300.0, 'koi_insol': 100.0,
        'koi_model_snr': 10.0, 'koi_steff': 5500.0, 'koi_slogg': 4.5,
        'koi_srad': 1.0, 'koi_kepmag': 14.0, 'koi_fpflag_nt': 0,
        'koi_fpflag_ss': 0, 'koi_fpflag_co': 0, 'koi_fpflag_ec': 0
    }

    user_input = {}
    cols = st.columns(3)
    for i, f in enumerate(features):
        col = cols[i % 3]
        if isinstance(default_values[f], int):
            user_input[f] = col.number_input(f, value=int(default_values[f]), step=1)
        else:
            user_input[f] = col.number_input(f, value=float(default_values[f]), step=0.1)

    if st.button("🚀 Predict from Manual Input"):
        input_df = pd.DataFrame([user_input])
        X_imp = imputer.transform(input_df)
        X_scaled = scaler.transform(X_imp)
        prob = model.predict_proba(X_scaled)[0, 1]
        pred = "🪐 Likely Exoplanet" if prob > 0.5 else "❌ Not an Exoplanet Candidate"

        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Prediction", pred)
            st.metric("Probability", f"{prob:.3f}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Exoplanet Likelihood (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#00c6ff"},
                       'steps': [
                           {'range': [0, 50], 'color': "#332f2f"},
                           {'range': [50, 100], 'color': "#0f3c5c"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)

            # 🌠 Animation when exoplanet detected
            if prob > 0.5 and planet_anim:
                st_lottie(planet_anim, height=250, key="anim2")

        with c2:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            shap.initjs()
            st.subheader("🧠 SHAP Force Plot")
            force_plot_html = shap.force_plot(
                explainer.expected_value,
                shap_values[0, :],
                pd.DataFrame(X_scaled, columns=features).iloc[0, :],
                matplotlib=False
            )
            st.components.v1.html(shap.getjs() + force_plot_html.html(), height=400)
