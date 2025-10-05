🪐 Exoplanet Discovery using Kepler Data (Machine Learning + Streamlit)
🔍 Overview
This project uses NASA Kepler exoplanet candidate data to train a machine learning model (XGBoost) that predicts whether a detected signal represents a real exoplanet or a false positive.
The model is trained using tabular data from the Kepler Objects of Interest (KOI) catalog and includes a Streamlit web interface that allows users to input key features (like orbital period, transit depth, etc.) and get an estimated exoplanet probability score.
🚀 Features
Machine Learning classification using XGBoost
Interactive Streamlit frontend for real-time predictions
SHAP explainability for feature impact visualization
Support for scientific data preprocessing and scaling
High interpretability for astrophysical analysis
📊 Input Features
Feature	Description
koi_period	Orbital period (days)
koi_duration	Transit duration (hours)
koi_depth	Transit depth (ppm)
koi_prad	Planetary radius (Earth radii)
koi_teq	Equilibrium temperature (K)
koi_insol	Insolation flux (Earth flux)
koi_model_snr	Transit signal-to-noise ratio
koi_steff	Stellar effective temperature (K)
koi_slogg	Stellar surface gravity
koi_srad	Stellar radius (solar radii)
koi_kepmag	Kepler-band magnitude
koi_fpflag_nt	Not-transit-like false positive flag
koi_fpflag_ss	Stellar eclipse false positive flag
koi_fpflag_co	Centroid offset false positive flag
koi_fpflag_ec	Ephemeris match contamination flag
🧠 Model
Algorithm: XGBoost Classifier
Goal: Binary classification → Planet (1) vs Not Planet (0)
Metrics: ROC-AUC, PR-AUC, Precision, Recall
Explainability: SHAP values to understand model reasoning
