import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load('models/xgboost_best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer will churn and understand **why** using SHAP explainability.")

st.sidebar.header("Customer Input")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 10, 120, 50)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])

# Build input vector matching your feature set
# (map inputs to encoded values matching your training features)

st.info("⚠️ Build input vector to match processed feature columns, then call model.predict_proba()")

# Example prediction block (wire up after feature engineering)
# input_df = build_input_vector(...)
# prob = model.predict_proba(scaler.transform(input_df))[0][1]
# st.metric("Churn Probability", f"{prob:.1%}")

st.markdown("---")
st.markdown("Built by **Jagmohan Prajapat** | [GitHub](https://github.com/Jagmohan-Prajapati)")
