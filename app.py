"""
STEP 8 & 9: Streamlit Deployment App
Run with: streamlit run app.py

This turns your model into an interactive web app that anyone
(including a recruiter) can use without writing any code.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# ── Load model and encoders (cached so it only loads once) ──
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    le_dict = joblib.load('label_encoders.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, le_dict, feature_columns

model, le_dict, feature_columns = load_artifacts()

# ── Header ────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("**Predict customer churn risk and understand *why* using SHAP explainability**")
st.divider()

# ── Sidebar Inputs ────────────────────────────────────────────
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, monthly_charges * tenure)

predict_btn = st.sidebar.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

# ── Build input row matching training feature engineering ──
def build_input_row():
    row = {
        'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
    }
    df_row = pd.DataFrame([row])

    # Same engineered features as training
    df_row['AvgMonthlySpend'] = df_row['TotalCharges'] / (df_row['tenure'] + 1)
    df_row['TenureGroup'] = pd.cut(df_row['tenure'], bins=[-1, 12, 24, 48, 72],
                                     labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])
    df_row['HasMultipleServices'] = (
        (df_row['OnlineSecurity'] == 'Yes').astype(int) +
        (df_row['OnlineBackup'] == 'Yes').astype(int) +
        (df_row['DeviceProtection'] == 'Yes').astype(int) +
        (df_row['TechSupport'] == 'Yes').astype(int)
    )

    # Encode using saved label encoders
    for col, le in le_dict.items():
        df_row[col] = le.transform(df_row[col].astype(str))

    return df_row[feature_columns]

# ── Main panel: Prediction + SHAP explanation ───────────────
col1, col2 = st.columns([1, 2])

if predict_btn:
    X_input = build_input_row()
    proba = model.predict_proba(X_input)[0][1]
    prediction = "Will Churn ⚠️" if proba > 0.5 else "Will Stay ✅"

    with col1:
        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{proba:.1%}")
        if proba > 0.5:
            st.error(f"**{prediction}** — High risk customer")
        else:
            st.success(f"**{prediction}** — Low risk customer")

        st.progress(float(proba))

        if proba > 0.7:
            st.warning("🎯 **Recommended Action:** Offer retention discount or contract upgrade incentive immediately.")
        elif proba > 0.4:
            st.info("🎯 **Recommended Action:** Monitor closely, consider proactive engagement.")
        else:
            st.info("🎯 **Recommended Action:** No immediate action needed.")

    with col2:
        st.subheader("Why this prediction? (SHAP Explanation)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.force_plot(
            explainer.expected_value, shap_values[0], X_input.iloc[0],
            matplotlib=True, show=False
        )
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Top contributing factors as text (great for non-technical stakeholders)
        shap_df = pd.DataFrame({
            'Feature': X_input.columns,
            'Impact': shap_values[0]
        }).sort_values('Impact', key=abs, ascending=False).head(5)

        st.markdown("**Top factors influencing this prediction:**")
        for _, row in shap_df.iterrows():
            direction = "🔴 increases" if row['Impact'] > 0 else "🟢 decreases"
            st.write(f"- **{row['Feature']}** {direction} churn risk")
else:
    st.info("👈 Fill in customer details in the sidebar and click **Predict Churn Risk** to get started.")

    st.subheader("About This Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC Score", "83.4%")
    c2.metric("Accuracy", "76.9%")
    c3.metric("Recall", "62.0%")
    c4.metric("Training Data", "7,043 customers")

    st.markdown("""
    **Model Pipeline:**
    1. Data cleaning (handled missing `TotalCharges` values)
    2. Feature engineering (AvgMonthlySpend, TenureGroup, HasMultipleServices)
    3. SMOTE oversampling to fix class imbalance (73% / 27% → 50% / 50%)
    4. XGBoost classifier trained on balanced data
    5. SHAP explainability layer for transparent predictions
    """)
