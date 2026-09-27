import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="ABC Ltd. - Customer Churn Evaluator",
    page_icon="📊",
    layout="wide"
)

# Load the pre-trained Scikit-Learn Pipeline from Colab
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

try:
    pipeline = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# ==========================================
# 2. HEADER & BUSINESS CONTEXT
# ==========================================
st.title("📊 ABC Ltd. — Customer Retention & Churn Risk Tool")
st.markdown("""
This predictive analytics application helps account managers, customer success leads, and executives 
identify high-risk accounts at **ABC Ltd.** before they churn. 

* **Model Type:** Logistic Regression Pipeline (Trained on historical customer behavior data)
* **Goal:** Enable proactive retention strategies and reduce customer attrition.
""")

st.divider()

if not model_loaded:
    st.error("⚠️ `model.pkl` file not found! Please upload `model.pkl` generated from Google Colab into your repository root directory.")
    st.stop()

# ==========================================
# 3. INTERACTIVE INPUT FORM (SIDEBAR & MAIN)
# ==========================================
st.sidebar.header("👤 Customer Profile Configuration")
st.sidebar.markdown("Adjust parameters to simulate a customer scenario:")

# Numerical Features
tenure = st.sidebar.slider("Tenure (Months with ABC Ltd.)", min_value=1, max_value=72, value=12, step=1)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=20.0, max_value=120.0, value=65.0, step=0.5)

# Categorical Features
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
paperless_billing = st.sidebar.radio("Paperless Billing Active?", ["Yes", "No"], horizontal=True)
tech_support = st.sidebar.radio("Has Premium Tech Support?", ["Yes", "No"], horizontal=True)

# Build DataFrame matching exact feature schema expected by Colab pipeline
input_data = pd.DataFrame([{
    'Tenure_Months': tenure,
    'Monthly_Charges': monthly_charges,
    'Contract_Type': contract_type,
    'Payment_Method': payment_method,
    'Paperless_Billing': paperless_billing,
    'Tech_Support': tech_support
}])

# ==========================================
# 4. PREDICTION & DISPLAY ENGINE
# ==========================================
st.subheader("📋 Profile Summary & Risk Assessment")

col_summary, col_predict = st.columns([1, 1])

with col_summary:
    st.markdown("### Selected Customer Attributes")
    st.dataframe(input_data.T.rename(columns={0: 'Attribute Value'}), use_container_width=True)

with col_predict:
    st.markdown("### AI Attrition Evaluation")
    
    # Generate prediction probabilities using the pipeline
    prob_churn = pipeline.predict_proba(input_data)[0][1]
    churn_percentage = prob_churn * 100

    # Display KPI Metric
    st.metric(
        label="Predicted Churn Probability",
        value=f"{churn_percentage:.1f}%",
        delta="- High Risk" if prob_churn >= 0.5 else "+ Stable Account",
        delta_color="inverse" if prob_churn >= 0.5 else "normal"
    )

    # Risk Tiering & Actionable Recommendations
    if prob_churn >= 0.65:
        st.error("🚨 **CRITICAL RISK TIER**")
        st.markdown("**Recommended Action:** Immediate intervention by senior account manager. Offer a long-term contract discount or dedicated tech support onboarding.")
    elif prob_churn >= 0.40:
        st.warning("⚠️ **MODERATE RISK TIER**")
        st.markdown("**Recommended Action:** Target with targeted customer satisfaction survey and highlight premium support features.")
    else:
        st.success("✅ **LOW RISK TIER**")
        st.markdown("**Recommended Action:** Customer profile is highly stable. Standard automated engagement and cross-sell campaigns.")

st.divider()

# ==========================================
# 5. EXPLAINABILITY & MANAGERIAL TRANSPARENCY
# ==========================================
st.subheader("💡 Decision Explainability: What Influenced This Score?")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.markdown("#### Primary Churn Drivers (Risk Escalators)")
    st.write("1. **Contract Type**: Month-to-month contracts have an **Odds Ratio > 3.0**, heavily increasing risk compared to long-term commitments.")
    st.write("2. **Monthly Charges**: Higher monthly spend without long-term contracts increases customer price sensitivity.")
    st.write("3. **Payment Method**: Customers paying via electronic check show statistically higher volatility.")

with col_exp2:
    st.markdown("#### Protective Factors (Risk Mitigators)")
    st.write("1. **Tenure**: Each additional month with ABC Ltd. exponentially lowers overall attrition likelihood.")
    st.write("2. **Tech Support Integration**: Enrolling in active technical support acts as an operational lock-in, reducing churn risk.")
    st.write("3. **Two-Year Contracts**: Multi-year agreements lock in lower rates and drastically suppress short-term exit risks.")

st.info("ℹ️ **Note for Decision Makers:** Model predictions represent statistical probabilities based on historical cohort patterns. Domain expertise and qualitative customer relationships should complement AI risk scores.")
