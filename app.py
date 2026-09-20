import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# STYLES AND CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* App Background */
.stApp {
    background-color: #F4F7F9;
}

/* Center and bound content container */
.block-container {
    max-width: 1100px !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

/* HERO SECTION CARD FIX - Visible Container */
.hero-card {
    background: linear-gradient(135deg, #E0F2FE 0%, #E6F4F1 100%);
    border: 1px solid #BEE3F8;
    border-radius: 20px;
    padding: 35px 40px;
    margin-bottom: 30px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
}

.badge {
    display: inline-block;
    background-color: #0284C7;
    color: #FFFFFF;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
    margin-bottom: 12px;
    letter-spacing: -0.5px;
}

.hero-text {
    color: #334155;
    font-size: 15px;
    line-height: 1.6;
    max-width: 550px;
}

/* Section Headings */
.section-title {
    color: #0F172A;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 6px;
}

.section-subtitle {
    color: #64748B;
    font-size: 13px;
    margin-bottom: 20px;
}

/* Input Form Controls Styling */
label {
    color: #1E293B !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: #CBD5E1 !important;
    background-color: #FFFFFF !important;
}

input {
    border-radius: 10px !important;
    border-color: #CBD5E1 !important;
    background-color: #FFFFFF !important;
}

/* Primary Action Button */
.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25);
    margin-top: 10px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
    color: #FFFFFF;
}

/* Output Card Styling */
.result-card {
    background: linear-gradient(145deg, #0F172A 0%, #1E293B 100%);
    border-radius: 18px;
    padding: 30px;
    min-height: 295px;
    color: #FFFFFF;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.15);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.result-heading {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 6px;
}

.result-subtitle {
    font-size: 13px;
    opacity: 0.8;
}

.result-price {
    font-size: 44px;
    font-weight: 800;
    color: #38BDF8;
    margin-top: 20px;
    letter-spacing: -1px;
}

/* Performance Metric Cards */
.info-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}

.info-title {
    color: #64748B;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.info-value {
    color: #0F172A;
    font-size: 22px;
    font-weight: 800;
    margin-top: 6px;
}

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 13px;
    margin-top: 25px;
}

hr {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 30px 0;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATASET LOADING
# =========================================================

@st.cache_data
def load_dataset():
    possible_files = [
        "insurance.csv",
        "./insurance.csv",
        "data/insurance.csv",
        "./data/insurance.csv"
    ]
    for file in possible_files:
        if os.path.exists(file):
            return pd.read_csv(file)
    return None

df = load_dataset()

if df is None:
    st.error("⚠️ `insurance.csv` repository mein nahi mila. Direct root folder mein push karein.")
    st.stop()

# Clean column headers
df.columns = df.columns.str.strip().str.lower()
if "expenses" in df.columns and "charges" not in df.columns:
    df = df.rename(columns={"expenses": "charges"})

required_columns = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error("Dataset columns missing: " + ", ".join(missing))
    st.stop()

# =========================================================
# MODEL TRAINING
# =========================================================

X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]

numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
r2 = r2_score(y_test, test_predictions)
mae = mean_absolute_error(y_test, test_predictions)

# =========================================================
# HERO SECTION (WITH VISIBLE BOX & HIGH QUALITY VECTOR SVG)
# =========================================================

st.markdown("""
<div class="hero-card">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;">
        <div style="flex: 1; min-width: 300px;">
            <div class="badge">✦ ML Powered Healthcare</div>
            <div class="hero-title">Medical Insurance Cost Predictor</div>
            <div class="hero-text">Estimate your annual medical insurance cost using a supervised machine learning model trained on historical healthcare demographic data.</div>
        </div>
        <div style="flex: 0.8; min-width: 260px; text-align: center;">
            <svg width="280" height="170" viewBox="0 0 300 180" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="300" height="180" rx="16" fill="#FFFFFF" fill-opacity="0.6"/>
                <path d="M70 90C70 65 90 45 115 45C140 45 160 65 160 90C160 120 115 145 115 145C115 145 70 120 70 90Z" fill="#0284C7" fill-opacity="0.15"/>
                <path d="M115 65V115M90 90H140" stroke="#0284C7" stroke-width="8" stroke-linecap="round"/>
                <circle cx="210" cy="80" r="35" fill="#38BDF8" fill-opacity="0.2"/>
                <path d="M190 125C190 105 200 95 210 95C220 95 230 105 230 125" stroke="#0F172A" stroke-width="5" stroke-linecap="round"/>
                <circle cx="210" cy="70" r="12" fill="#0F172A"/>
                <path d="M40 135L60 120L75 130L95 110" stroke="#0284C7" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MAIN SECTION (INPUTS & RESULT)
# =========================================================

left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Please enter your details below to estimate the insurance cost.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=100, value=30)
    with c2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    c1, c2 = st.columns(2)
    with c1:
        sex = st.selectbox("Gender", ["Female", "Male"])
    with c2:
        children = st.number_input("Children", min_value=0, max_value=10, value=0)

    c1, c2 = st.columns(2)
    with c1:
        smoker = st.selectbox("Smoking Status", ["No", "Yes"])
    with c2:
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

    st.write("")
    predict = st.button("Predict Insurance Cost")

with right_col:
    st.markdown('<div class="section-title">💰 Prediction Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Your estimated annual medical insurance cost.</div>', unsafe_allow_html=True)

    if predict:
        user_data = pd.DataFrame({
            "age": [age],
            "sex": [sex.lower()],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker.lower()],
            "region": [region.lower()]
        })
        prediction = model.predict(user_data)[0]

        st.markdown(
            f"""<div class="result-card">
<div>
    <div class="result-heading">Estimated Charges</div>
    <div class="result-subtitle">Calculated annual medical insurance estimate</div>
</div>
<div class="result-price">${prediction:,.2f}</div>
<div style="font-size: 12px; opacity: 0.7;">Based on Linear Regression Model</div>
</div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div class="result-card">
<div>
    <div class="result-heading">Prediction Result</div>
    <div class="result-subtitle">Your estimated annual medical insurance cost</div>
</div>
<div style="margin-top:25px; opacity:0.85; line-height:1.6; font-size: 15px;">
Enter details on the left and click "Predict Insurance Cost" to view the calculated estimate.
</div>
<div style="font-size: 12px; opacity: 0.5;">Awaiting user input...</div>
</div>""",
            unsafe_allow_html=True
        )

# =========================================================
# MODEL PERFORMANCE METRICS
# =========================================================

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Model Metrics & Performance</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Evaluation metrics of the trained regression model on the test dataset.</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""<div class="info-card"><div class="info-title">Algorithm</div><div class="info-value">Linear Regression</div></div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""<div class="info-card"><div class="info-title">R² Score</div><div class="info-value">{r2:.3f}</div></div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""<div class="info-card"><div class="info-title">MAE</div><div class="info-value">${mae:,.0f}</div></div>""", unsafe_allow_html=True)

with m4:
    st.markdown(f"""<div class="info-card"><div class="info-title">Dataset Size</div><div class="info-value">{len(df):,} rows</div></div>""", unsafe_allow_html=True)

# FOOTER
st.markdown("<hr>", unsafe_allow_html=True)
st.warning("⚠️ **Disclaimer:** This application is an educational machine learning project. The prediction is an estimate and should not be treated as an actual insurance quote.")
st.markdown('<div class="footer">Medical Insurance Cost Prediction · Machine Learning Portfolio Project</div>', unsafe_allow_html=True)
