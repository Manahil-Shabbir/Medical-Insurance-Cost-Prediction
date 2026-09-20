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

/* App layout fix */
.stApp {
    background-color: #F7FAFC;
}

.block-container {
    max-width: 1280px !important;
    padding-top: 20px !important;
    padding-bottom: 40px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Hero Section */
.hero-card {
    background: linear-gradient(135deg, #EBF8F6 0%, #F3F9FE 100%);
    border: 1px solid #D2E9E6;
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(8, 127, 115, 0.04);
}

.badge {
    display: inline-block;
    background-color: #D1F2EC;
    color: #087F73;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #0F324D;
    line-height: 1.15;
    margin-bottom: 12px;
    letter-spacing: -0.5px;
}

.hero-text {
    color: #5A7184;
    font-size: 15px;
    line-height: 1.6;
    max-width: 580px;
}

/* Headings */
.section-title {
    color: #0F324D;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Input Form Styling */
label {
    color: #1A3850 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: #D6E0E6 !important;
    background-color: #FFFFFF !important;
}

input {
    border-radius: 10px !important;
    border-color: #D6E0E6 !important;
    background-color: #FFFFFF !important;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #087F73 0%, #066C63 100%);
    color: white;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(8, 127, 115, 0.22);
    margin-top: 10px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #066C63 0%, #04524B 100%);
    color: white;
}

/* Prediction Card */
.result-card {
    background: linear-gradient(145deg, #0B3A53 0%, #0F4C64 100%);
    border-radius: 20px;
    padding: 32px;
    min-height: 290px;
    color: white;
    box-shadow: 0 10px 25px rgba(15, 76, 100, 0.15);
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
    font-size: 14px;
    opacity: 0.8;
}

.result-price {
    font-size: 46px;
    font-weight: 800;
    color: #4DE1C1;
    margin-top: 20px;
    letter-spacing: -1px;
}

/* Metric Cards */
.info-card {
    background: #FFFFFF;
    border: 1px solid #E1E8ED;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.02);
}

.info-title {
    color: #7A8B98;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.info-value {
    color: #0F324D;
    font-size: 22px;
    font-weight: 800;
    margin-top: 6px;
}

.footer {
    text-align: center;
    color: #94A2AC;
    font-size: 13px;
    margin-top: 30px;
}

hr {
    border: none;
    border-top: 1px solid #E1E8ED;
    margin: 32px 0;
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
    st.error("⚠️ insurance.csv file nahi mili. Kripya isko app folder mein rakhein.")
    st.stop()

# Clean columns
df.columns = df.columns.str.strip().str.lower()
if "expenses" in df.columns and "charges" not in df.columns:
    df = df.rename(columns={"expenses": "charges"})

required_columns = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error("Dataset columns missing hain: " + ", ".join(missing))
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
# HERO SECTION
# =========================================================

hero1, hero2 = st.columns([1.4, 1], gap="large")

with hero1:
    st.markdown('<div class="badge">✦ ML Powered Healthcare</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Medical Insurance Cost<br>Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-text">Estimate your medical insurance cost using a machine learning model trained on historical healthcare data. Get a quick estimate based on your personal information.</div>', unsafe_allow_html=True)

with hero2:
    # Embedded Vector Art (Direct SVG)
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
        <svg width="340" height="190" viewBox="0 0 340 190" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="340" height="190" rx="18" fill="#E2F4F2"/>
            <path d="M120 40C80 40 50 70 50 110C50 150 120 170 120 170C120 170 190 150 190 110C190 70 160 40 120 40Z" fill="#2DD4BF" opacity="0.25"/>
            <circle cx="230" cy="95" r="50" fill="#087F73" opacity="0.15"/>
            <rect x="105" y="80" width="30" height="50" rx="6" fill="#087F73"/>
            <rect x="95" y="90" width="50" height="30" rx="6" fill="#087F73"/>
            <path d="M200 120C200 100 220 85 240 85C260 85 280 100 280 120" stroke="#0F4C64" stroke-width="6" stroke-linecap="round"/>
            <circle cx="240" cy="65" r="14" fill="#0F4C64"/>
            <path d="M40 70Q60 50 80 70T120 70" stroke="#2DD4BF" stroke-width="4" stroke-linecap="round" fill="none"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# MAIN CONTENT (FORM & RESULT)
# =========================================================

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    st.caption("Please enter your details below to estimate the insurance cost.")

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

with right:
    st.markdown('<div class="section-title">💰 Prediction Result</div>', unsafe_allow_html=True)
    st.caption("Your estimated annual medical insurance cost.")

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
Enter the details on the left and click "Predict Insurance Cost" to generate the estimate.
</div>
<div style="font-size: 12px; opacity: 0.5;">Awaiting user input...</div>
</div>""",
            unsafe_allow_html=True
        )

# =========================================================
# MODEL METRICS
# =========================================================

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Model Metrics & Performance</div>', unsafe_allow_html=True)
st.caption("Evaluation metrics of the trained regression model on the test dataset.")

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
