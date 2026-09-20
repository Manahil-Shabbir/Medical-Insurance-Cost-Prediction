import streamlit as st
import pandas as pd
import os
import base64
from PIL import Image
from sklearn.model_selection import train_train_split
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
# HELPER FUNCTION FOR IMAGE ENCODING
# =========================================================

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    return None

img_base64 = get_base64_image("hero.png")

# =========================================================
# STYLES AND CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Hide Streamlit Default Header and Decoration Bar */
header[data-testid="stHeader"] {
    display: none !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Main App Background */
.stApp {
    background-color: #F8FAFC;
}

/* Container Spacing & Top Padding Fix */
.block-container {
    max-width: 1080px !important;
    padding-top: 4rem !important;
    padding-bottom: 3rem !important;
}

/* Integrated Hero Banner Card */
.hero-card-container {
    background: linear-gradient(135deg, #E0F2FE 0%, #F0FDFA 100%);
    border: 1px solid #BAE6FD;
    border-radius: 20px;
    padding: 32px 36px;
    margin-top: 10px;
    margin-bottom: 30px;
    box-shadow: 0 4px 20px rgba(2, 132, 199, 0.05);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
}

.hero-text-content {
    flex: 1.2;
}

.hero-image-content {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}

.hero-image-content img {
    max-width: 100%;
    max-height: 180px;
    object-fit: contain;
    border-radius: 12px;
}

.badge {
    display: inline-block;
    background-color: #0284C7;
    color: #FFFFFF;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 12px;
    letter-spacing: 0.3px;
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
    margin-bottom: 10px;
    letter-spacing: -0.5px;
}

.hero-text {
    color: #334155;
    font-size: 14px;
    line-height: 1.6;
}

/* Headings */
.section-title {
    color: #0F172A;
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #64748B;
    font-size: 13px;
    margin-bottom: 18px;
}

/* Input Fields Styling */
label {
    color: #1E293B !important;
    font-weight: 600 !important;
    font-size: 13px !important;
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

/* Action Button */
.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
    color: #FFFFFF;
    font-size: 15px;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25);
    margin-top: 8px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
    color: #FFFFFF;
}

/* Result Card */
.result-card {
    background: linear-gradient(145deg, #0F172A 0%, #1E293B 100%);
    border-radius: 18px;
    padding: 30px;
    min-height: 290px;
    color: #FFFFFF;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.result-heading {
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 4px;
}

.result-subtitle {
    font-size: 13px;
    opacity: 0.8;
}

.result-price {
    font-size: 42px;
    font-weight: 800;
    color: #38BDF8;
    margin-top: 15px;
    letter-spacing: -1px;
}

/* Metric Cards */
.info-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
}

.info-title {
    color: #64748B;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.info-value {
    color: #0F172A;
    font-size: 20px;
    font-weight: 800;
    margin-top: 4px;
}

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 13px;
    margin-top: 20px;
}

hr {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 28px 0;
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
# INTEGRATED HERO SECTION
# =========================================================

image_html = f'<img src="data:image/png;base64,{img_base64}" alt="Healthcare Illustration"/>' if img_base64 else ''

st.markdown(f"""
<div class="hero-card-container">
    <div class="hero-text-content">
        <div class="badge">✦ ML Powered Healthcare</div>
        <div class="hero-title">Medical Insurance Cost Predictor</div>
        <div class="hero-text">Estimate your annual medical insurance cost using a supervised machine learning model trained on historical healthcare demographic data.</div>
    </div>
    <div class="hero-image-content">
        {image_html}
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
<div style="margin-top:25px; opacity:0.85; line-height:1.6; font-size: 14px;">
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
