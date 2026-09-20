import streamlit as st
import pandas as pd
import os
import base64
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
    layout="wide"
)

# Helper function to convert local image to Base64
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

img_base64 = get_base64_of_bin_file("hero_banner.png")

# =========================================================
# CSS STYLING
# =========================================================

st.markdown(f"""
<style>

.stApp {{
    background: #F4F8FA;
}}

.block-container {{
    max-width: 1150px;
    padding-top: 30px;
    padding-bottom: 40px;
}}

/* HERO SECTION */

.hero-container {{
    background: linear-gradient(135deg, #E6F7F5 0%, #F0F9FF 100%);
    border: 1px solid #D2E9E6;
    border-radius: 24px;
    padding: 40px 45px;
    margin-bottom: 35px;
    box-shadow: 0 10px 30px rgba(8, 127, 115, 0.05);
}}

.badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #D1F2EC;
    color: #087F73;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 16px;
}}

.hero-title {{
    font-size: 42px;
    font-weight: 800;
    color: #0F324D;
    line-height: 1.15;
    margin-bottom: 14px;
    letter-spacing: -0.5px;
}}

.hero-text {{
    color: #5A7184;
    font-size: 15px;
    line-height: 1.7;
    max-width: 520px;
}}

/* HERO ILLUSTRATION */

.medical-art {{
    height: 230px;
    width: 100%;
    border-radius: 20px;
    {"background-image: url('data:image/png;base64," + img_base64 + "');" if img_base64 else ""}
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    display: flex;
    align-items: center;
    justify-content: center;
}}

/* HEADINGS */

.section-title {{
    color: #0F324D;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.section-subtitle {{
    color: #7A8B98;
    font-size: 14px;
    margin-bottom: 20px;
}}

/* INPUT CARD */

.card {{
    background: #FFFFFF;
    border: 1px solid #E1E8ED;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 24px rgba(15, 50, 77, 0.04);
}}

.card-subtitle {{
    color: #7A8B98;
    font-size: 13px;
    margin-bottom: 22px;
}}

/* INPUT FORM ELEMENTS */

label {{
    color: #1A3850 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}}

div[data-baseweb="select"] > div {{
    border-radius: 12px !important;
    border-color: #D6E0E6 !important;
}}

input {{
    border-radius: 12px !important;
    border-color: #D6E0E6 !important;
}}

/* BUTTON STYLING */

.stButton > button {{
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #087F73 0%, #066C63 100%);
    color: white;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(8, 127, 115, 0.25);
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(8, 127, 115, 0.35);
    color: white;
}}

/* PREDICTION CARD */

.result-card {{
    background: linear-gradient(145deg, #0B3A53 0%, #0F4C64 100%);
    border-radius: 20px;
    padding: 35px;
    min-height: 310px;
    color: white;
    box-shadow: 0 12px 28px rgba(15, 76, 100, 0.2);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}}

.result-heading {{
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 6px;
}}

.result-subtitle {{
    font-size: 14px;
    opacity: 0.8;
}}

.result-price {{
    font-size: 46px;
    font-weight: 800;
    color: #4DE1C1;
    margin-top: 30px;
    letter-spacing: -1px;
}}

/* METRIC / INFO CARDS */

.info-card {{
    background: #FFFFFF;
    border: 1px solid #E1E8ED;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.02);
}}

.info-title {{
    color: #7A8B98;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.info-value {{
    color: #0F324D;
    font-size: 22px;
    font-weight: 800;
    margin-top: 6px;
}}

.footer {{
    text-align: center;
    color: #94A2AC;
    font-size: 13px;
    margin-top: 40px;
}}

hr {{
    border: none;
    border-top: 1px solid #E1E8ED;
    margin: 35px 0;
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATASET
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
    st.error("⚠️ insurance.csv was not found in your repository.")
    st.info("Please upload insurance.csv to the same folder as app.py.")
    st.stop()

# Clean columns
df.columns = df.columns.str.strip().str.lower()
if "expenses" in df.columns and "charges" not in df.columns:
    df = df.rename(columns={"expenses": "charges"})

required_columns = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error("Dataset columns are missing: " + ", ".join(missing))
    st.stop()

# =========================================================
# TRAIN MODEL
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

hero1, hero2 = st.columns([1.35, 1], gap="large")

with hero1:
    st.markdown('<div class="badge">✦ ML Powered Healthcare</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Medical Insurance Cost<br>Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-text">Estimate your medical insurance cost using a machine learning model trained on historical healthcare data. Get a quick estimate based on your personal information.</div>', unsafe_allow_html=True)

with hero2:
    if img_base64:
        st.markdown('<div class="medical-art"></div>', unsafe_allow_html=True)
    else:
        # High quality inline vector SVG fallback if image is missing
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <svg width="220" height="180" viewBox="0 0 200 160" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="200" height="160" rx="16" fill="#E0F2FE"/>
                <path d="M100 40C70 40 50 60 50 90C50 120 100 140 100 140C100 140 150 120 150 90C150 60 130 40 100 40Z" fill="#38BDF8" opacity="0.3"/>
                <path d="M85 85H115M100 70V100" stroke="#0284C7" stroke-width="8" stroke-linecap="round"/>
            </svg>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# =========================================================
# INPUT / PREDICTION SECTION
# =========================================================

left_header, right_header = st.columns([1.05, 0.95], gap="large")

with left_header:
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)

with right_header:
    st.markdown('<div class="section-title">💰 Prediction Result</div>', unsafe_allow_html=True)


left, right = st.columns([1.05, 0.95], gap="large")

# INPUT FORM
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Please enter your details below to estimate the insurance cost.</div>', unsafe_allow_html=True)

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
    predict = st.button(" Predict Insurance Cost")
    st.markdown('</div>', unsafe_allow_html=True)

# RESULT CARD
with right:
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
<div style="margin-top:25px; opacity:0.8; line-height:1.6; font-size: 15px;">
Enter the patient information on the left and click "Predict Insurance Cost" to view the estimated charges.
</div>
<div style="font-size: 12px; opacity: 0.5;">Awaiting input...</div>
</div>""",
            unsafe_allow_html=True
        )

# =========================================================
# MODEL PERFORMANCE
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
