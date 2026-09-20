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

# Helper function to convert local image to Base64 for CSS background
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

# Save your banner image as 'hero_banner.png' in the same folder as app.py
img_base64 = get_base64_of_bin_file("hero_banner.png")
banner_bg = f"url('data:image/png;base64,{img_base64}')" if img_base64 else "#DDF5F3"

# =========================================================
# CSS STYLING
# =========================================================

st.markdown(f"""
<style>

.stApp {{
    background: #F7FAFC;
}}

.block-container {{
    max-width: 1150px;
    padding-top: 35px;
    padding-bottom: 40px;
}}

/* HERO SECTION */

.hero {{
    background: linear-gradient(135deg, #EAF8F7 0%, #F4FBFC 100%);
    border: 1px solid #D8ECEE;
    border-radius: 24px;
    padding: 38px 42px;
    margin-bottom: 35px;
}}

.badge {{
    display: inline-block;
    background: #D5F4EE;
    color: #087C70;
    border-radius: 25px;
    padding: 7px 15px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 14px;
}}

.hero-title {{
    font-size: 38px;
    font-weight: 800;
    color: #123B5D;
    line-height: 1.15;
    margin-bottom: 15px;
}}

.hero-text {{
    color: #64798A;
    font-size: 15px;
    line-height: 1.7;
    max-width: 500px;
}}

/* HERO ILLUSTRATION */

.medical-art {{
    height: 220px;
    border-radius: 20px;
    background-image: {banner_bg};
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
}}

/* SECTIONS & HEADINGS */

.section-title {{
    color: #123B5D;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 4px;
}}

.section-subtitle {{
    color: #7A8B98;
    font-size: 14px;
    margin-bottom: 20px;
}}

/* INPUT CARD */

.card {{
    background: white;
    border: 1px solid #E0E8ED;
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 5px 18px rgba(18,59,93,0.04);
}}

.card-subtitle {{
    color: #7A8B98;
    font-size: 13px;
    margin-bottom: 20px;
}}

/* INPUT FORM ELEMENTS */

label {{
    color: #29485E !important;
    font-weight: 600 !important;
}}

div[data-baseweb="select"] > div {{
    border-radius: 10px;
    border-color: #D8E2E8;
}}

input {{
    border-radius: 10px !important;
}}

/* BUTTON STYLING */

.stButton > button {{
    width: 100%;
    height: 50px;
    border-radius: 11px;
    border: none;
    background: #087F73;
    color: white;
    font-size: 16px;
    font-weight: 700;
}}

.stButton > button:hover {{
    background: #066C63;
    color: white;
}}

/* PREDICTION CARD */

.result-card {{
    background: #0F4C64;
    border-radius: 20px;
    padding: 32px;
    min-height: 280px;
    color: white;
    box-shadow: 0 10px 25px rgba(18,59,93,0.12);
}}

.result-heading {{
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 6px;
}}

.result-subtitle {{
    font-size: 14px;
    opacity: 0.8;
}}

.result-price {{
    font-size: 42px;
    font-weight: 800;
    margin-top: 40px;
}}

/* INFO CARDS */

.info-card {{
    background: white;
    border: 1px solid #E0E8ED;
    border-radius: 16px;
    padding: 20px;
    min-height: 120px;
}}

.info-title {{
    color: #123B5D;
    font-size: 16px;
    font-weight: 750;
}}

.info-text {{
    color: #788995;
    font-size: 13px;
    margin-top: 8px;
    line-height: 1.5;
}}

/* FOOTER */

.footer {{
    text-align: center;
    color: #94A2AC;
    font-size: 12px;
    margin-top: 30px;
}}

hr {{
    border: none;
    border-top: 1px solid #DCE6EB;
    margin: 32px 0;
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

hero1, hero2 = st.columns([1.4, 1], gap="large")

with hero1:
    st.markdown('<div class="badge">+ ML Powered</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Medical Insurance Cost<br>Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-text">Estimate your medical insurance cost using a machine learning model trained on historical healthcare data. Get a quick estimate based on your personal information.</div>', unsafe_allow_html=True)

with hero2:
    st.markdown('<div class="medical-art"></div>', unsafe_allow_html=True)

st.write("")

# =========================================================
# INPUT / PREDICTION SECTION
# =========================================================

left_header, right_header = st.columns([1.05, 0.95], gap="large")

with left_header:
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)

with right_header:
    st.markdown('<div class="section-title">💰 Prediction</div>', unsafe_allow_html=True)


left, right = st.columns([1.05, 0.95], gap="large")

# INPUT FORM
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Please enter your details to estimate the insurance cost.</div>', unsafe_allow_html=True)

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
    predict = st.button("💰 Predict Insurance Cost")
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
<div class="result-heading">Prediction Result</div>
<div class="result-subtitle">Your estimated annual medical insurance cost</div>
<div class="result-price">${prediction:,.2f}</div>
</div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div class="result-card">
<div class="result-heading">Prediction Result</div>
<div class="result-subtitle">Your estimated annual medical insurance cost</div>
<div style="margin-top:35px; opacity:0.75; line-height:1.6;">
Enter the information and click "Predict Insurance Cost".
</div>
</div>""",
            unsafe_allow_html=True
        )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Performance of the trained regression model on test data.</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""<div class="info-card"><div class="info-title">🤖 Algorithm</div><div class="info-text">Linear Regression</div></div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""<div class="info-card"><div class="info-title">📈 R² Score</div><div class="info-text">{r2:.3f}</div></div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""<div class="info-card"><div class="info-title">📉 MAE</div><div class="info-text">${mae:,.0f}</div></div>""", unsafe_allow_html=True)

with m4:
    st.markdown(f"""<div class="info-card"><div class="info-title">📚 Records</div><div class="info-text">{len(df):,} records</div></div>""", unsafe_allow_html=True)

# DISCLAIMER & FOOTER
st.markdown("<hr>", unsafe_allow_html=True)
st.warning("⚠️ **Disclaimer:** This application is an educational machine learning project. The prediction is an estimate and should not be treated as an actual insurance quote.")
st.markdown('<div class="footer">Medical Insurance Cost Prediction · Machine Learning Project</div>', unsafe_allow_html=True)
