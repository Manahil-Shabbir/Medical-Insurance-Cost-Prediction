import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HealthPredict | Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# COLORS / STYLE
# =========================================================
st.markdown("""
<style>

/* Main background */
.stApp {
    background:
        linear-gradient(rgba(247,250,252,0.94), rgba(247,250,252,0.96)),
        url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1800&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Main content width */
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Titles */
.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #12304A;
    line-height: 1.1;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 18px;
    color: #587083;
    line-height: 1.6;
}

/* Small badge */
.badge {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 30px;
    background: #E7F7F5;
    color: #087F73;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 15px;
}

/* Section headings */
.section-title {
    color: #12304A;
    font-size: 28px;
    font-weight: 750;
    margin-top: 30px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #718393;
    margin-bottom: 20px;
}

/* Cards */
.info-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #E4EBF0;
    border-radius: 18px;
    padding: 22px;
    min-height: 145px;
    box-shadow: 0 8px 25px rgba(18,48,74,0.06);
}

.info-card h3 {
    color: #12304A;
    margin-bottom: 8px;
}

.info-card p {
    color: #687A88;
    line-height: 1.5;
}

/* Prediction result */
.result-box {
    background: linear-gradient(135deg, #12304A, #087F73);
    border-radius: 22px;
    padding: 28px;
    color: white;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 12px 30px rgba(18,48,74,0.18);
}

.result-label {
    font-size: 15px;
    opacity: 0.85;
    margin-bottom: 6px;
}

.result-value {
    font-size: 42px;
    font-weight: 800;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #087F73, #0B9B8D);
    color: white;
    font-size: 17px;
    font-weight: 750;
    box-shadow: 0 7px 18px rgba(8,127,115,0.22);
    transition: 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(8,127,115,0.30);
}

/* Inputs */
.stSelectbox label,
.stNumberInput label {
    color: #29465A !important;
    font-weight: 650 !important;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: #DDE6EC;
    margin: 35px 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #82919D;
    font-size: 13px;
    padding-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    with open("insurance_model.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()


# =========================================================
# HERO SECTION
# =========================================================
hero_left, hero_right = st.columns([1.25, 1], gap="large")

with hero_left:
    st.markdown(
        '<div class="badge">🏥 AI-Powered Healthcare Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-title">HealthPredict</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Estimate medical insurance costs using a machine learning model
        based on personal and healthcare-related information.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.caption("✨ Fast • Simple • Data-Driven")

with hero_right:
    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1200&q=85",
        use_container_width=True
    )


st.markdown("<hr>", unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================
st.markdown(
    '<div class="section-title">Patient Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Enter the details below to estimate the insurance cost.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30,
        step=1
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

with col2:
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col3:
    smoker = st.selectbox(
        "Smoking Status",
        ["No", "Yes"]
    )

    region = st.selectbox(
        "Region",
        ["Southwest", "Southeast", "Northwest", "Northeast"]
    )


st.write("")


# =========================================================
# PREDICTION
# =========================================================
if st.button("💰  Predict Insurance Cost"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex.lower()],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker.lower()],
        "region": [region.lower()]
    })

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Estimated Annual Insurance Cost</div>
            <div class="result-value">${prediction:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PROJECT INFORMATION
# =========================================================
st.markdown(
    '<div class="section-title">About the Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">A machine learning workflow for estimating healthcare insurance charges.</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Data Driven</h3>
        <p>
        Uses patient characteristics such as age, BMI,
        smoking status, children and region.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h3>🤖 Machine Learning</h3>
        <p>
        Multiple Linear Regression is used to estimate
        the expected insurance charges.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h3>⚡ Instant Prediction</h3>
        <p>
        Enter patient information and receive an
        estimated insurance cost instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HOW IT WORKS
# =========================================================
st.markdown(
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True
)

step1, step2, step3 = st.columns(3, gap="large")

with step1:
    st.markdown("""
    <div class="info-card">
        <h3>01 · Input</h3>
        <p>Provide the patient's basic information and healthcare-related attributes.</p>
    </div>
    """, unsafe_allow_html=True)

with step2:
    st.markdown("""
    <div class="info-card">
        <h3>02 · Process</h3>
        <p>The trained machine learning pipeline processes the entered information.</p>
    </div>
    """, unsafe_allow_html=True)

with step3:
    st.markdown("""
    <div class="info-card">
        <h3>03 · Predict</h3>
        <p>The application generates an estimated annual insurance charge.</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# DISCLAIMER
# =========================================================
st.markdown("<hr>", unsafe_allow_html=True)

st.info(
    "⚠️ **Disclaimer:** This application provides a machine-learning-based "
    "estimate for educational and demonstration purposes only. "
    "It should not be considered a medical, financial, or insurance decision."
)

st.markdown(
    '<div class="footer">HealthPredict · Medical Insurance Cost Prediction · Machine Learning Project</div>',
    unsafe_allow_html=True
)
