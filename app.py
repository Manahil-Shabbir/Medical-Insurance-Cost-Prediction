import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HealthPredict | Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- MAIN BACKGROUND ---------- */

    .stApp {
        background-color: #f5f9fc;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }


    /* ---------- NAVBAR ---------- */

    .navbar {
        background: white;
        padding: 18px 28px;
        border-radius: 16px;
        border: 1px solid #e3ebf0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        box-shadow: 0 5px 18px rgba(20, 60, 90, 0.05);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        border-radius: 50%;
        background: #168579;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 23px;
        font-weight: bold;
    }

    .brand-name {
        color: #12395b;
        font-size: 21px;
        font-weight: 750;
        line-height: 1.1;
    }

    .brand-subtitle {
        color: #8aa4b8;
        font-size: 11px;
        margin-top: 4px;
    }

    .nav-right {
        color: #53738c;
        font-size: 12px;
        font-weight: 600;
    }


    /* ---------- HERO ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #eaf8fa 0%,
            #f4fbfd 55%,
            #e8f8f5 100%
        );

        border-radius: 22px;
        padding: 38px 42px;
        border: 1px solid #dcecef;
        margin-bottom: 25px;
    }

    .hero-badge {
        display: inline-block;
        background: #d7f4ee;
        color: #087f70;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #12395b;
        font-size: 38px;
        font-weight: 800;
        line-height: 1.15;
        margin: 0;
    }

    .hero-description {
        color: #52738c;
        font-size: 14px;
        line-height: 1.7;
        margin-top: 14px;
        max-width: 700px;
    }


    /* ---------- SECTION TITLE ---------- */

    .section-title {
        color: #12395b;
        font-size: 20px;
        font-weight: 750;
        margin: 10px 0 12px;
    }


    /* ---------- CARDS ---------- */

    .card {
        background: white;
        border: 1px solid #e2ebf0;
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 7px 25px rgba(25, 65, 90, 0.055);
        min-height: 350px;
    }

    .card-title {
        color: #163c5c;
        font-size: 19px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .card-subtitle {
        color: #7892a4;
        font-size: 12px;
        line-height: 1.6;
        margin-bottom: 20px;
    }


    /* ---------- PREDICTION CARD ---------- */

    .prediction-card {
        background: linear-gradient(
            145deg,
            #104c69,
            #155b76
        );

        border-radius: 18px;
        padding: 28px;
        min-height: 350px;
        box-shadow: 0 10px 30px rgba(16, 76, 105, 0.16);
    }

    .prediction-title {
        color: white;
        font-size: 21px;
        font-weight: 750;
    }

    .prediction-subtitle {
        color: #c5dce7;
        font-size: 12px;
        margin-top: 5px;
    }

    .prediction-empty {
        margin-top: 70px;
        text-align: center;
        color: #c7dce5;
        font-size: 14px;
        line-height: 1.7;
    }

    .result-box {
        margin-top: 55px;
        background: #17998d;
        border: 2px solid #46c8ba;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
    }

    .result-label {
        color: white;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .result-value {
        color: white;
        font-size: 35px;
        font-weight: 850;
        margin-top: 8px;
    }

    .result-period {
        color: #d3f0ec;
        font-size: 11px;
        margin-top: 3px;
    }


    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: white;
        border: 1px solid #e2ebf0;
        border-radius: 16px;
        padding: 20px;
        min-height: 145px;
        box-shadow: 0 5px 18px rgba(25, 65, 90, 0.04);
    }

    .info-number {
        width: 29px;
        height: 29px;
        border-radius: 50%;
        background: #e1f3f1;
        color: #087f70;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 800;
    }

    .info-title {
        color: #163c5c;
        font-size: 14px;
        font-weight: 750;
        margin-top: 10px;
    }

    .info-text {
        color: #7892a4;
        font-size: 11px;
        line-height: 1.6;
        margin-top: 5px;
    }


    /* ---------- MODEL CARD ---------- */

    .model-card {
        background: #e8f8f5;
        border: 1px solid #cfece7;
        border-radius: 16px;
        padding: 20px;
        min-height: 145px;
    }

    .model-title {
        color: #087f70;
        font-size: 15px;
        font-weight: 750;
        margin-bottom: 12px;
    }

    .model-line {
        color: #426b6a;
        font-size: 11px;
        margin-bottom: 7px;
    }

    .model-label {
        font-weight: 750;
        color: #32756f;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 10px;
        border: none;
        background: #159a8b;
        color: white;
        font-size: 14px;
        font-weight: 750;
        margin-top: 10px;
    }

    .stButton > button:hover {
        background: #117d72;
        color: white;
        border: none;
    }


    /* ---------- INPUTS ---------- */

    div[data-baseweb="input"] {
        border-radius: 9px;
    }

    div[data-baseweb="select"] {
        border-radius: 9px;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #89a1b2;
        font-size: 11px;
        margin-top: 30px;
        line-height: 1.7;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = "insurance.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# FEATURES + TARGET
# ============================================================

X = df[
    [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region"
    ]
]

y = df["charges"]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = [
    "age",
    "bmi",
    "children"
]

categorical_features = [
    "sex",
    "smoker",
    "region"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_features
        ),

        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train model
model.fit(
    X_train,
    y_train
)


# ============================================================
# NAVBAR
# ============================================================

st.markdown("""
<div class="navbar">

    <div class="brand">

        <div class="brand-icon">
            ♥
        </div>

        <div>
            <div class="brand-name">
                HealthPredict
            </div>

            <div class="brand-subtitle">
                Smarter Healthcare Decisions
            </div>
        </div>

    </div>

    <div class="nav-right">
        🛡️ ML Powered &nbsp; • &nbsp; Healthcare Insights
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-badge">
        ✣ &nbsp; MACHINE LEARNING PROJECT
    </div>

    <div class="hero-title">
        Medical Insurance Cost Predictor
    </div>

    <div class="hero-description">
        Estimate medical insurance costs using a Multiple Linear
        Regression model trained on historical healthcare data.
        Enter your personal information below to generate an
        estimated annual insurance cost.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN SECTION
# ============================================================

left, right = st.columns(
    [1.08, 0.92],
    gap="large"
)


# ============================================================
# INPUT CARD
# ============================================================

with left:

    st.markdown(
        '<div class="section-title">👤 Personal Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card-title">
            Enter Your Details
        </div>

        <div class="card-subtitle">
            Provide the following information to estimate
            your medical insurance cost.
        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=25,
            step=1
        )


    with col2:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=23.5,
            step=0.1
        )


    col3, col4 = st.columns(2)


    with col3:

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )


    with col4:

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )


    col5, col6 = st.columns(2)


    with col5:

        smoker = st.selectbox(
            "Smoking Status",
            [
                "No",
                "Yes"
            ]
        )


    with col6:

        region = st.selectbox(
            "Region",
            [
                "Northeast",
                "Northwest",
                "Southeast",
                "Southwest"
            ]
        )


    predict = st.button(
        "✣   Predict Insurance Cost"
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION CARD
# ============================================================

with right:

    st.markdown(
        '<div class="section-title">💰 Prediction</div>',
        unsafe_allow_html=True
    )


    # If prediction button clicked
    if predict:

        # IMPORTANT:
        # Dataset categorical values are lowercase.
        # Convert user input to lowercase before prediction.

        input_data = pd.DataFrame(
            {
                "age": [float(age)],

                "sex": [
                    gender.lower()
                ],

                "bmi": [float(bmi)],

                "children": [
                    int(children)
                ],

                "smoker": [
                    smoker.lower()
                ],

                "region": [
                    region.lower()
                ]
            }
        )


        prediction = model.predict(
            input_data
        )[0]


        prediction = max(
            0,
            float(prediction)
        )


        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-title">
                    Prediction Result
                </div>

                <div class="prediction-subtitle">
                    Estimated annual medical insurance cost
                </div>

                <div class="result-box">

                    <div class="result-label">
                        ESTIMATED INSURANCE COST
                    </div>

                    <div class="result-value">
                        ${prediction:,.2f}
                    </div>

                    <div class="result-period">
                        Estimated Annual Medical Insurance Cost
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.markdown(
            """
            <div class="prediction-card">

                <div class="prediction-title">
                    Prediction Result
                </div>

                <div class="prediction-subtitle">
                    Estimated annual medical insurance cost
                </div>

                <div class="prediction-empty">
                    ✦<br><br>
                    Enter your information and click<br>
                    <b>Predict Insurance Cost</b><br><br>
                    to generate your estimate.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-number">
                01
            </div>

            <div class="info-title">
                Enter Information
            </div>

            <div class="info-text">
                Provide age, BMI, smoking status,
                children, gender and region.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-number">
                02
            </div>

            <div class="info-title">
                Preprocessing
            </div>

            <div class="info-text">
                Numerical and categorical features
                are processed using a pipeline.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-number">
                03
            </div>

            <div class="info-title">
                Machine Learning
            </div>

            <div class="info-text">
                Multiple Linear Regression analyzes
                patterns learned from historical data.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        """
        <div class="model-card">

            <div class="model-title">
                ⓘ Model Information
            </div>

            <div class="model-line">
                <span class="model-label">Model:</span>
                Multiple Linear Regression
            </div>

            <div class="model-line">
                <span class="model-label">Task:</span>
                Regression
            </div>

            <div class="model-line">
                <span class="model-label">Target:</span>
                Medical Charges
            </div>

            <div class="model-line">
                <span class="model-label">Split:</span>
                80% Train / 20% Test
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <b>Medical Insurance Cost Prediction</b>
        &nbsp; • &nbsp;
        Machine Learning Project

        <br>

        This application provides an ML-based estimate for
        educational purposes and is not an actual insurance quotation.

    </div>
    """,
    unsafe_allow_html=True
)
