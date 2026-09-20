import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HealthPredict | Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM THEME
# ============================================================

st.markdown("""
<style>

    /* Main page */
    .stApp {
        background-color: #F4F8FB;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hide unnecessary Streamlit elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Headings */
    h1, h2, h3 {
        color: #123B5D !important;
    }

    /* Hero */
    .hero {
        background: linear-gradient(
            135deg,
            #123B5D 0%,
            #176B87 55%,
            #159A8B 100%
        );

        padding: 38px 42px;
        border-radius: 24px;
        margin-bottom: 28px;
        box-shadow: 0 12px 30px rgba(18, 59, 93, 0.15);
    }

    .hero-small {
        color: #BFEDE7;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .hero-title {
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-text {
        color: #E2F2F5;
        font-size: 15px;
        line-height: 1.7;
        max-width: 720px;
    }

    /* Section heading */
    .section-heading {
        color: #123B5D;
        font-size: 24px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 4px;
    }

    .section-text {
        color: #718594;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* Result area */
    .result-title {
        color: #123B5D;
        font-size: 24px;
        font-weight: 750;
        margin-top: 25px;
    }

    /* Button */
    .stButton > button {
        background: #159A8B;
        color: white;
        border: none;
        border-radius: 10px;
        height: 50px;
        font-size: 16px;
        font-weight: 700;
        box-shadow: 0 5px 14px rgba(21, 154, 139, 0.20);
    }

    .stButton > button:hover {
        background: #117D72;
        color: white;
    }

    /* Inputs */
    div[data-baseweb="input"],
    div[data-baseweb="select"] {
        border-radius: 8px;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #8798A5;
        font-size: 12px;
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid #DCE6EC;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")


df = load_data()


# ============================================================
# PREPARE DATA
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


# ============================================================
# PREPROCESSING
# ============================================================

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


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-small">
        🏥 MACHINE LEARNING • HEALTHCARE ANALYTICS
    </div>

    <div class="hero-title">
        HealthPredict
    </div>

    <div class="hero-text">
        Medical Insurance Cost Predictor
        <br><br>
        Estimate annual medical insurance costs using
        a Multiple Linear Regression model trained on
        historical healthcare data.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-heading">👤 Patient Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-text">'
    'Enter the information below to generate an estimated insurance cost.'
    '</div>',
    unsafe_allow_html=True
)


# First row
col1, col2, col3 = st.columns(3, gap="large")


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


with col3:

    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )


# Second row
col4, col5, col6 = st.columns(3, gap="large")


with col4:

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )


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


st.write("")


# ============================================================
# PREDICT BUTTON
# ============================================================

predict = st.button(
    "💰  Predict Insurance Cost",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    input_data = pd.DataFrame(
        {
            "age": [age],
            "sex": [gender.lower()],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker.lower()],
            "region": [region.lower()]
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
        '<div class="result-title">💰 Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.success(
        "Prediction generated successfully."
    )

    result_col1, result_col2, result_col3 = st.columns(
        3,
        gap="large"
    )

    with result_col1:

        st.metric(
            "Estimated Annual Cost",
            f"${prediction:,.2f}",
            border=True
        )

    with result_col2:

        st.metric(
            "Machine Learning Model",
            "Linear Regression",
            border=True
        )

    with result_col3:

        st.metric(
            "Prediction Type",
            "Regression",
            border=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-heading">📊 Model Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-text">'
    'Key information about the machine learning workflow.'
    '</div>',
    unsafe_allow_html=True
)


info1, info2, info3, info4 = st.columns(
    4,
    gap="medium"
)


with info1:

    st.info(
        "**Dataset**\n\n"
        "Medical Insurance Dataset"
    )


with info2:

    st.info(
        "**Algorithm**\n\n"
        "Multiple Linear Regression"
    )


with info3:

    st.info(
        "**Features**\n\n"
        "Age, BMI, Children, Gender, "
        "Smoking Status & Region"
    )


with info4:

    st.info(
        "**Data Split**\n\n"
        "80% Training / 20% Testing"
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-heading">⚙️ How It Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-text">'
    'From patient information to machine learning prediction.'
    '</div>',
    unsafe_allow_html=True
)


step1, step2, step3 = st.columns(
    3,
    gap="large"
)


with step1:

    st.markdown("### 01 · Enter Information")

    st.write(
        "Provide age, BMI, number of children, "
        "gender, smoking status and region."
    )


with step2:

    st.markdown("### 02 · Process Data")

    st.write(
        "Numerical and categorical variables are "
        "processed through a machine learning pipeline."
    )


with step3:

    st.markdown("### 03 · Generate Prediction")

    st.write(
        "The trained regression model estimates "
        "the expected annual insurance cost."
    )


# ============================================================
# DATASET PREVIEW
# ============================================================

with st.expander("🔎 View Dataset Preview"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "⚠️ **Disclaimer:** This application is an educational "
    "machine learning project. The predicted amount is an "
    "estimate and should not be considered an actual insurance "
    "quotation or professional medical advice."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        <b>HealthPredict</b> · Medical Insurance Cost Prediction
        · Machine Learning Project
        <br>
        Built with Python, Scikit-learn & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
