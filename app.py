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
    layout="wide"
)


# ============================================================
# COLORS / THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F9FC;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #123B5D;
}

div.stButton > button {
    background-color: #0F8B8D;
    color: white;
    border-radius: 10px;
    border: none;
    height: 48px;
    font-weight: 700;
    font-size: 15px;
}

div.stButton > button:hover {
    background-color: #0B7072;
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #DCE8EE;
    padding: 20px;
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("insurance.csv")


# ============================================================
# FEATURES
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
# PREPROCESSING + MODEL
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


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ============================================================
# TRAIN
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# HEADER
# ============================================================

header1, header2 = st.columns([2.2, 1])

with header1:

    st.markdown(
        "# 🏥 HealthPredict"
    )

    st.markdown(
        "### Medical Insurance Cost Predictor"
    )

    st.write(
        "A machine learning application that estimates "
        "annual medical insurance costs from personal and "
        "lifestyle information."
    )

with header2:

    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1200&q=80",
        use_container_width=True
    )

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("👤 Personal Information")

st.caption(
    "Enter the information below to generate an estimated "
    "annual insurance cost."
)


col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=25
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
        value=0
    )


col4, col5, col6 = st.columns(3)


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

predict_button = st.button(
    "💰  Predict Insurance Cost",
    use_container_width=True
)


# ============================================================
# RESULT
# ============================================================

if predict_button:

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

    prediction = model.predict(input_data)[0]

    prediction = max(0, float(prediction))

    st.success(
        "Prediction generated successfully!"
    )

    st.subheader("💰 Estimated Insurance Cost")

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Estimated Annual Cost",
            f"${prediction:,.2f}"
        )

    with result2:
        st.metric(
            "Model",
            "Linear Regression"
        )

    with result3:
        st.metric(
            "Prediction Type",
            "Regression"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header("📊 Project Information")


info1, info2, info3, info4 = st.columns(4)


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

st.divider()

st.header("⚙️ How It Works")

step1, step2, step3 = st.columns(3)


with step1:

    st.markdown("### 01 — Input")

    st.write(
        "The user enters demographic, lifestyle, "
        "and health-related information."
    )


with step2:

    st.markdown("### 02 — Processing")

    st.write(
        "Numerical and categorical features are "
        "processed through a machine learning pipeline."
    )


with step3:

    st.markdown("### 03 — Prediction")

    st.write(
        "The trained regression model estimates "
        "the expected medical insurance cost."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ Educational project only. The prediction is an "
    "estimated machine learning output and is not an actual "
    "insurance quotation or professional medical advice."
)

st.caption(
    "Medical Insurance Cost Prediction • Machine Learning Project"
)
