import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="HealthPredict - Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("🏥 HealthPredict")
st.subheader("Medical Insurance Cost Predictor")

st.write(
    "Estimate annual medical insurance costs using "
    "a Multiple Linear Regression machine learning model."
)

st.divider()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("insurance.csv")

# -----------------------------
# Features and target
# -----------------------------
X = df[
    ["age", "sex", "bmi", "children", "smoker", "region"]
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

# -----------------------------
# Preprocessing
# -----------------------------
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

# -----------------------------
# Model
# -----------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Input section
# -----------------------------
st.header("👤 Enter Personal Information")

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
        ["Female", "Male"]
    )

with col5:
    smoker = st.selectbox(
        "Smoking Status",
        ["No", "Yes"]
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

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button(
    "💰 Predict Insurance Cost",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [gender.lower()],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker.lower()],
        "region": [region.lower()]
    })

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    st.success("Prediction generated successfully!")

    st.metric(
        "Estimated Annual Medical Insurance Cost",
        f"${prediction:,.2f}"
    )

# -----------------------------
# Project information
# -----------------------------
st.divider()

st.header("📊 About the Model")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.info(
        "**Algorithm**\n\n"
        "Multiple Linear Regression"
    )

with info2:
    st.info(
        "**Task**\n\n"
        "Regression"
    )

with info3:
    st.info(
        "**Dataset**\n\n"
        "Medical Insurance Dataset"
    )

with info4:
    st.info(
        "**Train/Test Split**\n\n"
        "80% / 20%"
    )

st.divider()

st.caption(
    "⚠️ This application is an educational machine learning project. "
    "Predictions are estimates and should not be considered actual "
    "insurance quotations or professional medical advice."
)
