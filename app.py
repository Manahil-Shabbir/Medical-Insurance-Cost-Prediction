import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("insurance.csv")

# -----------------------------
# Features and Target
# -----------------------------
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]

numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        (
            "categorical",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# -----------------------------
# Model Pipeline
# -----------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# -----------------------------
# UI
# -----------------------------
st.title("🏥 Medical Insurance Cost Predictor")

st.write(
    "Enter the following information to estimate medical insurance cost "
    "using a Multiple Linear Regression model."
)

st.divider()

# Input fields
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=23.5,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

smoker = st.selectbox(
    "Smoking Status",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ["Northeast", "Northwest", "Southeast", "Southwest"]
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("💰 Predict Insurance Cost", use_container_width=True):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [gender],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    st.success(
        f"Estimated Medical Insurance Cost: **${prediction:,.2f}**"
    )

st.divider()

st.caption(
    "This prediction is for educational purposes only and should not "
    "be considered professional financial or medical advice."
)
