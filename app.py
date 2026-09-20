import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="HealthPredict",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# SIMPLE PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F8FA;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
    }

    h1, h2, h3 {
        color: #123B5D;
    }

    .stButton button {
        background-color: #0B8F87;
        color: white;
        border-radius: 10px;
        border: none;
        height: 48px;
        font-weight: 700;
    }

    .stButton button:hover {
        background-color: #08766F;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")


df = load_data()


# ============================================================
# FEATURES AND TARGET
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
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
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

model.fit(X_train, y_train)


# ============================================================
# HEADER
# ============================================================

st.title("🏥 HealthPredict")

st.subheader("Medical Insurance Cost Predictor")

st.write(
    "Estimate annual medical insurance costs using "
    "a Multiple Linear Regression machine learning model."
)

st.caption(
    "Healthcare Analytics • Machine Learning • Regression"
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("👤 Patient Information")

st.write(
    "Enter the patient's information to generate "
    "an estimated annual insurance cost."
)


col1, col2, col3 = st.columns(3)


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


st.write("")


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict = st.button(
    "💰 Predict Insurance Cost",
    use_container_width=True
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

    prediction = model.predict(input_data)[0]

    prediction = max(0, float(prediction))

    st.divider()

    st.header("💰 Prediction Result")

    st.success("Prediction generated successfully!")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Estimated Annual Cost",
            f"${prediction:,.2f}"
        )

    with result_col2:

        st.metric(
            "Model",
            "Linear Regression"
        )

    with result_col3:

        st.metric(
            "Task",
            "Regression"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header("📊 Model Information")

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
        "Smoking Status and Region"
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

    st.subheader("01 · Input")

    st.write(
        "Enter age, BMI, children, gender, "
        "smoking status and region."
    )


with step2:

    st.subheader("02 · Processing")

    st.write(
        "The data is processed using numerical "
        "and categorical feature preprocessing."
    )


with step3:

    st.subheader("03 · Prediction")

    st.write(
        "The trained regression model estimates "
        "the annual medical insurance charges."
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("🔎 View Dataset Preview"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "⚠️ This application is an educational machine learning "
    "project. Predictions are estimates and should not be "
    "considered actual insurance quotations or professional "
    "medical advice."
)


st.caption(
    "HealthPredict • Medical Insurance Cost Prediction • "
    "Built with Python, Scikit-learn and Streamlit"
)
