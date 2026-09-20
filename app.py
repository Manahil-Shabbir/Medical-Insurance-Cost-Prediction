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
# PROFESSIONAL THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F8FB;
}

.block-container {
    max-width: 1150px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Hero */
.hero-box {
    background: linear-gradient(
        135deg,
        #123B5D,
        #176B87
    );
    padding: 38px;
    border-radius: 24px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 10px 28px rgba(18,59,93,0.15);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 19px;
    font-weight: 600;
    color: #D7F1F0;
    margin-bottom: 12px;
}

.hero-text {
    font-size: 15px;
    color: #E6F2F5;
    line-height: 1.6;
}

/* Section titles */
.section-title {
    color: #123B5D;
    font-size: 25px;
    font-weight: 750;
    margin-top: 15px;
}

/* Prediction */
.result-box {
    background: linear-gradient(
        135deg,
        #123B5D,
        #0B8F87
    );
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin: 20px 0;
    box-shadow: 0 10px 25px rgba(11,143,135,0.20);
}

.result-label {
    font-size: 15px;
    color: #DDF4F2;
}

.result-value {
    font-size: 40px;
    font-weight: 800;
    margin-top: 5px;
}

/* Button */
.stButton > button {
    background-color: #0B8F87;
    color: white;
    border: none;
    border-radius: 11px;
    height: 50px;
    font-size: 16px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #08736D;
    color: white;
}

/* Footer */
.footer {
    text-align: center;
    color: #80909C;
    font-size: 12px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


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
# TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# HERO SECTION
# ============================================================

hero_left, hero_right = st.columns(
    [1.5, 1],
    gap="large"
)

with hero_left:

    st.markdown(
        """
        <div class="hero-box">

            <div class="hero-title">
                🏥 HealthPredict
            </div>

            <div class="hero-subtitle">
                Medical Insurance Cost Predictor
            </div>

            <div class="hero-text">
                Estimate annual medical insurance costs
                using machine learning and patient information.
                <br><br>
                Simple • Fast • Data-Driven
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with hero_right:

    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=900&q=85",
        use_container_width=True
    )


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the details below to generate an estimated "
    "annual insurance cost."
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
# PREDICT
# ============================================================

predict = st.button(
    "💰  Predict Insurance Cost",
    use_container_width=True
)


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

    st.success(
        "Prediction generated successfully!"
    )

    st.markdown(
        f"""
        <div class="result-box">

            <div class="result-label">
                ESTIMATED ANNUAL INSURANCE COST
            </div>

            <div class="result-value">
                ${prediction:,.2f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Model",
            "Linear Regression"
        )

    with result2:
        st.metric(
            "Prediction Type",
            "Regression"
        )

    with result3:
        st.metric(
            "Training Data",
            f"{len(X_train)} records"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📊 About This Project</div>',
    unsafe_allow_html=True
)

st.write(
    "This application uses the Medical Insurance dataset "
    "to predict insurance charges from demographic and "
    "lifestyle features."
)


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
        "Age • BMI • Children • Gender • "
        "Smoker • Region"
    )


with info4:

    st.info(
        "**Split**\n\n"
        "80% Training\n\n"
        "20% Testing"
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)

step1, step2, step3 = st.columns(3)


with step1:

    st.subheader("01 · Enter")

    st.write(
        "Provide the patient's age, BMI, "
        "children, gender, smoking status "
        "and region."
    )


with step2:

    st.subheader("02 · Process")

    st.write(
        "Categorical variables are encoded and "
        "combined with numerical features."
    )


with step3:

    st.subheader("03 · Predict")

    st.write(
        "The trained Linear Regression model "
        "generates an estimated insurance cost."
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

st.divider()

st.warning(
    "⚠️ **Educational Disclaimer:** This application is "
    "created for machine learning demonstration purposes. "
    "The predicted value is an estimate and should not be "
    "treated as an actual insurance quotation or professional "
    "medical advice."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🏥 HealthPredict · Medical Insurance Cost Prediction
        <br>
        Built with Python · Scikit-learn · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
