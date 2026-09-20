import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #F7FAFC;
}

.block-container {
    max-width: 1150px;
    padding-top: 35px;
    padding-bottom: 40px;
}

/* HERO */

.hero {
    background: linear-gradient(
        135deg,
        #EAF8F7 0%,
        #F4FBFC 100%
    );

    border: 1px solid #D8ECEE;
    border-radius: 24px;
    padding: 38px 42px;
    margin-bottom: 35px;
}

.badge {
    display: inline-block;
    background: #D5F4EE;
    color: #087C70;
    border-radius: 25px;
    padding: 7px 15px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    color: #123B5D;
    line-height: 1.1;
    margin-bottom: 15px;
}

.hero-text {
    color: #64798A;
    font-size: 15px;
    line-height: 1.7;
    max-width: 650px;
}

/* MEDICAL ILLUSTRATION */

.medical-art {
    height: 220px;
    border-radius: 20px;
    background: linear-gradient(
        145deg,
        #DDF5F3,
        #E8F4FB
    );
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
}

.medical-art .cross {
    position: absolute;
    font-size: 85px;
    color: #0A9A8B;
    opacity: 0.12;
}

.doctor {
    font-size: 105px;
    z-index: 2;
}

.heart {
    position: absolute;
    right: 30px;
    top: 35px;
    font-size: 45px;
}

.plus {
    position: absolute;
    left: 35px;
    bottom: 35px;
    font-size: 45px;
}

/* SECTION */

.section-title {
    color: #123B5D;
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #7A8B98;
    font-size: 14px;
    margin-bottom: 20px;
}

/* CARD */

.card {
    background: white;
    border: 1px solid #E0E8ED;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 5px 18px rgba(18,59,93,0.05);
}

/* INPUT */

label {
    color: #29485E !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div {
    border-radius: 10px;
    border-color: #D8E2E8;
}

input {
    border-radius: 10px !important;
}

/* BUTTON */

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 11px;
    border: none;
    background: #087F73;
    color: white;
    font-size: 16px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #066C63;
    color: white;
}

/* RESULT */

.result-card {
    background: linear-gradient(
        135deg,
        #123B5D,
        #17647D
    );

    border-radius: 20px;
    padding: 32px;
    min-height: 250px;
    color: white;
    box-shadow: 0 10px 25px rgba(18,59,93,0.15);
}

.result-label {
    font-size: 14px;
    opacity: 0.75;
}

.result-heading {
    font-size: 20px;
    font-weight: 700;
    margin-top: 8px;
}

.result-price {
    font-size: 40px;
    font-weight: 800;
    margin-top: 30px;
}

/* INFO */

.info-card {
    background: white;
    border: 1px solid #E0E8ED;
    border-radius: 16px;
    padding: 20px;
    min-height: 125px;
}

.info-title {
    color: #123B5D;
    font-size: 16px;
    font-weight: 750;
}

.info-text {
    color: #788995;
    font-size: 13px;
    margin-top: 8px;
    line-height: 1.5;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #94A2AC;
    font-size: 12px;
    margin-top: 30px;
}

hr {
    border: none;
    border-top: 1px solid #DCE6EB;
    margin: 32px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
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


# =========================================================
# DATA CHECK
# =========================================================

if df is None:

    st.error(
        "⚠️ insurance.csv was not found in your GitHub repository."
    )

    st.info(
        "Please upload insurance.csv to the same GitHub folder as app.py."
    )

    st.stop()


# Clean column names

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)


# Some versions use 'expenses' instead of 'charges'

if "expenses" in df.columns and "charges" not in df.columns:
    df = df.rename(columns={"expenses": "charges"})


required_columns = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges"
]


missing = [
    col for col in required_columns
    if col not in df.columns
]


if missing:

    st.error(
        "Dataset columns are missing: "
        + ", ".join(missing)
    )

    st.stop()


# =========================================================
# MODEL
# =========================================================

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


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features
        ),

        (
            "cat",
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
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LinearRegression()
        )
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


model.fit(X_train, y_train)


test_predictions = model.predict(X_test)

r2 = r2_score(
    y_test,
    test_predictions
)

mae = mean_absolute_error(
    y_test,
    test_predictions
)


# =========================================================
# HERO
# =========================================================

hero1, hero2 = st.columns(
    [1.55, 1],
    gap="large"
)


with hero1:

    st.markdown(
        '<div class="badge">✦ ML Powered Healthcare</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-title">
            Medical Insurance Cost<br>
            Predictor
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-text">
            Estimate medical insurance costs using a
            machine learning model trained on historical
            healthcare data. Get a quick estimate based
            on personal information.
        </div>
        """,
        unsafe_allow_html=True
    )


with hero2:

    st.markdown(
        """
        <div class="medical-art">

            <div class="cross">✚</div>

            <div class="doctor">👩‍⚕️</div>

            <div class="heart">💙</div>

            <div class="plus">➕</div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# INPUT / PREDICTION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Personal Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Enter the details below to estimate the insurance cost.'
    '</div>',
    unsafe_allow_html=True
)


left, right = st.columns(
    [1.05, 0.95],
    gap="large"
)


# =========================================================
# INPUT
# =========================================================

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### Personal Information"
    )

    st.caption(
        "Provide patient information below."
    )


    c1, c2 = st.columns(2)

    with c1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=30
        )

    with c2:

        sex = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )


    c1, c2 = st.columns(2)

    with c1:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

    with c2:

        children = st.number_input(
            "Children",
            min_value=0,
            max_value=10,
            value=0
        )


    smoker = st.selectbox(
        "Smoking Status",
        ["No", "Yes"]
    )


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


    predict = st.button(
        "💰  Predict Insurance Cost"
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# RESULT
# =========================================================

with right:

    st.markdown(
        '<div class="section-title">💰 Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Estimated annual medical insurance cost.'
        '</div>',
        unsafe_allow_html=True
    )


    if predict:

        user_data = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex.lower()],
                "bmi": [bmi],
                "children": [children],
                "smoker": [smoker.lower()],
                "region": [region.lower()]
            }
        )


        prediction = model.predict(
            user_data
        )[0]


        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Prediction Result
                </div>

                <div class="result-heading">
                    Estimated Annual Cost
                </div>

                <div class="result-price">
                    ${prediction:,.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-label">
                    Prediction Result
                </div>

                <div class="result-heading">
                    Your estimate will appear here
                </div>

                <div style="
                    margin-top:20px;
                    opacity:0.75;
                    line-height:1.6;
                ">
                    Enter the information and click
                    "Predict Insurance Cost".
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Performance of the trained regression model on the test data.'
    '</div>',
    unsafe_allow_html=True
)


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-title">
                🤖 Algorithm
            </div>

            <div class="info-text">
                Linear Regression
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m2:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-title">
                📈 R² Score
            </div>

            <div class="info-text">
                {r2:.3f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m3:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-title">
                📉 MAE
            </div>

            <div class="info-text">
                ${mae:,.0f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m4:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-title">
                📚 Records
            </div>

            <div class="info-text">
                {len(df):,} records
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# WORKFLOW
# =========================================================

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)


w1, w2, w3 = st.columns(3)


with w1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                01 · Enter Information
            </div>

            <div class="info-text">
                Enter age, BMI, smoking status,
                children, gender and region.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with w2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                02 · Process Data
            </div>

            <div class="info-text">
                Numerical and categorical features
                are processed by the ML pipeline.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with w3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                03 · Generate Prediction
            </div>

            <div class="info-text">
                The trained regression model
                estimates annual insurance charges.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("<hr>", unsafe_allow_html=True)

st.warning(
    "⚠️ **Disclaimer:** This application is an educational "
    "machine learning project. The prediction is an estimate "
    "and should not be treated as an actual insurance quote "
    "or professional advice."
)


st.markdown(
    """
    <div class="footer">
        Medical Insurance Cost Prediction · Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
