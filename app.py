import streamlit as st
import pandas as pd
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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp {
    background-color: #F7FBFC;
}

.block-container {
    max-width: 1120px;
    padding-top: 28px;
    padding-bottom: 40px;
}


/* ================= HERO ================= */

.hero {
    background: linear-gradient(
        135deg,
        #EAF8FA 0%,
        #F4FBFC 55%,
        #E5F5F8 100%
    );

    border-radius: 22px;
    padding: 35px 38px;
    border: 1px solid #D7EEF1;
    margin-bottom: 32px;
}

.hero-title {
    color: #123B5D;
    font-size: 38px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 12px;
}

.hero-text {
    color: #587083;
    font-size: 15px;
    line-height: 1.7;
    max-width: 600px;
}

.badge {
    display: inline-block;
    background: #D6F5EF;
    color: #087F73;
    padding: 7px 15px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 14px;
}


/* ================= SECTION ================= */

.section-title {
    color: #123B5D;
    font-size: 22px;
    font-weight: 750;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #718393;
    font-size: 14px;
    margin-bottom: 20px;
}


/* ================= CARDS ================= */

.card {
    background: white;
    border: 1px solid #E1E9EE;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 6px 20px rgba(18, 59, 93, 0.06);
}


/* ================= INPUTS ================= */

div[data-baseweb="select"] > div {
    border-radius: 10px;
}

input {
    border-radius: 10px !important;
}


/* ================= BUTTON ================= */

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 11px;
    border: none;

    background: #087F73;
    color: white;

    font-size: 16px;
    font-weight: 700;

    box-shadow: 0 6px 16px rgba(8,127,115,0.18);
}

.stButton > button:hover {
    background: #066E64;
    color: white;
}


/* ================= RESULT ================= */

.result-card {
    background: linear-gradient(
        135deg,
        #123B5D,
        #155B78
    );

    border-radius: 18px;
    padding: 30px;
    color: white;

    min-height: 190px;

    box-shadow: 0 10px 25px rgba(18,59,93,0.18);
}

.result-small {
    font-size: 14px;
    opacity: 0.8;
    margin-bottom: 8px;
}

.result-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 20px;
}

.result-value {
    font-size: 36px;
    font-weight: 800;
}


/* ================= INFO CARDS ================= */

.info {
    background: white;
    border: 1px solid #E1E9EE;
    border-radius: 16px;
    padding: 20px;
    min-height: 135px;
}

.info-title {
    color: #123B5D;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 8px;
}

.info-text {
    color: #718393;
    font-size: 13px;
    line-height: 1.6;
}


/* ================= DIVIDER ================= */

hr {
    border: none;
    border-top: 1px solid #DCE7EC;
    margin: 30px 0;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #8A9AA5;
    font-size: 12px;
    padding-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv("insurance.csv")

    return data


df = load_data()


# =========================================================
# MODEL TRAINING
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


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


model.fit(X_train, y_train)


# =========================================================
# MODEL EVALUATION
# =========================================================

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)


# =========================================================
# HERO
# =========================================================

hero_left, hero_right = st.columns(
    [1.55, 1],
    gap="large"
)


with hero_left:

    st.markdown(
        '<div class="badge">✦ ML Powered Healthcare Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-title">'
        'Medical Insurance Cost<br>'
        'Predictor'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-text">'
        'Estimate medical insurance costs using a machine learning '
        'model trained on historical healthcare data. '
        'Get a quick estimate based on personal information.'
        '</div>',
        unsafe_allow_html=True
    )


with hero_right:

    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=900&q=80",
        use_container_width=True
    )


# =========================================================
# MAIN AREA
# =========================================================

st.markdown(
    '<div class="section-title">👤 Personal Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Please enter your details to estimate the insurance cost.'
    '</div>',
    unsafe_allow_html=True
)


left, right = st.columns(
    [1.05, 0.95],
    gap="large"
)


# =========================================================
# INPUT CARD
# =========================================================

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        "### Personal Information"
    )

    st.caption(
        "Enter patient information below."
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

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PREDICTION CARD
# =========================================================

with right:

    st.markdown(
        "### 💰 Prediction"
    )

    st.caption(
        "Your estimated annual medical insurance cost."
    )


    if predict:

        input_data = pd.DataFrame(
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
            input_data
        )[0]


        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-small">
                    Prediction Result
                </div>

                <div class="result-title">
                    Estimated Annual Medical Insurance Cost
                </div>

                <div class="result-value">
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

                <div class="result-small">
                    Prediction Result
                </div>

                <div class="result-title">
                    Your prediction will appear here
                </div>

                <div style="
                    font-size:15px;
                    opacity:0.75;
                    line-height:1.6;
                ">
                    Enter the patient information and
                    click the prediction button.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# MODEL SUMMARY
# =========================================================

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">📊 Model Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Key information about the machine learning model.'
    '</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        """
        <div class="info">
            <div class="info-title">🤖 Algorithm</div>
            <div class="info-text">
                Multiple Linear Regression
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="info">
            <div class="info-title">📈 R² Score</div>
            <div class="info-text">
                {r2:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="info">
            <div class="info-title">📉 MAE</div>
            <div class="info-text">
                ${mae:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        """
        <div class="info">
            <div class="info-title">📚 Dataset</div>
            <div class="info-text">
                Medical Insurance Dataset
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'A simple machine learning workflow from input to prediction.'
    '</div>',
    unsafe_allow_html=True
)


a, b, c = st.columns(3)


with a:

    st.markdown(
        """
        <div class="info">
            <div class="info-title">
                01 · Enter Information
            </div>

            <div class="info-text">
                Provide age, BMI, smoking status,
                number of children, gender and region.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with b:

    st.markdown(
        """
        <div class="info">
            <div class="info-title">
                02 · Process Data
            </div>

            <div class="info-text">
                Numerical and categorical features
                are processed through the ML pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c:

    st.markdown(
        """
        <div class="info">
            <div class="info-title">
                03 · Generate Prediction
            </div>

            <div class="info-text">
                The trained regression model estimates
                the annual insurance charges.
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
    "machine learning project. The predicted amount is an "
    "estimate and is not an actual insurance quotation "
    "or professional medical advice."
)


st.markdown(
    '<div class="footer">'
    'Medical Insurance Cost Prediction • Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)
