
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("insurance.csv")

# Features and target
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]

# Feature types
numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("numeric", "passthrough", numeric_features),
    ("categorical",
     OneHotEncoder(drop="first", handle_unknown="ignore"),
     categorical_features)
])

# ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model.fit(X_train, y_train)

# Prediction function
def predict_insurance_cost(age, bmi, children, gender, smoker, region):

    input_data = pd.DataFrame({
        "age": [float(age)],
        "sex": [gender],
        "bmi": [float(bmi)],
        "children": [int(children)],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)

    return f"${prediction:,.2f}"

# Interface
app = gr.Interface(
    fn=predict_insurance_cost,
    inputs=[
        gr.Number(label="Age", value=25),
        gr.Number(label="BMI", value=23.5),
        gr.Number(label="Number of Children", value=0),
        gr.Dropdown(["Female", "Male"], label="Gender", value="Female"),
        gr.Dropdown(["No", "Yes"], label="Smoking Status", value="No"),
        gr.Dropdown(
            ["Northeast", "Northwest", "Southeast", "Southwest"],
            label="Region",
            value="Southeast"
        )
    ],
    outputs=gr.Textbox(label="Estimated Insurance Cost"),
    title="Medical Insurance Cost Predictor",
    description="Estimate medical insurance cost using Multiple Linear Regression."
)

import os

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
