# Medical Insurance Cost Prediction

## Project Overview

This project is a Machine Learning based Medical Insurance Cost Prediction system.
The objective is to predict estimated medical insurance charges based on personal
and demographic information.

The project follows a complete Machine Learning workflow including data loading,
data preprocessing, exploratory data analysis, model training, evaluation,
prediction, and deployment.

---

## Problem Statement

Medical insurance costs can vary depending on several factors such as age, BMI,
smoking status, gender, number of children, and geographical region.

The goal of this project is to develop a regression model that can estimate
medical insurance charges using these input features.

---

## Dataset

The project uses a Medical Cost Personal Dataset containing information about
individuals and their medical insurance charges.

### Dataset Features

| Feature | Description |
|---|---|
| Age | Age of the individual |
| Sex | Gender of the individual |
| BMI | Body Mass Index |
| Children | Number of children/dependents |
| Smoker | Smoking status |
| Region | Residential region |
| Charges | Medical insurance cost |

The target variable is **charges**.

---

## Machine Learning Workflow

The project follows these main steps:

```text
Dataset
   ↓
Data Cleaning
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Linear Regression Model
   ↓
Model Evaluation
   ↓
Prediction Application
   ↓
Deployment
Data Preprocessing

The following preprocessing steps were performed:

The dataset was loaded using Pandas.
Dataset structure and data types were inspected.
Missing values were checked.
Duplicate records were identified and removed.
Numerical and categorical features were identified.
Categorical variables were converted into numerical representations.
One-hot encoding was applied to categorical features in the final ML pipeline.
The dataset was divided into training and testing sets.

The final application uses a Scikit-learn preprocessing and regression pipeline.

Exploratory Data Analysis

Several visualizations were created to understand the relationships between
features and medical insurance charges.

Visualizations
Age vs Insurance Charges
BMI vs Insurance Charges
Smoking Status vs Insurance Charges
Correlation Analysis
Correlation Heatmap
Important Findings
Insurance charges generally show an increasing trend with age.
BMI shows a relationship with insurance charges, although the values are
relatively scattered.
Smokers generally have considerably higher insurance charges than non-smokers.
Age and smoking status appear to be particularly useful features for predicting
insurance charges.
Selected Machine Learning Model
Multiple Linear Regression

The selected model is Multiple Linear Regression.

Linear Regression is a supervised machine learning algorithm used for predicting
continuous numerical values.

In this project, the model learns the relationship between the input features
and the target variable charges.

Input Features
Age
Sex
BMI
Children
Smoker
Region
Target Variable
charges
Training Process

The dataset was divided into training and testing sets using:

Training data: 80%
Testing data: 20%
Random state: 42

The model was trained using the training dataset and then used to generate
predictions for unseen test data.

A Scikit-learn Pipeline was used to combine preprocessing and the
Multiple Linear Regression model.

Model Evaluation

The model was evaluated using the following regression metrics:

Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted charges.

Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted charges.

Root Mean Squared Error (RMSE)

Represents the square root of MSE and gives the error in the same unit as
the target variable.

R² Score

Measures how much of the variation in insurance charges is explained by the
model.

The actual numerical evaluation results are available in the project notebook.

Prediction Application

A web-based prediction application was developed using Gradio.

The user can enter:

Age
BMI
Number of Children
Gender
Smoking Status
Region

The application then generates an estimated annual medical insurance cost.

Application Workflow
User Input
    ↓
Preprocessing
    ↓
Trained Linear Regression Model
    ↓
Prediction
    ↓
Estimated Insurance Cost
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Gradio
Google Colab
GitHub
Project Structure
Medical-Insurance-Cost-Prediction/
│
├── Medical_Insurance_Cost_Prediction_Phase(1-8).ipynb
├── app.py
├── insurance.csv
├── requirements.txt
└── README.md
Deployment

The Machine Learning prediction application is prepared for deployment using
Gradio.

The deployed application link will be added here after deployment.

Live Application: Coming Soon

Screenshots
GitHub Repository

Project files and source code are available in this repository.

Prediction Application

The application provides an interactive interface where users can enter
personal information and receive an estimated insurance cost.

Working Prediction

A sample prediction is generated after entering the required information.

Screenshots will be added after deployment.

Limitations
The model provides an estimated insurance cost and not an actual insurance quotation.
Prediction quality depends on the dataset used for training.
The dataset may not represent current insurance pricing in every location.
Real-world insurance costs can depend on additional factors that are not included
in this dataset.
The model is intended for educational and machine learning demonstration purposes.
Future Improvements

Possible improvements include:

Comparing multiple regression algorithms.
Hyperparameter tuning.
Feature engineering.
Using a larger and more diverse dataset.
Adding additional relevant healthcare and demographic features.
Improving the prediction interface.
Monitoring model performance after deployment.
Conclusion

This project demonstrates a complete Machine Learning workflow for predicting
medical insurance costs.

The project covers data preprocessing, exploratory data analysis, supervised
learning, model evaluation, and development of an interactive prediction
application using Gradio.

The final application allows users to enter personal information and receive an
estimated medical insurance cost.

Disclaimer

This project is developed for educational and Machine Learning demonstration
purposes. The predicted value should not be considered an actual insurance
quotation or professional financial or medical advice.
