import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Smart Salary Predictor", page_icon="💼")

st.title("💼 Smart Salary Prediction System")
st.write("Predict AI & Data Science salaries using Machine Learning")

# Inputs
work_year = st.selectbox("Work Year", [2020, 2021, 2022, 2023])

experience = st.selectbox(
    "Experience Level",
    encoders["experience_level"].classes_
)

employment = st.selectbox(
    "Employment Type",
    encoders["employment_type"].classes_
)

job = st.selectbox(
    "Job Title",
    encoders["job_title"].classes_
)

residence = st.selectbox(
    "Employee Residence",
    encoders["employee_residence"].classes_
)

remote = st.selectbox(
    "Remote Ratio",
    [0, 50, 100]
)

company = st.selectbox(
    "Company Location",
    encoders["company_location"].classes_
)

size = st.selectbox(
    "Company Size",
    encoders["company_size"].classes_
)

if st.button("Predict Salary"):

    input_df = pd.DataFrame({
        "work_year": [work_year],
        "experience_level": [
            encoders["experience_level"].transform([experience])[0]
        ],
        "employment_type": [
            encoders["employment_type"].transform([employment])[0]
        ],
        "job_title": [
            encoders["job_title"].transform([job])[0]
        ],
        "employee_residence": [
            encoders["employee_residence"].transform([residence])[0]
        ],
        "remote_ratio": [remote],
        "company_location": [
            encoders["company_location"].transform([company])[0]
        ],
        "company_size": [
            encoders["company_size"].transform([size])[0]
        ]
    })

    prediction = model.predict(input_df)

    st.success(f"💰 Predicted Salary: ${prediction[0]:,.2f}")