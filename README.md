# Salary-prediction
💼 Smart Salary Prediction System | Machine Learning project using Random Forest and Streamlit to predict AI &amp; Data Science salaries.
# 💼 Smart Salary Prediction System

A Machine Learning web application that predicts AI & Data Science salaries based on experience, employment type, job role, company location, company size, remote work ratio, and work year.

## 🚀 Features
- Predict salary using Machine Learning
- Random Forest Regression model
- Interactive Streamlit web application
- Automatic categorical encoding using LabelEncoder
- User-friendly dropdown interface
- Real-time salary prediction

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- Streamlit
- Joblib

## 📊 Machine Learning Workflow
- Data Collection
- Data Cleaning
- Label Encoding
- Train-Test Split
- Model Training
- Model Evaluation
- Model Deployment

## 📈 Model Performance
- **Algorithm:** Random Forest Regressor
- **MAE:** 27,727.77
- **RMSE:** 41,329.93
- **R² Score:** 0.5543

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
smart-salary-prediction/
│── app.py
│── train_model.py
│── salary_model.pkl
│── encoders.pkl
│── requirements.txt
│── README.md
│── data/
│   └── ds_salaries.csv
```

