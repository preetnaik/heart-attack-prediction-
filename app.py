# app.py
import streamlit as st
import numpy as np
import joblib

# Load the trained Logistic Regression model
model = joblib.load("heart_models.pkl")["LogisticRegression"]

st.set_page_config(page_title="Heart Risk Predictor", page_icon="💓")
st.title("💓 Heart Attack Risk Prediction")
st.markdown("Enter the patient's health data to assess risk of a heart attack.")

# Input form
age = st.number_input("Age", 0, 120, 50)
sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
exang = st.radio("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of the ST Segment", [0, 1, 2])

# Collect features
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope]])

# Prediction button
if st.button("Predict Risk"):
    prediction = model.predict(features)[0]
    outcome = "💔 Risk of Heart Attack" if prediction == 1 else "✅ No Risk"
    st.success(f"Prediction: {outcome}")
