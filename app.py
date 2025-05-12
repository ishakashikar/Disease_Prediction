# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Title and Description
st.set_page_config(page_title="Disease Prediction System")
st.title("🩺 Disease Prediction System")
st.write("""
This application predicts the risk of diabetes based on your health inputs. 
Please adjust the sliders below and click 'Predict' to get your results.
""")

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("diabetes.csv")
    return data

data = load_data()

# Prepare Data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# User Inputs
st.sidebar.header("User Input Features")

def user_input_features():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose Level", 0, 200, 100)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 80)
    bmi = st.sidebar.slider("BMI", 10.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 10, 100, 30)

    features = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    return pd.DataFrame([features])

input_df = user_input_features()

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ At Risk of Diabetes (Confidence: {prediction_proba:.2%})")
    else:
        st.success(f"✅ Not at Risk (Confidence: {1 - prediction_proba:.2%})")

# Accuracy Display
st.sidebar.markdown("---")
if st.sidebar.checkbox("Show Model Accuracy"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.sidebar.write("Model Accuracy on Test Set:", f"{acc:.2%}")
    st.sidebar.text(classification_report(y_test, y_pred))
