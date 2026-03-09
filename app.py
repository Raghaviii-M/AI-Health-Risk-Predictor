import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")

st.title("AI Health Risk Predictor")

st.write("Predict patient health test result risk")

# INPUTS
age = st.slider("Age",1,100)

gender = st.selectbox("Gender",["Male","Female"])

blood = st.selectbox("Blood Type",["A+","A-","B+","B-","AB+","AB-","O+","O-"])

condition = st.selectbox("Medical Condition",
["Diabetes","Hypertension","Cancer","Asthma","Obesity","Heart Disease"])

admission = st.selectbox("Admission Type",
["Emergency","Urgent","Elective"])

billing = st.number_input("Billing Amount")

# Encoding

gender_val = 1 if gender=="Male" else 0

blood_types = ["A+","A-","B+","B-","AB+","AB-","O+","O-"]
blood_val = blood_types.index(blood)

conditions = ["Diabetes","Hypertension","Cancer","Asthma","Obesity","Heart Disease"]
condition_val = conditions.index(condition)

admissions = ["Emergency","Urgent","Elective"]
admission_val = admissions.index(admission)

if st.button("Predict Test Result"):

    data = np.array([[age,gender_val,blood_val,condition_val,admission_val,billing]])

    prediction = model.predict(data)

    probability = model.predict_proba(data)

    confidence = np.max(probability)*100

    if prediction[0] == 0:

        st.success("Test Result: NORMAL")

        st.write("Confidence Score:",round(confidence,2),"%")

        st.info("Suggestion: Maintain healthy lifestyle and regular checkups")

    elif prediction[0] == 1:

        st.error("Test Result: ABNORMAL")

        st.write("Confidence Score:",round(confidence,2),"%")

        st.warning("Suggestion: Consult doctor, monitor health, follow medication")

    else:

        st.warning("Test Result: INCONCLUSIVE")

        st.write("Confidence Score:",round(confidence,2),"%")

        st.info("Suggestion: Further medical tests recommended")


# PERSONAL HEALTH DASHBOARD


st.header("Patient Health Dashboard")

df = pd.read_csv("dataset.csv")

# Risk Score Calculation
risk_score = (age/100)*30 + (billing/100000)*40 + (condition_val/5)*30

st.subheader("Health Risk Score")

st.progress(int(risk_score))

st.write("Risk Score:", round(risk_score,2),"%")


# AGE RISK TREND CHART

st.subheader("Age vs Risk Trend")

age_range = list(range(1,100))

risk_trend = [(a/100)*30 + (billing/100000)*40 + (condition_val/5)*30 for a in age_range]

fig, ax = plt.subplots()

ax.plot(age_range, risk_trend)

ax.axvline(age, linestyle="--")

ax.set_xlabel("Age")

ax.set_ylabel("Risk Level")

ax.set_title("Health Risk Trend by Age")

st.pyplot(fig)


