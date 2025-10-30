import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Trained Model
# -----------------------------
model = pickle.load(open('model.pkl', 'rb'))

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.markdown("<h1 style='text-align:center;'>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.write("### Enter details to check the risk of heart disease")

# -----------------------------
# Input Fields with Range Limits
# -----------------------------
age = st.number_input('Age', min_value=1, max_value=120, value=40)
cp = st.selectbox('Chest Pain Type (0–3)', [0,1,2,3])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', [0,1])
oldpeak = st.number_input('ST Depression (Oldpeak)', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox('Slope of ST Segment (0–2)', [0,1,2])
ca = st.selectbox('Number of Major Vessels (0–3)', [0,1,2,3])
thal = st.selectbox('Thalassemia (1: Normal, 2: Fixed, 3: Reversible)', [1,2,3])

# -----------------------------
# Input Validation
# -----------------------------
warning_flag = False
if chol < 120:
    st.warning("⚠️ Cholesterol value is below normal range (should be ≥ 120 mg/dl).")
    warning_flag = True
if trestbps < 90:
    st.warning("⚠️ Resting Blood Pressure is too low (should be ≥ 90 mm Hg).")
    warning_flag = True
if thalach < 80:
    st.warning("⚠️ Maximum Heart Rate is unusually low (should be ≥ 80 bpm).")
    warning_flag = True
if oldpeak < 0:
    st.warning("⚠️ ST Depression cannot be negative.")
    warning_flag = True

# -----------------------------
# Prediction Logic
# -----------------------------
features = np.array([[age, cp, trestbps, chol, thalach, exang, oldpeak, slope, ca, thal]])

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

if st.button('🔍 Predict Heart Disease'):
    if warning_flag:
        st.error("⚠️ Please correct the highlighted input values before prediction.")
    else:
        prediction = model.predict(features_scaled)
        if prediction[0] == 1:
            st.error("🚨 The person is likely to have **Heart Disease.**")
        else:
            st.success("✅ The person is **Healthy** — no sign of heart disease.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'>Developed by <b>Sanjana 💻</b> | Powered by scikit-learn & Streamlit 🧠</div>", unsafe_allow_html=True)
