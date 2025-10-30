import streamlit as st
import pickle
import json
import numpy as np

# -----------------------------
# Load Model and Artifacts
# -----------------------------
with open("best_model.pkl", "rb") as f:  # ✅ Corrected filename
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #ffffff; padding: 2rem; }
    .stButton button {
        background-color: #e91e63; color: white; font-weight: bold;
        border-radius: 8px; padding: 0.6rem 1.2rem;
    }
    .stButton button:hover { background-color: #ad1457; color: white; }
    .result-box {
        border-radius: 10px; padding: 15px; margin-top: 15px; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter details to predict the risk of heart disease.</p>", unsafe_allow_html=True)

# -----------------------------
# Important Features
# -----------------------------
st.subheader("🩺 Enter Patient Details")

inputs = {}
inputs['age'] = st.number_input("Age", min_value=20, max_value=100, value=45)
inputs['sex'] = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
inputs['cp'] = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
inputs['trestbps'] = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
inputs['chol'] = st.number_input("Cholesterol (mg/dl)", min_value=120, max_value=600, value=200)
inputs['thalach'] = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
inputs['exang'] = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
inputs['oldpeak'] = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
inputs['slope'] = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
inputs['ca'] = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
inputs['thal'] = st.selectbox("Thalassemia (0=Normal, 1=Fixed, 2=Reversible)", [0, 1, 2])

# -----------------------------
# Validation Check
# -----------------------------
if inputs['chol'] < 120:
    st.warning("⚠️ Cholesterol value is too low! Please enter at least 120 mg/dl.")
elif inputs['trestbps'] < 90:
    st.warning("⚠️ Resting Blood Pressure is too low! Enter a value above 90 mm Hg.")

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("🔍 Predict"):
    try:
        X = np.array([inputs[col] if col in inputs else 0 for col in feature_columns]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1] * 100

        st.markdown(f"<div class='result-box' style='background-color:#e3f2fd;'><b>Prediction Value:</b> {prediction}</div>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("<div class='result-box' style='background-color:#ffebee; color:#d32f2f;'><b>⚠️ Heart Disease Detected!</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box' style='background-color:#e8f5e9; color:#2e7d32;'><b>✅ No Heart Disease Detected.</b></div>", unsafe_allow_html=True)

        st.markdown(f"<div class='result-box' style='background-color:#ede7f6;'><b>Probability:</b> {prob:.2f}%</div>", unsafe_allow_html=True)

        if prob < 40:
            st.info("🟢 **Low Risk** – Keep up the healthy lifestyle.")
        elif prob < 70:
            st.warning("🟡 **Moderate Risk** – Consider lifestyle improvements.")
        else:
            st.error("🔴 **High Risk** – Please consult a cardiologist soon.")

    except Exception as e:
        st.error(f"Error occurred: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'>Developed by <b>Sanjana 💻</b> | Powered by <b>Streamlit</b> & <b>scikit-learn</b> 🧠</div>", unsafe_allow_html=True)
