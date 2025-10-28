import streamlit as st
import pickle
import json
import numpy as np

# -----------------------------
# Load Model and Artifacts
# -----------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        padding: 2rem;
    }
    .stButton button {
        background-color: #2e8b57;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #228b22;
        color: white;
    }
    .result-box {
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title & Description
# -----------------------------
st.markdown("<h1 style='text-align:center;'>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter patient details to check the possibility of heart disease.</p>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Input Fields
# -----------------------------
col1, col2 = st.columns(2)
inputs = {}
    inputs['age'] = st.number_input("Age", min_value=0, max_value=120, value=45)
    inputs['sex'] = st.selectbox("Sex", ["Male", "Female"])
    inputs['cp'] = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    inputs['trestbps'] = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    inputs['chol'] = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=220)
    inputs['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    inputs['restecg'] = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
    inputs['thalach'] = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    inputs['exang'] = st.selectbox("Exercise Induced Angina", [0, 1])
    inputs['oldpeak'] = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    inputs['slope'] = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
    inputs['ca'] = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    inputs['thal'] = st.selectbox("Thalassemia (0=normal, 1=fixed, 2=reversible)", [0, 1, 2])

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("🔍 Predict"):
    try:
        # Convert inputs to correct format
        input_order = feature_columns
        X = np.array([inputs[col] if col in inputs else 0 for col in input_order]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1] * 100

        st.markdown("<div class='result-box' style='background-color:#e3f2fd;'><b>Prediction Value:</b> {}</div>".format(prediction), unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("<div class='result-box' style='background-color:#ffebee; color:#d32f2f;'><b>⚠️ Heart Disease Detected!</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box' style='background-color:#e8f5e9; color:#2e7d32;'><b>✅ No Heart Disease Detected.</b></div>", unsafe_allow_html=True)

        st.markdown(f"<div class='result-box' style='background-color:#f3e5f5;'><b>Probability:</b> {prob:.2f}%</div>", unsafe_allow_html=True)

        risk_msg = "🟢 **Low Risk** – Continue healthy habits." if prob < 40 else \
                   "🟡 **Moderate Risk** – Consider lifestyle changes." if prob < 70 else \
                   "🔴 **High Risk** – Please consult a cardiologist."
        st.markdown(f"<div class='result-box' style='background-color:#e3f2fd;'>{risk_msg}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error occurred: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;'>
Developed by <b>Sanjana 💻</b> | Powered by <b>scikit-learn</b> & <b>Streamlit</b> 🧠 | ❤️ Machine Learning
</div>
""", unsafe_allow_html=True)

