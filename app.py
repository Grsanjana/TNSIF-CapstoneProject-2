import streamlit as st
import pickle
import json
import numpy as np

# -----------------------------
# Load model and other files
# -----------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction App 💓", layout="centered")
st.title("💓 Heart Disease Risk Prediction")
st.markdown("This AI-powered app predicts the risk of heart disease based on your health details.")

st.divider()

# Create user inputs dynamically
input_data = []
st.subheader("🩺 Enter Patient Health Details")

for col in feature_columns:
    if "sex" in col.lower():
        val = st.selectbox(f"{col}", ["Male", "Female"])
        val = 1 if val == "Male" else 0

    elif "chest_pain" in col.lower():
        val = st.selectbox(f"{col}", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(val)

    elif "fasting_blood_sugar" in col.lower():
        val = st.selectbox(f"{col}", ["< 120 mg/dl", "> 120 mg/dl"])
        val = 0 if val == "< 120 mg/dl" else 1

    elif "rest_ecg" in col.lower():
        val = st.selectbox(f"{col}", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
        val = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(val)

    elif "exercise_induced_angina" in col.lower():
        val = st.selectbox(f"{col}", ["No", "Yes"])
        val = 1 if val == "Yes" else 0

    elif "thalassemia" in col.lower():
        val = st.selectbox(f"{col}", ["Normal", "Fixed Defect", "Reversible Defect"])
        val = ["Normal", "Fixed Defect", "Reversible Defect"].index(val)

    else:
        val = st.slider(f"{col}", min_value=0.0, max_value=10.0, step=0.1)

    input_data.append(val)

st.divider()

if st.button("🔍 Predict"):
    try:
        # Prepare input data
        X = np.array(input_data).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        # Get probability if model supports predict_proba
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_scaled)[0][1] * 100
        else:
            prob = 50  # fallback if model doesn't support proba

        st.subheader("🧩 Prediction Results")
        st.markdown(f"**Prediction Value:** `{int(prediction)}`")

        if prediction == 1:
            risk_text = "🚨 Heart Disease Detected!"
            risk_level = "High Risk – Please consult a doctor immediately."
            st.error(f"{risk_text}\n\n🧠 Probability: **{prob:.2f}%**\n💔 Risk Level: **{risk_level}**")
        else:
            risk_text = "✅ No Heart Disease Detected."
            risk_level = "Low Risk – Continue healthy habits."
            st.success(f"{risk_text}\n\n🧠 Probability: **{prob:.2f}%**\n💚 Risk Level: **{risk_level}**")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
