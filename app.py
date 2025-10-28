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
st.title("Heart Disease Prediction App 💓")
st.write("Enter patient details below to predict heart disease risk.")

# Dynamically generate input fields
input_data = []
for col in feature_columns:
    val = st.number_input(f"{col}", step=0.1)
    input_data.append(val)

if st.button("Predict"):
    try:
        # Convert to numpy array
        X = np.array(input_data).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        
        if prediction == 1:
            st.error("⚠️ The model predicts a **high risk of heart disease.** Please consult a doctor.")
        else:
            st.success("✅ The model predicts a **low risk of heart disease.**")
    except Exception as e:
        st.write("Error:", e)

    
     
        
