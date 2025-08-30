# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("credit_fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection App")
st.write("Detect if a transaction is Fraudulent or Legitimate")

option = st.radio("Select Input Type", ["Manual Entry", "Upload CSV"])

if option == "Manual Entry":
    st.subheader("Enter Transaction Details")

    time = st.number_input("Time (standardized)", value=0.0)
    amount = st.number_input("Amount (standardized)", value=0.0)
    
    feature_inputs = []
    for i in range(1, 29):
        val = st.number_input(f"V{i}", value=0.0)
        feature_inputs.append(val)
    
    input_data = [[time, amount] + feature_inputs]
    
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success("🚨 Fraud Transaction" if prediction == 1 else "✅ Legitimate Transaction")

else:
    st.subheader("Upload a CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Check if all required columns are present
        expected_cols = ['Time', 'Amount'] + [f"V{i}" for i in range(1, 29)]
        if all(col in data.columns for col in expected_cols):
            if st.button("Predict"):
                predictions = model.predict(data[expected_cols])
                data['Prediction'] = ['Fraud' if x == 1 else 'Legit' for x in predictions]
                st.dataframe(data)
                csv = data.to_csv(index=False)
                st.download_button("Download Results", csv, "results.csv", "text/csv")
        else:
            st.warning("Uploaded CSV must contain Time, Amount, and V1 to V28 columns.")
