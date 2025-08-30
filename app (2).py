#app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("credit_fraud_model.pkl")

model = load_model()

st.title("💳 Credit Card Fraud Detection App")
st.write("Upload transaction data to check for fraudulent activity.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Ensure required columns
    expected_cols = ['Time', 'Amount'] + [f"V{i}" for i in range(1, 29)] + ['Hour', 'Transaction_Frequency']

    # Add Hour feature if missing
    if 'Hour' not in data.columns:
        data['Hour'] = (data['Time'] // 3600) % 24

    # Add Transaction_Frequency feature if missing
    if 'Transaction_Frequency' not in data.columns:
        # Placeholder logic – adjust as per training preprocessing
        data['Transaction_Frequency'] = 1  

    # Check if all expected columns exist
    if all(col in data.columns for col in expected_cols):
        if st.button("Predict"):
            predictions = model.predict(data[expected_cols])
            data['Prediction'] = ['Fraud' if x == 1 else 'Legit' for x in predictions]

            st.subheader("Prediction Results")
            st.dataframe(data)

            # Download results
            csv = data.to_csv(index=False)
            st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
    else:
        st.error(f"Missing required columns. Expected: {expected_cols}")

else:
    st.info("Please upload a CSV file with transaction data.")

# Optional: Single transaction input form
st.subheader("🔎 Check a Single Transaction")

with st.form("single_transaction"):
    time_val = st.number_input("Time (seconds since first transaction)", min_value=0, step=1)
    amount_val = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    v_features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]
    hour_val = (time_val // 3600) % 24
    freq_val = st.number_input("Transaction Frequency (default=1)", min_value=1, value=1)

    submitted = st.form_submit_button("Check Transaction")

    if submitted:
        single_df = pd.DataFrame([[time_val, amount_val] + v_features + [hour_val, freq_val]],
                                 columns=['Time', 'Amount'] + [f"V{i}" for i in range(1, 29)] + ['Hour', 'Transaction_Frequency'])

        prediction = model.predict(single_df)[0]
        result = "Fraud ❌" if prediction == 1 else "Legit ✅"

        st.success(f"Prediction: {result}")
