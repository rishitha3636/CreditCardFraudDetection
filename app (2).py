import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("credit_fraud_model.pkl")

model = load_model()

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Upload transaction data to check for fraudulent activity and visualize results.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data (first 5 rows)")
    st.dataframe(data.head())

    # Expected columns from training
    expected_cols = ['Time', 'Amount'] + [f"V{i}" for i in range(1, 29)] + ['Hour', 'Transaction_Frequency']

    # Add Hour feature if missing
    if 'Hour' not in data.columns:
        data['Hour'] = (data['Time'] // 3600) % 24

    # Add Transaction_Frequency feature if missing
    if 'Transaction_Frequency' not in data.columns:
        data['Transaction_Frequency'] = 1  # Placeholder, adjust logic if needed

    # Run prediction
    if all(col in data.columns for col in expected_cols):
        predictions = model.predict(data[expected_cols])
        data['Prediction'] = ['Fraud' if x == 1 else 'Legit' for x in predictions]

        st.subheader("✅ Prediction Results")
        st.dataframe(data[['Time', 'Amount', 'Prediction']].head(20))

        # Download results
        csv = data.to_csv(index=False)
        st.download_button("💾 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

        # ---- Dashboard Visuals ----
        st.subheader("📊 Dashboard Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Fraud vs Legit Transactions")
            counts = data['Prediction'].value_counts()
            fig, ax = plt.subplots()
            counts.plot(kind="bar", color=["green", "red"], ax=ax)
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col2:
            st.write("Transaction Amount Distribution (Fraud vs Legit)")
            fig, ax = plt.subplots()
            sns.histplot(data=data, x="Amount", hue="Prediction", bins=50, log_scale=(False, True), ax=ax)
            st.pyplot(fig)

        st.write("Fraud Transactions Over Time")
        fig, ax = plt.subplots(figsize=(10,4))
        fraud_data = data[data['Prediction'] == 'Fraud']
        sns.histplot(fraud_data['Hour'], bins=24, kde=False, ax=ax, color="red")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Fraud Count")
        st.pyplot(fig)

    else:
        st.error(f"Missing required columns. Expected: {expected_cols}")

else:
    st.info("Please upload a CSV file to continue.")
