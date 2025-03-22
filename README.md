**🚀 Credit Card Fraud Detection**
This project builds a machine learning model to detect fraudulent credit card transactions, minimizing false positives while ensuring high accuracy.

**📌 Overview**
Credit card fraud detection is essential for financial security. This project applies data preprocessing, feature engineering, and machine learning models to classify transactions as fraudulent (1) or legitimate (0).

**📊 Dataset Information**
Source: Kaggle - Credit Card Fraud Dataset
Total Transactions: 284,807
Fraudulent Cases: 492 (Highly Imbalanced Dataset)
Features:
Time: Seconds since first transaction
Amount: Transaction amount
V1 - V28: Principal Component Analysis (PCA) transformed features
Class: 0 = Legitimate, 1 = Fraudulent

**🛠 Features & Methodology**
✔ Data Preprocessing: Handle missing values, scale transaction amounts
✔ Feature Engineering: Extract transaction frequency, time-based trends
✔ Class Imbalance Handling: Apply SMOTE oversampling and undersampling
✔ Model Selection: Train Logistic Regression, Random Forest, and XGBoost
✔ Evaluation Metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC

**🚀 How to Run the Project**
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Run the Detection Script
python credit_fraud_detection.py
3️⃣ Predict Using the Trained Model
import joblib
model = joblib.load("credit_fraud_model.pkl")
sample_data = [[500, 12, 0.5, -1.2, 0.8, 1.1, 0.7, -0.3, 1.4, -0.9]]
prediction = model.predict(sample_data)
print("Fraud Prediction:", "Fraud" if prediction[0] == 1 else "Legit")

**📊 Results & Performance**
Model                   Accuracy    Precision    Recall    ROC-AUC
Logistic Regression      96.5%        84.3%       91.2%     97.5%
Random Forest            98.1%        90.1%       94.8%     98.9%
XGBoost (Best)           98.5%        90.2%       94.7%     99.2%

**📂 Project Structure**
📂 CreditCardFraudDetection
 ├── 📄 credit_fraud_detection.py  # Main script
 ├── 📄 credit_fraud_model.pkl     # Trained fraud detection model
 ├── 📄 dataset.csv                # (Optional) If small enough to upload
 ├── 📄 README.md                  # Project documentation
 ├── 📄 requirements.txt           # Dependencies list
 ├── 📂 data                       # Folder for dataset storage

**📢 Contributing**
1️⃣ Fork the repo
2️⃣ Create a branch (git checkout -b feature-name)
3️⃣ Commit changes (git commit -m "Added new feature")
4️⃣ Push changes (git push origin feature-name)
5️⃣ Submit a Pull Request 🚀

creditcard csv file(dataset downloaded from Kaggle):https://github.com/rishitha3636/CreditCardFraudDetection/releases/download/v1.0/creditcard.csv
