# 🔎 Fraud Detection System

A Machine Learning web application to detect fraudulent transactions based on customer behavior patterns. This project utilizes clustering and classification techniques to identify anomalies in financial transactions.

🔗 **Live Demo:** [Click Here to Try the App](https://fraud-clustering-and-classification.streamlit.app/)

![App Screenshot]<img width="1919" height="985" alt="image" src="https://github.com/user-attachments/assets/407126c8-97da-4bc2-a924-e5c74993c327" />
<img width="1919" height="977" alt="image" src="https://github.com/user-attachments/assets/eaf03415-759f-46a4-835b-8b57ca3c7e95" />
<img width="1919" height="980" alt="image" src="https://github.com/user-attachments/assets/a28d8aac-c5dc-4c2c-91cb-0bf7539f91bd" />


## 📋 About the Project
Financial fraud is a critical issue in the banking sector. This application helps identify suspicious transactions by analyzing features such as:
- Transaction Amount & Duration
- Customer Age & Location
- Login Attempts
- Account Balance

The system classifies transactions into **Fraud (High Risk)** or **Non-Fraud (Normal)** and provides actionable recommendations (Hold, Verify, or Block).

## 🛠️ Tech Stack
- **Python**: Core language.
- **Streamlit**: Web application framework.
- **Scikit-Learn**: Machine Learning model building.
- **Pandas & NumPy**: Data manipulation.
- **Joblib**: Model serialization.

## 📂 Features
1.  **Input Manual Simulation**: Input single transaction details to test the model logic.
2.  **Batch Processing**: Upload CSV files containing bulk transaction data.
3.  **Auto-Inverse Scaling**: Automatically detects if uploaded data is normalized (numerical) or original, and converts it back to a human-readable format before prediction.
4.  **Actionable Insights**: Provides specific SOP recommendations based on the prediction result.

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/Khazior/Machine-Learning-Project.git](https://github.com/Khazior/Machine-Learning-Project.git)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
