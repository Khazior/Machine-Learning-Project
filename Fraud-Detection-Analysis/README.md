# 🔎 Fraud Detection System

A Machine Learning web application to detect fraudulent transactions based on customer behavior patterns. This project utilizes clustering and classification techniques to identify anomalies in financial transactions.

🔗 **Live Demo:** [Click Here to Try the App](https://fraud-detection-applications.streamlit.app/)

![App Screenshot]
<img width="1919" height="980" alt="image" src="https://github.com/user-attachments/assets/85671b01-b4d4-403c-bb5a-3e1e7db0911a" />
<img width="1919" height="519" alt="image" src="https://github.com/user-attachments/assets/bd8435ac-9223-4df9-b275-e9cc72e9b35b" />
<img width="1919" height="557" alt="image" src="https://github.com/user-attachments/assets/aaf65c6f-2052-487f-824d-c5f6c4cc6aa2" />
<img width="1919" height="980" alt="image" src="https://github.com/user-attachments/assets/0f012fb0-dc49-4439-9784-5349e0ca2c44" />



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
