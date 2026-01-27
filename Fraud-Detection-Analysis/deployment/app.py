import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. SETUP & DATA STATISTICS
# ==========================================
st.set_page_config(page_title="Fraud Detection System", page_icon="🚀", layout="wide")

# Load Model
try:
    # Ensure this path matches your GitHub repository structure
    model = joblib.load('analisa-ml-dicoding/deployment/tuning_classification.h5')
except FileNotFoundError:
    st.error("❌ File 'analisa-ml-dicoding/deployment/tuning_classification.h5' not found!")
    st.stop()

# --- STATISTICS FOR INVERSE SCALING (NUMERICAL) ---
STATS = {
    'TransactionAmount': {'mean': 256.84, 'std': 218.37},
    'CustomerAge':       {'mean': 44.69, 'std': 17.74},
    'TransactionDuration': {'mean': 119.22, 'std': 70.60},
    'LoginAttempts':     {'mean': 1.0, 'std': 0.0},
    'AccountBalance':    {'mean': 5100.81, 'std': 3907.15}
}

# --- COMPLETE CATEGORY MAPPING (NUMERICAL -> TEXT) ---
CAT_MAPPING = {
    'TransactionType': {1: 'Debit', 0: 'Credit'},
    'Channel': {0: 'ATM', 2: 'Online', 1: 'Branch'},
    'CustomerOccupation': {0: 'Doctor', 3: 'Student', 2: 'Retired', 1: 'Engineer'},
    'Location': {
        36: 'San Diego', 15: 'Houston', 23: 'Mesa', 33: 'Raleigh', 28: 'Oklahoma City', 
        39: 'Seattle', 16: 'Indianapolis', 11: 'Detroit', 26: 'Nashville', 0: 'Albuquerque', 
        22: 'Memphis', 21: 'Louisville', 10: 'Denver', 2: 'Austin', 8: 'Columbus', 
        20: 'Los Angeles', 19: 'Las Vegas', 25: 'Milwaukee', 24: 'Miami', 3: 'Baltimore', 
        37: 'San Francisco', 35: 'San Antonio', 30: 'Philadelphia', 5: 'Charlotte', 
        40: 'Tucson', 18: 'Kansas City', 41: 'Virginia Beach', 29: 'Omaha', 9: 'Dallas', 
        1: 'Atlanta', 4: 'Boston', 17: 'Jacksonville', 13: 'Fort Worth', 7: 'Colorado Springs', 
        34: 'Sacramento', 14: 'Fresno', 32: 'Portland', 42: 'Washington', 6: 'Chicago', 
        27: 'New York', 31: 'Phoenix', 38: 'San Jose', 12: 'El Paso'
    }
}

# Mandatory Columns for Model
EXPECTED_COLUMNS = [
    'TransactionAmount', 'CustomerAge', 'TransactionDuration', 'LoginAttempts', 'AccountBalance',
    'TransactionType_Debit', 'Location_Atlanta', 'Location_Austin', 'Location_Baltimore', 
    'Location_Boston', 'Location_Charlotte', 'Location_Chicago', 'Location_Colorado Springs', 
    'Location_Columbus', 'Location_Dallas', 'Location_Denver', 'Location_Detroit', 
    'Location_El Paso', 'Location_Fort Worth', 'Location_Fresno', 'Location_Houston', 
    'Location_Indianapolis', 'Location_Jacksonville', 'Location_Kansas City', 'Location_Las Vegas', 
    'Location_Los Angeles', 'Location_Louisville', 'Location_Memphis', 'Location_Mesa', 
    'Location_Miami', 'Location_Milwaukee', 'Location_Nashville', 'Location_New York', 
    'Location_Oklahoma City', 'Location_Omaha', 'Location_Philadelphia', 'Location_Phoenix', 
    'Location_Portland', 'Location_Raleigh', 'Location_Sacramento', 'Location_San Antonio', 
    'Location_San Diego', 'Location_San Francisco', 'Location_San Jose', 'Location_Seattle', 
    'Location_Tucson', 'Location_Virginia Beach', 'Location_Washington', 
    'Channel_Branch', 'Channel_Online', 
    'CustomerOccupation_Engineer', 'CustomerOccupation_Retired', 'CustomerOccupation_Student', 
    'AgeGroup_Old', 'AgeGroup_Young'
]

# ==========================================
# 2. LOGIC FUNCTIONS (BACKEND)
# ==========================================

def determine_age_group(age):
    if age <= 32: return 'Young'
    elif age <= 55: return 'Mature'
    else: return 'Old'

def inverse_transform_data(df, stats, mapping):
    """Converts normalized data back to original format."""
    df_inv = df.copy()
    
    # 1. Inverse Numerical
    for col, stat in stats.items():
        if col in df_inv.columns:
            if stat['std'] == 0:
                df_inv[col] = stat['mean']
            else:
                df_inv[col] = (df_inv[col] * stat['std']) + stat['mean']
            
            # Formatting
            if col == 'CustomerAge':
                df_inv[col] = df_inv[col].round().astype(int)
            elif col in ['TransactionAmount', 'AccountBalance']:
                df_inv[col] = df_inv[col].round(2)
            elif col == 'TransactionDuration':
                df_inv[col] = df_inv[col].round(0)
    
    # 2. Inverse Categorical
    for col, map_dict in mapping.items():
        if col in df_inv.columns:
            try:
                df_inv[col] = pd.to_numeric(df_inv[col], errors='coerce').fillna(0).astype(int)
                df_inv[col] = df_inv[col].map(map_dict)
            except:
                pass 
    return df_inv

def process_prediction_pipeline(df_input, is_normalized=False):
    # Step 1: Inverse if data is normalized
    if is_normalized:
        df_proc = inverse_transform_data(df_input, STATS, CAT_MAPPING)
    else:
        df_proc = df_input.copy()

    # Step 2: Handle Age Group
    if 'CustomerAge' in df_proc.columns:
        df_proc['AgeGroup'] = df_proc['CustomerAge'].apply(determine_age_group)

    # Step 3: One-Hot Encoding
    df_encoded = pd.get_dummies(df_proc)

    # Step 4: Reindex
    df_final = df_encoded.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    
    return df_final, df_proc

# ==========================================
# 3. USER INTERFACE DISPLAY
# ==========================================
st.title("Fraud vs Non-Fraud Detection 🔎")
st.write("Supports Auto-Normalization Detection & Automatic Labeling")

tab1, tab2 = st.tabs(["👤 Manual Input", "📂 Upload File (Batch)"])

# --- TAB 1: MANUAL INPUT ---
with tab1:
    st.write("Single customer data simulation for security detection.")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        balance = st.number_input("Account Balance", 0.0, 100000.0, 5000.0)
        amount = st.number_input("Transaction Amount", 0.0, 10000.0, 100.0)
        duration = st.number_input("Duration (Sec)", 0.0, 1000.0, 60.0)
        login = st.number_input("Login Attempts", 0, 10, 1)
    with col2:
        trans_type = st.selectbox("Transaction Type", ['Debit', 'Credit'])
        channel = st.selectbox("Channel", ['ATM', 'Online', 'Branch'])
        occupation = st.selectbox("Occupation", ['Doctor', 'Student', 'Retired', 'Engineer', 'Others'])
        # Display list of locations from mapping
        loc_options = sorted(list(CAT_MAPPING['Location'].values()))
        location = st.selectbox("Location", loc_options)
        
    if st.button("Predict"):
        data_row = pd.DataFrame({
            'TransactionAmount': [amount], 'CustomerAge': [age], 'TransactionDuration': [duration],
            'LoginAttempts': [login], 'AccountBalance': [balance], 'TransactionType': [trans_type],
            'Location': [location], 'Channel': [channel], 'CustomerOccupation': [occupation]
        })
        
        # Process & Predict
        X_pred, _ = process_prediction_pipeline(data_row, is_normalized=False)
        hasil = model.predict(X_pred)[0]
        
        # --- RESULT & RECOMMENDATION (FRAUD CONTEXT) ---
        st.divider()
        st.subheader("Security Analysis Result:")
        
        if hasil == 1:
            # CASE: FRAUD (DANGER - RED)
            st.error("### ⚠️ WARNING: Potential Fraud Detected!")
            
            

            st.markdown("""
            **Risk Analysis:**
            This transaction shows **anomalies** or patterns deviating from the customer's habitual profile. The system detects high-risk suspicious activity indicators.
            
            **Required Actions (SOP):**
            * ⛔ **HOLD Transaction:** Do not automatically approve this transaction.
            * 📞 **Manual Verification:** Contact the customer immediately via registered number for confirmation.
            * 🔒 **Temporary Block:** If the customer cannot be reached or does not recognize the transaction, freeze the account immediately.
            """)
        else:
            # CASE: NON-FRAUD (SAFE - GREEN)
            st.success("### ✅ SAFE: Normal Transaction (Non-Fraud)")
            
            

            st.markdown("""
            **Risk Analysis:**
            Transaction pattern is consistent with the historical profile of the customer (Genuine User). No indications of account misuse found.
            
            **Required Actions:**
            * ✅ **Approve:** Proceed with transaction processing.
            * ✅ **No Friction:** No additional verification (OTP/Call) required to maintain customer convenience.
            """)

# --- TAB 2: UPLOAD FILE (BATCH) ---
with tab2:
    st.header("Upload CSV File")
    st.write("Please upload a CSV file containing customer transaction data.")
    
    # NOTE: Radio button has been removed for auto-detection
    
    uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write("Initial Data Preview:")
        st.dataframe(df_upload.head())
        
        if st.button("Process & Batch Predict"):
            try:
                # --- AUTO-DETECT LOGIC ---
                # We check the average of the CustomerAge column.
                # Normalized data usually ranges from -2 to 2 (mean close to 0).
                # Original data age must be above 17 years.
                
                is_norm = False # Default assume original data
                
                if 'CustomerAge' in df_upload.columns:
                    mean_age = df_upload['CustomerAge'].mean()
                    if mean_age < 15: # Safe threshold
                        is_norm = True
                        st.info("ℹ️ System detected input is **Numerical Data (Normalized)**. Converting to Original Data...")
                    else:
                        is_norm = False
                        st.info("ℹ️ System detected input is **Original Data**.")
                
                # 1. Pipeline (Inverse automatically if is_norm=True)
                X_batch, df_clean = process_prediction_pipeline(df_upload, is_normalized=is_norm)
                
                # 2. Predict
                prediksi_raw = model.predict(X_batch)
                
                # 3. Merge Results
                df_hasil = df_clean.copy()
                df_hasil['Label_Cluster'] = pd.Series(prediksi_raw).map({1: 'Fraud', 0: 'Non-Fraud'})
                
                # 4. Show Success Info
                st.success("✅ Process Completed! Fraud/Non-Fraud classification successful.")
                
                st.write("Final Result Preview:")
                st.dataframe(df_hasil.head())
                
                # 5. Download
                csv = df_hasil.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Result (CSV)",
                    data=csv,
                    file_name="fraud_detection_result.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Ensure the CSV file has appropriate columns (TransactionAmount, CustomerAge, etc.).")