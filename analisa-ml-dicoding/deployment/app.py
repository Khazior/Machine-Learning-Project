import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. SETUP & DATA STATISTIK
# ==========================================
st.set_page_config(page_title="Prediksi Segmen Nasabah (Final)", page_icon="🚀", layout="wide")

# Load Model
try:
    model = joblib.load('analisa-ml-dicoding/deployment/tuning_classification.h5')
except FileNotFoundError:
    st.error("❌ File 'analisa-ml-dicoding/deployment/tuning_classification.h5' tidak ditemukan!")
    st.stop()

# --- STATISTIK UNTUK INVERSE SCALING (NUMERIK) ---
STATS = {
    'TransactionAmount': {'mean': 256.84, 'std': 218.37},
    'CustomerAge':       {'mean': 44.69, 'std': 17.74},
    'TransactionDuration': {'mean': 119.22, 'std': 70.60},
    'LoginAttempts':     {'mean': 1.0, 'std': 0.0},
    'AccountBalance':    {'mean': 5100.81, 'std': 3907.15}
}

# --- MAPPING KATEGORI LENGKAP (NUMERIK -> TEKS) ---
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

# Daftar Kolom Wajib Model
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
    'CustomerAgeGroup_Muda', 'CustomerAgeGroup_Tua'
]

# ==========================================
# 2. FUNGSI LOGIKA (BACKEND)
# ==========================================

def tentukan_age_group(age):
    if age <= 32: return 'Muda'
    elif age <= 55: return 'Dewasa'
    else: return 'Tua'

def inverse_transform_data(df, stats, mapping):
    """Mengubah data normalisasi kembali ke data asli."""
    df_inv = df.copy()
    
    # 1. Inverse Numerik
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

def proses_prediction_pipeline(df_input, is_normalized=False):
    # Step 1: Inverse jika data normalisasi
    if is_normalized:
        df_proc = inverse_transform_data(df_input, STATS, CAT_MAPPING)
    else:
        df_proc = df_input.copy()

    # Step 2: Handle Age Group
    if 'CustomerAge' in df_proc.columns:
        df_proc['CustomerAgeGroup'] = df_proc['CustomerAge'].apply(tentukan_age_group)

    # Step 3: One-Hot Encoding
    df_encoded = pd.get_dummies(df_proc)

    # Step 4: Reindex
    df_final = df_encoded.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    
    return df_final, df_proc

# ==========================================
# 3. TAMPILAN USER INTERFACE
# ==========================================
st.title("Mendeteksi Fraud vs Non-Fraud 🔎")
st.write("Mendukung Normalisasi & Labeling Otomatis")

tab1, tab2 = st.tabs(["👤 Input Manual", "📂 Upload File (Batch)"])

# --- TAB 1: INPUT MANUAL ---
with tab1:
    st.write("Simulasi data satu nasabah untuk deteksi keamanan.")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Umur", 18, 100, 30)
        balance = st.number_input("Saldo", 0.0, 100000.0, 5000.0)
        amount = st.number_input("Transaksi", 0.0, 10000.0, 100.0)
        duration = st.number_input("Durasi", 0.0, 1000.0, 60.0)
        login = st.number_input("Login Attempts", 0, 10, 1)
    with col2:
        trans_type = st.selectbox("Tipe", ['Debit', 'Credit'])
        channel = st.selectbox("Channel", ['ATM', 'Online', 'Branch'])
        occupation = st.selectbox("Pekerjaan", ['Doctor', 'Student', 'Retired', 'Engineer', 'Others'])
        # Tampilkan list lokasi dari mapping
        loc_options = sorted(list(CAT_MAPPING['Location'].values()))
        location = st.selectbox("Lokasi", loc_options)
        
    if st.button("Prediksi"):
        data_row = pd.DataFrame({
            'TransactionAmount': [amount], 'CustomerAge': [age], 'TransactionDuration': [duration],
            'LoginAttempts': [login], 'AccountBalance': [balance], 'TransactionType': [trans_type],
            'Location': [location], 'Channel': [channel], 'CustomerOccupation': [occupation]
        })
        
        # Proses & Prediksi
        X_pred, _ = proses_prediction_pipeline(data_row, is_normalized=False)
        hasil = model.predict(X_pred)[0]
        
        # --- HASIL & REKOMENDASI (KONTEKS FRAUD) ---
        st.divider()
        st.subheader("Hasil Analisis Keamanan:")
        
        if hasil == 1:
            # KASUS: FRAUD (BAHAYA - MERAH)
            st.error("### ⚠️ PERINGATAN: Potensi Fraud Terdeteksi!")
            
            

            st.markdown("""
            **Analisis Risiko:**
            Transaksi ini menunjukkan **anomali** atau pola yang menyimpang dari profil kebiasaan nasabah. Sistem mendeteksi indikasi aktivitas mencurigakan yang berisiko tinggi.
            
            **Tindakan yang Harus Dilakukan (SOP):**
            * ⛔ **HOLD Transaksi:** Jangan setujui transaksi ini secara otomatis.
            * 📞 **Verifikasi Manual:** Hubungi nasabah segera melalui nomor terdaftar untuk konfirmasi.
            * 🔒 **Blokir Sementara:** Jika nasabah tidak dapat dihubungi atau tidak mengenali transaksi, segera bekukan akun.
            """)
        else:
            # KASUS: NON-FRAUD (AMAN - HIJAU)
            st.success("### ✅ AMAN: Transaksi Normal (Non-Fraud)")
            
            

            st.markdown("""
            **Analisis Risiko:**
            Pola transaksi konsisten dengan profil historis nasabah (Genuine User). Tidak ditemukan indikasi penyalahgunaan akun.
            
            **Tindakan yang Harus Dilakukan:**
            * ✅ **Approve:** Lanjutkan pemrosesan transaksi.
            * ✅ **No Friction:** Tidak diperlukan verifikasi tambahan (OTP/Call) agar kenyamanan nasabah terjaga.
            """)

# --- TAB 2: UPLOAD FILE (BATCH) ---
with tab2:
    st.header("Upload File CSV")
    
    data_type = st.radio(
        "Jenis data yang diupload:",
        ("Data asli ", 
         "Data angka desimal")
    )
    
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write("Preview Data Awal:")
        st.dataframe(df_upload.head())
        
        if st.button("Proses & Prediksi Batch"):
            try:
                is_norm = True if "Normalisasi" in data_type else False
                
                # 1. Pipeline
                X_batch, df_clean = proses_prediction_pipeline(df_upload, is_normalized=is_norm)
                
                # 2. Prediksi
                prediksi_raw = model.predict(X_batch)
                
                # 3. Gabung Hasil
                df_hasil = df_clean.copy()
                
                # --- UPDATE: HANYA LABEL_CLUSTER, TANPA PREDIKSI_CLUSTER ---
                # Kita langsung mapping dari hasil prediksi raw ke Label
                df_hasil['Label_Cluster'] = pd.Series(prediksi_raw).map({1: 'Fraud', 0: 'Non-Fraud'})
                
                # 4. Tampilkan Info
                if is_norm:
                    st.success("✅ Data berhasil dikembalikan ke format asli!")
                else:
                    st.success("✅ Data berhasil diproses!")
                
                st.write("Preview Hasil Akhir:")
                st.dataframe(df_hasil.head())
                
                # 5. Download
                csv = df_hasil.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv,
                    file_name="hasil_prediksi_final.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.warning("Tips: Pastikan pilihan 'Jenis Data' sesuai dengan file yang diupload.")