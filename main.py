import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("bank_final.csv")
    return df

df = load_data()

# ----------------------------
# Preprocessing
# ----------------------------
X = df[['age', 'job', 'marital', 'education', 'default', 'balance',
        'housing', 'loan', 'contact', 'day', 'month', 'duration',
        'campaign', 'pdays', 'previous', 'poutcome']]
y = df['deposit']

# Label Encoding untuk fitur kategorikal
categorical_cols = X.select_dtypes(include=['object']).columns
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encoding target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)  # yes = 1, no = 0

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ----------------------------
# Streamlit App
# ----------------------------
st.title("Prediksi Deposit Nasabah (Bank Marketing Dataset)")

st.subheader("Masukkan Data Nasabah:")

# Input form
with st.form("form"):
    age = st.number_input("Umur", min_value=18, max_value=100, value=30)
    job = st.selectbox("Pekerjaan", df['job'].unique())
    marital = st.selectbox("Status Pernikahan", df['marital'].unique())
    education = st.selectbox("Pendidikan", df['education'].unique())
    default = st.selectbox("Apakah memiliki kredit macet?", df['default'].unique())
    balance = st.number_input("Saldo", value=1000)
    housing = st.selectbox("Memiliki pinjaman rumah?", df['housing'].unique())
    loan = st.selectbox("Memiliki pinjaman pribadi?", df['loan'].unique())
    contact = st.selectbox("Jenis kontak", df['contact'].unique())
    day = st.number_input("Hari (kontak terakhir)", min_value=1, max_value=31, value=15)
    month = st.selectbox("Bulan (kontak terakhir)", df['month'].unique())
    duration = st.number_input("Durasi kontak terakhir (detik)", value=100)
    campaign = st.number_input("Jumlah kontak dalam kampanye ini", value=1)
    pdays = st.number_input("Hari sejak kontak terakhir sebelumnya", value=999)
    previous = st.number_input("Jumlah kontak sebelumnya", value=0)
    poutcome = st.selectbox("Hasil kampanye sebelumnya", df['poutcome'].unique())

    submit = st.form_submit_button("Prediksi")

if submit:
    # Bentuk input user menjadi dataframe
    user_input = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }])

    # Encode input dengan encoder yang sama
    for col in categorical_cols:
        le = encoders[col]
        user_input[col] = le.transform(user_input[col])

    # Prediksi
    prediction = model.predict(user_input)[0]
    result = target_encoder.inverse_transform([prediction])[0]

    # Output
    st.subheader("Hasil Prediksi:")
    if result == 'yes':
        st.success("🎉 Selamat! Deposit anda diterima.")
    else:
        st.error("❌ Mohon maaf, deposit anda tidak diterima.")