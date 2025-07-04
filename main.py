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
st.markdown("<h1 style='text-align: center;'>Bank Term Deposit Prediction</h1>", unsafe_allow_html=True)

st.subheader("Input Customer Data:")

# Input form
with st.form("form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", df['job'].unique())
    marital = st.selectbox("Marital", df['marital'].unique())
    education = st.selectbox("Education", df['education'].unique())
    default = st.selectbox("is there any bad debt??", df['default'].unique())
    balance = st.number_input("Account Balance", value=1000)
    housing = st.selectbox("Housing Loan?", df['housing'].unique())
    loan = st.selectbox("Personal Loan?", df['loan'].unique())
    contact = st.selectbox("Contact type", df['contact'].unique())
    day = st.number_input("Day (Last Contact)", min_value=1, max_value=31, value=15)
    month = st.selectbox("Month (Last Contact)", df['month'].unique())
    duration = st.number_input("Last Contact Duration (seconds)", value=100)
    campaign = st.number_input("Number of contacts in this campaign", value=1)
    pdays = st.number_input("Days since last previous contact", value=999)
    previous = st.number_input("Number of previous contacts", value=0)
    poutcome = st.selectbox("Previous campaign results", df['poutcome'].unique())

    submit = st.form_submit_button("Predict")

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
    st.subheader("Prediction Result:")
    if result == 'yes':
        st.success("🎉 Congratulations! Your term deposit application has been accepted.")
    else:
        st.error("❌ Sorry, your term deposit application was not accepted.")
