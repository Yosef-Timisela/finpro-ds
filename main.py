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

# Label Encoding for categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target
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

# Realtime Input Fields
age = st.slider("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", df['job'].unique())
marital = st.radio("Marital", df['marital'].unique())
education = st.radio("Education", df['education'].unique())
default = st.radio("Is there any bad debt?", df['default'].unique())
balance = st.slider("Account Balance", min_value=int(df['balance'].min()), max_value=int(df['balance'].max()), value=1000)
housing = st.radio("Housing Loan?", df['housing'].unique())
loan = st.radio("Personal Loan?", df['loan'].unique())
contact = st.radio("Contact type", df['contact'].unique())
day = st.slider("Day (Last Contact)", min_value=1, max_value=31, value=15)
month_num = st.slider("Month (Last Contact)", min_value=1, max_value=12, value=6)
duration = st.number_input("Last Contact Duration (seconds)", value=100)
campaign = st.slider("Number of contacts in this campaign", min_value=1, max_value=50, value=1)
pdays = st.slider("Days since last previous contact", min_value=0, max_value=999, value=999)
previous = st.slider("Number of previous contacts", min_value=0, max_value=50, value=0)
poutcome = st.radio("Previous campaign results", df['poutcome'].unique())

# Convert month number to label if categorical
if 'month' in categorical_cols:
    month_labels = encoders['month'].classes_
    month = month_labels[month_num - 1] if 1 <= month_num <= 12 else 'may'
else:
    month = month_num

# Create input dataframe
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

# Encode categorical input
for col in categorical_cols:
    le = encoders[col]
    user_input[col] = le.transform(user_input[col])

# Prediction and confidence
prediction = model.predict(user_input)[0]
proba = model.predict_proba(user_input)[0]
result = target_encoder.inverse_transform([prediction])[0]
confidence = proba[prediction] * 100  # percentage

# Display result
st.subheader("Prediction Result:")
if result == 'yes':
    st.success(f"🎉 Congratulations! Your term deposit application has been accepted.\n\n✅ Confidence: {confidence:.2f}%")
else:
    st.error(f"❌ Sorry, your term deposit application was not accepted.\n\n🧠 Confidence: {confidence:.2f}%")
