import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


model = tf.keras.models.load_model("model.h5", compile=False)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('labelencoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')
st.write("Enter customer details to predict churn")

credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, step=1)
geography = st.selectbox("Geography", ["France", "Germany", "Spain", "Other"])
gender = st.radio("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, step=1)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, step=1)
balance = st.number_input("Balance", min_value=0.0, step=100.0, format="%.2f")
num_products = st.number_input("Number of Products", min_value=1, max_value=4, step=1)
has_crcard = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0, step=100)

if st.button("Submit"):
    
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [label_encoder.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [1 if has_crcard == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary],
    })

    
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    
    input_data.drop(columns=['Geography'], inplace=True)
    input_data_df = pd.concat([input_data, geo_encoded_df], axis=1)

    
    input_df_scaled = scaler.transform(input_data_df)

    
    prediction = model.predict(input_df_scaled)
    prediction_proba = float(prediction[0])

    
    if prediction_proba > 0.5:
        st.error(f'⚠️ The customer is likely to churn. Probability: {prediction_proba:.2f}')
    else:
        st.success(f'✅ The customer is not likely to churn. Probability: {prediction_proba:.2f}')
