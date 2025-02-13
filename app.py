import numpy as np 
import streamlit as st 
import tensorflow as tf
import pickle
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the trained model
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('onehotEncoder_geo.pkl', 'rb') as file:
    onehotEncoder_geo = pickle.load(file)

# Streamlit App
st.title("Customer Churn Prediction")

# User Input
geography = st.selectbox('Geography', onehotEncoder_geo.categories_[0])  # OneHotEncoder uses categories_
gender = st.selectbox('Gender', label_encoder_gender.classes_)  # LabelEncoder uses classes_
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Encode 'Gender'
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-hot encode 'Geography'
geo_encoded = onehotEncoder_geo.transform(np.array(geography).reshape(-1, 1))  # Fix reshape issue
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotEncoder_geo.get_feature_names_out())  # Fix column issue

# Input Data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
