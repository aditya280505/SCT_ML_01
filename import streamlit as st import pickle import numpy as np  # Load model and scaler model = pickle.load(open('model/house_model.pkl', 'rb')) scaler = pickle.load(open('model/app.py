import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model/house_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

st.title("🏠 House Price Prediction")

# Input fields
area = st.number_input("Area (in sq ft):", min_value=0)
bedrooms = st.number_input("Number of Bedrooms:", min_value=1)
bathrooms = st.number_input("Number of Bathrooms:", min_value=1)

# When user clicks Predict
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")
