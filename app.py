import streamlit as st
import pickle
import numpy as np

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏠 House Price Prediction")

# Inputs
area = st.number_input("Enter Area (sq ft)")
bedrooms = st.slider("Number of Bedrooms", 1, 5, 3)
bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)

# Predict button
if st.button("Predict"):
    input_data = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")
