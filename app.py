#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Sales Prediction App")

st.markdown("Predict sales based on time, city, and product details.")

# Input fields
month = st.number_input("📅 Month", min_value=1, max_value=12, value=3)
hour = st.number_input("⏰ Hour", min_value=0, max_value=23, value=14)
city = st.number_input("🏙️ City (Encoded)", min_value=0, value=0)
quantity_ordered = st.number_input("📦 Quantity Ordered", min_value=1, value=1)
price_each = st.number_input("💰 Price Each", min_value=1.0, value=10.0)

# Predict Button
if st.button("🔍 Predict Sales"):
    try:
        # Prepare input
        features = np.array([[month, hour, city, quantity_ordered, price_each]])
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        st.success(f"🤑 Predicted Sales: **${round(float(prediction), 2)}**")
    except Exception as e:
        st.error(f"Error: {e}")


# In[ ]:




