import streamlit as st
import joblib
import numpy as np

model = joblib.load("sales_model_tiny.pkl")

st.title("Retail Sales Forecasting")

store = st.number_input("Store", value=1)
dept = st.number_input("Department", value=1)
holiday = st.number_input("Holiday (0/1)", value=0)
temp = st.number_input("Temperature", value=70.0)
fuel = st.number_input("Fuel Price", value=3.0)
cpi = st.number_input("CPI", value=200.0)
unemp = st.number_input("Unemployment", value=7.0)
type_ = st.number_input("Store Type", value=1)
size = st.number_input("Store Size", value=150000)
year = st.number_input("Year", value=2012)
month = st.number_input("Month", value=1)
week = st.number_input("Week", value=1)

if st.button("Predict Sales"):
    features = np.array([[store, dept, holiday, temp, fuel,
                          cpi, unemp, type_, size,
                          year, month, week]])

    prediction = model.predict(features)

    st.success(f"Predicted Weekly Sales: {prediction[0]:,.2f}")