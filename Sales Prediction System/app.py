import streamlit as st
import joblib
import pandas as pd

st.title("Boston House Price Prediction App")

# Load trained components
model = joblib.load("regression.pkl")
scaler = joblib.load("scaler.pkl")

st.write("Enter the input values:")

# User Inputs (same features as training)

crim = st.number_input("Crime rate per capita by town")
zn = st.number_input("Proportion of residential land zoned for lots over 25,000 sq.ft")
indus = st.number_input("Proportion of non-retail business acres per town (industrial land)")
chas = st.number_input("Charles River dummy variable (1 if tract bounds river; 0 otherwise)")
nox = st.number_input("Nitric oxides concentration (parts per 10 million)")
rm = st.number_input("Average number of rooms per dwelling")
age = st.number_input("Weighted distances to five Boston employment centers")
dis = st.number_input("Proportion of owner-occupied units built prior to 1940")
rad = st.number_input("Index of accessibility to radial highways")
tax = st.number_input("Full-value property tax rate per $10,000")
ptratio = st.number_input("Pupil-teacher ratio by town")
b = st.number_input("1000(Bk – 0.63)² where Bk is the proportion of Black residents")
lstat = st.number_input("Percentage of lower-status of the population")

if st.button("Predict Sales"):
    # Create input DataFrame
    data = {
        "CRIM": crim,
        "ZN": zn,
        "INDUS": indus,
        "CHAS": chas,
        "NOX": nox,
        "RM": rm,
        "AGE": age,
        "DIS": dis,
        "RAD": rad,
        "TAX": tax,
        "PTRATIO": ptratio,
        "B": b,
        "LSTAT": lstat
    }

    input_df = pd.DataFrame([data])

    # Handle missing values
    df = input_df.fillna(0)

    # Scale features
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]

    # Show result
    st.success(f"Predicted Sales: {prediction:.2f}")
