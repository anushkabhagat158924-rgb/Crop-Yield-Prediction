import streamlit as st
import pandas as pd
import joblib

# Load models
linear = joblib.load("linear_model (2).pkl")
poly_model = joblib.load("polynomial_model (2).pkl")
poly = joblib.load("polynomial_features (2).pkl")

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Crop Yield Prediction System")

st.write("Predict crop yield using Machine Learning")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("Linear Regression", "Polynomial Regression")
)

st.header("Enter Crop Details")

crop = st.number_input("Crop", 0, 50, 0)

crop_year = st.number_input("Crop Year", 2000, 2030, 2020)

season = st.number_input("Season", 0, 10, 1)

state = st.number_input("State", 0, 50, 2)

area = st.number_input("Area", 1.0, 100000.0, 10000.0)

annual_rainfall = st.number_input(
    "Annual Rainfall", 0.0, 5000.0, 1500.0
)

fertilizer = st.number_input(
    "Fertilizer", 0.0, 1000000.0, 100000.0
)

pesticide = st.number_input(
    "Pesticide", 0.0, 100000.0, 3000.0
)

input_data = pd.DataFrame({
    "Crop": [crop],
    "Crop_Year": [crop_year],
    "Season": [season],
    "State": [state],
    "Area": [area],
    "Annual_Rainfall": [annual_rainfall],
    "Fertilizer": [fertilizer],
    "Pesticide": [pesticide]
})

if st.button("Predict Yield"):

    if model_choice == "Linear Regression":
        prediction = linear.predict(input_data)

    else:
        poly_input = poly.transform(input_data)
        prediction = poly_model.predict(poly_input)

    st.success(
        f"Predicted Crop Yield: {prediction[0]:.2f}"
    )