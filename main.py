import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
try:
    with open('knn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'knn_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Set up the Streamlit app layout
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")

st.title("K-Nearest Neighbors Classifier for Cancer Prediction")
st.write("This application uses a pre-trained KNN model to predict whether a breast mass is malignant or benign based on its features.")
st.write("Please enter the feature values below to get a prediction.")

# --- Sidebar for user input ---
st.sidebar.header("Input Features")

# Create a dictionary to store user inputs
input_dict = {}

# List of features for the input fields
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Create input fields in the sidebar
for feature in features:
    input_dict[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", value=0.0)

# Create a DataFrame from the user inputs
input_df = pd.DataFrame([input_dict])

# --- Main content area for prediction ---
st.header("Prediction")

if st.sidebar.button("Predict"):
    # Scale the input data using the pre-trained scaler
    try:
        scaled_input = scaler.transform(input_df)
        
        # Make a prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Display the result
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error("The model predicts this is a Malignant (M) mass.")
        else:
            st.success("The model predicts this is a Benign (B) mass.")

        st.subheader("Prediction Probabilities:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Benign (B) Probability", f"{prediction_proba[0][0]:.2%}")
        with col2:
            st.metric("Malignant (M) Probability", f"{prediction_proba[0][1]:.2%}")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input values are valid numbers.")

st.markdown("---")
st.write("This app is for educational purposes only and should not be used for medical diagnosis.")
