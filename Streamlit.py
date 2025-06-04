import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model with error handling
try:
    with open('l_model.pkl', 'rb') as file:
        model = pickle.load(file)
        st.write(f"Loaded model type: {type(model)}")
except Exception as e1:
    try:
        with open('linear_model.pkl', 'rb') as file:
            model = pickle.load(file)
            st.write(f"Loaded model type: {type(model)}")
    except Exception as e2:
        import streamlit as st
        st.error(f"Error loading model files: l_model.pkl error: {e1}, linear_model.pkl error: {e2}")
        model = None

st.title("Lung Cancer Prediction App")
st.write("Provide the following details to predict the likelihood of lung cancer.")

# Input features
AGE = st.slider("Age", 20, 100, step=1)
GENDER = st.selectbox("Gender", ['M', 'F'])
SMOKING = st.selectbox("Smoking", ['Yes', 'No'])
YELLOW_FINGERS = st.selectbox("Yellow Fingers", ['Yes', 'No'])
ANXIETY = st.selectbox("Anxiety", ['Yes', 'No'])
PEER_PRESSURE = st.selectbox("Peer Pressure", ['Yes', 'No'])
CHRONIC_DISEASE = st.selectbox("Chronic Disease", ['Yes', 'No'])
FATIGUE = st.selectbox("Fatigue", ['Yes', 'No'])
ALLERGY = st.selectbox("Allergy", ['Yes', 'No'])
WHEEZING = st.selectbox("Wheezing", ['Yes', 'No'])
ALCOHOL_CONSUMING = st.selectbox("Alcohol Consuming", ['Yes', 'No'])
COUGHING = st.selectbox("Coughing", ['Yes', 'No'])
SHORTNESS_OF_BREATH = st.selectbox("Shortness of Breath", ['Yes', 'No'])
SWALLOWING_DIFFICULTY = st.selectbox("Swallowing Difficulty", ['Yes', 'No'])
CHEST_PAIN = st.selectbox("Chest Pain", ['Yes', 'No'])

# Convert inputs to numerical format
def binary_encode(val):
    # For binary features except GENDER: map 'Yes' to 2, 'No' to 1
    if val == 'Yes':
        return 2
    elif val == 'No':
        return 1
    else:
        return 0

def encode_gender(val):
    # Map 'M' to 1, 'F' to 0
    if val == 'M':
        return 1
    elif val == 'F':
        return 0
    else:
        return 0

input_data = pd.DataFrame([[
    encode_gender(GENDER),
    AGE,
    binary_encode(SMOKING),
    binary_encode(YELLOW_FINGERS),
    binary_encode(ANXIETY),
    binary_encode(PEER_PRESSURE),
    binary_encode(CHRONIC_DISEASE),
    binary_encode(FATIGUE),
    binary_encode(ALLERGY),
    binary_encode(WHEEZING),
    binary_encode(ALCOHOL_CONSUMING),
    binary_encode(COUGHING),
    binary_encode(SHORTNESS_OF_BREATH),
    binary_encode(SWALLOWING_DIFFICULTY),
    binary_encode(CHEST_PAIN)
]], columns=[
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
    'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
    'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
    'SWALLOWING DIFFICULTY', 'CHEST PAIN'
])

# Predict
if st.button("Predict"):
    if model is None:
        st.error("Model is not loaded. Cannot perform prediction.")
    else:
        try:
            result = model.predict(input_data)[0]
            if result == 1:
                st.error("⚠️ The person is likely to have lung cancer.")
            else:
                st.success("✅ The person is unlikely to have lung cancer.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
