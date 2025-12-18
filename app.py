import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Iris Species Predictor", layout="centered")

# --- CUSTOM CSS FOR UI ---
# This block sets the background image and forces all text to be black
st.markdown(
    """
    <style>
    /* Background Image */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1550747528-cdb4592f1ddf?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Force all text (headings, labels, paragraphs) to be Black */
    h1, h2, h3, p, label, .stMarkdown, .stNumberInput label {
        color: #000000 !important;
        font-weight: 800 !important; /* Extra bold for better readability against background */
    }

    /* Make input fields semi-transparent so we can see the flower, but text stays clear */
    div[data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 10px;
    }

    /* Style the prediction result box */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌸 Iris Flower Classification")
st.write("Enter the measurements below to identify the Iris species.")

# --- LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_model():
    try:
        with open("svm_iris_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'svm_iris_model.pkl' not found! Please make sure it's in the same folder.")
        return None

model = load_model()

# --- INPUT SECTION ---
# Using columns to organize inputs cleanly
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.1, max_value=10.0, value=5.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.1, max_value=10.0, value=3.5)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.1, max_value=10.0, value=1.4)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=10.0, value=0.2)

# --- PREDICTION ---
if st.button("Identify Species"):
    if model is not None:
        # Create a DataFrame with the exact column names used during training
        # to avoid the 'feature names' warning seen in your PDF.
        features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
        
        prediction = model.predict(features)
        
        # Display the result
        st.success(f"### Predicted Species: **{prediction[0]}**")
    else:
        st.error("Model is not loaded. Check your .pkl file.")

st.markdown("---")
st.write("Model trained using Support Vector Machine (Linear Kernel)")
