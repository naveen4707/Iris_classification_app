import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Iris Species Predictor", layout="centered")

# --- CUSTOM CSS FOR UI ---
# This section injects CSS to set the background and force black fonts
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1550747528-cdb4592f1ddf?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
    }

    /* Force all text elements to be black */
    h1, h2, h3, p, label, .stMarkdown, .stNumberInput label {
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* Make the input boxes slightly transparent to see background but readable */
    div[data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.6) !important;
        border-radius: 5px;
    }

    /* Styling the result box */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.8) !important;
        color: #000000 !important;
        border: 2px solid #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌸 Iris Flower Classifier")
st.write("Input the measurements below to predict the species of the flower.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        with open("svm_iris_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'svm_iris_model.pkl' not found. Please ensure it is in the same directory.")
        return None

model = load_model()

# --- USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# --- PREDICTION LOGIC ---
if st.button("Predict Species"):
    if model is not None:
        # We create a DataFrame to match the feature names used during training 
        # (This prevents the 'feature names' warning seen in your PDF page 4)
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
        
        prediction = model.predict(input_data)
        
        st.success(f"### Predicted Species: {prediction[0]}")
    else:
        st.error("Model not loaded.")

# Footer info
st.markdown("---")
st.write("Built with Streamlit based on SVM Linear Model.")
