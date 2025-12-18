import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Iris Species Predictor", layout="centered")

# --- CUSTOM CSS FOR BACKGROUND AND BLACK FONTS ---
st.markdown(
    """
    <style>
    /* Full Page Background */
    .stApp {
        background: url("https://images.unsplash.com/photo-1599932025779-11100366112d?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Force all text to be Black and Bold */
    h1, h2, h3, p, label, .stMarkdown, .stNumberInput label {
        color: #000000 !important;
        font-weight: 900 !important;
    }

    /* Input box styling for visibility */
    div[data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.8) !important;
        border: 2px solid #000000 !important;
        border-radius: 10px;
    }
    
    /* Input text color */
    input {
        color: #000000 !important;
    }

    /* Prediction Result Box */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 3px solid #000000;
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌸 Iris Flower Classification")
st.write("Enter the measurements below to identify the species.")

# --- SPECIES IMAGE DATABASE ---
species_images = {
    "Iris-setosa": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
    "Iris-versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "Iris-virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

# --- LOAD THE MODEL ---
@st.cache_resource
def load_model():
    try:
        with open("svm_iris_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'svm_iris_model.pkl' not found.")
        return None

model = load_model()

# --- INPUT SECTION ---
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.1, max_value=10.0, value=5.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.1, max_value=10.0, value=3.5)
with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.1, max_value=10.0, value=1.4)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=10.0, value=0.2)

# --- PREDICTION AND IMAGE OUTPUT ---
if st.button("Identify Species"):
    if model is not None:
        # Match feature names from training
        features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
        
        prediction = model.predict(features)[0]
        
        # Output result
        st.success(f"### Predicted Species: {prediction}")
        
        # Display the flower image
        if prediction in species_images:
            st.image(species_images[prediction], caption=f"Visual of {prediction}", width=400)
    else:
        st.error("Model not loaded.")

st.markdown("---")
st.write("Model trained using Support Vector Machine (Linear Kernel)")
