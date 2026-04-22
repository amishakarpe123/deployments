import streamlit as st
import pandas as pd
import pickle
import time
from streamlit_lottie import st_lottie
import requests

# Page configuration
st.set_page_config(page_title="Purchase Predictor", page_icon="🛍️", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Assets
lottie_shopping = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5ngm4806.json")

# Load the model
@st.cache_resource
def load_model():
    with open("model (4).pkl", "rb") as f:
        model = pickle.pi_load(f)
    return model

model = load_model()

# Header Section
st.title("🛍️ Customer Purchase Predictor")
st.write("Predict whether a customer will purchase a product based on their profile.")

if lottie_shopping:
    st_lottie(lottie_shopping, height=200, key="coding")

# Input Form
with st.container():
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 30)
    
    with col2:
        salary = st.number_input("Estimated Annual Salary ($)", min_value=1000, max_value=200000, value=50000, step=500)

# Pre-processing input
# Note: Ensure your training encoding matches (e.g., Male=1, Female=0)
gender_encoded = 1 if gender == "Male" else 0

input_data = pd.DataFrame([[gender_encoded, age, salary]], 
                          columns=['Gender', 'Age', 'EstimatedSalary'])

# Prediction
if st.button("Analyze Customer"):
    with st.spinner('Calculating probabilities...'):
        time.sleep(1) # For animation effect
        prediction = model.predict(input_data)
        
        st.divider()
        
        if prediction[0] == 1:
            st.balloons()
            st.success("### Result: Likely to Purchase! 🎯")
            st.write("This customer shows high intent patterns based on their profile.")
        else:
            st.warning("### Result: Unlikely to Purchase ❌")
            st.write("This customer is currently projected to be a non-buyer.")
