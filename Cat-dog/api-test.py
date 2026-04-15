import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict/"

st.title("🐱🐶 Cat vs Dog Classifier")

st.write("Upload an image and get prediction from your FastAPI model.")

# Upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result['prediction']}")
                else:
                    st.error(f"Error: {response.text}")

            except Exception as e:
                st.error(f"Failed to connect to API: {e}")