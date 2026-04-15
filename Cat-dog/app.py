import streamlit as st
import keras
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Cat vs Dog Image Classification")

# Load model
model = keras.models.load_model('cat-dog-classification.keras')

# Upload image
uploaded_file = st.file_uploader("Upload an Image")

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    
if st.button("Predict"):
    # Resize image
    image = image.resize((224, 224))  # adjust if needed

    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    result = np.argmax(prediction)

    # Output
    if result == 0:
            st.write("🐱 Cat")
    else:
            st.write("🐶 Dog")