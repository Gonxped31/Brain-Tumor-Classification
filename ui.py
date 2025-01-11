import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib

def load_model(model_path):
    return joblib.load(model_path)

def preprocess_image(image, target_size=(32, 32)):
    image = image.convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, target_size)
    image = image.flatten().reshape(1, -1)
    return image

def predict_image(model, image):
    prediction = model.predict(image)
    return 'Tumor' if prediction[0] == 1 else 'No Tumor'

st.title("Brain Tumor Classification")
st.write("Upload a brain scan image to detect the presence of a tumor.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Processing...")

    model = load_model("svc.pkl")
    processed_image = preprocess_image(image)
    result = predict_image(model, processed_image)

    st.write(f"Prediction: **{result}**")