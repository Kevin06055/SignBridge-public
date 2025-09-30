import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(page_title="Sign Language Predictor", layout="centered")

st.title("🖐 Sign Language Predictor")
st.write("Image upload panna → model predict panna → annotated result kaamikum.")

# Model load panna
@st.cache_resource
def load_model():
    return YOLO('D:/MyWorks/SignBridge/best.pt')

model = load_model()

# Image upload panna
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Predict panna
    results = model(img)

    # Annotate panna
    annotated_img = results[0].plot()

    # Convert BGR to RGB for display
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show original and result
    st.subheader("Original Image")
    st.image(img_rgb, use_column_width=True)

    st.subheader("Prediction Result")
    st.image(annotated_img, use_column_width=True)

    # Optionally, show details
    st.subheader("Detection Info")
    st.write(results[0].boxes.data.cpu().numpy())
