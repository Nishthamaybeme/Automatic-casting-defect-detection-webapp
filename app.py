import torch
import streamlit as st
import requests
from PIL import Image
import os

# Download model from Google Drive (only if it doesn't exist locally)
model_url = "https://drive.google.com/file/d/1Nmi4h-cIXeYbz0Dcf_1oI1Vzm8OObFtG/view?usp=sharing"
model_path = "crime_game_model.pth"

if not os.path.exists(model_path):
    st.info("Downloading model... This might take a while.")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    st.success("Model downloaded!")

# Load the model
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

# Streamlit app for image upload
st.title("Upload Image for Prediction")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Add prediction logic here
    st.write("Prediction: Placeholder")
