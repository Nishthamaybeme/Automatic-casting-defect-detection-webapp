import torch
import streamlit as st
from PIL import Image

# Load the model locally
model_path = "E:\defect detection\crime_game_model.pth"  # Adjust the path if needed
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
