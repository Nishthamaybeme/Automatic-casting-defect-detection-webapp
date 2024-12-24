import gdown
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
import os

# Correct Google Drive file ID for gdown
model_url = 'https://drive.google.com/uc?id=1Nmi4h-cIXeYbz0Dcf_1oI1Vzm8OObFtG'

# Download the model from Google Drive
gdown.download(model_url, 'crime_game_model.pth', quiet=False)

# Load the model (ensure the correct path and extension are used)
try:
    model = torch.load('crime_game_model.pth', map_location=torch.device("cpu"))
    model.eval()  # Set the model to evaluation mode
except Exception as e:
    st.error(f"Failed to load the model: {e}")

# Streamlit app
st.title("Automatic Casting Defect Detection")
st.write("Upload an image to predict if it's a defect or not.")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust based on your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization values for ImageNet-like models
    ])
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run the prediction
    with torch.no_grad():
        prediction = model(input_image)
        predicted_class = torch.argmax(prediction, dim=1).item()

    # Display the prediction
    class_names = ["def_front", "ok_front"]  # Adjust based on your classes
    st.write(f"Prediction: **{class_names[predicted_class]}**")
