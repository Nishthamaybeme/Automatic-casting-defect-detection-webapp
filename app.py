import torch
import streamlit as st
from PIL import Image
import requests
import os

# Function to create a direct download link for Google Drive
def download_file_from_google_drive(url, destination):
    file_id = url.split("/d/")[1].split("/view")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            f.write(response.content)
    else:
        st.error("Failed to download the model. Please check the link.")

# Model file path
model_url = "https://drive.google.com/file/d/1Nmi4h-cIXeYbz0Dcf_1oI1Vzm8OObFtG/view?usp=sharing"
model_path = "crime_game_model.pth"

# Download model if not already downloaded
if not os.path.exists(model_path):
    st.info("Downloading model... This might take a while.")
    download_file_from_google_drive(model_url, model_path)
    st.success("Model downloaded!")

# Load the model
try:
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
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
    transform = torch.nn.Sequential(
        torch.nn.Resize((224, 224)),  # Adjust based on your model's input size
        torch.nn.ToTensor()
    )
    input_image = transform(image).unsqueeze(0)

    # Run the prediction
    with torch.no_grad():
        prediction = model(input_image)
        predicted_class = torch.argmax(prediction, dim=1).item()

    # Display the prediction
    class_names = ["def_front", "ok_front"]  # Adjust based on your classes
    st.write(f"Prediction: **{class_names[predicted_class]}**")
