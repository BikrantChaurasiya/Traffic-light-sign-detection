import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import urllib.request
import os

MODEL_URL = "https://huggingface.co/bikrantchaurasiya/traffic_light_model/resolve/main/traffic_light_model.keras"
MODEL_PATH = "traffic_light_model.keras"

# Download model from Hugging Face
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# IMPORTANT:
# Replace these with the exact class_names from your training notebook.
class_names = [
    "go",
    "goLeft",
    "goForward",
    "stop",
    "warning",
    "stopLeft",
    "warningLeft"
]

st.title("🚦 Traffic Light Sign Detection")

st.write("Upload a traffic light image to classify the detected sign.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    if st.button("Predict"):

        img = image.resize((224, 224))
        img = np.array(img)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)[0]

        index = np.argmax(prediction)
        confidence = prediction[index]

        st.success(
            f"Detected Sign: {class_names[index]}"
        )

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )
