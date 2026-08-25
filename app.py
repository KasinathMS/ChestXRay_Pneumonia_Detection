import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenetv2.keras")

model = load_model()

st.title("Pneumonia Detection from Chest X-Ray")

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Chest X-Ray")

    # Resize
    image_resized = image.resize((224, 224))

    # Convert to NumPy array
    img_array = np.array(image_resized)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # MobileNetV2 preprocessing
    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    probability = prediction[0][0]

    if probability >= 0.5:
        result = "PNEUMONIA"
        confidence = probability
    else:
        result = "NORMAL"
        confidence = 1 - probability

    st.subheader("Prediction")
    st.write(f"**{result}**")
    st.write(f"Probability: **{confidence * 100:.2f}%**")