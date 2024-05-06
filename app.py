import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model("my_pneumonia_detection_model.h5")
img_width, img_height = 224, 224  # Update the target size to match the model input shape

def preprocess_image(image):
    # Convert image to RGB if it has fewer than 3 channels
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize image to match model input shape
    image = image.resize((img_width, img_height))
    # Convert image to numpy array
    image = np.array(image)
    # Ensure the image has 3 color channels (RGB)
    if image.shape[-1] != 3:
        raise ValueError("Image does not have 3 color channels (RGB)")
    # Reshape image to add batch dimension
    image = np.expand_dims(image, axis=0)
    # Normalize pixel values
    image = image / 255.0
    return image

def image_prediction(new_image_path):
    test_image = Image.open(new_image_path)
    test_image = preprocess_image(test_image)
    prediction = model.predict(test_image)
    plt.imshow(test_image.squeeze())  # Plot the image
    plt.axis('off')  # Turn off axis
    if prediction[0] > 0.5:
        st.error('Affected  by Pneumonia', icon="🚨")
    else:
        st.success('Result is NORMAL !!', icon="😊")

# Streamlit app
st.title('🌡️🦠Pneumonia Detection Model')

uploaded_file = st.file_uploader("Choose chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, width=400)
    image_prediction(uploaded_file)
