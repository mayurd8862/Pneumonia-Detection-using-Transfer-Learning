import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('my_pneumonia_detection_model.h5')
img_width, img_height = 256, 256

# from tensorflow.keras.preprocessing import image as keras_image

def preprocess_image(image):
    # Convert image to RGB if it has fewer than 3 channels
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize image to match model input shape
    image = image.resize((img_width, img_height))
    # Convert image to numpy array
    image = keras_image.img_to_array(image)
    # Ensure the image has 3 color channels (RGB)
    if image.shape[-1] != 3:
        raise ValueError("Image does not have 3 color channels (RGB)")
    # Reshape image to add batch dimension
    image = np.expand_dims(image, axis=0)
    # Normalize pixel values
    image = image / 255.0
    return image


# Streamlit app
st.title('🌡️🦠Pneumonia Detection Model')

uploaded_file = st.file_uploader("Choose chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image,width=400)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)

    # Get class label
    class_label = np.argmax(prediction)
    
    # Display prediction
    if class_label==1:
        st.error('Affected  by Pneumonia', icon="🚨")
    else:
        st.success('Result is NORMAL !!', icon="😊")
       
