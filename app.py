import streamlit as st
import os
import glob
import tempfile
import cv2
import numpy as np
from anti_spoof_predict_onnx import AntiSpoofPredict


model_path = os.path.join('model', '2.7_80x80_MiniFASNetV2.onnx')
model_test = AntiSpoofPredict(0, model_path)

def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    return temp_dir

def detect_objects_in_images(image_dir):
    results = []
    for image_path in glob.glob(os.path.join(image_dir, '*')):
        image = cv2.imread(image_path)
        prediction = np.zeros((1, 3))
        prediction += model_test.predict(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results.append([cv2.resize(image,(256,256)),prediction])  # Perform object detection
    return results

st.title("Face Anti Spoofing")
st.write("Upload a folder of images and the app will check spoofing in each image.")

uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    image_dir = save_uploaded_files(uploaded_files)
    st.write("Uploaded images:")
    cols = st.columns(4)  # Adjust the number of columns as needed
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i % 4]:
            st.image(os.path.join(image_dir, uploaded_file.name), width=100)

    if st.button("Check Spoofing"):
        st.write("Checking...")
        results = detect_objects_in_images(image_dir)
        cols = st.columns(4)  # Adjust the number of columns as needed
        for i, result in enumerate(results):
            with cols[i % 4]:
                st.image(result[0])  # Display the image with detections
                st.write(result[1])  # Display the detection results as a dataframe
