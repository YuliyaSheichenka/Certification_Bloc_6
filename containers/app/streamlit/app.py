import streamlit as st
import numpy as np
import tensorflow as tf
import warnings
import platform
from PIL import Image
from utils import image2tensor, pred2label
import requests
import json

warnings.filterwarnings('ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)

# Add a select box to choose between Alzheimer and brain tumors
options = {"Alzheimer": "https://api.brainsight.tech/predictionalz", "Brain Tumors": "https://api.brainsight.tech/predictionbt"}
selected_option = st.sidebar.radio("Select a condition to predict", options.keys())

# Upload an image and set some options for demo purposes
img_file = st.sidebar.file_uploader(label='Upload MRI scan file', type=['png', 'jpg'])

if img_file:
    img = Image.open(img_file)
    st.image(img, caption='Uploaded scan', use_column_width=True)


def on_submit():
    '''
        consume deep learning model
    '''
    if img_file:
        img = img_file.getvalue()
        files = {'file': img}
        
        # API Endpoint
        endpoint = options[selected_option]
        response = requests.post(endpoint, files=files)

        # response = requests.post('https://api.brainsight.tech/predictionalz', files=files)
        prediction = response.json()
        print(prediction["predicted_label"])
        st.header(f"Prediction: {prediction['predicted_label']}")
    print('ok')


st.sidebar.button('Start prediction', key=None, help=None,
                  on_click=on_submit, disabled=False)


