from PIL import Image
import numpy as np
import streamlit as st
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image
import keras.backend as K
model = load_model('../Downloads/TrioCNNBM.h5')

classes = {
    0:'No DR',
    1:'Mild',
    2:'Moderate',
    3:'Severe',
    4:' Proliferative DR'
}

st.markdown(
    """
    <style>
    .header-style {
        font-size:25px;
        font-family:sans-serif;
        position:absolute;
        text-align: center;
        color: 032131;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .font-style {
        font-size:20px;
        font-family:sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .footer-style {
        font-size: 15px;
        font-family: sans-serif;
        position: fixed;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #262339;
        color: white;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header('')
st.markdown(
    '<p class="header-style">Automated System of Diabetic Ratinopathy Detection</p>',
    unsafe_allow_html=True
)

def load_image(img):
    test_image = image.load_img(img,target_size=(224,224,3))
    test_image = np.array(test_image)
    return test_image


def classify(img):
    K.clear_session()
    test_image = np.expand_dims(img,axis=0)
    prediction_result = model.predict(test_image)
    status = np.argmax(prediction_result,axis=1)[0]

    column_1, column_2 = st.columns(2)
    column_1.write("Patient Status : ")
    column_2.write(f"{classes[status]}")


uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png','jpeg'],accept_multiple_files = False)


if uploadFile is not None:
    img = load_image(uploadFile)
    st.image(img)
    st.write("Image Uploaded Successfully")
    if st.button('Classify Image'):
        classify(img)
   