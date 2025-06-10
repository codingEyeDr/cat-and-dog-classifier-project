import streamlit as st
import tensorflow as tf
from PIL import image
import numpy as np

model = keras.models.load_model('CatandDog.h5')

st.title('Cat and Dog Classifier')

upload = st.file_uploader('Upload an image', type=['jpg', 'png'])

img_size = 100

if upload():
    image = Image.open(upload)
    st.image(image, caption='Uploaded image')

    image = image.resize((img_size, img_size))
    image = image.array(image)/255
    image = np.expand_dims(image, axis = 0)


    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    st.write(f'Predicted class {class_idx}')
