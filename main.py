
import streamlit as st
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras. preprocessing import image
import cv2
st.title("Feed")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
l='model1.h5'
model2 = tf.keras.models.load_model(
       (l),
       custom_objects={'KerasLayer':hub.KerasLayer}
)
while True:
    _, frame1 = camera.read()
    time.sleep(0.1)

    _, frame2 = camera.read()
    time.sleep(0.1)


    image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (256, 256))
    image1=  np.dstack([image1]*3)
    image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.resize(image2, (256, 256))
    image2 = np.dstack([image2]* 3)
    absdiff = cv2.absdiff(image1,image2)

    absdiff1 = np.expand_dims(absdiff, axis=0)


    a=model2.predict(absdiff1)
    if a==0:
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (0, 30)

        # fontScale
        fontScale = 0.5

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.putText() method
        absdiff = cv2.putText(absdiff, 'Signed', org, font,
                            fontScale, color, thickness, cv2.LINE_AA)


    FRAME_WINDOW.image(absdiff)





else:
    st.write('Stopped')
