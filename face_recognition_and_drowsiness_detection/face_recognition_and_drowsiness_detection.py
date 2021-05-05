"""Main module."""
import streamlit as st

st.write('hi pradip!')

import cv2
import time


camera = cv2.VideoCapture(0)
# out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,480))

# @st.cache
# while(camera.isOpened()):
#     ret, frame = camera.read()
#     if ret==True:
#         # write the flipped frame
# #         out.write(frame)
        
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

@st.cache(suppress_st_warning=True)
def show_image():
   _, image = camera.read()
   st.image(image)

    
image_holder = st.empty()

while True:
    time.sleep(0.1)
    ret, frame = camera.read()
    image_holder.image(frame)
# st.video(out)
camera.release()
# out.release()
