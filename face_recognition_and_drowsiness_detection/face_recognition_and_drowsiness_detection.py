"""Module to run demo on streamlit"""

import cv2
import time
import numpy as np
import streamlit as st
import face_recognition as fr

# function to capture video from webcam and output processed video on app
def run_live_monitoring(image):
    '''
    Runs facial recognition and drowsinees detection model on live video feed
    
    Arguments:
        image: input image as reference for face recogintion and mark attandance
    Output:
        processed video on app with drawsiness alert
    
    '''
    # use face recognition library to encode reference image in 128-dimension vector
    # face_encodings will return a list of encodings for each face in image
    ref_enc = fr.face_encodings(np.array(image))[0]  # extract first encoding from the list
    # create camera object to get live video
    camera = cv2.VideoCapture(0)
    # create widget to output processed video
    image_holder = st.empty()
    # read from camera
    webcam, frame = camera.read()
    # video is generated frame by frame
    # each frame will be processed individualy
    # loop to run model till video is availabe
    while webcam:
        # pause current video frames for processing
        time.sleep(0.1)
        # process current frame 
        # encode current frame 
        test_enc = fr.face_encodings(frame)[0]  # extract first encoding from the list
        result = fr.compare_faces([ref_enc],test_enc,tolerance=0.6)
        print((result))
        
        
        
        
        output = f.process(image,frame)
        # output processed image
        image_holder.image(output)
        # get next frame from camera
        webcam, frame = camera.read()
    # release resources 
    camera.release()


# write header
st.header('LIVE CLASS MONITORING SYSTEM')

# create a text field to input student name
student_name = st.text_input('Enter full name of student')

# first ask for student name 
# if student name is provided then as for student image for reference
if len(student_name) > 0:
    upload = st.file_uploader('Choose an image...', type='jpg')
    # once a image is uploaded start the video for face recognition
    if upload != None:
        ref_image = upload.read()
        # pass uploaded image as reference for facial recognition
        run_live_monitoring(ref_image)
        
