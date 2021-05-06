"""Module to run demo on streamlit"""

import cv2
import time
import PIL
import beepy
import threading
import os, sys
from io import BytesIO    
import numpy as np
import pandas as pd
import streamlit as st
import face_recognition as fr

def match_encoding(ref_enc, test_image, tolerance):
    '''
    Given an image performs face encoding and compare it with given list of encodings.
    If distance between images is less than tolerance then index of encoding from the list that matched
    with the test image is returned.
    '''
    index = -1
    try:
        # encode the test image
        enc_test = fr.face_encodings(test_image)[0]  # extract first encoding from the list
        # compare a list of face encodings against a test encoding to see if they match
        # euclidean distance for each face encoding is calculated and compared with tolerance value
        # tolerance is the distance between faces to consider it a match
        result = fr.face_distance(ref_enc,enc_test)
        # get the index of minimum distance
        min_dist_index = np.argmin(result)
        # compare with tolerance value
        if result[min_dist_index] <= tolerance:
            index = min_dist_index
    except:
        # face encoding failed, there is no face present in image or can not match face encoding within tolerance limit
        pass
    return index

def mark_attendance(register, index, session_start_time):
    # mark attendance in register to given index
    register.iloc[[index],[1]] = 'P'
    # add session time
    prev_session_time_str = register.iloc[index][2]
    # convert previous session time to int(in seconds) from string(h:mm:ss)
    h, m, s = prev_session_time_str.split(':')
    prev_time = int(h)*3600 + int(m)*60 + int(s)
    # calculate new session time
    new_time = prev_time + time.time() - session_start_time
    # convert new session time to string(h:mm:ss)
    time_str = time.strftime('%H:%M:%S', time.gmtime(new_time))
    register.iloc[[index],[2]] = time_str

def ratio(points):
    # from list of tuples calculate aspect ratio
    # initialize default values for extreme points
    left = 1000000
    right = 0
    up = 1000000
    down = 0
    # iterate over all points to find extreme points
    for p in points:
        if p[0] < left:
            left = p[0]
        if p[0] > right:
            right = p[0]
        if p[1] < up:
            up = p[1]
        if p[1] > down:
            down = p[1]
    # calculate aspect ratio
    ratio = (down - up) / (right - left)
    return ratio

def calculate_ear(image):
    '''
    From given image, detect facial features and extracts eyes.
    If eye feature is extracted calculate eye aspect ratio and return the average of ratio from both eyes.
    Argument:
        image: input image 
    returns:
        ear: float indicating average eye aspect ratio
    '''
    ear = 0.5 # default start ratio
    try:
        # get facial landmarks as dictionary
        landmarks = fr.face_landmarks(image)
        # extract left and right eye points from landmarks
        left_eye_points = landmarks[0]['left_eye']
        right_eye_points = landmarks[0]['right_eye']
        ear_left = ratio(left_eye_points)
        ear_right = ratio(right_eye_points)
        ear = (ear_left + ear_right)/2
    except:
        # unable to load facial features
        pass
    return ear

def drowsiness_alert(image):
    '''Adds text in image for drowsiness alert'''
    return cv2.putText(image,text='Drowsiness Alert!',org=(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=1,color=(255,0,0),thickness=2)

def not_attentive(image):
    '''Adds text in image for attention alert'''
    return cv2.putText(image,text='Not attentive!',org=(10,20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.5,color=(0,0,0),thickness=1)

def attendance_notification(image,name):
    '''Adds text in image to indicate attendance is marked'''
    msg = 'Welcome ' + name + ' your attendance is marked.'
    return cv2.putText(image,text=msg,org=(10,20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.5,color=(0,255,0),thickness=1)

# function to capture video from webcam and output processed video on app
def run_live_monitoring(register, ref_enc, tolerance=0.6, ear_threshold=0.2, time_delta=2):
    '''
    Runs facial recognition and drowsinees detection model on live video feed    
    Arguments:
        image: input image as reference for face recogintion and mark attandance
    Output:
        processed video on app with drawsiness alert
    
    '''
    # use thread for plying sound in background while main thread can execute program 
    t_sound = threading.Thread(target= beepy.beep,args=(6,)) # play alarm sound when running
    # declare varaibles to detect time difference
    session_start_time = None
    gaze_away_start_time = None
    eye_closed_start_time = None
    attendance_notification_start_time = None
    prev_index = None
    counter = 0  # counter for iterations
    
    # capture frames from webcam
    camera = cv2.VideoCapture(0)
    # read first frame
    webcam, frame = camera.read()
    # create empty holder to output images to create video on app screen
    image_holder = st.empty()
    webcam, frame = camera.read()
    # video is generated frame by frame
    # each frame will be processed individualy
    # loop to run model till video is availabe
    # create a button to stop video
    run = True
    if st.button('stop'):
        run = False
    while webcam and run:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = frame
        # get index of matching student encoding
        index = match_encoding(ref_enc, frame, tolerance)
        # if index is not -1 means matching student ecoding found
        if index != -1:
            prev_index = index  # update previous index value
            # there are 2 possibilities either student is already attending class
            # or just joined the class
            # if session start time is 'None', means student just joined the class
            # we can mark student's attendance
            if session_start_time == None:
                # start new session for student
                session_start_time = time.time()
                # if student have not marked as present then show notification
                if register.iloc[index]['Attendance'] == 'A':
                    # start attendance notification time
                    attendance_notification_start_time = time.time()
                    # output notification
                    name = register.iloc[index]['Name']
                    output = attendance_notification(frame, name)
                    # mark student's attendance
                    mark_attendance(register, index, session_start_time)
            else:
                # if student is already attending class, check for notification time out
                # we will show notification for 3 seconds
                if attendance_notification_start_time != None:
                    if (time.time() - attendance_notification_start_time) < 3:
                        # continue showing notification for 3 second
                        name = register.iloc[index]['Name']
                        output = attendance_notification(frame, name)
                    else:
                        # close notification
                        attendance_notification_start_time = None
                # check for drowsiness
                ear = calculate_ear(frame)
                if ear < ear_threshold:
                    # if eyes are closed there are 2 posibilities 
                    # 1. it's blink
                    # 2. drowsiness
                    # first check for blink
                    if eye_closed_start_time == None:
                        # start timer for closed eye
                        eye_closed_start_time = time.time()
                    else:
                        # if eyes already closed, check for duration 
                        # when duration is more than time_delta we consider it as drowsiness
                        if (time.time() - eye_closed_start_time) > time_delta:
                            # put notification alert
                            output = drowsiness_alert(frame)
                            # play alarm sound
                            if not t_sound.is_alive():
                                t_sound = threading.Thread(target= beepy.beep,args=(6,))
                                t_sound.start()
                else:
                    # when eyes are open stop the timers
                    eye_closed_start_time = None
        # when matching index is not found we can assume either student not attending class
        # or algorithm unable to detect face of student
        # to remove second possiblity we calculate gaze away time
        else:
            # if session is not started then no one in front of webcam is attending the class 
            if session_start_time != None:
                if gaze_away_start_time == None:
                    gaze_away_start_time = time.time()
                else:
                    # calculate the gaze away time to detect if student not attentive
                    if (time.time() - gaze_away_start_time) > time_delta:
                        # end current session time for student and show notification
                        mark_attendance(register, prev_index, session_start_time)
                        gaze_away_start_time = None # reset gazez away time
                        session_start_time = None # reset session time
                        output = not_attentive(frame)
            else:
                # if student was attending class continue showing notification
                if prev_index != None:
                    if register.iloc[prev_index]['Attendance'] == 'P':
                        output = not_attentive(frame)
        # show output image on a window
        image_holder.image(output)
        # read next frame
        webcam, frame = camera.read()
        # add session time every 10 iterations
        if (prev_index != None) and (session_start_time != None) and (counter > 10):
            counter = 0
            mark_attendance(register, prev_index, session_start_time)
        else:
            counter += 1
    camera.release()
    cv2.destroyAllWindows()
    

@st.cache(allow_output_mutation=True)
def create_register(names):
    register = pd.DataFrame()
    # add name of student
    register['Name'] = names
    # initially mark student as absent
    register['Attendance'] = 'A'
    # record session time for student
    register['Session Time'] = '0:00:00'
    return register


# write header
st.header('LIVE CLASS MONITORING SYSTEM')

# create a text field to input student name
student_name = st.sidebar.text_input('Enter full name of student')
# Add a slider for tolerance
tolerance = st.sidebar.slider('Select tolerance value', 0.0, 1.0, 0.6)
# Add a slider for ear_threshold
ear_threshold = st.sidebar.slider('Select eye aspect ratio threshold value', 0.0, 1.0, 0.2)
# Add a slider for drowsiness detection time
time_delta = st.sidebar.slider('Select drowsiness detection time value', 0, 10, 2)
        
# first ask for student name 
# if student name is provided then as for student image for reference
if len(student_name) > 0:
    upload = st.sidebar.file_uploader('Choose an image...', type='jpg')
    # once a image is uploaded start the video for face recognition
    if upload != None:
        ref_image = fr.load_image_file(upload)
        # create dataframe to keep track of attendace
        attendance_register = create_register([student_name])
        # create a list of face encoding from student image
        enc_ref = fr.face_encodings(ref_image)     
        # run live monitoring sysetem
        run_live_monitoring(attendance_register, enc_ref, tolerance, ear_threshold, time_delta)
        # show attendance register at end
        st.dataframe(attendance_register)
    
