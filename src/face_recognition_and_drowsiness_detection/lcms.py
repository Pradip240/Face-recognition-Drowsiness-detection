"""Module to run demo on streamlit"""
import cv2
import time
import beepy
import threading
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
import face_recognition as fr

class Camera:
    '''
    Camera object to get video from remote source

    use read() method to read frames from video
    '''
    def __init__(self) -> None:
        self.cam = cv2.VideoCapture(0)
    
    def read(self):
        _, frame = self.cam.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class LCMS:
    '''
    class to represent Live Class Monitoring System
    '''
    # constructor to initialize class
    def __init__(self) -> None:
        self.processed = None  # processed images to output
        self.image = None   # raw frame from webcam
        self.counter = 0    # counter from frame processing
        self.student_name = ""
        self.tolerance = 0.6    # threshold for face recognition 
        self.ear_threshold = 0.26    # threshold for drowsiness detection
        self.time_delta = 2     # time for eyes closed to detecting drowsiness
        self.enc_ref = None     # computed encoding for reference image
        self.attendance_register = None   # register to keep track of attendance
        # initialize states
        self.is_drowsy = False
        self.is_present = False
        self.eye_closed = None    # time when eye closed
        self.time = 0          # total session time
        self.time_stamp = None             
         

    # method to setup class demo
    def start(self):        
        # write header
        st.header('LIVE CLASS MONITORING SYSTEM')

        # create a text field to input student name
        self.student_name = st.sidebar.text_input('Enter full name of student')
        # Add a slider for tolerance
        self.tolerance = st.sidebar.slider('Select tolerance value', 0.0, 1.0, 0.6)
        # Add a slider for ear_threshold
        self.ear_threshold = st.sidebar.slider('Select eye aspect ratio threshold value', 0.0, 1.0, 0.26)
        # Add a slider for drowsiness detection time
        self.time_delta = st.sidebar.slider('Select drowsiness detection time value', 0, 10, 2)
        
        # first ask for student name 
        # if student name is provided then as for student image for reference
        if len(self.student_name) > 0:
            upload = st.sidebar.file_uploader('Choose an image...', type='jpg')
            # once a image is uploaded start the video for face recognition
            if upload != None:
                ref_image = fr.load_image_file(upload)
                # create dataframe to keep track of attendace
                self.attendance_register = create_register(self.student_name)
                # create a list of face encoding from student image
                # save encoding to avoid repeating computation
                self.enc_ref = fr.face_encodings(ref_image)     
                # run live monitoring sysetem
                self.run_live_monitoring()
                # show attendance register at end
                st.dataframe(self.attendance_register)


    # method to process input video and produce resulting video
    def run_live_monitoring(self):
        '''
        Runs facial recognition and drowsinees detection model on live video feed    
        Arguments:
            image: input image from camera
        Output:
            processed video on app with drawsiness alert        
        '''
        # use thread for plying sound in background while main thread can execute program 
        thread_sound = threading.Thread(target=beepy.beep, args=(6,), daemon=True) # play alarm sound when running
                                
        # capture frames from webcam
        camera = Camera()
        
        image_holder = st.empty()

        # video is generated frame by frame
        # each frame will be processed individualy
        # loop to run model till video is availabe
        while True:
            try:
                # read next frame from camera
                self.image = camera.read()
                # process current image
                self.process()
                # annote image
                self.annote()
                # play alarm to wake-up drowsy student
                if self.is_drowsy:
                    if not thread_sound.is_alive():
                        thread_sound = threading.Thread(target=beepy.beep, args=(6,), daemon=True)
                        thread_sound.start()
                # output image
                image_holder.image(self.processed)
            except:
                break
        # at end of class add session time to the attendance register    
        self.mark_attendance(0, self.time)

    # method to run all calculations in background
    def process(self):
        # process every 2'nd frame for speedup 
        self.counter -= 1
        if self.counter > 0:
            return
        # reset counter
        self.counter = 2
        self.face_recognition()
        # update session time
        if self.is_present:
            if self.time_stamp != None:
                self.time += (time.time() - self.time_stamp)
            self.time_stamp = time.time()
            # check for drowsiness
            self.drowsiness_detection()
        # if student is not present in frame we can assume student not attending class
        else:
            # stop current session time for student
            self.time_stamp = None # reset time stamp
        

    def face_recognition(self):
        '''
        Given an image performs face encoding and compare it with given list of encodings.
        If distance between images is less than tolerance then the student with given encoding is marked 
        as present
        '''
        self.is_present = False
        try:
            # reduce image size to speed up processing
            image = cv2.resize(self.image, (0,0), None, 0.25, 0.25)
            # find face locations
            face_locations = fr.face_locations(image)
            # encode the test image
            enc_test = fr.face_encodings(image, face_locations)[0]  # extract first encoding from the list
            # compare a list of face encodings against a test encoding to see if they match
            # euclidean distance for each face encoding is calculated and compared with tolerance value
            # tolerance is the distance between faces to consider it a match
            result = fr.face_distance(self.enc_ref,enc_test)
            # get the index of minimum distance
            min_dist_index = np.argmin(result)
            # compare with tolerance value
            if result[min_dist_index] <= self.tolerance:
                self.is_present = True
        except:
            # face encoding failed, there is no face present in image or can not match face encoding within tolerance limit
            pass
        
    def mark_attendance(self, index, session_time):
        # add session time
        prev_session_time_str = self.attendance_register.iloc[index][2]
        # convert previous session time to int(in seconds) from string(h:mm:ss)
        h, m, s = prev_session_time_str.split(':')
        prev_time = int(h)*3600 + int(m)*60 + int(s)
        # calculate new session time
        new_time = prev_time + session_time
        # convert new session time to string(h:mm:ss)
        time_str = time.strftime('%H:%M:%S', time.gmtime(new_time))
        self.attendance_register.iloc[[index],[2]] = time_str

    
    def ratio(self, points):
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

    
    def drowsiness_detection(self):
        '''
        From given image, detect facial features and extracts eyes.
        If eye feature is extracted calculate eye aspect ratio and return the average of ratio from both eyes.
        With eye aspect ratio and threshold values for EAR and Time, detects the drowsiness 
        '''
        ear = 0.5 # default start ratio
        try:
            # reduce image size to speed up processing
            image = cv2.resize(self.image, (0,0), None, 0.25, 0.25)
            # find face locations
            face_locations = fr.face_locations(image)
            # get facial landmarks as dictionary
            landmarks = fr.face_landmarks(image, face_locations)
            # extract left and right eye points from landmarks
            left_eye_points = landmarks[0]['left_eye']
            right_eye_points = landmarks[0]['right_eye']
            ear_left = self.ratio(left_eye_points)
            ear_right = self.ratio(right_eye_points)
            ear = (ear_left + ear_right)/2
        except:
            # unable to load facial features
            return
        if ear < self.ear_threshold:
            # if eyes are closed there are 2 posibilities 
            # 1. it's blink
            # 2. drowsiness
            # first check for blink
            if self.eye_closed == None:
                # start timer for closed eye
                self.eye_closed = time.time()
            else:
                # if eyes already closed, check for duration 
                # when duration is more than time_delta we consider it as drowsiness
                if (time.time() - self.eye_closed) > self.time_delta:
                    # put drowsiness notification
                    self.is_drowsy = True
                    # reset timer
                    self.eye_closed = None
        else:
            self.is_drowsy = False
            self.eye_closed = None
            
    
    def drowsiness_alert(self):
        '''Adds text in image for drowsiness alert'''
        return cv2.putText(self.image,text='Drowsiness Alert!',org=(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,color=(255,0,0),thickness=2)


    def show_session_time(self):
        '''Adds session time in image to indicate attendance is marked'''
        time = "session time (in seconds): " + str(int(self.time))
        return cv2.putText(self.image,text=time,org=(10,20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,color=(0,0,0),thickness=1)


    # method to annote image after processing it
    def annote(self):
        # check states and annote image
        if self.is_drowsy:
            self.processed = self.drowsiness_alert()
        elif self.is_present:
            self.processed = self.show_session_time()
        else:
            self.processed = self.image


# function to create register
@st.cache(allow_output_mutation=True)
def create_register(name):
    register = pd.DataFrame()
    # add today's date
    register['Date'] = [date.today().strftime("%d/%m/%Y")]
    # add name of student
    register['Name'] = [name]
    # record session time for student
    register['Session Time'] = '0:00:00'
    return register

