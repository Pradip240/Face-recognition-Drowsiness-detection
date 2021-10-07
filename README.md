Face Recocognition & Drowsiness Detection
=========================================

[![Build Status](https://img.shields.io/pypi/v/face_recognition_and_drowsiness_detection.svg)](https://pypi.python.org/pypi/face_recognition_and_drowsiness_detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Problem statement: 
Digital classrooms are conducted via video telephony software program where it is not possible to see all students and access the mood. Students are not focusing on content due to lack of surveillance, apply deep learning algorithms to live video data from webcam to overcome this challenge.

## Approach:
Capture frames from video and process each frame individually as image.
Apply HOG model on image to get histogram of gradients.
Use these gradients to identify facial region in image.
From all the facial regions extract 68 facial features for each facial region detected in input image.
Use the facial features to encode image in 128 dimension vector. This vector is used for face recognition.
To recognize student by face we first take an image of student. This image will act as reference image.
This image is encoded and saved as 128 dimension vector in our dataset. Dataset also contains name of each student with encoded image.
Live video is processed in real time and each frame is encoded with 128 dimension vector, this vector is compared with all the vectors from saved dataset and if distance between these vectors is less than a threshold value then student with matching encoding is marked as present and start recording session time for that student. Session time is calculated as total time thee student present in front of web camera.
Eye region can be extracted from facial features and we can calculate Eye Aspect Ratio (EAR). If EAR is lower than threshold value then eyes are closed and if eyes remain closed for longer time then it is marked as drowsiness and raise a drowsiness alert.

## Conclusion:
Deep learning can be used to process video data in real time. With Face recognition and drowsiness detection it can improve students attention span and help them learn the concept well. 

Installing
--------

Requires Python3.6 or higher. Run the command to install Face Recognition & Drowsiness Detection

```
pip install face-recognition-and-drowsiness-detection
```

Run the command for Demo App

```
streamlit run app.py
```

* Free software: MIT license


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
