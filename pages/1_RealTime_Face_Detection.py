import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(layout="wide")
st.title("Real-time Face Detection")

start = st.button("Start Face Detection")

if start:
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Unable to read from webcam. Exiting...")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

            stframe.image(frame, channels="BGR")

            # Exit button
            if st.button("Stop"):
                break

    cap.release()
