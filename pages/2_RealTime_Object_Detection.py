import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(layout="wide")
st.title("Real-time Object Detection (Simulated)")

st.warning("Note: MediaPipe doesn’t have general object detection. This is a placeholder.")

start = st.button("Start Object Detection")

if start:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Unable to read from webcam. Exiting...")
            break

        frame = cv2.flip(frame, 1)

        # Simulate detection with a rectangle (placeholder)
        height, width, _ = frame.shape
        cv2.rectangle(frame, (100, 100), (width - 100, height - 100), (0, 255, 0), 2)
        cv2.putText(frame, 'Simulated Object', (110, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

        if st.button("Stop"):
            break

    cap.release()
