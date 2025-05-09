import streamlit as st
import cv2
import mediapipe as mp

# Set up MediaPipe object detection
mp_object_detection = mp.solutions.object_detection
mp_drawing = mp.solutions.drawing_utils

st.title('Real-time Object Detection')

# Set up webcam for real-time video capture
cap = cv2.VideoCapture(0)

# Initialize object detection model
with mp_object_detection.ObjectDetection(min_detection_confidence=0.5) as object_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect objects
        results = object_detection.process(rgb_frame)

        # Draw object detections on the frame
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Convert BGR back to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the Streamlit app
        st.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
