import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# MediaPipe Face Detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Custom Video Transformer class for streamlit-webrtc
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame to an OpenCV image
        img = frame.to_ndarray(format="bgr24")
        
        # Convert the image to RGB (for MediaPipe processing)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(img_rgb)
        
        # Draw face detections on the frame
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)
        
        # Return the frame back to Streamlit
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC streamer to display video
def run_webrtc():
    webrtc_streamer(
        key="face-detection-webcam",
        video_transformer_factory=VideoTransformer,
        async_mode=True
    )

# Streamlit app UI
def app():
    st.title("Real-Time Face Detection with Webcam")
    st.write(
        "This app uses OpenCV and MediaPipe for real-time face detection. "
        "It leverages streamlit-webrtc to stream webcam footage."
    )
    
    run_webrtc()

# Run the Streamlit app
if __name__ == "__main__":
    app()
