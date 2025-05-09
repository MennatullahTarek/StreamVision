import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import av

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.title("😀 Real-Time Face Detection (WebRTC)")

class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)

        return img

webrtc_streamer(
    key="face-detect",
    video_processor_factory=FaceDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False}
)
