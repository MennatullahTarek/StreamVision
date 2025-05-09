import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import av

st.title("📦 Real-Time Object Detection (Cup)")

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

class ObjectronTransformer(VideoTransformerBase):
    def __init__(self):
        self.objectron = mp_objectron.Objectron(static_image_mode=False,
                                                max_num_objects=5,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.7,
                                                model_name='Cup')

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.objectron.process(image_rgb)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(
                    image,
                    detected_object.landmarks_2d,
                    mp_objectron.BOX_CONNECTIONS
                )
                mp_drawing.draw_axis(
                    image,
                    detected_object.rotation,
                    detected_object.translation
                )
        return image

webrtc_streamer(
    key="objectron-detect",
    video_processor_factory=ObjectronTransformer,
    media_stream_constraints={"video": True, "audio": False}
)
