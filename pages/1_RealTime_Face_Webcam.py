import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="😎 Real-Time Face Detection (Webcam)", layout="centered")
st.title("😎 Real-Time Face Detection from Webcam")

class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, bbox, (0, 255, 0), 2)
                score = round(detection.score[0] * 100, 2)
                cv2.putText(img, f'{score}%', (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img

webrtc_streamer(key="face-detect", video_processor_factory=FaceDetectionTransformer)
