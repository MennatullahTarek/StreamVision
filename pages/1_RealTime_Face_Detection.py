import streamlit as st
import tempfile
import cv2
import time
import numpy as np
from PIL import Image
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="😎 Face Detection", layout="centered")
st.title("😎 Real-Time Face Detection with MediaPipe")
st.markdown("Upload a video and choose your funky face detection style ✨")

style = st.selectbox("🎭 Choose your detection style:", ["🟢 Classic Box", "🤩 Funky Mode"])

uploaded_video = st.file_uploader("📼 Upload a video", type=["mp4", "mov", "avi"])

# Load MediaPipe Face Detector
@st.cache_resource
def load_face_detector():
    base_options = python.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
    return vision.FaceDetector.create_from_options(options)

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    detector = load_face_detector()
    cap = cv2.VideoCapture(temp_video_path)
    FRAME_WINDOW = st.empty()
    face_count = 0
    frame_count = 0

    st.info("🧠 Detecting faces...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(mp_image)

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2, y2 = x1 + bbox.width, y1 + bbox.height
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            face_count += 1

            if style == "🟢 Classic Box":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            else:
                color = tuple(random.choices(range(256), k=3))
                emoji = random.choice(["😎", "🤖", "👽", "🧠", "🎭", "😜"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, emoji, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, color, 2)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img, caption=f"Frame {frame_count}", use_column_width=True)
        time.sleep(0.02)
        frame_count += 1

    cap.release()
    st.success(f"✔️ Finished! Detected {face_count} faces in {frame_count} frames.")
    st.balloons()
else:
    st.warning("📂 Upload a video to begin.")
