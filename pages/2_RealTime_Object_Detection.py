import streamlit as st
import tempfile
import cv2
import numpy as np
import time
from PIL import Image
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="🎯 Object Detection", layout="centered")
st.title("🎯 Real-Time Object Detection with MediaPipe")
st.markdown("Upload a video file and choose how you want it detected 👇")

# Style options
style = st.selectbox("🎨 Choose your detection style:", ["✅ Classic Detection", "🌈 Funky Mode"])

# Upload video
uploaded_video = st.file_uploader("📹 Upload a video file", type=["mp4", "mov", "avi"])

# Load MediaPipe object detection model
@st.cache_resource
def load_model():
    base_options = python.BaseOptions(model_asset_path="efficientdet_lite0.tflite")
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           score_threshold=0.5)
    return vision.ObjectDetector.create_from_options(options)

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    detector = load_model()
    cap = cv2.VideoCapture(temp_video_path)
    FRAME_WINDOW = st.empty()
    total_detections = 0
    frame_count = 0

    st.info("⏳ Processing your video... please wait!")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Correcting the mp_image import
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(mp_image)

        for det in detection_result.detections:
            bbox = det.bounding_box
            startX = int(bbox.origin_x)
            startY = int(bbox.origin_y)
            endX = int(bbox.origin_x + bbox.width)
            endY = int(bbox.origin_y + bbox.height)

            category = det.categories[0].category_name or "object"
            confidence = det.categories[0].score
            total_detections += 1

            if style == "✅ Classic Detection":
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"{category} ({confidence:.2f})", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:  # Funky Mode
                color = tuple(random.choices(range(256), k=3))
                emoji = random.choice(["🎯", "🚀", "💥", "🧠", "🦄", "🍕"])
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                cv2.putText(frame, f"{emoji} {category}", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img, caption=f"Frame {frame_count}", use_column_width=True)
        time.sleep(0.02)
        frame_count += 1

    cap.release()
    st.success(f"🎉 Done! {total_detections} objects detected in {frame_count} frames.")
    st.balloons()
else:
    st.warning("📂 Upload a video to start detection.")
