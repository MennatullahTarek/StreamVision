import streamlit as st
import cv2
import tempfile
import time
from PIL import Image

st.set_page_config(page_title="🎭 Funky Face Detector", layout="centered")
st.title("🕶️ Funky Real-Time Face Detection (Video Upload)")
st.markdown("Upload a video and choose your detection style ✨")

# Mode selection
mode = st.selectbox("🎨 Choose your detection style:", ["🧑‍🦲 Classic Face Detection", "😎 Funky Mode"])

uploaded_file = st.file_uploader("🎥 Upload your video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())

    st.success("✅ Video uploaded successfully!")
    st.info("⏳ Processing video... please wait")

    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(tfile.name)
    FRAME_WINDOW = st.empty()

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / frame_rate if frame_rate > 0 else 0.03

    frame_count = 0
    detected_faces = 0

    with st.spinner("🎬 Detecting faces in each frame..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                detected_faces += 1
                if mode == "🧑‍🦲 Classic Face Detection":
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                elif mode == "😎 Funky Mode":
                    # Draw sunglasses
                    cv2.rectangle(frame, (x + int(w * 0.15), y + int(h * 0.25)),
                                  (x + int(w * 0.85), y + int(h * 0.45)), (0, 0, 0), -1)
                    # Draw mustache
                    cv2.ellipse(frame, (x + w//2, y + int(h * 0.75)), (w//4, h//10), 0, 0, 180, (0, 0, 0), -1)

            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            FRAME_WINDOW.image(img, caption=f"Frame {frame_count}", use_column_width=True)
            time.sleep(delay)

    cap.release()

    st.success(f"🥳 Done! {detected_faces} faces found in {frame_count} frames.")
    st.balloons()
else:
    st.warning("📂 Please upload a video file to get started.")
