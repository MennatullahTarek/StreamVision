import streamlit as st

st.title("About")

st.write("""
This Streamlit app demonstrates real-time face and object detection using MediaPipe and OpenCV.

It uses your webcam to detect faces (using MediaPipe's Face Detection solution) and displays the results in real-time.

**Note:** Object detection is currently simulated. You may extend it with YOLO or TensorFlow Lite for better object detection capabilities.
""")

st.markdown("""
**Technologies used:**
- [Streamlit](https://streamlit.io)
- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)

Created with ❤️ by MennatullahTarek.
""")
