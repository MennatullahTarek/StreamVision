import streamlit as st

st.set_page_config(page_title="AI Vision Streamlit", page_icon="👁️", layout="wide")

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Real-time Face Detection", "Real-time Object Detection", "About"])

if selection == "Real-time Face Detection":
    st.markdown("### Real-time Face Detection")
    st.write("Navigate to the **pages/1_RealTime_Face_Detection.py** page.")
elif selection == "Real-time Object Detection":
    st.markdown("### Real-time Object Detection")
    st.write("Navigate to the **pages/2_RealTime_Object_Detection.py** page.")
else:
    st.markdown("### About")
    st.write("This is an AI-based Vision Streamlit App using MediaPipe for real-time face and object detection.")
