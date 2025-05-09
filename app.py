import streamlit as st

# Set up the app layout and page navigation
st.set_page_config(page_title="AI Vision Streamlit", page_icon="👁️", layout="wide")

# Sidebar with navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Real-time Face Detection", "Real-time Object Detection", "About"])

# Load the respective page based on selection
if selection == "Real-time Face Detection":
    st.markdown("### Real-time Face Detection")
    st.write("This page will display real-time face detection using MediaPipe.")
    st.write("Please check the pages section to navigate to the **Face Detection** page.")
elif selection == "Real-time Object Detection":
    st.markdown("### Real-time Object Detection")
    st.write("This page will display real-time object detection using MediaPipe.")
    st.write("Please check the pages section to navigate to the **Object Detection** page.")
else:
    st.markdown("### About")
    st.write("This is an AI-based Vision Streamlit App using MediaPipe for real-time face and object detection.")
