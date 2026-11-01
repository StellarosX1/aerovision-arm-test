import tempfile
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Page Configuration
st.set_page_config(page_title="Drone SAR Survivor Detection Dashboard", layout="wide")
st.title("🚁 Drone SAR — Survivor Detection Dashboard")

# Load YOLO Model (Update path to your visdrone_best.pt if needed)
@st.cache_resource
def load_detection_model(model_path="visdrone_best.pt"):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        return None

model = load_detection_model("visdrone_best.pt")

# Sidebar Controls
st.sidebar.header("Media Selection")
source_type = st.sidebar.radio("Choose Input Type:", ("Image", "Video"))
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

def detect_survivors(frame):
    """Runs survivor detection on a single image frame using YOLO."""
    if model is None:
        return frame, 0
    
    results = model.predict(frame, conf=confidence_threshold, verbose=False)
    annotated_frame = results[0].plot()
    
    # Count detections (assuming class 0 or person detections)
    detections_count = len(results[0].boxes)
    return annotated_frame, detections_count

# Image Processing Mode
if source_type == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Input")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

        with col2:
            st.subheader("Survivor Detection Output")
            processed_image, count = detect_survivors(image)
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.success(f"Detections found: **{count}**")

# Video Processing Mode
elif source_type == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        status_text = st.empty()
        
        stop_btn = st.button("Stop Processing")

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, count = detect_survivors(frame)
            st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            status_text.markdown(f"**Current Frame Detections:** {count}")

        cap.release()