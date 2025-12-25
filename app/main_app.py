import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# 1. Page Config
st.set_page_config(page_title="MaskGuard AI", layout="wide")

# 2. Model Loading (Safety First)
@st.cache_resource
def load_yolo_model():
    try:
        # Use .pt if you have it, or stick to .onnx
        return YOLO("best_face_mask.onnx", task='detect')
    except Exception as e:
        st.error(f"Model file not found! Error: {e}")
        return None

model = load_yolo_model()

st.title("🛡️ MaskGuard AI (Stable Version)")

# 3. Sidebar Setup
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)
mode = st.sidebar.selectbox("Mode", ["Image", "Video"])

# 4. Image Logic
if mode == "Image":
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file:
        # Load image safely
        img = Image.open(file).convert("RGB")
        img_array = np.array(img)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Source")
            # 2025 Fix: Using width="stretch" instead of use_container_width
            st.image(img, width="stretch") 
            
        if st.button("Run Detection"):
            if model:
                results = model.predict(img_array, conf=conf_threshold)[0]
                # Get the plotted image (BGR from OpenCV)
                res_plotted = results.plot()
                # Convert BGR to RGB
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("Result")
                    st.image(res_rgb, width="stretch")
            else:
                st.error("Model not loaded correctly.")

# 5. Video Logic
elif mode == "Video":
    video_file = st.file_uploader("Upload Video", type=['mp4', 'mov'])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO
            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            # Convert BGR (OpenCV) to RGB (Streamlit)
            res_frame = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
            
            # Show frame
            st_frame.image(res_frame, width="stretch")
            
        cap.release()
        os.remove(tfile.name)
