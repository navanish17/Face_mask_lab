import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from collections import Counter

# 1. Basic Config
st.set_page_config(page_title="MaskGuard AI", layout="wide")

# 2. Model Loading
@st.cache_resource
def load_model():
    # task='detect' is mandatory for ONNX models in 2025
    return YOLO("best_face_mask.onnx", task='detect')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model error: {e}")
    st.stop()

st.title("🛡️ MaskGuard AI - Stable Version")

# 3. Sidebar
with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence", 0.1, 1.0, 0.5)
    mode = st.selectbox("Mode", ["Image", "Video"])

# 4. Image Mode
if mode == "Image":
    up_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if up_file:
        # Step 1: PIL Image load aur RGB conversion (v. important)
        img = Image.open(up_file).convert("RGB")
        img_array = np.array(img)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Input")
            # 2025 Fix: width="stretch" use karein ya st.write(img)
            st.image(img, use_container_width=True) 

        if st.button("Start Analysis"):
            # Step 2: Prediction
            results = model.predict(img_array, conf=conf)[0]
            
            # Step 3: YOLO plot generates BGR, must convert to RGB
            res_plotted = results.plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.write("Output")
                st.image(res_rgb, use_container_width=True)

            # Step 4: Results count
            if len(results.boxes) > 0:
                labels = [model.names[int(b.cls[0])] for b in results.boxes]
                counts = Counter(labels)
                st.success(f"Detected: {dict(counts)}")
            else:
                st.warning("No faces detected.")

# 5. Video Mode
elif mode == "Video":
    v_file = st.file_uploader("Upload Video", type=["mp4", "mov"])
    if v_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(v_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Prediction on frame
            res = model.predict(frame, conf=conf, verbose=False)[0]
            # Convert BGR (OpenCV) to RGB (Streamlit)
            res_frame = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
            
            frame_placeholder.image(res_frame, use_container_width=True)
            
        cap.release()
        os.remove(tfile.name)
