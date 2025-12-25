import streamlit as st
import numpy as np
import time
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from collections import Counter

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="MaskGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Modern Dark UI Styling
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: radial-gradient(circle at top left, #1e293b, #0f172a); color: #f8fafc; }
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { 
        font-weight: 800; 
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
    }
    .image-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; }
    .cyan-text { color: #38bdf8; }
    .green-text { color: #4ade80; }
    .yellow-text { color: #fbbf24; }
    .red-text { color: #f87171; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load YOLO Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    # task='detect' is important for ONNX models
    return YOLO("best_face_mask.onnx", task='detect')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error: Model file 'best_face_mask.onnx' not found. {e}")
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2576/2576762.png", width=80)
    st.markdown("### 🛠️ Control Panel")
    conf_threshold = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    mode = st.selectbox("Select Workflow", ["📸 Single Image", "📂 Batch Processing", "📽️ Video Analysis"])
    st.markdown("---")
    st.info("✅ Green: Mask | ⚠️ Yellow: Improper | 🚨 Red: No Mask")

# ==================================================
# UI LOGIC: IMAGE MODE
# ==================================================
if mode == "📸 Single Image":
    uploaded_file = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        # 1. Convert to RGB to avoid Alpha channel/TypeError
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### Input Source")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            # Fix for TypeError: Ensure image is valid before display
            if image:
                st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Analyze Framework"):
            with st.spinner("Processing..."):
                start = time.time()
                results = model.predict(img_np, conf=conf_threshold)[0]
                end = time.time()

                # YOLO plot uses BGR, convert to RGB for Streamlit
                annotated = results.plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                with col2:
                    st.markdown("#### AI Inference")
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(annotated_rgb, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Metrics Row
            st.markdown("---")
            if len(results.boxes) == 0:
                st.warning("No subjects detected in the frame.")
            else:
                labels = [model.names[int(b.cls[0])] for b in results.boxes]
                counts = Counter(labels)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"<div class='metric-card'><div class='metric-value green-text'>{counts.get('with_mask',0)}</div><small>SAFE</small><br>With Mask</div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div class='metric-value yellow-text'>{counts.get('mask_weared_incorrect',0)}</div><small>RISK</small><br>Incorrect</div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div class='metric-value red-text'>{counts.get('without_mask',0)}</div><small>DANGER</small><br>No Mask</div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='metric-card'><div class='metric-value cyan-text'>{end-start:.2f}s</div><small>LATENCY</small><br>Time</div>", unsafe_allow_html=True)

# ==================================================
# UI LOGIC: BATCH MODE
# ==================================================
elif mode == "📂 Batch Processing":
    files = st.file_uploader("Upload Image Dataset", type=["jpg", "png"], accept_multiple_files=True)

    if files:
        if st.button("🔥 Process All"):
            total_counts = Counter()
            progress_bar = st.progress(0)
            
            for i, file in enumerate(files):
                img = Image.open(file).convert("RGB")
                res = model.predict(np.array(img), conf=conf_threshold)[0]
                total_counts.update([model.names[int(b.cls[0])] for b in res.boxes])
                progress_bar.progress((i + 1) / len(files))

            st.markdown("### 📈 Cumulative Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Safe", total_counts.get('with_mask', 0))
            c2.metric("Total Warnings", total_counts.get('mask_weared_incorrect', 0))
            c3.metric("Total Violations", total_counts.get('without_mask', 0))

# ==================================================
# UI LOGIC: VIDEO MODE
# ==================================================
elif mode == "📽️ Video Analysis":
    video = st.file_uploader("Upload Security Footage", type=["mp4", "mov", "avi"])

    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st.info("🔄 Processing stream... please wait.")
        frame_area = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Inference on each frame
            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            annotated = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
            frame_area.image(annotated, use_container_width=True)

        cap.release()
        os.remove(tfile.name)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(f"""
<div style="text-align: center; margin-top: 5rem; padding: 2rem; opacity: 0.6;">
    MaskGuard AI v2.0 • Powered by YOLOv8 & Streamlit • {time.strftime("%Y")}
</div>
""", unsafe_allow_html=True)
