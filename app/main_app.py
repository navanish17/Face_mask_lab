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
# Modern Dark UI Styling (Look Enhancement Only)
# --------------------------------------------------
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #1e293b);
        color: #f8fafc;
    }

    /* Enhanced Header */
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.3);
    }
    .main-header h1 { 
        font-weight: 800; 
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        margin-bottom: 0.5rem;
    }
    .main-header p { font-size: 1.2rem; opacity: 0.8; letter-spacing: 1px; }

    /* Glass Containers for Images */
    .image-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .image-container:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }

    /* Modern Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 8px;
    }

    /* Accent Colors */
    .green-text { color: #4ade80; }
    .yellow-text { color: #fbbf24; }
    .red-text { color: #f87171; }
    .blue-text { color: #38bdf8; }

    /* Buttons Style */
    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(90deg, #38bdf8, #6366f1);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
    }
    
    .footer {
        text-align: center;
        padding: 3rem;
        opacity: 0.5;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🛡️ MASKGUARD AI</h1>
    <p>Advanced Real-Time Face Mask Detection System</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load YOLO Model (Logic Unchanged)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best_face_mask.onnx")

model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    conf_threshold = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    mode = st.radio(
        "📂 Select Detection Source",
        ["📤 Single Image", "📁 Batch Processing", "🎥 Live Stream/Video"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Live Legend")
    st.success("😷 Mask Detected")
    st.warning("⚠️ Incorrect Position")
    st.error("🚨 No Mask Found")

# ==================================================
# UI LOGIC: SINGLE IMAGE
# ==================================================
if mode == "📤 Single Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### Input Source")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Run Analysis"):
            start = time.time()
            results = model(img_np, conf=conf_threshold)[0]
            end = time.time()

            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with col2:
                st.markdown("#### AI Detection")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(annotated, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### 📉 Real-time Metrics")
            if len(results.boxes) == 0:
                st.warning("No faces detected in the current frame.")
            else:
                labels = [model.names[int(b.cls[0])] for b in results.boxes]
                counts = Counter(labels)

                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"<div class='metric-card'><div class='metric-value green-text'>{counts.get('with_mask',0)}</div>Safe</div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div class='metric-value yellow-text'>{counts.get('mask_weared_incorrect',0)}</div>Risk</div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div class='metric-value red-text'>{counts.get('without_mask',0)}</div>Danger</div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='metric-card'><div class='metric-value blue-text'>{end-start:.2f}s</div>Latency</div>", unsafe_allow_html=True)

# ==================================================
# UI LOGIC: BATCH PROCESSING
# ==================================================
elif mode == "📁 Batch Processing":
    files = st.file_uploader("Upload Dataset", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files:
        if st.button("🔥 Process All Images"):
            total_counts = Counter()
            progress_bar = st.progress(0)

            for i, file in enumerate(files):
                img = Image.open(file).convert("RGB")
                res = model(np.array(img), conf=conf_threshold)[0]
                total_counts.update([model.names[int(b.cls[0])] for b in res.boxes])
                progress_bar.progress((i + 1) / len(files))

            st.markdown("### 📈 Cumulative Batch Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Safe (Mask)", total_counts.get('with_mask', 0))
            c2.metric("Total Warnings", total_counts.get('mask_weared_incorrect', 0))
            c3.metric("Total Violations", total_counts.get('without_mask', 0))

# ==================================================
# UI LOGIC: VIDEO MODE
# ==================================================
elif mode == "🎥 Live Stream/Video":
    video = st.file_uploader("Upload Footage", type=["mp4", "mov", "avi"])

    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        st.info("🔄 Initializing Neural Network Engine...")
        frame_area = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = model(frame, conf=conf_threshold)[0]
            annotated = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
            frame_area.image(annotated, use_column_width=True)

        cap.release()
        os.remove(tfile.name)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(f"""
<div class="footer">
    MaskGuard AI Engine v2.5 • Developed for Health Compliance • {time.strftime("%Y")}
</div>
""", unsafe_allow_html=True)
