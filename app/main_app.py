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
# Modern Dark/Glass UI Styling
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
# Header
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🛡️ MASKGUARD AI</h1>
    <p style="font-size: 1.1rem; opacity: 0.8;">Real-time Computer Vision for Mask Compliance</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load YOLO Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    # Make sure 'best_face_mask.onnx' exists in your root folder
    return YOLO("best_face_mask.onnx", task='detect')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2576/2576762.png", width=80)
    st.markdown("### 🛠️ Settings")
    conf_threshold = st.slider("🎯 Confidence", 0.1, 1.0, 0.45, 0.05)
    mode = st.selectbox("Workflow", ["📸 Single Image", "📂 Batch Processing", "📽️ Video Analysis"])
    st.markdown("---")
    st.info("✅ Green: Safe | ⚠️ Yellow: Risk | 🚨 Red: No Mask")

# ==================================================
# LOGIC: SINGLE IMAGE
# ==================================================
if mode == "📸 Single Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        # Load and fix orientation/format
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Source")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Run Detection"):
            with st.spinner('AI is processing...'):
                results = model.predict(img_array, conf=conf_threshold)[0]
                
                # Plot results and convert BGR (OpenCV) to RGB (Streamlit)
                annotated_img = results.plot() 
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                with col2:
                    st.markdown("#### Result")
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(annotated_img_rgb, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Metrics
            if len(results.boxes) > 0:
                labels = [model.names[int(b.cls[0])] for b in results.boxes]
                counts = Counter(labels)
                
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"<div class='metric-card'><div class='metric-value green-text'>{counts.get('with_mask',0)}</div>Safe</div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div class='metric-value yellow-text'>{counts.get('mask_weared_incorrect',0)}</div>Warning</div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div class='metric-value red-text'>{counts.get('without_mask',0)}</div>Danger</div>", unsafe_allow_html=True)
            else:
                st.warning("No faces detected.")

# ==================================================
# LOGIC: BATCH PROCESSING
# ==================================================
elif mode == "📂 Batch Processing":
    files = st.file_uploader("Upload multiple images", type=["jpg", "png"], accept_multiple_files=True)
    if files and st.button("Start Batch Processing"):
        total_counts = Counter()
        bar = st.progress(0)
        
        for i, file in enumerate(files):
            img = Image.open(file).convert("RGB")
            res = model.predict(np.array(img), conf=conf_threshold)[0]
            total_counts.update([model.names[int(b.cls[0])] for b in res.boxes])
            bar.progress((i + 1) / len(files))

        st.success("Processing Complete!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Safe", total_counts.get('with_mask', 0))
        c2.metric("Total Warnings", total_counts.get('mask_weared_incorrect', 0))
        c3.metric("Total Violations", total_counts.get('without_mask', 0))

# ==================================================
# LOGIC: VIDEO ANALYSIS
# ==================================================
elif mode == "📽️ Video Analysis":
    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st.info("Press 'Stop' in the top right to interrupt.")
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # YOLO inference on frame
            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_container_width=True)

        cap.release()
        os.remove(tfile.name)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(f"""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; opacity: 0.5; font-size: 0.8rem;">
    MaskGuard AI • {time.strftime("%Y")} • Built with Ultralytics ONNX
</div>
""", unsafe_allow_html=True)
