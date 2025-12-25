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
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Light UI Styling
# --------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #f7f9fc; }

.main-header {
    background: linear-gradient(90deg, #667eea, #764ba2);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
}
.main-header h1 { color: white; margin: 0; }
.main-header p { color: rgba(255,255,255,0.9); }

.image-container {
    background: white;
    border-radius: 14px;
    padding: 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
}

.green { color: #00a65a; }
.yellow { color: #f39c12; }
.red { color: #e74c3c; }

.footer {
    text-align: center;
    padding: 2rem;
    color: #6b7280;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>😷 Face Mask Detection System</h1>
    <p>AI-Powered Surveillance & Compliance Monitoring</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load YOLO Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best_face_mask.onnx")

model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    conf_threshold = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

    mode = st.radio(
        "Detection Mode",
        ["📤 Upload Image", "📁 Batch Processing", "🎥 Upload Video"]
    )

    st.markdown("### 🏷️ Class Labels")
    st.success("😷 With Mask")
    st.warning("⚠️ Incorrect Mask")
    st.error("🚨 Without Mask")

# ==================================================
# IMAGE MODE
# ==================================================
if mode == "📤 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="Original Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Run Detection"):
            start = time.time()
            results = model(img_np, conf=conf_threshold)[0]
            end = time.time()

            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with col2:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(annotated, caption="Detected Output", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### 📊 Detection Summary")

            if len(results.boxes) == 0:
                st.warning("⚠️ No faces detected.")
            else:
                labels = [model.names[int(b.cls[0])] for b in results.boxes]
                counts = Counter(labels)

                for box in results.boxes:
                    label = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])

                    if label == "without_mask":
                        st.error(f"🚨 No Mask ({conf:.2f})")
                    elif label == "mask_weared_incorrect":
                        st.warning(f"⚠️ Incorrect Mask ({conf:.2f})")
                    else:
                        st.success(f"✅ With Mask ({conf:.2f})")

                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"<div class='metric-card'><div class='metric-value green'>{counts.get('with_mask',0)}</div>With Mask</div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div class='metric-value yellow'>{counts.get('mask_weared_incorrect',0)}</div>Incorrect</div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div class='metric-value red'>{counts.get('without_mask',0)}</div>No Mask</div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='metric-card'><div class='metric-value'>{len(results.boxes)}</div>{end-start:.2f}s</div>", unsafe_allow_html=True)

# ==================================================
# BATCH PROCESSING MODE
# ==================================================
elif mode == "📁 Batch Processing":

    files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if files:
        st.info(f"📂 {len(files)} images uploaded")

        if st.button("🚀 Process Batch"):
            total_counts = Counter()
            progress = st.progress(0)

            for i, file in enumerate(files):
                image = Image.open(file).convert("RGB")
                img_np = np.array(image)

                results = model(img_np, conf=conf_threshold)[0]
                labels = [model.names[int(b.cls[0])] for b in results.boxes]
                total_counts.update(labels)

                progress.progress((i + 1) / len(files))

            st.markdown("### 📊 Batch Summary")

            st.success(f"😷 With Mask: {total_counts.get('with_mask', 0)}")
            st.warning(f"⚠️ Incorrect Mask: {total_counts.get('mask_weared_incorrect', 0)}")
            st.error(f"🚨 No Mask: {total_counts.get('without_mask', 0)}")

# ==================================================
# VIDEO MODE
# ==================================================
elif mode == "🎥 Upload Video":

    video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        frame_area = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf_threshold)[0]
            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            frame_area.image(annotated, channels="RGB", use_column_width=True)

        cap.release()
        os.remove(tfile.name)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<div class="footer">
    Face Mask Detection System • YOLOv8 • Streamlit
</div>
""", unsafe_allow_html=True)
