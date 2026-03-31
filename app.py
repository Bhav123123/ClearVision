"""
Smart Vision — Face Detection System
=====================================
Main Streamlit application entry point.
Run with:  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from datetime import datetime

from modules.detector import FaceDetector
from modules.preprocessor import Preprocessor
from modules.annotator import Annotator
from modules.attendance import AttendanceLogger
from modules.reporter import ReportGenerator
from utils.helpers import get_frame_fps, overlay_stats

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Vision | Face Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0D1B2A; }
    .metric-box {
        background: #1E2D40; border-radius: 10px;
        padding: 12px; text-align: center; color: white;
    }
    .title-text { color: #1E88E5; font-size: 36px; font-weight: bold; }
    .sub-text   { color: #90CAF9; font-size: 14px; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00E5FF; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "session_start"    not in st.session_state:
    st.session_state.session_start = datetime.now()
if "running"          not in st.session_state:
    st.session_state.running = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/280x80/0D1B2A/1E88E5?text=Smart+Vision", width=280)
    st.markdown("---")

    st.subheader("⚙️ Configuration")
    model_choice = st.selectbox(
        "Detection Model",
        ["Haar Cascade (Fast)", "HOG + SVM (Balanced)", "CNN / dlib (Accurate)"],
        index=1,
    )
    model_map = {
        "Haar Cascade (Fast)":    "haar",
        "HOG + SVM (Balanced)":   "hog",
        "CNN / dlib (Accurate)":  "cnn",
    }
    selected_model = model_map[model_choice]

    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    dedup_interval       = st.slider("De-dup Interval (sec)", 5, 120, 30, 5)
    max_faces            = st.slider("Max Faces to Display", 1, 20, 10)

    st.markdown("---")
    st.subheader("📥 Input Source")
    input_type = st.radio("Select Input", ["📷 Live Webcam", "🎬 Upload Video", "🖼️ Upload Image"])

    st.markdown("---")
    st.subheader("📊 Session Stats")
    stat_detections = st.empty()
    stat_uptime     = st.empty()
    stat_fps        = st.empty()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="title-text">👁️ Smart Vision — Face Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Real-time face detection powered by OpenCV · dlib · MediaPipe</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Layout columns ────────────────────────────────────────────────────────────
col_feed, col_info = st.columns([3, 1])

with col_info:
    st.subheader("📈 Live Metrics")
    m1 = st.empty()
    m2 = st.empty()
    m3 = st.empty()
    st.markdown("---")
    st.subheader("📋 Attendance Log")
    attendance_table = st.empty()
    st.markdown("---")
    btn_report = st.button("📥 Generate Report")
    download_area = st.empty()

with col_feed:
    feed_placeholder = st.empty()
    status_bar       = st.empty()

# ── Initialize modules ────────────────────────────────────────────────────────
@st.cache_resource
def load_modules(model):
    detector   = FaceDetector(model=model)
    preprocessor = Preprocessor(target_size=(640, 480))
    annotator  = Annotator()
    logger     = AttendanceLogger(log_path="logs/attendance.csv", dedup_interval=dedup_interval)
    return detector, preprocessor, annotator, logger

detector, preprocessor, annotator, logger = load_modules(selected_model)
reporter = ReportGenerator(log_path="logs/attendance.csv", report_dir="reports")

# ── Process single frame ──────────────────────────────────────────────────────
def process_frame(raw_frame):
    preprocessed = preprocessor.process(raw_frame)
    faces        = detector.detect(preprocessed, max_faces=max_faces)
    annotated    = annotator.draw(raw_frame.copy(), faces)
    annotated    = overlay_stats(annotated, len(faces))

    for i, face in enumerate(faces):
        logged = logger.log(face_id=f"FACE_{i:03d}", confidence=face.get("confidence", 1.0))
        if logged:
            st.session_state.total_detections += 1
            annotator.save_snapshot(raw_frame, face, snap_dir="snapshots")

    return annotated, faces

# ── WEBCAM INPUT ──────────────────────────────────────────────────────────────
if input_type == "📷 Live Webcam":
    start_col, stop_col = st.columns(2)
    with start_col:
        if st.button("▶️ Start Detection", type="primary"):
            st.session_state.running = True
    with stop_col:
        if st.button("⏹️ Stop"):
            st.session_state.running = False

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Webcam not found. Please check your camera connection.")
            st.session_state.running = False
        else:
            prev_time = time.time()
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    status_bar.warning("⚠️ Frame read failed — retrying...")
                    continue

                annotated, faces = process_frame(frame)
                fps = get_frame_fps(prev_time)
                prev_time = time.time()

                # Display
                feed_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_column_width=True
                )
                status_bar.success(f"🟢 LIVE  |  Faces: {len(faces)}  |  FPS: {fps:.1f}")

                # Metrics
                m1.metric("Faces This Frame", len(faces))
                m2.metric("Total Detections", st.session_state.total_detections)
                uptime = int((datetime.now() - st.session_state.session_start).total_seconds())
                m3.metric("Session Uptime", f"{uptime // 60}m {uptime % 60}s")

                # Sidebar
                stat_detections.metric("Detections", st.session_state.total_detections)
                stat_fps.metric("FPS", f"{fps:.1f}")

                # Attendance table
                df = logger.get_dataframe()
                if not df.empty:
                    attendance_table.dataframe(df.tail(10), use_container_width=True)

            cap.release()

# ── VIDEO INPUT ───────────────────────────────────────────────────────────────
elif input_type == "🎬 Upload Video":
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())

        cap   = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prog  = st.progress(0)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 2 == 0:   # process every other frame for speed
                annotated, faces = process_frame(frame)
                feed_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_column_width=True
                )
                m1.metric("Faces This Frame", len(faces))
                m2.metric("Total Detections", st.session_state.total_detections)

            prog.progress(min(frame_idx / max(total, 1), 1.0))
            frame_idx += 1

        cap.release()
        os.unlink(tfile.name)
        status_bar.success("✅ Video processing complete!")

        df = logger.get_dataframe()
        if not df.empty:
            attendance_table.dataframe(df, use_container_width=True)

# ── IMAGE INPUT ───────────────────────────────────────────────────────────────
elif input_type == "🖼️ Upload Image":
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_img:
        img_array = np.array(Image.open(uploaded_img).convert("RGB"))
        frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        annotated, faces = process_frame(frame_bgr)
        feed_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB", use_column_width=True
        )
        m1.metric("Faces Detected", len(faces))
        m2.metric("Total Detections", st.session_state.total_detections)
        status_bar.success(f"✅ Detection complete — {len(faces)} face(s) found.")

        df = logger.get_dataframe()
        if not df.empty:
            attendance_table.dataframe(df, use_container_width=True)

# ── Report Generation ─────────────────────────────────────────────────────────
if btn_report:
    with st.spinner("Generating report..."):
        report_path = reporter.generate()
    if report_path and os.path.exists(report_path):
        with open(report_path, "rb") as f:
            download_area.download_button(
                "⬇️ Download Report CSV",
                data=f,
                file_name=os.path.basename(report_path),
                mime="text/csv",
            )
        st.success("✅ Report ready!")
    else:
        st.warning("⚠️ No attendance data yet. Run detection first.")
