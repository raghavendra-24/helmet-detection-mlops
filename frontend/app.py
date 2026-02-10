"""
Streamlit Frontend for Real-Time Helmet Detection
Interactive webcam demo with metrics dashboard.
"""

import streamlit as st
import cv2
import numpy as np
import time
import requests
import base64
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.inference import InferenceEngine
from backend.drift_detector import DriftDetector
from backend.model_downloader import ensure_models_exist

# Ensure models are downloaded (for cloud deployment)
ensure_models_exist()

# Configuration
MODEL_PATH = Path(__file__).parent.parent / "models" / "helmet_yolov8s_best.pt"
BASELINE_PATH = Path(__file__).parent.parent / "models" / "baseline_stats.json"

# Page config
st.set_page_config(
    page_title="Helmet Detection | MLOps Demo",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
    .status-normal { color: #00ff00; }
    .status-warning { color: #ffff00; }
    .status-alert { color: #ff0000; }
    .drift-alert {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load detection model (cached)."""
    if not MODEL_PATH.exists():
        return None
    return InferenceEngine(
        model_path=str(MODEL_PATH),
        confidence_threshold=0.5,
        device="cpu"
    )


@st.cache_resource
def load_drift_detector():
    """Load drift detector (cached)."""
    if not BASELINE_PATH.exists():
        return None
    return DriftDetector(
        baseline_path=str(BASELINE_PATH),
        threshold=2.0
    )


    # Header
    st.markdown('<p class="header-gradient">🦺 Real-Time Helmet Detection</p>', unsafe_allow_html=True)
    st.markdown("*YOLOv8 with MLOps Monitoring*")
    st.markdown("---")
    
    # Load model and detector
    engine = load_model()
    drift_detector = load_drift_detector()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model status
        st.subheader("Model Status")
        if engine:
            st.success("✅ Model Loaded")
            st.json(engine.get_model_info())
        else:
            st.error("❌ Model Not Found")
            st.info(f"Place model at:\n`{MODEL_PATH}`")
        
        st.markdown("---")
        
        # Drift detector status
        st.subheader("Drift Detector")
        if drift_detector:
            st.success("✅ Baseline Loaded")
            st.json(drift_detector.baseline)
        else:
            st.warning("⚠️ No Baseline")
        
        st.markdown("---")
        
        # Controls
        st.subheader("Controls")
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        drift_threshold = st.slider("Drift Threshold", 1.0, 5.0, 2.0, 0.5)
        
        if drift_detector:
            drift_detector.threshold = drift_threshold
    
    # Main content
    if engine is None:
        st.error("⚠️ Model not loaded. Please place your model files in the `models/` directory.")
        ### Required Files:
        1. `models/helmet_yolov8s_best.pt` - YOLOv8 model
        2. `models/baseline_stats.json` - Baseline statistics for drift detection
        """)
        return
    
    # Update confidence threshold
    engine.confidence_threshold = confidence
    
    # Tabs for different modes
    tab1, tab2, tab3 = st.tabs(["📷 Webcam", "📁 Upload Image", "📊 Metrics"])
    
    with tab1:
        st.subheader("Live Webcam Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Webcam placeholder
            video_placeholder = st.empty()
            
        with col2:
            # Live metrics
            st.markdown("### 📊 Live Metrics")
            fps_metric = st.empty()
            latency_metric = st.empty()
            count_metric = st.empty()
            drift_metric = st.empty()
            alert_placeholder = st.empty()
        
        # Start/Stop button
        run_detection = st.checkbox("🎬 Start Detection", value=False)
        
        if run_detection:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not access webcam. Please check permissions.")
            else:
                fps_history = []
                
                while run_detection:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to read frame")
                        break
                    
                    # Run inference
                    start_time = time.time()
                    result = engine.predict(frame)
                    inference_time = time.time() - start_time
                    
                    # Calculate FPS
                    fps = 1.0 / inference_time if inference_time > 0 else 0
                    fps_history.append(fps)
                    if len(fps_history) > 30:
                        fps_history.pop(0)
                    avg_fps = np.mean(fps_history)
                    
                    # Check drift
                    drift_score = 0.0
                    drift_status = "normal"
                    if drift_detector:
                        drift_score, drift_details = drift_detector.compute_drift_score(frame)
                        drift_status = drift_detector.get_drift_status(drift_score)
                    
                    # Annotate frame
                    annotated = engine.annotate_frame(frame, result)
                    
                    # Add overlay info
                    cv2.putText(
                        annotated,
                        f"FPS: {avg_fps:.1f} | Objects: {result.count} | Drift: {drift_score:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    # Display frame
                    video_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )
                    
                    # Update metrics
                    fps_metric.metric("⚡ FPS", f"{avg_fps:.1f}")
                    latency_metric.metric("⏱️ Latency", f"{result.inference_time_ms:.1f} ms")
                    count_metric.metric("👥 Detections", result.count)
                    drift_metric.metric("📉 Drift Score", f"{drift_score:.2f}")
                    
                    # Show alert if drift detected
                    if drift_status == "alert":
                        alert_placeholder.markdown(
                            '<div class="drift-alert">⚠️ DRIFT DETECTED! Input distribution has shifted.</div>',
                            unsafe_allow_html=True
                        )
                    elif drift_status == "warning":
                        alert_placeholder.warning("⚠️ Drift Warning: Distribution shift detected")
                    else:
                        alert_placeholder.empty()
                    
                    # Check if checkbox is still checked
                    time.sleep(0.01)
                
                cap.release()
    
    with tab2:
        st.subheader("Upload Image for Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            key="image_upload"
        )
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Run detection
            result = engine.predict(image)
            annotated = engine.annotate_frame(image, result)
            
            # Check drift
            drift_score = 0.0
            drift_status = "normal"
            if drift_detector:
                drift_score, drift_details = drift_detector.compute_drift_score(image)
                drift_status = drift_detector.get_drift_status(drift_score)
            
            with col2:
                st.markdown("**Detected Objects**")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Results
            st.markdown("### Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("👥 Detections", result.count)
            col2.metric("⏱️ Inference", f"{result.inference_time_ms:.1f} ms")
            col3.metric("📉 Drift Score", f"{drift_score:.2f}")
            col4.metric("🚦 Status", drift_status.upper())
            
            if drift_status == "alert":
                st.error("⚠️ DRIFT ALERT: This image shows significant distribution shift from training data!")
    
    with tab3:
        st.subheader("📊 Model Performance Metrics")
        
        # Training metrics
        st.markdown("### Training Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("mAP@50", "66.2%")
        col2.metric("Precision", "80.4%")
        col3.metric("Recall", "56.0%")
        col4.metric("Inference", "8.5 ms")
        
        st.markdown("---")
        
        # Prometheus endpoint info
        st.markdown("### 📈 Monitoring Endpoints")
        st.code("""
# Prometheus metrics
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info
        """)
        
        st.info("💡 Run the FastAPI backend to access Prometheus metrics: `uvicorn backend.main:app --reload`")


if __name__ == "__main__":
    main()
