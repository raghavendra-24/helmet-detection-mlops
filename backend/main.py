"""
FastAPI Backend for Real-Time Helmet Detection
Provides REST API and WebSocket for helmet compliance inference, metrics, and health checks.
"""

import io
import base64
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

# from .inference import InferenceEngine  # Moved to local import to allow ONNX-only runtime
from .drift_detector import DriftDetector
from .metrics import (
    get_metrics,
    get_content_type,
    update_fps,
    update_drift_score
)


# Configuration
MODEL_PATH = Path(__file__).parent.parent / "models" / "helmet_yolov8s_best.pt"
BASELINE_PATH = Path(__file__).parent.parent / "models" / "baseline_stats.json"

# Global instances
engine: Optional[Any] = None
drift_detector: Optional[DriftDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and drift detector on startup."""
    global engine, drift_detector
    
    print("🚀 Starting Real-Time Detection API...")
    
    # Download models from HuggingFace if not present
    from .model_downloader import ensure_models_exist
    ensure_models_exist()
    
    # Load inference engine
    MODEL_PATH_ONNX = Path(__file__).parent.parent / "models" / "best.onnx"
    
    # Priority 1: Semantic ONNX (Lightweight, ideal for Render)
    if MODEL_PATH_ONNX.exists():
        try:
            from .onnx_inference import OnnxInferenceEngine
            engine = OnnxInferenceEngine(
                model_path=str(MODEL_PATH_ONNX),
                confidence_threshold=0.3
            )
            print(f"✅ Loaded ONNX Model: {MODEL_PATH_ONNX}")
        except Exception as e:
            print(f"⚠️ Failed to load ONNX model: {e}")
            
    # Priority 2: PyTorch (Heavy, fallback)
    if engine is None and MODEL_PATH.exists():
        try:
            from .inference import InferenceEngine
            engine = InferenceEngine(
                model_path=str(MODEL_PATH),
                confidence_threshold=0.3,
                device="cpu"
            )
            print(f"✅ Loaded PyTorch Model: {MODEL_PATH}")
        except ImportError:
            print("⚠️ PyTorch/Ultralytics not installed. Install requirements.txt for .pt support.")
        except Exception as e:
            print(f"❌ Failed to load PyTorch model: {e}")
            
    if engine is None:
        print(f"⚠️ No model loaded. Please ensure models are present.")
        print("   Required: models/best.onnx OR models/helmet_yolov8s_best.pt")
    
    # Load drift detector
    if BASELINE_PATH.exists():
        drift_detector = DriftDetector(
            baseline_path=str(BASELINE_PATH),
            threshold=2.0
        )
        print(f"✅ Drift detector loaded with baseline from {BASELINE_PATH}")
    else:
        print(f"⚠️ Baseline stats not found at {BASELINE_PATH}")
    
    yield
    
    print("👋 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Real-Time Helmet Detection API",
    description="YOLOv8-based helmet compliance detection with drift monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class PredictionResponse(BaseModel):
    success: bool
    detections: list
    count: int
    inference_time_ms: float
    drift_score: Optional[float] = None
    drift_status: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    drift_detector_ready: bool


class ModelInfoResponse(BaseModel):
    model_path: str
    device: str
    confidence_threshold: float
    class_names: dict
    num_classes: int


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health."""
    return HealthResponse(
        status="healthy",
        model_loaded=engine is not None,
        drift_detector_ready=drift_detector is not None
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model metadata."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = engine.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    annotate: bool = Query(False, description="Return annotated image")
):
    """
    Run helmet detection on an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG)
        annotate: If True, returns base64 encoded annotated image
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
    # Run inference
    result = engine.predict(frame)
    
    # Check drift
    drift_score = None
    drift_status = None
    if drift_detector is not None:
        drift_score, drift_details = drift_detector.compute_drift_score(frame)
        drift_status = drift_detector.get_drift_status(drift_score)
        update_drift_score(drift_score, drift_status == "alert")
    
    response_data = {
        "success": True,
        "detections": result.to_dict()["detections"],
        "count": result.count,
        "inference_time_ms": result.inference_time_ms,
        "drift_score": drift_score,
        "drift_status": drift_status
    }
    
    # Optionally return annotated image
    if annotate:
        annotated = engine.annotate_frame(frame, result)
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        response_data["annotated_image"] = img_base64
    
    return JSONResponse(content=response_data)


@app.post("/predict-base64")
async def predict_base64(data: dict):
    """
    Run detection on base64 encoded image.
    Useful for Streamlit webcam integration.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        img_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    
    # Run inference
    result = engine.predict(frame)
    
    # Check drift
    drift_score = None
    drift_status = None
    if drift_detector is not None:
        drift_score, drift_details = drift_detector.compute_drift_score(frame)
        drift_status = drift_detector.get_drift_status(drift_score)
        update_drift_score(drift_score, drift_status == "alert")
    
    # Annotate frame
    annotated = engine.annotate_frame(frame, result)
    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "success": True,
        "annotated_image": img_base64,
        "count": result.count,
        "inference_time_ms": result.inference_time_ms,
        "drift_score": drift_score,
        "drift_status": drift_status,
        "detections": result.to_dict()["detections"]
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type=get_content_type()
    )


@app.get("/drift-status")
async def get_drift_status():
    """Get current drift detection status."""
    if drift_detector is None:
        return {"status": "not_configured", "message": "Drift detector not loaded"}
    
    return {
        "status": "ready",
        "threshold": drift_detector.threshold,
        "baseline": drift_detector.baseline,
        "history_size": len(drift_detector.brightness_history)
    }


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection.
    Receives base64 images, returns detection results.
    """
    await websocket.accept()
    print("🔌 WebSocket client connected")
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if engine is None:
                await websocket.send_json({
                    "error": "Model not loaded",
                    "count": 0,
                    "detections": [],
                    "inference_time_ms": 0,
                    "drift_score": 0
                })
                continue
            
            try:
                # Decode base64 image
                img_data = base64.b64decode(message.get("image", ""))
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Invalid image"})
                    continue
                
                # Update confidence if provided
                if "confidence" in message:
                    engine.confidence_threshold = float(message["confidence"])
                
                # Run inference
                result = engine.predict(frame)
                
                # Check drift
                drift_score = 0.0
                drift_status = "normal"
                if drift_detector is not None:
                    drift_score, _ = drift_detector.compute_drift_score(frame)
                    drift_status = drift_detector.get_drift_status(drift_score)
                    update_drift_score(drift_score, drift_status == "alert")
                
                # Send results
                await websocket.send_json({
                    "success": True,
                    "count": result.count,
                    "detections": result.to_dict()["detections"],
                    "inference_time_ms": result.inference_time_ms,
                    "drift_score": drift_score,
                    "drift_status": drift_status
                })
                
            except Exception as e:
                await websocket.send_json({
                    "error": str(e),
                    "count": 0,
                    "detections": [],
                    "inference_time_ms": 0,
                    "drift_score": 0
                })
    
    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")


# Run with: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

