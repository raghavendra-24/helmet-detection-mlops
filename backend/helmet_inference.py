"""
Helmet Detection Inference Utilities
=====================================
Inference helper for helmet compliance detection.

Compatible with the existing InferenceEngine interface pattern.

Author: Raghavendra
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .metrics import track_inference_time, update_detection_count, set_model_loaded


@dataclass
class HelmetDetectionResult:
    """Container for helmet detection results."""
    
    boxes: List[List[float]]      # [[x1, y1, x2, y2], ...]
    scores: List[float]            # Confidence scores
    class_ids: List[int]           # Class IDs
    class_names: List[str]         # Class names
    inference_time_ms: float       # Inference time in milliseconds
    frame_shape: Tuple[int, int]   # (width, height)
    
    @property
    def count(self) -> int:
        """Total detection count."""
        return len(self.boxes)
    
    @property
    def helmet_count(self) -> int:
        """Number of helmets detected."""
        return sum(1 for c in self.class_ids if c == 0)
    
    @property
    def no_helmet_count(self) -> int:
        """Number of no_helmet detections."""
        return sum(1 for c in self.class_ids if c == 1)
    
    @property
    def counts(self) -> Dict[str, int]:
        """Per-class counts."""
        return {
            "helmet": self.helmet_count,
            "no_helmet": self.no_helmet_count
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "detections": [
                {
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": float(score),
                    "bbox": [float(x) for x in box]
                }
                for box, score, class_id, class_name
                in zip(self.boxes, self.scores, self.class_ids, self.class_names)
            ],
            "counts": self.counts,
            "count": self.count,
            "inference_time_ms": self.inference_time_ms,
            "frame_shape": self.frame_shape
        }


class HelmetInferenceEngine:
    """
    Helmet detection inference engine.
    
    Provides a production-ready interface for helmet compliance detection,
    compatible with the existing FastAPI backend.
    
    Example:
        engine = HelmetInferenceEngine("models/helmet_yolov8s_best.pt")
        result = engine.predict(frame)
        print(f"Helmets: {result.helmet_count}, Violations: {result.no_helmet_count}")
    """
    
    # Class configuration
    CLASS_NAMES = {0: "helmet", 1: "no_helmet"}
    
    # Visualization colors (BGR format)
    COLORS = {
        0: (0, 255, 0),    # Green for helmet
        1: (0, 0, 255)     # Red for no_helmet
    }
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """
        Initialize the helmet inference engine.
        
        Args:
            model_path: Path to YOLOv8 model (.pt, .onnx, or .torchscript)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device for inference ("cpu" or "cuda")
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        self.model = None
        self.class_names = self.CLASS_NAMES
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8 model."""
        try:
            from ultralytics import YOLO
            
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Update class names from model if available
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
            
            set_model_loaded(True)
            print(f"✅ Helmet model loaded: {self.model_path}")
            print(f"   Classes: {self.class_names}")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            set_model_loaded(False)
            raise RuntimeError(f"Failed to load helmet model: {e}")
    
    @track_inference_time
    def predict(self, frame: np.ndarray) -> HelmetDetectionResult:
        """
        Run helmet detection on a frame.
        
        Args:
            frame: BGR image as numpy array (OpenCV format)
        
        Returns:
            HelmetDetectionResult with detections and counts
        """
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Extract detections
        boxes = []
        scores = []
        class_ids = []
        class_names = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                boxes.append(xyxy)
                scores.append(conf)
                class_ids.append(cls_id)
                class_names.append(self.class_names.get(cls_id, "unknown"))
        
        # Update metrics
        update_detection_count(len(boxes))
        
        return HelmetDetectionResult(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            inference_time_ms=inference_time_ms,
            frame_shape=(frame.shape[1], frame.shape[0])
        )
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        result: HelmetDetectionResult,
        thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: BGR image
            result: HelmetDetectionResult from predict()
            thickness: Line thickness
            font_scale: Font scale for labels
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for box, score, class_id, class_name in zip(
            result.boxes, result.scores, result.class_ids, result.class_names
        ):
            x1, y1, x2, y2 = map(int, box)
            color = self.COLORS.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background
            label = f"{class_name}: {score:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 5, y1),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text
                1,
                cv2.LINE_AA
            )
        
        # Draw summary overlay
        summary = f"Helmets: {result.helmet_count} | Violations: {result.no_helmet_count}"
        cv2.putText(
            annotated,
            summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        return annotated
    
    def get_model_info(self) -> Dict:
        """Get model metadata."""
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "class_names": self.class_names,
            "num_classes": len(self.class_names)
        }


# Alias for backward compatibility with existing InferenceEngine
InferenceEngine = HelmetInferenceEngine
DetectionResult = HelmetDetectionResult
