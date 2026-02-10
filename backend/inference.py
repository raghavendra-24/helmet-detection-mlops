"""
YOLOv8 Inference Engine
Handles model loading and real-time helmet detection.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

from .metrics import track_inference_time, update_detection_count, set_model_loaded


class DetectionResult:
    """Container for detection results."""
    
    def __init__(
        self,
        boxes: List[List[float]],
        scores: List[float],
        class_ids: List[int],
        class_names: List[str],
        inference_time_ms: float,
        frame_shape: Tuple[int, int]
    ):
        self.boxes = boxes  # [[x1, y1, x2, y2], ...]
        self.scores = scores
        self.class_ids = class_ids
        self.class_names = class_names
        self.inference_time_ms = inference_time_ms
        self.frame_shape = frame_shape
        self.count = len(boxes)
    
    def to_dict(self) -> Dict:
        return {
            "detections": [
                {
                    "box": box,
                    "score": float(score),
                    "class_id": int(class_id),
                    "class_name": class_name
                }
                for box, score, class_id, class_name 
                in zip(self.boxes, self.scores, self.class_ids, self.class_names)
            ],
            "count": self.count,
            "inference_time_ms": self.inference_time_ms,
            "frame_shape": self.frame_shape
        }


class InferenceEngine:
    """YOLOv8 inference engine for real-time detection."""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to YOLOv8 .pt model file
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        self.model = None
        self.class_names = []
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.class_names = self.model.names
            set_model_loaded(True)
            print(f"✅ Model loaded: {self.model_path}")
            print(f"   Classes: {self.class_names}")
            print(f"   Device: {self.device}")
        except Exception as e:
            set_model_loaded(False)
            raise RuntimeError(f"Failed to load model: {e}")
    
    @track_inference_time
    def predict(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            DetectionResult with boxes, scores, and class info
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
            result = results[0]
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                boxes.append(xyxy)
                scores.append(conf)
                class_ids.append(cls_id)
                class_names.append(self.class_names.get(cls_id, "unknown"))
        
        # Update metrics
        update_detection_count(len(boxes))
        
        return DetectionResult(
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
        result: DetectionResult,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes on frame.
        
        Args:
            frame: BGR image
            result: DetectionResult from predict()
            color: BGR color for boxes
            thickness: Line thickness
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for box, score, class_name in zip(result.boxes, result.scores, result.class_names):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
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
