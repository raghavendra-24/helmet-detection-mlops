"""
Helmet Detection Model Abstraction
===================================
Clean, modular model abstraction for YOLOv8-based helmet compliance detection.

Supports:
- PyTorch (.pt)
- ONNX (.onnx)
- TorchScript (.torchscript)

Author: Raghavendra
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass, field


class ModelBackend(Enum):
    """Supported model backends."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"


class ModelMode(Enum):
    """Model operation modes."""
    TRAINING = "training"
    INFERENCE = "inference"


@dataclass
class Detection:
    """Single detection result."""
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    
    def to_dict(self) -> Dict:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": [round(x, 2) for x in self.bbox]
        }


@dataclass
class DetectionOutput:
    """Structured detection output."""
    detections: List[Detection] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    frame_shape: Tuple[int, int] = (0, 0)
    
    def to_dict(self) -> Dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "counts": self.counts,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "frame_shape": self.frame_shape
        }


class HelmetDetectionModel:
    """
    Helmet compliance detection model abstraction.
    
    Provides a unified interface for loading and running inference
    with YOLOv8 models across different backends.
    
    Example:
        model = HelmetDetectionModel("helmet_yolov8s_best.pt")
        model.set_mode(ModelMode.INFERENCE)
        result = model.predict(frame, confidence=0.5)
        print(result.counts)  # {'helmet': 5, 'no_helmet': 2}
    """
    
    # Class configuration
    CLASS_NAMES = {0: "helmet", 1: "no_helmet"}
    CLASS_IDS = {"helmet": 0, "no_helmet": 1}
    
    def __init__(
        self,
        model_path: str,
        backend: Optional[ModelBackend] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        """
        Initialize the helmet detection model.
        
        Args:
            model_path: Path to model file (.pt, .onnx, or .torchscript)
            backend: Model backend (auto-detected if None)
            device: Device for inference ("cpu", "cuda", "cuda:0")
            confidence_threshold: Default confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.mode = ModelMode.INFERENCE
        
        # Auto-detect backend from file extension
        if backend is None:
            backend = self._detect_backend()
        self.backend = backend
        
        # Model instance (lazy loaded)
        self._model = None
        self._load_model()
    
    def _detect_backend(self) -> ModelBackend:
        """Detect backend from file extension."""
        suffix = self.model_path.suffix.lower()
        if suffix == ".pt":
            return ModelBackend.PYTORCH
        elif suffix == ".onnx":
            return ModelBackend.ONNX
        elif suffix in [".torchscript", ".pt"]:
            # Check if it's torchscript by trying to load
            return ModelBackend.PYTORCH
        else:
            raise ValueError(f"Unknown model format: {suffix}")
    
    def _load_model(self):
        """Load the model based on backend."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if self.backend == ModelBackend.PYTORCH:
            self._load_pytorch_model()
        elif self.backend == ModelBackend.ONNX:
            self._load_onnx_model()
        elif self.backend == ModelBackend.TORCHSCRIPT:
            self._load_torchscript_model()
        
        print(f"✅ Model loaded: {self.model_path}")
        print(f"   Backend: {self.backend.value}")
        print(f"   Device: {self.device}")
        print(f"   Classes: {list(self.CLASS_NAMES.values())}")
    
    def _load_pytorch_model(self):
        """Load PyTorch/Ultralytics model."""
        from ultralytics import YOLO
        self._model = YOLO(str(self.model_path))
        
        if self.device != "cpu":
            self._model.to(self.device)
    
    def _load_onnx_model(self):
        """Load ONNX model."""
        import onnxruntime as ort
        
        providers = ['CPUExecutionProvider']
        if 'cuda' in self.device.lower():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self._model = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        print(f"   ONNX providers: {self._model.get_providers()}")
    
    def _load_torchscript_model(self):
        """Load TorchScript model."""
        import torch
        self._model = torch.jit.load(str(self.model_path))
        
        if self.device != "cpu":
            self._model = self._model.to(self.device)
        
        self._model.eval()
    
    def set_mode(self, mode: ModelMode):
        """
        Set the model operation mode.
        
        Args:
            mode: TRAINING or INFERENCE
        """
        self.mode = mode
        
        if self.backend == ModelBackend.PYTORCH and hasattr(self._model, 'model'):
            if mode == ModelMode.TRAINING:
                self._model.model.train()
            else:
                self._model.model.eval()
    
    def predict(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        iou: Optional[float] = None
    ) -> DetectionOutput:
        """
        Run inference on a single frame.
        
        Args:
            frame: BGR image as numpy array (OpenCV format)
            confidence: Confidence threshold (uses default if None)
            iou: IoU threshold for NMS (uses default if None)
        
        Returns:
            DetectionOutput with detections and counts
        """
        import time
        
        conf = confidence or self.confidence_threshold
        iou_thresh = iou or self.iou_threshold
        
        start_time = time.time()
        
        if self.backend == ModelBackend.PYTORCH:
            detections = self._predict_pytorch(frame, conf, iou_thresh)
        elif self.backend == ModelBackend.ONNX:
            detections = self._predict_onnx(frame, conf, iou_thresh)
        elif self.backend == ModelBackend.TORCHSCRIPT:
            detections = self._predict_torchscript(frame, conf, iou_thresh)
        else:
            detections = []
        
        inference_time = (time.time() - start_time) * 1000
        
        # Resolve conflicting detections
        detections = self._resolve_conflicts(detections)
        
        # Count per class
        counts = {"helmet": 0, "no_helmet": 0}
        for det in detections:
            counts[det.class_name] += 1
        
        return DetectionOutput(
            detections=detections,
            counts=counts,
            inference_time_ms=inference_time,
            frame_shape=(frame.shape[1], frame.shape[0])
        )
    
    def _predict_pytorch(
        self,
        frame: np.ndarray,
        conf: float,
        iou: float
    ) -> List[Detection]:
        """Run PyTorch inference."""
        results = self._model.predict(
            frame,
            conf=conf,
            iou=iou,
            verbose=False
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                score = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                detections.append(Detection(
                    class_name=self.CLASS_NAMES.get(cls_id, "unknown"),
                    class_id=cls_id,
                    confidence=score,
                    bbox=xyxy
                ))
        
        return detections
    
    def _predict_onnx(
        self,
        frame: np.ndarray,
        conf: float,
        iou: float
    ) -> List[Detection]:
        """Run ONNX inference."""
        # Preprocess
        input_tensor = self._preprocess_onnx(frame)
        
        # Run inference
        input_name = self._model.get_inputs()[0].name
        output_name = self._model.get_outputs()[0].name
        
        outputs = self._model.run([output_name], {input_name: input_tensor})
        
        # Postprocess
        return self._postprocess_onnx(outputs[0], frame.shape, conf, iou)
    
    def _preprocess_onnx(self, frame: np.ndarray, imgsz: int = 640) -> np.ndarray:
        """Preprocess image for ONNX inference."""
        # Resize with letterbox
        h, w = frame.shape[:2]
        scale = min(imgsz / h, imgsz / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad to square
        pad_h = (imgsz - new_h) // 2
        pad_w = (imgsz - new_w) // 2
        
        padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Convert to tensor format
        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
        tensor = np.expand_dims(tensor, 0)  # Add batch dim
        
        return tensor
    
    def _postprocess_onnx(
        self,
        output: np.ndarray,
        original_shape: Tuple[int, int, int],
        conf: float,
        iou: float,
        imgsz: int = 640
    ) -> List[Detection]:
        """Postprocess ONNX output."""
        # Output shape: (1, 6, 8400) for YOLOv8
        # 6 = x, y, w, h, class0_conf, class1_conf
        predictions = output[0].T  # (8400, 6)
        
        detections = []
        h, w = original_shape[:2]
        scale = min(imgsz / h, imgsz / w)
        pad_h = (imgsz - int(h * scale)) // 2
        pad_w = (imgsz - int(w * scale)) // 2
        
        for pred in predictions:
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]
            
            # Get best class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence < conf:
                continue
            
            # Convert to xyxy
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Remove padding and scale
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale
            
            # Clip to image bounds
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            detections.append(Detection(
                class_name=self.CLASS_NAMES.get(class_id, "unknown"),
                class_id=class_id,
                confidence=float(confidence),
                bbox=[x1, y1, x2, y2]
            ))
        
        # Apply NMS
        detections = self._apply_nms(detections, iou)
        
        return detections
    
    def _predict_torchscript(
        self,
        frame: np.ndarray,
        conf: float,
        iou: float
    ) -> List[Detection]:
        """Run TorchScript inference."""
        import torch
        
        # Preprocess
        tensor = self._preprocess_onnx(frame)  # Same preprocessing
        tensor = torch.from_numpy(tensor)
        
        if self.device != "cpu":
            tensor = tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self._model(tensor)
        
        # Postprocess (same as ONNX)
        return self._postprocess_onnx(
            output.cpu().numpy(),
            frame.shape,
            conf,
            iou
        )
    
    def _apply_nms(
        self,
        detections: List[Detection],
        iou_threshold: float
    ) -> List[Detection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                d for d in detections
                if self._compute_iou(best.bbox, d.bbox) < iou_threshold
            ]
        
        return keep
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def _resolve_conflicts(
        self,
        detections: List[Detection],
        overlap_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Resolve conflicting helmet vs no_helmet detections.
        
        If helmet and no_helmet boxes overlap significantly,
        keep the one with higher confidence.
        """
        if len(detections) < 2:
            return detections
        
        resolved = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            # Check for conflicts with other detections
            best = det1
            for j, det2 in enumerate(detections):
                if j <= i or j in used:
                    continue
                
                # Check if same location (overlapping)
                iou = self._compute_iou(det1.bbox, det2.bbox)
                if iou > overlap_threshold:
                    # Keep higher confidence detection
                    if det2.confidence > best.confidence:
                        best = det2
                    used.add(j)
            
            resolved.append(best)
            used.add(i)
        
        return resolved
    
    def get_model_info(self) -> Dict:
        """Get model metadata."""
        return {
            "model_path": str(self.model_path),
            "backend": self.backend.value,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "class_names": self.CLASS_NAMES,
            "num_classes": len(self.CLASS_NAMES),
            "mode": self.mode.value
        }
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "helmet_yolov8s_best",
        models_dir: Optional[str] = None,
        **kwargs
    ) -> "HelmetDetectionModel":
        """
        Load a pretrained model by name.
        
        Args:
            model_name: Model name without extension
            models_dir: Directory containing models
            **kwargs: Additional arguments for __init__
        
        Returns:
            HelmetDetectionModel instance
        """
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / "models"
        else:
            models_dir = Path(models_dir)
        
        # Try different extensions
        for ext in [".pt", ".onnx", ".torchscript"]:
            model_path = models_dir / f"{model_name}{ext}"
            if model_path.exists():
                return cls(str(model_path), **kwargs)
        
        raise FileNotFoundError(
            f"Model not found: {model_name} in {models_dir}"
        )
