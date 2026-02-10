
import cv2
import numpy as np
import onnxruntime as ort
import time
from typing import List, Tuple, Dict
from pathlib import Path
from .metrics import track_inference_time, update_detection_count, set_model_loaded

class DetectionResult:
    """Container for detection results."""
    def __init__(self, boxes, scores, class_ids, class_names, inference_time_ms, frame_shape):
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        self.class_names = class_names
        self.inference_time_ms = inference_time_ms
        self.frame_shape = frame_shape
        self.count = len(boxes)
    
    def to_dict(self):
        return {
            "detections": [
                {
                    "box": box,
                    "score": float(score),
                    "class_id": int(class_id),
                    "class_name": class_name
                } for box, score, class_id, class_name in zip(self.boxes, self.scores, self.class_ids, self.class_names)
            ],
            "count": self.count,
            "inference_time_ms": self.inference_time_ms,
            "frame_shape": self.frame_shape
        }

class OnnxInferenceEngine:
    def __init__(self, model_path: str, confidence_threshold: float = 0.3, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_thres = confidence_threshold
        self.iou_thres = iou_threshold
        self.class_names = {0: 'helmet', 1: 'no_helmet'}
        
        # Load ONNX model
        try:
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.inputs = self.session.get_inputs()
            self.input_name = self.inputs[0].name
            self.input_shape = self.inputs[0].shape  # [1, 3, 640, 640]
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
            set_model_loaded(True)
            print(f"✅ ONNX Model loaded: {model_path}")
        except Exception as e:
            set_model_loaded(False)
            print(f"❌ Failed to load ONNX model: {e}")
            raise e

    def preprocess(self, image):
        self.img_height, self.img_width = image.shape[:2]
        
        # Scale input
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        
        # Normalize
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        input_tensor = img[np.newaxis, :, :, :].astype(np.float32)
        
        return input_tensor, self.img_width, self.img_height

    def postprocess(self, outputs, orig_w, orig_h):
        # Outputs: [1, 6, 8400] (4 box + 2 classes)
        predictions = np.squeeze(outputs[0]).T
        
        # Filter by confidence
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thres, :]
        scores = scores[scores > self.conf_thres]
        
        if len(scores) == 0:
            return [], [], [], []
            
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        # Get boxes (x, y, w, h) -> (x1, y1, x2, y2)
        boxes = predictions[:, :4]
        input_w, input_h = self.input_width, self.input_height
        
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        
        # Rescale to original image
        scale_w = orig_w / input_w
        scale_h = orig_h / input_h
        
        x1 = (x - w / 2) * scale_w
        y1 = (y - h / 2) * scale_h
        x2 = (x + w / 2) * scale_w
        y2 = (y + h / 2) * scale_h
        
        final_boxes = np.stack((x1, y1, x2, y2), axis=1)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            final_boxes.tolist(), scores.tolist(), self.conf_thres, self.iou_thres
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return final_boxes[indices].tolist(), scores[indices].tolist(), class_ids[indices].tolist(), [self.class_names.get(c, 'unknown') for c in class_ids[indices]]
        
        return [], [], [], []

    @track_inference_time
    def predict(self, frame: np.ndarray) -> DetectionResult:
        start_time = time.time()
        
        # Preprocess
        input_tensor, orig_w, orig_h = self.preprocess(frame)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess
        boxes, scores, class_ids, class_names = self.postprocess(outputs, orig_w, orig_h)
        
        inference_time_ms = (time.time() - start_time) * 1000
        update_detection_count(len(boxes))
        
        return DetectionResult(boxes, scores, class_ids, class_names, inference_time_ms, (orig_w, orig_h))

    def get_model_info(self):
        return {
            "model_path": str(self.model_path),
            "type": "ONNX",
            "confidence": self.conf_thres,
            "classes": self.class_names
        }
