"""
Backend package initialization.
"""

from .inference import InferenceEngine, DetectionResult
from .drift_detector import DriftDetector
from .metrics import (
    get_metrics,
    get_content_type,
    update_fps,
    update_drift_score
)

__all__ = [
    "InferenceEngine",
    "DetectionResult", 
    "DriftDetector",
    "get_metrics",
    "get_content_type",
    "update_fps",
    "update_drift_score"
]
