"""
Drift Detection Module
Compares current frame statistics to training baseline to detect distribution shifts.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from scipy import stats


class DriftDetector:
    """Detects input drift by comparing image statistics to training baseline."""
    
    def __init__(self, baseline_path: str, threshold: float = 2.0):
        """
        Initialize drift detector with baseline statistics.
        
        Args:
            baseline_path: Path to baseline_stats.json from training
            threshold: Number of standard deviations for drift alert (default: 2.0)
        """
        self.threshold = threshold
        self.baseline = self._load_baseline(baseline_path)
        
        # Rolling window for smoothing
        self.history_size = 30
        self.brightness_history = []
        self.contrast_history = []
        
    def _load_baseline(self, path: str) -> Dict:
        """Load baseline statistics from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def compute_frame_stats(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Compute brightness and contrast statistics for a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Dictionary with brightness and contrast values
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame
            
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        return {
            "brightness": brightness,
            "contrast": contrast
        }
    
    def compute_drift_score(self, frame: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute drift score for a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Tuple of (drift_score, details_dict)
        """
        frame_stats = self.compute_frame_stats(frame)
        
        # Update rolling history
        self.brightness_history.append(frame_stats["brightness"])
        self.contrast_history.append(frame_stats["contrast"])
        
        if len(self.brightness_history) > self.history_size:
            self.brightness_history.pop(0)
            self.contrast_history.pop(0)
        
        # Compute Z-scores against baseline
        brightness_z = abs(
            (frame_stats["brightness"] - self.baseline["brightness_mean"]) 
            / self.baseline["brightness_std"]
        )
        contrast_z = abs(
            (frame_stats["contrast"] - self.baseline["contrast_mean"]) 
            / self.baseline["contrast_std"]
        )
        
        # Combined drift score (max of the two)
        drift_score = max(brightness_z, contrast_z)
        
        details = {
            "brightness": frame_stats["brightness"],
            "contrast": frame_stats["contrast"],
            "brightness_z": brightness_z,
            "contrast_z": contrast_z,
            "drift_score": drift_score,
            "is_drifted": drift_score > self.threshold,
            "baseline_brightness": self.baseline["brightness_mean"],
            "baseline_contrast": self.baseline["contrast_mean"]
        }
        
        return drift_score, details
    
    def get_drift_status(self, drift_score: float) -> str:
        """Get human-readable drift status."""
        if drift_score < 1.0:
            return "normal"
        elif drift_score < self.threshold:
            return "warning"
        else:
            return "alert"
    
    def reset_history(self):
        """Reset rolling history."""
        self.brightness_history = []
        self.contrast_history = []
