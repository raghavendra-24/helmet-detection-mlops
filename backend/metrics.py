"""
Prometheus Metrics Module
Exposes metrics for monitoring inference performance and drift.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
from functools import wraps


# Counters
INFERENCE_COUNT = Counter(
    'detection_inference_total',
    'Total number of inference requests',
    ['status']  # success, error
)

DETECTION_COUNT = Counter(
    'detection_objects_total',
    'Total number of objects detected'
)

DRIFT_ALERT_COUNT = Counter(
    'detection_drift_alerts_total',
    'Total number of drift alerts triggered'
)

# Histograms
INFERENCE_LATENCY = Histogram(
    'detection_inference_latency_seconds',
    'Inference latency in seconds',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)

# Gauges
CURRENT_FPS = Gauge(
    'detection_current_fps',
    'Current frames per second'
)

CURRENT_DRIFT_SCORE = Gauge(
    'detection_drift_score',
    'Current drift score'
)

MODEL_LOADED = Gauge(
    'detection_model_loaded',
    'Whether the model is loaded (1) or not (0)'
)


def track_inference_time(func):
    """Decorator to track inference time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            INFERENCE_COUNT.labels(status='success').inc()
            return result
        except Exception as e:
            INFERENCE_COUNT.labels(status='error').inc()
            raise e
        finally:
            latency = time.time() - start_time
            INFERENCE_LATENCY.observe(latency)
    return wrapper


def update_detection_count(count: int):
    """Update detection count metric."""
    DETECTION_COUNT.inc(count)


def update_fps(fps: float):
    """Update current FPS gauge."""
    CURRENT_FPS.set(fps)


def update_drift_score(score: float, is_alert: bool = False):
    """Update drift score gauge and increment alert counter if needed."""
    CURRENT_DRIFT_SCORE.set(score)
    if is_alert:
        DRIFT_ALERT_COUNT.inc()


def set_model_loaded(loaded: bool):
    """Set model loaded status."""
    MODEL_LOADED.set(1 if loaded else 0)


def get_metrics():
    """Get all metrics in Prometheus format."""
    return generate_latest()


def get_content_type():
    """Get Prometheus content type."""
    return CONTENT_TYPE_LATEST
