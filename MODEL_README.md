# Helmet Detection Model - Quick Reference

## 📁 Project Files

```
realtime-helmet-detection/
├── kaggle_training.py       # Kaggle training script (copy to notebook)
├── backend/
│   ├── models.py            # Clean model abstraction (multi-backend)
│   ├── helmet_inference.py  # Helmet-specific inference engine
│   ├── inference.py         # Original inference (still works)
│   └── main.py              # FastAPI backend
├── configs/
│   ├── dataset.yaml         # YOLO dataset configuration
│   └── training.yaml        # Training hyperparameters
└── models/
    ├── helmet_yolov8s_best.pt        # PyTorch model (from Kaggle)
    ├── best.onnx                     # ONNX model
    ├── baseline_stats.json           # Drift detection baseline
    └── metrics.json                  # Training metrics
```

## 🚀 Training on Kaggle

1. Create new Kaggle notebook with **GPU T4 x2**
2. Add datasets:
   - `andrewmvd/hard-hat-detection`
   - `coco-2017-dataset` (for COCO negatives)
3. Copy cells from `kaggle_training.py` sequentially
4. Run all cells (~30-40 minutes)
5. Download exports from Output section

## 🎯 Training Results

| Metric | Achieved | 
|--------|----------|
| mAP@50 | **86.7%** |
| mAP@50-95 | 50.0% |
| Precision | 87.5% |
| Recall | 77.5% |
| Helmet Precision | 91.4% |
| No-Helmet Recall | 78.1% |

### Per-Class Performance

| Class | Images | Instances | Precision | Recall | mAP@50 |
|-------|--------|-----------|-----------|--------|--------|
| helmet | 911 | 3451 | 91.4% | 76.9% | 87.9% |
| no_helmet | 168 | 1132 | 83.6% | 78.1% | 85.6% |

## 🔄 Swap Models

After training, update `backend/main.py`:

```python
MODEL_PATH = Path(__file__).parent.parent / "models" / "helmet_yolov8s_best.pt"
```

Or use the model abstraction:

```python
from backend.models import HelmetDetectionModel

model = HelmetDetectionModel.from_pretrained("helmet_yolov8s_best")
result = model.predict(frame, confidence=0.5)
print(result.to_dict())
```

## 🏃 Inference Performance

| Stage | Time |
|-------|------|
| Preprocess | 0.2ms |
| Inference | 6.6ms |
| Postprocess | 1.5ms |
| **Total** | **~8.3ms** |
| Throughput | ~120 FPS |

## 📝 Resume Highlights

> **Real-Time Helmet Compliance Detection with MLOps Monitoring**
> - Trained YOLOv8s on merged helmet datasets achieving **86.7% mAP@50**
> - Built **two-class detector** with **91.4% helmet precision** and **78.1% no-helmet recall**
> - Implemented **drift detection** for production monitoring
> - Built REST API with **FastAPI** and React dashboard
> - Integrated **Prometheus/Grafana** with custom alerts
> - Containerized with **Docker Compose**
