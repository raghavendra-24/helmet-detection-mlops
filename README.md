# Real-Time Helmet Detection with MLOps Monitoring

A production-ready helmet compliance detection system with drift monitoring, deployed as a microservices architecture.

## 🌐 Live Demo

👉 **[Live Demo Coming Soon]** | [API Docs Coming Soon] | [GitHub](https://github.com/raghavendra-24/helmet-detection-mlops)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-orange)
![React](https://img.shields.io/badge/React-Vite-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

## 🎯 Features

- **Real-Time Helmet Detection**: YOLOv8s trained on Hard Hat Detection dataset (~120 FPS on GPU)
- **Two Classes**: `helmet` (safety compliant) and `no_helmet` (violation)
- **React Frontend**: WebRTC webcam streaming with live bounding boxes
- **Drift Detection**: Automatic monitoring for input distribution shifts
- **MLOps Monitoring**: Prometheus metrics + Grafana dashboard
- **REST API + WebSocket**: FastAPI backend for production integration
- **Cloud Deployed**: Frontend on Vercel, Backend on Render

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | **86.7%** |
| mAP@50-95 | 50.0% |
| Precision | 87.5% |
| Recall | 77.5% |
| Helmet Precision | 91.4% |
| No-Helmet Recall | 78.1% |
| Inference | 6.6ms (GPU) |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam (for live demo)

### Local Setup

```bash
cd realtime-helmet-detection
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Model files should be in models/ folder:
# - helmet_yolov8s_best.pt
# - baseline_stats.json

# Run Backend
uvicorn backend.main:app --reload

# Run Frontend (in new terminal)
cd frontend-react
npm install
npm run dev
```

### Docker

```bash
docker-compose up --build
# Frontend: http://localhost:3000
# API: http://localhost:8000
# Grafana: http://localhost:3001
```

## ☁️ Cloud Deployment

### Option 1: Streamlit Cloud (Recommended)

1. **Upload model to Hugging Face Hub:**
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload your-username/helmet-yolov8s ./models/
   ```

2. **Push code to GitHub**

3. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repo
   - Set main file: `frontend/app.py`
   - Add secret: `HF_MODEL_REPO = "your-username/helmet-yolov8s"`

### Option 2: Railway/Render

1. Push to GitHub
2. Connect to Railway/Render
3. Set environment variables:
   - `MODEL_URL`: Direct URL to model file
   - `BASELINE_URL`: Direct URL to baseline_stats.json

## 📁 Project Structure

```
realtime-helmet-detection/
├── models/                    # Model files
│   ├── helmet_yolov8s_best.pt
│   ├── best.onnx
│   ├── baseline_stats.json
│   └── metrics.json
├── backend/                   # FastAPI backend
│   ├── main.py               # API endpoints
│   ├── helmet_inference.py   # Helmet detection engine
│   ├── drift_detector.py     # Drift detection
│   └── metrics.py            # Prometheus metrics
├── frontend-react/           # React + Vite frontend
├── frontend/app.py           # Streamlit UI (alternative)
├── kaggle_training.py        # Training script
├── monitoring/               # Prometheus + Grafana
├── docker-compose.yml
└── requirements.txt
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Run helmet detection on image |
| `/model-info` | GET | Get model metadata |
| `/metrics` | GET | Prometheus metrics |
| `/ws/detect` | WebSocket | Real-time detection stream |

## 🧠 Training

Training was performed on Kaggle with:
- **Dataset**: Hard Hat Detection + COCO negatives
- **Model**: YOLOv8s (11M parameters)
- **Epochs**: 40
- **Augmentations**: Mosaic, MixUp, Copy-Paste, HSV jitter

See `kaggle_training.py` for full training pipeline.

## 📝 Resume Highlights

> **Real-Time Helmet Compliance Detection with MLOps Monitoring**
> - Trained YOLOv8s on Hard Hat Detection achieving **86.7% mAP@50** with **120 FPS**
> - Built **two-class detector** for `helmet` and `no_helmet` with **91.4% helmet precision**
> - Implemented **drift detection** using statistical analysis
> - Built REST API with **FastAPI** and React dashboard with **Vite**
> - Integrated **Prometheus/Grafana** monitoring with alerts
> - Containerized with **Docker Compose**

## 📄 License

MIT License
