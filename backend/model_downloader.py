"""
Model downloader utility.
Downloads model files from Hugging Face Hub or direct URL if not present locally.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, destination: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))


def ensure_models_exist(models_dir: Path = None):
    """
    Ensure model files exist, downloading if necessary.
    
    For deployment, set these environment variables:
    - MODEL_URL: Direct URL to download the .pt model
    - BASELINE_URL: Direct URL to download baseline_stats.json
    
    Or use Hugging Face Hub:
    - HF_MODEL_REPO: Repository name (e.g., "username/helmet-yolov8s")
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"
    
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "helmet_yolov8s_best.pt"
    baseline_path = models_dir / "baseline_stats.json"
    
    # Check if models already exist
    if model_path.exists() and baseline_path.exists():
        print("✅ Model files found locally")
        return True
    
    # Try to download from environment variables
    model_url = os.environ.get("MODEL_URL")
    baseline_url = os.environ.get("BASELINE_URL")
    
    if model_url and not model_path.exists():
        print(f"📥 Downloading model from {model_url}...")
        try:
            download_file(model_url, model_path)
            print("✅ Model downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return False
    
    if baseline_url and not baseline_path.exists():
        print(f"📥 Downloading baseline stats from {baseline_url}...")
        try:
            download_file(baseline_url, baseline_path)
            print("✅ Baseline stats downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download baseline: {e}")
            return False
    
    # Try Hugging Face Hub
    hf_repo = os.environ.get("HF_MODEL_REPO")
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            
            if not model_path.exists():
                print(f"📥 Downloading model from Hugging Face: {hf_repo}")
                hf_hub_download(
                    repo_id=hf_repo,
                    filename="helmet_yolov8s_best.pt",
                    local_dir=str(models_dir)
                )
            
            if not (models_dir / "best.onnx").exists():
                try:
                    print(f"📥 Downloading ONNX model from Hugging Face: {hf_repo}")
                    hf_hub_download(
                        repo_id=hf_repo,
                        filename="best.onnx",
                        local_dir=str(models_dir)
                    )
                except Exception as e:
                    print(f"⚠️ ONNX model not found in repo (skipping): {e}")

            if not baseline_path.exists():
                print(f"📥 Downloading baseline from Hugging Face: {hf_repo}")
                hf_hub_download(
                    repo_id=hf_repo,
                    filename="baseline_stats.json",
                    local_dir=str(models_dir)
                )
            
            print("✅ Files downloaded from Hugging Face")
            return True
        except Exception as e:
            print(f"⚠️ Hugging Face download failed: {e}")
    
    # Check final status
    if model_path.exists() and baseline_path.exists():
        return True
    
    print("⚠️ Model files not found. Please:")
    print("   1. Place files in models/ directory, OR")
    print("   2. Set MODEL_URL and BASELINE_URL environment variables, OR")
    print("   3. Set HF_MODEL_REPO to your Hugging Face repository")
    
    return False


if __name__ == "__main__":
    ensure_models_exist()
