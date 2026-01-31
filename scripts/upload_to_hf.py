
import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

def upload_to_huggingface(repo_id, model_path, token=None):
    """
    Uploads the model folder to Hugging Face Hub.
    """
    if token:
        login(token)
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository {repo_id} ready.")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    print(f"🚀 Uploading files from {model_path} to {repo_id}...")
    
    # Upload essential model files only
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["*.pt", "*.onnx", "baseline_stats.json"]
    )
    
    print(f"✅ Upload complete! Check your model at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face")
    parser.add_argument("--repo", required=True, help="Hugging Face Repository ID (e.g., username/model-name)")
    parser.add_argument("--path", default="../models", help="Path to models directory")
    parser.add_argument("--token", help="Hugging Face API Token (optional if already logged in)")
    
    args = parser.parse_args()
    
    model_dir = Path(args.path)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).parent / args.path
        
    upload_to_huggingface(args.repo, model_dir.resolve(), args.token)
