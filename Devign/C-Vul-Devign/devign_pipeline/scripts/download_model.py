"""Download model from GitHub Release."""

import os
import sys
from pathlib import Path

import requests


def download_model(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model from {url}")
    print(f"Destination: {dest}")
    
    headers = {}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
        headers["Accept"] = "application/octet-stream"
    
    resp = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    resp.raise_for_status()
    
    total_size = int(resp.headers.get("content-length", 0))
    downloaded = 0
    
    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\rProgress: {pct:.1f}%", end="", flush=True)
    
    print(f"\nDownloaded {downloaded:,} bytes")


def main() -> None:
    model_url = os.environ.get("MODEL_URL")
    if not model_url:
        print("ERROR: MODEL_URL environment variable is not set")
        print("Set MODEL_URL to the GitHub Release asset URL for the model file")
        sys.exit(1)
    
    dest = Path(os.environ.get("MODEL_DEST", "models/best_v2_seed42.pt"))
    
    try:
        download_model(model_url, dest)
        print(f"Model downloaded successfully to {dest}")
    except requests.RequestException as e:
        print(f"ERROR: Failed to download model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
