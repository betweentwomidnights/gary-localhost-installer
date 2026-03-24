#!/usr/bin/env python3
"""
Download Foundation-1 model files from HuggingFace.
Run inside the container with a mounted volume so files persist on host.
"""

import os
from huggingface_hub import hf_hub_download

REPO_ID = "RoyalCities/Foundation-1"
MODEL_DIR = os.environ.get("FOUNDATION_MODEL_DIR", "/models/foundation-1")

FILES = [
    "Foundation_1.safetensors",
    "model_config.json",
]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename in FILES:
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest):
            print(f"Already exists: {dest}")
            continue

        print(f"Downloading {filename} from {REPO_ID}...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODEL_DIR,
        )
        print(f"  -> {downloaded}")

    print("All model files ready.")


if __name__ == "__main__":
    main()
