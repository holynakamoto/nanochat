"""
Standalone IaC-GPT Training Script for Kaggle
Optimized for processing Crossplane manifests and other IaC definitions.
"""

import os
import sys
import json
import argparse
import subprocess

# 1. Environment Setup (Kaggle specific)
def setup_kaggle():
    print("Setting up Kaggle environment...")
    # Install dependencies if they are missing
    try:
        import torch
        import transformers
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "datasets", "wandb", "pyyaml"])

# 2. Project Bootstrap
def bootstrap_repo():
    if not os.path.exists("gpt.py"):
        print("Cloning nanochat repository for core modules...")
        subprocess.check_call(["git", "clone", "https://github.com/holynakamoto/iacgpt.git", "."])
        sys.path.append(os.getcwd())

# 3. Training Logic
def train(args):
    print(f"Starting IaC training on accelerator: {args.accelerator}")
    print(f"Processing manifests from: {args.manifests}")
    
    manifest_path = args.manifests
    jsonl_path = "manifests/crossplane_training.jsonl"
    os.makedirs("manifests", exist_ok=True)
    
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    with open(jsonl_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Converted manifest to JSONL: {jsonl_path}")
    os.environ["WANDB_PROJECT"] = "iacgpt-kaggle"
    
    sys.argv = [
        "scripts.chat_sft",
        "--run", f"kaggle-train-{args.slug.split('/')[-1]}",
        "--num-iterations", "100",
        "--device-batch-size", "4"
    ]
    
    try:
        from scripts.chat_sft import main as sft_main
        sft_main()
    except Exception as e:
        print(f"Training failed: {e}")
        subprocess.check_call([sys.executable, "-m", "scripts.chat_sft", "--run", "kaggle-fallback"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", default="manifests/crossplane_definitions.json")
    parser.add_argument("--accelerator", default="gpu_p100")
    parser.add_argument("--slug", default="nicholasmoore/iacgpt-bootstrap-train")
    args = parser.parse_args()
    
    setup_kaggle()
    bootstrap_repo()
    train(args)
