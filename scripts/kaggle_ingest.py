import os
import argparse
import shutil
import glob
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def download_datasets(datasets, raw_dir):
    """Download and unzip datasets from Kaggle."""
    api = KaggleApi()
    api.authenticate()
    
    os.makedirs(raw_dir, exist_ok=True)
    
    for dataset in datasets:
        print(f"Downloading {dataset}...")
        # dataset format is 'username/dataset-name'
        api.dataset_download_files(dataset, path=raw_dir, unzip=True)

def process_kaggle_data(raw_dir, processed_dir):
    """
    Search for IaC files in the downloaded Kaggle data and move them 
    to the format expected by repackage_iac_data.py.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # Types we care about
    iac_types = ["terraform", "kubernetes", "ansible", "crossplane", "docker"]
    for t in iac_types:
        os.makedirs(processed_path / t, exist_ok=True)
    
    # Simple extension mapping
    ext_mapping = {
        ".tf": "terraform",
        ".tfvars": "terraform",
        ".yaml": "yaml_files", # need further check
        ".yml": "yaml_files",
        "Dockerfile": "docker",
    }
    
    print(f"Processing data from {raw_dir} to {processed_dir}...")
    
    file_count = 0
    for file_path in raw_path.rglob("*"):
        if not file_path.is_file():
            continue
            
        ext = file_path.suffix
        name = file_path.name
        
        target_type = None
        if name == "Dockerfile" or name.startswith("Dockerfile."):
            target_type = "docker"
        elif ext in [".tf", ".tfvars"]:
            target_type = "terraform"
        elif ext in [".yaml", ".yml"]:
            # Basic keyword check for K8s/Ansible/Crossplane
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(4000)
                    if "apiVersion:" in content and "kind:" in content:
                        target_type = "kubernetes"
                    elif "tasks:" in content or "hosts:" in content:
                        target_type = "ansible"
                    elif "crossplane" in content.lower():
                        target_type = "crossplane"
            except:
                continue
        
        if target_type:
            # Flatten name to avoid collisions
            safe_name = str(file_path.relative_to(raw_path)).replace(os.sep, "_")
            shutil.copy2(file_path, processed_path / target_type / safe_name)
            file_count += 1
            
    print(f"Scanned {file_count} IaC files from Kaggle datasets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull training data from Kaggle")
    parser.add_argument("--datasets", nargs="+", help="Kaggle dataset identifiers (user/repo)")
    parser.add_argument("--raw-dir", default="data/kaggle_raw")
    parser.add_argument("--processed-dir", default="data/iac_raw_cloned")
    args = parser.parse_args()
    
    if args.datasets:
        download_datasets(args.datasets, args.raw_dir)
        process_kaggle_data(args.raw_dir, args.processed_dir)
    else:
        print("No datasets provided.")
