#!/usr/bin/env python3
"""
Kaggle Kernel Runner - Automate notebook execution on Kaggle GPUs
Reads configuration from claude.yml and manages kernel lifecycle.
"""

import os
import sys
import json
import time
import yaml
import argparse
from pathlib import Path
from typing import Optional

try:
    from kaggle import api
except ImportError:
    print("ERROR: Kaggle API not installed. Run: pip install kaggle")
    sys.exit(1)


class KaggleRunner:
    """Manages Kaggle kernel creation and execution."""

    def __init__(self, config_path: str = "claude.yml"):
        self.config = self._load_config(config_path)
        self.kaggle_config = self.config.get("kaggle_config", {})
        self.slug = self.kaggle_config.get("slug")
        self.username = os.environ.get("KAGGLE_USERNAME") or api.read_config_file()["username"]

        if not self.slug:
            raise ValueError("kaggle_config.slug not found in claude.yml")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def get_accelerator_type(self) -> str:
        """Map claude.yml accelerator to Kaggle kernel type."""
        training_config = self.config.get("training", {})
        accelerator = training_config.get("accelerator", "gpu_p100")

        # Map to Kaggle's accelerator types
        mapping = {
            "gpu_p100": "GPU",
            "gpu_t4": "GPU",
            "tpu": "TPU",
            "cpu": "CPU"
        }
        return mapping.get(accelerator, "GPU")

    def create_kernel_metadata(self, notebook_path: str) -> dict:
        """Generate kernel metadata for Kaggle API."""
        kernel_type = self.get_accelerator_type()

        metadata = {
            "id": self.slug,
            "title": self.config.get("project", "IaC-GPT Training"),
            "code_file": os.path.basename(notebook_path),
            "language": "python",
            "kernel_type": "notebook",
            "is_private": self.kaggle_config.get("is_private", True),
            "enable_gpu": kernel_type == "GPU",
            "enable_tpu": kernel_type == "TPU",
            "enable_internet": self.kaggle_config.get("internet", True),
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": []
        }

        return metadata

    def push_and_run(self, notebook_path: str, working_dir: Optional[str] = None,
                     timeout_hours: int = 4) -> bool:
        """Push notebook to Kaggle and trigger execution."""
        if working_dir is None:
            working_dir = Path(notebook_path).parent

        # Create kernel-metadata.json
        metadata = self.create_kernel_metadata(notebook_path)
        metadata_path = Path(working_dir) / "kernel-metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        timeout_seconds = timeout_hours * 3600

        print(f"📤 Pushing kernel to Kaggle: {self.slug}")
        print(f"   GPU: {metadata['enable_gpu']}")
        print(f"   Internet: {metadata['enable_internet']}")
        print(f"   Private: {metadata['is_private']}")
        print(f"   Max Runtime: {timeout_hours}h ({timeout_seconds}s)")
        print(f"\n📤 Pushing notebook to Kaggle...")
        print(f"⚠️  Note: Kaggle API cannot trigger execution automatically")
        print(f"   You'll need to click 'Run' on Kaggle after push completes")

        try:
            # Push the kernel with timeout (sets max runtime limit)
            api.kernels_push(str(working_dir), timeout=timeout_seconds)
            print(f"✅ Kernel pushed successfully!")
            return True
        except Exception as e:
            print(f"❌ Error pushing kernel: {e}")
            return False

    def get_kernel_status(self) -> Optional[str]:
        """Get current status of the kernel."""
        try:
            response = api.kernels_status(self.slug)
            status = response.get("status", "unknown")
            return status
        except Exception as e:
            print(f"⚠️  Error getting kernel status: {e}")
            return None

    def wait_for_status(self, target_status: str = "running", timeout: int = 300, poll_interval: int = 5) -> bool:
        """Poll kernel status until it reaches target status or timeout."""
        print(f"\n⏳ Waiting for kernel status: '{target_status}' (timeout: {timeout}s)")

        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout:
            current_status = self.get_kernel_status()

            if current_status != last_status:
                print(f"   Status: {current_status}")
                last_status = current_status

            if current_status and current_status.lower() == target_status.lower():
                print(f"✅ Kernel is now {target_status}!")
                return True

            if current_status and current_status.lower() in ["error", "failed", "cancelled"]:
                print(f"❌ Kernel execution failed with status: {current_status}")
                return False

            time.sleep(poll_interval)

        print(f"⏱️  Timeout waiting for status '{target_status}'")
        return False

    def get_kernel_url(self) -> str:
        """Get the Kaggle URL for this kernel."""
        return f"https://www.kaggle.com/code/{self.slug}"


def main():
    parser = argparse.ArgumentParser(description="Run IaC-GPT training on Kaggle")
    parser.add_argument("--notebook", default="kaggle_iacgpt_training.ipynb",
                       help="Path to notebook to run")
    parser.add_argument("--config", default="claude.yml",
                       help="Path to claude.yml config file")
    parser.add_argument("--timeout-hours", type=int, default=4,
                       help="Max execution time in hours (default: 4)")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor execution output (experimental)")

    args = parser.parse_args()

    # Check if notebook exists
    if not os.path.exists(args.notebook):
        print(f"❌ Notebook not found: {args.notebook}")
        sys.exit(1)

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    # Initialize runner
    try:
        runner = KaggleRunner(args.config)
    except Exception as e:
        print(f"❌ Error initializing runner: {e}")
        sys.exit(1)

    # Push and run kernel
    success = runner.push_and_run(args.notebook, timeout_hours=args.timeout_hours)

    if not success:
        sys.exit(1)

    print(f"\n🔗 Kernel URL: {runner.get_kernel_url()}")
    print(f"\n⚠️  IMPORTANT: Click 'Run' on Kaggle to start training!")
    print(f"\n📋 Next steps:")
    print(f"   1. Visit: {runner.get_kernel_url()}")
    print(f"   2. Click the 'Run' button (top right)")
    print(f"   3. Training will start on GPU P100 (max {args.timeout_hours}h)")
    print(f"\n💡 To monitor output (after starting):")
    print(f"   kaggle kernels output {runner.slug}")

    if args.monitor:
        print(f"\n📊 Fetching initial output...")
        try:
            # Wait a bit for execution to start
            time.sleep(10)
            output = api.kernels_output(runner.slug)
            print(output)
        except Exception as e:
            print(f"⚠️  Could not fetch output yet: {e}")
            print(f"   Check the URL above for live updates")


if __name__ == "__main__":
    main()
