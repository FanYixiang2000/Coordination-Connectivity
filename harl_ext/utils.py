"""Shared utilities for the experiment pipeline."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def get_git_commit() -> str:
    """Get current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_env_info() -> dict:
    """Collect environment info for reproducibility."""
    import torch
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def save_command_txt(output_dir: str, argv: list):
    """Save the command that launched this script."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "command.txt"), "w") as f:
        f.write(" ".join(argv) + "\n")


def save_json(data: dict, path: str):
    """Save dict to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    """Load dict from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def ensure_project_path():
    """Add project root and HARL root to sys.path."""
    project_dir = str(Path(__file__).resolve().parents[1])
    harl_root = str(Path(__file__).resolve().parents[3])
    for p in [project_dir, harl_root]:
        if p not in sys.path:
            sys.path.insert(0, p)
