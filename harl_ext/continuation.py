"""
Continuation (two-stage) training utilities.

Handles:
  - Finding model directories from stage-1 runs
  - Generating transition manifests
  - Checkpoint compatibility checking
"""
from __future__ import annotations

import json
import os
import glob
import torch
from pathlib import Path
from typing import Optional

from harl_ext.run_selection import find_smac_models_dir, find_smac_run_dir


def find_model_dir(log_base: str, env_name: str, map_name: str,
                   algo_name: str, exp_name: str) -> Optional[str]:
    """
    Find the models/ directory from a completed HARL training run.
    Searches: log_base/env_name/map_name/algo_name/exp_name/seed-*/models/
    Returns the most recent one if multiple exist.
    """
    pattern = os.path.join(
        log_base, env_name, map_name, algo_name, exp_name, "seed-*", "models"
    )
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    for c in reversed(candidates):
        if os.path.isfile(os.path.join(c, "actor_agent0.pt")):
            return c
    return None


def find_model_dir_for_seed(log_base: str, env_name: str, map_name: str,
                            algo_name: str, exp_name: str, seed: int) -> Optional[str]:
    """Find model dir matching a specific seed (prefers threads=20, complete runs)."""
    if env_name == "smac":
        return find_smac_models_dir(
            log_base, map_name, algo_name, exp_name, seed
        )
    pattern = os.path.join(
        log_base, env_name, map_name, algo_name, exp_name,
        f"seed-{seed:05d}-*", "models"
    )
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    for c in reversed(candidates):
        if os.path.isfile(os.path.join(c, "actor_agent0.pt")):
            return c
    return None


def find_run_dir_for_seed(log_base: str, env_name: str, map_name: str,
                          algo_name: str, exp_name: str, seed: int) -> Optional[str]:
    """Find the run directory (parent of models/) for a specific seed."""
    if env_name == "smac":
        return find_smac_run_dir(
            log_base, map_name, algo_name, exp_name, seed
        )
    model_dir = find_model_dir_for_seed(
        log_base, env_name, map_name, algo_name, exp_name, seed
    )
    if model_dir is None:
        return None
    return str(Path(model_dir).parent)


def check_checkpoint_compatibility(
    source_model_dir: str,
    target_algo_name: str,
    target_share_param: bool,
    num_agents: int,
    hidden_sizes: list,
) -> dict:
    """
    Check if a source checkpoint can be loaded into the target config.
    Returns a detailed report.
    """
    report = {
        "source_model_dir": source_model_dir,
        "target_algo": target_algo_name,
        "target_share_param": target_share_param,
        "target_hidden_sizes": hidden_sizes,
        "num_agents": num_agents,
        "actor_files": {},
        "critic_file": {},
        "compatible": True,
        "warnings": [],
    }

    for agent_id in range(num_agents):
        fpath = os.path.join(source_model_dir, f"actor_agent{agent_id}.pt")
        if os.path.exists(fpath):
            sd = torch.load(fpath, map_location="cpu")
            shapes = {k: list(v.shape) for k, v in sd.items()}
            report["actor_files"][f"actor_agent{agent_id}"] = {
                "exists": True, "param_shapes": shapes,
            }
        else:
            report["actor_files"][f"actor_agent{agent_id}"] = {"exists": False}
            report["compatible"] = False
            report["warnings"].append(f"Missing {fpath}")

    critic_path = os.path.join(source_model_dir, "critic_agent.pt")
    if os.path.exists(critic_path):
        sd = torch.load(critic_path, map_location="cpu")
        shapes = {k: list(v.shape) for k, v in sd.items()}
        report["critic_file"] = {"exists": True, "param_shapes": shapes}
    else:
        report["critic_file"] = {"exists": False}
        report["warnings"].append(f"Missing critic: will initialize fresh")

    return report


def write_transition_manifest(
    output_path: str,
    source_algo: str,
    target_algo: str,
    source_checkpoint_path: str,
    target_config_path: str,
    share_param_source: bool,
    share_param_target: bool,
    hidden_sizes: list,
    load_report: dict,
    notes: str = "",
):
    """Write a JSON manifest documenting the stage1->stage2 transition."""
    manifest = {
        "source_algo": source_algo,
        "target_algo": target_algo,
        "source_checkpoint_path": source_checkpoint_path,
        "target_config_path": target_config_path,
        "share_param_source": share_param_source,
        "share_param_target": share_param_target,
        "representation_hidden_sizes": hidden_sizes,
        "actor_handling": "direct load (same architecture, [128,128] for all)",
        "critic_handling": "load if compatible, else fresh init",
        "optimizer_handling": "fresh init (new optimizer for stage2)",
        "rnn_state_handling": "reset to zero (standard for new run)",
        "load_report": load_report,
        "notes": notes,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Transition manifest: {output_path}")
    return manifest
