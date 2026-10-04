"""
Centralized method and config registry for the SMAC robustness experiment.

5 methods × 3 maps × 4 seeds — designed for clean causal decomposition:

  Q1: is "just more training" enough?            -> compare (1) vs (3)
  Q2: does shared->nonshared decoupling help?    -> compare (3) vs (4)
  Q3: does HAPPO add further gains?              -> compare (4) vs (5)

  (1) MAPPO-shared scratch          — baseline
  (2) HAPPO-nonshared scratch       — upper-bound reference
  (3) MAPPO-shared → MAPPO-shared   — "more training" control
  (4) MAPPO-shared → MAPPO-nonshared— param decoupling only
  (5) MAPPO-shared → HAPPO-nonshared— M2H (decoupling + algo switch)

Hidden sizes fixed at [128,128] (standard, no controversy).
Episode length per map matches SMAC environment limit.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

HARL_ROOT = str(Path(__file__).resolve().parents[3])
TUNED_CFG_DIR = os.path.join(HARL_ROOT, "tuned_configs", "smac")
PROJECT_DIR = str(Path(__file__).resolve().parents[1])

MAPS = ["3s5z", "MMM2", "corridor"]
MAPS_EXTENDED = MAPS + ["5m_vs_6m", "8m_vs_9m", "6h_vs_8z", "25m", "3s5z_vs_3s6z"]
SEEDS = [1, 2, 3, 4]
HIDDEN_SIZES = [128, 128]

N_ROLLOUT_THREADS = 20

MAP_EPISODE_LENGTH = {
    "3s5z":         150,
    "MMM2":         180,
    "corridor":     400,
    "5m_vs_6m":      70,
    "8m_vs_9m":     120,
    "6h_vs_8z":     150,
    "25m":          150,
    "3s5z_vs_3s6z": 170,
}

MAP_AGENTS = {
    "3s5z": 8, "MMM2": 10, "corridor": 6,
    "5m_vs_6m": 5, "8m_vs_9m": 8, "6h_vs_8z": 6, "25m": 25, "3s5z_vs_3s6z": 8,
}

BUDGET = {
    "3s5z": {
        "total":    10_002_000,
        "pretrain":  6_666_000,
        "continue":  3_336_000,
    },
    "MMM2": {
        "total":    19_998_000,
        "pretrain": 13_334_400,
        "continue":  6_663_600,
    },
    "corridor": {
        "total":    20_000_000,
        "pretrain": 13_336_000,
        "continue":  6_664_000,
    },
    "5m_vs_6m": {
        "total":    19_992_000,
        "pretrain": 13_328_000,
        "continue":  6_664_000,
    },
    "8m_vs_9m": {
        "total":    19_999_200,
        "pretrain": 13_332_000,
        "continue":  6_667_200,
    },
    "6h_vs_8z": {
        "total":    19_998_000,
        "pretrain": 13_332_000,
        "continue":  6_666_000,
    },
    "25m": {
        "total":    19_998_000,
        "pretrain": 13_332_000,
        "continue":  6_666_000,
    },
    "3s5z_vs_3s6z": {
        "total":    19_998_800,
        "pretrain": 13_334_800,
        "continue":  6_664_000,
    },
}

for m in MAPS_EXTENDED:
    b = BUDGET[m]
    spu = MAP_EPISODE_LENGTH[m] * N_ROLLOUT_THREADS
    assert b["pretrain"] + b["continue"] == b["total"], f"Budget mismatch for {m}"
    assert b["total"] % spu == 0, f"Total budget for {m} not divisible by {spu}"
    assert b["pretrain"] % spu == 0, f"Pretrain budget for {m} not divisible by {spu}"
    assert b["continue"] % spu == 0, f"Continue budget for {m} not divisible by {spu}"

# ---- Main-paper methods (5) ----
METHODS = {
    # (1) MAPPO-shared scratch — baseline
    "mappo_shared_full": {
        "algo": "mappo", "share_param": True,
        "is_continuation": False,
        "description": "MAPPO(shared), full budget baseline",
    },
    # (2) HAPPO-nonshared scratch — upper-bound reference
    "happo_nonshared_full": {
        "algo": "happo", "share_param": False,
        "is_continuation": False,
        "description": "HAPPO(nonshared), full budget baseline",
    },
    # (3) MAPPO-shared continuation — "more training" control
    "mappo_to_mappo_shared": {
        "algo_stage1": "mappo", "share_param_stage1": True,
        "algo_stage2": "mappo", "share_param_stage2": True,
        "is_continuation": True,
        "description": "MAPPO-shared continuation (more-training control)",
    },
    # (4) MAPPO-shared → MAPPO-nonshared — param decoupling only
    "mappo_to_mappo_nonshared": {
        "algo_stage1": "mappo", "share_param_stage1": True,
        "algo_stage2": "mappo", "share_param_stage2": False,
        "is_continuation": True,
        "description": "MAPPO shared→nonshared (param decoupling control)",
    },
    # (5) MAPPO-shared → HAPPO-nonshared — M2H
    "mappo_to_happo_nonshared": {
        "algo_stage1": "mappo", "share_param_stage1": True,
        "algo_stage2": "happo", "share_param_stage2": False,
        "is_continuation": True,
        "description": "MAPPO→HAPPO nonshared (M2H, decoupling + algo switch)",
    },
}

CHECKPOINT_FRACTIONS = [1/3, 2/3, 1.0]


def _load_tuned_base(map_name: str) -> dict:
    """Load official HARL tuned config for given map (happo base)."""
    cfg_path = os.path.join(TUNED_CFG_DIR, map_name, "happo", "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Tuned config not found: {cfg_path}\n"
            f"Refusing to run without a verified tuned config."
        )
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return cfg


def build_config(
    map_name: str,
    algo: str,
    share_param: bool,
    seed: int,
    num_env_steps: int,
    exp_name: str,
    model_dir: str | None = None,
    log_dir: str | None = None,
    save_interval_pct: float = 10.0,
) -> dict:
    """Build a complete HARL config dict from tuned base + overrides.

    save_interval_pct: save checkpoint every N% of total episodes (default 10%).
    """
    cfg = _load_tuned_base(map_name)

    cfg["main_args"]["algo"] = algo
    cfg["main_args"]["env"] = "smac"
    cfg["main_args"]["exp_name"] = exp_name
    cfg["main_args"]["load_config"] = ""

    cfg["env_args"]["map_name"] = map_name

    cfg["algo_args"]["model"]["hidden_sizes"] = list(HIDDEN_SIZES)

    ep_len = MAP_EPISODE_LENGTH[map_name]
    cfg["algo_args"]["train"]["episode_length"] = ep_len
    cfg["algo_args"]["train"]["n_rollout_threads"] = N_ROLLOUT_THREADS

    cfg["algo_args"]["algo"]["share_param"] = share_param

    if algo == "mappo":
        cfg["algo_args"]["algo"]["fixed_order"] = True
    elif algo == "happo":
        cfg["algo_args"]["algo"]["fixed_order"] = False

    cfg["algo_args"]["seed"]["seed"] = seed
    cfg["algo_args"]["seed"]["seed_specify"] = True

    cfg["algo_args"]["train"]["num_env_steps"] = num_env_steps
    cfg["algo_args"]["train"]["model_dir"] = model_dir

    cfg["algo_args"]["eval"]["eval_episodes"] = 20
    cfg["algo_args"]["eval"]["n_eval_rollout_threads"] = 10

    total_episodes = num_env_steps // (ep_len * N_ROLLOUT_THREADS)
    eval_interval = max(1, int(total_episodes * save_interval_pct / 100.0))
    cfg["algo_args"]["train"]["eval_interval"] = eval_interval
    cfg["algo_args"]["train"]["log_interval"] = max(1, eval_interval // 5)

    if log_dir is not None:
        cfg["algo_args"]["logger"]["log_dir"] = log_dir

    return cfg


def get_full_budget_config(
    method_name: str, map_name: str, seed: int, log_dir: str
) -> dict:
    """Get config for a full-budget (non-continuation) method."""
    mdef = METHODS[method_name]
    assert not mdef["is_continuation"], f"{method_name} is continuation, not full-budget"
    return build_config(
        map_name=map_name,
        algo=mdef["algo"],
        share_param=mdef["share_param"],
        seed=seed,
        num_env_steps=BUDGET[map_name]["total"],
        exp_name=method_name,
        model_dir=None,
        log_dir=log_dir,
    )


def get_continuation_stage1_config(
    method_name: str, map_name: str, seed: int, log_dir: str
) -> dict:
    """Get stage-1 (pretrain) config for a continuation method."""
    mdef = METHODS[method_name]
    assert mdef["is_continuation"], f"{method_name} is not continuation"
    return build_config(
        map_name=map_name,
        algo=mdef["algo_stage1"],
        share_param=mdef["share_param_stage1"],
        seed=seed,
        num_env_steps=BUDGET[map_name]["pretrain"],
        exp_name=f"{method_name}_stage1",
        model_dir=None,
        log_dir=log_dir,
    )


def get_continuation_stage2_config(
    method_name: str, map_name: str, seed: int, log_dir: str,
    model_dir: str,
) -> dict:
    """Get stage-2 (continuation) config."""
    mdef = METHODS[method_name]
    assert mdef["is_continuation"], f"{method_name} is not continuation"
    return build_config(
        map_name=map_name,
        algo=mdef["algo_stage2"],
        share_param=mdef["share_param_stage2"],
        seed=seed,
        num_env_steps=BUDGET[map_name]["continue"],
        exp_name=f"{method_name}_stage2",
        model_dir=model_dir,
        log_dir=log_dir,
    )


def resolve_config_summary(cfg: dict) -> dict:
    """Return a flat summary of the resolved config for logging."""
    aa = cfg["algo_args"]
    return {
        "algo_name": cfg["main_args"]["algo"],
        "env_name": cfg["main_args"]["env"],
        "map_name": cfg["env_args"]["map_name"],
        "exp_name": cfg["main_args"]["exp_name"],
        "seed": aa["seed"]["seed"],
        "hidden_sizes": aa["model"]["hidden_sizes"],
        "share_param": aa["algo"]["share_param"],
        "fixed_order": aa["algo"]["fixed_order"],
        "num_env_steps": aa["train"]["num_env_steps"],
        "model_dir": aa["train"]["model_dir"],
        "n_rollout_threads": aa["train"]["n_rollout_threads"],
        "episode_length": aa["train"]["episode_length"],
        "use_recurrent_policy": aa["model"]["use_recurrent_policy"],
        "state_type": cfg["env_args"].get("state_type", "EP"),
        "gamma": aa["algo"]["gamma"],
        "lr": aa["model"]["lr"],
        "clip_param": aa["algo"]["clip_param"],
        "entropy_coef": aa["algo"]["entropy_coef"],
        "log_dir": aa["logger"]["log_dir"],
        "source": "tuned_configs/smac/<map>/happo/config.json + overrides",
    }
