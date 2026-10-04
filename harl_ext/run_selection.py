"""
Select the best SMAC training run among multiple seed-* directories.

Prefers runs that match the registry budget and n_rollout_threads=20,
avoiding accidental --threads 8 resume runs that change PPO batch size.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Optional

from configs.method_registry import BUDGET, METHODS, N_ROLLOUT_THREADS


def infer_expected_steps(map_name: str, exp_name: str) -> Optional[int]:
    """Return the intended num_env_steps for a method/stage on this map."""
    if map_name not in BUDGET:
        return None
    if exp_name.endswith("_stage1"):
        base = exp_name[: -len("_stage1")]
        if base in METHODS:
            return BUDGET[map_name]["pretrain"]
    if exp_name.endswith("_stage2"):
        base = exp_name[: -len("_stage2")]
        if base in METHODS:
            return BUDGET[map_name]["continue"]
    if exp_name in METHODS:
        return BUDGET[map_name]["total"]
    return None


def _load_run_config(run_dir: str) -> Optional[dict]:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path) as f:
        return json.load(f)


def score_smac_run(
    run_dir: str,
    expected_threads: int = N_ROLLOUT_THREADS,
    expected_steps: Optional[int] = None,
) -> int:
    """Higher score = better run. Wrong-thread runs are heavily penalized."""
    cfg = _load_run_config(run_dir)
    if cfg is None:
        return -10**9

    aa = cfg["algo_args"]
    threads = aa["train"]["n_rollout_threads"]
    steps = aa["train"]["num_env_steps"]

    score = 0
    if threads == expected_threads:
        score += 10_000
    else:
        score -= 5_000

    if os.path.isfile(os.path.join(run_dir, "_TRAINING_COMPLETE")):
        score += 5_000

    model_path = os.path.join(run_dir, "models", "actor_agent0.pt")
    if os.path.isfile(model_path):
        score += 1_000

    if expected_steps is not None:
        if steps == expected_steps:
            score += 2_000
        elif steps >= int(expected_steps * 0.95):
            score += 500
        else:
            score -= max(0, (expected_steps - steps) // 1000)

    summary = os.path.join(run_dir, "logs", "summary.json")
    if os.path.isfile(summary):
        try:
            with open(summary) as f:
                payload = json.load(f)
            n_pts = sum(len(v) for v in payload.values())
            score += min(n_pts, 500)
        except (json.JSONDecodeError, TypeError, AttributeError):
            score += 10

    progress = os.path.join(run_dir, "progress.txt")
    if os.path.isfile(progress) and os.path.getsize(progress) > 0:
        score += 50

    return score


def pick_best_run_dir(
    run_dirs: list[str],
    expected_threads: int = N_ROLLOUT_THREADS,
    expected_steps: Optional[int] = None,
) -> Optional[str]:
    """Pick the best run directory from candidates."""
    if not run_dirs:
        return None

    good_thread_dirs = []
    for run_dir in run_dirs:
        cfg = _load_run_config(run_dir)
        if cfg is None:
            continue
        if cfg["algo_args"]["train"]["n_rollout_threads"] == expected_threads:
            good_thread_dirs.append(run_dir)

    if not good_thread_dirs:
        return None

    scored = [
        (score_smac_run(d, expected_threads, expected_steps), d)
        for d in good_thread_dirs
    ]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, best_dir = scored[0]
    if best_score < 0:
        return None
    return best_dir


def _smac_method_root(log_base: str, map_name: str) -> str:
    """Resolve .../map_<map>/smac/<map> from log_base."""
    nested = os.path.join(log_base, f"map_{map_name}", "smac", map_name)
    if os.path.isdir(nested):
        return nested
    flat = os.path.join(log_base, "smac", map_name)
    if os.path.isdir(flat):
        return flat
    return nested


def glob_smac_run_dirs(
    log_base: str,
    map_name: str,
    algo: str,
    exp_name: str,
    seed: int,
) -> list[str]:
    base = _smac_method_root(log_base, map_name)
    pattern = os.path.join(base, algo, exp_name, f"seed-{seed:05d}-*")
    return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def find_smac_run_dir(
    log_base: str,
    map_name: str,
    algo: str,
    exp_name: str,
    seed: int,
    expected_threads: int = N_ROLLOUT_THREADS,
    expected_steps: Optional[int] = None,
) -> Optional[str]:
    run_dirs = glob_smac_run_dirs(log_base, map_name, algo, exp_name, seed)
    if expected_steps is None:
        expected_steps = infer_expected_steps(map_name, exp_name)
    return pick_best_run_dir(run_dirs, expected_threads, expected_steps)


def find_smac_models_dir(
    log_base: str,
    map_name: str,
    algo: str,
    exp_name: str,
    seed: int,
    expected_threads: int = N_ROLLOUT_THREADS,
    expected_steps: Optional[int] = None,
) -> Optional[str]:
    run_dir = find_smac_run_dir(
        log_base, map_name, algo, exp_name, seed,
        expected_threads, expected_steps,
    )
    if run_dir is None:
        return None
    models = os.path.join(run_dir, "models")
    if os.path.isfile(os.path.join(models, "actor_agent0.pt")):
        return models
    return None


def read_progress_steps(run_dir: str) -> int:
    progress_file = os.path.join(run_dir, "progress.txt")
    if not os.path.isfile(progress_file) or os.path.getsize(progress_file) == 0:
        return 0
    try:
        with open(progress_file) as f:
            lines = f.readlines()
        if not lines:
            return 0
        last_line = lines[-1].strip()
        return int(last_line.split(",")[0])
    except (ValueError, IndexError):
        return 0


def find_best_smac_checkpoint(
    log_base: str,
    map_name: str,
    algo: str,
    exp_name: str,
    seed: int,
    expected_threads: int = N_ROLLOUT_THREADS,
    expected_steps: Optional[int] = None,
) -> tuple[Optional[str], int, Optional[str]]:
    """Return (model_dir, completed_steps, run_dir) for resume."""
    if expected_steps is None:
        expected_steps = infer_expected_steps(map_name, exp_name)

    run_dirs = glob_smac_run_dirs(log_base, map_name, algo, exp_name, seed)
    eligible = []
    for run_dir in run_dirs:
        cfg = _load_run_config(run_dir)
        if cfg is None:
            continue
        threads = cfg["algo_args"]["train"]["n_rollout_threads"]
        if threads != expected_threads:
            continue
        model_dir = os.path.join(run_dir, "models")
        if not os.path.isfile(os.path.join(model_dir, "critic_agent.pt")):
            continue
        eligible.append(run_dir)

    if not eligible:
        return None, 0, None

    best_run = None
    best_steps = -1
    for run_dir in eligible:
        steps = read_progress_steps(run_dir)
        if steps > best_steps:
            best_steps = steps
            best_run = run_dir

    if best_run is None:
        return None, 0, None
    return os.path.join(best_run, "models"), best_steps, best_run


def validate_rollout_threads(threads: Optional[int]) -> int:
    """Reject overrides that change PPO batch size."""
    if threads is None:
        return N_ROLLOUT_THREADS
    if threads != N_ROLLOUT_THREADS:
        raise ValueError(
            f"n_rollout_threads must be {N_ROLLOUT_THREADS} for SMAC experiments "
            f"(got {threads}). Overriding threads changes PPO batch size and "
            f"invalidates cross-method comparisons. Remove --threads and use "
            f"train_full_budget.py / train_continuation.py instead."
        )
    return threads
