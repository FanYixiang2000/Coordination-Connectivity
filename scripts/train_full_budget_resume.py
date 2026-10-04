"""
Train a full-budget method with automatic checkpoint resume.

If a previous run exists for the same method/map/seed with a checkpoint,
resumes from that checkpoint for the remaining steps instead of starting over.

Usage:
    python scripts/train_full_budget_resume.py --method mappo_shared_full --map 3s5z_vs_3s6z --seed 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = str(SCRIPT_DIR.parent)
HARL_ROOT = str(SCRIPT_DIR.parents[3])
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, HARL_ROOT)

import harl_ext.smac_fix  # noqa: F401

from configs.method_registry import (
    METHODS, MAPS_EXTENDED, SEEDS, HIDDEN_SIZES,
    get_full_budget_config, resolve_config_summary, BUDGET,
)
from harl_ext.utils import get_git_commit, get_env_info, save_command_txt, save_json
from harl_ext.run_selection import find_best_smac_checkpoint, validate_rollout_threads


def find_best_checkpoint(log_dir: str, map_name: str, algo: str, exp_name: str, seed: int):
    """Find the most advanced checkpoint for this job (threads=20 only).

    Returns (model_dir, completed_steps) or (None, 0) if no checkpoint found.
    """
    model_dir, completed_steps, _ = find_best_smac_checkpoint(
        log_dir, map_name, algo, exp_name, seed
    )
    return model_dir, completed_steps


def main():
    parser = argparse.ArgumentParser(description="Full-budget training with auto-resume")
    parser.add_argument("--method", type=str, required=True,
                        choices=[m for m, d in METHODS.items() if not d["is_continuation"]])
    parser.add_argument("--map", type=str, required=True, choices=MAPS_EXTENDED)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--threads", type=int, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        validate_rollout_threads(args.threads)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if args.log_dir is None:
        args.log_dir = os.path.join(PROJECT_DIR, "exp", "smac_robustness",
                                    f"map_{args.map}")

    mdef = METHODS[args.method]
    total_budget = BUDGET[args.map]["total"]

    checkpoint_dir, completed_steps = find_best_checkpoint(
        args.log_dir, args.map, mdef["algo"], args.method, args.seed
    )

    remaining_steps = total_budget - completed_steps

    if completed_steps > 0 and checkpoint_dir:
        pct = completed_steps * 100 // total_budget
        print(f"[RESUME] Found checkpoint at {pct}% ({completed_steps:,}/{total_budget:,} steps)")
        print(f"  Checkpoint: {checkpoint_dir}")
        print(f"  Remaining:  {remaining_steps:,} steps")

        if remaining_steps <= 0 or pct >= 95:
            print(f"[SKIP] Already completed ({pct}% >= 95%). Nothing to do.")
            return
    else:
        print(f"[FRESH] No checkpoint found, starting from scratch.")
        checkpoint_dir = None
        remaining_steps = total_budget

    cfg = get_full_budget_config(args.method, args.map, args.seed, args.log_dir)

    cfg["algo_args"]["train"]["num_env_steps"] = remaining_steps
    cfg["algo_args"]["train"]["model_dir"] = checkpoint_dir

    summary = resolve_config_summary(cfg)

    print("=" * 70)
    print(f"  FULL-BUDGET TRAINING {'(RESUMED)' if checkpoint_dir else '(FRESH)'}")
    print(f"  Method:      {args.method}")
    print(f"  Map:         {args.map}")
    print(f"  Seed:        {args.seed}")
    print(f"  Algo:        {summary['algo_name']}")
    print(f"  share_param: {summary['share_param']}")
    print(f"  Steps:       {remaining_steps:,} (of {total_budget:,} total)")
    print(f"  Resumed from:{checkpoint_dir or 'N/A'}")
    print(f"  Log dir:     {args.log_dir}")
    print("=" * 70)

    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[cfg["main_args"]["algo"]](
        args=cfg["main_args"],
        algo_args=cfg["algo_args"],
        env_args=cfg["env_args"],
    )

    meta_dir = runner.run_dir
    save_command_txt(meta_dir, sys.argv)
    save_json(get_env_info(), os.path.join(meta_dir, "env_info.json"))
    save_json(summary, os.path.join(meta_dir, "config_resolved.json"))
    save_json({"method": args.method, "map": args.map, "seed": args.seed,
               "is_continuation": False, "resumed_from": checkpoint_dir,
               "completed_steps_before_resume": completed_steps,
               "description": METHODS[args.method]["description"]},
              os.path.join(meta_dir, "method_manifest.json"))

    runner.run()
    runner.close()

    with open(os.path.join(meta_dir, "_TRAINING_COMPLETE"), "w") as f:
        f.write(f"method={args.method} map={args.map} seed={args.seed}\n")
        f.write(f"resumed_from_steps={completed_steps}\n")

    print(f"\n[DONE] method={args.method} map={args.map} seed={args.seed}")
    print(f"  Results: {meta_dir}")


if __name__ == "__main__":
    main()
