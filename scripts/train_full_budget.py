"""
Train a full-budget method (MAPPO_shared, HAPPO_shared, HAPPO_nonshared).

Usage:
    python scripts/train_full_budget.py --method mappo_shared_full --map 3s5z --seed 1
    python scripts/train_full_budget.py --method happo_shared_full --map corridor --seed 2
    python scripts/train_full_budget.py --method happo_nonshared_full --map MMM2 --seed 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = str(SCRIPT_DIR.parent)
HARL_ROOT = str(SCRIPT_DIR.parents[3])
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, HARL_ROOT)

import harl_ext.smac_fix  # noqa: F401 — patch pysc2 DuplicateMapError

from configs.method_registry import (
    METHODS, MAPS, MAPS_EXTENDED, SEEDS, HIDDEN_SIZES,
    get_full_budget_config, resolve_config_summary, BUDGET,
)
from harl_ext.utils import get_git_commit, get_env_info, save_command_txt, save_json


def main():
    parser = argparse.ArgumentParser(description="Full-budget training")
    parser.add_argument("--method", type=str, required=True,
                        choices=[m for m, d in METHODS.items() if not d["is_continuation"]])
    parser.add_argument("--map", type=str, required=True, choices=MAPS_EXTENDED)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Override log directory")
    args = parser.parse_args()

    if args.log_dir is None:
        args.log_dir = os.path.join(PROJECT_DIR, "exp", "smac_robustness",
                                    f"map_{args.map}")

    cfg = get_full_budget_config(args.method, args.map, args.seed, args.log_dir)
    summary = resolve_config_summary(cfg)

    print("=" * 70)
    print(f"  FULL-BUDGET TRAINING")
    print(f"  Method:      {args.method}")
    print(f"  Map:         {args.map}")
    print(f"  Seed:        {args.seed}")
    print(f"  Algo:        {summary['algo_name']}")
    print(f"  share_param: {summary['share_param']}")
    print(f"  hidden_sizes:{summary['hidden_sizes']}")
    print(f"  Steps:       {summary['num_env_steps']:,}")
    print(f"  state_type:  {summary['state_type']}")
    print(f"  Log dir:     {args.log_dir}")
    print("=" * 70)

    assert summary["hidden_sizes"] == list(HIDDEN_SIZES), \
        f"hidden_sizes mismatch: {summary['hidden_sizes']} != {HIDDEN_SIZES}"

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
               "is_continuation": False, "description": METHODS[args.method]["description"]},
              os.path.join(meta_dir, "method_manifest.json"))
    with open(os.path.join(meta_dir, "git_commit.txt"), "w") as f:
        f.write(get_git_commit())

    runner.run()
    runner.close()

    with open(os.path.join(meta_dir, "_TRAINING_COMPLETE"), "w") as f:
        f.write(f"method={args.method} map={args.map} seed={args.seed}\n")

    print(f"\n[DONE] method={args.method} map={args.map} seed={args.seed}")
    print(f"  Results: {meta_dir}")


if __name__ == "__main__":
    main()
