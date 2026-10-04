"""
Two-stage continuation training.

Stage 1: Pretrain (e.g. MAPPO shared)
Stage 2: Continue (e.g. HAPPO shared or nonshared)

Usage:
    python scripts/train_continuation.py --method mappo_to_happo_shared --map corridor --seed 1
    python scripts/train_continuation.py --method mappo_to_mappo_shared --map 3s5z --seed 2
    python scripts/train_continuation.py --method mappo_to_happo_nonshared --map MMM2 --seed 3

    # Run only stage2 if stage1 already done:
    python scripts/train_continuation.py --method mappo_to_happo_shared --map corridor --seed 1 --stage2_only
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
    METHODS, MAPS, MAPS_EXTENDED, SEEDS, HIDDEN_SIZES, BUDGET,
    get_continuation_stage1_config, get_continuation_stage2_config,
    resolve_config_summary,
)
from harl_ext.continuation import (
    find_model_dir_for_seed, check_checkpoint_compatibility,
    write_transition_manifest,
)
from harl_ext.utils import get_git_commit, get_env_info, save_command_txt, save_json


def run_stage(cfg: dict, stage_label: str, meta_extra: dict) -> str:
    """Run a single training stage, return the run_dir path."""
    from harl.runners import RUNNER_REGISTRY

    summary = resolve_config_summary(cfg)
    print(f"\n{'='*70}")
    print(f"  {stage_label}")
    print(f"  Algo:        {summary['algo_name']}")
    print(f"  share_param: {summary['share_param']}")
    print(f"  hidden_sizes:{summary['hidden_sizes']}")
    print(f"  Steps:       {summary['num_env_steps']:,}")
    print(f"  model_dir:   {summary['model_dir']}")
    print(f"{'='*70}")

    assert summary["hidden_sizes"] == list(HIDDEN_SIZES), \
        f"hidden_sizes mismatch: {summary['hidden_sizes']} != {HIDDEN_SIZES}"

    runner = RUNNER_REGISTRY[cfg["main_args"]["algo"]](
        args=cfg["main_args"],
        algo_args=cfg["algo_args"],
        env_args=cfg["env_args"],
    )

    meta_dir = runner.run_dir
    save_command_txt(meta_dir, sys.argv)
    save_json(get_env_info(), os.path.join(meta_dir, "env_info.json"))
    save_json(summary, os.path.join(meta_dir, "config_resolved.json"))
    save_json({**meta_extra, "stage": stage_label},
              os.path.join(meta_dir, "method_manifest.json"))
    with open(os.path.join(meta_dir, "git_commit.txt"), "w") as f:
        f.write(get_git_commit())

    runner.run()
    runner.close()

    with open(os.path.join(meta_dir, "_TRAINING_COMPLETE"), "w") as f:
        f.write(f"{stage_label} done\n")

    print(f"  [{stage_label} DONE] Results: {meta_dir}")
    return meta_dir


def main():
    parser = argparse.ArgumentParser(description="Continuation training (2-stage)")
    parser.add_argument("--method", type=str, required=True,
                        choices=[m for m, d in METHODS.items() if d["is_continuation"]])
    parser.add_argument("--map", type=str, required=True, choices=MAPS_EXTENDED)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--stage2_only", action="store_true",
                        help="Skip stage1, assume it's already done")
    parser.add_argument("--stage1_model_dir", type=str, default=None,
                        help="Explicit path to stage1 model dir (for --stage2_only)")
    args = parser.parse_args()

    mdef = METHODS[args.method]
    if args.log_dir is None:
        args.log_dir = os.path.join(PROJECT_DIR, "exp", "smac_robustness",
                                    f"map_{args.map}")

    meta_extra = {
        "method": args.method,
        "map": args.map,
        "seed": args.seed,
        "is_continuation": True,
        "description": mdef["description"],
    }

    stage1_model_dir = args.stage1_model_dir

    if not args.stage2_only:
        cfg1 = get_continuation_stage1_config(args.method, args.map, args.seed, args.log_dir)
        stage1_run_dir = run_stage(cfg1, "STAGE 1 (pretrain)", meta_extra)
        stage1_model_dir = os.path.join(stage1_run_dir, "models")
    else:
        if stage1_model_dir is None:
            stage1_model_dir = find_model_dir_for_seed(
                args.log_dir, "smac", args.map,
                mdef["algo_stage1"], f"{args.method}_stage1", args.seed,
            )
        if stage1_model_dir is None:
            print(f"[ERROR] Cannot find stage1 model for {args.method} map={args.map} seed={args.seed}")
            print(f"  Searched in: {args.log_dir}/smac/{args.map}/{mdef['algo_stage1']}/{args.method}_stage1/")
            sys.exit(1)

    print(f"\n  Stage1 model dir: {stage1_model_dir}")

    from harl.envs.smac.smac_maps import get_map_params
    n_agents = get_map_params(args.map)["n_agents"]
    load_report = check_checkpoint_compatibility(
        stage1_model_dir, mdef["algo_stage2"], mdef["share_param_stage2"],
        n_agents, list(HIDDEN_SIZES),
    )

    if not load_report["compatible"]:
        report_path = os.path.join(args.log_dir, f"load_report_FAIL_{args.method}_s{args.seed}.json")
        save_json(load_report, report_path)
        print(f"[ERROR] Checkpoint incompatible! Report: {report_path}")
        for w in load_report["warnings"]:
            print(f"  WARNING: {w}")
        sys.exit(1)

    cfg2 = get_continuation_stage2_config(
        args.method, args.map, args.seed, args.log_dir,
        model_dir=stage1_model_dir,
    )
    stage2_run_dir = run_stage(cfg2, "STAGE 2 (continuation)", meta_extra)

    manifest_path = os.path.join(stage2_run_dir, "transition_manifest.json")
    write_transition_manifest(
        output_path=manifest_path,
        source_algo=mdef["algo_stage1"],
        target_algo=mdef["algo_stage2"],
        source_checkpoint_path=stage1_model_dir,
        target_config_path=os.path.join(stage2_run_dir, "config_resolved.json"),
        share_param_source=mdef["share_param_stage1"],
        share_param_target=mdef["share_param_stage2"],
        hidden_sizes=list(HIDDEN_SIZES),
        load_report=load_report,
    )

    save_json(load_report, os.path.join(stage2_run_dir, "load_report.json"))

    print(f"\n[ALL DONE] method={args.method} map={args.map} seed={args.seed}")
    print(f"  Stage2 results: {stage2_run_dir}")


if __name__ == "__main__":
    main()
