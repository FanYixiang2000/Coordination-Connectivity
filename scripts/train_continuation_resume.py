"""
Two-stage continuation training with automatic checkpoint resume.

If a previous run for either stage exists with a checkpoint, resumes from there.

Usage:
    python scripts/train_continuation_resume.py --method mappo_to_happo_nonshared --map 6h_vs_8z --seed 2
    python scripts/train_continuation_resume.py --method mappo_to_happo_nonshared --map 6h_vs_8z --seed 2 --stage2_only
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
    METHODS, MAPS_EXTENDED, SEEDS, HIDDEN_SIZES, BUDGET,
    MAP_EPISODE_LENGTH, N_ROLLOUT_THREADS,
    get_continuation_stage1_config, get_continuation_stage2_config,
    resolve_config_summary,
)
from harl_ext.continuation import (
    find_model_dir_for_seed, check_checkpoint_compatibility,
    write_transition_manifest,
)
from harl_ext.utils import get_git_commit, get_env_info, save_command_txt, save_json
from harl_ext.run_selection import find_best_smac_checkpoint, validate_rollout_threads


def find_best_checkpoint(log_dir: str, map_name: str, algo: str, exp_name: str, seed: int):
    """Find the most advanced checkpoint for this job (threads=20 only).

    Returns (model_dir, completed_steps, run_dir) or (None, 0, None).
    """
    return find_best_smac_checkpoint(log_dir, map_name, algo, exp_name, seed)


def is_stage_complete(log_dir: str, map_name: str, algo: str, exp_name: str, seed: int):
    """Check if a stage is already fully completed."""
    seed_str = f"{seed:05d}"
    base_path = os.path.join(log_dir, "smac", map_name, algo, exp_name)
    if not os.path.isdir(base_path):
        return False
    for seed_dir in glob.glob(os.path.join(base_path, f"seed-{seed_str}-*")):
        if os.path.isfile(os.path.join(seed_dir, "_TRAINING_COMPLETE")):
            return True
    return False


def run_stage_with_resume(cfg: dict, stage_label: str, meta_extra: dict,
                          stage_budget: int, log_dir: str, map_name: str,
                          algo: str, exp_name: str, seed: int) -> str:
    """Run a training stage with checkpoint resume support."""
    from harl.runners import RUNNER_REGISTRY

    checkpoint_dir, completed_steps, _ = find_best_checkpoint(
        log_dir, map_name, algo, exp_name, seed
    )

    remaining_steps = stage_budget - completed_steps

    if completed_steps > 0 and checkpoint_dir:
        pct = completed_steps * 100 // stage_budget
        print(f"  [RESUME] Found checkpoint at {pct}% ({completed_steps:,}/{stage_budget:,} steps)")
        print(f"    Checkpoint: {checkpoint_dir}")
        print(f"    Remaining:  {remaining_steps:,} steps")

        if remaining_steps <= 0 or pct >= 95:
            print(f"  [SKIP] Stage already completed ({pct}% >= 95%).")
            parent = os.path.dirname(checkpoint_dir)
            return parent

        cfg["algo_args"]["train"]["num_env_steps"] = remaining_steps
        cfg["algo_args"]["train"]["model_dir"] = checkpoint_dir

        ep_len = MAP_EPISODE_LENGTH[map_name]
        total_episodes = remaining_steps // (ep_len * N_ROLLOUT_THREADS)
        eval_interval = max(1, int(total_episodes * 10 / 100))
        cfg["algo_args"]["train"]["eval_interval"] = eval_interval
        cfg["algo_args"]["train"]["log_interval"] = max(1, eval_interval // 5)
    else:
        print(f"  [FRESH] No checkpoint found, starting from scratch.")

    summary = resolve_config_summary(cfg)
    print(f"\n{'='*70}")
    print(f"  {stage_label} {'(RESUMED)' if checkpoint_dir and completed_steps > 0 else '(FRESH)'}")
    print(f"  Algo:        {summary['algo_name']}")
    print(f"  share_param: {summary['share_param']}")
    print(f"  Steps:       {summary['num_env_steps']:,} (of {stage_budget:,} total)")
    print(f"  model_dir:   {summary['model_dir']}")
    print(f"{'='*70}")

    runner = RUNNER_REGISTRY[cfg["main_args"]["algo"]](
        args=cfg["main_args"],
        algo_args=cfg["algo_args"],
        env_args=cfg["env_args"],
    )

    meta_dir = runner.run_dir
    save_command_txt(meta_dir, sys.argv)
    save_json(get_env_info(), os.path.join(meta_dir, "env_info.json"))
    save_json(summary, os.path.join(meta_dir, "config_resolved.json"))
    save_json({**meta_extra, "stage": stage_label,
               "resumed_from": checkpoint_dir,
               "completed_steps_before_resume": completed_steps},
              os.path.join(meta_dir, "method_manifest.json"))

    runner.run()
    runner.close()

    with open(os.path.join(meta_dir, "_TRAINING_COMPLETE"), "w") as f:
        f.write(f"{stage_label} done\n")
        f.write(f"resumed_from_steps={completed_steps}\n")

    print(f"  [{stage_label} DONE] Results: {meta_dir}")
    return meta_dir


def main():
    parser = argparse.ArgumentParser(description="Continuation training with auto-resume")
    parser.add_argument("--method", type=str, required=True,
                        choices=[m for m, d in METHODS.items() if d["is_continuation"]])
    parser.add_argument("--map", type=str, required=True, choices=MAPS_EXTENDED)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--stage2_only", action="store_true",
                        help="Skip stage1, assume it's already done")
    parser.add_argument("--stage1_model_dir", type=str, default=None,
                        help="Explicit path to stage1 model dir (for --stage2_only)")
    parser.add_argument("--threads", type=int, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        validate_rollout_threads(args.threads)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    mdef = METHODS[args.method]
    if args.log_dir is None:
        args.log_dir = os.path.join(PROJECT_DIR, "exp", "smac_robustness",
                                    f"map_{args.map}")

    meta_extra = {
        "method": args.method, "map": args.map, "seed": args.seed,
        "is_continuation": True, "description": mdef["description"],
    }

    stage1_model_dir = args.stage1_model_dir

    if not args.stage2_only:
        stage1_budget = BUDGET[args.map]["pretrain"]
        cfg1 = get_continuation_stage1_config(args.method, args.map, args.seed, args.log_dir)

        stage1_complete = is_stage_complete(
            args.log_dir, args.map, mdef["algo_stage1"],
            f"{args.method}_stage1", args.seed
        )

        if stage1_complete:
            print(f"[STAGE1 ALREADY COMPLETE] Skipping to stage2.")
            stage1_model_dir = find_model_dir_for_seed(
                args.log_dir, "smac", args.map,
                mdef["algo_stage1"], f"{args.method}_stage1", args.seed,
            )
        else:
            stage1_run_dir = run_stage_with_resume(
                cfg1, "STAGE 1 (pretrain)", meta_extra,
                stage1_budget, args.log_dir, args.map,
                mdef["algo_stage1"], f"{args.method}_stage1", args.seed
            )
            stage1_model_dir = os.path.join(stage1_run_dir, "models")
    else:
        if stage1_model_dir is None:
            stage1_model_dir = find_model_dir_for_seed(
                args.log_dir, "smac", args.map,
                mdef["algo_stage1"], f"{args.method}_stage1", args.seed,
            )
        if stage1_model_dir is None:
            print(f"[ERROR] Cannot find stage1 model for {args.method} map={args.map} seed={args.seed}")
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

    stage2_budget = BUDGET[args.map]["continue"]
    cfg2 = get_continuation_stage2_config(
        args.method, args.map, args.seed, args.log_dir,
        model_dir=stage1_model_dir,
    )
    stage2_run_dir = run_stage_with_resume(
        cfg2, "STAGE 2 (continuation)", meta_extra,
        stage2_budget, args.log_dir, args.map,
        mdef["algo_stage2"], f"{args.method}_stage2", args.seed
    )

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
