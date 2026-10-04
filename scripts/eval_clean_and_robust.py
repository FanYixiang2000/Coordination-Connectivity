"""
Evaluate trained models under clean and perturbed conditions (100 episodes each).

Usage:
    # Evaluate a specific model directory:
    python scripts/eval_clean_and_robust.py \
        --model_dir exp/smac_robustness/map_corridor/smac/corridor/mappo/mappo_shared_full/seed-00001-.../models \
        --map corridor --algo mappo --share_param \
        --output_dir exp/smac_robustness/map_corridor/eval_results/mappo_shared_full_s1

    # Or use the auto-discovery mode:
    python scripts/eval_clean_and_robust.py \
        --method mappo_shared_full --map corridor --seed 1 --auto
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

from configs.method_registry import METHODS, MAPS, SEEDS, HIDDEN_SIZES
from harl_ext.evaluator import evaluate_model, save_eval_result
from harl_ext.perturbations import get_perturbation_configs
from harl_ext.continuation import find_model_dir_for_seed
from harl_ext.utils import save_json


def find_auto_model_dir(method: str, map_name: str, seed: int, log_dir: str) -> str:
    """Auto-discover model directory from method/map/seed."""
    mdef = METHODS[method]
    if mdef["is_continuation"]:
        algo = mdef["algo_stage2"]
        exp = f"{method}_stage2"
    else:
        algo = mdef["algo"]
        exp = method

    model_dir = find_model_dir_for_seed(log_dir, "smac", map_name, algo, exp, seed)
    if model_dir is None:
        raise FileNotFoundError(
            f"Cannot find model for method={method} map={map_name} seed={seed}\n"
            f"Searched: {log_dir}/smac/{map_name}/{algo}/{exp}/seed-{seed:05d}-*/models"
        )
    return model_dir


def load_config_from_run(model_dir: str) -> dict:
    """Load the config.json from the run directory (parent of models/)."""
    run_dir = str(Path(model_dir).parent)
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Clean + robustness evaluation (100 ep)")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--map", type=str, required=True, choices=MAPS)
    parser.add_argument("--algo", type=str, default=None, choices=["mappo", "happo"])
    parser.add_argument("--share_param", action="store_true", default=None)
    parser.add_argument("--no_share_param", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=100)

    parser.add_argument("--method", type=str, default=None, choices=list(METHODS.keys()))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover model dir from --method/--map/--seed")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--clean_only", action="store_true")
    parser.add_argument("--condition", type=str, default=None,
                        help="Evaluate only one condition, e.g. 'missing_0.2' or 'delay_2'")
    args = parser.parse_args()

    if args.auto:
        assert args.method and args.seed, "--auto requires --method and --seed"
        if args.log_dir is None:
            args.log_dir = os.path.join(PROJECT_DIR, "exp", "smac_robustness",
                                        f"map_{args.map}")
        args.model_dir = find_auto_model_dir(args.method, args.map, args.seed, args.log_dir)

        mdef = METHODS[args.method]
        if mdef["is_continuation"]:
            args.algo = mdef["algo_stage2"]
            sp = mdef["share_param_stage2"]
        else:
            args.algo = mdef["algo"]
            sp = mdef["share_param"]
        if args.share_param is None and not args.no_share_param:
            args.share_param = sp

        if args.output_dir is None:
            args.output_dir = os.path.join(
                args.log_dir, "eval_results",
                f"{args.method}_s{args.seed}"
            )

    assert args.model_dir, "Must provide --model_dir or use --auto"
    assert args.algo, "Must provide --algo or use --auto"

    if args.no_share_param:
        share_param = False
    elif args.share_param:
        share_param = True
    else:
        share_param = False

    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_DIR, "exp", "eval_results",
                                       f"{args.map}_{args.algo}")

    run_cfg = load_config_from_run(args.model_dir)

    conditions = get_perturbation_configs()
    if args.clean_only:
        conditions = [c for c in conditions if c["type"] == "clean"]
    elif args.condition:
        parts = args.condition.split("_")
        ptype = parts[0]
        if ptype == "clean":
            conditions = [{"type": "clean"}]
        elif ptype == "missing":
            conditions = [{"type": "missing", "missing_rate": float(parts[1])}]
        elif ptype == "noise":
            conditions = [{"type": "noise", "sigma": float(parts[1])}]
        elif ptype == "delay":
            conditions = [{"type": "delay", "delay": int(parts[1])}]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"  ROBUSTNESS EVALUATION")
    print(f"  Model:     {args.model_dir}")
    print(f"  Map:       {args.map}")
    print(f"  Algo:      {args.algo}")
    print(f"  Shared:    {share_param}")
    print(f"  Episodes:  {args.num_episodes}")
    print(f"  Conditions:{len(conditions)}")
    print(f"  Output:    {args.output_dir}")
    print("=" * 70)

    all_results = []
    for cond in conditions:
        ctype = cond["type"]
        if ctype == "clean":
            fname = "clean_eval.json"
            label = "clean"
        elif ctype == "missing":
            fname = f"missing_eval_rate_{cond['missing_rate']}.json"
            label = f"missing(rate={cond['missing_rate']})"
        elif ctype == "noise":
            fname = f"noise_eval_sigma_{cond['sigma']}.json"
            label = f"noise(sigma={cond['sigma']})"
        elif ctype == "delay":
            fname = f"delay_eval_{cond['delay']}.json"
            label = f"delay(d={cond['delay']})"
        else:
            continue

        out_path = os.path.join(args.output_dir, fname)
        if os.path.exists(out_path):
            print(f"  [SKIP] {label} — already exists: {out_path}")
            with open(out_path) as f:
                result = json.load(f)
            all_results.append(result)
            continue

        print(f"\n  Evaluating: {label} ...")
        t0 = time.time()

        algo_args = run_cfg if run_cfg else {
            "algo_args": {
                "model": {
                    "hidden_sizes": list(HIDDEN_SIZES),
                    "recurrent_n": 1,
                    "use_recurrent_policy": False,
                    "use_naive_recurrent_policy": False,
                    "activation_func": "relu",
                    "use_feature_normalization": True,
                    "initialization_method": "orthogonal_",
                    "gain": 0.01,
                    "lr": 0.0005,
                    "critic_lr": 0.0005,
                    "opti_eps": 1e-5,
                    "weight_decay": 0,
                    "std_x_coef": 1,
                    "std_y_coef": 0.5,
                    "data_chunk_length": 10,
                },
                "algo": {
                    "ppo_epoch": 5, "critic_epoch": 5,
                    "use_clipped_value_loss": True, "clip_param": 0.2,
                    "actor_num_mini_batch": 1, "critic_num_mini_batch": 1,
                    "entropy_coef": 0.01, "value_loss_coef": 1,
                    "use_max_grad_norm": True, "max_grad_norm": 10.0,
                    "use_gae": True, "gamma": 0.99, "gae_lambda": 0.95,
                    "use_huber_loss": True, "use_policy_active_masks": True,
                    "huber_delta": 10.0, "action_aggregation": "prod",
                    "share_param": share_param, "fixed_order": False,
                },
            },
            "env_args": {"map_name": args.map, "state_type": "EP"},
        }

        result = evaluate_model(
            model_dir=args.model_dir,
            map_name=args.map,
            algo_name=args.algo,
            share_param=share_param,
            algo_args=algo_args,
            num_episodes=args.num_episodes,
            perturbation=cond,
        )
        elapsed = time.time() - t0
        result["eval_time_seconds"] = elapsed

        save_eval_result(result, out_path)
        all_results.append(result)
        print(f"    Win rate: {result['win_rate']:.3f} "
              f"(±{result['win_rate_stderr']:.3f}) [{elapsed:.1f}s]")

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "perturbation": r["perturbation"],
            "win_rate": r["win_rate"],
            "win_rate_ci95_low": r.get("win_rate_ci95_low"),
            "win_rate_ci95_high": r.get("win_rate_ci95_high"),
            "avg_reward": r["avg_reward"],
            "avg_episode_length": r["avg_episode_length"],
        })
    save_json({"summary": summary_rows}, summary_path)

    print(f"\n[ALL DONE] {len(all_results)} conditions evaluated.")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
