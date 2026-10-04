#!/usr/bin/env python3
"""
MPE training and coordination-connectivity evaluation.

Trains MAPPO-shared, HAPPO-nonshared, and MAPPO->HAPPO continuation on
PettingZoo MPE cooperative scenarios, then runs interpolation evaluation.

Usage:
    python scripts/exp_mpe_train_and_eval.py --phase train --scenario simple_spread_v2 --seed 1
    python scripts/exp_mpe_train_and_eval.py --phase eval --scenario simple_spread_v2
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, PROJECT_DIR)
from configs.method_registry import HARL_ROOT, HIDDEN_SIZES
sys.path.insert(0, HARL_ROOT)

MPE_SCENARIOS = {
    "simple_spread_v2": {
        "n_agents": 3,
        "continuous": False,
        "episode_length": 25,
        "total_steps": 10_000_000,
    },
    "simple_reference_v2": {
        "n_agents": 2,
        "continuous": False,
        "episode_length": 25,
        "total_steps": 10_000_000,
    },
}

MPE_EXP_DIR = os.path.join(PROJECT_DIR, "exp", "mpe_robustness")
MPE_RESULTS_DIR = os.path.join(PROJECT_DIR, "exp", "mpe_cc")

SEEDS = [1, 2, 3, 4]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_EVAL_EPISODES = 50


def load_mpe_tuned_config(scenario, continuous=False):
    suffix = "continuous" if continuous else "discrete"
    cfg_path = os.path.join(
        HARL_ROOT, "tuned_configs", "pettingzoo_mpe",
        f"{scenario}-{suffix}", "happo", "config.json"
    )
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["algo_args"]["model"]["hidden_sizes"] = list(HIDDEN_SIZES)
    return cfg


def build_mpe_config(scenario, algo, share_param, seed, num_env_steps,
                     exp_name, model_dir=None, log_dir=None):
    info = MPE_SCENARIOS[scenario]
    cfg = load_mpe_tuned_config(scenario, continuous=info["continuous"])
    cfg["main_args"]["algo"] = algo
    cfg["main_args"]["env"] = "pettingzoo_mpe"
    cfg["main_args"]["exp_name"] = exp_name
    cfg["main_args"]["load_config"] = ""
    cfg["env_args"]["scenario"] = scenario
    cfg["env_args"]["continuous_actions"] = info["continuous"]
    cfg["algo_args"]["model"]["hidden_sizes"] = list(HIDDEN_SIZES)
    cfg["algo_args"]["algo"]["share_param"] = share_param
    if algo == "mappo":
        cfg["algo_args"]["algo"]["fixed_order"] = True
    elif algo == "happo":
        cfg["algo_args"]["algo"]["fixed_order"] = False
    cfg["algo_args"]["seed"]["seed"] = seed
    cfg["algo_args"]["seed"]["seed_specify"] = True
    cfg["algo_args"]["train"]["num_env_steps"] = num_env_steps
    cfg["algo_args"]["train"]["model_dir"] = model_dir
    cfg["algo_args"]["train"]["episode_length"] = info["episode_length"]
    cfg["algo_args"]["train"]["n_rollout_threads"] = 20
    if log_dir:
        cfg["algo_args"]["logger"]["log_dir"] = log_dir
    return cfg


def train_method(scenario, method, seed):
    from harl.runners import RUNNER_REGISTRY

    info = MPE_SCENARIOS[scenario]
    total = info["total_steps"]
    pretrain_steps = int(total * 2 / 3)
    continue_steps = total - pretrain_steps
    log_dir = os.path.join(MPE_EXP_DIR, f"scenario_{scenario}")

    if method == "mappo_shared_full":
        cfg = build_mpe_config(scenario, "mappo", True, seed, total,
                               "mappo_shared_full", log_dir=log_dir)
        runner = RUNNER_REGISTRY["mappo"](
            args=cfg["main_args"], algo_args=cfg["algo_args"], env_args=cfg["env_args"])
        runner.run()
        runner.close()

    elif method == "happo_nonshared_full":
        cfg = build_mpe_config(scenario, "happo", False, seed, total,
                               "happo_nonshared_full", log_dir=log_dir)
        runner = RUNNER_REGISTRY["happo"](
            args=cfg["main_args"], algo_args=cfg["algo_args"], env_args=cfg["env_args"])
        runner.run()
        runner.close()

    elif method == "mappo_to_happo_nonshared":
        cfg1 = build_mpe_config(scenario, "mappo", True, seed, pretrain_steps,
                                "mappo_to_happo_nonshared_stage1", log_dir=log_dir)
        runner1 = RUNNER_REGISTRY["mappo"](
            args=cfg1["main_args"], algo_args=cfg1["algo_args"], env_args=cfg1["env_args"])
        runner1.run()
        model_dir = os.path.join(runner1.run_dir, "models")
        runner1.close()

        cfg2 = build_mpe_config(scenario, "happo", False, seed, continue_steps,
                                "mappo_to_happo_nonshared_stage2",
                                model_dir=model_dir, log_dir=log_dir)
        runner2 = RUNNER_REGISTRY["happo"](
            args=cfg2["main_args"], algo_args=cfg2["algo_args"], env_args=cfg2["env_args"])
        runner2.run()
        runner2.close()

    elif method == "mappo_to_mappo_nonshared":
        cfg1 = build_mpe_config(scenario, "mappo", True, seed, pretrain_steps,
                                "mappo_to_mappo_nonshared_stage1", log_dir=log_dir)
        runner1 = RUNNER_REGISTRY["mappo"](
            args=cfg1["main_args"], algo_args=cfg1["algo_args"], env_args=cfg1["env_args"])
        runner1.run()
        model_dir = os.path.join(runner1.run_dir, "models")
        runner1.close()

        cfg2 = build_mpe_config(scenario, "mappo", False, seed, continue_steps,
                                "mappo_to_mappo_nonshared_stage2",
                                model_dir=model_dir, log_dir=log_dir)
        runner2 = RUNNER_REGISTRY["mappo"](
            args=cfg2["main_args"], algo_args=cfg2["algo_args"], env_args=cfg2["env_args"])
        runner2.run()
        runner2.close()

    print(f"[DONE] {method} / {scenario} / seed={seed}", flush=True)


def find_mpe_model_dir(scenario, method_name, seed):
    suffix = "continuous" if MPE_SCENARIOS[scenario]["continuous"] else "discrete"
    task = f"{scenario}-{suffix}"
    base = os.path.join(MPE_EXP_DIR, f"scenario_{scenario}", "pettingzoo_mpe",
                        task)
    algo = "happo" if "happo" in method_name and "to_mappo" not in method_name else "mappo"
    pattern = os.path.join(base, algo, method_name, f"seed-{seed:05d}-*", "models")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No model found: {pattern}")
    return candidates[-1]


def load_mpe_state_dicts(scenario, method_name, seed):
    model_dir = find_mpe_model_dir(scenario, method_name, seed)
    info = MPE_SCENARIOS[scenario]
    n_agents = info["n_agents"]
    is_shared = "shared" in method_name and "nonshared" not in method_name

    sds = []
    if is_shared:
        sd = torch.load(os.path.join(model_dir, "actor_agent0.pt"),
                        map_location="cpu", weights_only=False)
        sds = [copy.deepcopy(sd) for _ in range(n_agents)]
    else:
        for i in range(n_agents):
            sd = torch.load(os.path.join(model_dir, f"actor_agent{i}.pt"),
                            map_location="cpu", weights_only=False)
            sds.append(sd)
    return sds


def interpolate_sds(sds_a, sds_b, alpha):
    return [{k: (1.0 - alpha) * sds_a[i][k] + alpha * sds_b[i][k]
             for k in sds_a[i]} for i in range(len(sds_a))]


def make_mpe_env(scenario):
    from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
    info = MPE_SCENARIOS[scenario]
    env_args = {
        "scenario": scenario,
        "continuous_actions": info["continuous"],
        "max_cycles": info["episode_length"],
    }
    return PettingZooMPEEnv(env_args)


def create_mpe_actors(scenario, sds):
    from harl.models.policy_models.stochastic_policy import StochasticPolicy
    from gym.spaces import Discrete, Box

    cfg = load_mpe_tuned_config(scenario, MPE_SCENARIOS[scenario]["continuous"])
    model_cfg = copy.deepcopy(cfg["algo_args"]["model"])
    for k, v in cfg["algo_args"]["algo"].items():
        model_cfg.setdefault(k, v)

    env = make_mpe_env(scenario)
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    env.close()

    model_cfg["action_space"] = act_space
    actors = []
    for sd in sds:
        actor = StochasticPolicy(model_cfg, obs_space, act_space,
                                 device=torch.device("cpu"))
        actor.load_state_dict(sd)
        actor.eval()
        actors.append(actor)
    return actors


def run_mpe_episode(env, actors, scenario):
    info = MPE_SCENARIOS[scenario]
    n_agents = info["n_agents"]
    hidden_size = HIDDEN_SIZES[-1]
    rnn_states = [torch.zeros(1, 1, hidden_size) for _ in range(n_agents)]
    masks = [torch.ones(1, 1) for _ in range(n_agents)]

    obs, _, avail = env.reset()
    total_reward = 0.0

    for step in range(info["episode_length"]):
        actions = []
        for i in range(n_agents):
            obs_t = torch.FloatTensor(np.array(obs[i])).unsqueeze(0)
            avail_t = None
            if avail is not None:
                avail_t = torch.FloatTensor(np.array(avail[i])).unsqueeze(0)
            with torch.no_grad():
                action, _, rnn_out = actors[i](
                    obs_t, rnn_states[i], masks[i],
                    available_actions=avail_t, deterministic=True)
                rnn_states[i] = rnn_out
            actions.append(action.item())

        obs, _, rewards, dones, infos, avail = env.step(np.array(actions))
        total_reward += float(rewards[0][0])

        if all(dones):
            break

    return total_reward


def eval_mpe_reward(sds, scenario, n_ep):
    actors = create_mpe_actors(scenario, sds)
    env = make_mpe_env(scenario)
    rewards = []
    for _ in range(n_ep):
        r = run_mpe_episode(env, actors, scenario)
        rewards.append(r)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def compute_barrier_reward(rewards_over_alpha):
    endpoints_min = min(rewards_over_alpha[0], rewards_over_alpha[-1])
    worst = min(rewards_over_alpha)
    return max(0, endpoints_min - worst)


def eval_mpe_interpolation(scenario, seeds):
    os.makedirs(MPE_RESULTS_DIR, exist_ok=True)

    interp_paths = [
        ("P1", "mappo_to_mappo_nonshared_stage2", "mappo_to_happo_nonshared_stage2",
         "MAPPO-ns vs M2H (same source)"),
        ("P2", "mappo_shared_full", "happo_nonshared_full",
         "MAPPO vs HAPPO (cross-history)"),
    ]

    results = {}
    for path_id, method_a, method_b, desc in interp_paths:
        print(f"\n  {path_id}: {desc}", flush=True)
        seed_rews, seed_barriers = [], []
        for seed in seeds:
            print(f"    seed={seed}", end="", flush=True)
            try:
                sds_a = load_mpe_state_dicts(scenario, method_a, seed)
                sds_b = load_mpe_state_dicts(scenario, method_b, seed)
            except FileNotFoundError as e:
                print(f" SKIP: {e}", flush=True)
                continue

            rews = []
            for alpha in ALPHAS:
                sds_interp = interpolate_sds(sds_a, sds_b, alpha)
                mean_r, _ = eval_mpe_reward(sds_interp, scenario, N_EVAL_EPISODES)
                rews.append(mean_r)
                print(f" {alpha:.1f}:{mean_r:.1f}", end="", flush=True)

            barrier = compute_barrier_reward(rews)
            seed_rews.append(rews)
            seed_barriers.append(barrier)
            print(f"  B={barrier:.2f}", flush=True)

        if not seed_rews:
            continue

        arr = np.array(seed_rews)
        results[path_id] = {
            "description": desc,
            "alphas": ALPHAS,
            "mean_reward": arr.mean(axis=0).tolist(),
            "std_reward": arr.std(axis=0).tolist(),
            "per_seed_reward": seed_rews,
            "mean_barrier": float(np.mean(seed_barriers)),
            "std_barrier": float(np.std(seed_barriers)),
        }

    outfile = os.path.join(MPE_RESULTS_DIR, f"mpe_cc_{scenario}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "eval"], required=True)
    parser.add_argument("--scenario", type=str, default="simple_spread_v2",
                        choices=list(MPE_SCENARIOS.keys()))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--method", type=str, default=None,
                        choices=["mappo_shared_full", "happo_nonshared_full",
                                 "mappo_to_happo_nonshared", "mappo_to_mappo_nonshared"])
    parser.add_argument("--seeds", type=str, default="1,2,3,4")
    args = parser.parse_args()

    if args.phase == "train":
        methods = [args.method] if args.method else [
            "mappo_shared_full", "happo_nonshared_full",
            "mappo_to_happo_nonshared", "mappo_to_mappo_nonshared",
        ]
        for m in methods:
            print(f"\n{'='*60}", flush=True)
            print(f"  Training {m} on {args.scenario} seed={args.seed}", flush=True)
            print(f"{'='*60}", flush=True)
            train_method(args.scenario, m, args.seed)

    elif args.phase == "eval":
        seeds = [int(s) for s in args.seeds.split(",")]
        eval_mpe_interpolation(args.scenario, seeds)


if __name__ == "__main__":
    main()
