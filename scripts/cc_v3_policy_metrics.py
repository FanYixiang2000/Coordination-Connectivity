#!/usr/bin/env python3
"""Q4 rebuttal metrics for coordination connectivity.

This script is read-only with respect to trained runs and existing CC JSON. It
computes parameter distance and policy-behavior divergence for existing CC v3
paths, then saves new results under exp/q4_rebuttal/.
"""

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_DIR, "exp", "q4_rebuttal")
os.makedirs(OUT_DIR, exist_ok=True)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

sys.path.insert(0, SCRIPT_DIR)
import coordination_connectivity_v3 as cc  # noqa: E402
from cc_v3_supplement import EXTRA_INTERP_PATHS  # noqa: E402


ALL_PATHS = {pid: (pid, a, b, desc) for pid, a, b, desc in cc.INTERP_PATHS}
ALL_PATHS.update({pid: (pid, a, b, desc) for pid, a, b, desc in EXTRA_INTERP_PATHS})


def flatten_sd(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.detach().cpu().float().reshape(-1) for v in sd.values()])


def weight_pair_metrics(sds_a, sds_b) -> Dict:
    per_agent = []
    for sd_a, sd_b in zip(sds_a, sds_b):
        va = flatten_sd(sd_a)
        vb = flatten_sd(sd_b)
        diff = va - vb
        denom = torch.norm(va) + torch.norm(vb) + 1e-12
        cos = torch.dot(va, vb) / (torch.norm(va) * torch.norm(vb) + 1e-12)
        per_agent.append(
            {
                "l2": float(torch.norm(diff).item()),
                "normalized_l2": float((torch.norm(diff) / denom).item()),
                "cosine": float(cos.item()),
            }
        )
    return {
        "mean_l2": float(np.mean([x["l2"] for x in per_agent])),
        "mean_normalized_l2": float(np.mean([x["normalized_l2"] for x in per_agent])),
        "mean_cosine": float(np.mean([x["cosine"] for x in per_agent])),
        "per_agent": per_agent,
    }


def _to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


@torch.no_grad()
def actor_probs_and_state(actor, obs, rnn_state, mask, avail):
    obs_t = _to_tensor(obs).unsqueeze(0)
    avail_t = _to_tensor(avail).unsqueeze(0)
    actor_features = actor.base(obs_t)
    if actor.use_naive_recurrent_policy or actor.use_recurrent_policy:
        actor_features, next_state = actor.rnn(actor_features, rnn_state, mask)
    else:
        next_state = rnn_state
    logits = actor.act.get_logits(actor_features, avail_t)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probs, next_state


def divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pq = float(np.sum(p * (np.log(p) - np.log(q))))
    kl_qp = float(np.sum(q * (np.log(q) - np.log(p))))
    js = 0.5 * np.sum(p * (np.log(p) - np.log(m))) + 0.5 * np.sum(q * (np.log(q) - np.log(m)))
    tv = 0.5 * np.sum(np.abs(p - q))
    return {
        "kl_ab": kl_pq,
        "kl_ba": kl_qp,
        "sym_kl": 0.5 * (kl_pq + kl_qp),
        "js": float(js),
        "tv": float(tv),
        "argmax_disagree": float(int(np.argmax(p) != np.argmax(q))),
    }


def init_rnn_states(map_name):
    cfg = cc.load_base_config(map_name)
    recurrent_n = cfg["algo_args"]["model"].get("recurrent_n", 1)
    hidden_size = cc.HIDDEN_SIZES[-1]
    n_agents = cc.MAP_AGENTS[map_name]
    return [torch.zeros(1, recurrent_n, hidden_size) for _ in range(n_agents)]


def eval_policy_metrics_for_alpha(
    map_name: str,
    sds_a,
    sds_b,
    alpha: float,
    episodes: int,
) -> Dict:
    n_agents = cc.MAP_AGENTS[map_name]
    sds_interp = cc.interpolate_sds(sds_a, sds_b, alpha)
    actors_interp = cc.create_actors(map_name, sds_interp)
    actors_a = cc.create_actors(map_name, sds_a)
    actors_b = cc.create_actors(map_name, sds_b)
    env = cc.make_env(map_name)

    mask = torch.ones(1, 1)
    sums = {
        "kl_ab": 0.0,
        "kl_ba": 0.0,
        "sym_kl": 0.0,
        "js": 0.0,
        "tv": 0.0,
        "argmax_disagree": 0.0,
        "joint_sym_kl": 0.0,
        "joint_js_sum": 0.0,
        "joint_tv_bound": 0.0,
        "joint_any_argmax_disagree": 0.0,
    }
    steps = 0
    agent_steps = 0
    wins = 0

    for _ in range(episodes):
        rnn_i = init_rnn_states(map_name)
        rnn_a = init_rnn_states(map_name)
        rnn_b = init_rnn_states(map_name)
        env.reset()
        terminated = False
        info = {}
        while not terminated:
            obs_list = [env.get_obs_agent(i) for i in range(n_agents)]
            avail_list = [env.get_avail_agent_actions(i) for i in range(n_agents)]
            actions = []
            joint_sym_kl = 0.0
            joint_js = 0.0
            joint_tv_bound = 0.0
            any_disagree = 0.0

            for i in range(n_agents):
                p_a, rnn_a[i] = actor_probs_and_state(
                    actors_a[i], obs_list[i], rnn_a[i], mask, avail_list[i]
                )
                p_b, rnn_b[i] = actor_probs_and_state(
                    actors_b[i], obs_list[i], rnn_b[i], mask, avail_list[i]
                )
                div = divergence(p_a, p_b)
                for key in ["kl_ab", "kl_ba", "sym_kl", "js", "tv", "argmax_disagree"]:
                    sums[key] += div[key]
                joint_sym_kl += div["sym_kl"]
                joint_js += div["js"]
                joint_tv_bound += div["tv"]
                any_disagree = max(any_disagree, div["argmax_disagree"])
                agent_steps += 1

                with torch.no_grad():
                    obs_t = _to_tensor(obs_list[i]).unsqueeze(0)
                    avail_t = _to_tensor(avail_list[i]).unsqueeze(0)
                    action, _, rnn_out = actors_interp[i](
                        obs_t, rnn_i[i], mask, available_actions=avail_t, deterministic=True
                    )
                    rnn_i[i] = rnn_out
                    actions.append(action.item())

            sums["joint_sym_kl"] += joint_sym_kl
            sums["joint_js_sum"] += joint_js
            sums["joint_tv_bound"] += min(1.0, joint_tv_bound)
            sums["joint_any_argmax_disagree"] += any_disagree
            steps += 1

            _, _, _, dones, infos, _ = env.step(actions)
            terminated = bool(np.all(dones))
            info = infos[0] if infos else {}
        wins += int(info.get("won", False))

    try:
        env.close()
    except Exception:
        pass

    out = {k: float(sums[k] / max(1, agent_steps)) for k in ["kl_ab", "kl_ba", "sym_kl", "js", "tv", "argmax_disagree"]}
    out.update(
        {
            "joint_sym_kl": float(sums["joint_sym_kl"] / max(1, steps)),
            "joint_js_sum": float(sums["joint_js_sum"] / max(1, steps)),
            "joint_tv_bound": float(sums["joint_tv_bound"] / max(1, steps)),
            "joint_any_argmax_disagree": float(sums["joint_any_argmax_disagree"] / max(1, steps)),
            "probe_win_rate": float(wins / max(1, episodes)),
            "episodes": episodes,
            "steps": steps,
            "agent_steps": agent_steps,
        }
    )
    return out


def existing_barrier_for_seed(existing: Dict, map_name: str, path_id: str, seed_index: int):
    try:
        path = existing[map_name]["interpolation"][path_id]
        return float(path["per_seed_barriers"][seed_index])
    except Exception:
        return None


def existing_wr_for_seed(existing: Dict, map_name: str, path_id: str, seed_index: int):
    try:
        path = existing[map_name]["interpolation"][path_id]
        return [float(x) for x in path["per_seed_wr"][seed_index]]
    except Exception:
        return None


def summarize_path(seed_rows: List[Dict]) -> Dict:
    alpha_rows = []
    all_alphas = sorted({r["alpha"] for s in seed_rows for r in s["alpha_metrics"]})
    for alpha in all_alphas:
        rows = [r for s in seed_rows for r in s["alpha_metrics"] if r["alpha"] == alpha]
        alpha_rows.append(
            {
                "alpha": alpha,
                "mean_js": float(np.mean([r["js"] for r in rows])),
                "mean_tv": float(np.mean([r["tv"] for r in rows])),
                "mean_sym_kl": float(np.mean([r["sym_kl"] for r in rows])),
                "mean_joint_sym_kl": float(np.mean([r["joint_sym_kl"] for r in rows])),
                "mean_action_disagreement": float(np.mean([r["argmax_disagree"] for r in rows])),
                "mean_probe_win_rate": float(np.mean([r["probe_win_rate"] for r in rows])),
            }
        )
    barriers = [s["existing_barrier"] for s in seed_rows if s["existing_barrier"] is not None]
    return {
        "n_seeds": len(seed_rows),
        "mean_existing_barrier": float(np.mean(barriers)) if barriers else None,
        "mean_weight_normalized_l2": float(np.mean([s["weight_metrics"]["mean_normalized_l2"] for s in seed_rows])),
        "mean_weight_cosine": float(np.mean([s["weight_metrics"]["mean_cosine"] for s in seed_rows])),
        "alpha_summary": alpha_rows,
    }


def run_map(map_name: str, seeds: List[int], path_ids: List[str], alphas: List[float], episodes: int) -> Dict:
    cc_file = os.path.join(cc.RESULTS_DIR, f"all_results_v3_{map_name}.json")
    existing = {}
    if os.path.exists(cc_file):
        with open(cc_file) as f:
            existing = json.load(f)

    result = {
        "map": map_name,
        "seeds": seeds,
        "alphas": alphas,
        "episodes_per_alpha": episodes,
        "paths": {},
    }

    for path_id in path_ids:
        if path_id not in ALL_PATHS:
            print(f"[{map_name}] skip unknown path {path_id}", flush=True)
            continue
        _, method_a, method_b, desc = ALL_PATHS[path_id]
        print(f"\n[{map_name}] {path_id}: {desc}", flush=True)
        seed_rows = []
        for seed_index, seed in enumerate(seeds):
            try:
                sds_a = cc.load_state_dicts(map_name, method_a, seed)
                sds_b = cc.load_state_dicts(map_name, method_b, seed)
            except FileNotFoundError as exc:
                print(f"  seed={seed} skip: {exc}", flush=True)
                continue
            w_metrics = weight_pair_metrics(sds_a, sds_b)
            alpha_rows = []
            print(
                f"  seed={seed} w_norm_l2={w_metrics['mean_normalized_l2']:.4f} "
                f"w_cos={w_metrics['mean_cosine']:.4f}",
                flush=True,
            )
            for alpha in alphas:
                metrics = eval_policy_metrics_for_alpha(map_name, sds_a, sds_b, alpha, episodes)
                metrics["alpha"] = alpha
                alpha_rows.append(metrics)
                print(
                    f"    a={alpha:.2f} js={metrics['js']:.4f} tv={metrics['tv']:.4f} "
                    f"jointKL={metrics['joint_sym_kl']:.3f} disagree={metrics['argmax_disagree']:.3f} "
                    f"wr={metrics['probe_win_rate']:.2f}",
                    flush=True,
                )
            seed_rows.append(
                {
                    "seed": seed,
                    "method_a": method_a,
                    "method_b": method_b,
                    "description": desc,
                    "existing_barrier": existing_barrier_for_seed(existing, map_name, path_id, seed_index),
                    "existing_wr": existing_wr_for_seed(existing, map_name, path_id, seed_index),
                    "weight_metrics": w_metrics,
                    "alpha_metrics": alpha_rows,
                }
            )
        result["paths"][path_id] = {
            "description": desc,
            "method_a": method_a,
            "method_b": method_b,
            "seeds": seed_rows,
            "summary": summarize_path(seed_rows) if seed_rows else {},
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=str, default="3s5z")
    parser.add_argument("--seeds", type=str, default="1,2,3,4")
    parser.add_argument("--paths", type=str, default="P1,P2,P3,P5,P6,P7")
    parser.add_argument("--alphas", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    maps = [x.strip() for x in args.maps.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    path_ids = [x.strip() for x in args.paths.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    all_results = {}
    for map_name in maps:
        all_results[map_name] = run_map(map_name, seeds, path_ids, alphas, args.episodes)

    if args.output:
        out_path = args.output
    else:
        map_tag = "_".join(maps)
        out_path = os.path.join(OUT_DIR, f"policy_metrics_{map_tag}_ep{args.episodes}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Q4 policy metrics to {out_path}", flush=True)


if __name__ == "__main__":
    main()
