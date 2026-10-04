#!/usr/bin/env python3
"""Pathwise behavior divergence for Q4 rebuttal on SMAC corridor.

This directly probes the Appendix-B link between policy behavior and return:
for P6 (same-source) and P4 (cross-history), it builds a shared local-history
bank from both endpoints, then computes per-alpha individual and joint policy
divergence on exactly the same histories.
"""

import copy
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT_DIR = PROJECT_DIR / "exp" / "q4_rebuttal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, str(SCRIPT_DIR))

import coordination_connectivity_v3 as cc  # noqa: E402
from cc_v3_policy_metrics import actor_probs_and_state, divergence, init_rnn_states  # noqa: E402


PATHS = {
    "P6": {
        "label": "P6: MAPPO-ns -> M2H (same source)",
        "short": "P6 same-source",
        "method_a": "mappo_to_mappo_nonshared_stage2",
        "method_b": "mappo_to_happo_nonshared_stage2",
        "color": "#4C72B0",
    },
    "P4": {
        "label": "P4: M2H -> HAPPO-Full (cross-history)",
        "short": "P4 cross-history",
        "method_a": "mappo_to_happo_nonshared_stage2",
        "method_b": "happo_nonshared_full",
        "color": "#C44E52",
    },
}
ALPHAS = cc.ALPHAS
SEEDS = [1, 2, 3, 4]


def _to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def collect_history_bank(map_name, sds_a, sds_b, episodes_per_endpoint=3, max_steps_per_ep=220):
    """Collect shared local histories from both endpoint rollouts."""
    n_agents = cc.MAP_AGENTS[map_name]
    bank = []
    for source_name, sds in [("A", sds_a), ("B", sds_b)]:
        actors = cc.create_actors(map_name, sds)
        env = cc.make_env(map_name)
        for _ in range(episodes_per_endpoint):
            rnn_states = init_rnn_states(map_name)
            mask = torch.ones(1, 1)
            env.reset()
            terminated = False
            steps = 0
            while not terminated and steps < max_steps_per_ep:
                obs_list = [copy.deepcopy(env.get_obs_agent(i)) for i in range(n_agents)]
                avail_list = [copy.deepcopy(env.get_avail_agent_actions(i)) for i in range(n_agents)]
                bank.append({"source": source_name, "obs": obs_list, "avail": avail_list})
                actions = []
                for i in range(n_agents):
                    obs_t = _to_tensor(obs_list[i]).unsqueeze(0)
                    avail_t = _to_tensor(avail_list[i]).unsqueeze(0)
                    with torch.no_grad():
                        action, _, rnn_out = actors[i](
                            obs_t,
                            rnn_states[i],
                            mask,
                            available_actions=avail_t,
                            deterministic=True,
                        )
                    rnn_states[i] = rnn_out
                    actions.append(action.item())
                _, _, _, dones, _, _ = env.step(actions)
                terminated = bool(np.all(dones))
                steps += 1
        try:
            env.close()
        except Exception:
            pass
    return bank


def compute_on_bank(map_name, sds_ref, sds_alpha, bank):
    n_agents = cc.MAP_AGENTS[map_name]
    actors_ref = cc.create_actors(map_name, sds_ref)
    actors_alpha = cc.create_actors(map_name, sds_alpha)
    mask = torch.ones(1, 1)
    sums = {
        "ind_js": 0.0,
        "ind_tv": 0.0,
        "ind_sym_kl": 0.0,
        "ind_argmax_disagree": 0.0,
        "joint_sym_kl": 0.0,
        "joint_js_sum": 0.0,
        "joint_tv_bound": 0.0,
        "joint_any_argmax_disagree": 0.0,
    }
    agent_count = 0
    joint_count = 0
    # corridor is feed-forward in our setup. We still pass zero states so the
    # same helper works with HARL's policy wrapper.
    for item in bank:
        rnn_ref = init_rnn_states(map_name)
        rnn_alpha = init_rnn_states(map_name)
        joint_sym_kl = 0.0
        joint_js = 0.0
        joint_tv = 0.0
        any_disagree = 0.0
        for i in range(n_agents):
            p_ref, _ = actor_probs_and_state(
                actors_ref[i], item["obs"][i], rnn_ref[i], mask, item["avail"][i]
            )
            p_alpha, _ = actor_probs_and_state(
                actors_alpha[i], item["obs"][i], rnn_alpha[i], mask, item["avail"][i]
            )
            div = divergence(p_alpha, p_ref)
            sums["ind_js"] += div["js"]
            sums["ind_tv"] += div["tv"]
            sums["ind_sym_kl"] += div["sym_kl"]
            sums["ind_argmax_disagree"] += div["argmax_disagree"]
            joint_sym_kl += div["sym_kl"]
            joint_js += div["js"]
            joint_tv += div["tv"]
            any_disagree = max(any_disagree, div["argmax_disagree"])
            agent_count += 1
        sums["joint_sym_kl"] += joint_sym_kl
        sums["joint_js_sum"] += joint_js
        sums["joint_tv_bound"] += min(1.0, joint_tv)
        sums["joint_any_argmax_disagree"] += any_disagree
        joint_count += 1
    return {
        "ind_js": float(sums["ind_js"] / max(1, agent_count)),
        "ind_tv": float(sums["ind_tv"] / max(1, agent_count)),
        "ind_sym_kl": float(sums["ind_sym_kl"] / max(1, agent_count)),
        "ind_argmax_disagree": float(sums["ind_argmax_disagree"] / max(1, agent_count)),
        "joint_sym_kl": float(sums["joint_sym_kl"] / max(1, joint_count)),
        "joint_js_sum": float(sums["joint_js_sum"] / max(1, joint_count)),
        "joint_tv_bound": float(sums["joint_tv_bound"] / max(1, joint_count)),
        "joint_any_argmax_disagree": float(sums["joint_any_argmax_disagree"] / max(1, joint_count)),
    }


def existing_wr(map_name, path_id):
    path = json.load(open(PROJECT_DIR / "exp" / "cc_v3" / f"all_results_v3_{map_name}.json"))
    arr = np.array(path[map_name]["interpolation"][path_id]["per_seed_wr"], dtype=float)
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }


def run():
    map_name = "corridor"
    results = {"map": map_name, "seeds": SEEDS, "alphas": ALPHAS, "paths": {}}
    for path_id, spec in PATHS.items():
        print(f"\n=== {path_id}: {spec['label']} ===", flush=True)
        seed_rows = []
        for seed in SEEDS:
            print(f" seed={seed}", flush=True)
            sds_a = cc.load_state_dicts(map_name, spec["method_a"], seed)
            sds_b = cc.load_state_dicts(map_name, spec["method_b"], seed)
            bank = collect_history_bank(map_name, sds_a, sds_b, episodes_per_endpoint=3)
            alpha_rows = []
            for alpha in ALPHAS:
                sds_alpha = cc.interpolate_sds(sds_a, sds_b, alpha)
                to_a = compute_on_bank(map_name, sds_a, sds_alpha, bank)
                to_b = compute_on_bank(map_name, sds_b, sds_alpha, bank)
                row = {
                    "alpha": alpha,
                    "to_A": to_a,
                    "to_B": to_b,
                    "nearest_ind_tv": min(to_a["ind_tv"], to_b["ind_tv"]),
                    "nearest_joint_tv_bound": min(to_a["joint_tv_bound"], to_b["joint_tv_bound"]),
                    "avg_ind_tv_to_endpoints": 0.5 * (to_a["ind_tv"] + to_b["ind_tv"]),
                    "avg_joint_kl_to_endpoints": 0.5 * (to_a["joint_sym_kl"] + to_b["joint_sym_kl"]),
                }
                alpha_rows.append(row)
                print(
                    f"  a={alpha:.1f} TV_A={to_a['ind_tv']:.3f} TV_B={to_b['ind_tv']:.3f} "
                    f"jointKL_A={to_a['joint_sym_kl']:.3f} jointKL_B={to_b['joint_sym_kl']:.3f}",
                    flush=True,
                )
            seed_rows.append({"seed": seed, "bank_size": len(bank), "alpha_metrics": alpha_rows})
            dump(results)
        wr = existing_wr(map_name, path_id)
        summary = []
        for idx, alpha in enumerate(ALPHAS):
            rows = [s["alpha_metrics"][idx] for s in seed_rows]
            summary.append(
                {
                    "alpha": alpha,
                    "win_rate_mean": wr["mean"][idx],
                    "win_rate_std": wr["std"][idx],
                    "to_A_ind_tv_mean": float(np.mean([r["to_A"]["ind_tv"] for r in rows])),
                    "to_B_ind_tv_mean": float(np.mean([r["to_B"]["ind_tv"] for r in rows])),
                    "to_A_joint_kl_mean": float(np.mean([r["to_A"]["joint_sym_kl"] for r in rows])),
                    "to_B_joint_kl_mean": float(np.mean([r["to_B"]["joint_sym_kl"] for r in rows])),
                    "nearest_joint_tv_bound_mean": float(np.mean([r["nearest_joint_tv_bound"] for r in rows])),
                    "avg_ind_tv_to_endpoints_mean": float(np.mean([r["avg_ind_tv_to_endpoints"] for r in rows])),
                    "avg_joint_kl_to_endpoints_mean": float(np.mean([r["avg_joint_kl_to_endpoints"] for r in rows])),
                }
            )
        results["paths"][path_id] = {
            **spec,
            "win_rate": wr,
            "seeds": seed_rows,
            "summary": summary,
        }
        dump(results)
    plot(results)
    dump(results)


def dump(results):
    out = OUT_DIR / "q4_corridor_pathwise_behavior_p6_p4.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


def plot(results):
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 7.2), dpi=220, sharex=True)
    for col, path_id in enumerate(["P6", "P4"]):
        path = results["paths"][path_id]
        spec = PATHS[path_id]
        color = spec["color"]
        xs = np.array(results["alphas"])
        summ = path["summary"]
        wr = np.array([r["win_rate_mean"] for r in summ])
        wr_std = np.array([r["win_rate_std"] for r in summ])
        tv_a = np.array([r["to_A_ind_tv_mean"] for r in summ])
        tv_b = np.array([r["to_B_ind_tv_mean"] for r in summ])
        kl_a = np.array([r["to_A_joint_kl_mean"] for r in summ])
        kl_b = np.array([r["to_B_joint_kl_mean"] for r in summ])

        axes[0, col].plot(xs, wr, marker="o", color=color, linewidth=2.4)
        axes[0, col].fill_between(xs, np.clip(wr - wr_std, 0, 1), np.clip(wr + wr_std, 0, 1), color=color, alpha=0.18)
        axes[0, col].set_ylim(-0.05, 1.05)
        axes[0, col].set_title(spec["short"])
        axes[0, col].set_ylabel("Win rate")

        axes[1, col].plot(xs, tv_a, marker="o", color="#4C72B0", linewidth=2.0, label="to endpoint A")
        axes[1, col].plot(xs, tv_b, marker="s", color="#C44E52", linewidth=2.0, label="to endpoint B")
        axes[1, col].set_ylabel("Individual TV")
        axes[1, col].set_ylim(-0.02, max(0.55, 1.1 * max(tv_a.max(), tv_b.max())))

        axes[2, col].plot(xs, kl_a, marker="o", color="#4C72B0", linewidth=2.0, label="to endpoint A")
        axes[2, col].plot(xs, kl_b, marker="s", color="#C44E52", linewidth=2.0, label="to endpoint B")
        axes[2, col].set_ylabel("Joint sym-KL")
        axes[2, col].set_xlabel("Interpolation coefficient alpha")

        for row in range(3):
            axes[row, col].grid(True, alpha=0.25)
            axes[row, col].set_xticks(xs)
    axes[1, 1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.suptitle("Q4: Pathwise Behavior Divergence on Shared Corridor Histories", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "q4_corridor_pathwise_behavior_p6_p4.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "q4_corridor_pathwise_behavior_p6_p4.pdf", bbox_inches="tight")
    print(OUT_DIR / "q4_corridor_pathwise_behavior_p6_p4.png", flush=True)


if __name__ == "__main__":
    run()
