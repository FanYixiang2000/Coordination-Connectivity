#!/usr/bin/env python3
"""All corridor paths: behavior divergence along interpolation paths.

Extends the P6/P4 mechanism check to the full corridor interpolation set:
P1/P2/P6 same-source and P3/P4/P5/P7 cross-history.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT_DIR = PROJECT_DIR / "exp" / "q4_rebuttal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, str(SCRIPT_DIR))

import coordination_connectivity_v3 as cc  # noqa: E402
from q4_corridor_pathwise_behavior import collect_history_bank, compute_on_bank, existing_wr  # noqa: E402


PATHS = {
    "P1": {
        "group": "same_source",
        "label": "P1 MAPPO-ct -> M2H",
        "method_a": "mappo_to_mappo_shared_stage2",
        "method_b": "mappo_to_happo_nonshared_stage2",
    },
    "P2": {
        "group": "same_source",
        "label": "P2 MAPPO-ct -> MAPPO-ns",
        "method_a": "mappo_to_mappo_shared_stage2",
        "method_b": "mappo_to_mappo_nonshared_stage2",
    },
    "P6": {
        "group": "same_source",
        "label": "P6 MAPPO-ns -> M2H",
        "method_a": "mappo_to_mappo_nonshared_stage2",
        "method_b": "mappo_to_happo_nonshared_stage2",
    },
    "P3": {
        "group": "cross_history",
        "label": "P3 MAPPO-Full -> HAPPO-Full",
        "method_a": "mappo_shared_full",
        "method_b": "happo_nonshared_full",
    },
    "P4": {
        "group": "cross_history",
        "label": "P4 M2H -> HAPPO-Full",
        "method_a": "mappo_to_happo_nonshared_stage2",
        "method_b": "happo_nonshared_full",
    },
    "P5": {
        "group": "cross_history",
        "label": "P5 MAPPO-ct -> HAPPO-Full",
        "method_a": "mappo_to_mappo_shared_stage2",
        "method_b": "happo_nonshared_full",
    },
    "P7": {
        "group": "cross_history",
        "label": "P7 MAPPO-ns -> HAPPO-Full",
        "method_a": "mappo_to_mappo_nonshared_stage2",
        "method_b": "happo_nonshared_full",
    },
}

ALPHAS = cc.ALPHAS
SEEDS = [1, 2, 3, 4]
MAP_NAME = "corridor"
PRE_METHOD = "mappo_to_mappo_shared_stage1"


def summarize_path(seed_rows, path_id):
    wr = existing_wr(MAP_NAME, path_id)
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
                "to_pre_ind_tv_mean": float(np.mean([r["to_pre"]["ind_tv"] for r in rows])),
                "to_A_joint_kl_mean": float(np.mean([r["to_A"]["joint_sym_kl"] for r in rows])),
                "to_B_joint_kl_mean": float(np.mean([r["to_B"]["joint_sym_kl"] for r in rows])),
                "to_pre_joint_kl_mean": float(np.mean([r["to_pre"]["joint_sym_kl"] for r in rows])),
                "avg_ind_tv_to_endpoints_mean": float(np.mean([r["avg_ind_tv_to_endpoints"] for r in rows])),
                "avg_joint_kl_to_endpoints_mean": float(np.mean([r["avg_joint_kl_to_endpoints"] for r in rows])),
                "nearest_joint_tv_bound_mean": float(np.mean([r["nearest_joint_tv_bound"] for r in rows])),
            }
        )
    return summary


def dump(results):
    with open(OUT_DIR / "q4_corridor_all_paths_behavior.json", "w") as f:
        json.dump(results, f, indent=2)


def run():
    results = {"map": MAP_NAME, "seeds": SEEDS, "alphas": ALPHAS, "paths": {}}
    for path_id, spec in PATHS.items():
        print(f"\n=== {path_id}: {spec['label']} ===", flush=True)
        seed_rows = []
        for seed in SEEDS:
            print(f" seed={seed}", flush=True)
            sds_a = cc.load_state_dicts(MAP_NAME, spec["method_a"], seed)
            sds_b = cc.load_state_dicts(MAP_NAME, spec["method_b"], seed)
            sds_pre = cc.load_state_dicts(MAP_NAME, PRE_METHOD, seed)
            bank = collect_history_bank(MAP_NAME, sds_a, sds_b, episodes_per_endpoint=2)
            alpha_rows = []
            for alpha in ALPHAS:
                sds_alpha = cc.interpolate_sds(sds_a, sds_b, alpha)
                to_a = compute_on_bank(MAP_NAME, sds_a, sds_alpha, bank)
                to_b = compute_on_bank(MAP_NAME, sds_b, sds_alpha, bank)
                to_pre = compute_on_bank(MAP_NAME, sds_pre, sds_alpha, bank)
                row = {
                    "alpha": alpha,
                    "to_A": to_a,
                    "to_B": to_b,
                    "to_pre": to_pre,
                    "avg_ind_tv_to_endpoints": 0.5 * (to_a["ind_tv"] + to_b["ind_tv"]),
                    "avg_joint_kl_to_endpoints": 0.5 * (to_a["joint_sym_kl"] + to_b["joint_sym_kl"]),
                    "nearest_joint_tv_bound": min(to_a["joint_tv_bound"], to_b["joint_tv_bound"]),
                }
                alpha_rows.append(row)
                print(
                    f"  a={alpha:.1f} avgTV={row['avg_ind_tv_to_endpoints']:.3f} "
                    f"avgJointKL={row['avg_joint_kl_to_endpoints']:.2f} preTV={to_pre['ind_tv']:.3f}",
                    flush=True,
                )
            seed_rows.append({"seed": seed, "bank_size": len(bank), "alpha_metrics": alpha_rows})
            dump(results)
        results["paths"][path_id] = {
            **spec,
            "seeds": seed_rows,
            "summary": summarize_path(seed_rows, path_id),
        }
        dump(results)
    plot_all(results)
    plot_pre(results)
    dump(results)


def plot_all(results):
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    xs = np.array(results["alphas"])
    groups = [
        ("same_source", "Same-source paths", ["P1", "P2", "P6"], ["#4C72B0", "#55A868", "#8172B3"]),
        ("cross_history", "Cross-history paths", ["P3", "P4", "P5", "P7"], ["#C44E52", "#DD8452", "#937860", "#DA8BC3"]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(9.1, 6.6), dpi=230, sharex=True)
    for col, (_, title, pids, colors) in enumerate(groups):
        axes[0, col].set_title(title)
        for pid, color in zip(pids, colors):
            path = results["paths"][pid]
            summ = path["summary"]
            wr = np.array([r["win_rate_mean"] for r in summ])
            wr_std = np.array([r["win_rate_std"] for r in summ])
            avg_tv = np.array([r["avg_ind_tv_to_endpoints_mean"] for r in summ])
            avg_kl = np.array([r["avg_joint_kl_to_endpoints_mean"] for r in summ])
            label = pid
            axes[0, col].plot(xs, wr, color=color, marker="o", linewidth=2.0, markersize=3.5, label=label)
            axes[0, col].fill_between(xs, np.clip(wr - wr_std, 0, 1), np.clip(wr + wr_std, 0, 1), color=color, alpha=0.12, linewidth=0)
            axes[1, col].plot(xs, avg_tv, color=color, marker="o", linewidth=2.0, markersize=3.5, label=label)
            axes[2, col].plot(xs, avg_kl, color=color, marker="o", linewidth=2.0, markersize=3.5, label=label)
        axes[0, col].set_ylim(-0.05, 1.05)
        axes[0, col].set_ylabel("Win rate")
        axes[1, col].set_ylabel("Individual TV")
        axes[2, col].set_ylabel("Joint sym-KL")
        axes[2, col].set_xlabel("Interpolation coefficient alpha")
        axes[0, col].legend(frameon=False, loc="lower center", ncol=2)
        for row in range(3):
            axes[row, col].grid(True, alpha=0.25)
            axes[row, col].set_xlim(0, 1)
            axes[row, col].set_xticks(xs)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q4_corridor_all_paths_behavior_clean.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "q4_corridor_all_paths_behavior_clean.pdf", bbox_inches="tight")


def plot_pre(results):
    xs = np.array(results["alphas"])
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.2), dpi=230, sharex=True)
    colors = {"P1": "#4C72B0", "P2": "#55A868", "P6": "#8172B3"}
    for pid in ["P1", "P2", "P6"]:
        summ = results["paths"][pid]["summary"]
        axes[0].plot(xs, [r["to_pre_ind_tv_mean"] for r in summ], marker="o", linewidth=2.0, color=colors[pid], label=pid)
        axes[1].plot(xs, [r["to_pre_joint_kl_mean"] for r in summ], marker="o", linewidth=2.0, color=colors[pid], label=pid)
    axes[0].set_ylabel("Individual TV to MAPPO-Pre")
    axes[1].set_ylabel("Joint sym-KL to MAPPO-Pre")
    axes[1].set_xlabel("Interpolation coefficient alpha")
    axes[0].legend(frameon=False, loc="upper center", ncol=3)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q4_corridor_same_source_pre_anchor_clean.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "q4_corridor_same_source_pre_anchor_clean.pdf", bbox_inches="tight")


if __name__ == "__main__":
    run()
