#!/usr/bin/env python3
"""Generate Figure `fig:behavior-divergence` from exp/q4_rebuttal JSONs (read-only)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "exp/q4_rebuttal/q4_corridor_all_paths_behavior.json"
CCV3 = BASE / "exp/cc_v3/all_results_v3_corridor.json"  # win-rate rows reuse the main interpolation evaluations
OUT = BASE / "paper/11_6.22/fig8_behavior_divergence.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)  # allow running from a standalone checkout

results = json.load(open(SRC))
cc = json.load(open(CCV3))

LABELS = {  # paper-consistent method names
    "P1": "P1 MAPPO-Shared $\\rightarrow$ M2H",
    "P2": "P2 MAPPO-Shared $\\rightarrow$ MAPPO-Sep",
    "P6": "P6 MAPPO-Sep $\\rightarrow$ M2H",
    "P3": "P3 MAPPO-Full $\\rightarrow$ HAPPO-Full",
    "P4": "P4 M2H $\\rightarrow$ HAPPO-Full",
    "P5": "P5 MAPPO-Shared $\\rightarrow$ HAPPO-Full",
    "P7": "P7 MAPPO-Sep $\\rightarrow$ HAPPO-Full",
}


def wr_for(pid):
    arr = np.array(cc["corridor"]["interpolation"][pid]["per_seed_wr"], dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


def summarize(pid):
    seed_rows = results["paths"][pid]["seeds"]
    n_alpha = len(results["alphas"])
    tv, kl = [], []
    for idx in range(n_alpha):
        rows = [s["alpha_metrics"][idx] for s in seed_rows]
        tv.append(np.mean([0.5 * (r["to_A"]["ind_tv"] + r["to_B"]["ind_tv"]) for r in rows]))  # mean distance to the two endpoints
        kl.append(np.mean([0.5 * (r["to_A"]["joint_sym_kl"] + r["to_B"]["joint_sym_kl"]) for r in rows]))
    return np.array(tv), np.array(kl)


plt.rcParams.update({
    "font.size": 13,  # sized for display at 0.62\linewidth in the paper
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,  # embed TrueType fonts for camera-ready
    "ps.fonttype": 42,
})
xs = np.array(results["alphas"])
groups = [
    ("Same-source paths", ["P1", "P2", "P6"], ["#4C72B0", "#55A868", "#8172B3"]),
    ("Cross-history paths", ["P3", "P4", "P5", "P7"], ["#C44E52", "#DD8452", "#937860", "#DA8BC3"]),
]
fig, axes = plt.subplots(3, 2, figsize=(9.1, 5.0), dpi=230, sharex=True)
for col, (title, pids, colors) in enumerate(groups):
    axes[0, col].set_title(title)
    for pid, color in zip(pids, colors):
        wr_m, wr_s = wr_for(pid)
        tv, kl = summarize(pid)
        lab = LABELS[pid]
        axes[0, col].plot(xs, wr_m, color=color, marker="o", linewidth=2.0, markersize=3.5, label=lab)
        axes[0, col].fill_between(xs, np.clip(wr_m - wr_s, 0, 1), np.clip(wr_m + wr_s, 0, 1),
                                  color=color, alpha=0.12, linewidth=0)  # +/- 1 cross-seed std
        axes[1, col].plot(xs, tv, color=color, marker="o", linewidth=2.0, markersize=3.5, label=lab)
        axes[2, col].plot(xs, kl, color=color, marker="o", linewidth=2.0, markersize=3.5, label=lab)
    axes[0, col].set_ylim(-0.05, 1.05)
    axes[0, col].set_ylabel("Win rate")
    axes[1, col].set_ylabel("Individual TV")
    axes[2, col].set_ylabel("Joint sym-KL")
    axes[2, col].set_xlabel("Interpolation coefficient $\\alpha$")
    for row in range(3):
        axes[row, col].grid(True, alpha=0.25)
        axes[row, col].set_xlim(0, 1)
        axes[row, col].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # sparser ticks for readability at small display size
handles = axes[2, 0].get_legend_handles_labels()[0] + axes[2, 1].get_legend_handles_labels()[0]
labels = axes[2, 0].get_legend_handles_labels()[1] + axes[2, 1].get_legend_handles_labels()[1]
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=4,
           frameon=False, fontsize=11.5, handlelength=1.6,
           columnspacing=1.2)  # one shared legend below the whole figure, outside all panels
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(str(OUT).replace(".pdf", ".png"), bbox_inches="tight")
print("saved", OUT)
