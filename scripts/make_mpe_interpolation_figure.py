#!/usr/bin/env python3
"""Generate Figure `fig:mpe` from exp/mpe_cc evaluation JSONs (read-only)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "exp" / "mpe_cc"
OUT_DIR = PROJECT_DIR / "paper" / "11_6.22"
OUT_DIR.mkdir(parents=True, exist_ok=True)  # allow running from a standalone checkout

SCENARIOS = {
    "simple_reference_v2": "fig9_mpe_reference",
    "simple_spread_v2": "fig9_mpe_spread",
}

STYLE = {  # path id -> (paper-consistent label, color, linestyle)
    "P1": ("MAPPO-Sep $\\rightarrow$ M2H (same source)", "#2E7D32", "-"),
    "P3": ("MAPPO-Full $\\rightarrow$ HAPPO-Full", "#B22222", "-"),
    "P4": ("M2H $\\rightarrow$ HAPPO-Full", "#E07B67", "--"),
    "P7": ("MAPPO-Sep $\\rightarrow$ HAPPO-Full", "#8B5A2B", "-."),
}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,  # embed TrueType fonts for camera-ready
    "ps.fonttype": 42,
})

for scenario, out_name in SCENARIOS.items():
    data = json.load(open(SRC_DIR / f"mpe_cc_{scenario}_expanded.json"))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=230)
    for pid, (label, color, ls) in STYLE.items():
        path = data[pid]
        xs = np.array(path["alphas"])
        per_seed = np.array(path["per_seed_reward"], dtype=float)  # shape (4 seeds, 11 alphas)
        mean = per_seed.mean(axis=0)
        std = per_seed.std(axis=0)
        ax.plot(xs, mean, color=color, linestyle=ls, marker="o",
                linewidth=2.0, markersize=4, label=label)
        ax.fill_between(xs, mean - std, mean + std, color=color,
                        alpha=0.15, linewidth=0)  # +/- 1 cross-seed std, as stated in the caption
    ax.set_xlabel("Interpolation coefficient $\\alpha$")
    ax.set_ylabel("Mean return (higher is better)")
    ax.set_title(f"MPE {scenario}: coordination connectivity")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              frameon=False, fontsize=9, handlelength=1.8,
              columnspacing=1.2)  # legend below the axes, outside the plotting area
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_name}.png", bbox_inches="tight")
    print("saved", OUT_DIR / f"{out_name}.pdf")
