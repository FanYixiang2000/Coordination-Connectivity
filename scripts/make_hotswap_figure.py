#!/usr/bin/env python3
"""Redraw the single-agent hot-swap bar chart (fig5_hotswap.pdf) without
the in-bar drop annotations. Data, bar layout, colors, and fonts are
identical to the submitted figure; only the "<down-arrow>XX%" text labels
inside the "After swap" bars are removed."""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "pdf.fonttype": 42,  # NeurIPS requires Type 42 (TrueType) embedding
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "font.weight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_FILE = PROJECT_DIR / "exp" / "cc_v3" / "all_results_merged.json"
OUTPUT_DIR = PROJECT_DIR / "paper" / "11_6.22"

MAPS = ["3s5z", "corridor", "MMM2"]

NAME_MAP = {
    "MAPPO-ct": "MAPPO-Shared",
    "MAPPO-ns": "MAPPO-Sep",
    "M2H": "M2H",
    "HAPPO": "HAPPO-Full",
    "MAPPO": "MAPPO-Full",
    "MAPPO-pre": "MAPPO-Pre",
}


def dn(name):
    return NAME_MAP.get(name, name)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # allow running from a standalone checkout
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    available = [m for m in MAPS if m in data and "hotswap" in data[m]]
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 4.2), squeeze=False)
    for idx, mn in enumerate(available):
        ax = axes[0, idx]
        hs = data[mn]["hotswap"]
        hs_ids = sorted(hs.keys())

        labels = [f"{dn(hs[h]['team'])} \u2190 {dn(hs[h]['donor'])}" for h in hs_ids]
        base = [hs[h]["mean_base_wr"] for h in hs_ids]
        swap = [hs[h]["mean_swap_wr_overall"] for h in hs_ids]
        swap_std = [hs[h]["std_swap_wr_overall"] for h in hs_ids]

        x = np.arange(len(hs_ids))
        w = 0.28
        for i in range(len(hs_ids)):
            is_cross = ("HAPPO" in hs[hs_ids[i]]["team"]
                        or "HAPPO" in hs[hs_ids[i]]["donor"])
            c_base = "#b2182b" if is_cross else "#2166ac"
            c_swap = "#d6604d" if is_cross else "#67a9cf"
            ax.bar(x[i] - w / 2, base[i], w, color=c_base, alpha=0.4,
                   edgecolor=c_base, linewidth=1.2)
            ax.bar(x[i] + w / 2, swap[i], w, color=c_swap,
                   edgecolor="black", linewidth=0.8,
                   yerr=swap_std[i], capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
        if idx == 0:
            ax.set_ylabel("Win rate", fontweight="bold")
        ax.set_title(mn, fontweight="bold")
        ax.set_ylim(0, 1.15)
        # Neutral gray patches (light = Original, solid = After swap), same style as the specialization figure
        ax.legend([Patch(facecolor="gray", alpha=0.3, edgecolor="gray", lw=1),
                   Patch(facecolor="gray", edgecolor="gray", lw=1)],
                  ["Original", "After swap"],
                  fontsize=9, framealpha=0.9, loc="lower left")
        ax.grid(True, axis="y", alpha=0.1, lw=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig5_hotswap.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
