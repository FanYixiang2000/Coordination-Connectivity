#!/usr/bin/env python3
"""Generate Figure `fig:vdn-pilot` (appendix VDN pilot on MPE Reference) from
the raw evaluation JSON in exp/mpe_vd_cc (read-only). Replaces the run-time
diagnostic plot with a paper-style figure: serif fonts, proper alpha symbol,
no scenario id suffixes or barrier values baked into the legend."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC = (PROJECT_DIR / "exp" / "mpe_vd_cc" /
       "mpe_vd_cc_simple_reference_v2_vdn_seed1_src2000_br1000.json")
OUT_DIR = PROJECT_DIR / "paper" / "11_6.22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

# blue = same-source branches, orange = branch vs scratch (as stated in the caption)
STYLE = {
    "VDN same-source A <-> B": ("Branch A $\\leftrightarrow$ branch B (same source)",
                                "#1f77b4", "-"),
    "VDN branch-A <-> scratch": ("Branch A $\\leftrightarrow$ independent scratch",
                                 "#ff7f0e", "--"),
}

data = json.load(open(SRC))
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=230)
for path in data["paths"]:
    label, color, ls = STYLE[path["name"]]
    xs = np.array([pt["alpha"] for pt in path["points"]])
    ys = np.array([pt["mean_return"] for pt in path["points"]])
    ax.plot(xs, ys, color=color, linestyle=ls, marker="o",
            linewidth=2.0, markersize=4.5, label=label)
ax.set_xlabel("Interpolation coefficient $\\alpha$")
ax.set_ylabel("Episode return")
ax.set_title("VDN pilot: MPE Reference ($N=2$)")
ax.set_xlim(-0.02, 1.02)
ax.grid(True, alpha=0.25)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
          frameon=False, fontsize=9, handlelength=1.8, columnspacing=1.2)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig10_vdn_reference.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig10_vdn_reference.png", bbox_inches="tight")
print("saved", OUT_DIR / "fig10_vdn_reference.pdf")
