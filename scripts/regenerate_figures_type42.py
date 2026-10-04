#!/usr/bin/env python3
"""
Regenerate ALL paper figures with Type 42 (TrueType) font embedding.

NeurIPS requires that all fonts in PDF figures be Type 1 or TrueType (Type 42).
The original figures used DejaVu Serif/Sans which were embedded as Type 3.

This script re-runs all figure generation with proper rcParams to force
Type 42 embedding, keeping the visual content identical.
"""
import json, csv, os, sys, argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Force Type 42 (TrueType) font embedding — must be set BEFORE any figure creation
plt.rcParams.update({
    "pdf.fonttype": 42,
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
PROJECT_DIR = str(SCRIPT_DIR.parent)
RESULTS_FILE = os.path.join(PROJECT_DIR, "exp", "cc_v3", "all_results_merged.json")
TABLES_DIR = os.path.join(PROJECT_DIR, "exp", "smac_robustness", "plot_ready_tables")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "13")

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

STYLES = {
    "P1": {"color": "#2166ac", "ls": "-",  "marker": "o", "label": "MAPPO-Shared \u2192 M2H"},
    "P2": {"color": "#67a9cf", "ls": "--", "marker": "s", "label": "MAPPO-Shared \u2192 MAPPO-Sep"},
    "P6": {"color": "#1b7837", "ls": ":",  "marker": "p", "label": "MAPPO-Sep \u2192 M2H"},
    "P3": {"color": "#b2182b", "ls": "-",  "marker": "^", "label": "MAPPO-Full \u2192 HAPPO-Full"},
    "P4": {"color": "#d6604d", "ls": "--", "marker": "D", "label": "M2H \u2192 HAPPO-Full"},
    "P5": {"color": "#f4a582", "ls": "-.", "marker": "v", "label": "MAPPO-Shared \u2192 HAPPO-Full"},
    "P7": {"color": "#8c510a", "ls": "-.", "marker": "h", "label": "MAPPO-Sep \u2192 HAPPO-Full"},
}

METHOD_ORDER = ["MAPPO-Shared", "MAPPO-Sep", "M2H", "MAPPO-Full", "HAPPO-Full"]
METHOD_ORDER_INT = ["MAPPO-ct", "MAPPO-ns", "M2H", "MAPPO", "HAPPO"]
METHOD_ORDER_SPEC = ["MAPPO-Shared", "MAPPO-Sep", "M2H", "HAPPO-Full"]
METHOD_ORDER_SPEC_INT = ["MAPPO-ct", "MAPPO-ns", "M2H", "HAPPO"]

MC = {
    "MAPPO-Shared": "#2166ac", "MAPPO-Sep": "#67a9cf",
    "M2H": "#1b7837", "HAPPO-Full": "#b2182b", "MAPPO-Full": "#762a83",
    "MAPPO-ct": "#2166ac", "MAPPO-ns": "#67a9cf",
    "HAPPO": "#b2182b", "MAPPO": "#762a83",
}

# Main-results plotting config
MAIN_METHODS = [
    "mappo_shared_full",
    "happo_nonshared_full",
    "mappo_to_mappo_shared",
    "mappo_to_mappo_nonshared",
    "mappo_to_happo_nonshared",
]

LABEL_MAP = {
    "mappo_shared_full": "MAPPO-Full",
    "happo_nonshared_full": "HAPPO-Full",
    "mappo_to_mappo_shared": "MAPPO-Shared",
    "mappo_to_mappo_nonshared": "MAPPO-Sep",
    "mappo_to_happo_nonshared": "M2H",
}

COLORS = {
    "mappo_shared_full": "#762a83",
    "happo_nonshared_full": "#b2182b",
    "mappo_to_mappo_shared": "#2166ac",
    "mappo_to_mappo_nonshared": "#67a9cf",
    "mappo_to_happo_nonshared": "#1b7837",
}

LINESTYLES = {
    "mappo_shared_full": "-",
    "happo_nonshared_full": "-",
    "mappo_to_mappo_shared": "--",
    "mappo_to_mappo_nonshared": "-.",
    "mappo_to_happo_nonshared": "-.",
}

PARAM_COMPARE_METHODS = [
    "mappo_to_mappo_shared",
    "mappo_to_mappo_nonshared",
    "mappo_to_happo_nonshared",
]


def load_csv(path):
    if not os.path.exists(path):
        print(f"  [WARN] Not found: {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_fig(fig, path_stem):
    for ext in ["pdf", "png"]:
        p = f"{path_stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  Saved: {path_stem}.{{pdf,png}}")


def interpolate_curves(steps_list, vals_list, n_points=200):
    if not steps_list:
        return None, None, None
    max_s = min(s[-1] for s in steps_list)
    min_s = max(s[0] for s in steps_list)
    if max_s <= min_s:
        return None, None, None
    common = np.linspace(min_s, max_s, n_points)
    interp = [np.interp(common, s, v) for s, v in zip(steps_list, vals_list)]
    arr = np.array(interp)
    return common, arr.mean(axis=0), arr.std(axis=0)


# FIG 1: TRAINING CURVES (combined)

def fig1_training_curves(data):
    available = [m for m in MAPS if m in data and "training" in data[m]]
    if not available:
        print("  No training data for fig1"); return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.3), squeeze=False)
    for idx, mn in enumerate(available):
        ax = axes[0, idx]
        tr = data[mn]["training"]
        for method_int in METHOD_ORDER_INT:
            if method_int not in tr:
                continue
            r = tr[method_int]
            disp = dn(method_int)
            c = MC.get(method_int, "gray")
            steps = np.array(r["steps"])
            mean = np.array(r["mean_wr"])
            std = np.array(r["std_wr"])
            ax.plot(steps / 1e6, mean, color=c, linewidth=2.0, label=disp)
            ax.fill_between(steps / 1e6, np.clip(mean - std, 0, 1),
                            np.clip(mean + std, 0, 1), alpha=0.12, color=c)
        ax.set_xlabel("Steps (M)", fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Win rate", fontweight="bold")
        ax.set_title(mn, fontweight="bold")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.15, lw=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(handles), 5), bbox_to_anchor=(0.5, -0.18),
               frameon=True, edgecolor="#cccccc", fontsize=8.5,
               handlelength=2.5, columnspacing=1.0)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    path = os.path.join(OUTPUT_DIR, "fig1_training_curves.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# FIG 2: INTERPOLATION

def fig2_interpolation(data):
    available = [m for m in MAPS if m in data and "interpolation" in data[m]]
    n = len(available)
    if n == 0:
        print("  No interpolation data"); return

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.3), squeeze=False)
    path_order = ["P1", "P2", "P6", "P3", "P4", "P5", "P7"]

    for idx, mn in enumerate(available):
        ax = axes[0, idx]
        interp = data[mn]["interpolation"]
        for pid in path_order:
            if pid not in interp:
                continue
            r = interp[pid]
            st = STYLES[pid]
            mean = np.array(r["mean_wr"])
            std = np.array(r["std_wr"])
            ax.plot(r["alphas"], mean, color=st["color"], ls=st["ls"],
                    marker=st["marker"], markersize=4.5, linewidth=2.0,
                    label=st["label"], markeredgecolor=st["color"],
                    markeredgewidth=0.8, zorder=3)
            ax.fill_between(r["alphas"], np.maximum(mean - std, 0),
                            np.minimum(mean + std, 1.05),
                            alpha=0.12, color=st["color"], zorder=1)
        ax.set_xlabel(r"Interpolation coefficient $\alpha$", fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Win rate", fontweight="bold")
        ax.set_title(mn, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(y=0.9, color="#999999", ls=":", lw=0.6)
        ax.grid(True, alpha=0.1, lw=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(handles), 4), bbox_to_anchor=(0.5, -0.22),
               frameon=True, edgecolor="#cccccc", fontsize=8.5,
               handlelength=2.5, columnspacing=1.0)
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    path = os.path.join(OUTPUT_DIR, "fig2_interpolation.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# FIG 3: BARRIER MATRIX

def fig3_barrier_matrix(data):
    available = [m for m in MAPS if m in data and "interpolation" in data[m]]
    if not available:
        print("  No barrier data"); return

    path_to_pair = {
        "P1": ("MAPPO-ct", "M2H"), "P2": ("MAPPO-ct", "MAPPO-ns"),
        "P4": ("M2H", "HAPPO"),
        "P5": ("MAPPO-ct", "HAPPO"), "P6": ("MAPPO-ns", "M2H"),
        "P7": ("MAPPO-ns", "HAPPO"),
    }
    methods_int = METHOD_ORDER_SPEC_INT
    methods_disp = METHOD_ORDER_SPEC
    nm = len(methods_int)
    n = len(available)

    fig = plt.figure(figsize=(4.8 * n + 1.2, 4.8))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1]*n + [0.05],
                          wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cbar_ax = fig.add_subplot(gs[0, n])

    im = None
    for idx, mn in enumerate(available):
        ax = axes[idx]
        interp = data[mn].get("interpolation", {})
        mat = np.full((nm, nm), np.nan)
        for pid, (a, b) in path_to_pair.items():
            if pid not in interp:
                continue
            if a in methods_int and b in methods_int:
                i, j = methods_int.index(a), methods_int.index(b)
                val = interp[pid]["mean_barrier"]
                mat[i, j] = val
                mat[j, i] = val
        np.fill_diagonal(mat, 0.0)

        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(nm))
        ax.set_xticklabels(methods_disp, rotation=40, ha="right", fontsize=8.5,
                           fontweight="bold")
        ax.set_yticks(range(nm))
        if idx == 0:
            ax.set_yticklabels(methods_disp, fontsize=8.5, fontweight="bold")
        else:
            ax.set_yticklabels([])
        ax.set_title(mn, fontweight="bold", pad=8)
        ax.tick_params(axis="both", length=0, pad=3)

        for i in range(nm):
            for j in range(nm):
                if np.isnan(mat[i, j]):
                    ax.text(j, i, "\u2014", ha="center", va="center",
                            fontsize=11, color="#999999")
                else:
                    v = mat[i, j]
                    txt_color = "white" if v > 0.65 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=11, color=txt_color, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_linewidth(1.8)

    if im is not None:
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label(r"Barrier $\mathcal{B}$", fontweight="bold", fontsize=10)
        cb.ax.tick_params(labelsize=9, width=1.0)
        cb.outline.set_linewidth(1.5)

    fig.subplots_adjust(bottom=0.18, top=0.92, left=0.06, right=0.96)
    path = os.path.join(OUTPUT_DIR, "fig3_barrier_matrix.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# FIG 4: SPECIALIZATION

def fig4_specialization(data):
    available = [m for m in MAPS if m in data]
    has_clone = any("clone_all" in data[m] for m in available)
    has_spec = any("specialization" in data[m] for m in available)
    if not has_clone and not has_spec:
        print("  No specialization data"); return

    maps_with_clone = [m for m in available if "clone_all" in data[m]]
    maps_with_spec = [m for m in available if "specialization" in data[m]]
    n_cols = max(len(maps_with_clone), len(maps_with_spec), 3)
    n_rows = (1 if not has_clone or not has_spec else 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    int_methods = METHOD_ORDER_SPEC_INT

    row_clone = 0
    row_spec = 1 if n_rows == 2 else 0

    if has_clone:
        for col, mn in enumerate(maps_with_clone):
            ax = axes[row_clone, col]
            clone = data[mn].get("clone_all", {})
            labels_disp = []
            base_vals = []
            clone_vals = []
            clone_stds = []
            colors = []
            for label_int in int_methods:
                if label_int not in clone:
                    continue
                r = clone[label_int]
                c = MC.get(label_int, "gray")
                labels_disp.append(dn(label_int))
                base_vals.append(r["mean_base_wr"])
                clone_vals.append(r["mean_clone_wr"])
                clone_stds.append(r["std_clone_wr"])
                colors.append(c)

            x = np.arange(len(labels_disp))
            w = 0.32
            for i in range(len(x)):
                ax.bar(x[i] - w/2, base_vals[i], w, color=colors[i], alpha=0.3,
                       edgecolor=colors[i], linewidth=1.2)
                ax.bar(x[i] + w/2, clone_vals[i], w, color=colors[i],
                       edgecolor=colors[i], linewidth=1.2,
                       yerr=clone_stds[i], capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_disp, fontsize=9, fontweight="bold",
                               rotation=30, ha="right")
            if col == 0:
                ax.set_ylabel("Win rate", fontweight="bold", fontsize=11)
            ax.set_title(f"Clone-all \u2014 {mn}", fontweight="bold", fontsize=12)
            ax.set_ylim(0, 1.15)
            ax.grid(True, axis="y", alpha=0.15)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            if col == 0:
                ax.legend([Patch(facecolor="gray", alpha=0.3, edgecolor="gray", lw=1),
                           Patch(facecolor="gray", edgecolor="gray", lw=1)],
                          ["Original", "Clone-all"],
                          fontsize=9, framealpha=0.9, loc="upper right")
        for col in range(len(maps_with_clone), n_cols):
            axes[row_clone, col].set_visible(False)

    if has_spec:
        for col, mn in enumerate(maps_with_spec):
            ax = axes[row_spec, col]
            spec = data[mn].get("specialization", {})
            labels_disp = []
            drop_vals = []
            drop_stds = []
            colors = []
            for label_int in int_methods:
                if label_int not in spec:
                    continue
                r = spec[label_int]
                c = MC.get(label_int, "gray")
                labels_disp.append(dn(label_int))
                drop_vals.append(r["mean_swap_drop"])
                drop_stds.append(r["std_swap_drop"])
                colors.append(c)

            x = np.arange(len(labels_disp))
            ax.bar(x, drop_vals, width=0.55, color=colors,
                   edgecolor="black", linewidth=0.8,
                   yerr=drop_stds, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_disp, fontsize=9, fontweight="bold",
                               rotation=30, ha="right")
            if col == 0:
                ax.set_ylabel("Role-swap drop", fontweight="bold", fontsize=11)
            ax.set_title(f"Role-swap \u2014 {mn}", fontweight="bold", fontsize=12)
            ax.axhline(y=0.05, color="#b2182b", ls="--", lw=1.2, alpha=0.5)
            ax.grid(True, axis="y", alpha=0.15)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
        for col in range(len(maps_with_spec), n_cols):
            axes[row_spec, col].set_visible(False)

    plt.tight_layout(h_pad=3.0)
    path = os.path.join(OUTPUT_DIR, "fig4_specialization.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# FIG 5: HOT-SWAP

def fig5_hotswap(data):
    available = [m for m in MAPS if m in data and "hotswap" in data[m]]
    if not available:
        print("  No hot-swap data"); return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 4.2), squeeze=False)
    for idx, mn in enumerate(available):
        ax = axes[0, idx]
        hs = data[mn]["hotswap"]
        hs_ids = sorted(hs.keys())

        labels = []
        for h in hs_ids:
            team = dn(hs[h]["team"])
            donor = dn(hs[h]["donor"])
            labels.append(f"{team} \u2190 {donor}")

        base = [hs[h]["mean_base_wr"] for h in hs_ids]
        swap = [hs[h]["mean_swap_wr_overall"] for h in hs_ids]
        swap_std = [hs[h]["std_swap_wr_overall"] for h in hs_ids]

        x = np.arange(len(hs_ids))
        w = 0.28
        for i in range(len(hs_ids)):
            team_int = hs[hs_ids[i]]["team"]
            is_cross = "HAPPO" in team_int or "HAPPO" in hs[hs_ids[i]]["donor"]
            c_base = "#b2182b" if is_cross else "#2166ac"
            c_swap = "#d6604d" if is_cross else "#67a9cf"
            ax.bar(x[i] - w/2, base[i], w, color=c_base, alpha=0.4,
                   edgecolor=c_base, linewidth=1.2,
                   label="Original" if i == 0 else None)
            ax.bar(x[i] + w/2, swap[i], w, color=c_swap,
                   edgecolor="black", linewidth=0.8,
                   yerr=swap_std[i], capsize=3,
                   label="After swap" if i == 0 else None)
            drop = base[i] - swap[i]
            if drop > 0.02:
                ax.annotate(f"\u2193{drop:.0%}", xy=(x[i] + w/2, swap[i]),
                            xytext=(0, -14), textcoords="offset points",
                            ha="center", fontsize=8, fontweight="bold",
                            color="#b2182b")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
        if idx == 0:
            ax.set_ylabel("Win rate", fontweight="bold")
        ax.set_title(mn, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
        ax.grid(True, axis="y", alpha=0.1, lw=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_hotswap.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# FIG 6: WEIGHT SPACE

def fig6_weight_space(data):
    available = [m for m in MAPS if m in data and "weights" in data[m]]
    if not available:
        print("  No weight data"); return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.0), squeeze=False)
    all_methods = ["MAPPO", "HAPPO", "MAPPO-pre", "MAPPO-ct", "MAPPO-ns", "M2H"]
    colors_w = {
        "MAPPO": "#762a83", "HAPPO": "#b2182b", "MAPPO-pre": "#888888",
        "MAPPO-ct": "#2166ac", "MAPPO-ns": "#67a9cf", "M2H": "#1b7837",
    }
    markers_w = {
        "MAPPO": "D", "HAPPO": "^", "MAPPO-pre": "s",
        "MAPPO-ct": "o", "MAPPO-ns": "p", "M2H": "*",
    }

    for idx, mn in enumerate(available):
        ax = axes[0, idx]
        w = data[mn]["weights"]
        for label_int in all_methods:
            if label_int not in w:
                continue
            r = w[label_int]
            disp = dn(label_int)
            ax.scatter(r["mean_base_cos"], r["mean_intra_cos"],
                       marker=markers_w.get(label_int, "o"), s=130,
                       color=colors_w[label_int], edgecolors="black",
                       linewidths=1.0, label=disp, zorder=5)
        if idx == 0:
            ax.set_ylabel("Intra-team cosine similarity", fontweight="bold")
        ax.set_xlabel("Ancestor cosine similarity", fontweight="bold")
        ax.set_title(mn, fontweight="bold")
        ax.set_xlim(0.20, 1.08)
        ax.set_ylim(0.20, 1.08)
        ax.plot([0.20, 1.08], [0.20, 1.08], "k--", lw=0.5, alpha=0.15)
        ax.grid(True, alpha=0.1, lw=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95,
                  markerscale=1.1, edgecolor="#cccccc",
                  handletextpad=0.5, borderpad=0.6)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig6_weight_space.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# FIG 7: MIXED TEAM

def fig7_mixed_team(data):
    available = [m for m in MAPS if m in data and "mixed_team" in data[m]]
    if not available:
        print("  No mixed-team data"); return
    SHORT = {
        "MAPPO-Shared": "M-Shared", "MAPPO-Sep": "M-Sep",
        "M2H": "M2H", "HAPPO-Full": "HAPPO-F", "MAPPO-Full": "M-Full",
    }
    def sn(name):
        return SHORT.get(name, name)

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    for idx, mn in enumerate(available):
        ax = axes[0, idx]
        mt = data[mn]["mixed_team"]
        mt_ids = sorted(mt.keys())
        labels = []
        for m_ in mt_ids:
            a = sn(dn(mt[m_].get("method_a", "")))
            b = sn(dn(mt[m_].get("method_b", "")))
            labels.append(f"{a}\n+ {b}")
        wrs = [mt[m_]["mean_wr"] for m_ in mt_ids]
        stds = [mt[m_]["std_wr"] for m_ in mt_ids]
        desc = [mt[m_].get("description", "") for m_ in mt_ids]
        colors = ["#67a9cf" if "HAPPO" not in d else "#d6604d" for d in desc]
        x = np.arange(len(mt_ids))
        ax.bar(x, wrs, color=colors, edgecolor="black", linewidth=0.8,
               yerr=stds, capsize=3, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, linespacing=1.15)
        if idx == 0:
            ax.set_ylabel("Win rate", fontweight="bold")
        ax.set_title(mn, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.grid(True, axis="y", alpha=0.1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig7_mixed_team.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# APPENDIX: TRAINING CURVES (per-map)

def plot_training_curves_permap(rows):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        m, method, seed = r["map"], r["method"], int(r["seed"])
        step, val = int(r["step"]), float(r["eval_win_rate"])
        data[m][method][seed].append((step, val))

    for map_name in MAPS:
        if map_name not in data:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in MAIN_METHODS:
            if method not in data[map_name]:
                continue
            all_steps, all_vals = [], []
            for seed, pts in data[map_name][method].items():
                pts.sort()
                all_steps.append([p[0] for p in pts])
                all_vals.append([p[1] for p in pts])
            common, mean, std = interpolate_curves(all_steps, all_vals)
            if common is None:
                continue
            label = LABEL_MAP.get(method, method)
            color = COLORS.get(method, None)
            ls = LINESTYLES.get(method, "-")
            ax.plot(common / 1e6, mean, label=label, color=color,
                    linewidth=2, linestyle=ls)
            ax.fill_between(common / 1e6, mean - std, mean + std,
                            color=color, alpha=0.15)
        ax.set_xlabel("Environment Steps (M)")
        ax.set_ylabel("Eval Win Rate")
        ax.set_title(f"Training Curve \u2014 {map_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(-0.05, 1.05)
        save_fig(fig, os.path.join(OUTPUT_DIR, f"training_curve_{map_name}"))


# APPENDIX: SEVERITY SWEEPS

def _collect_severity_data(map_rows, method, ptype, severities, include_clean=True):
    xs, means, stds = [], [], []
    if include_clean:
        clean_vals = [float(r["win_rate"]) for r in map_rows
                      if r["method"] == method
                      and r["perturbation_type"] == "clean"
                      and r["win_rate"] is not None]
        if clean_vals:
            xs.append(0.0)
            means.append(np.mean(clean_vals))
            stds.append(np.std(clean_vals) if len(clean_vals) > 1 else 0)
    for sev in severities:
        sev_str = str(sev)
        vals = [float(r["win_rate"]) for r in map_rows
                if r["method"] == method
                and r["perturbation_type"] == ptype
                and r["severity"] == sev_str
                and r["win_rate"] is not None]
        if vals:
            xs.append(float(sev))
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
    return np.array(xs), np.array(means), np.array(stds)


def plot_severity_sweeps(rob_rows):
    for map_name in MAPS:
        map_rows = [r for r in rob_rows if r["map"] == map_name]
        if not map_rows:
            continue
        for ptype, xlabel, severities in [
            ("missing", "Missing Rate", ["0.1", "0.2", "0.3", "0.4", "0.5"]),
            ("noise", "Noise Sigma", ["0.05", "0.1", "0.2", "0.3", "0.4"]),
            ("delay", "Delay Steps", ["1", "2", "3", "4"]),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            for method in MAIN_METHODS:
                xs, mean, std = _collect_severity_data(
                    map_rows, method, ptype, severities)
                if len(xs) == 0:
                    continue
                label = LABEL_MAP.get(method, method)
                color = COLORS.get(method, None)
                ls = LINESTYLES.get(method, "-")
                ax.plot(xs, mean, label=label, color=color,
                        linewidth=2.2, linestyle=ls, marker="o", markersize=5)
                ax.fill_between(xs, np.clip(mean - std, 0, 1),
                                np.clip(mean + std, 0, 1),
                                color=color, alpha=0.15)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Win Rate")
            ax.set_title(f"{ptype.capitalize()} Severity Sweep \u2014 {map_name}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            save_fig(fig, os.path.join(OUTPUT_DIR, f"severity_{ptype}_{map_name}"))


# APPENDIX: ROBUSTNESS (combined 3-panel)

def plot_robustness_severity(rob_rows):
    perturbation_configs = [
        ("noise", "Observation Noise (\u03c3)",
         ["0.05", "0.1", "0.2", "0.3", "0.4"]),
        ("delay", "Action Delay (steps)",
         ["1", "2", "3", "4"]),
        ("missing", "Observation Missing Rate",
         ["0.1", "0.2", "0.3", "0.4", "0.5"]),
    ]

    for map_name in MAPS:
        map_rows = [r for r in rob_rows if r["map"] == map_name]
        if not map_rows:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, (ptype, xlabel, severities) in zip(axes, perturbation_configs):
            for method in MAIN_METHODS:
                xs, mean, std = _collect_severity_data(
                    map_rows, method, ptype, severities)
                if len(xs) == 0:
                    continue
                label = LABEL_MAP.get(method, method)
                color = COLORS.get(method, None)
                ls = LINESTYLES.get(method, "-")
                ax.plot(xs, mean, label=label, color=color,
                        linewidth=2.2, linestyle=ls, marker="o", markersize=5)
                ax.fill_between(xs, np.clip(mean - std, 0, 1),
                                np.clip(mean + std, 0, 1),
                                color=color, alpha=0.15)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Win Rate")
            ax.set_title(f"{ptype.capitalize()} \u2014 {map_name}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

        axes[0].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"Robustness under Increasing Perturbation \u2014 {map_name}",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        save_fig(fig, os.path.join(OUTPUT_DIR, f"robustness_severity_{map_name}"))

        for ptype, xlabel, severities in perturbation_configs:
            fig_s, ax_s = plt.subplots(figsize=(7, 5))
            for method in MAIN_METHODS:
                xs, mean, std = _collect_severity_data(
                    map_rows, method, ptype, severities)
                if len(xs) == 0:
                    continue
                label = LABEL_MAP.get(method, method)
                color = COLORS.get(method, None)
                ls = LINESTYLES.get(method, "-")
                ax_s.plot(xs, mean, label=label, color=color,
                          linewidth=2.2, linestyle=ls, marker="o", markersize=5)
                ax_s.fill_between(xs, np.clip(mean - std, 0, 1),
                                  np.clip(mean + std, 0, 1),
                                  color=color, alpha=0.15)
            ax_s.set_xlabel(xlabel)
            ax_s.set_ylabel("Win Rate")
            ax_s.set_title(f"{ptype.capitalize()} Robustness \u2014 {map_name}")
            ax_s.legend(loc="upper right", fontsize=8.5)
            ax_s.grid(True, alpha=0.3)
            ax_s.set_ylim(-0.05, 1.05)
            fig_s.tight_layout()
            save_fig(fig_s, os.path.join(OUTPUT_DIR, f"robustness_{ptype}_{map_name}"))


# APPENDIX: PARAM COMPARISON

def plot_param_comparison(rob_rows):
    for map_name in MAPS:
        map_rows = [r for r in rob_rows if r["map"] == map_name]
        if not map_rows:
            continue

        perturbation_groups = [
            ("clean", "clean", ""),
            ("missing_0.1", "missing", "0.1"),
            ("missing_0.3", "missing", "0.3"),
            ("noise_0.1", "noise", "0.1"),
            ("noise_0.2", "noise", "0.2"),
            ("delay_1", "delay", "1"),
            ("delay_2", "delay", "2"),
        ]

        fig, ax = plt.subplots(figsize=(12, 5))
        n_groups = len(perturbation_groups)
        methods = PARAM_COMPARE_METHODS
        n_methods = len(methods)
        width = 0.8 / n_methods
        x = np.arange(n_groups)

        for mi, method in enumerate(methods):
            means, stds = [], []
            for label, ptype, severity in perturbation_groups:
                vals = [float(r["win_rate"]) for r in map_rows
                        if r["method"] == method
                        and r["perturbation_type"] == ptype
                        and r["severity"] == severity
                        and r["win_rate"] is not None]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if len(vals) > 1 else 0)

            offset = (mi - n_methods / 2 + 0.5) * width
            mlabel = LABEL_MAP.get(method, method)
            color = COLORS.get(method, None)
            ax.bar(x + offset, means, width, yerr=stds,
                   label=mlabel, color=color, edgecolor="black",
                   linewidth=0.5, capsize=3, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in perturbation_groups], rotation=30, ha="right")
        ax.set_ylabel("Win Rate")
        ax.set_title(f"Continuation Methods Comparison \u2014 {map_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        save_fig(fig, os.path.join(OUTPUT_DIR, f"param_comparison_{map_name}"))


# APPENDIX: CLEAN VS DELAY SCATTER

def plot_clean_vs_delay(scatter_rows):
    for map_name in MAPS:
        fig, ax = plt.subplots(figsize=(6, 6))
        has_data = False
        for method in MAIN_METHODS:
            clean_vals, delay_vals = [], []
            for r in scatter_rows:
                if r["map"] != map_name or r["method"] != method:
                    continue
                if r.get("clean_win_rate") and r.get("delay1_win_rate"):
                    clean_vals.append(float(r["clean_win_rate"]))
                    delay_vals.append(float(r["delay1_win_rate"]))
            if clean_vals:
                has_data = True
                label = LABEL_MAP.get(method, method)
                color = COLORS.get(method, None)
                ax.scatter(clean_vals, delay_vals, label=label,
                           color=color, s=60, edgecolors="black", linewidths=0.5)
        if has_data:
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
            ax.set_xlabel("Clean Win Rate")
            ax.set_ylabel("Delay-1 Win Rate")
            ax.set_title(f"Clean vs Robust \u2014 {map_name}")
            ax.legend(loc="lower right", fontsize=8)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            save_fig(fig, os.path.join(OUTPUT_DIR, f"clean_vs_delay_scatter_{map_name}"))
        else:
            plt.close(fig)


# APPENDIX: COMBINED TRAINING CURVES

def plot_training_curves_combined(rows):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        m, method, seed = r["map"], r["method"], int(r["seed"])
        step, val = int(r["step"]), float(r["eval_win_rate"])
        data[m][method][seed].append((step, val))

    available = [m for m in MAPS if m in data]
    if not available:
        return
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.0), squeeze=False)
    for idx, map_name in enumerate(available):
        ax = axes[0, idx]
        for method in MAIN_METHODS:
            if method not in data[map_name]:
                continue
            all_steps, all_vals = [], []
            for seed, pts in data[map_name][method].items():
                pts.sort()
                all_steps.append([p[0] for p in pts])
                all_vals.append([p[1] for p in pts])
            common, mean, std = interpolate_curves(all_steps, all_vals)
            if common is None:
                continue
            label = LABEL_MAP.get(method, method)
            color = COLORS.get(method, None)
            ls = LINESTYLES.get(method, "-")
            ax.plot(common / 1e6, mean, label=label, color=color,
                    linewidth=2, linestyle=ls)
            ax.fill_between(common / 1e6, mean - std, mean + std,
                            color=color, alpha=0.15)
        ax.set_xlabel("Environment Steps (M)")
        if idx == 0:
            ax.set_ylabel("Eval Win Rate")
        ax.set_title(map_name, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, "training_curves_combined"))


# MAIN

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CC v3 data for main figures (fig1-7)
    cc_data = None
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            cc_data = json.load(f)
        print(f"Loaded CC data for maps: {list(cc_data.keys())}")
    else:
        print(f"[WARN] CC results not found: {RESULTS_FILE}")

    # Load robustness tables for appendix figures
    tc_rows = load_csv(os.path.join(TABLES_DIR, "training_curve.csv"))
    rob_rows = load_csv(os.path.join(TABLES_DIR, "robustness_summary.csv"))
    scatter_rows = load_csv(os.path.join(TABLES_DIR, "clean_vs_delay_scatter.csv"))

    print("\n=== Generating main figures (fig1-fig7) ===")
    if cc_data:
        print("\n[Fig 1] Training curves (combined)...")
        fig1_training_curves(cc_data)
        print("\n[Fig 2] Interpolation curves...")
        fig2_interpolation(cc_data)
        print("\n[Fig 3] Barrier matrix...")
        fig3_barrier_matrix(cc_data)
        print("\n[Fig 4] Specialization (clone-all + role-swap)...")
        fig4_specialization(cc_data)
        print("\n[Fig 5] Hot-swap...")
        fig5_hotswap(cc_data)
        print("\n[Fig 6] Weight space...")
        fig6_weight_space(cc_data)
        print("\n[Fig 7] Mixed team...")
        fig7_mixed_team(cc_data)

    print("\n=== Generating appendix figures ===")
    if tc_rows:
        print("\n[Appendix] Per-map training curves...")
        plot_training_curves_permap(tc_rows)
        print("\n[Appendix] Combined training curves...")
        plot_training_curves_combined(tc_rows)

    if rob_rows:
        print("\n[Appendix] Severity sweeps...")
        plot_severity_sweeps(rob_rows)
        print("\n[Appendix] Robustness severity (3-panel)...")
        plot_robustness_severity(rob_rows)
        print("\n[Appendix] Parameter comparison...")
        plot_param_comparison(rob_rows)

    if scatter_rows:
        print("\n[Appendix] Clean vs delay scatter...")
        plot_clean_vs_delay(scatter_rows)

    print(f"\n[DONE] All figures saved to: {OUTPUT_DIR}")
    print("All PDFs should now use Type 42 (TrueType) fonts only.")


if __name__ == "__main__":
    main()
