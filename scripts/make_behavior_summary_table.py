#!/usr/bin/env python3
"""Reproduce Appendix Table `tab:behavior-midpoint` from exp/q4_rebuttal JSONs (read-only)."""
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
Q4_DIR = PROJECT_DIR / "exp" / "q4_rebuttal"

GROUPS = {
    "Same-source": ["P1", "P2", "P6"],
    "Cross-history": ["P3", "P5", "P7"],
}
MAPS = ["3s5z", "corridor"]


def main():
    header = f"{'Map':<10} {'Group':<14} {'B':>6} {'TV':>6} {'JS':>6} {'disagr':>7} {'w-dist':>7} {'midWR':>6}"
    print(header)
    print("-" * len(header))
    for map_name in MAPS:
        data = json.load(open(Q4_DIR / f"policy_metrics_{map_name}_core_ep20.json"))[map_name]
        for group, pids in GROUPS.items():
            B, tv, js, dis, w, wr = [], [], [], [], [], []
            for pid in pids:
                for seed_row in data["paths"][pid]["seeds"]:
                    B.append(seed_row["existing_barrier"])  # barrier from the main interpolation evaluations
                    w.append(seed_row["weight_metrics"]["mean_normalized_l2"])  # normalized L2 between endpoint actors
                    mid = [r for r in seed_row["alpha_metrics"] if r["alpha"] == 0.5][0]  # midpoint measurements
                    tv.append(mid["tv"])
                    js.append(mid["js"])
                    dis.append(mid["argmax_disagree"])
                    wr.append(mid["probe_win_rate"])
            assert len(B) == 12, f"expected 12 path-seed rows, got {len(B)}"  # 3 paths x 4 seeds per group
            print(
                f"{map_name:<10} {group:<14} "
                f"{np.mean(B):>6.2f} {np.mean(tv):>6.2f} {np.mean(js):>6.2f} "
                f"{np.mean(dis):>7.2f} {np.mean(w):>7.2f} {np.mean(wr):>6.2f}"
            )


if __name__ == "__main__":
    main()
