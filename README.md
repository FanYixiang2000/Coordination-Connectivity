# Coordination Connectivity: Shared Initialization Shapes the Joint-Policy Landscape in MARL

Code and result data for the NeurIPS 2026 paper.

## Contents

```
configs/
  method_registry.py            training-history configs (MAPPO-Shared, MAPPO-Sep, M2H, MAPPO-Full, HAPPO-Full)
harl_ext/                       extensions on top of HARL: continuation training, checkpoint
                                selection, deterministic evaluation, perturbation utilities
scripts/
  train_full_budget.py              train MAPPO-Full / HAPPO-Full from scratch
  train_continuation.py             shared pre-training -> continuation (MAPPO-Shared, MAPPO-Sep, M2H)
  train_*_resume.py                 resume variants of the above
  eval_clean_and_robust.py          clean and perturbed evaluation (appendix robustness)
  coordination_connectivity_v3.py   main probes: interpolation, hot-swap, specialization, weight analysis (SMAC)
  cc_v3_supplement.py               clone-all, extra interpolation paths, mixed-team probes
  cc_v3_policy_metrics.py           endpoint-vs-endpoint behavioral divergence (Protocol B)
  q4_corridor_pathwise_behavior.py  per-alpha behavioral divergence on corridor (Protocol A)
  q4_corridor_all_paths_behavior.py Protocol A over all seven paths
  exp_mpe_train_and_eval.py         MPE (PettingZoo) actor-critic training + interpolation
  exp_mpe_value_decomp_cc.py        VDN/QMIX pilot on MPE
  regenerate_figures_type42.py      figures 2-3, appendix figures 5-8, robustness figures
  make_behavior_divergence_figure.py  Figure 4 (TV/KL along interpolation paths)
  make_mpe_interpolation_figure.py    Figure 9 (MPE interpolation curves)
  make_hotswap_figure.py              appendix Figure 6 (hot-swap bars)
  make_behavior_summary_table.py      Table 8 (midpoint divergence summary)
exp/
  cc_v3/                 SMAC results (3s5z, corridor, MMM2), one JSON per map + merged
  q4_rebuttal/           behavioral divergence results (Protocols A and B)
  mpe_cc/                MPE interpolation results
  mpe_vd_cc/             VDN/QMIX pilot results
  smac_robustness/       plot-ready tables for appendix robustness figures
```

## Reproducing figures and tables

All figures and tables in the paper can be regenerated from the JSON/CSV
result files in `exp/` without re-running training:

```bash
python scripts/regenerate_figures_type42.py        # figures 2-3, appendix figures 5-8, robustness figures
python scripts/make_behavior_divergence_figure.py  # figure 4
python scripts/make_mpe_interpolation_figure.py    # figure 9 (both panels)
python scripts/make_hotswap_figure.py              # appendix figure 6, camera-ready style (no in-bar labels)
python scripts/make_vdn_figure.py                  # appendix figure 10 (VDN pilot)
python scripts/make_behavior_summary_table.py      # table 8
```

Figure 1 is a hand-made schematic. Figure/table reproduction requires only
Python 3.8+, numpy, and matplotlib.

## Re-running experiments

The SMAC experiments run on top of the HARL framework
(https://github.com/PKU-MARL/HARL) with StarCraft II / SMAC installed; we do
not fork HARL, all extensions live in `harl_ext/` and the scripts import HARL
as installed. PyTorch and gym are required. Training the five histories:
`train_full_budget.py` for the from-scratch baselines, `train_continuation.py`
for the shared-pretraining continuations; `configs/method_registry.py`
specifies the exact configuration for each history (hyperparameters in
Appendix A). The probe scripts (`coordination_connectivity_v3.py`,
`cc_v3_supplement.py`, and the behavioral-divergence scripts) expect the
resulting checkpoints for the five training histories described in Section 3.

The MPE experiments (`exp_mpe_train_and_eval.py`, `exp_mpe_value_decomp_cc.py`)
require PettingZoo (MPE scenarios `simple_reference_v2` and `simple_spread_v2`)
and are self-contained otherwise; protocols are described in Appendix E.

Evaluation protocol: greedy (deterministic) actions, 100 episodes per
interpolation point on SMAC, 50 on MPE, seeds 1-4, alpha grid {0, 0.1, ..., 1.0}.

## License

The code and result data in this repository are released under the MIT License
(see `LICENSE`). Third-party dependencies are installed separately and are not
distributed here: HARL (MIT), PettingZoo (MIT), SMAC (MIT), and StarCraft II,
which is subject to Blizzard's own end-user license agreement.
