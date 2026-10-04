#!/usr/bin/env python3
"""
Coordination Connectivity v3 — Step-Aligned + Parallel Evaluation
==================================================================
Key: uses N_PARALLEL SC2 instances to evaluate episodes concurrently.
     With 20 workers on a 128-core machine, ~20x speedup.
"""
import argparse, copy, glob, json, os, subprocess, sys, time, tempfile
import numpy as np
import torch
import torch.multiprocessing as mp

import matplotlib
matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
EXP_DIR = os.path.join(PROJECT_DIR, "exp", "smac_robustness")
RESULTS_DIR = os.path.join(PROJECT_DIR, "exp", "cc_v3")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)
from configs.method_registry import HARL_ROOT, HIDDEN_SIZES
from harl_ext.run_selection import find_smac_models_dir

sys.path.insert(0, HARL_ROOT)

try:
    import pysc2.maps.lib as _lib
    def _safe_get_maps():
        maps = {}
        for m in _lib.Map.all_subclasses():
            fn = getattr(m, 'filename', None)
            bn = getattr(m, 'battle_net', None)
            if fn or bn:
                name = m.__name__
                if name not in maps:
                    maps[name] = m
        return maps
    _lib.get_maps = _safe_get_maps
except Exception:
    pass

from harl.models.policy_models.stochastic_policy import StochasticPolicy
from gym.spaces import Discrete, Box
from harl.envs.smac.StarCraft2_Env import StarCraft2Env

MAP_AGENTS = {
    "3s5z": 8, "MMM2": 10, "corridor": 6,
    "5m_vs_6m": 5, "8m_vs_9m": 8, "6h_vs_8z": 6, "25m": 25, "3s5z_vs_3s6z": 8,
}
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_EPISODES = 100
N_PARALLEL = 20

METHOD_INFO = {
    "mappo_shared_full":                {"shared": True,  "algo": "mappo"},
    "happo_nonshared_full":             {"shared": False, "algo": "happo"},
    "mappo_to_mappo_shared_stage1":     {"shared": True,  "algo": "mappo"},
    "mappo_to_mappo_shared_stage2":     {"shared": True,  "algo": "mappo"},
    "mappo_to_mappo_nonshared_stage2":  {"shared": False, "algo": "mappo"},
    "mappo_to_happo_nonshared_stage2":  {"shared": False, "algo": "happo"},
}

SHORT = {
    "mappo_shared_full":                "MAPPO",
    "happo_nonshared_full":             "HAPPO",
    "mappo_to_mappo_shared_stage1":     "MAPPO-pre",
    "mappo_to_mappo_shared_stage2":     "MAPPO-ct",
    "mappo_to_mappo_nonshared_stage2":  "MAPPO-ns",
    "mappo_to_happo_nonshared_stage2":  "M2H",
}

INTERP_PATHS = [
    ("P1", "mappo_to_mappo_shared_stage2", "mappo_to_happo_nonshared_stage2",
     "MAPPO-ct → M2H (same source)"),
    ("P2", "mappo_to_mappo_shared_stage2", "mappo_to_mappo_nonshared_stage2",
     "MAPPO-ct → MAPPO-ns (same source)"),
    ("P3", "mappo_shared_full", "happo_nonshared_full",
     "MAPPO → HAPPO (cross-history)"),
    ("P4", "mappo_to_happo_nonshared_stage2", "happo_nonshared_full",
     "M2H → HAPPO (same algo, diff init)"),
]

HOTSWAP_PAIRS = [
    ("HS1", "mappo_to_happo_nonshared_stage2", "mappo_to_mappo_shared_stage2",
     "M2H team ← MAPPO-ct"),
    ("HS2", "mappo_to_mappo_nonshared_stage2", "mappo_to_mappo_shared_stage2",
     "MAPPO-ns team ← MAPPO-ct"),
    ("HS3", "happo_nonshared_full", "mappo_shared_full",
     "HAPPO team ← MAPPO"),
    ("HS4", "happo_nonshared_full", "mappo_to_mappo_shared_stage2",
     "HAPPO team ← MAPPO-ct"),
]

SPEC_METHODS = [
    ("mappo_to_happo_nonshared_stage2",  "M2H"),
    ("mappo_to_mappo_nonshared_stage2",  "MAPPO-ns"),
    ("happo_nonshared_full",             "HAPPO"),
    ("mappo_to_mappo_shared_stage2",     "MAPPO-ct"),
]


def find_model_dir(map_name, method_name, seed):
    algo_dir = "happo" if "happo" in method_name and "to_mappo" not in method_name else "mappo"
    model_dir = find_smac_models_dir(
        EXP_DIR, map_name, algo_dir, method_name, seed
    )
    if model_dir is None:
        raise FileNotFoundError(f"No models: {method_name}, map={map_name}, seed={seed}")
    return model_dir


def load_base_config(map_name):
    cfg_path = os.path.join(HARL_ROOT, "tuned_configs", "smac", map_name, "happo", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["algo_args"]["model"]["hidden_sizes"] = list(HIDDEN_SIZES)
    cfg["env_args"]["map_name"] = map_name
    return cfg


_SPACES = {}
def get_spaces(map_name):
    if map_name not in _SPACES:
        cfg = load_base_config(map_name)
        tmp = StarCraft2Env(copy.deepcopy(cfg["env_args"]))
        obs_shape = tmp.get_obs_size()[0]
        n_actions = tmp.get_total_actions()
        tmp.close()
        _SPACES[map_name] = {
            "obs": Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32),
            "act": Discrete(n_actions),
        }
    return _SPACES[map_name]


def load_state_dicts(map_name, method_name, seed):
    model_dir = find_model_dir(map_name, method_name, seed)
    info = METHOD_INFO[method_name]
    n_agents = MAP_AGENTS[map_name]
    sds = []
    if info["shared"]:
        sd = torch.load(os.path.join(model_dir, "actor_agent0.pt"),
                        map_location="cpu", weights_only=False)
        sds = [copy.deepcopy(sd) for _ in range(n_agents)]
    else:
        for i in range(n_agents):
            sd = torch.load(os.path.join(model_dir, f"actor_agent{i}.pt"),
                            map_location="cpu", weights_only=False)
            sds.append(sd)
    return sds


def interpolate_sds(sds_a, sds_b, alpha):
    return [{k: (1.0 - alpha) * sds_a[i][k] + alpha * sds_b[i][k]
             for k in sds_a[i]} for i in range(len(sds_a))]


def create_actors(map_name, sds):
    sp = get_spaces(map_name)
    cfg = load_base_config(map_name)
    model_cfg = copy.deepcopy(cfg["algo_args"]["model"])
    model_cfg["action_space"] = sp["act"]
    for k, v in cfg["algo_args"]["algo"].items():
        model_cfg.setdefault(k, v)
    actors = []
    for sd in sds:
        actor = StochasticPolicy(model_cfg, sp["obs"], sp["act"],
                                 device=torch.device("cpu"))
        actor.load_state_dict(sd)
        actor.eval()
        actors.append(actor)
    return actors


def make_env(map_name):
    cfg = load_base_config(map_name)
    env = StarCraft2Env(copy.deepcopy(cfg["env_args"]))
    env.seed(42)
    return env


def run_episode(env, actors, map_name):
    cfg = load_base_config(map_name)
    recurrent_n = cfg["algo_args"]["model"].get("recurrent_n", 1)
    hidden_size = HIDDEN_SIZES[-1]
    n_agents = MAP_AGENTS[map_name]
    rnn_states = [torch.zeros(1, recurrent_n, hidden_size) for _ in range(n_agents)]
    masks = [torch.ones(1, 1) for _ in range(n_agents)]
    env.reset()
    terminated = False
    info = {}
    while not terminated:
        obs_list = [env.get_obs_agent(i) for i in range(n_agents)]
        avail_list = [env.get_avail_agent_actions(i) for i in range(n_agents)]
        actions = []
        for i in range(n_agents):
            obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0)
            avail_t = torch.FloatTensor(avail_list[i]).unsqueeze(0)
            with torch.no_grad():
                action, _, rnn_out = actors[i](
                    obs_t, rnn_states[i], masks[i],
                    available_actions=avail_t, deterministic=True)
                rnn_states[i] = rnn_out
            actions.append(action.item())
        _, _, _, dones, infos, _ = env.step(actions)
        terminated = bool(np.all(dones))
        info = infos[0] if infos else {}
    return info.get("won", False)


def _worker_eval(args):
    """Worker process: create own SC2 env, load actors, run n_eps episodes."""
    sds_path, map_name, n_eps, worker_id = args
    sds = torch.load(sds_path, map_location="cpu", weights_only=False)
    actors = create_actors(map_name, sds)
    env = make_env(map_name)
    wins = 0
    for _ in range(n_eps):
        try:
            won = run_episode(env, actors, map_name)
            if won:
                wins += 1
        except Exception:
            try: env.close()
            except: pass
            time.sleep(2)
            env = make_env(map_name)
    try: env.close()
    except: pass
    return wins


def eval_wr_parallel(sds, map_name, n_ep=None, n_parallel=None):
    """Evaluate win rate using parallel SC2 instances."""
    if n_ep is None:
        n_ep = N_EPISODES
    if n_parallel is None:
        n_parallel = N_PARALLEL

    if n_parallel <= 1 or n_ep < n_parallel:
        actors = create_actors(map_name, sds)
        env = make_env(map_name)
        wins = 0
        for _ in range(n_ep):
            try:
                won = run_episode(env, actors, map_name)
                if won: wins += 1
            except Exception:
                try: env.close()
                except: pass
                time.sleep(2)
                env = make_env(map_name)
        try: env.close()
        except: pass
        return wins / n_ep

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp")
    tmp_path = tmp.name
    tmp.close()
    torch.save(sds, tmp_path)

    eps_per_worker = n_ep // n_parallel
    remainder = n_ep % n_parallel
    worker_args = []
    for i in range(n_parallel):
        n = eps_per_worker + (1 if i < remainder else 0)
        if n > 0:
            worker_args.append((tmp_path, map_name, n, i))

    try:
        with mp.Pool(processes=len(worker_args)) as pool:
            results = pool.map(_worker_eval, worker_args)
        total_wins = sum(results)
        total_eps = sum(a[2] for a in worker_args)
    finally:
        os.unlink(tmp_path)

    return total_wins / total_eps


def compute_barrier(win_rates):
    return max(0, min(win_rates[0], win_rates[-1]) - min(win_rates))


# EXP 1: INTERPOLATION

def exp1_interpolation(map_name, seeds, n_parallel):
    print(f"\n{'='*70}")
    print(f"EXP1: INTERPOLATION (parallel={n_parallel})  map={map_name}")
    print(f"{'='*70}")
    results = {}
    for path_id, method_a, method_b, desc in INTERP_PATHS:
        print(f"\n  --- {path_id}: {desc} ---")
        seed_wrs, seed_barriers = [], []
        for seed in seeds:
            print(f"    [seed={seed}]", end="", flush=True)
            t0 = time.time()
            try:
                sds_a = load_state_dicts(map_name, method_a, seed)
                sds_b = load_state_dicts(map_name, method_b, seed)
            except FileNotFoundError as e:
                print(f" SKIP: {e}")
                continue
            wrs = []
            for alpha in ALPHAS:
                sds_interp = interpolate_sds(sds_a, sds_b, alpha)
                wr = eval_wr_parallel(sds_interp, map_name, n_parallel=n_parallel)
                wrs.append(wr)
                print(f" {alpha:.1f}:{wr:.0%}", end="", flush=True)
            barrier = compute_barrier(wrs)
            seed_wrs.append(wrs)
            seed_barriers.append(barrier)
            dt = time.time() - t0
            print(f"  B={barrier:.3f} ({dt:.0f}s)")
        if not seed_wrs: continue
        arr = np.array(seed_wrs)
        results[path_id] = {
            "description": desc,
            "method_a": SHORT[method_a], "method_b": SHORT[method_b],
            "alphas": ALPHAS,
            "mean_wr": arr.mean(axis=0).tolist(),
            "std_wr": arr.std(axis=0).tolist(),
            "per_seed_wr": seed_wrs,
            "mean_barrier": float(np.mean(seed_barriers)),
            "std_barrier": float(np.std(seed_barriers)),
            "per_seed_barriers": seed_barriers,
            "connected": float(np.mean(seed_barriers)) < 0.1,
            "n_seeds": len(seed_wrs),
        }
        s = "CONNECTED" if results[path_id]["connected"] else "BARRIER"
        print(f"    → B={np.mean(seed_barriers):.3f}±{np.std(seed_barriers):.3f} [{s}]")
    return results


# EXP 2: HOT-SWAP

def exp2_hotswap(map_name, seeds, n_parallel):
    print(f"\n{'='*70}")
    print(f"EXP2: HOT-SWAP (parallel={n_parallel})  map={map_name}")
    print(f"{'='*70}")
    n_agents = MAP_AGENTS[map_name]
    results = {}
    for hs_id, team_method, donor_method, desc in HOTSWAP_PAIRS:
        print(f"\n  --- {hs_id}: {desc} ---")
        seed_base, seed_swap = [], []
        for seed in seeds:
            print(f"    [seed={seed}]", end="", flush=True)
            t0 = time.time()
            try:
                sds_team = load_state_dicts(map_name, team_method, seed)
                sds_donor = load_state_dicts(map_name, donor_method, seed)
            except FileNotFoundError as e:
                print(f" SKIP: {e}")
                continue
            base_wr = eval_wr_parallel(sds_team, map_name, n_parallel=n_parallel)
            print(f" base={base_wr:.0%}", end="", flush=True)
            seed_base.append(base_wr)
            agent_wrs = []
            for i in range(n_agents):
                sds_sw = [copy.deepcopy(sd) for sd in sds_team]
                sds_sw[i] = copy.deepcopy(sds_donor[i])
                sw_wr = eval_wr_parallel(sds_sw, map_name, n_parallel=n_parallel)
                agent_wrs.append(sw_wr)
            mean_sw = np.mean(agent_wrs)
            print(f" swap_avg={mean_sw:.0%}", end="", flush=True)
            seed_swap.append(agent_wrs)
            dt = time.time() - t0
            print(f" ({dt:.0f}s)")
        if not seed_swap: continue
        swap_arr = np.array(seed_swap)
        base_arr = np.array(seed_base)
        results[hs_id] = {
            "description": desc, "team": SHORT[team_method], "donor": SHORT[donor_method],
            "n_agents": n_agents,
            "mean_base_wr": float(base_arr.mean()), "std_base_wr": float(base_arr.std()),
            "mean_swap_wr_per_agent": swap_arr.mean(axis=0).tolist(),
            "std_swap_wr_per_agent": swap_arr.std(axis=0).tolist(),
            "mean_swap_wr_overall": float(swap_arr.mean()),
            "std_swap_wr_overall": float(swap_arr.std()),
            "mean_drop": float(base_arr.mean() - swap_arr.mean()),
            "n_seeds": len(seed_swap),
        }
        print(f"    → base={base_arr.mean():.1%} swap={swap_arr.mean():.1%} drop={base_arr.mean()-swap_arr.mean():+.1%}")
    return results


# EXP 3: SPECIALIZATION

def exp3_specialization(map_name, seeds, n_parallel):
    print(f"\n{'='*70}")
    print(f"EXP3: SPECIALIZATION (parallel={n_parallel})  map={map_name}")
    print(f"{'='*70}")
    n_agents = MAP_AGENTS[map_name]
    n_swaps = min(n_agents, 6)
    results = {}
    for method, label in SPEC_METHODS:
        print(f"\n  --- {label} ---")
        seed_base, seed_drops = [], []
        for seed in seeds:
            print(f"    [seed={seed}]", end="", flush=True)
            t0 = time.time()
            try:
                sds = load_state_dicts(map_name, method, seed)
            except FileNotFoundError as e:
                print(f" SKIP: {e}")
                continue
            base_wr = eval_wr_parallel(sds, map_name, n_parallel=n_parallel)
            print(f" base={base_wr:.0%}", end="", flush=True)
            seed_base.append(base_wr)
            drops = []
            for si in range(n_swaps):
                i, j = si, (si + 1) % n_agents
                sds_sw = [copy.deepcopy(sd) for sd in sds]
                sds_sw[i], sds_sw[j] = sds_sw[j], sds_sw[i]
                sw_wr = eval_wr_parallel(sds_sw, map_name, n_parallel=n_parallel)
                drops.append(base_wr - sw_wr)
            mean_drop = np.mean(drops)
            print(f" swap_drop={mean_drop:+.0%}", end="", flush=True)
            seed_drops.append(drops)
            dt = time.time() - t0
            print(f" ({dt:.0f}s)")
        if not seed_drops: continue
        drops_arr = np.array(seed_drops)
        base_arr = np.array(seed_base)
        results[label] = {
            "method": method,
            "mean_base_wr": float(base_arr.mean()),
            "std_base_wr": float(base_arr.std()),
            "mean_swap_drop": float(drops_arr.mean()),
            "std_swap_drop": float(drops_arr.std()),
            "mean_drop_per_pair": drops_arr.mean(axis=0).tolist(),
            "n_seeds": len(seed_drops),
            "specialized": float(drops_arr.mean()) > 0.05,
        }
        sp = "SPECIALIZED" if drops_arr.mean() > 0.05 else "NOT SPECIALIZED"
        print(f"    → base={base_arr.mean():.1%} drop={drops_arr.mean():+.1%} [{sp}]")
    return results


# EXP 4: WEIGHT ANALYSIS

def exp4_weights(map_name, seeds):
    print(f"\n{'='*70}")
    print(f"EXP4: WEIGHT ANALYSIS  map={map_name}")
    print(f"{'='*70}")
    methods_to_load = [
        "mappo_shared_full", "happo_nonshared_full",
        "mappo_to_mappo_shared_stage1", "mappo_to_mappo_shared_stage2",
        "mappo_to_mappo_nonshared_stage2", "mappo_to_happo_nonshared_stage2",
    ]
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    results = {}
    for m in methods_to_load:
        all_intra, all_base, all_mag = [], [], []
        for seed in seeds:
            try:
                sds = load_state_dicts(map_name, m, seed)
                sds_pre = load_state_dicts(map_name, "mappo_to_mappo_shared_stage1", seed)
            except FileNotFoundError:
                continue
            vecs = [torch.cat([p.flatten() for p in sd.values()]).numpy() for sd in sds]
            base_vec = torch.cat([p.flatten() for p in sds_pre[0].values()]).numpy()
            intra = [cos(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i+1, len(vecs))]
            all_intra.append(np.mean(intra) if intra else 1.0)
            all_base.append(np.mean([cos(v, base_vec) for v in vecs]))
            all_mag.append(np.mean([float(np.linalg.norm(v - base_vec)) for v in vecs]))
        if not all_intra: continue
        results[SHORT[m]] = {
            "mean_intra_cos": float(np.mean(all_intra)),
            "std_intra_cos": float(np.std(all_intra)),
            "mean_base_cos": float(np.mean(all_base)),
            "std_base_cos": float(np.std(all_base)),
            "mean_magnitude": float(np.mean(all_mag)),
            "std_magnitude": float(np.std(all_mag)),
        }
        print(f"  {SHORT[m]:>12s}  intra={np.mean(all_intra):.4f}±{np.std(all_intra):.4f}  "
              f"base={np.mean(all_base):.4f}±{np.std(all_base):.4f}")
    return results


# MAIN

def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="1,2,3,4")
    parser.add_argument("--skip-interp", action="store_true")
    parser.add_argument("--skip-hotswap", action="store_true")
    parser.add_argument("--skip-spec", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=20)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file. Default: all_results_v3_{map}.json")
    args = parser.parse_args()

    global N_EPISODES, N_PARALLEL
    N_EPISODES = args.episodes
    N_PARALLEL = args.parallel

    master_file = args.output or os.path.join(RESULTS_DIR, f"all_results_v3_{args.map}.json")
    all_data = {}
    if os.path.exists(master_file):
        with open(master_file) as f:
            all_data = json.load(f)

    mn = args.map
    seeds = [int(s) for s in args.seeds.split(",")]
    if mn not in all_data:
        all_data[mn] = {}

    print(f"Config: map={mn}, seeds={seeds}, episodes={N_EPISODES}, parallel={N_PARALLEL}")

    all_data[mn]["weights"] = exp4_weights(mn, seeds)
    if not args.skip_interp:
        all_data[mn]["interpolation"] = exp1_interpolation(mn, seeds, N_PARALLEL)
    if not args.skip_hotswap:
        all_data[mn]["hotswap"] = exp2_hotswap(mn, seeds, N_PARALLEL)
    if not args.skip_spec:
        all_data[mn]["specialization"] = exp3_specialization(mn, seeds, N_PARALLEL)

    with open(master_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved to {master_file}")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for mn_ in all_data:
        for pid in sorted(all_data[mn_].get("interpolation", {})):
            r = all_data[mn_]["interpolation"][pid]
            s = "CONN" if r["connected"] else "BARR"
            print(f"  {mn_:>10s} {pid} B={r['mean_barrier']:.3f}±{r['std_barrier']:.3f} [{s}] {r['description']}")
        for hid in sorted(all_data[mn_].get("hotswap", {})):
            r = all_data[mn_]["hotswap"][hid]
            print(f"  {mn_:>10s} {hid} base={r['mean_base_wr']:.1%} swap={r['mean_swap_wr_overall']:.1%} drop={r['mean_drop']:+.1%}  {r['description']}")
        for label, r in all_data[mn_].get("specialization", {}).items():
            sp = "SPEC" if r["specialized"] else "SAME"
            print(f"  {mn_:>10s} SP[{label:>10s}] base={r['mean_base_wr']:.1%} drop={r['mean_swap_drop']:+.1%} [{sp}]")


if __name__ == "__main__":
    main()
