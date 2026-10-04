#!/usr/bin/env python3
"""
Coordination Connectivity v3 — Supplementary Experiments (Parallel)
====================================================================
Uses N_PARALLEL SC2 instances for ~20x speedup.

  1. Clone-All: copy agent i's weights to ALL positions → proves specialization
  2. Extra interpolation paths for full pairwise barrier matrix (P5-P7)
  3. Mixed-Team: half agents from method A, half from method B

Usage:
  python scripts/cc_v3_supplement.py --map 3s5z --seeds 1,2,3,4 --parallel 20
"""
import argparse, copy, glob, json, os, subprocess, sys, time, tempfile
import numpy as np
import torch
import torch.multiprocessing as mp

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

EXTRA_INTERP_PATHS = [
    ("P5", "mappo_to_mappo_shared_stage2", "happo_nonshared_full",
     "MAPPO-ct → HAPPO (shared-source vs independent)"),
    ("P6", "mappo_to_mappo_nonshared_stage2", "mappo_to_happo_nonshared_stage2",
     "MAPPO-ns → M2H (same source, both decoupled)"),
    ("P7", "mappo_to_mappo_nonshared_stage2", "happo_nonshared_full",
     "MAPPO-ns → HAPPO (decoupled vs independent)"),
]

CLONE_METHODS = [
    ("mappo_to_happo_nonshared_stage2",  "M2H"),
    ("mappo_to_mappo_nonshared_stage2",  "MAPPO-ns"),
    ("happo_nonshared_full",             "HAPPO"),
    ("mappo_to_mappo_shared_stage2",     "MAPPO-ct"),
]

MIXED_TEAM_PAIRS = [
    ("MT1", "mappo_to_happo_nonshared_stage2", "mappo_to_mappo_shared_stage2",
     "M2H + MAPPO-ct"),
    ("MT2", "mappo_to_mappo_nonshared_stage2", "mappo_to_mappo_shared_stage2",
     "MAPPO-ns + MAPPO-ct"),
    ("MT3", "happo_nonshared_full", "mappo_shared_full",
     "HAPPO + MAPPO"),
    ("MT4", "happo_nonshared_full", "mappo_to_mappo_shared_stage2",
     "HAPPO + MAPPO-ct"),
    ("MT5", "mappo_to_happo_nonshared_stage2", "mappo_to_mappo_nonshared_stage2",
     "M2H + MAPPO-ns"),
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
    """Worker: create own SC2 env, load actors, run n_eps episodes."""
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


# CLONE-ALL TEST

def exp_clone_all(map_name, seeds, n_parallel):
    print(f"\n{'='*70}")
    print(f"CLONE-ALL TEST (parallel={n_parallel})  map={map_name}")
    print(f"{'='*70}")
    n_agents = MAP_AGENTS[map_name]
    results = {}
    for method, label in CLONE_METHODS:
        print(f"\n  --- {label} ---")
        seed_base_wrs, seed_clone_wrs = [], []
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
            seed_base_wrs.append(base_wr)

            clone_wrs = []
            n_test = min(n_agents, 4)
            for i in range(n_test):
                sds_cloned = [copy.deepcopy(sds[i]) for _ in range(n_agents)]
                cl_wr = eval_wr_parallel(sds_cloned, map_name, n_parallel=n_parallel)
                clone_wrs.append(cl_wr)
                print(f" clone{i}:{cl_wr:.0%}", end="", flush=True)
            seed_clone_wrs.append(clone_wrs)
            dt = time.time() - t0
            print(f" ({dt:.0f}s)")

        if not seed_clone_wrs: continue
        clone_arr = np.array(seed_clone_wrs)
        base_arr = np.array(seed_base_wrs)
        results[label] = {
            "method": method,
            "mean_base_wr": float(base_arr.mean()),
            "std_base_wr": float(base_arr.std()),
            "mean_clone_wr": float(clone_arr.mean()),
            "std_clone_wr": float(clone_arr.std()),
            "mean_clone_per_agent": clone_arr.mean(axis=0).tolist(),
            "mean_drop": float(base_arr.mean() - clone_arr.mean()),
            "n_seeds": len(seed_clone_wrs),
            "specialized": float(base_arr.mean() - clone_arr.mean()) > 0.10,
        }
        sp = "SPECIALIZED" if results[label]["specialized"] else "NOT SPECIALIZED"
        print(f"    → base={base_arr.mean():.1%}  clone={clone_arr.mean():.1%}  drop={base_arr.mean()-clone_arr.mean():+.1%}  [{sp}]")
    return results


# EXTRA INTERPOLATION

def exp_extra_interp(map_name, seeds, n_parallel):
    print(f"\n{'='*70}")
    print(f"EXTRA INTERPOLATION (parallel={n_parallel})  map={map_name}")
    print(f"{'='*70}")
    results = {}
    for path_id, method_a, method_b, desc in EXTRA_INTERP_PATHS:
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
            "per_seed_wr": [w for w in seed_wrs],
            "mean_barrier": float(np.mean(seed_barriers)),
            "std_barrier": float(np.std(seed_barriers)),
            "per_seed_barriers": seed_barriers,
            "connected": float(np.mean(seed_barriers)) < 0.1,
            "n_seeds": len(seed_wrs),
        }
        s = "CONNECTED" if results[path_id]["connected"] else "BARRIER"
        print(f"    → B={np.mean(seed_barriers):.3f}±{np.std(seed_barriers):.3f} [{s}]")
    return results


# MIXED-TEAM

def exp_mixed_team(map_name, seeds, n_parallel):
    print(f"\n{'='*70}")
    print(f"MIXED-TEAM TEST (parallel={n_parallel})  map={map_name}")
    print(f"{'='*70}")
    n_agents = MAP_AGENTS[map_name]
    half = n_agents // 2
    results = {}
    for mt_id, method_a, method_b, desc in MIXED_TEAM_PAIRS:
        print(f"\n  --- {mt_id}: {desc} ---")
        seed_wrs = []
        for seed in seeds:
            print(f"    [seed={seed}]", end="", flush=True)
            t0 = time.time()
            try:
                sds_a = load_state_dicts(map_name, method_a, seed)
                sds_b = load_state_dicts(map_name, method_b, seed)
            except FileNotFoundError as e:
                print(f" SKIP: {e}")
                continue
            sds_mixed = [copy.deepcopy(sds_a[i]) if i < half
                         else copy.deepcopy(sds_b[i]) for i in range(n_agents)]
            wr = eval_wr_parallel(sds_mixed, map_name, n_parallel=n_parallel)
            seed_wrs.append(wr)
            dt = time.time() - t0
            print(f" wr={wr:.0%} ({dt:.0f}s)")
        if not seed_wrs: continue
        arr = np.array(seed_wrs)
        results[mt_id] = {
            "description": desc,
            "method_a": SHORT[method_a], "method_b": SHORT[method_b],
            "mean_wr": float(arr.mean()),
            "std_wr": float(arr.std()),
            "per_seed_wr": [float(w) for w in seed_wrs],
            "n_seeds": len(seed_wrs),
        }
        print(f"    → wr={arr.mean():.1%}±{arr.std():.1%}")
    return results


# MAIN

def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="1,2,3,4")
    parser.add_argument("--experiments", type=str, default="clone_all,extra_interp,mixed_team")
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
    exps = [e.strip() for e in args.experiments.split(",")]
    if mn not in all_data:
        all_data[mn] = {}

    print(f"Config: map={mn}, seeds={seeds}, episodes={N_EPISODES}, parallel={N_PARALLEL}")

    if "clone_all" in exps:
        all_data[mn]["clone_all"] = exp_clone_all(mn, seeds, N_PARALLEL)
    if "extra_interp" in exps:
        extra = exp_extra_interp(mn, seeds, N_PARALLEL)
        if "interpolation" not in all_data[mn]:
            all_data[mn]["interpolation"] = {}
        all_data[mn]["interpolation"].update(extra)
    if "mixed_team" in exps:
        all_data[mn]["mixed_team"] = exp_mixed_team(mn, seeds, N_PARALLEL)

    with open(master_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved to {master_file}")

    print(f"\n{'='*70}\nSUPPLEMENT SUMMARY for {mn}\n{'='*70}")
    if "clone_all" in all_data[mn]:
        for label, r in all_data[mn]["clone_all"].items():
            sp = "SPEC" if r["specialized"] else "SAME"
            print(f"  Clone-All [{label:>10s}] base={r['mean_base_wr']:.1%} clone={r['mean_clone_wr']:.1%} [{sp}]")
    for pid in ["P5", "P6", "P7"]:
        if pid in all_data[mn].get("interpolation", {}):
            r = all_data[mn]["interpolation"][pid]
            s = "CONN" if r["connected"] else "BARR"
            print(f"  Interp {pid} B={r['mean_barrier']:.3f}±{r['std_barrier']:.3f} [{s}] {r['description']}")
    if "mixed_team" in all_data[mn]:
        for mt_id, r in all_data[mn]["mixed_team"].items():
            print(f"  Mixed {mt_id} wr={r['mean_wr']:.1%}±{r['std_wr']:.1%} {r['description']}")


if __name__ == "__main__":
    main()
