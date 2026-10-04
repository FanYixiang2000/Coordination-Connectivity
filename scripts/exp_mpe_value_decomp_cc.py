#!/usr/bin/env python3
"""VDN/QMIX coordination-connectivity pilot on PettingZoo MPE.

This script is independent from the HARL MAPPO/HAPPO MPE runs. It trains
simple value-decomposition agents on the same MPE scenarios, then evaluates
utility-network interpolation for same-source and independent-source paths.
"""

import argparse
import copy
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_DIR, "exp", "mpe_vd_cc")
os.makedirs(OUT_DIR, exist_ok=True)

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))

sys.path.insert(0, SCRIPT_DIR)
from exp_mpe_train_and_eval import ALPHAS, MPE_SCENARIOS, make_mpe_env  # noqa: E402


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class QMIXMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, mixing_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_dim = mixing_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, n_agents * mixing_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, mixing_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, 1),
        )

    def forward(self, q_values, state):
        batch = q_values.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.n_agents, self.mixing_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.mixing_dim)
        hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.mixing_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(batch)


class VDNMixer(nn.Module):
    def forward(self, q_values, state):
        del state
        return q_values.sum(dim=1)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []

    def add(self, transition: Dict):
        self.data.append(transition)
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.data), batch_size, replace=False)
        return [self.data[i] for i in idx]


@dataclass
class TrainConfig:
    scenario: str = "simple_reference_v2"
    seed: int = 1
    source_episodes: int = 2000
    branch_episodes: int = 1000
    eval_episodes: int = 50
    hidden_dim: int = 128
    mixing_dim: int = 32
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 256
    buffer_size: int = 50000
    epsilon_start: float = 1.0
    epsilon_branch: float = 0.10
    epsilon_end: float = 0.05
    epsilon_decay_updates: int = 10000
    target_update_interval: int = 200


class ValueDecompAgent:
    def __init__(self, algo: str, n_agents: int, obs_dim: int, state_dim: int, n_actions: int, cfg: TrainConfig):
        self.algo = algo
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.epsilon = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay_updates = cfg.epsilon_decay_updates
        self.target_update_interval = cfg.target_update_interval
        self.update_count = 0
        self.q_networks = nn.ModuleList([QNetwork(obs_dim, n_actions, cfg.hidden_dim) for _ in range(n_agents)])
        self.target_q_networks = nn.ModuleList([QNetwork(obs_dim, n_actions, cfg.hidden_dim) for _ in range(n_agents)])
        if algo == "qmix":
            self.mixer = QMIXMixer(n_agents, state_dim, cfg.mixing_dim)
            self.target_mixer = QMIXMixer(n_agents, state_dim, cfg.mixing_dim)
        elif algo == "vdn":
            self.mixer = VDNMixer()
            self.target_mixer = VDNMixer()
        else:
            raise ValueError(f"Unknown algo: {algo}")
        self.params = list(self.q_networks.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.update_targets()

    def update_targets(self):
        self.target_q_networks.load_state_dict(self.q_networks.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def act(self, obs_list, avail_list=None, deterministic=False):
        actions = []
        for i in range(self.n_agents):
            obs_t = torch.as_tensor(obs_list[i], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = self.q_networks[i](obs_t).squeeze(0)
            if avail_list is not None:
                avail = torch.as_tensor(avail_list[i], dtype=torch.bool)
                q = q.masked_fill(~avail, -1e9)
            if deterministic or np.random.rand() > self.epsilon:
                actions.append(int(torch.argmax(q).item()))
            else:
                if avail_list is None:
                    actions.append(int(np.random.randint(self.n_actions)))
                else:
                    legal = np.flatnonzero(np.asarray(avail_list[i]) > 0)
                    actions.append(int(np.random.choice(legal)))
        return actions

    def store(self, obs, state, actions, reward, next_obs, next_state, done):
        self.buffer.add(
            {
                "obs": [np.asarray(o, dtype=np.float32).copy() for o in obs],
                "state": np.asarray(state, dtype=np.float32).copy(),
                "actions": list(actions),
                "reward": float(reward),
                "next_obs": [np.asarray(o, dtype=np.float32).copy() for o in next_obs],
                "next_state": np.asarray(next_state, dtype=np.float32).copy(),
                "done": float(done),
            }
        )

    def can_update(self):
        return len(self.buffer) >= self.batch_size

    def update(self):
        if not self.can_update():
            return None
        batch = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(np.array([[t["obs"][i] for t in batch] for i in range(self.n_agents)]), dtype=torch.float32)
        next_obs = torch.as_tensor(np.array([[t["next_obs"][i] for t in batch] for i in range(self.n_agents)]), dtype=torch.float32)
        actions = torch.as_tensor(np.array([[t["actions"][i] for t in batch] for i in range(self.n_agents)]), dtype=torch.long)
        state = torch.as_tensor(np.array([t["state"] for t in batch]), dtype=torch.float32)
        next_state = torch.as_tensor(np.array([t["next_state"] for t in batch]), dtype=torch.float32)
        rewards = torch.as_tensor(np.array([t["reward"] for t in batch]), dtype=torch.float32)
        dones = torch.as_tensor(np.array([t["done"] for t in batch]), dtype=torch.float32)

        q_vals = []
        for i in range(self.n_agents):
            q_i = self.q_networks[i](obs[i]).gather(1, actions[i].unsqueeze(1)).squeeze(1)
            q_vals.append(q_i)
        q_vals = torch.stack(q_vals, dim=1)
        q_tot = self.mixer(q_vals, state)

        with torch.no_grad():
            target_q_vals = []
            for i in range(self.n_agents):
                target_q_i = self.target_q_networks[i](next_obs[i]).max(dim=1)[0]
                target_q_vals.append(target_q_i)
            target_q_vals = torch.stack(target_q_vals, dim=1)
            target_q_tot = self.target_mixer(target_q_vals, next_state)
            targets = rewards + self.gamma * (1.0 - dones) * target_q_tot

        loss = F.mse_loss(q_tot, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, 10.0)
        self.optimizer.step()
        self.update_count += 1
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / max(1, self.epsilon_decay_updates),
        )
        if self.update_count % self.target_update_interval == 0:
            self.update_targets()
        return float(loss.item())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dims(scenario: str):
    env = make_mpe_env(scenario)
    obs_dim = int(env.observation_space[0].shape[0])
    state_dim = int(env.share_observation_space[0].shape[0])
    n_actions = int(env.action_space[0].n)
    n_agents = int(env.n_agents)
    env.close()
    return n_agents, obs_dim, state_dim, n_actions


def q_state(agent: ValueDecompAgent):
    return copy.deepcopy(agent.q_networks.state_dict())


def full_state(agent: ValueDecompAgent):
    return {
        "q": copy.deepcopy(agent.q_networks.state_dict()),
        "mixer": copy.deepcopy(agent.mixer.state_dict()),
    }


def load_q_state(agent: ValueDecompAgent, state: Dict):
    agent.q_networks.load_state_dict(state)
    agent.target_q_networks.load_state_dict(state)


def load_full_state(agent: ValueDecompAgent, state: Dict):
    load_q_state(agent, state["q"])
    agent.mixer.load_state_dict(state["mixer"])
    agent.target_mixer.load_state_dict(state["mixer"])


def train_agent(agent: ValueDecompAgent, cfg: TrainConfig, episodes: int, seed: int, label: str):
    set_seed(seed)
    env = make_mpe_env(cfg.scenario)
    curve = []
    eval_interval = max(100, episodes // 4)
    for ep in range(episodes):
        obs, state_list, avail = env.reset()
        state = state_list[0]
        ep_return = 0.0
        for _ in range(MPE_SCENARIOS[cfg.scenario]["episode_length"]):
            actions = agent.act(obs, avail, deterministic=False)
            next_obs, next_state_list, rewards, dones, _, next_avail = env.step(np.array(actions))
            reward = float(rewards[0][0])
            done = bool(all(dones))
            agent.store(obs, state, actions, reward, next_obs, next_state_list[0], done)
            agent.update()
            ep_return += reward
            obs, state, avail = next_obs, next_state_list[0], next_avail
            if done:
                break
        if (ep + 1) % eval_interval == 0 or ep == episodes - 1:
            metrics = evaluate_agent(agent, cfg, max(10, cfg.eval_episodes // 2), seed + 100000)
            curve.append({"episode": ep + 1, **metrics})
            print(
                f"[{label}] ep={ep+1}/{episodes} eval_return={metrics['mean_return']:.3f} "
                f"eps={agent.epsilon:.3f}",
                flush=True,
            )
    env.close()
    return curve


@torch.no_grad()
def evaluate_agent(agent: ValueDecompAgent, cfg: TrainConfig, episodes: int, seed: int):
    env = make_mpe_env(cfg.scenario)
    env.seed(seed)
    returns = []
    for _ in range(episodes):
        obs, _, avail = env.reset()
        ep_return = 0.0
        for _ in range(MPE_SCENARIOS[cfg.scenario]["episode_length"]):
            actions = agent.act(obs, avail, deterministic=True)
            obs, _, rewards, dones, _, avail = env.step(np.array(actions))
            ep_return += float(rewards[0][0])
            if bool(all(dones)):
                break
        returns.append(ep_return)
    env.close()
    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}


def interpolate_q_states(state_a: Dict, state_b: Dict, alpha: float):
    return {k: (1.0 - alpha) * state_a[k] + alpha * state_b[k] for k in state_a}


def evaluate_path(name: str, algo: str, state_a: Dict, state_b: Dict, cfg: TrainConfig, dims):
    n_agents, obs_dim, state_dim, n_actions = dims
    points = []
    for alpha in ALPHAS:
        agent = ValueDecompAgent(algo, n_agents, obs_dim, state_dim, n_actions, cfg)
        load_q_state(agent, interpolate_q_states(state_a, state_b, alpha))
        metrics = evaluate_agent(agent, cfg, cfg.eval_episodes, cfg.seed + 200000 + int(alpha * 1000))
        points.append({"alpha": alpha, **metrics})
        print(f"[path {name}] alpha={alpha:.1f} return={metrics['mean_return']:.3f}", flush=True)
    vals = [p["mean_return"] for p in points]
    barrier = max(0.0, min(vals[0], vals[-1]) - min(vals))
    return {"name": name, "points": points, "return_barrier": float(barrier)}


def plot_results(result: Dict, png_path: str):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for path in result["paths"]:
        x = [p["alpha"] for p in path["points"]]
        y = [p["mean_return"] for p in path["points"]]
        label = f"{path['name']} (B={path['return_barrier']:.2f})"
        ax.plot(x, y, marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("Interpolation coefficient alpha")
    ax.set_ylabel("Episode return")
    ax.set_title(f"MPE Value-Decomposition CC: {result['scenario']} / {result['algo'].upper()}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    fig.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_algo(algo: str, cfg: TrainConfig):
    dims = make_dims(cfg.scenario)
    n_agents, obs_dim, state_dim, n_actions = dims
    print(f"\n=== {cfg.scenario} {algo.upper()} seed={cfg.seed} ===", flush=True)
    print(json.dumps(asdict(cfg), indent=2), flush=True)

    source = ValueDecompAgent(algo, n_agents, obs_dim, state_dim, n_actions, cfg)
    source_curve = train_agent(source, cfg, cfg.source_episodes, cfg.seed, f"{algo}-source")
    source_state = full_state(source)

    branch_a = ValueDecompAgent(algo, n_agents, obs_dim, state_dim, n_actions, cfg)
    load_full_state(branch_a, source_state)
    branch_a.epsilon = cfg.epsilon_branch
    curve_a = train_agent(branch_a, cfg, cfg.branch_episodes, cfg.seed + 101, f"{algo}-branch-A")

    branch_b = ValueDecompAgent(algo, n_agents, obs_dim, state_dim, n_actions, cfg)
    load_full_state(branch_b, source_state)
    branch_b.epsilon = cfg.epsilon_branch
    curve_b = train_agent(branch_b, cfg, cfg.branch_episodes, cfg.seed + 202, f"{algo}-branch-B")

    scratch = ValueDecompAgent(algo, n_agents, obs_dim, state_dim, n_actions, cfg)
    curve_scratch = train_agent(
        scratch, cfg, cfg.source_episodes + cfg.branch_episodes, cfg.seed + 303, f"{algo}-scratch"
    )

    q_a = q_state(branch_a)
    q_b = q_state(branch_b)
    q_s = q_state(scratch)
    paths = [
        evaluate_path(f"{algo.upper()} same-source A <-> B", algo, q_a, q_b, cfg, dims),
        evaluate_path(f"{algo.upper()} branch-A <-> scratch", algo, q_a, q_s, cfg, dims),
    ]
    result = {
        "scenario": cfg.scenario,
        "algo": algo,
        "config": asdict(cfg),
        "curves": {
            "source": source_curve,
            "branch_a": curve_a,
            "branch_b": curve_b,
            "scratch": curve_scratch,
        },
        "endpoint_eval": {
            "branch_a": evaluate_agent(branch_a, cfg, cfg.eval_episodes, cfg.seed + 300001),
            "branch_b": evaluate_agent(branch_b, cfg, cfg.eval_episodes, cfg.seed + 300002),
            "scratch": evaluate_agent(scratch, cfg, cfg.eval_episodes, cfg.seed + 300003),
        },
        "paths": paths,
    }
    prefix = f"mpe_vd_cc_{cfg.scenario}_{algo}_seed{cfg.seed}_src{cfg.source_episodes}_br{cfg.branch_episodes}"
    json_path = os.path.join(OUT_DIR, prefix + ".json")
    png_path = os.path.join(OUT_DIR, prefix + ".png")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    plot_results(result, png_path)
    print(f"Saved JSON: {json_path}", flush=True)
    print(f"Saved plot: {png_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="simple_reference_v2", choices=list(MPE_SCENARIOS.keys()))
    parser.add_argument("--algos", type=str, default="vdn,qmix")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source_episodes", type=int, default=2000)
    parser.add_argument("--branch_episodes", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epsilon_decay_updates", type=int, default=10000)
    parser.add_argument("--epsilon_branch", type=float, default=0.10)
    args = parser.parse_args()

    cfg = TrainConfig(
        scenario=args.scenario,
        seed=args.seed,
        source_episodes=args.source_episodes,
        branch_episodes=args.branch_episodes,
        eval_episodes=args.eval_episodes,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epsilon_decay_updates=args.epsilon_decay_updates,
        epsilon_branch=args.epsilon_branch,
    )
    algos = [x.strip() for x in args.algos.split(",") if x.strip()]
    for algo in algos:
        set_seed(cfg.seed)
        run_algo(algo, cfg)


if __name__ == "__main__":
    main()
