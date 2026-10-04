"""
Unified robustness evaluator for SMAC experiments.

Loads trained actor models and runs 100-episode evaluations with
optional perturbations. Saves per-episode results to JSON.
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
import torch
from pathlib import Path
from typing import Any, Optional

import sys
HARL_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, HARL_ROOT)

from harl.envs.smac.StarCraft2_Env import StarCraft2Env
from harl.algorithms.actors import ALGO_REGISTRY
from harl.utils.envs_tools import get_num_agents
from harl.envs.smac.smac_maps import get_map_params
from harl_ext.perturbations import ObsMissing, ObsNoise, ActionDelay


def _load_actors(
    model_dir: str,
    algo_name: str,
    share_param: bool,
    obs_space,
    act_space,
    num_agents: int,
    hidden_sizes: list,
    algo_args_algo: dict,
    algo_args_model: dict,
    device: torch.device,
):
    """Load actor models from checkpoint directory."""
    merged_args = {**algo_args_model, **algo_args_algo}
    merged_args["hidden_sizes"] = hidden_sizes

    actors = []
    if share_param:
        agent = ALGO_REGISTRY[algo_name](
            merged_args, obs_space[0], act_space[0], device=device,
        )
        sd = torch.load(os.path.join(model_dir, "actor_agent0.pt"), map_location=device)
        agent.actor.load_state_dict(sd)
        agent.prep_rollout()
        for _ in range(num_agents):
            actors.append(agent)
    else:
        for agent_id in range(num_agents):
            agent = ALGO_REGISTRY[algo_name](
                merged_args, obs_space[agent_id], act_space[agent_id], device=device,
            )
            sd = torch.load(
                os.path.join(model_dir, f"actor_agent{agent_id}.pt"),
                map_location=device,
            )
            agent.actor.load_state_dict(sd)
            agent.prep_rollout()
            actors.append(agent)
    return actors


def evaluate_model(
    model_dir: str,
    map_name: str,
    algo_name: str,
    share_param: bool,
    algo_args: dict,
    num_episodes: int = 100,
    perturbation: dict | None = None,
    eval_seed: int = 99999,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Run evaluation episodes and return detailed results.

    perturbation: dict with 'type' key and type-specific params.
        e.g. {"type": "clean"}, {"type": "missing", "missing_rate": 0.2},
             {"type": "noise", "sigma": 0.1}, {"type": "delay", "delay": 2}
    """
    if perturbation is None:
        perturbation = {"type": "clean"}

    device = torch.device(device_str)
    map_params = get_map_params(map_name)
    n_agents = map_params["n_agents"]
    episode_limit = map_params["limit"]

    state_type = algo_args.get("env_args", {}).get("state_type", "EP")
    if isinstance(algo_args.get("env_args"), dict):
        state_type = algo_args["env_args"].get("state_type", "EP")

    env_args = {"map_name": map_name, "state_type": state_type}
    env = StarCraft2Env(env_args, seed=eval_seed)

    # StarCraft2Env.observation_space is a list of ints (obs sizes)
    # StarCraft2Env.action_space is a list of gym.spaces.Discrete
    obs_space = env.observation_space
    act_space = env.action_space

    aa_model = algo_args.get("algo_args", algo_args).get("model", {})
    aa_algo = algo_args.get("algo_args", algo_args).get("algo", {})
    hidden_sizes = aa_model.get("hidden_sizes", [128, 128])
    recurrent_n = aa_model.get("recurrent_n", 1)
    use_recurrent = aa_model.get("use_recurrent_policy", False)
    rnn_hidden_size = hidden_sizes[-1]

    # For _load_actors, obs_space entries can be int (obs_size);
    # HARL's ALGO_REGISTRY accepts raw int obs_space for SMAC
    if share_param:
        os_list = [obs_space[0]] * n_agents
        as_list = [act_space[0]] * n_agents
    else:
        os_list = obs_space
        as_list = act_space

    actors = _load_actors(
        model_dir=model_dir,
        algo_name=algo_name,
        share_param=share_param,
        obs_space=os_list,
        act_space=as_list,
        num_agents=n_agents,
        hidden_sizes=hidden_sizes,
        algo_args_algo=aa_algo,
        algo_args_model=aa_model,
        device=device,
    )

    rng = np.random.RandomState(eval_seed)
    ptype = perturbation["type"]

    obs_perturb = None
    action_delay = None
    if ptype == "missing":
        obs_perturb = ObsMissing(perturbation["missing_rate"], rng)
    elif ptype == "noise":
        obs_perturb = ObsNoise(perturbation["sigma"], rng)
    elif ptype == "delay":
        action_delay = ActionDelay(perturbation["delay"], 1, n_agents)

    episode_results = []
    wins = 0

    for ep in range(num_episodes):
        # StarCraft2Env.reset() → (local_obs, global_state, avail_actions)
        obs_list, _state, avail_actions = env.reset()
        if action_delay is not None:
            action_delay.reset_thread(0)

        rnn_states = np.zeros((n_agents, recurrent_n, rnn_hidden_size), dtype=np.float32)
        masks = np.ones((n_agents, 1), dtype=np.float32)

        ep_reward = 0.0
        ep_length = 0
        won = False

        for t in range(episode_limit):
            obs_array = np.array(obs_list, dtype=np.float32)
            if obs_perturb is not None:
                obs_array = obs_perturb.apply(obs_array)

            avail_array = np.array(avail_actions, dtype=np.float32)

            actions_list = []
            new_rnn_states = []
            for agent_id in range(n_agents):
                obs_t = torch.FloatTensor(obs_array[agent_id:agent_id+1]).to(device)
                rnn_t = torch.FloatTensor(rnn_states[agent_id:agent_id+1]).to(device)
                mask_t = torch.FloatTensor(masks[agent_id:agent_id+1]).to(device)
                avail_t = torch.FloatTensor(avail_array[agent_id:agent_id+1]).to(device)

                with torch.no_grad():
                    action, new_rnn = actors[agent_id].act(
                        obs_t, rnn_t, mask_t, avail_t, deterministic=True,
                    )
                actions_list.append(action.cpu().numpy()[0])
                new_rnn_states.append(new_rnn.cpu().numpy()[0])

            rnn_states = np.array(new_rnn_states)
            actions = np.array(actions_list)

            if action_delay is not None:
                actions_3d = actions.reshape(1, n_agents, -1)
                actions_3d = action_delay.apply(actions_3d)
                actions = actions_3d[0]

            actions_int = actions.flatten().tolist()

            # Use avail_actions from the current obs to project illegal actions
            for i in range(n_agents):
                a = int(actions_int[i])
                avail_i = avail_actions[i] if isinstance(avail_actions[i], list) \
                    else avail_actions[i].tolist()
                if len(avail_i) > a and avail_i[a] == 0:
                    legal = [j for j, v in enumerate(avail_i) if v == 1]
                    actions_int[i] = legal[0] if legal else 0

            # StarCraft2Env.step() → (local_obs, global_state, rewards, dones, infos, avail_actions)
            obs_list, _state, rewards, dones, infos, avail_actions = env.step(actions_int)
            step_reward = rewards[0][0] if isinstance(rewards[0], (list, np.ndarray)) else rewards[0]
            ep_reward += step_reward
            ep_length += 1

            terminated = all(dones) if hasattr(dones, '__iter__') else dones
            if terminated:
                won = infos[0].get("won", False) if infos else False
                break

        if won:
            wins += 1

        episode_results.append({
            "episode": ep,
            "reward": float(ep_reward),
            "length": ep_length,
            "win": won,
        })

    env.close()

    win_rate = wins / num_episodes
    rewards = [r["reward"] for r in episode_results]
    lengths = [r["length"] for r in episode_results]
    win_array = [1.0 if r["win"] else 0.0 for r in episode_results]

    se = np.std(win_array) / np.sqrt(num_episodes)
    ci95 = 1.96 * se

    result = {
        "perturbation": perturbation,
        "num_episodes": num_episodes,
        "win_rate": win_rate,
        "win_rate_std": float(np.std(win_array)),
        "win_rate_stderr": float(se),
        "win_rate_ci95_low": max(0, win_rate - ci95),
        "win_rate_ci95_high": min(1, win_rate + ci95),
        "wins": wins,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_episode_length": float(np.mean(lengths)),
        "std_episode_length": float(np.std(lengths)),
        "episodes": episode_results,
        "model_dir": model_dir,
        "map_name": map_name,
        "algo_name": algo_name,
        "share_param": share_param,
        "eval_seed": eval_seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result


def save_eval_result(result: dict, output_path: str):
    """Save evaluation result to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {output_path}")
