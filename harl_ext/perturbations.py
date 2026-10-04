"""
Unified perturbation implementations for robustness evaluation.

Three perturbation types:
  1. Observation missing (zero-masking random features)
  2. Observation noise (additive Gaussian)
  3. Action delay (buffered execution with warm-up)

All perturbations operate on numpy arrays during evaluation only.
"""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional


class ObsMissing:
    """Zero-mask a random fraction of observation features."""

    def __init__(self, missing_rate: float, rng: np.random.RandomState):
        assert 0.0 <= missing_rate <= 1.0
        self.missing_rate = missing_rate
        self.rng = rng

    def apply(self, obs: np.ndarray) -> np.ndarray:
        """obs shape: (n_threads, obs_dim) or (obs_dim,)"""
        if self.missing_rate == 0.0:
            return obs
        obs = obs.copy()
        mask = self.rng.random(obs.shape) < self.missing_rate
        obs[mask] = 0.0
        return obs


class ObsNoise:
    """Add Gaussian noise to observations."""

    def __init__(self, sigma: float, rng: np.random.RandomState):
        assert sigma >= 0.0
        self.sigma = sigma
        self.rng = rng

    def apply(self, obs: np.ndarray) -> np.ndarray:
        if self.sigma == 0.0:
            return obs
        obs = obs.copy()
        noise = self.rng.normal(0.0, self.sigma, size=obs.shape).astype(obs.dtype)
        return obs + noise


class ActionDelay:
    """
    Buffer-based action delay with warm-up semantics.

    During the first `delay` steps, the current action is executed (warm-up).
    After that, the delayed action from the buffer is executed.
    Buffer stores discrete action IDs (integers), not logits.
    """

    def __init__(self, delay: int, n_threads: int, n_agents: int):
        assert delay >= 0
        self.delay = delay
        self.n_threads = n_threads
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.buffers = [
            [deque(maxlen=self.delay + 1) for _ in range(self.n_agents)]
            for _ in range(self.n_threads)
        ]
        self.step_count = [0] * self.n_threads

    def reset_thread(self, thread_id: int):
        for a in range(self.n_agents):
            self.buffers[thread_id][a].clear()
        self.step_count[thread_id] = 0

    def apply(self, actions: np.ndarray) -> np.ndarray:
        """
        actions shape: (n_threads, n_agents, action_dim)
        Returns delayed actions with the same shape.
        """
        if self.delay == 0:
            return actions
        out = actions.copy()
        for t in range(self.n_threads):
            for a in range(self.n_agents):
                self.buffers[t][a].append(actions[t, a].copy())
                if self.step_count[t] < self.delay:
                    out[t, a] = actions[t, a]
                else:
                    out[t, a] = self.buffers[t][a][0]
            self.step_count[t] += 1
        return out


def get_perturbation_configs():
    """Return the full set of perturbation conditions for evaluation."""
    conditions = [{"type": "clean"}]

    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        conditions.append({"type": "missing", "missing_rate": rate})

    for sigma in [0.05, 0.10, 0.20, 0.30, 0.40]:
        conditions.append({"type": "noise", "sigma": sigma})

    for delay in [1, 2, 3, 4]:
        conditions.append({"type": "delay", "delay": delay})

    return conditions
