from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class BanditEnvConfig:
    """Base config for 2D distribution fitting in [-1, 1]^2."""
    num_pre_sampled: int = 4096
    kernel_bandwidth: float = 0.1
    noise_std: float = 0.03
    reward_scale: float = 1.0
    obs_dim: int = 1

    def sample_points(self, rng: np.random.Generator, n: int) -> np.ndarray:
        raise NotImplementedError


@dataclass
class LemniscateConfig(BanditEnvConfig):
    a: float = 0.9

    def sample_points(self, rng: np.random.Generator, n: int) -> np.ndarray:
        t = rng.uniform(0.0, 2.0 * np.pi, size=n)
        denom = 1.0 + np.sin(t) ** 2
        x = self.a * np.cos(t) / denom
        y = self.a * np.sin(t) * np.cos(t) / denom
        pts = np.stack([x, y], axis=1)
        if self.noise_std > 0:
            pts = pts + rng.normal(scale=self.noise_std, size=pts.shape)
        return np.clip(pts, -1.0, 1.0)


@dataclass
class TwoMoonsConfig(BanditEnvConfig):
    def sample_points(self, rng: np.random.Generator, n: int) -> np.ndarray:
        n1 = n // 2
        n2 = n - n1
        t1 = rng.uniform(0.0, np.pi, size=n1)
        t2 = rng.uniform(0.0, np.pi, size=n2)
        radius = 0.55
        moon1 = np.stack([np.cos(t1), np.sin(t1)], axis=1) * radius + np.array([-0.25, 0.0])
        moon2 = np.stack([np.cos(t2), -np.sin(t2)], axis=1) * radius + np.array([0.25, 0.0])
        pts = np.concatenate([moon1, moon2], axis=0)
        if self.noise_std > 0:
            pts = pts + rng.normal(scale=self.noise_std, size=pts.shape)
        return np.clip(pts, -1.0, 1.0)


class BanditEnv(gym.Env):
    """
    One-step bandit environment for 2D distribution fitting.

    Observation:
        - Constant vector in [-1, 1] (default dimension 1).

    Action:
        - 2D action in [-1, 1]^2.

    Reward:
        - miniexp(-min distance^2 / (2*sigma^2)) to the target distribution point cloud.
    """

    metadata = {"render_modes": ["matplotlib"], "render_fps": 4}

    def __init__(
        self,
        task: Literal["lemniscate", "twomoons"] = "lemniscate",
        config: Optional[BanditEnvConfig] = None,
        max_steps: int = 1,
    ):
        super().__init__()
        if config is None:
            if task == "lemniscate":
                config = LemniscateConfig()
            elif task == "twomoons":
                config = TwoMoonsConfig()
            else:
                raise ValueError(f"Invalid task: {task}")

        self.config = config
        self.max_steps = max_steps
        self.obs_dim = config.obs_dim
        self.action_dim = 2

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self.np_random = np.random.default_rng()
        self.target_points = self.config.sample_points(
            self.np_random, self.config.num_pre_sampled
        )
        self.steps = 0
        self.last_action: Optional[np.ndarray] = None

        self._obs = np.zeros(self.obs_dim, dtype=np.float32)
        self._fig = None
        self._ax = None

    def _get_obs(self) -> np.ndarray:
        return self._obs.copy()

    def compute_reward(self, action: np.ndarray) -> float:
        action = np.asarray(action, dtype=float).reshape(1, -1)
        diffs = self.target_points[None, :, :] - action[:, None, :]
        sq_dist = np.sum(diffs ** 2, axis=-1)
        d2 = float(np.min(sq_dist))
        sigma = float(self.config.kernel_bandwidth)
        reward = self.config.reward_scale * np.exp(-d2 / (2.0 * sigma ** 2))
        return float(reward)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.target_points = self.config.sample_points(
                self.np_random, self.config.num_pre_sampled
            )
        self.steps = 0
        self.last_action = None
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)
        reward = self.compute_reward(action)
        self.last_action = action.copy()

        self.steps += 1
        terminated = self.steps >= self.max_steps
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def render(
        self,
        action_samples: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
        *_args,
        **_kwargs,
    ):
        if self._fig is None or self._ax is None:
            plt.ion()
            self._fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            self._ax = np.asarray(axes)

        if not isinstance(self._ax, np.ndarray) or self._ax.size != 2:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            return self.render(action_samples=action_samples, save_path=save_path)

        left_ax, right_ax = self._ax
        left_ax.clear()
        right_ax.clear()
        points = self.target_points
        if points.shape[0] > 2000:
            idx = self.np_random.choice(points.shape[0], size=2000, replace=False)
            points = points[idx]
        left_ax.scatter(points[:, 0], points[:, 1], s=8, c="#8ecae6", alpha=0.6)

        if action_samples is not None:
            action_samples = np.asarray(action_samples, dtype=float)
            if action_samples.ndim == 1:
                action_samples = action_samples.reshape(1, -1)
            if action_samples.shape[0] > 1500:
                idx = self.np_random.choice(action_samples.shape[0], size=1500, replace=False)
                action_samples = action_samples[idx]
            right_ax.scatter(
                action_samples[:, 0],
                action_samples[:, 1],
                s=12,
                c="#ffb703",
                alpha=0.6,
                edgecolors="none",
                zorder=2,
            )
        elif self.last_action is not None:
            right_ax.scatter(
                self.last_action[0],
                self.last_action[1],
                s=80,
                c="#fb8500",
                edgecolors="#023047",
                linewidths=1.5,
                zorder=3,
            )

        for ax in (left_ax, right_ax):
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal", "box")

        self._fig.canvas.draw()
        if save_path is not None:
            self._fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            plt.ioff()
            plt.close(self._fig)
        self._fig = None
        self._ax = None
