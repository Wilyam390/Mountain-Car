"""Utilities for evaluation and visualization of continuous Mountain Car agents."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _action_scalar(action) -> float:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    return float(arr[0])


def run_continuous_episode(
    env,
    model,
    *,
    deterministic: bool = True,
    seed: int | None = None,
) -> Dict:
    """Roll out one episode and collect reward decomposition + trajectory info."""
    obs, _ = env.reset(seed=seed)

    positions = [float(obs[0])]
    velocities = [float(obs[1])]
    actions = []

    engineered_return = 0.0
    base_return = 0.0
    fuel_cost_total = 0.0
    quadratic_cost_total = 0.0
    time_cost_total = 0.0
    non_null_count = 0
    action_abs_total = 0.0
    steps = 0
    success = 0

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        base_reward = float(info.get("base_reward", reward))
        engineered_reward = float(info.get("engineered_reward", reward))
        fuel_cost = float(info.get("fuel_cost", 0.0))
        quadratic_cost = float(info.get("quadratic_cost", 0.0))
        time_cost = float(info.get("time_cost", 0.0))
        non_null = int(info.get("non_null_action", 0))
        action_abs = float(info.get("action_abs", abs(_action_scalar(action))))

        engineered_return += engineered_reward
        base_return += base_reward
        fuel_cost_total += fuel_cost
        quadratic_cost_total += quadratic_cost
        time_cost_total += time_cost
        non_null_count += non_null
        action_abs_total += action_abs
        steps += 1
        success = int(terminated)

        actions.append(_action_scalar(action))
        positions.append(float(next_obs[0]))
        velocities.append(float(next_obs[1]))
        obs = next_obs

    return {
        "engineered_return": engineered_return,
        "base_return": base_return,
        "fuel_cost": fuel_cost_total,
        "quadratic_cost": quadratic_cost_total,
        "time_cost": time_cost_total,
        "non_null_count": non_null_count,
        "mean_action_abs": action_abs_total / max(1, steps),
        "steps": steps,
        "success": success,
        "positions": np.asarray(positions, dtype=np.float32),
        "velocities": np.asarray(velocities, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
    }


def evaluate_continuous_model(
    env_factory: Callable[[], object],
    model,
    *,
    n_episodes: int = 30,
    deterministic: bool = True,
    seed: int | None = 123,
) -> Dict[str, float]:
    """Evaluate a model over many episodes using a fresh environment."""
    env = env_factory()
    try:
        episodes = []
        for idx in range(n_episodes):
            ep_seed = None if seed is None else int(seed + idx)
            episodes.append(
                run_continuous_episode(
                    env,
                    model,
                    deterministic=deterministic,
                    seed=ep_seed,
                )
            )
    finally:
        env.close()

    summary = {
        "n_episodes": n_episodes,
        "mean_engineered_return": float(np.mean([ep["engineered_return"] for ep in episodes])),
        "std_engineered_return": float(np.std([ep["engineered_return"] for ep in episodes])),
        "mean_base_return": float(np.mean([ep["base_return"] for ep in episodes])),
        "mean_steps": float(np.mean([ep["steps"] for ep in episodes])),
        "success_rate": float(np.mean([ep["success"] for ep in episodes])),
        "mean_fuel_cost": float(np.mean([ep["fuel_cost"] for ep in episodes])),
        "mean_quadratic_cost": float(np.mean([ep["quadratic_cost"] for ep in episodes])),
        "mean_time_cost": float(np.mean([ep["time_cost"] for ep in episodes])),
        "mean_non_null_count": float(np.mean([ep["non_null_count"] for ep in episodes])),
        "mean_action_abs": float(np.mean([ep["mean_action_abs"] for ep in episodes])),
    }
    return summary


def collect_model_trajectories(
    env_factory: Callable[[], object],
    model,
    *,
    n_episodes: int = 8,
    deterministic: bool = True,
    seed: int | None = 1000,
) -> List[Dict]:
    """Collect a small bundle of trajectories for a phase portrait figure."""
    env = env_factory()
    trajectories: List[Dict] = []
    try:
        for idx in range(n_episodes):
            ep_seed = None if seed is None else int(seed + idx)
            trajectories.append(
                run_continuous_episode(
                    env,
                    model,
                    deterministic=deterministic,
                    seed=ep_seed,
                )
            )
    finally:
        env.close()
    return trajectories


def plot_eval_curve_from_npz(npz_path: str | Path, *, title: str = "Evaluation curve", ax=None):
    """Plot EvalCallback results saved by Stable-Baselines3."""
    data = np.load(npz_path)
    timesteps = np.asarray(data["timesteps"]).reshape(-1)
    results = np.asarray(data["results"])
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, mean_rewards, label="Mean eval reward")
    ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Eval reward")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_action_surface(
    model,
    *,
    low: np.ndarray,
    high: np.ndarray,
    n_pos: int = 100,
    n_vel: int = 100,
    deterministic: bool = True,
    title: str = "Policy action surface",
    ax=None,
):
    """Visualize the deterministic action selected over a state grid."""
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    pos_values = np.linspace(low[0], high[0], n_pos)
    vel_values = np.linspace(low[1], high[1], n_vel)

    action_grid = np.zeros((n_pos, n_vel), dtype=np.float32)
    for i, position in enumerate(pos_values):
        for j, velocity in enumerate(vel_values):
            obs = np.array([position, velocity], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=deterministic)
            action_grid[i, j] = _action_scalar(action)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        action_grid.T,
        origin="lower",
        aspect="auto",
        extent=[low[0], high[0], low[1], high[1]],
        vmin=-1.0,
        vmax=1.0,
    )
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("Action")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title)
    return ax


def plot_action_histogram(trajectories: List[Dict], *, title: str = "Action histogram", ax=None):
    """Quick sanity-check plot: did the agent learn a non-trivial action distribution?"""
    actions = np.concatenate([traj["actions"] for traj in trajectories]) if trajectories else np.array([])
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(actions, bins=41)
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return ax
