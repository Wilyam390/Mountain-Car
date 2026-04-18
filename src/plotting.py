"""
═══════════════════════════════════════════════════════════════════════════════
PLOTTING UTILITIES FOR MOUNTAIN CAR RL EXPERIMENTS

Standardized matplotlib visualization functions for:
  - Training curves and learning progress
  - Policy visualization (heatmaps, phase portraits)
  - Value function analysis (3D surfaces)
  - State visitation patterns
  - Trajectory analysis

All functions support optional matplotlib axes for subplot integration.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .environment_utils import StateDiscretizer


# ═══════════════════════════════════════════════════════════════════════════════
# SMOOTHING & AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def moving_average(values: Sequence[float], window: int = 100) -> np.ndarray:
    """
    Compute moving average using convolution.

    Args:
        values: Sequence of values
        window: Size of moving window

    Returns:
        Moving average values (shorter than input by window-1)
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values.copy()
    return np.convolve(values, np.ones(window, dtype=np.float32) / window, mode="valid")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING PROGRESS VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curve(
    values: Sequence[float],
    *,
    window: int = 100,
    title: str = "Training curve",
    ylabel: str = "Value",
    ax: Optional[Any] = None,
) -> Any:
    """Plot training progress with moving average overlay."""
    values = np.asarray(values, dtype=np.float32)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(values, alpha=0.35, label="Episode value")
    ax.plot(moving_average(values, window=window), label=f"{window}-episode MA", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_success_curve(
    successes: Sequence[int],
    *,
    window: int = 100,
    title: str = "Success rate",
    ax: Optional[Any] = None,
) -> Any:
    """Plot rolling success rate."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    successes_float = np.asarray(successes, dtype=np.float32)
    ma = moving_average(successes_float, window=window)
    ax.plot(ma * 100, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_policy_map(
    q_table: np.ndarray,
    discretizer: StateDiscretizer,
    *,
    visit_counts: Optional[np.ndarray] = None,
    mask_unvisited: bool = True,
    title: str = "Learned greedy policy",
    ax: Optional[Any] = None,
) -> Any:
    """Visualize learned policy as heatmap."""
    policy = np.argmax(q_table, axis=2).astype(np.float32)

    if visit_counts is not None and mask_unvisited:
        policy = np.ma.masked_where(visit_counts <= 0, policy)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        policy.T,
        origin="lower",
        aspect="auto",
        extent=[
            discretizer.pos_range[0],
            discretizer.pos_range[1],
            discretizer.vel_range[0],
            discretizer.vel_range[1],
        ],
        cmap="RdYlGn",
        vmin=0,
        vmax=2,
    )

    cbar = plt.colorbar(image, ax=ax)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["Push Left", "No Action", "Push Right"])

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(title)

    return ax


def plot_visitation_heatmap(
    visit_counts: np.ndarray,
    discretizer: StateDiscretizer,
    *,
    log_scale: bool = True,
    title: str = "State visitation frequency",
    ax: Optional[Any] = None,
) -> Any:
    """Visualize which states were visited during training."""
    values = np.log1p(visit_counts) if log_scale else visit_counts

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        values.T,
        origin="lower",
        aspect="auto",
        extent=[
            discretizer.pos_range[0],
            discretizer.pos_range[1],
            discretizer.vel_range[0],
            discretizer.vel_range[1],
        ],
        cmap="viridis",
    )

    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("log(1 + visits)" if log_scale else "visits")

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(title)

    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# VALUE FUNCTION VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_value_surface(
    q_table: np.ndarray,
    discretizer: StateDiscretizer,
    *,
    mode: str = "max",
    action: Optional[int] = None,
    title: str = "Value function surface",
    elev: int = 30,
    azim: int = -60,
) -> Any:
    """Visualize value function as 3D surface."""
    if mode == "max":
        values = np.max(q_table, axis=2)
    elif mode == "action":
        if action is None:
            raise ValueError("action must be provided when mode='action'")
        values = q_table[:, :, action]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'max' or 'action'")

    pos_centers = 0.5 * (discretizer.pos_bins[:-1] + discretizer.pos_bins[1:])
    vel_centers = 0.5 * (discretizer.vel_bins[:-1] + discretizer.vel_bins[1:])
    X, Y = np.meshgrid(pos_centers, vel_centers, indexing="ij")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, values, rstride=2, cstride=2, alpha=0.8)

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_zlabel("Q-value")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY COLLECTION & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_greedy_trajectories(
    env: gym.Env,
    q_table: np.ndarray,
    discretizer: StateDiscretizer,
    *,
    n_episodes: int = 10,
    seed: Optional[int] = 42,
) -> List[dict]:
    """Collect multiple greedy episodes for visualization."""
    trajectories = []

    for episode_idx in range(n_episodes):
        ep_seed = None if seed is None else int(seed + episode_idx)

        state, _ = env.reset(seed=ep_seed)
        positions = [state[0]]
        velocities = [state[1]]
        actions = []

        done = False
        max_steps = 200

        while not done and len(positions) < max_steps + 1:
            state_disc = discretizer.discretize(state)
            action = int(np.argmax(q_table[state_disc]))
            actions.append(action)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            positions.append(state[0])
            velocities.append(state[1])

        trajectories.append(
            {
                "positions": np.array(positions),
                "velocities": np.array(velocities),
                "actions": np.array(actions),
                "success": bool(done),
                "steps": len(actions),
            }
        )

    return trajectories


def plot_phase_portrait(
    trajectories: Sequence[dict],
    *,
    title: str = "Greedy trajectories in phase space",
    ax: Optional[Any] = None,
) -> Any:
    """Plot trajectories in position-velocity phase space."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))

    for episode, color in zip(trajectories, colors):
        ax.plot(
            episode["positions"],
            episode["velocities"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            alpha=0.7,
            color=color,
        )

        ax.plot(episode["positions"][0], episode["velocities"][0], "g*", markersize=12)
        ax.plot(episode["positions"][-1], episode["velocities"][-1], "r*", markersize=12)

    ax.axvline(x=0.6, color="red", linestyle="--", alpha=0.5, linewidth=2, label="Goal")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_method_comparison(
    method_results: dict,
    *,
    metric: str = "rewards",
    window: int = 100,
    figsize: tuple = (14, 5),
) -> Any:
    """Compare learning curves for multiple RL methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for method_name, values in method_results.items():
        values = np.asarray(values)
        ax1.plot(values, alpha=0.2)
        ax2.plot(moving_average(values, window=window), label=method_name, linewidth=2)

    ax1.set_title(f"Raw {metric} per episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(metric)
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f"{window}-episode moving average")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel(metric)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("✓ Plotting utilities loaded successfully")
