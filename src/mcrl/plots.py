
"""Matplotlib helpers for the Mountain Car project."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .tabular import Discretizer, TrainingResult, run_greedy_episode


def moving_average(values: Sequence[float], window: int = 100) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values.copy()
    return np.convolve(values, np.ones(window, dtype=np.float32) / window, mode="valid")


def plot_training_curve(
    values: Sequence[float],
    *,
    window: int = 100,
    title: str = "Training curve",
    ylabel: str = "Value",
    ax=None,
):
    values = np.asarray(values, dtype=np.float32)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values, alpha=0.35, label="Episode value")
    ax.plot(moving_average(values, window=window), label=f"{window}-episode moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_success_curve(
    successes: Sequence[int],
    *,
    window: int = 100,
    title: str = "Success rate",
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ma = moving_average(successes, window=window)
    ax.plot(ma)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    return ax


def plot_policy_map(
    q_table: np.ndarray,
    discretizer: Discretizer,
    *,
    visit_counts: np.ndarray | None = None,
    mask_unvisited: bool = True,
    title: str = "Greedy policy",
    ax=None,
):
    policy = np.argmax(q_table, axis=2).astype(np.float32)
    if visit_counts is not None and mask_unvisited:
        policy = np.ma.masked_where(visit_counts <= 0, policy)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        policy.T,
        origin="lower",
        aspect="auto",
        extent=[
            discretizer.low[0],
            discretizer.high[0],
            discretizer.low[1],
            discretizer.high[1],
        ],
    )
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["left", "idle", "right"])
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title)
    return ax


def plot_visitation_heatmap(
    visit_counts: np.ndarray,
    discretizer: Discretizer,
    *,
    log_scale: bool = True,
    title: str = "State visitation heatmap",
    ax=None,
):
    values = np.log1p(visit_counts) if log_scale else visit_counts
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        values.T,
        origin="lower",
        aspect="auto",
        extent=[
            discretizer.low[0],
            discretizer.high[0],
            discretizer.low[1],
            discretizer.high[1],
        ],
    )
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("log(1 + visits)" if log_scale else "visits")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title)
    return ax


def plot_value_surface(
    q_table: np.ndarray,
    discretizer: Discretizer,
    *,
    mode: str = "max",
    action: int | None = None,
    title: str = "Value surface",
    elev: int = 30,
    azim: int = -60,
):
    if mode == "max":
        values = np.max(q_table, axis=2)
    elif mode == "action":
        if action is None:
            raise ValueError("action must be provided when mode='action'")
        values = q_table[:, :, action]
    else:
        raise ValueError("mode must be 'max' or 'action'")

    pos_centers, vel_centers = discretizer.bin_centers()
    X, Y = np.meshgrid(pos_centers, vel_centers, indexing="ij")

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, values, rstride=1, cstride=1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Value")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    return ax


def collect_greedy_trajectories(
    env,
    q_table: np.ndarray,
    discretizer: Discretizer,
    *,
    n_episodes: int = 10,
    seed: int | None = 1000,
) -> List[dict]:
    trajectories = []
    for i in range(n_episodes):
        ep_seed = None if seed is None else int(seed + i)
        trajectories.append(run_greedy_episode(env, q_table, discretizer, seed=ep_seed))
    return trajectories


def plot_phase_portrait(
    trajectories: Iterable[dict],
    *,
    title: str = "Greedy trajectories in phase space",
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    for episode in trajectories:
        ax.plot(episode["positions"], episode["velocities"], marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title)
    return ax
