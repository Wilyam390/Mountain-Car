
"""Discretization, tabular training loops, and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class Discretizer:
    low: np.ndarray
    high: np.ndarray
    n_pos_bins: int
    n_vel_bins: int

    def __post_init__(self):
        self.low = np.asarray(self.low, dtype=np.float32)
        self.high = np.asarray(self.high, dtype=np.float32)
        self.pos_edges = np.linspace(self.low[0], self.high[0], self.n_pos_bins + 1)
        self.vel_edges = np.linspace(self.low[1], self.high[1], self.n_vel_bins + 1)

    @classmethod
    def from_env(cls, env: gym.Env, n_pos_bins: int = 30, n_vel_bins: int = 30) -> "Discretizer":
        return cls(
            low=env.observation_space.low,
            high=env.observation_space.high,
            n_pos_bins=n_pos_bins,
            n_vel_bins=n_vel_bins,
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_pos_bins, self.n_vel_bins)

    def discretize(self, state: np.ndarray) -> Tuple[int, int]:
        position, velocity = state
        pos_idx = np.digitize(position, self.pos_edges) - 1
        vel_idx = np.digitize(velocity, self.vel_edges) - 1
        pos_idx = int(np.clip(pos_idx, 0, self.n_pos_bins - 1))
        vel_idx = int(np.clip(vel_idx, 0, self.n_vel_bins - 1))
        return pos_idx, vel_idx

    def bin_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        pos_centers = 0.5 * (self.pos_edges[:-1] + self.pos_edges[1:])
        vel_centers = 0.5 * (self.vel_edges[:-1] + self.vel_edges[1:])
        return pos_centers, vel_centers


@dataclass
class TrainingResult:
    algorithm: str
    q_table: np.ndarray
    visit_counts: np.ndarray
    engineered_returns: List[float]
    base_returns: List[float]
    steps: List[int]
    successes: List[int]
    fuel_costs: List[float]
    thrust_counts: List[int]
    epsilons: List[float]


def init_q_table(discretizer: Discretizer, n_actions: int, q_init: float = 0.0) -> np.ndarray:
    return np.full((*discretizer.shape, n_actions), fill_value=q_init, dtype=np.float32)


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, q_values.shape[0]))
    return int(np.argmax(q_values))


def _episode_seed(base_seed: int | None, episode_idx: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed + episode_idx)


def _extract_metrics(info: Dict, reward: float) -> Tuple[float, float, float, int]:
    base_reward = float(info.get("base_reward", reward))
    fuel_cost = float(info.get("fuel_cost", 0.0))
    thrust_count = int(info.get("thrust_action", info.get("non_null_action", 0)))
    return base_reward, float(reward), fuel_cost, thrust_count


def train_q_learning(
    env: gym.Env,
    discretizer: Discretizer,
    *,
    episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.999,
    epsilon_min: float = 0.05,
    seed: int | None = None,
    q_init: float = 0.0,
    verbose_every: int = 500,
) -> TrainingResult:
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    q_table = init_q_table(discretizer, n_actions, q_init=q_init)
    visit_counts = np.zeros(discretizer.shape, dtype=np.int32)

    engineered_returns: List[float] = []
    base_returns: List[float] = []
    steps_list: List[int] = []
    successes: List[int] = []
    fuel_costs: List[float] = []
    thrust_counts: List[int] = []
    epsilons: List[float] = []

    epsilon = float(epsilon_start)

    for episode in range(episodes):
        state, _ = env.reset(seed=_episode_seed(seed, episode))
        state_disc = discretizer.discretize(state)

        done = False
        engineered_return = 0.0
        base_return = 0.0
        fuel_cost_total = 0.0
        thrust_total = 0
        steps = 0
        success = 0

        while not done:
            visit_counts[state_disc] += 1
            action = epsilon_greedy(q_table[state_disc], epsilon, rng)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_disc = discretizer.discretize(next_state)
            done = terminated or truncated

            td_target = reward
            if not done:
                td_target += gamma * float(np.max(q_table[next_state_disc]))
            td_error = td_target - float(q_table[state_disc + (action,)])
            q_table[state_disc + (action,)] += alpha * td_error

            raw_base_reward, engineered_reward, fuel_cost, thrust_inc = _extract_metrics(info, reward)
            engineered_return += engineered_reward
            base_return += raw_base_reward
            fuel_cost_total += fuel_cost
            thrust_total += thrust_inc
            steps += 1
            success = int(terminated)

            state_disc = next_state_disc

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        engineered_returns.append(engineered_return)
        base_returns.append(base_return)
        steps_list.append(steps)
        successes.append(success)
        fuel_costs.append(fuel_cost_total)
        thrust_counts.append(thrust_total)
        epsilons.append(epsilon)

        if verbose_every and (episode + 1) % verbose_every == 0:
            avg_reward = float(np.mean(engineered_returns[-verbose_every:]))
            avg_success = float(np.mean(successes[-verbose_every:]))
            avg_steps = float(np.mean(steps_list[-verbose_every:]))
            print(
                f"[Q-learning] Episode {episode + 1}/{episodes} | "
                f"avg engineered return: {avg_reward:.3f} | "
                f"avg steps: {avg_steps:.2f} | "
                f"success rate: {avg_success:.3f} | "
                f"epsilon: {epsilon:.3f}"
            )

    return TrainingResult(
        algorithm="q_learning",
        q_table=q_table,
        visit_counts=visit_counts,
        engineered_returns=engineered_returns,
        base_returns=base_returns,
        steps=steps_list,
        successes=successes,
        fuel_costs=fuel_costs,
        thrust_counts=thrust_counts,
        epsilons=epsilons,
    )


def train_sarsa(
    env: gym.Env,
    discretizer: Discretizer,
    *,
    episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.999,
    epsilon_min: float = 0.05,
    seed: int | None = None,
    q_init: float = 0.0,
    verbose_every: int = 500,
) -> TrainingResult:
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    q_table = init_q_table(discretizer, n_actions, q_init=q_init)
    visit_counts = np.zeros(discretizer.shape, dtype=np.int32)

    engineered_returns: List[float] = []
    base_returns: List[float] = []
    steps_list: List[int] = []
    successes: List[int] = []
    fuel_costs: List[float] = []
    thrust_counts: List[int] = []
    epsilons: List[float] = []

    epsilon = float(epsilon_start)

    for episode in range(episodes):
        state, _ = env.reset(seed=_episode_seed(seed, episode))
        state_disc = discretizer.discretize(state)
        action = epsilon_greedy(q_table[state_disc], epsilon, rng)

        done = False
        engineered_return = 0.0
        base_return = 0.0
        fuel_cost_total = 0.0
        thrust_total = 0
        steps = 0
        success = 0

        while not done:
            visit_counts[state_disc] += 1
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_disc = discretizer.discretize(next_state)
            done = terminated or truncated

            if done:
                td_target = reward
            else:
                next_action = epsilon_greedy(q_table[next_state_disc], epsilon, rng)
                td_target = reward + gamma * float(q_table[next_state_disc + (next_action,)])

            td_error = td_target - float(q_table[state_disc + (action,)])
            q_table[state_disc + (action,)] += alpha * td_error

            raw_base_reward, engineered_reward, fuel_cost, thrust_inc = _extract_metrics(info, reward)
            engineered_return += engineered_reward
            base_return += raw_base_reward
            fuel_cost_total += fuel_cost
            thrust_total += thrust_inc
            steps += 1
            success = int(terminated)

            state_disc = next_state_disc
            if not done:
                action = next_action

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        engineered_returns.append(engineered_return)
        base_returns.append(base_return)
        steps_list.append(steps)
        successes.append(success)
        fuel_costs.append(fuel_cost_total)
        thrust_counts.append(thrust_total)
        epsilons.append(epsilon)

        if verbose_every and (episode + 1) % verbose_every == 0:
            avg_reward = float(np.mean(engineered_returns[-verbose_every:]))
            avg_success = float(np.mean(successes[-verbose_every:]))
            avg_steps = float(np.mean(steps_list[-verbose_every:]))
            print(
                f"[SARSA] Episode {episode + 1}/{episodes} | "
                f"avg engineered return: {avg_reward:.3f} | "
                f"avg steps: {avg_steps:.2f} | "
                f"success rate: {avg_success:.3f} | "
                f"epsilon: {epsilon:.3f}"
            )

    return TrainingResult(
        algorithm="sarsa",
        q_table=q_table,
        visit_counts=visit_counts,
        engineered_returns=engineered_returns,
        base_returns=base_returns,
        steps=steps_list,
        successes=successes,
        fuel_costs=fuel_costs,
        thrust_counts=thrust_counts,
        epsilons=epsilons,
    )


def run_greedy_episode(
    env: gym.Env,
    q_table: np.ndarray,
    discretizer: Discretizer,
    *,
    seed: int | None = None,
) -> Dict:
    state, _ = env.reset(seed=seed)
    state_disc = discretizer.discretize(state)

    positions = [float(state[0])]
    velocities = [float(state[1])]
    actions = []

    done = False
    engineered_return = 0.0
    base_return = 0.0
    fuel_cost_total = 0.0
    thrust_total = 0
    steps = 0
    success = 0

    while not done:
        action = int(np.argmax(q_table[state_disc]))
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state_disc = discretizer.discretize(next_state)
        done = terminated or truncated

        raw_base_reward, engineered_reward, fuel_cost, thrust_inc = _extract_metrics(info, reward)
        engineered_return += engineered_reward
        base_return += raw_base_reward
        fuel_cost_total += fuel_cost
        thrust_total += thrust_inc
        steps += 1
        success = int(terminated)

        actions.append(action)
        positions.append(float(next_state[0]))
        velocities.append(float(next_state[1]))
        state_disc = next_state_disc

    return {
        "engineered_return": engineered_return,
        "base_return": base_return,
        "fuel_cost": fuel_cost_total,
        "thrust_count": thrust_total,
        "steps": steps,
        "success": success,
        "positions": np.asarray(positions, dtype=np.float32),
        "velocities": np.asarray(velocities, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int32),
    }


def evaluate_greedy_policy(
    env: gym.Env,
    q_table: np.ndarray,
    discretizer: Discretizer,
    *,
    n_episodes: int = 100,
    seed: int | None = 123,
) -> Dict[str, float]:
    episodes = [
        run_greedy_episode(env, q_table, discretizer, seed=_episode_seed(seed, i))
        for i in range(n_episodes)
    ]
    summary = {
        "n_episodes": n_episodes,
        "mean_engineered_return": float(np.mean([ep["engineered_return"] for ep in episodes])),
        "std_engineered_return": float(np.std([ep["engineered_return"] for ep in episodes])),
        "mean_base_return": float(np.mean([ep["base_return"] for ep in episodes])),
        "mean_steps": float(np.mean([ep["steps"] for ep in episodes])),
        "success_rate": float(np.mean([ep["success"] for ep in episodes])),
        "mean_fuel_cost": float(np.mean([ep["fuel_cost"] for ep in episodes])),
        "mean_thrust_count": float(np.mean([ep["thrust_count"] for ep in episodes])),
    }
    return summary
