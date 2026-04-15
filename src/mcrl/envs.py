"""Environment factories and reward wrappers for Mountain Car variants.

This file preserves the discrete wrappers and upgrades the continuous
wrappers so the continuous experiments can be re-run with a less degenerate
reward setup and better logging.
"""

from __future__ import annotations

from typing import Literal, Optional

import gymnasium as gym
import numpy as np


ContinuousRewardMode = Literal["replace", "augment_original"]
ContinuousFuelMode = Literal["binary", "l1"]


def _action_scalar(action: np.ndarray | list[float] | tuple[float] | float | int) -> float:
    """Robustly convert an action into a scalar float."""
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    return float(arr[0])


class DiscreteRewardInfoWrapper(gym.Wrapper):
    """Adds consistent metrics to the info dict for the standard discrete task."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action_int = int(action)
        info = dict(info)
        info.update(
            {
                "base_reward": float(reward),
                "engineered_reward": float(reward),
                "fuel_cost": 0.0,
                "time_cost": 0.0,
                "thrust_action": int(action_int != 1),
                "scenario": "discrete_standard",
                "position": float(obs[0]),
                "velocity": float(obs[1]),
            }
        )
        return obs, float(reward), terminated, truncated, info


class DiscreteFuelCostWrapper(gym.Wrapper):
    """Adapted discrete scenario: extra penalty whenever the agent uses thrust."""

    def __init__(self, env: gym.Env, fuel_cost: float = 0.25):
        super().__init__(env)
        self.fuel_cost = float(fuel_cost)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action_int = int(action)
        thrust = int(action_int != 1)
        fuel_cost = self.fuel_cost * thrust
        shaped_reward = float(reward) - fuel_cost

        info = dict(info)
        info.update(
            {
                "base_reward": float(reward),
                "engineered_reward": float(shaped_reward),
                "fuel_cost": float(fuel_cost),
                "time_cost": 0.0,
                "thrust_action": thrust,
                "scenario": "discrete_fuel",
                "position": float(obs[0]),
                "velocity": float(obs[1]),
            }
        )
        return obs, float(shaped_reward), terminated, truncated, info


class ContinuousRewardInfoWrapper(gym.Wrapper):
    """Adds metrics to the info dict for the standard continuous task."""

    def __init__(self, env: gym.Env, non_null_threshold: float = 0.05):
        super().__init__(env)
        self.non_null_threshold = float(non_null_threshold)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action_value = _action_scalar(action)
        quadratic_cost = 0.1 * (action_value ** 2)
        goal_bonus = 100.0 if terminated else 0.0

        info = dict(info)
        info.update(
            {
                "base_reward": float(reward),
                "engineered_reward": float(reward),
                "fuel_cost": float(quadratic_cost),
                "quadratic_cost": float(quadratic_cost),
                "time_cost": 0.0,
                "goal_bonus": goal_bonus,
                "non_null_action": int(abs(action_value) > self.non_null_threshold),
                "action_abs": abs(action_value),
                "scenario": "continuous_standard",
                "position": float(obs[0]),
                "velocity": float(obs[1]),
            }
        )
        return obs, float(reward), terminated, truncated, info


class ContinuousFuelCostWrapper(gym.Wrapper):
    """Adapted continuous scenario with explicit time and fuel costs.

    Parameters
    ----------
    mode:
        "binary" -> one fixed cost whenever |action| > threshold
        "l1"     -> a linear cost proportional to |action|
    reward_mode:
        "replace"          -> reward = goal_bonus(if success) - time_cost - fuel_cost
        "augment_original" -> reward = original_env_reward - time_cost - fuel_cost

    Notes
    -----
    The previous continuous experiments collapsed to near-zero actions. The
    ``replace`` + ``per_step_time_cost`` combination makes inactivity costly,
    which is usually what we want when studying a minimum-fuel *while still
    reaching the goal* variant.
    """

    def __init__(
        self,
        env: gym.Env,
        mode: ContinuousFuelMode = "binary",
        cost_coef: float = 0.05,
        non_null_threshold: float = 0.05,
        per_step_time_cost: float = 0.10,
        goal_bonus: float = 100.0,
        reward_mode: ContinuousRewardMode = "replace",
    ):
        super().__init__(env)
        if mode not in {"binary", "l1"}:
            raise ValueError("mode must be 'binary' or 'l1'")
        if reward_mode not in {"replace", "augment_original"}:
            raise ValueError("reward_mode must be 'replace' or 'augment_original'")
        self.mode = mode
        self.cost_coef = float(cost_coef)
        self.non_null_threshold = float(non_null_threshold)
        self.per_step_time_cost = float(per_step_time_cost)
        self.goal_bonus = float(goal_bonus)
        self.reward_mode = reward_mode

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        action_value = _action_scalar(action)
        action_abs = abs(action_value)

        if self.mode == "binary":
            fuel_cost = self.cost_coef * float(action_abs > self.non_null_threshold)
        else:
            fuel_cost = self.cost_coef * action_abs

        time_cost = self.per_step_time_cost
        if self.reward_mode == "replace":
            shaped_reward = (self.goal_bonus if terminated else 0.0) - time_cost - fuel_cost
        else:
            shaped_reward = float(raw_reward) - time_cost - fuel_cost

        info = dict(info)
        info.update(
            {
                "base_reward": float(raw_reward),
                "engineered_reward": float(shaped_reward),
                "fuel_cost": float(fuel_cost),
                "quadratic_cost": float(0.1 * (action_value ** 2)),
                "time_cost": float(time_cost),
                "goal_bonus": self.goal_bonus if terminated else 0.0,
                "non_null_action": int(action_abs > self.non_null_threshold),
                "action_abs": action_abs,
                "reward_mode": self.reward_mode,
                "scenario": f"continuous_fuel_{self.mode}",
                "position": float(obs[0]),
                "velocity": float(obs[1]),
            }
        )
        return obs, float(shaped_reward), terminated, truncated, info


class ContinuousStartRangeWrapper(gym.Wrapper):
    """Optional reset-range wrapper for robustness / controlled experiments."""

    def __init__(self, env: gym.Env, low: float = -0.6, high: float = -0.4):
        super().__init__(env)
        self.low = float(low)
        self.high = float(high)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options.setdefault("low", self.low)
        options.setdefault("high", self.high)
        return self.env.reset(seed=seed, options=options)


def make_discrete_env(
    *,
    render_mode: Optional[str] = None,
    goal_velocity: float = 0.0,
    fuel_cost: float | None = None,
) -> gym.Env:
    """Factory for the discrete MountainCar scenarios."""
    env = gym.make("MountainCar-v0", render_mode=render_mode, goal_velocity=goal_velocity)
    if fuel_cost is None or fuel_cost <= 0.0:
        return DiscreteRewardInfoWrapper(env)
    return DiscreteFuelCostWrapper(env, fuel_cost=fuel_cost)


def make_continuous_env(
    *,
    render_mode: Optional[str] = None,
    goal_velocity: float = 0.0,
    adapted: bool = False,
    fuel_mode: ContinuousFuelMode = "binary",
    cost_coef: float = 0.05,
    non_null_threshold: float = 0.05,
    per_step_time_cost: float = 0.10,
    goal_bonus: float = 100.0,
    reward_mode: ContinuousRewardMode = "replace",
    reset_low: float = -0.6,
    reset_high: float = -0.4,
) -> gym.Env:
    """Factory for the continuous MountainCar scenarios."""
    env = gym.make(
        "MountainCarContinuous-v0",
        render_mode=render_mode,
        goal_velocity=goal_velocity,
    )
    env = ContinuousStartRangeWrapper(env, low=reset_low, high=reset_high)
    if not adapted:
        return ContinuousRewardInfoWrapper(env, non_null_threshold=non_null_threshold)
    return ContinuousFuelCostWrapper(
        env,
        mode=fuel_mode,
        cost_coef=cost_coef,
        non_null_threshold=non_null_threshold,
        per_step_time_cost=per_step_time_cost,
        goal_bonus=goal_bonus,
        reward_mode=reward_mode,
    )
