"""
Environment utilities for Mountain Car RL experiments.

Provides:
- Factory for creating configured environments
- State discretization utilities
- Custom reward wrappers for different scenarios
- State preprocessing/normalization
- Environment validation
"""

import numpy as np
import gymnasium as gym
from gymnasium import Wrapper


# ============================================================================
# ENVIRONMENT FACTORY
# ============================================================================

def create_env(
    env_type: str = "discrete",
    scenario: str = "min_steps",
    render: bool = False,
    seed: int = None,
) -> gym.Env:
    """
    Create and configure a Mountain Car environment.

    Args:
        env_type: "discrete" or "continuous"
        scenario: "min_steps", "min_fuel", "linear_cost", "quadratic_cost"
        render: Whether to enable rendering
        seed: Random seed for reproducibility

    Returns:
        Configured Gymnasium environment

    Examples:
        env = create_env("discrete", "min_steps", seed=42)
        env = create_env("continuous", "quadratic_cost", seed=42)
    """
    if env_type == "discrete":
        env = gym.make("MountainCar-v0", render_mode="rgb_array" if render else None)
    elif env_type == "continuous":
        env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array" if render else None)
    else:
        raise ValueError(f"Unknown env_type: {env_type}. Use 'discrete' or 'continuous'")

    # Apply scenario-based reward wrapper
    env = _apply_reward_wrapper(env, env_type, scenario)

    if seed is not None:
        env.reset(seed=seed)

    return env


def _apply_reward_wrapper(env, env_type: str, scenario: str) -> gym.Env:
    """Wrap environment with scenario-specific reward function."""

    if env_type == "discrete":
        if scenario == "min_steps":
            return StandardRewardWrapper(env)
        elif scenario == "min_fuel":
            return MinFuelRewardWrapper(env)
        else:
            raise ValueError(f"Unknown discrete scenario: {scenario}")

    elif env_type == "continuous":
        if scenario == "quadratic_cost":
            return env  # Default continuous env already uses this
        elif scenario == "linear_cost":
            return LinearActionCostWrapper(env)
        else:
            raise ValueError(f"Unknown continuous scenario: {scenario}")


# ============================================================================
# REWARD WRAPPERS
# ============================================================================

class StandardRewardWrapper(Wrapper):
    """
    Discrete Mountain Car with standard reward: -1 per step.
    Optimizes for minimum number of steps to goal.
    """

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # Default reward is -1 per step, +0 if goal reached
        # (goal termination adds implicit bonus)
        return next_state, reward, terminated, truncated, info


class MinFuelRewardWrapper(Wrapper):
    """
    Discrete Mountain Car with fuel cost penalty.
    Reward: -1 (step cost) - fuel_cost
    Fuel cost: 1 if action=left(0) or right(2), 0 if action=none(1)

    Optimizes for fuel efficiency + reaching goal.
    """

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        fuel_cost = 1.0 if action in [0, 2] else 0.0
        modified_reward = reward - fuel_cost
        return next_state, modified_reward, terminated, truncated, info


class LinearActionCostWrapper(Wrapper):
    """
    Continuous Mountain Car with linear action cost.
    Reward: -0.1 * |action| + goal_bonus

    Penalizes large actions linearly.
    """

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # In continuous env:
        # reward = -0.1 * action^2 (default)
        # We can override with linear cost:
        action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        linear_action_cost = -0.1 * abs(action_val)
        # Preserve goal bonus if reached
        goal_bonus = 100.0 if terminated else 0.0
        modified_reward = linear_action_cost + goal_bonus
        return next_state, modified_reward, terminated, truncated, info


# ============================================================================
# STATE DISCRETIZATION
# ============================================================================

class StateDiscretizer:
    """
    Converts continuous Mountain Car state to discrete indices.

    State: [position, velocity]
    Position range: [-1.2, 0.6]
    Velocity range: [-0.07, 0.07]

    Creates a grid of n_pos_bins × n_vel_bins bins.
    """

    def __init__(self, n_pos_bins: int = 20, n_vel_bins: int = 20):
        """
        Initialize discretizer.

        Args:
            n_pos_bins: Number of position bins
            n_vel_bins: Number of velocity bins
        """
        self.n_pos_bins = n_pos_bins
        self.n_vel_bins = n_vel_bins

        # Hard-coded ranges from Mountain Car environment
        self.pos_range = (-1.2, 0.6)
        self.vel_range = (-0.07, 0.07)

        # Create bin edges
        self.pos_bins = np.linspace(self.pos_range[0], self.pos_range[1], n_pos_bins + 1)
        self.vel_bins = np.linspace(self.vel_range[0], self.vel_range[1], n_vel_bins + 1)

    def discretize(self, state: np.ndarray) -> tuple:
        """
        Convert continuous state to discrete (pos_idx, vel_idx).

        Args:
            state: [position, velocity] array

        Returns:
            (pos_idx, vel_idx) tuple of integers
        """
        position, velocity = state

        # digitize returns index in bins (1-indexed), so subtract 1
        pos_idx = np.digitize(position, self.pos_bins) - 1
        vel_idx = np.digitize(velocity, self.vel_bins) - 1

        # Clip to valid range
        pos_idx = np.clip(pos_idx, 0, self.n_pos_bins - 1)
        vel_idx = np.clip(vel_idx, 0, self.n_vel_bins - 1)

        return (int(pos_idx), int(vel_idx))

    def get_grid_shape(self) -> tuple:
        """Return shape of discretized state space."""
        return (self.n_pos_bins, self.n_vel_bins)

    def index_to_state_region(self, pos_idx: int, vel_idx: int) -> tuple:
        """
        Convert discrete indices back to continuous state region (center point).

        Useful for visualization and physical interpretation.

        Returns:
            (position, velocity) center of bin region
        """
        pos_center = (self.pos_bins[pos_idx] + self.pos_bins[pos_idx + 1]) / 2
        vel_center = (self.vel_bins[vel_idx] + self.vel_bins[vel_idx + 1]) / 2
        return (pos_center, vel_center)


# ============================================================================
# STATE NORMALIZATION
# ============================================================================

class StateNormalizer:
    """Normalize continuous states to [-1, 1] range (useful for neural networks)."""

    def __init__(
        self,
        pos_range: tuple = (-1.2, 0.6),
        vel_range: tuple = (-0.07, 0.07),
    ):
        """
        Initialize normalizer with state ranges.

        Args:
            pos_range: (min, max) for position
            vel_range: (min, max) for velocity
        """
        self.pos_range = pos_range
        self.vel_range = vel_range

    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1]."""
        position, velocity = state
        norm_pos = 2 * (position - self.pos_range[0]) / (self.pos_range[1] - self.pos_range[0]) - 1
        norm_vel = 2 * (velocity - self.vel_range[0]) / (self.vel_range[1] - self.vel_range[0]) - 1
        return np.array([norm_pos, norm_vel], dtype=np.float32)

    def denormalize(self, norm_state: np.ndarray) -> np.ndarray:
        """Denormalize from [-1, 1] back to original ranges."""
        norm_pos, norm_vel = norm_state
        position = (norm_pos + 1) / 2 * (self.pos_range[1] - self.pos_range[0]) + self.pos_range[0]
        velocity = (norm_vel + 1) / 2 * (self.vel_range[1] - self.vel_range[0]) + self.vel_range[0]
        return np.array([position, velocity], dtype=np.float32)


# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================

def validate_environment(env: gym.Env, n_steps: int = 100) -> dict:
    """
    Validate environment by taking random steps and checking outputs.

    Args:
        env: Gymnasium environment
        n_steps: Number of steps to take

    Returns:
        Dictionary with validation results

    Raises:
        AssertionError if validation fails
    """
    results = {
        "obs_space_shape": env.observation_space.shape,
        "action_space_type": env.action_space.__class__.__name__,
        "is_discrete_action": hasattr(env.action_space, "n"),
        "reward_stats": {"min": np.inf, "max": -np.inf, "mean": 0, "std": 0},
        "trajectory_length": 0,
    }

    rewards = []
    state, _ = env.reset(seed=42)

    for _ in range(n_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        assert isinstance(next_state, (np.ndarray, tuple))
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        rewards.append(reward)
        results["trajectory_length"] += 1

        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    rewards = np.array(rewards)
    results["reward_stats"] = {
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
    }

    return results


if __name__ == "__main__":
    # Quick tests
    print("Testing environment creation...")

    env_disc = create_env("discrete", "min_steps", seed=42)
    print(f"✓ Created discrete min_steps env")

    env_fuel = create_env("discrete", "min_fuel", seed=42)
    print(f"✓ Created discrete min_fuel env")

    env_cont = create_env("continuous", "quadratic_cost", seed=42)
    print(f"✓ Created continuous quadratic_cost env")

    print("\nTesting discretization...")
    discretizer = StateDiscretizer(n_pos_bins=20, n_vel_bins=20)
    state, _ = env_disc.reset(seed=42)
    disc_state = discretizer.discretize(state)
    print(f"State {state} → Discrete {disc_state}")
    print(f"Grid shape: {discretizer.get_grid_shape()}")

    print("\nTesting normalization...")
    normalizer = StateNormalizer()
    norm_state = normalizer.normalize(state)
    denorm_state = normalizer.denormalize(norm_state)
    print(f"Original: {state}")
    print(f"Normalized: {norm_state}")
    print(f"Denormalized: {denorm_state}")

    print("\nValidating environments...")
    val_results = validate_environment(env_disc, n_steps=50)
    print(f"Validation results: {val_results}")

    print("\n✓ All basic tests passed!")
