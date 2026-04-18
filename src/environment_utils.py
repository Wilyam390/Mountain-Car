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
    fuel_cost: float = 0.25,
    cost_coef: float = 0.05,
    per_step_time_cost: float = 0.10,
) -> gym.Env:
    """
    Create and configure a Mountain Car environment.

    Args:
        env_type: "discrete" or "continuous"
        scenario: Environment variant:
            Discrete: "min_steps", "min_fuel", "discrete_standard", "discrete_fuel"
            Continuous: "quadratic_cost", "linear_cost", "continuous_standard",
                       "continuous_linear", "continuous_fuel_binary", "continuous_fuel_l1"
        render: Whether to enable rendering
        seed: Random seed for reproducibility
        fuel_cost: Fuel cost coefficient for discrete (default 0.25)
        cost_coef: Fuel cost coefficient for continuous (default 0.05)
        per_step_time_cost: Per-step time penalty for continuous (default 0.10)

    Returns:
        Configured Gymnasium environment

    Examples:
        env = create_env("discrete", "min_steps", seed=42)
        env = create_env("continuous", "continuous_fuel_binary", seed=42)
        env = create_env("discrete", "min_fuel", fuel_cost=0.25, seed=42)
    """
    if env_type == "discrete":
        env = gym.make("MountainCar-v0", render_mode="rgb_array" if render else None)
    elif env_type == "continuous":
        env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array" if render else None)
    else:
        raise ValueError(f"Unknown env_type: {env_type}. Use 'discrete' or 'continuous'")

    # Apply scenario-based reward wrapper
    env = _apply_reward_wrapper(
        env,
        env_type,
        scenario,
        fuel_cost=fuel_cost,
        cost_coef=cost_coef,
        per_step_time_cost=per_step_time_cost,
    )

    if seed is not None:
        env.reset(seed=seed)

    return env


def _apply_reward_wrapper(
    env,
    env_type: str,
    scenario: str,
    fuel_cost: float = 0.25,
    cost_coef: float = 0.05,
    per_step_time_cost: float = 0.10,
) -> gym.Env:
    """Wrap environment with scenario-specific reward function."""

    if env_type == "discrete":
        if scenario == "min_steps" or scenario == "discrete_standard":
            return StandardRewardWrapper(env)
        elif scenario == "min_fuel" or scenario == "discrete_fuel":
            return MinFuelRewardWrapper(env, fuel_cost=fuel_cost)
        else:
            raise ValueError(f"Unknown discrete scenario: {scenario}")

    elif env_type == "continuous":
        # Add start range wrapper for consistency
        env = ContinuousStartRangeWrapper(env)

        if scenario == "quadratic_cost" or scenario == "continuous_standard":
            return ContinuousRewardInfoWrapper(env)
        elif scenario == "linear_cost" or scenario == "continuous_linear":
            return LinearActionCostWrapper(env)
        elif scenario == "continuous_fuel_binary":
            return ContinuousFuelCostWrapper(
                env,
                mode="binary",
                cost_coef=cost_coef,
                per_step_time_cost=per_step_time_cost,
            )
        elif scenario == "continuous_fuel_l1":
            return ContinuousFuelCostWrapper(
                env,
                mode="l1",
                cost_coef=cost_coef,
                per_step_time_cost=per_step_time_cost,
            )
        else:
            raise ValueError(f"Unknown continuous scenario: {scenario}")


# ============================================================================
# REWARD WRAPPERS
# ============================================================================

# Helper function for continuous wrappers
def _action_scalar(action: np.ndarray | list | tuple | float | int) -> float:
    """Robustly convert an action into a scalar float."""
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    return float(arr[0])


class StandardRewardWrapper(Wrapper):
    """
    Discrete Mountain Car with standard reward: -1 per step.
    Optimizes for minimum number of steps to goal.
    Adds consistent info dict metrics.
    """

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


class MinFuelRewardWrapper(Wrapper):
    """
    Discrete Mountain Car with fuel cost penalty.
    Reward: -1 (step cost) - fuel_cost
    Fuel cost: proportional to fuel_cost param if action=left(0) or right(2)

    Optimizes for fuel efficiency + reaching goal.
    """

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


class LinearActionCostWrapper(Wrapper):
    """
    Continuous Mountain Car with linear action cost.
    Reward: -0.1 * |action| + goal_bonus

    Penalizes large actions linearly. Smoother alternative to quadratic.
    """

    def __init__(self, env: gym.Env, non_null_threshold: float = 0.05):
        super().__init__(env)
        self.non_null_threshold = float(non_null_threshold)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action_value = _action_scalar(action)
        linear_cost = 0.1 * abs(action_value)
        goal_bonus = 100.0 if terminated else 0.0
        shaped_reward = goal_bonus - linear_cost

        info = dict(info)
        info.update(
            {
                "base_reward": float(reward),
                "engineered_reward": float(shaped_reward),
                "fuel_cost": float(linear_cost),
                "linear_cost": float(linear_cost),
                "time_cost": 0.0,
                "goal_bonus": goal_bonus,
                "non_null_action": int(abs(action_value) > self.non_null_threshold),
                "action_abs": abs(action_value),
                "scenario": "continuous_linear",
                "position": float(obs[0]),
                "velocity": float(obs[1]),
            }
        )
        return obs, float(shaped_reward), terminated, truncated, info


class ContinuousRewardInfoWrapper(Wrapper):
    """
    Continuous Mountain Car with standard (quadratic) action cost.
    Adds metrics to the info dict for the standard continuous task.
    Reward: Original env reward (which is -0.1 * action^2 + goal_bonus)
    """

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


class ContinuousFuelCostWrapper(Wrapper):
    """
    Continuous Mountain Car with explicit time and fuel costs.

    Supports both binary and L1 fuel modes with customizable reward composition.
    This wrapper enables control over action penalties and time costs for
    energy-efficient control studies.

    Parameters
    ----------
    mode : {'binary', 'l1'}
        'binary' -> fixed cost when |action| > threshold
        'l1'     -> linear cost proportional to |action|
    cost_coef : float
        Scaling factor for fuel cost (default 0.05)
    per_step_time_cost : float
        Per-step penalty for exploration (default 0.10)
    reward_mode : {'replace', 'augment_original'}
        'replace' -> reward = goal_bonus - time_cost - fuel_cost
        'augment_original' -> reward = env_reward - time_cost - fuel_cost
    """

    def __init__(
        self,
        env: gym.Env,
        mode: str = "binary",
        cost_coef: float = 0.05,
        non_null_threshold: float = 0.05,
        per_step_time_cost: float = 0.10,
        goal_bonus: float = 100.0,
        reward_mode: str = "replace",
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
        else:  # l1
            fuel_cost = self.cost_coef * action_abs

        time_cost = self.per_step_time_cost
        if self.reward_mode == "replace":
            shaped_reward = (self.goal_bonus if terminated else 0.0) - time_cost - fuel_cost
        else:  # augment_original
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


class ContinuousStartRangeWrapper(Wrapper):
    """Optional reset-range wrapper for robustness and controlled experiments."""

    def __init__(self, env: gym.Env, low: float = -0.6, high: float = -0.4):
        super().__init__(env)
        self.low = float(low)
        self.high = float(high)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options.setdefault("low", self.low)
        options.setdefault("high", self.high)
        return self.env.reset(seed=seed, options=options)


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
