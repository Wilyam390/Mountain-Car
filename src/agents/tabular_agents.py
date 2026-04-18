"""
Tabular RL agents: Q-learning and SARSA

These agents implement value-based reinforcement learning using discretized state spaces.
Both inherit from BaseAgent to ensure compatibility with the generic training loop.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import gymnasium as gym

from .base_agent import BaseAgent, EpsilonGreedyStrategy


class StateDiscretizer:
    """
    Converts continuous Mountain Car states to discrete indices.

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

    def discretize(self, state: np.ndarray) -> Tuple[int, int]:
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

    def get_grid_shape(self) -> Tuple[int, int]:
        """Return shape of discretized state space."""
        return (self.n_pos_bins, self.n_vel_bins)

    def index_to_state_region(self, pos_idx: int, vel_idx: int) -> Tuple[float, float]:
        """
        Convert discrete indices back to continuous state region (center point).

        Useful for visualization and physical interpretation.

        Returns:
            (position, velocity) center of bin region
        """
        pos_center = (self.pos_bins[pos_idx] + self.pos_bins[pos_idx + 1]) / 2
        vel_center = (self.vel_bins[vel_idx] + self.vel_bins[vel_idx + 1]) / 2
        return (pos_center, vel_center)


class QLearning(BaseAgent):
    """
    Q-learning agent for discrete, tabular Mountain Car.

    Off-policy value-based method that learns the optimal action-value function.
    Uses state discretization to approximate the continuous state space.

    Update rule: Q[s,a] ← Q[s,a] + α(r + γ max Q[s',a'] - Q[s,a])
    """

    def __init__(
        self,
        n_actions: int = 3,
        n_pos_bins: int = 20,
        n_vel_bins: int = 20,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05,
    ):
        """
        Initialize Q-learning agent.

        Args:
            n_actions: Number of discrete actions (default 3 for Mountain Car)
            n_pos_bins: Position discretization bins
            n_vel_bins: Velocity discretization bins
            alpha: Learning rate (0-1)
            gamma: Discount factor (0-1)
            epsilon_start: Initial exploration rate
            epsilon_decay: Exploration decay per episode
            epsilon_min: Minimum exploration rate
        """
        state_shape = (n_pos_bins, n_vel_bins)
        super().__init__(n_actions=n_actions, state_shape=state_shape, agent_name="Q-learning")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.discretizer = StateDiscretizer(n_pos_bins, n_vel_bins)

        # Q-table: shape (n_pos_bins, n_vel_bins, n_actions)
        self.Q = np.zeros(state_shape + (n_actions,), dtype=np.float32)
        self.visit_counts = np.zeros(state_shape, dtype=np.int32)

        # Exploration strategy
        self.strategy = EpsilonGreedyStrategy(
            epsilon_start=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
        )

        # Metrics tracking
        self.episode_steps = 0
        self.episode_reward = 0.0

    def train_step(self, state: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Choose action during training (with epsilon-greedy exploration)."""
        state_disc = self.discretizer.discretize(state)
        greedy_action = int(np.argmax(self.Q[state_disc]))
        action = self.strategy.select_action(greedy_action, self.n_actions)
        return action, {"epsilon": self.strategy.get_epsilon()}

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Q-learning update: Q[s,a] ← Q[s,a] + α(r + γ max Q[s',a'] - Q[s,a])
        """
        state_disc = self.discretizer.discretize(state)
        next_state_disc = self.discretizer.discretize(next_state)

        self.visit_counts[state_disc] += 1

        # Q-learning: off-policy, use max of next state
        best_next_q = np.max(self.Q[next_state_disc])
        td_target = float(reward) + self.gamma * best_next_q * (not done)
        td_error = td_target - float(self.Q[state_disc + (action,)])
        self.Q[state_disc + (action,)] += self.alpha * td_error

        # Decay exploration
        self.strategy.decay()

    def act(self, state: np.ndarray, training: bool = False) -> int:
        """Greedy action selection (no exploration in eval)."""
        state_disc = self.discretizer.discretize(state)
        return int(np.argmax(self.Q[state_disc]))

    def get_hyperparams(self) -> Dict[str, Any]:
        """Return current hyperparameters."""
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.strategy.get_epsilon(),
            "n_pos_bins": self.discretizer.n_pos_bins,
            "n_vel_bins": self.discretizer.n_vel_bins,
        }

    def set_hyperparams(self, params: Dict[str, Any]) -> None:
        """Update hyperparameters from dictionary."""
        if "alpha" in params:
            self.alpha = float(params["alpha"])
        if "gamma" in params:
            self.gamma = float(params["gamma"])

    def save(self, path: str) -> None:
        """Save Q-table and visit counts."""
        np.savez(path, Q=self.Q, visit_counts=self.visit_counts)

    def load(self, path: str) -> None:
        """Load Q-table and visit counts."""
        data = np.load(path)
        self.Q = data["Q"]
        self.visit_counts = data["visit_counts"]

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about learned Q-values."""
        return {
            "q_mean": float(np.mean(self.Q)),
            "q_std": float(np.std(self.Q)),
            "q_max": float(np.max(self.Q)),
            "visited_states": int(np.sum(self.visit_counts > 0)),
            "total_visits": int(np.sum(self.visit_counts)),
        }


class SARSA(BaseAgent):
    """
    SARSA agent for discrete, tabular Mountain Car.

    On-policy value-based method that learns from the actual policy being followed.
    Uses state discretization to approximate the continuous state space.

    Update rule: Q[s,a] ← Q[s,a] + α(r + γ Q[s',a'] - Q[s,a])
    where a' is the actual action taken in s' (not greedy)
    """

    def __init__(
        self,
        n_actions: int = 3,
        n_pos_bins: int = 20,
        n_vel_bins: int = 20,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05,
    ):
        """
        Initialize SARSA agent.

        Args:
            n_actions: Number of discrete actions (default 3 for Mountain Car)
            n_pos_bins: Position discretization bins
            n_vel_bins: Velocity discretization bins
            alpha: Learning rate (0-1)
            gamma: Discount factor (0-1)
            epsilon_start: Initial exploration rate
            epsilon_decay: Exploration decay per episode
            epsilon_min: Minimum exploration rate
        """
        state_shape = (n_pos_bins, n_vel_bins)
        super().__init__(n_actions=n_actions, state_shape=state_shape, agent_name="SARSA")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.discretizer = StateDiscretizer(n_pos_bins, n_vel_bins)

        # Q-table: shape (n_pos_bins, n_vel_bins, n_actions)
        self.Q = np.zeros(state_shape + (n_actions,), dtype=np.float32)
        self.visit_counts = np.zeros(state_shape, dtype=np.int32)

        # Exploration strategy
        self.strategy = EpsilonGreedyStrategy(
            epsilon_start=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
        )

        # Track next action for SARSA update
        self._next_action = None

    def train_step(self, state: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Choose action during training (with epsilon-greedy exploration)."""
        state_disc = self.discretizer.discretize(state)
        greedy_action = int(np.argmax(self.Q[state_disc]))
        action = self.strategy.select_action(greedy_action, self.n_actions)
        return action, {"epsilon": self.strategy.get_epsilon()}

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        SARSA update: Q[s,a] ← Q[s,a] + α(r + γ Q[s',a'] - Q[s,a])
        where a' is sampled from the policy (on-policy)
        """
        state_disc = self.discretizer.discretize(state)
        next_state_disc = self.discretizer.discretize(next_state)

        self.visit_counts[state_disc] += 1

        # SARSA: on-policy, need to get action from next state
        if not done:
            # Choose next action using same exploration strategy
            greedy_next_action = int(np.argmax(self.Q[next_state_disc]))
            next_action = self.strategy.select_action(greedy_next_action, self.n_actions)
            next_q_value = float(self.Q[next_state_disc + (next_action,)])
        else:
            next_q_value = 0.0

        td_target = float(reward) + self.gamma * next_q_value
        td_error = td_target - float(self.Q[state_disc + (action,)])
        self.Q[state_disc + (action,)] += self.alpha * td_error

        # Decay exploration
        self.strategy.decay()

    def act(self, state: np.ndarray, training: bool = False) -> int:
        """Greedy action selection (no exploration in eval)."""
        state_disc = self.discretizer.discretize(state)
        return int(np.argmax(self.Q[state_disc]))

    def get_hyperparams(self) -> Dict[str, Any]:
        """Return current hyperparameters."""
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.strategy.get_epsilon(),
            "n_pos_bins": self.discretizer.n_pos_bins,
            "n_vel_bins": self.discretizer.n_vel_bins,
        }

    def set_hyperparams(self, params: Dict[str, Any]) -> None:
        """Update hyperparameters from dictionary."""
        if "alpha" in params:
            self.alpha = float(params["alpha"])
        if "gamma" in params:
            self.gamma = float(params["gamma"])

    def save(self, path: str) -> None:
        """Save Q-table and visit counts."""
        np.savez(path, Q=self.Q, visit_counts=self.visit_counts)

    def load(self, path: str) -> None:
        """Load Q-table and visit counts."""
        data = np.load(path)
        self.Q = data["Q"]
        self.visit_counts = data["visit_counts"]

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about learned Q-values."""
        return {
            "q_mean": float(np.mean(self.Q)),
            "q_std": float(np.std(self.Q)),
            "q_max": float(np.max(self.Q)),
            "visited_states": int(np.sum(self.visit_counts > 0)),
            "total_visits": int(np.sum(self.visit_counts)),
        }


class MonteCarloAgent(BaseAgent):
    """
    Monte Carlo agent for tabular RL.

    Updates Q-values based on actual returns from complete episodes.
    Every-visit Monte Carlo implementation.
    """

    def __init__(
        self,
        q_table: np.ndarray,
        discretizer: StateDiscretizer,
        visit_counts: np.ndarray,
        gamma: float = 0.99,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Initialize Monte Carlo agent.

        Args:
            q_table: Initial Q-table (n_pos × n_vel × n_actions)
            discretizer: StateDiscretizer instance
            visit_counts: Tracking state visitation
            gamma: Discount factor
            epsilon_decay: Decay rate for exploration
            epsilon_min: Minimum epsilon value
        """
        super().__init__(q_table=q_table, discretizer=discretizer, visit_counts=visit_counts)
        self.gamma = float(gamma)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)

        # Monte Carlo specific: store episode trajectories
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # Initialize strategy
        self.n_actions = q_table.shape[-1]
        self.strategy = EpsilonGreedyStrategy(
            epsilon=1.0,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
        )

    def train_step(self, state: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Choose action during training (with epsilon-greedy exploration)."""
        state_disc = self.discretizer.discretize(state)
        greedy_action = int(np.argmax(self.Q[state_disc]))
        action = self.strategy.select_action(greedy_action, self.n_actions)
        return action, {"epsilon": self.strategy.get_epsilon()}

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store trajectory data. Actual update happens at episode end."""
        state_disc = self.discretizer.discretize(state)

        # Track state visitation
        self.visit_counts[state_disc] += 1

        # Store trajectory
        self.episode_states.append(state_disc)
        self.episode_actions.append(action)
        self.episode_rewards.append(float(reward))

    def act(self, state: np.ndarray, training: bool = False) -> int:
        """Greedy action selection (no exploration in eval)."""
        state_disc = self.discretizer.discretize(state)
        return int(np.argmax(self.Q[state_disc]))

    def end_episode(self) -> None:
        """
        Update Q-values based on episode returns (Monte Carlo update).
        Called at the end of each episode.
        """
        if len(self.episode_rewards) == 0:
            return

        # Compute returns (discounted cumulative rewards)
        returns = []
        g = 0.0
        for reward in reversed(self.episode_rewards):
            g = reward + self.gamma * g
            returns.insert(0, g)

        # Update Q-values (every-visit Monte Carlo)
        for state_disc, action, return_val in zip(
            self.episode_states,
            self.episode_actions,
            returns,
        ):
            # Simple average update (every-visit MC)
            old_q = float(self.Q[state_disc + (action,)])
            self.Q[state_disc + (action,)] = old_q + 0.1 * (return_val - old_q)

        # Decay exploration
        self.strategy.decay()

        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def get_hyperparams(self) -> Dict[str, Any]:
        """Return current hyperparameters."""
        return {
            "gamma": self.gamma,
            "epsilon": self.strategy.get_epsilon(),
            "n_pos_bins": self.discretizer.n_pos_bins,
            "n_vel_bins": self.discretizer.n_vel_bins,
        }

    def set_hyperparams(self, params: Dict[str, Any]) -> None:
        """Update hyperparameters from dictionary."""
        if "gamma" in params:
            self.gamma = float(params["gamma"])

    def save(self, path: str) -> None:
        """Save Q-table and visit counts."""
        np.savez(path, Q=self.Q, visit_counts=self.visit_counts)

    def load(self, path: str) -> None:
        """Load Q-table and visit counts."""
        data = np.load(path)
        self.Q = data["Q"]
        self.visit_counts = data["visit_counts"]

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about learned Q-values."""
        return {
            "q_mean": float(np.mean(self.Q)),
            "q_std": float(np.std(self.Q)),
            "q_max": float(np.max(self.Q)),
            "visited_states": int(np.sum(self.visit_counts > 0)),
            "total_visits": int(np.sum(self.visit_counts)),
        }


# Aliases for consistency across naming conventions
QLearningAgent = QLearning
SARSAAgent = SARSA
