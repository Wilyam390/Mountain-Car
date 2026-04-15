"""
Abstract base class for Mountain Car RL agents.

All agents (Q-learning, SARSA, DQN, Policy Gradient, etc.) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class defining a common interface for all RL agents.

    Subclasses must implement:
        - train_step(state): Decide action during training
        - learn(state, action, reward, next_state, done): Learn from transition
        - act(state, training): Decide action (exploration or greedy)
        - get_hyperparams(): Return current hyperparameters
        - set_hyperparams(params): Set hyperparameters
        - save(path): Save trained model/weights
        - load(path): Load trained model/weights
    """

    def __init__(
        self,
        n_actions: int,
        state_shape: Tuple,
        agent_name: str = "BaseAgent",
        **kwargs
    ):
        """
        Initialize agent.

        Args:
            n_actions: Number of discrete actions
            state_shape: Shape of state (e.g., (2,) for Mountain Car, or (20, 20) for discretized)
            agent_name: Name of agent for logging/identification
            **kwargs: Additional hyperparameters
        """
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.agent_name = agent_name
        self.step_count = 0

    # ========================================================================
    # REQUIRED INTERFACE
    # ========================================================================

    @abstractmethod
    def train_step(self, state: Any) -> Tuple[Any, Dict]:
        """
        Select action during training (with exploration).

        Args:
            state: Current state

        Returns:
            (action, info_dict)
            - action: Selected action
            - info_dict: Metadata (e.g., epsilon, Q-values)
        """
        pass

    @abstractmethod
    def learn(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """
        Update agent based on experience (state, action, reward, next_state, done).

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        pass

    @abstractmethod
    def act(self, state: Any, training: bool = False) -> Any:
        """
        Select action given state (without exploration during eval).

        Args:
            state: Current state
            training: If True, may use exploration. If False, greedy policy.

        Returns:
            action
        """
        pass

    @abstractmethod
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return dictionary of current hyperparameters."""
        pass

    @abstractmethod
    def set_hyperparams(self, params: Dict[str, Any]) -> None:
        """Set hyperparameters from dictionary."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save trained agent (weights, Q-table, policy, etc.) to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load trained agent from disk."""
        pass

    # ========================================================================
    # OPTIONAL UTILS (can override)
    # ========================================================================

    def reset(self) -> None:
        """Reset any internal counters (called at episode start if needed)."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return current statistics about the agent (e.g., Q-value statistics).
        Optional, default returns empty dict.
        """
        return {}

    def __repr__(self) -> str:
        return f"{self.agent_name}(n_actions={self.n_actions}, state_shape={self.state_shape})"


# ============================================================================
# EPSILON-GREEDY STRATEGY (COMMON)
# ============================================================================

class EpsilonGreedyStrategy:
    """
    Epsilon-greedy exploration strategy.

    With probability epsilon, take random action.
    With probability 1-epsilon, take greedy action.
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
    ):
        """
        Initialize strategy.

        Args:
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate (never go below this)
            epsilon_decay: Multiplicative decay per step (0-1)
        """
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.step_count = 0

    def select_action(self, greedy_action: int, n_actions: int) -> int:
        """
        Select action using epsilon-greedy.

        Args:
            greedy_action: Best action according to current policy
            n_actions: Total number of actions

        Returns:
            Selected action (int)
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(n_actions)
        return greedy_action

    def decay(self) -> None:
        """Decay epsilon after each step."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1

    def get_epsilon(self) -> float:
        """Return current epsilon value."""
        return self.epsilon

    def reset(self, epsilon: float = None) -> None:
        """Reset epsilon to start value."""
        self.epsilon = epsilon or self.epsilon_start
        self.step_count = 0


# ============================================================================
# EXAMPLE STUB FOR QUICK TESTING
# ============================================================================

class DummyAgent(BaseAgent):
    """
    Dummy agent that takes random actions (for testing infrastructure).
    """

    def __init__(self, n_actions: int, state_shape: Tuple):
        super().__init__(n_actions, state_shape, agent_name="DummyAgent")

    def train_step(self, state: Any) -> Tuple[Any, Dict]:
        action = np.random.randint(self.n_actions)
        return action, {"dummy": True}

    def learn(self, state, action, reward, next_state, done):
        pass

    def act(self, state: Any, training: bool = False) -> Any:
        return np.random.randint(self.n_actions)

    def get_hyperparams(self) -> Dict:
        return {"n_actions": self.n_actions}

    def set_hyperparams(self, params: Dict) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


if __name__ == "__main__":
    print("Testing base agent interface...")

    agent = DummyAgent(n_actions=3, state_shape=(2,))
    print(f"Created agent: {agent}")

    # Test interface
    state = np.array([0.5, 0.1])
    action, info = agent.train_step(state)
    print(f"Action during training: {action}, info: {info}")

    greedy_action = agent.act(state, training=False)
    print(f"Greedy action: {greedy_action}")

    agent.learn(state, action, reward=-1.0, next_state=state, done=False)
    print("✓ Learn step executed")

    print(f"Hyperparams: {agent.get_hyperparams()}")

    print("\nTesting epsilon-greedy strategy...")
    strategy = EpsilonGreedyStrategy(epsilon_start=1.0, epsilon_decay=0.99)
    print(f"Initial epsilon: {strategy.get_epsilon()}")

    for i in range(10):
        strategy.decay()

    print(f"Epsilon after 10 steps: {strategy.get_epsilon():.4f}")

    print("\n✓ Base agent interface working!")
