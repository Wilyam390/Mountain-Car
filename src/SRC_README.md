# Mountain Car RL Infrastructure

This directory contains the reusable infrastructure for all phases of the Mountain Car comparative study.

## Overview

The `src/` module is designed to be **generic, modular, and parameterizable** so that:
1. Different environments can be created with a single function call
2. Any agent implementation works with the same training loop
3. Metrics and statistics are collected uniformly across all experiments
4. Code is reused across all 8 phases (no duplication)

## Architecture

```
src/
├── environment_utils.py    # Environment creation & state handling
├── evaluation.py           # Training loops, metrics, statistics
├── agents/
│   ├── base_agent.py      # Abstract agent interface
│   ├── tabular_agents.py  # Q-learning, SARSA (to implement)
│   ├── monte_carlo_agent.py  # Monte Carlo (to implement)
│   ├── dqn_agent.py       # DQN, DDQN (to implement)
│   └── policy_gradient_agent.py  # REINFORCE, Actor-Critic (to implement)
└── __init__.py
```

## Core Components

### 1. Environment Utils (`environment_utils.py`)

**Purpose**: Create and configure environments with different reward schemes.

**Key Classes**:
- `create_env(env_type, scenario, seed)` - Factory function
- `StateDiscretizer(n_pos_bins, n_vel_bins)` - Converts continuous to discrete states
- `StateNormalizer()` - Normalizes states to [-1, 1] (for neural nets)
- Reward wrappers:
  - `StandardRewardWrapper` - Discrete, min-steps (default)
  - `MinFuelRewardWrapper` - Discrete, min-fuel
  - `LinearActionCostWrapper` - Continuous, linear penalty

**Usage**:
```python
from src.environment_utils import create_env, StateDiscretizer

# Create different environments
env_discrete = create_env("discrete", "min_steps", seed=42)
env_fuel = create_env("discrete", "min_fuel", seed=42)
env_continuous = create_env("continuous", "quadratic_cost", seed=42)

# Discretize states
discretizer = StateDiscretizer(n_pos_bins=20, n_vel_bins=20)
state_discrete = discretizer.discretize(continuous_state)
```

**Tweaking Parameters**:
- Change `n_pos_bins`, `n_vel_bins` for Phase 3 (discretization analysis)
- Change `scenario` to test different reward structures (Phase 2)
- Change `env_type` for discrete vs continuous (Phase 6)

---

### 2. Evaluation (`evaluation.py`)

**Purpose**: Training loops, metrics collection, statistical analysis.

**Key Functions**:
- `train_agent(agent, env, n_episodes, ...)` - Generic training loop
- `evaluate_agent(agent, env, ...)` - Evaluate trained agent (greedy policy)
- `export_results(metrics, path)` - Save results to JSON

**Key Classes**:
- `EpisodeMetrics` - Single episode metrics (reward, steps, success, etc.)
- `StatisticalAnalyzer` - Aggregate metrics across seeds, compute CI, convergence
- `moving_average()`, `exponential_smoothing()` - Utility smoothing functions

**Usage**:
```python
from src.evaluation import train_agent, evaluate_agent, StatisticalAnalyzer

# Train agent
metrics, log_dir = train_agent(
    agent=my_agent,
    env=my_env,
    n_episodes=5000,
    eval_freq=500,
    log_dir="./logs/exp1",
    seed=42
)

# Evaluate
eval_metrics, avg_reward, success_rate, avg_steps = evaluate_agent(
    agent=my_agent,
    env=my_env,
    n_eval_episodes=100,
    seed=42
)

# Analyze across multiple seeds
analyzer = StatisticalAnalyzer()
stats = analyzer.aggregate_metrics(
    {42: metrics_seed42, 123: metrics_seed123, 999: metrics_seed999},
    method_name="Q-learning",
    scenario_name="min_steps"
)
```

**Tweaking Parameters**:
- Adjust `n_episodes` for faster/slower training
- Change `eval_freq` for more/less frequent logging
- Set `log_dir` for tensorboard visualization
- Run with multiple seeds for statistical rigor

---

### 3. Base Agent (`agents/base_agent.py`)

**Purpose**: Abstract interface that all agents must implement.

**Key Classes**:
- `BaseAgent` - Abstract base class defining required methods
- `EpsilonGreedyStrategy` - Reusable exploration strategy
- `DummyAgent` - Example stub for testing

**Required Methods**:
```python
class CustomAgent(BaseAgent):
    def train_step(self, state) -> (action, info_dict):
        """Choose action during training (with exploration)"""
        pass

    def learn(self, state, action, reward, next_state, done):
        """Update agent from experience"""
        pass

    def act(self, state, training=False) -> action:
        """Choose action (greedy if training=False)"""
        pass

    def get_hyperparams(self) -> dict:
        """Return current hyperparameters"""
        pass

    def set_hyperparams(self, params):
        """Update hyperparameters"""
        pass

    def save(self, path):
        """Save trained model/weights"""
        pass

    def load(self, path):
        """Load trained model/weights"""
        pass
```

**Example Q-learning Implementation**:
```python
from src.agents.base_agent import BaseAgent, EpsilonGreedyStrategy
import numpy as np

class QLearning(BaseAgent):
    def __init__(self, n_actions, state_shape, alpha=0.1, gamma=0.99):
        super().__init__(n_actions, state_shape, agent_name="Q-learning")
        self.Q = np.zeros(state_shape + (n_actions,))
        self.alpha = alpha
        self.gamma = gamma
        self.strategy = EpsilonGreedyStrategy()

    def train_step(self, state):
        greedy_action = np.argmax(self.Q[state])
        action = self.strategy.select_action(greedy_action, self.n_actions)
        return action, {"epsilon": self.strategy.get_epsilon()}

    def learn(self, state, action, reward, next_state, done):
        best_next_q = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next_q * (not done)
        self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])
        self.strategy.decay()

    def act(self, state, training=False):
        return np.argmax(self.Q[state])

    # ... other required methods
```

## How to Implement a New Agent

1. **Create file** (e.g., `src/agents/my_agent.py`)
2. **Inherit from `BaseAgent`**
3. **Implement required methods** (see template above)
4. **Use in training loop** (no changes needed to training code!)

Example:
```python
# src/agents/sarsa_agent.py
from src.agents.base_agent import BaseAgent

class SARSA(BaseAgent):
    def __init__(self, n_actions, state_shape, alpha=0.1, gamma=0.99):
        super().__init__(n_actions, state_shape, agent_name="SARSA")
        self.Q = np.zeros(state_shape + (n_actions,))
        # ... init code ...

    def learn(self, state, action, reward, next_state, done):
        # SARSA update (different from Q-learning!)
        next_action = self.strategy.select_action(...)
        q_next = self.Q[next_state][next_action]  # ← On-policy!
        td_target = reward + self.gamma * q_next * (not done)
        # ... rest of update ...
```

Then use it:
```python
from src.agents.sarsa_agent import SARSA

agent = SARSA(n_actions=3, state_shape=(20, 20))
metrics, _ = train_agent(agent, env, n_episodes=5000)
# Same training loop, different agent!
```

---

## Statistical Analysis Workflow

```python
from src.evaluation import StatisticalAnalyzer, EpisodeMetrics

# 1. Run experiments with multiple seeds
metrics_by_seed = {}
for seed in [42, 123, 999]:
    agent = MyAgent(...)
    metrics_by_seed[seed] = train_agent(agent, env, seed=seed)

# 2. Aggregate statistics
analyzer = StatisticalAnalyzer()
stats = analyzer.aggregate_metrics(
    metrics_by_seed,
    method_name="Q-learning",
    scenario_name="min_steps"
)

# 3. Access results
print(stats["mean_across_seeds"]["avg_reward"])  # Mean ± std
print(stats["mean_across_seeds"]["success_rate"])

# 4. Compute confidence intervals
ci = analyzer.compute_confidence_interval(np.array([...]))

# 5. Analyze convergence
convergence = analyzer.compute_convergence_metrics(rewards_list)
```

---

## Tensorboard Monitoring

Training with tensorboard logging:
```python
metrics, log_dir = train_agent(
    agent, env,
    n_episodes=5000,
    log_dir="./logs/q_learning_discrete_min_steps"
)
```

View in tensorboard:
```bash
tensorboard --logdir=./logs
```

Then navigate to `http://localhost:6006` in browser.


## Tips

1. **Always use `seed`** parameter for reproducibility
2. **Save results to JSON** with `export_results()` for later analysis
3. **Use tensorboard** to visualize training in real-time
4. **Run multiple seeds** for statistical significance 
5. **Profile code** to identify bottlenecks before scaling
6. **Version your experiments** (e.g., `logs/phase3_q_learning_v1/`)
7. **Document hyperparameters** in experiment metadata

---

## File Structure After Running

```
Mountain-Car/
├── src/
│   ├── environment_utils.py
│   ├── evaluation.py
│   └── agents/
│       ├── base_agent.py
│       ├── tabular_agents.py (to implement)
│       └── ...
├── notebooks/
│   ├── 01_environment_and_baselines.ipynb (started)
│   ├── 02_reward_design_and_scenarios.ipynb
│   └── ...
├── logs/  ← Generated during training
│   ├── phase1_qlearning/
│   ├── phase2_reward_scenarios/
│   └── ...
└── results/  ← Generated results
    ├── metrics_qlearning_discrete_minsteps.json
    └── ...
```

---