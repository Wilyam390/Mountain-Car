"""
Evaluation and training utilities for Mountain Car RL agents.

Provides:
- Generic training loop (works with any agent)
- Metrics calculation (reward, success rate, steps, efficiency)
- Statistical analysis (mean, std, confidence intervals, convergence)
- Tensorboard integration
- Results export/saving
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import time

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# ============================================================================
# DATA CLASSES & METRICS
# ============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics collected during a single episode."""

    episode: int
    reward: float
    steps: int
    success: bool
    fuel_used: Optional[float] = None
    wall_hits: int = 0
    max_velocity: float = 0.0
    final_position: float = 0.0


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over multiple episodes."""

    method_name: str
    scenario_name: str
    seed: int
    n_episodes: int

    # Performance
    avg_reward: float
    std_reward: float
    min_reward: float
    max_reward: float

    # Success
    success_rate: float
    episodes_to_80pct_success: Optional[int]

    # Steps
    avg_steps: float
    std_steps: float

    # Convergence
    final_100_avg_reward: float
    final_500_avg_reward: Optional[float]

    # Custom
    avg_fuel: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return asdict(self)


# ============================================================================
# GENERIC TRAINING LOOP
# ============================================================================

def train_agent(
    agent: Any,
    env: Any,
    n_episodes: int = 5000,
    seed: int = 42,
    eval_freq: int = 500,
    log_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[List[EpisodeMetrics], str]:
    """
    Generic training loop for any RL agent.

    Trains agent on environment for n_episodes and collects metrics.
    Works with any agent that implements: agent.train_step(state, action, reward, next_state, done)

    Args:
        agent: RL agent with train() method (returns (next_state_disc, reward, done, info))
        env: Gymnasium environment
        n_episodes: Total training episodes
        seed: Random seed for reproducibility
        eval_freq: Evaluation frequency (log metrics every N episodes)
        log_dir: Directory for tensorboard logs (optional)
        verbose: Print progress to stdout

    Returns:
        (episode_metrics_list, log_dir_path)
        - List of EpisodeMetrics for each episode
        - Path to tensorboard logs (if provided)
    """
    np.random.seed(seed)
    env.reset(seed=seed)

    metrics_list = []
    writer = None

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if HAS_TENSORBOARD:
            try:
                writer = SummaryWriter(log_dir=str(log_dir))
            except Exception as e:
                print(f"Warning: Failed to initialize TensorBoard writer: {e}")
        else:
            print(f"Warning: TensorBoard not available. Logs will be saved to {log_dir} but not viewable in TensorBoard.")

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        success = False

        while True:
            # Agent decides action and learns
            action, learn_info = agent.train_step(state)

            # Environment step
            next_state, reward, terminated, truncated, env_info = env.step(action)
            done = terminated or truncated

            # Accumulate metrics
            total_reward += reward
            steps += 1

            # Agent learns from transition
            agent.learn(state, action, reward, next_state, done)

            if terminated:
                success = True

            state = next_state

            if done:
                break

        # Record episode metrics
        ep_metrics = EpisodeMetrics(
            episode=episode,
            reward=total_reward,
            steps=steps,
            success=success,
            fuel_used=learn_info.get("fuel_used", None),
            final_position=float(state[0]),
            max_velocity=float(np.abs(state[1])),
        )
        metrics_list.append(ep_metrics)

        # Log to tensorboard
        if writer is not None:
            writer.add_scalar("reward/episode", total_reward, episode)
            writer.add_scalar("steps/episode", steps, episode)
            writer.add_scalar("success/episode", int(success), episode)

        # Evaluation checkpoint
        if (episode + 1) % eval_freq == 0:
            recent_metrics = metrics_list[max(0, episode - eval_freq) : episode + 1]
            recent_rewards = [m.reward for m in recent_metrics]
            recent_success = sum(m.success for m in recent_metrics) / len(recent_metrics)

            if verbose:
                print(
                    f"Episode {episode + 1:5d}/{n_episodes} | "
                    f"Avg Reward: {np.mean(recent_rewards):8.2f} | "
                    f"Success Rate: {recent_success:6.1%}"
                )

            if writer is not None:
                writer.add_scalar("metrics/window_avg_reward", np.mean(recent_rewards), episode)
                writer.add_scalar("metrics/window_success_rate", recent_success, episode)
                writer.flush()

    if writer is not None:
        writer.close()

    # Return log_dir path if it was provided, otherwise empty string
    if log_dir is not None:
        return metrics_list, str(log_dir)

    return metrics_list, ""


def _extract_training_lists(metrics_list: List[EpisodeMetrics]) -> Tuple[List[float], List[int], List[int]]:
    """
    Extract rewards, successes, and steps from metrics list.

    Utility for compatibility with notebooks that expect these lists.
    """
    episode_rewards = [m.reward for m in metrics_list]
    episode_successes = [1 if m.success else 0 for m in metrics_list]
    episode_steps = [m.steps for m in metrics_list]
    return episode_rewards, episode_successes, episode_steps



# ============================================================================
# EVALUATION (GREEDY POLICY)
# ============================================================================

def evaluate_agent(
    agent: Any,
    env: Any,
    n_eval_episodes: int = 100,
    seed: int = 42,
) -> Tuple[List[EpisodeMetrics], float, float, float]:
    """
    Evaluate trained agent using greedy (no exploration) policy.

    Args:
        agent: Trained RL agent with act() method (returns action)
        env: Gymnasium environment
        n_eval_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        (metrics_list, avg_reward, success_rate, avg_steps)
    """
    np.random.seed(seed)
    metrics_list = []

    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        success = False

        while True:
            # Greedy action (no exploration)
            action = agent.act(state, training=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            if terminated:
                success = True

            state = next_state

            if done:
                break

        ep_metrics = EpisodeMetrics(
            episode=episode,
            reward=total_reward,
            steps=steps,
            success=success,
            final_position=float(state[0]),
        )
        metrics_list.append(ep_metrics)

    avg_reward = np.mean([m.reward for m in metrics_list])
    success_rate = np.mean([m.success for m in metrics_list])
    avg_steps = np.mean([m.steps for m in metrics_list])

    return metrics_list, float(avg_reward), float(success_rate), float(avg_steps)


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class StatisticalAnalyzer:
    """Compute statistics over multiple seeds/runs."""

    @staticmethod
    def aggregate_metrics(
        metrics_list_by_seed: Dict[int, List[EpisodeMetrics]],
        method_name: str,
        scenario_name: str,
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple seeds.

        Args:
            metrics_list_by_seed: {seed: [EpisodeMetrics]}
            method_name: Name of training method (e.g., "Q-learning")
            scenario_name: Name of scenario (e.g., "min_steps")

        Returns:
            Dictionary with aggregated statistics
        """
        all_results = []

        for seed, metrics_list in metrics_list_by_seed.items():
            rewards = np.array([m.reward for m in metrics_list])
            steps = np.array([m.steps for m in metrics_list])
            successes = np.array([int(m.success) for m in metrics_list])

            # Find episode to reach 80% success rate
            window_size = 100
            episodes_to_80 = None
            for i in range(window_size, len(successes)):
                window_success = np.mean(successes[i - window_size : i])
                if window_success >= 0.8:
                    episodes_to_80 = i
                    break

            result = AggregatedMetrics(
                method_name=method_name,
                scenario_name=scenario_name,
                seed=seed,
                n_episodes=len(metrics_list),
                avg_reward=float(np.mean(rewards)),
                std_reward=float(np.std(rewards)),
                min_reward=float(np.min(rewards)),
                max_reward=float(np.max(rewards)),
                success_rate=float(np.mean(successes)),
                episodes_to_80pct_success=episodes_to_80,
                avg_steps=float(np.mean(steps)),
                std_steps=float(np.std(steps)),
                final_100_avg_reward=float(np.mean(rewards[-100:])),
                final_500_avg_reward=float(np.mean(rewards[-500:]))
                if len(rewards) >= 500
                else None,
            )
            all_results.append(result)

        return {
            "results_per_seed": [r.to_dict() for r in all_results],
            "mean_across_seeds": {
                "avg_reward": float(np.mean([r.avg_reward for r in all_results])),
                "std_reward": float(np.std([r.avg_reward for r in all_results])),
                "success_rate": float(np.mean([r.success_rate for r in all_results])),
                "avg_steps": float(np.mean([r.avg_steps for r in all_results])),
                "episodes_to_80pct": float(
                    np.mean([r.episodes_to_80pct_success for r in all_results if r.episodes_to_80pct_success])
                )
                if any(r.episodes_to_80pct_success for r in all_results)
                else None,
            },
        }

    @staticmethod
    def compute_confidence_interval(
        data: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for data (assumes normal distribution).

        Args:
            data: Array of values
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            (lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        std_err = np.std(data) / np.sqrt(n)
        # Approximation: 1.96 for 95% CI (z-score)
        z_score = 1.96 if confidence == 0.95 else 2.576  # 99% CI
        margin = z_score * std_err
        return (mean - margin, mean + margin)

    @staticmethod
    def compute_convergence_metrics(
        rewards: List[float], window_size: int = 100
    ) -> Dict[str, float]:
        """
        Analyze convergence of training.

        Args:
            rewards: List of episode rewards
            window_size: Size of moving average window

        Returns:
            Dictionary with convergence metrics
        """
        rewards = np.array(rewards)
        windows = [
            np.mean(rewards[i - window_size : i])
            for i in range(window_size, len(rewards))
        ]
        windows = np.array(windows)

        # Stability: std of recent windows
        final_windows = windows[-10:] if len(windows) > 10 else windows
        stability = float(np.std(final_windows))

        # Convergence speed: abrupt changes in moving average
        if len(windows) > 10:
            early_avg = np.mean(windows[:10])
            late_avg = np.mean(windows[-10:])
            improvement = late_avg - early_avg
        else:
            improvement = 0.0

        return {
            "stability_score": stability,
            "improvement": float(improvement),
            "final_window_mean": float(np.mean(final_windows)),
            "final_window_std": float(np.std(final_windows)),
        }


# ============================================================================
# RESULTS EXPORT
# ============================================================================

def export_results(
    metrics_list: List[EpisodeMetrics],
    output_path: str,
    metadata: Optional[Dict] = None,
):
    """
    Export episode metrics to JSON file.

    Args:
        metrics_list: List of EpisodeMetrics
        output_path: Path to save JSON
        metadata: Optional metadata to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": metadata or {},
        "episodes": [
            {
                "episode": m.episode,
                "reward": m.reward,
                "steps": m.steps,
                "success": m.success,
                "fuel_used": m.fuel_used,
                "final_position": m.final_position,
            }
            for m in metrics_list
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results exported to {output_path}")


# ============================================================================
# MOVING AVERAGE & SMOOTHING
# ============================================================================

def moving_average(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute moving average with given window size."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def exponential_smoothing(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Apply exponential smoothing to data.

    Args:
        data: Input data
        alpha: Smoothing factor (0-1). Higher = more responsive.

    Returns:
        Smoothed data
    """
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


if __name__ == "__main__":
    print("Testing evaluation utilities...")

    # Create dummy metrics
    metrics = [
        EpisodeMetrics(episode=i, reward=-100 - i, steps=50 + i, success=i > 80)
        for i in range(100)
    ]

    rewards = np.array([m.reward for m in metrics])
    print(f"Sample metrics created: {len(metrics)} episodes")

    # Test statistics
    analyzer = StatisticalAnalyzer()
    convergence = analyzer.compute_convergence_metrics([m.reward for m in metrics])
    print(f"Convergence metrics: {convergence}")

    # Test confidence interval
    ci = analyzer.compute_confidence_interval(rewards)
    print(f"95% CI for rewards: {ci}")

    # Test moving average
    ma = moving_average(rewards, window=10)
    print(f"Moving average length: {len(ma)} (original: {len(rewards)})")

    print("✓ All evaluation utilities working!")
