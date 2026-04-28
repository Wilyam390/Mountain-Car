#!/usr/bin/env python
"""
Debug script for Phase 7: Check environment setup and DQN agent
Verifies:
1. Environment (continuous) can be reset and stepped
2. Random policy can reach goal
3. DQN action discretization works
4. DQN can train for a few steps without errors
"""

import numpy as np
import sys
from pathlib import Path
import os

# Add src to path
# Go up one level from /notebooks/ to /Mountain-Car/
root_path = os.path.abspath("..")

if root_path not in sys.path:
    sys.path.append(root_path)


# Add src to path for imports
project_root = Path(".").resolve().parent
sys.path.insert(0, str(project_root / "src"))

from src.environment_utils import create_env
from src.agents.continuous_agents import DQNAgent

print("=" * 70)
print("PHASE 7 DEBUG SCRIPT: Environment & DQN Setup Check")
print("=" * 70)

# ============================================================================
# 1. ENVIRONMENT BASIC CHECKS
# ============================================================================
print("\n[1] Environment Basic Setup (Continuous)")
print("-" * 70)

try:
    env = create_env(env_type="continuous", scenario="continuous_standard", seed=42)
    print(f"✓ Environment created via create_env()")
    print(f"  - Env type: continuous")
    print(f"  - Scenario: continuous_standard")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    # Note: continuous env may not have max_episode_steps in spec
    if hasattr(env, '_max_episode_steps'):
        print(f"  - Max episode steps: {env._max_episode_steps}")
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 2. TEST RANDOM POLICY (should reach goal sometimes)
# ============================================================================
print("\n[2] Random Policy Test (10 episodes)")
print("-" * 70)

successes = 0
max_steps_seen = 0

for ep in range(10):
    obs, info = env.reset()
    episode_success = False

    for step in range(500):
        action = env.action_space.sample()  # Random action in [-1, 1]
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:  # Mountain Car terminates only on goal reach
            successes += 1
            episode_success = True
            max_steps_seen = max(max_steps_seen, step + 1)
            print(f"  Episode {ep+1}: SUCCESS in {step+1} steps")
            break

    if not episode_success:
        max_steps_seen = max(max_steps_seen, 500)

if successes == 0:
    print(f"  ⚠ WARNING: Random policy never reached goal (0/10)")
    print(f"    This suggests the environment or success condition may be broken.")
else:
    print(f"✓ Random policy succeeded {successes}/10 times")
    print(f"  Max steps seen: {max_steps_seen}")

# ============================================================================
# 3. TEST DQN INITIALIZATION
# ============================================================================
print("\n[3] DQN Initialization & Action Bins")
print("-" * 70)

try:
    agent = DQNAgent(n_actions=11, state_shape=(2,), learning_rate=0.001)
    print(f"✓ DQNAgent created")
    print(f"  - State shape: (2,)")
    print(f"  - N actions: 11")
    print(f"  - Learning rate: 0.001")
    print(f"  - Gamma: {agent.gamma}")
    print(f"  - Epsilon: {agent.epsilon}")
    print(f"  - Epsilon decay: {agent.epsilon_decay}")
    print(f"  - Update freq: {agent.update_freq}")
except Exception as e:
    print(f"✗ Failed to create DQNAgent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check action bins
print(f"\n  Action bins ({len(agent.action_bins)} total):")
print(f"    {agent.action_bins}")
if len(agent.action_bins) != 11:
    print(f"    ✗ WARNING: Expected 11 bins, got {len(agent.action_bins)}")
else:
    print(f"    ✓ Correct number of bins")

if not (agent.action_bins.min() >= -1.0 and agent.action_bins.max() <= 1.0):
    print(f"    ✗ WARNING: Action bins outside [-1, 1] range")
else:
    print(f"    ✓ Bins within [-1, 1]")

# ============================================================================
# 4. TEST DQN ACTIONS ON KNOWN STATES
# ============================================================================
print("\n[4] DQN Action Output on Test States")
print("-" * 70)

test_states = [
    np.array([0.0, 0.0]),      # Center
    np.array([-0.9, 0.0]),     # Left side
    np.array([0.3, 0.05]),     # Near goal
]

for i, state in enumerate(test_states):
    try:
        action = agent.act(state)
        print(f"  State {i} {state}: action = {action} (type: {type(action).__name__})")

        if not isinstance(action, np.ndarray):
            print(f"    ✗ WARNING: Action is not np.ndarray")
        elif action.shape != (1,):
            print(f"    ✗ WARNING: Action shape is {action.shape}, expected (1,)")
        elif not (-1 <= action[0] <= 1):
            print(f"    ✗ WARNING: Action value {action[0]} outside [-1, 1]")
        else:
            print(f"    ✓ Valid action format")
    except Exception as e:
        print(f"  State {i}: ✗ ERROR: {e}")

# ============================================================================
# 5. TEST DQN TRAINING STEP
# ============================================================================
print("\n[5] DQN Training Step (1 episode)")
print("-" * 70)

try:
    obs, _ = env.reset()
    total_reward = 0

    for step in range(100):  # Just 100 steps
        # train_step returns (action, info_dict), then learn separately
        action, _ = agent.train_step(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from the transition
        agent.learn(obs, action, reward, next_obs, terminated or truncated)
        total_reward += reward

        obs = next_obs

        if terminated or truncated:
            break

    print(f"✓ Training step completed")
    print(f"  - Steps taken: {step + 1}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Agent epsilon after training: {agent.epsilon:.6f}")
    print(f"  - Buffer size: {len(agent.buffer)} transitions stored")

except Exception as e:
    print(f"✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 6. QUICK TRAINING RUN (100 episodes)
# ============================================================================
print("\n[6] Quick Training Run (100 episodes, continuous_standard scenario)")
print("-" * 70)

successes = 0
total_rewards = []

for ep in range(100):
    obs, _ = env.reset()
    episode_reward = 0

    for step in range(500):
        # train_step returns (action, info_dict)
        action, _ = agent.train_step(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from transition
        agent.learn(obs, action, reward, next_obs, terminated or truncated)
        episode_reward += reward

        obs = next_obs

        if terminated or truncated:
            if terminated:  # Terminated = reached goal
                successes += 1
            break

    total_rewards.append(episode_reward)

    if (ep + 1) % 25 == 0:
        avg_reward = np.mean(total_rewards[-25:])
        print(f"  Episodes {ep-24:3d}-{ep+1:3d}: "
              f"Avg Reward: {avg_reward:8.2f}, "
              f"Success Rate: {successes}%, "
              f"Epsilon: {agent.epsilon:.6f}")

print(f"\n✓ Training complete")
print(f"  - Total successes: {successes}/100")
print(f"  - Avg reward (last 25): {np.mean(total_rewards[-25:]):.2f}")
print(f"  - Final epsilon: {agent.epsilon:.6f}")

if successes == 0:
    print(f"\n  ⚠ WARNING: No successes after 100 episodes!")
    print(f"    Debugging suggestions:")
    print(f"    1. Check if random policy can succeed (see above)")
    print(f"    2. Try increasing epsilon_decay to 0.9998 (slow decay = more exploration)")
    print(f"    3. Try decreasing learning_rate to 0.0001")
    print(f"    4. Verify action format matches environment expectations")
else:
    print(f"\n  ✓ Learning is working! DQN succeeded {successes} times.")

env.close()

print("\n" + "=" * 70)
print("DEBUG CHECK COMPLETE")
print("=" * 70)
