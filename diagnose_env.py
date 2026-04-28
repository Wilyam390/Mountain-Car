#!/usr/bin/env python
"""
Diagnose why random policy can't solve the environment.
"""
import gymnasium as gym
import numpy as np

print("=" * 70)
print("DIAGNOSIS: Why can't random policy solve MountainCarContinuous?")
print("=" * 70)

# Test 1: Check raw gymnasium env properties
print("\n[1] Raw Gymnasium Environment Properties")
print("-" * 70)
env = gym.make("MountainCarContinuous-v0")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
print(f"Spec max_episode_steps: {env.spec.max_episode_steps}")
if hasattr(env, '_max_episode_steps'):
    print(f"Env _max_episode_steps: {env._max_episode_steps}")

# Test 2: Run random policy with DETAILED logging
print("\n[2] Random Policy - First 3 Episodes (verbose)")
print("-" * 70)

for ep_num in range(3):
    obs, info = env.reset()
    print(f"\nEpisode {ep_num + 1}:")
    print(f"  Initial state: pos={obs[0]:.4f}, vel={obs[1]:.4f}")

    total_steps = 0
    for step in range(1000):  # Try 1000 steps
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        if step < 5 or step % 100 == 0:
            print(f"    Step {step+1:4d}: pos={obs[0]:7.4f}, vel={obs[1]:7.4f}, reward={reward:7.2f}, terminated={terminated}, truncated={truncated}")

        if terminated:
            print(f"  ✓ SUCCESS in {step + 1} steps")
            break

        if truncated:
            print(f"  ✗ TRUNCATED at step {step + 1} (episode length limit)")
            break
    else:
        print(f"  ✗ FAILED: No success after 1000 steps")

env.close()

# Test 3: Check wrapped environment
print("\n[3] Wrapped Environment via create_env()")
print("-" * 70)

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from environment_utils import create_env

env = create_env("continuous", "continuous_standard", seed=42)
print(f"Wrapped env action space: {env.action_space}")
print(f"Wrapped env spec.max_episode_steps: {env.spec.max_episode_steps}")

obs, info = env.reset()
print(f"Initial state: pos={obs[0]:.4f}, vel={obs[1]:.4f}")

total_steps = 0
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_steps += 1

    if step < 5 or step % 100 == 0:
        print(f"  Step {step+1:4d}: pos={obs[0]:7.4f}, vel={obs[1]:7.4f}, reward={reward:7.2f}, terminated={terminated}, truncated={truncated}")

    if terminated:
        print(f"✓ SUCCESS in {step + 1} steps, final_reward={reward:.2f}")
        break

    if truncated:
        print(f"✗ TRUNCATED at step {step + 1}")
        break
else:
    print(f"✗ FAILED after 1000 steps")

env.close()
