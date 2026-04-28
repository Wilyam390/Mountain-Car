#!/usr/bin/env python
"""
Minimal debug: Test raw continuous env, then wrapped env.
"""
import numpy as np
import gymnasium as gym

print("=" * 70)
print("TEST 1: Raw MountainCarContinuous-v0 (no wrappers)")
print("=" * 70)

env = gym.make("MountainCarContinuous-v0")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

successes = 0
for ep in range(10):
    obs, _ = env.reset()
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            successes += 1
            print(f"  Ep {ep+1}: SUCCESS in {step+1} steps")
            break

print(f"✓ Raw environment: {successes}/10 successes\n")

env.close()

print("=" * 70)
print("TEST 2: With create_env wrapper")
print("=" * 70)

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from environment_utils import create_env

env = create_env("continuous", "continuous_standard", seed=42)
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

successes = 0
for ep in range(10):
    obs, info = env.reset()
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            successes += 1
            print(f"  Ep {ep+1}: SUCCESS in {step+1} steps, reward={reward:.2f}")
            break

print(f"✓ Wrapped environment: {successes}/10 successes")

env.close()
