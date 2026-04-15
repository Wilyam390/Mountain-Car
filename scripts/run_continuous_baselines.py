"""Train PPO and SAC on continuous Mountain Car scenarios.

This script is the next stage after the tabular discrete baseline.
It supports:
- continuous_standard: original MountainCarContinuous-v0 reward
- continuous_fuel_binary: adapted reward with a fixed cost for non-null actions
- continuous_fuel_l1: adapted reward with an L1 action penalty

Examples
--------
python scripts/run_continuous_baselines.py --timesteps 80000 --seeds 0 1 2
python scripts/run_continuous_baselines.py --algorithms sac --scenarios continuous_standard continuous_fuel_binary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from mcrl.continuous_utils import (  # noqa: E402
    collect_model_trajectories,
    evaluate_continuous_model,
    plot_action_surface,
    plot_eval_curve_from_npz,
)
from mcrl.envs import make_continuous_env  # noqa: E402
from mcrl.plots import plot_phase_portrait  # noqa: E402

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:  # pragma: no cover - handled for user friendliness
    raise ImportError(
        "stable-baselines3 is not installed. Add 'stable-baselines3[extra]' to requirements.txt "
        "and run pip install -r requirements.txt before launching this script."
    ) from exc


ALGO_CLASSES = {
    "ppo": PPO,
    "sac": SAC,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", choices=["ppo", "sac"], default=["ppo", "sac"])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["continuous_standard", "continuous_fuel_binary", "continuous_fuel_l1"],
        default=["continuous_standard", "continuous_fuel_binary"],
    )
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cost-coef", type=float, default=0.20)
    parser.add_argument("--non-null-threshold", type=float, default=0.05)
    parser.add_argument("--per-step-time-cost", type=float, default=0.0)
    parser.add_argument("--goal-velocity", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--outdir", type=str, default="results/continuous")
    return parser.parse_args()


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def make_env_factory(scenario: str, args: argparse.Namespace):
    if scenario == "continuous_standard":
        return lambda: make_continuous_env(
            goal_velocity=args.goal_velocity,
            adapted=False,
            non_null_threshold=args.non_null_threshold,
        )
    if scenario == "continuous_fuel_binary":
        return lambda: make_continuous_env(
            goal_velocity=args.goal_velocity,
            adapted=True,
            fuel_mode="binary",
            cost_coef=args.cost_coef,
            non_null_threshold=args.non_null_threshold,
            per_step_time_cost=args.per_step_time_cost,
        )
    if scenario == "continuous_fuel_l1":
        return lambda: make_continuous_env(
            goal_velocity=args.goal_velocity,
            adapted=True,
            fuel_mode="l1",
            cost_coef=args.cost_coef,
            non_null_threshold=args.non_null_threshold,
            per_step_time_cost=args.per_step_time_cost,
        )
    raise ValueError(f"Unknown scenario: {scenario}")


def build_model(algo_name: str, env, *, seed: int, device: str, tensorboard_log: str):
    if algo_name == "ppo":
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.98,
            ent_coef=0.0,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[64, 64]),
            tensorboard_log=tensorboard_log,
            seed=seed,
            device=device,
            verbose=1,
        )
    if algo_name == "sac":
        return SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log=tensorboard_log,
            seed=seed,
            device=device,
            verbose=1,
        )
    raise ValueError(f"Unknown algorithm: {algo_name}")


def main() -> None:
    args = parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for scenario in args.scenarios:
        env_factory = make_env_factory(scenario, args)
        probe_env = env_factory()
        obs_low = np.asarray(probe_env.observation_space.low, dtype=np.float32)
        obs_high = np.asarray(probe_env.observation_space.high, dtype=np.float32)
        probe_env.close()

        for algo_name in args.algorithms:
            model_cls = ALGO_CLASSES[algo_name]
            for seed in args.seeds:
                print("=" * 80)
                print(f"Scenario={scenario} | Algorithm={algo_name} | Seed={seed}")

                run_dir = outdir / scenario / algo_name / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                train_env = Monitor(env_factory(), filename=str(run_dir / "train_monitor.csv"))
                eval_env = Monitor(env_factory(), filename=str(run_dir / "eval_monitor.csv"))

                try:
                    model = build_model(
                        algo_name,
                        train_env,
                        seed=seed,
                        device=args.device,
                        tensorboard_log=str(run_dir / "tb"),
                    )

                    eval_callback = EvalCallback(
                        eval_env,
                        best_model_save_path=str(run_dir / "best_model"),
                        log_path=str(run_dir / "eval_callback"),
                        eval_freq=args.eval_freq,
                        n_eval_episodes=args.eval_episodes,
                        deterministic=True,
                        render=False,
                    )

                    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=False)
                    model.save(run_dir / "final_model")

                    best_model_path = run_dir / "best_model" / "best_model.zip"
                    selected_model = (
                        model_cls.load(best_model_path, device=args.device)
                        if best_model_path.exists()
                        else model
                    )
                    selected_model_name = "best" if best_model_path.exists() else "final"

                    summary = evaluate_continuous_model(
                        env_factory,
                        selected_model,
                        n_episodes=args.eval_episodes,
                        deterministic=True,
                        seed=10_000 + seed,
                    )
                    summary["selected_model"] = selected_model_name
                    summary["timesteps"] = args.timesteps
                    print(json.dumps(summary, indent=2))

                    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2)

                    eval_npz = run_dir / "eval_callback" / "evaluations.npz"
                    if eval_npz.exists():
                        plot_eval_curve_from_npz(
                            eval_npz,
                            title=f"{algo_name} | {scenario} | eval curve",
                        )
                        save_fig(run_dir / "training_eval_curve.png")

                    plot_action_surface(
                        selected_model,
                        low=obs_low,
                        high=obs_high,
                        title=f"{algo_name} | {scenario} | action surface",
                    )
                    save_fig(run_dir / "action_surface.png")

                    trajectories = collect_model_trajectories(
                        env_factory,
                        selected_model,
                        n_episodes=8,
                        deterministic=True,
                        seed=20_000 + seed,
                    )
                    plot_phase_portrait(
                        trajectories,
                        title=f"{algo_name} | {scenario} | greedy trajectories",
                    )
                    save_fig(run_dir / "phase_portrait.png")

                    rows.append(
                        {
                            "scenario": scenario,
                            "algorithm": algo_name,
                            "seed": seed,
                            "selected_model": selected_model_name,
                            "timesteps": args.timesteps,
                            "eval_mean_engineered_return": summary["mean_engineered_return"],
                            "eval_std_engineered_return": summary["std_engineered_return"],
                            "eval_mean_base_return": summary["mean_base_return"],
                            "eval_mean_steps": summary["mean_steps"],
                            "eval_success_rate": summary["success_rate"],
                            "eval_mean_fuel_cost": summary["mean_fuel_cost"],
                            "eval_mean_quadratic_cost": summary["mean_quadratic_cost"],
                            "eval_mean_non_null_count": summary["mean_non_null_count"],
                            "eval_mean_action_abs": summary["mean_action_abs"],
                        }
                    )
                finally:
                    train_env.close()
                    eval_env.close()

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "summary.csv", index=False)
    aggregate = (
        df.groupby(["scenario", "algorithm"], as_index=False)
        .agg(
            eval_mean_engineered_return_mean=("eval_mean_engineered_return", "mean"),
            eval_mean_engineered_return_std=("eval_mean_engineered_return", "std"),
            eval_success_rate_mean=("eval_success_rate", "mean"),
            eval_mean_steps_mean=("eval_mean_steps", "mean"),
            eval_mean_fuel_cost_mean=("eval_mean_fuel_cost", "mean"),
            eval_mean_non_null_count_mean=("eval_mean_non_null_count", "mean"),
        )
        .sort_values(["scenario", "eval_mean_engineered_return_mean"], ascending=[True, False])
    )
    aggregate.to_csv(outdir / "summary_aggregate.csv", index=False)
    print("\nSaved:")
    print(f"  - {outdir / 'summary.csv'}")
    print(f"  - {outdir / 'summary_aggregate.csv'}")


if __name__ == "__main__":
    main()
