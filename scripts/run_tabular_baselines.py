
"""
Run Q-learning and SARSA on the standard and fuel-penalized discrete Mountain Car scenarios.

Examples
--------
python scripts/run_tabular_baselines.py
python scripts/run_tabular_baselines.py --episodes 7000 --seeds 0 1 2 3 4
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

from mcrl.envs import make_discrete_env  # noqa: E402
from mcrl.plots import (  # noqa: E402
    collect_greedy_trajectories,
    plot_phase_portrait,
    plot_policy_map,
    plot_success_curve,
    plot_training_curve,
    plot_value_surface,
    plot_visitation_heatmap,
)
from mcrl.tabular import Discretizer, evaluate_greedy_policy, train_q_learning, train_sarsa  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--pos-bins", type=int, default=30)
    parser.add_argument("--vel-bins", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--fuel-cost", type=float, default=0.25)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--outdir", type=str, default="results/tabular")
    return parser.parse_args()


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "discrete_standard": lambda: make_discrete_env(),
        "discrete_fuel": lambda: make_discrete_env(fuel_cost=args.fuel_cost),
    }
    trainers = {
        "q_learning": train_q_learning,
        "sarsa": train_sarsa,
    }

    rows = []

    for scenario_name, env_factory in scenarios.items():
        for algo_name, trainer in trainers.items():
            for seed in args.seeds:
                print("=" * 80)
                print(f"Scenario={scenario_name} | Algorithm={algo_name} | Seed={seed}")

                env = env_factory()
                discretizer = Discretizer.from_env(
                    env,
                    n_pos_bins=args.pos_bins,
                    n_vel_bins=args.vel_bins,
                )

                result = trainer(
                    env,
                    discretizer,
                    episodes=args.episodes,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    epsilon_start=args.epsilon_start,
                    epsilon_decay=args.epsilon_decay,
                    epsilon_min=args.epsilon_min,
                    seed=seed,
                    verbose_every=max(1, args.episodes // 10),
                )

                eval_env = env_factory()
                summary = evaluate_greedy_policy(
                    eval_env,
                    result.q_table,
                    discretizer,
                    n_episodes=args.eval_episodes,
                    seed=10_000 + seed,
                )
                print(json.dumps(summary, indent=2))

                run_dir = outdir / scenario_name / algo_name / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                np.savez_compressed(
                    run_dir / "artifacts.npz",
                    q_table=result.q_table,
                    visit_counts=result.visit_counts,
                    engineered_returns=np.asarray(result.engineered_returns, dtype=np.float32),
                    base_returns=np.asarray(result.base_returns, dtype=np.float32),
                    steps=np.asarray(result.steps, dtype=np.int32),
                    successes=np.asarray(result.successes, dtype=np.int32),
                    fuel_costs=np.asarray(result.fuel_costs, dtype=np.float32),
                    thrust_counts=np.asarray(result.thrust_counts, dtype=np.int32),
                    epsilons=np.asarray(result.epsilons, dtype=np.float32),
                )

                # Main figures for the report.
                plot_training_curve(
                    result.engineered_returns,
                    title=f"{algo_name} | {scenario_name} | engineered return",
                    ylabel="Engineered return",
                )
                save_fig(run_dir / "learning_curve_engineered.png")

                plot_training_curve(
                    result.base_returns,
                    title=f"{algo_name} | {scenario_name} | base return",
                    ylabel="Base return",
                )
                save_fig(run_dir / "learning_curve_base.png")

                plot_success_curve(
                    result.successes,
                    title=f"{algo_name} | {scenario_name} | success rate",
                )
                save_fig(run_dir / "success_curve.png")

                plot_policy_map(
                    result.q_table,
                    discretizer,
                    visit_counts=result.visit_counts,
                    mask_unvisited=True,
                    title=f"{algo_name} | {scenario_name} | policy map",
                )
                save_fig(run_dir / "policy_map.png")

                plot_visitation_heatmap(
                    result.visit_counts,
                    discretizer,
                    title=f"{algo_name} | {scenario_name} | visitation heatmap",
                )
                save_fig(run_dir / "visitation_heatmap.png")

                plot_value_surface(
                    result.q_table,
                    discretizer,
                    mode="max",
                    title=f"{algo_name} | {scenario_name} | max-Q value surface",
                )
                save_fig(run_dir / "value_surface_maxq.png")

                trajectories = collect_greedy_trajectories(
                    env_factory(),
                    result.q_table,
                    discretizer,
                    n_episodes=8,
                    seed=20_000 + seed,
                )
                plot_phase_portrait(
                    trajectories,
                    title=f"{algo_name} | {scenario_name} | greedy trajectories",
                )
                save_fig(run_dir / "phase_portrait.png")

                rows.append(
                    {
                        "scenario": scenario_name,
                        "algorithm": algo_name,
                        "seed": seed,
                        "train_last100_engineered_return": float(np.mean(result.engineered_returns[-100:])),
                        "train_last100_base_return": float(np.mean(result.base_returns[-100:])),
                        "train_last100_success_rate": float(np.mean(result.successes[-100:])),
                        "train_last100_steps": float(np.mean(result.steps[-100:])),
                        "eval_mean_engineered_return": summary["mean_engineered_return"],
                        "eval_std_engineered_return": summary["std_engineered_return"],
                        "eval_mean_base_return": summary["mean_base_return"],
                        "eval_mean_steps": summary["mean_steps"],
                        "eval_success_rate": summary["success_rate"],
                        "eval_mean_fuel_cost": summary["mean_fuel_cost"],
                        "eval_mean_thrust_count": summary["mean_thrust_count"],
                    }
                )

                env.close()
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
        )
        .sort_values(["scenario", "eval_success_rate_mean"], ascending=[True, False])
    )
    aggregate.to_csv(outdir / "summary_aggregate.csv", index=False)
    print("\nSaved:")
    print(f"  - {outdir / 'summary.csv'}")
    print(f"  - {outdir / 'summary_aggregate.csv'}")


if __name__ == "__main__":
    main()
