"""Small follow-up sweeps for the discrete tabular stage.

This script is intentionally narrower than run_tabular_baselines.py.
It focuses on the questions we still need to settle before moving on:
1) Is 30x30 discretization still the best practical choice?
2) How sensitive is the fuel-penalty scenario to the chosen fuel cost?
3) With the chosen settings, how do Q-learning and SARSA compare over 5 seeds?

Examples
--------
python scripts/run_tabular_cleanup.py --task bin_sweep
python scripts/run_tabular_cleanup.py --task fuel_sweep
python scripts/run_tabular_cleanup.py --task final_compare
python scripts/run_tabular_cleanup.py --task all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from mcrl.envs import make_discrete_env  # noqa: E402
from mcrl.tabular import Discretizer, evaluate_greedy_policy, train_q_learning, train_sarsa  # noqa: E402


TRAINERS = {
    "q_learning": train_q_learning,
    "sarsa": train_sarsa,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["bin_sweep", "fuel_sweep", "final_compare", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=7000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--bin-values", type=int, nargs="+", default=[20, 30, 40])
    parser.add_argument("--fuel-cost-values", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    parser.add_argument("--best-pos-bins", type=int, default=30)
    parser.add_argument("--best-vel-bins", type=int, default=30)
    parser.add_argument("--best-fuel-cost", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--outdir", type=str, default="results/tabular_cleanup")
    return parser.parse_args()


def train_and_eval(
    *,
    algorithm: str,
    scenario: str,
    seed: int,
    pos_bins: int,
    vel_bins: int,
    fuel_cost: float,
    episodes: int,
    eval_episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_decay: float,
    epsilon_min: float,
) -> dict:
    if scenario == "discrete_standard":
        env_factory = lambda: make_discrete_env()
    elif scenario == "discrete_fuel":
        env_factory = lambda: make_discrete_env(fuel_cost=fuel_cost)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    env = env_factory()
    try:
        discretizer = Discretizer.from_env(env, n_pos_bins=pos_bins, n_vel_bins=vel_bins)
        trainer = TRAINERS[algorithm]
        result = trainer(
            env,
            discretizer,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            seed=seed,
            verbose_every=max(1, episodes // 5),
        )
    finally:
        env.close()

    eval_env = env_factory()
    try:
        summary = evaluate_greedy_policy(
            eval_env,
            result.q_table,
            discretizer,
            n_episodes=eval_episodes,
            seed=10_000 + seed,
        )
    finally:
        eval_env.close()

    return {
        "scenario": scenario,
        "algorithm": algorithm,
        "seed": seed,
        "pos_bins": pos_bins,
        "vel_bins": vel_bins,
        "fuel_cost": fuel_cost,
        "train_last100_engineered_return": float(np.mean(result.engineered_returns[-100:])),
        "train_last100_success_rate": float(np.mean(result.successes[-100:])),
        "eval_mean_engineered_return": summary["mean_engineered_return"],
        "eval_std_engineered_return": summary["std_engineered_return"],
        "eval_mean_base_return": summary["mean_base_return"],
        "eval_mean_steps": summary["mean_steps"],
        "eval_success_rate": summary["success_rate"],
        "eval_mean_fuel_cost": summary["mean_fuel_cost"],
        "eval_mean_thrust_count": summary["mean_thrust_count"],
    }


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            eval_mean_engineered_return_mean=("eval_mean_engineered_return", "mean"),
            eval_mean_engineered_return_std=("eval_mean_engineered_return", "std"),
            eval_success_rate_mean=("eval_success_rate", "mean"),
            eval_success_rate_std=("eval_success_rate", "std"),
            eval_mean_steps_mean=("eval_mean_steps", "mean"),
            eval_mean_fuel_cost_mean=("eval_mean_fuel_cost", "mean"),
            eval_mean_thrust_count_mean=("eval_mean_thrust_count", "mean"),
        )
        .sort_values(group_cols)
    )


def run_bin_sweep(args: argparse.Namespace) -> None:
    print("\n=== BIN SWEEP (SARSA, standard discrete scenario) ===")
    rows = []
    for bins in args.bin_values:
        for seed in args.seeds:
            print(f"bins={bins}x{bins} | seed={seed}")
            rows.append(
                train_and_eval(
                    algorithm="sarsa",
                    scenario="discrete_standard",
                    seed=seed,
                    pos_bins=bins,
                    vel_bins=bins,
                    fuel_cost=0.0,
                    episodes=args.episodes,
                    eval_episodes=args.eval_episodes,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    epsilon_start=args.epsilon_start,
                    epsilon_decay=args.epsilon_decay,
                    epsilon_min=args.epsilon_min,
                )
            )

    outdir = ROOT / args.outdir / "bin_sweep"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    agg_df = aggregate(raw_df, ["algorithm", "pos_bins", "vel_bins"])
    raw_df.to_csv(outdir / "raw.csv", index=False)
    agg_df.to_csv(outdir / "aggregate.csv", index=False)
    print(agg_df.to_string(index=False))
    print(f"Saved bin sweep to: {outdir}")


def run_fuel_sweep(args: argparse.Namespace) -> None:
    print("\n=== FUEL-COST SWEEP (SARSA, adapted discrete scenario) ===")
    rows = []
    for fuel_cost in args.fuel_cost_values:
        for seed in args.seeds:
            print(f"fuel_cost={fuel_cost:.3f} | seed={seed}")
            rows.append(
                train_and_eval(
                    algorithm="sarsa",
                    scenario="discrete_fuel",
                    seed=seed,
                    pos_bins=args.best_pos_bins,
                    vel_bins=args.best_vel_bins,
                    fuel_cost=fuel_cost,
                    episodes=args.episodes,
                    eval_episodes=args.eval_episodes,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    epsilon_start=args.epsilon_start,
                    epsilon_decay=args.epsilon_decay,
                    epsilon_min=args.epsilon_min,
                )
            )

    outdir = ROOT / args.outdir / "fuel_sweep"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    agg_df = aggregate(raw_df, ["algorithm", "fuel_cost"])
    raw_df.to_csv(outdir / "raw.csv", index=False)
    agg_df.to_csv(outdir / "aggregate.csv", index=False)
    print(agg_df.to_string(index=False))
    print(f"Saved fuel sweep to: {outdir}")


def run_final_compare(args: argparse.Namespace) -> None:
    print("\n=== FINAL TABULAR COMPARISON (Q-learning vs SARSA) ===")
    rows = []
    for scenario in ["discrete_standard", "discrete_fuel"]:
        for algorithm in ["q_learning", "sarsa"]:
            for seed in args.seeds:
                print(f"scenario={scenario} | algorithm={algorithm} | seed={seed}")
                rows.append(
                    train_and_eval(
                        algorithm=algorithm,
                        scenario=scenario,
                        seed=seed,
                        pos_bins=args.best_pos_bins,
                        vel_bins=args.best_vel_bins,
                        fuel_cost=args.best_fuel_cost,
                        episodes=args.episodes,
                        eval_episodes=args.eval_episodes,
                        alpha=args.alpha,
                        gamma=args.gamma,
                        epsilon_start=args.epsilon_start,
                        epsilon_decay=args.epsilon_decay,
                        epsilon_min=args.epsilon_min,
                    )
                )

    outdir = ROOT / args.outdir / "final_compare"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    agg_df = aggregate(raw_df, ["scenario", "algorithm"])
    raw_df.to_csv(outdir / "raw.csv", index=False)
    agg_df.to_csv(outdir / "aggregate.csv", index=False)
    print(agg_df.to_string(index=False))
    print(f"Saved final comparison to: {outdir}")


def main() -> None:
    args = parse_args()
    tasks = [args.task] if args.task != "all" else ["bin_sweep", "fuel_sweep", "final_compare"]
    print(json.dumps(vars(args), indent=2))
    for task in tasks:
        if task == "bin_sweep":
            run_bin_sweep(args)
        elif task == "fuel_sweep":
            run_fuel_sweep(args)
        elif task == "final_compare":
            run_final_compare(args)
        else:
            raise ValueError(f"Unsupported task: {task}")


if __name__ == "__main__":
    main()
