#!/usr/bin/env python3
"""Evaluate saved agents on held-out OGBench AntMaze branch states."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_bundle", type=str, required=True, help="Path to `probe_states.npz`.")
    parser.add_argument(
        "--run_dir",
        action="append",
        default=[],
        help="Run directory containing `flags.json` and `params_*.pkl`. Can be passed multiple times.",
    )
    parser.add_argument(
        "--run_glob",
        action="append",
        default=[],
        help="Glob pattern for run directories. Can be passed multiple times.",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Checkpoint step to restore. If omitted, the latest `params_*.pkl` is used.",
    )
    parser.add_argument("--samples_per_state", type=int, default=16)
    parser.add_argument("--rollout_horizon", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min_delta_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--skip_plot", action="store_true")
    return parser.parse_args()


def expand_run_dirs(run_dirs: Sequence[str], run_globs: Sequence[str]) -> List[Path]:
    expanded = {Path(run_dir).resolve() for run_dir in run_dirs}
    for pattern in run_globs:
        expanded.update(Path(path).resolve() for path in glob.glob(pattern))
    expanded = {path for path in expanded if (path / "flags.json").exists()}
    if not expanded:
        raise ValueError("No valid run directories were provided.")
    return sorted(expanded)


def load_probe_bundle(path: str) -> Dict[str, np.ndarray]:
    bundle = np.load(path, allow_pickle=False)
    return {key: bundle[key] for key in bundle.files}


def list_checkpoints(run_dir: Path) -> List[int]:
    pattern = re.compile(r"params_(\d+)\.pkl$")
    steps = []
    for path in run_dir.glob("params_*.pkl"):
        match = pattern.search(path.name)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise ValueError(f"No checkpoints found under {run_dir}.")
    return sorted(steps)


def resolve_checkpoint_step(run_dir: Path, checkpoint_step: Optional[int]) -> int:
    steps = list_checkpoints(run_dir)
    if checkpoint_step is None:
        return steps[-1]
    if checkpoint_step not in steps:
        raise ValueError(f"Checkpoint step {checkpoint_step} is not available in {run_dir}. Found: {steps}.")
    return checkpoint_step


def load_run_flags(run_dir: Path) -> Mapping[str, object]:
    with open(run_dir / "flags.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def create_and_restore_agent(
    run_dir: Path,
    checkpoint_step: int,
    probe_bundle: Mapping[str, np.ndarray],
):
    from agents import agents as agent_registry
    from utils.flax_utils import restore_agent

    flags = load_run_flags(run_dir)
    config = dict(flags["agent"])
    agent_name = config["agent_name"]
    agent_class = agent_registry[agent_name]

    example_obs = probe_bundle["observation"][:1]
    example_actions = probe_bundle["action"][:1]
    seed = int(flags.get("seed", 0))
    agent = agent_class.create(seed, example_obs, example_actions, config)
    agent = restore_agent(agent, str(run_dir), checkpoint_step)
    return agent, flags


def load_ogbench_env(env_name: str):
    import ogbench

    return ogbench.make_env_and_datasets(env_name, env_only=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def classify_delta(delta: np.ndarray, good_dir: np.ndarray, bad_dir: np.ndarray, min_delta_norm: float) -> str:
    if float(np.linalg.norm(delta)) < min_delta_norm:
        return "neutral"
    good_score = cosine_similarity(delta, good_dir)
    bad_score = cosine_similarity(delta, bad_dir)
    if good_score > bad_score:
        return "good"
    if bad_score > good_score:
        return "bad"
    return "neutral"


def restore_probe_state(env, qpos: np.ndarray, qvel: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    base_env = env.unwrapped
    base_env.set_goal(goal_xy=goal_xy.astype(np.float64))
    base_env.set_state(qpos.astype(np.float64), qvel.astype(np.float64))
    return np.asarray(base_env.get_ob(), dtype=np.float32)


def sample_action(agent, observation: np.ndarray, rng, temperature: float) -> np.ndarray:
    action = agent.sample_actions(observations=observation[None], temperature=temperature, seed=rng)
    return np.asarray(action[0], dtype=np.float32)


def evaluate_run(
    run_dir: Path,
    checkpoint_step: int,
    probe_bundle: Mapping[str, np.ndarray],
    samples_per_state: int,
    rollout_horizon: int,
    temperature: float,
    seed: int,
    min_delta_norm: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    import jax

    agent, flags = create_and_restore_agent(run_dir, checkpoint_step, probe_bundle)
    env_name = str(flags["env_name"])
    env = load_ogbench_env(env_name)
    env.reset()

    rng = jax.random.PRNGKey(seed)
    rows: List[Dict[str, object]] = []

    num_probe_states = len(probe_bundle["branch_id"])
    for probe_idx in range(num_probe_states):
        branch_id = int(probe_bundle["branch_id"][probe_idx])
        label = int(probe_bundle["label"][probe_idx])
        qpos = probe_bundle["qpos"][probe_idx]
        qvel = probe_bundle["qvel"][probe_idx]
        goal_xy = probe_bundle["goal_xy"][probe_idx]
        good_dir = probe_bundle["good_direction_vector"][probe_idx]
        bad_dir = probe_bundle["bad_direction_vector"][probe_idx]

        for sample_idx in range(samples_per_state):
            rng, rollout_seed = jax.random.split(rng)
            observation = restore_probe_state(env, qpos, qvel, goal_xy)
            start_xy = np.asarray(env.unwrapped.get_xy(), dtype=np.float32)

            terminated = False
            truncated = False
            for _ in range(rollout_horizon):
                rollout_seed, action_seed = jax.random.split(rollout_seed)
                action = sample_action(agent, observation, action_seed, temperature)
                observation, _, terminated, truncated, _ = env.step(np.clip(action, -1.0, 1.0))
                observation = np.asarray(observation, dtype=np.float32)
                if terminated or truncated:
                    break

            end_xy = np.asarray(env.unwrapped.get_xy(), dtype=np.float32)
            delta = end_xy - start_xy
            preference = classify_delta(delta, good_dir, bad_dir, min_delta_norm)
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "checkpoint_step": int(checkpoint_step),
                    "env_name": env_name,
                    "probe_index": probe_idx,
                    "branch_id": branch_id,
                    "source_label": label,
                    "sample_index": sample_idx,
                    "start_x": float(start_xy[0]),
                    "start_y": float(start_xy[1]),
                    "end_x": float(end_xy[0]),
                    "end_y": float(end_xy[1]),
                    "delta_x": float(delta[0]),
                    "delta_y": float(delta[1]),
                    "delta_norm": float(np.linalg.norm(delta)),
                    "goal_progress": float(np.linalg.norm(goal_xy - start_xy) - np.linalg.norm(goal_xy - end_xy)),
                    "classification": preference,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

    good_fraction = np.mean([row["classification"] == "good" for row in rows]) if rows else 0.0
    bad_fraction = np.mean([row["classification"] == "bad" for row in rows]) if rows else 0.0
    neutral_fraction = np.mean([row["classification"] == "neutral" for row in rows]) if rows else 0.0
    goal_progress = np.mean([row["goal_progress"] for row in rows]) if rows else 0.0

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_step": int(checkpoint_step),
        "env_name": env_name,
        "num_probe_states": int(num_probe_states),
        "samples_per_state": int(samples_per_state),
        "rollout_horizon": int(rollout_horizon),
        "good_fraction": float(good_fraction),
        "bad_fraction": float(bad_fraction),
        "neutral_fraction": float(neutral_fraction),
        "mean_goal_progress": float(goal_progress),
    }
    env.close()
    return rows, summary


def write_csv(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No probe rows to write.")
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(summary: Mapping[str, object], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def maybe_plot(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["run_dir"])].append(row)

    labels = []
    good = []
    bad = []
    neutral = []
    for run_dir, run_rows in grouped.items():
        labels.append(Path(run_dir).name)
        good.append(np.mean([row["classification"] == "good" for row in run_rows]))
        bad.append(np.mean([row["classification"] == "bad" for row in run_rows]))
        neutral.append(np.mean([row["classification"] == "neutral" for row in run_rows]))

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    ax.bar(x - width, good, width=width, label="good", color="#1b9e77")
    ax.bar(x, bad, width=width, label="bad", color="#d95f02")
    ax.bar(x + width, neutral, width=width, label="neutral", color="#7570b3")
    ax.set_ylabel("Fraction")
    ax.set_title("OGBench branch probe preference by run")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dirs = expand_run_dirs(args.run_dir, args.run_glob)
    probe_bundle = load_probe_bundle(args.probe_bundle)

    all_rows: List[Dict[str, object]] = []
    summaries = []
    for run_dir in run_dirs:
        checkpoint_step = resolve_checkpoint_step(run_dir, args.checkpoint_step)
        rows, summary = evaluate_run(
            run_dir=run_dir,
            checkpoint_step=checkpoint_step,
            probe_bundle=probe_bundle,
            samples_per_state=args.samples_per_state,
            rollout_horizon=args.rollout_horizon,
            temperature=args.temperature,
            seed=args.seed,
            min_delta_norm=args.min_delta_norm,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, output_csv)

    overall = {
        "num_runs": len(summaries),
        "runs": summaries,
        "mean_good_fraction": float(np.mean([summary["good_fraction"] for summary in summaries])),
        "mean_bad_fraction": float(np.mean([summary["bad_fraction"] for summary in summaries])),
        "mean_neutral_fraction": float(np.mean([summary["neutral_fraction"] for summary in summaries])),
        "mean_goal_progress": float(np.mean([summary["mean_goal_progress"] for summary in summaries])),
    }

    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(overall, output_json)

    if not args.skip_plot:
        maybe_plot(all_rows, output_csv.with_suffix(".png"))

    print(f"Wrote OGBench probe rows to {output_csv}")


if __name__ == "__main__":
    main()
