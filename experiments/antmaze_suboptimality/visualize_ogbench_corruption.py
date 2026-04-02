#!/usr/bin/env python3
"""Visualize OGBench AntMaze branch corruption datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.antmaze_suboptimality.build_ogbench_datasets import (
    build_ogbench_transitions,
    dataset_path_from_env_name,
    parse_ogbench_env_name,
    task_goal_xy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory produced by build_ogbench_datasets.py",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default="~/.ogbench/data")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--branch_id", type=int, default=None)
    parser.add_argument("--max_full_episodes", type=int, default=80)
    parser.add_argument("--max_snippets_per_label", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_branch_rows(path: Path) -> List[Dict[str, object]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "branch_id": int(row["branch_id"]),
                    "split": row["split"],
                    "label": row["label"],
                    "label_value": int(row["label_value"]),
                    "source_index": int(row["source_index"]),
                    "raw_index": int(row["raw_index"]),
                    "episode_id": int(row["episode_id"]),
                    "step_in_episode": int(row["step_in_episode"]),
                    "progress": float(row["progress"]),
                    "future_success": row["future_success"] == "True",
                    "full_episode_success": row["full_episode_success"] == "True",
                    "start_x": float(row["start_x"]),
                    "start_y": float(row["start_y"]),
                    "goal_x": float(row["goal_x"]),
                    "goal_y": float(row["goal_y"]),
                    "direction": int(row["direction"]),
                    "direction_name": row["direction_name"],
                    "dir_dx": float(row["dir_dx"]),
                    "dir_dy": float(row["dir_dy"]),
                    "cell_row": int(row["cell_row"]),
                    "cell_col": int(row["cell_col"]),
                }
            )
    return rows


def group_episode_ranges(terminals: np.ndarray) -> List[Tuple[int, int]]:
    starts = []
    start = 0
    for idx, terminal in enumerate(terminals):
        if terminal > 0:
            starts.append((start, idx))
            start = idx + 1
    if start < len(terminals):
        starts.append((start, len(terminals) - 1))
    return starts


def plot_maze_background(ax: plt.Axes, xy: np.ndarray) -> None:
    ax.set_xlim(float(np.min(xy[:, 0])) - 1.0, float(np.max(xy[:, 0])) + 1.0)
    ax.set_ylim(float(np.min(xy[:, 1])) - 1.0, float(np.max(xy[:, 1])) + 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12, linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def sample_without_replacement(items: Sequence[int], max_items: int, rng: np.random.Generator) -> List[int]:
    if len(items) <= max_items:
        return list(items)
    return list(rng.choice(np.asarray(items, dtype=np.int32), size=max_items, replace=False))


def plot_original_episodes(
    output_path: Path,
    transitions: Mapping[str, np.ndarray],
    episode_ranges: Sequence[Tuple[int, int]],
    branch_definitions: Sequence[Mapping[str, object]],
    goal_xy: np.ndarray,
    max_full_episodes: int,
    rng: np.random.Generator,
) -> None:
    episode_ids = sample_without_replacement(list(range(len(episode_ranges))), max_full_episodes, rng)
    fig, ax = plt.subplots(figsize=(10, 7))
    xy = transitions["xy"]
    for episode_id in episode_ids:
        start, end = episode_ranges[episode_id]
        traj = xy[start : end + 1]
        ax.plot(traj[:, 0], traj[:, 1], color="0.75", linewidth=0.8, alpha=0.55)

    branch_xy = np.asarray([branch["cell_center_xy"] for branch in branch_definitions], dtype=np.float32)
    gap = np.asarray([branch["progress_gap"] for branch in branch_definitions], dtype=np.float32)
    scat = ax.scatter(
        branch_xy[:, 0],
        branch_xy[:, 1],
        c=gap,
        s=28,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.2,
        alpha=0.95,
        label="Candidate branches",
    )
    ax.scatter(goal_xy[0], goal_xy[1], marker="*", s=240, color="royalblue", edgecolors="black", label="Goal")
    plot_maze_background(ax, xy)
    ax.set_title("Original OGBench trajectories with mined branch cells")
    ax.legend(loc="lower right")
    cbar = fig.colorbar(scat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Progress gap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def select_branch(
    branch_definitions: Sequence[Mapping[str, object]],
    branch_rows: Sequence[Mapping[str, object]],
    branch_id: int | None,
) -> Mapping[str, object]:
    if branch_id is not None:
        for branch in branch_definitions:
            if int(branch["branch_id"]) == branch_id:
                return branch
        raise ValueError(f"branch_id={branch_id} not found.")

    counts = defaultdict(int)
    for row in branch_rows:
        if row["split"] == "train":
            counts[int(row["branch_id"])] += 1

    ranked = sorted(
        branch_definitions,
        key=lambda branch: (float(branch["progress_gap"]), counts[int(branch["branch_id"])]),
        reverse=True,
    )
    return ranked[0]


def branch_rows_for_id(branch_rows: Sequence[Mapping[str, object]], branch_id: int) -> List[Mapping[str, object]]:
    return [row for row in branch_rows if int(row["branch_id"]) == branch_id]


def line_for_record(
    transitions: Mapping[str, np.ndarray],
    record: Mapping[str, object],
    horizon: int,
) -> np.ndarray:
    idx = int(record["source_index"])
    return transitions["xy"][idx : idx + horizon]


def plot_branch_context(
    output_path: Path,
    transitions: Mapping[str, np.ndarray],
    episode_ranges: Sequence[Tuple[int, int]],
    branch: Mapping[str, object],
    branch_rows: Sequence[Mapping[str, object]],
    goal_xy: np.ndarray,
    horizon: int,
    max_snippets_per_label: int,
    rng: np.random.Generator,
) -> None:
    branch_id = int(branch["branch_id"])
    rows = branch_rows_for_id(branch_rows, branch_id)
    good_rows = [row for row in rows if row["label_value"] == 1]
    bad_rows = [row for row in rows if row["label_value"] == 0]
    good_rows = [good_rows[i] for i in sample_without_replacement(list(range(len(good_rows))), max_snippets_per_label, rng)]
    bad_rows = [bad_rows[i] for i in sample_without_replacement(list(range(len(bad_rows))), max_snippets_per_label, rng)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    xy = transitions["xy"]
    for ax, rows_subset, title, color in (
        (axes[0], good_rows, "Good continuations from same branch cell", "#2ca02c"),
        (axes[1], bad_rows, "Bad continuations from same branch cell", "#d62728"),
    ):
        episode_ids = sorted({int(row["episode_id"]) for row in rows_subset})
        for episode_id in episode_ids:
            start, end = episode_ranges[episode_id]
            traj = xy[start : end + 1]
            ax.plot(traj[:, 0], traj[:, 1], color="0.85", linewidth=0.7, alpha=0.5)

        for row in rows_subset:
            line = line_for_record(transitions, row, horizon)
            ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.5, alpha=0.85)
            ax.scatter(line[0, 0], line[0, 1], color=color, s=10, alpha=0.85)

        center = np.asarray(branch["cell_center_xy"], dtype=np.float32)
        ax.scatter(center[0], center[1], marker="P", s=120, color="gold", edgecolors="black", zorder=5)
        ax.scatter(goal_xy[0], goal_xy[1], marker="*", s=220, color="royalblue", edgecolors="black", zorder=5)
        plot_maze_background(ax, xy)
        ax.set_title(title)

    fig.suptitle(
        f"Branch {branch_id}: good={branch['good_direction_name']} bad={branch['bad_direction_name']} "
        f"gap={float(branch['progress_gap']):.2f}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selection_mechanism(
    output_path: Path,
    transitions: Mapping[str, np.ndarray],
    branch: Mapping[str, object],
    branch_rows: Sequence[Mapping[str, object]],
    goal_xy: np.ndarray,
    horizon: int,
    max_snippets_per_label: int,
    rng: np.random.Generator,
) -> None:
    branch_id = int(branch["branch_id"])
    rows = branch_rows_for_id(branch_rows, branch_id)
    train_good = [row for row in rows if row["split"] == "train" and row["label_value"] == 1]
    train_bad = [row for row in rows if row["split"] == "train" and row["label_value"] == 0]
    probe_good = [row for row in rows if row["split"] == "probe" and row["label_value"] == 1]
    probe_bad = [row for row in rows if row["split"] == "probe" and row["label_value"] == 0]

    train_good = [train_good[i] for i in sample_without_replacement(list(range(len(train_good))), max_snippets_per_label, rng)]
    train_bad = [train_bad[i] for i in sample_without_replacement(list(range(len(train_bad))), max_snippets_per_label, rng)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    panels = (
        (axes[0], train_good, "Selected if rho = 0", "#2ca02c"),
        (axes[1], train_bad, "Selected if rho = 1", "#d62728"),
        (axes[2], probe_good + probe_bad, "Held-out probe states", "#9467bd"),
    )
    xy = transitions["xy"]
    center = np.asarray(branch["cell_center_xy"], dtype=np.float32)
    for ax, rows_subset, title, color in panels:
        ax.hexbin(xy[:, 0], xy[:, 1], gridsize=55, mincnt=1, cmap="Greys", linewidths=0, alpha=0.35)
        for row in rows_subset:
            line = line_for_record(transitions, row, horizon)
            ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.2, alpha=0.85)
            ax.scatter(line[0, 0], line[0, 1], color=color, s=8, alpha=0.85)
        ax.scatter(center[0], center[1], marker="P", s=120, color="gold", edgecolors="black", zorder=5)
        ax.scatter(goal_xy[0], goal_xy[1], marker="*", s=220, color="royalblue", edgecolors="black", zorder=5)
        plot_maze_background(ax, xy)
        ax.set_title(title)

    fig.suptitle(f"How the dataset is constructed at branch {branch_id}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_overlay(
    output_path: Path,
    transitions: Mapping[str, np.ndarray],
    corrupt_variant: Mapping[str, np.ndarray],
    goal_xy: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    xy = transitions["xy"]
    branch_mask = corrupt_variant["branch_id"] >= 0
    corrupt_xy = corrupt_variant["xy"]

    axes[0].hexbin(xy[:, 0], xy[:, 1], gridsize=65, mincnt=1, cmap="Greys")
    axes[0].scatter(goal_xy[0], goal_xy[1], marker="*", s=220, color="royalblue", edgecolors="black")
    axes[0].set_title("Original dataset density")
    plot_maze_background(axes[0], xy)

    axes[1].hexbin(corrupt_xy[:, 0], corrupt_xy[:, 1], gridsize=65, mincnt=1, cmap="Greys", alpha=0.55)
    axes[1].scatter(
        corrupt_xy[branch_mask, 0],
        corrupt_xy[branch_mask, 1],
        c=np.where(corrupt_variant["branch_label"][branch_mask] == 1, "#2ca02c", "#d62728"),
        s=2,
        alpha=0.45,
    )
    axes[1].scatter(goal_xy[0], goal_xy[1], marker="*", s=220, color="royalblue", edgecolors="black")
    axes[1].set_title("Corrupt variant: controlled windows overlaid")
    plot_maze_background(axes[1], corrupt_xy)

    fig.suptitle("Where corruption acts in the dataset", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else dataset_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary = load_json(dataset_root / "dataset_summary.json")
    branch_definitions = load_json(dataset_root / "branch_definitions.json")
    branch_rows = load_branch_rows(dataset_root / "branch_bank.csv")

    parsed = parse_ogbench_env_name(summary["env_name"])
    raw_dataset_path = dataset_path_from_env_name(summary["env_name"], args.dataset_dir, args.dataset_path)
    raw = np.load(raw_dataset_path)
    transitions = build_ogbench_transitions({key: raw[key] for key in raw.files})
    transitions["xy"] = transitions["qpos"][:, :2].astype(np.float32)
    episode_ranges = group_episode_ranges(transitions["terminals"])
    goal_xy = task_goal_xy(parsed["maze_type"], int(summary["task_id"]))
    branch = select_branch(branch_definitions, branch_rows, args.branch_id)
    horizon = int(summary["snippet_horizon"])

    corrupt_dataset_path = Path(summary["variant_summaries"][0]["dataset_path"])
    corrupt_variant = np.load(corrupt_dataset_path)

    plot_original_episodes(
        output_dir / "01_original_episodes_with_branch_cells.png",
        transitions,
        episode_ranges,
        branch_definitions,
        goal_xy,
        args.max_full_episodes,
        rng,
    )
    plot_dataset_overlay(
        output_dir / "02_dataset_overlay_original_vs_corrupt.png",
        transitions,
        corrupt_variant,
        goal_xy,
    )
    plot_branch_context(
        output_dir / f"03_branch_{int(branch['branch_id']):03d}_good_vs_bad_context.png",
        transitions,
        episode_ranges,
        branch,
        branch_rows,
        goal_xy,
        horizon,
        args.max_snippets_per_label,
        rng,
    )
    plot_selection_mechanism(
        output_dir / f"04_branch_{int(branch['branch_id']):03d}_selection_mechanism.png",
        transitions,
        branch,
        branch_rows,
        goal_xy,
        horizon,
        args.max_snippets_per_label,
        rng,
    )

    manifest = {
        "dataset_root": str(dataset_root),
        "raw_dataset_path": str(raw_dataset_path),
        "output_dir": str(output_dir),
        "chosen_branch_id": int(branch["branch_id"]),
        "chosen_branch_cell": branch["cell"],
        "chosen_branch_gap": float(branch["progress_gap"]),
        "chosen_branch_good_direction": branch["good_direction_name"],
        "chosen_branch_bad_direction": branch["bad_direction_name"],
        "figures": sorted(str(path) for path in output_dir.glob("*.png")),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
