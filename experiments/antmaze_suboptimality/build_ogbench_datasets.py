#!/usr/bin/env python3
"""Build controlled OGBench AntMaze suboptimality datasets.

This script targets OGBench singletask AntMaze datasets such as
`antmaze-large-navigate-singletask-task1-v0`. It reconstructs the transition
format used by the repo, mines branch states with good and bad in-support
continuations, and exports dataset variants with different bad-branch ratios.

The output layout matches the D4RL helper:

    <output_dir>/
      config.json
      dataset_summary.json
      branch_bank.csv
      branch_definitions.json
      probe_states.npz
      variants/
        rho_0.00/
          dataset.npz
          metadata.json
        ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


MAZE_CELL_SIZE = 4.0
MAZE_OFFSET_X = 4.0
MAZE_OFFSET_Y = 4.0
OGBENCH_GOAL_TOL = 0.5

OGBENCH_MAZE_LAYOUTS = {
    "medium": (
        (1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 0, 0, 1, 0, 0, 0, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (1, 0, 0, 1, 0, 0, 0, 1),
        (1, 0, 1, 0, 0, 1, 0, 1),
        (1, 0, 0, 0, 1, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1),
    ),
    "large": (
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1),
        (1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1),
        (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1),
        (1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1),
        (1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1),
        (1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1),
        (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ),
    "giant": (
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1),
        (1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1),
        (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1),
        (1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1),
        (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1),
        (1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1),
        (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1),
        (1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1),
        (1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1),
        (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ),
    "teleport": (
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1),
        (1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1),
        (1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1),
        (1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1),
        (1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1),
        (1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1),
        (1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1),
        (1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1),
        (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ),
}

OGBENCH_TASKS = {
    "medium": (
        ((1, 1), (6, 6)),
        ((6, 1), (1, 6)),
        ((5, 3), (4, 2)),
        ((6, 5), (6, 1)),
        ((2, 6), (1, 1)),
    ),
    "large": (
        ((1, 1), (7, 10)),
        ((5, 4), (7, 1)),
        ((7, 4), (1, 10)),
        ((3, 8), (5, 4)),
        ((1, 1), (5, 4)),
    ),
    "giant": (
        ((1, 1), (10, 14)),
        ((1, 14), (10, 1)),
        ((8, 14), (1, 1)),
        ((8, 3), (5, 12)),
        ((5, 9), (3, 8)),
    ),
    "teleport": (
        ((1, 10), (7, 1)),
        ((1, 1), (7, 10)),
        ((5, 6), (7, 10)),
        ((7, 1), (7, 10)),
        ((5, 6), (7, 1)),
    ),
}

DIRECTION_NAMES = {
    0: "east",
    1: "north",
    2: "west",
    3: "south",
    4: "stay",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", type=str, default="antmaze-large-navigate-singletask-task1-v0")
    parser.add_argument("--dataset_dir", type=str, default="~/.ogbench/data")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional explicit path to the OGBench .npz dataset. Overrides --dataset_dir.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rho_values", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--spatial_bin_size", type=float, default=2.0)
    parser.add_argument("--direction_lookahead", type=int, default=6)
    parser.add_argument("--snippet_horizon", type=int, default=20)
    parser.add_argument("--distance_metric", choices=("euclidean", "geodesic"), default="geodesic")
    parser.add_argument("--min_direction_norm", type=float, default=1.0)
    parser.add_argument("--min_cell_visits", type=int, default=30)
    parser.add_argument("--min_dir_visits", type=int, default=8)
    parser.add_argument("--min_progress_gap", type=float, default=1.0)
    parser.add_argument("--min_branch_distance_to_goal", type=float, default=2.0)
    parser.add_argument("--num_probe_per_label", type=int, default=2)
    parser.add_argument(
        "--max_train_snippets_per_label",
        type=int,
        default=32,
        help=(
            "Base per-branch/per-label train-snippet cap after probe holdout. "
            "If `--target_branch_transitions` requires a larger cap, the builder "
            "raises this automatically."
        ),
    )
    parser.add_argument(
        "--target_branch_transitions",
        type=int,
        default=200000,
        help=(
            "Approximate target number of branch-controlled transitions in each "
            "rho variant. The builder expands the per-branch snippet budget to "
            "hit this target when enough snippets are available."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uncompressed", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args()


def parse_rho_values(rho_values: str) -> List[float]:
    rhos = [float(token.strip()) for token in rho_values.split(",") if token.strip()]
    if not rhos:
        raise ValueError("No rho values were provided.")
    for rho in rhos:
        if rho < 0.0 or rho > 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}.")
    return rhos


def resolve_train_snippet_cap(num_candidate_branches: int, args: argparse.Namespace) -> int:
    """Pick a per-branch/per-label cap that can support the requested branch mass."""

    base_cap = max(int(args.max_train_snippets_per_label), 1)
    target_branch_transitions = int(getattr(args, "target_branch_transitions", 0))
    if target_branch_transitions <= 0 or num_candidate_branches <= 0:
        return base_cap

    required_per_branch = math.ceil(
        target_branch_transitions / (num_candidate_branches * max(int(args.snippet_horizon), 1))
    )
    return max(base_cap, required_per_branch)


def parse_ogbench_env_name(env_name: str) -> Dict[str, object]:
    splits = env_name.split("-")
    if "singletask" not in splits:
        raise ValueError(f"{env_name} is not an OGBench singletask environment.")
    if not env_name.startswith("antmaze-"):
        raise ValueError(f"{env_name} is not an OGBench AntMaze environment.")

    pos = splits.index("singletask")
    task_id = 1
    if pos + 1 < len(splits) - 1:
        task_token = splits[pos + 1]
        match = re.fullmatch(r"task(\d+)", task_token)
        if match is not None:
            task_id = int(match.group(1))

    maze_type = splits[1]
    dataset_name = "-".join(splits[:pos] + splits[-1:])
    gym_env_name = "-".join(splits[: pos - 1] + splits[pos:])
    return {
        "task_id": task_id,
        "maze_type": maze_type,
        "dataset_name": dataset_name,
        "gym_env_name": gym_env_name,
    }


def infer_maze_layout(maze_type: str) -> Tuple[Tuple[int, ...], ...]:
    if maze_type not in OGBENCH_MAZE_LAYOUTS:
        raise ValueError(f"Unsupported OGBench AntMaze type: {maze_type}")
    return OGBENCH_MAZE_LAYOUTS[maze_type]


def ij_to_xy(ij: Tuple[int, int]) -> np.ndarray:
    i, j = ij
    x = j * MAZE_CELL_SIZE - MAZE_OFFSET_X
    y = i * MAZE_CELL_SIZE - MAZE_OFFSET_Y
    return np.asarray([x, y], dtype=np.float32)


def xy_to_ij(xy: np.ndarray) -> Tuple[int, int]:
    i = int((float(xy[1]) + MAZE_OFFSET_Y + 0.5 * MAZE_CELL_SIZE) / MAZE_CELL_SIZE)
    j = int((float(xy[0]) + MAZE_OFFSET_X + 0.5 * MAZE_CELL_SIZE) / MAZE_CELL_SIZE)
    return (i, j)


def task_goal_xy(maze_type: str, task_id: int) -> np.ndarray:
    tasks = OGBENCH_TASKS.get(maze_type)
    if tasks is None:
        raise ValueError(f"No task definitions available for maze type {maze_type}.")
    if task_id < 1 or task_id > len(tasks):
        raise ValueError(f"task_id must be in [1, {len(tasks)}], got {task_id}.")
    _, goal_ij = tasks[task_id - 1]
    return ij_to_xy(goal_ij)


def dataset_path_from_env_name(env_name: str, dataset_dir: str, dataset_path: Optional[str]) -> Path:
    if dataset_path is not None:
        return Path(dataset_path).expanduser().resolve()
    parsed = parse_ogbench_env_name(env_name)
    root = Path(dataset_dir).expanduser().resolve()
    return root / f"{parsed['dataset_name']}.npz"


def load_npz_dataset(dataset_path: Path) -> Dict[str, np.ndarray]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    bundle = np.load(dataset_path)
    return {key: bundle[key] for key in bundle.files}


def build_ogbench_transitions(raw: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    required = ("observations", "actions", "terminals", "qpos", "qvel")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"OGBench dataset is missing keys: {missing}")

    observations = np.asarray(raw["observations"], dtype=np.float32)
    actions = np.asarray(raw["actions"], dtype=np.float32)
    terminals = np.asarray(raw["terminals"]).astype(np.float32)
    qpos = np.asarray(raw["qpos"], dtype=np.float32)
    qvel = np.asarray(raw["qvel"], dtype=np.float32)

    ob_mask = (1.0 - terminals).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    new_terminals = np.concatenate([terminals[1:], [1.0]]).astype(np.float32)

    raw_index = np.flatnonzero(ob_mask).astype(np.int32)
    next_raw_index = np.flatnonzero(next_ob_mask).astype(np.int32)

    if len(raw_index) != len(next_raw_index):
        raise ValueError("OGBench transition reconstruction produced misaligned raw indices.")

    return {
        "observations": observations[ob_mask].astype(np.float32),
        "actions": actions[ob_mask].astype(np.float32),
        "next_observations": observations[next_ob_mask].astype(np.float32),
        "terminals": new_terminals[ob_mask].astype(np.float32),
        "qpos": qpos[ob_mask].astype(np.float32),
        "qvel": qvel[ob_mask].astype(np.float32),
        "next_qpos": qpos[next_ob_mask].astype(np.float32),
        "next_qvel": qvel[next_ob_mask].astype(np.float32),
        "raw_index": raw_index,
        "next_raw_index": next_raw_index,
    }


def apply_ogbench_singletask_rewards(
    transitions: MutableMapping[str, np.ndarray],
    goal_xy: np.ndarray,
    goal_tol: float,
) -> None:
    xy = transitions["qpos"][:, :2]
    successes = (np.linalg.norm(xy - goal_xy[None], axis=-1) <= goal_tol).astype(np.float32)
    transitions["rewards"] = successes - 1.0
    transitions["masks"] = 1.0 - successes
    transitions["goal_xy"] = np.repeat(goal_xy[None], len(xy), axis=0).astype(np.float32)
    transitions["xy"] = xy.astype(np.float32)
    transitions["next_xy"] = transitions["next_qpos"][:, :2].astype(np.float32)


def reconstruct_episode_metadata(transitions: MutableMapping[str, np.ndarray]) -> None:
    episode_ids = np.zeros_like(transitions["terminals"], dtype=np.int32)
    step_in_episode = np.zeros_like(transitions["terminals"], dtype=np.int32)
    episode_starts = [0]
    episode_ends = []

    episode_id = 0
    step = 0
    for idx, terminal in enumerate(transitions["terminals"]):
        episode_ids[idx] = episode_id
        step_in_episode[idx] = step
        if terminal > 0:
            episode_ends.append(idx)
            if idx + 1 < len(transitions["terminals"]):
                episode_id += 1
                episode_starts.append(idx + 1)
                step = 0
        else:
            step += 1

    transitions["episode_id"] = episode_ids
    transitions["step_in_episode"] = step_in_episode
    transitions["episode_starts"] = np.asarray(episode_starts, dtype=np.int32)
    transitions["episode_ends"] = np.asarray(episode_ends, dtype=np.int32)


def xy_to_grid_bin(xy: np.ndarray, bin_size: float) -> Tuple[int, int]:
    return (int(math.floor(float(xy[0]) / bin_size)), int(math.floor(float(xy[1]) / bin_size)))


def direction_bin(delta: np.ndarray, min_norm: float) -> int:
    norm = float(np.linalg.norm(delta))
    if norm < min_norm:
        return 4
    if abs(delta[0]) >= abs(delta[1]):
        return 0 if delta[0] >= 0 else 2
    return 1 if delta[1] >= 0 else 3


def is_free_cell(maze_layout: Tuple[Tuple[int, ...], ...], row: int, col: int) -> bool:
    if row < 0 or col < 0 or row >= len(maze_layout) or col >= len(maze_layout[0]):
        return False
    return maze_layout[row][col] != 1


@lru_cache(maxsize=None)
def shortest_path_steps(
    maze_layout: Tuple[Tuple[int, ...], ...],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[int]:
    if start == goal:
        return 0
    if not is_free_cell(maze_layout, *start) or not is_free_cell(maze_layout, *goal):
        return None

    frontier = [start]
    visited = {start}
    steps = 0
    while frontier:
        steps += 1
        next_frontier = []
        for row, col in frontier:
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                candidate = (row + d_row, col + d_col)
                if candidate in visited or not is_free_cell(maze_layout, *candidate):
                    continue
                if candidate == goal:
                    return steps
                visited.add(candidate)
                next_frontier.append(candidate)
        frontier = next_frontier
    return None


def point_distance(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    metric: str,
    maze_layout: Tuple[Tuple[int, ...], ...],
) -> float:
    if metric == "euclidean":
        return float(np.linalg.norm(goal_xy - start_xy))

    start_cell = xy_to_ij(start_xy)
    goal_cell = xy_to_ij(goal_xy)
    steps = shortest_path_steps(maze_layout, start_cell, goal_cell)
    if steps is None:
        return float(np.linalg.norm(goal_xy - start_xy))
    return float(steps * MAZE_CELL_SIZE)


def goal_steps_map(
    maze_layout: Tuple[Tuple[int, ...], ...],
    goal: Tuple[int, int],
) -> np.ndarray:
    steps = np.full((len(maze_layout), len(maze_layout[0])), -1, dtype=np.int32)
    if not is_free_cell(maze_layout, *goal):
        return steps

    frontier = [goal]
    steps[goal[0], goal[1]] = 0
    while frontier:
        row, col = frontier.pop(0)
        cur_steps = steps[row, col]
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (row + d_row, col + d_col)
            if not is_free_cell(maze_layout, *nxt):
                continue
            if steps[nxt[0], nxt[1]] != -1:
                continue
            steps[nxt[0], nxt[1]] = cur_steps + 1
            frontier.append(nxt)
    return steps


def geodesic_distance_array(
    xy: np.ndarray,
    goal_xy: np.ndarray,
    maze_layout: Tuple[Tuple[int, ...], ...],
) -> np.ndarray:
    steps = goal_steps_map(maze_layout, xy_to_ij(goal_xy))
    euclidean = np.linalg.norm(xy - goal_xy[None], axis=-1).astype(np.float32)

    cells_i = ((xy[:, 1] + MAZE_OFFSET_Y + 0.5 * MAZE_CELL_SIZE) / MAZE_CELL_SIZE).astype(np.int32)
    cells_j = ((xy[:, 0] + MAZE_OFFSET_X + 0.5 * MAZE_CELL_SIZE) / MAZE_CELL_SIZE).astype(np.int32)
    in_bounds = (
        (cells_i >= 0)
        & (cells_i < steps.shape[0])
        & (cells_j >= 0)
        & (cells_j < steps.shape[1])
    )

    distances = euclidean.copy()
    valid_i = cells_i[in_bounds]
    valid_j = cells_j[in_bounds]
    valid_steps = steps[valid_i, valid_j]
    valid_mask = valid_steps >= 0
    if np.any(valid_mask):
        in_bound_indices = np.flatnonzero(in_bounds)
        replace_indices = in_bound_indices[valid_mask]
        distances[replace_indices] = valid_steps[valid_mask].astype(np.float32) * MAZE_CELL_SIZE
    return distances


def build_transition_features(
    transitions: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    maze_layout: Tuple[Tuple[int, ...], ...],
) -> Dict[str, np.ndarray]:
    num_transitions = transitions["observations"].shape[0]
    episode_ends = transitions["episode_ends"]
    episode_id = transitions["episode_id"]

    progress = np.full(num_transitions, np.nan, dtype=np.float32)
    direction = np.full(num_transitions, -1, dtype=np.int32)
    direction_delta = np.zeros((num_transitions, 2), dtype=np.float32)
    valid = np.zeros(num_transitions, dtype=bool)
    goal_distance = np.zeros(num_transitions, dtype=np.float32)
    future_success = np.zeros(num_transitions, dtype=bool)
    full_episode_success = np.zeros(num_transitions, dtype=bool)
    spatial_bins = np.zeros((num_transitions, 2), dtype=np.int32)

    goal_xy = transitions["goal_xy"][0]
    if args.distance_metric == "geodesic":
        goal_distance = geodesic_distance_array(transitions["xy"], goal_xy, maze_layout)
    else:
        goal_distance = np.linalg.norm(transitions["xy"] - goal_xy[None], axis=-1).astype(np.float32)

    success_now = goal_distance <= OGBENCH_GOAL_TOL
    success_prefix = np.concatenate([[0], np.cumsum(success_now.astype(np.int32))])
    episode_end_by_id = {int(episode_id[end_idx]): int(end_idx) for end_idx in episode_ends}
    episode_success = {}
    for ep_id, end_idx in episode_end_by_id.items():
        start_idx = int(np.searchsorted(episode_id, ep_id, side="left"))
        episode_success[ep_id] = bool(success_now[start_idx : end_idx + 1].any())

    for idx in range(num_transitions):
        ep_id = int(episode_id[idx])
        episode_end = episode_end_by_id[ep_id]
        full_episode_success[idx] = episode_success[ep_id]
        spatial_bins[idx] = xy_to_grid_bin(transitions["xy"][idx], args.spatial_bin_size)

        future_idx = idx + args.direction_lookahead
        snippet_end_idx = idx + args.snippet_horizon
        if future_idx > episode_end or snippet_end_idx > episode_end:
            continue
        if goal_distance[idx] < args.min_branch_distance_to_goal:
            continue

        future_xy = transitions["xy"][future_idx]
        delta = future_xy - transitions["xy"][idx]
        direction[idx] = direction_bin(delta, args.min_direction_norm)
        direction_delta[idx] = delta.astype(np.float32)
        progress[idx] = goal_distance[idx] - goal_distance[snippet_end_idx]
        future_success[idx] = bool(success_prefix[snippet_end_idx + 1] - success_prefix[idx] > 0)
        valid[idx] = direction[idx] != 4

    return {
        "progress": progress,
        "direction": direction,
        "direction_delta": direction_delta,
        "valid": valid,
        "goal_distance": goal_distance,
        "future_success": future_success,
        "full_episode_success": full_episode_success,
        "spatial_bins": spatial_bins,
    }


def candidate_branches(
    transitions: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    cell_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, is_valid in enumerate(features["valid"]):
        if is_valid:
            cell_to_indices[tuple(int(v) for v in features["spatial_bins"][idx])].append(idx)

    branches: List[Dict[str, object]] = []
    branch_id = 0
    for cell, indices in sorted(cell_to_indices.items()):
        if len(indices) < args.min_cell_visits:
            continue

        direction_groups: Dict[int, List[int]] = defaultdict(list)
        for idx in indices:
            direction_groups[int(features["direction"][idx])].append(idx)

        direction_groups = {
            direction: group for direction, group in direction_groups.items() if len(group) >= args.min_dir_visits
        }
        if len(direction_groups) < 2:
            continue

        direction_stats = []
        for direction, group in direction_groups.items():
            group_progress = features["progress"][group]
            direction_stats.append(
                {
                    "direction": direction,
                    "indices": group,
                    "count": len(group),
                    "mean_progress": float(np.mean(group_progress)),
                    "success_rate": float(np.mean(features["full_episode_success"][group])),
                    "future_success_rate": float(np.mean(features["future_success"][group])),
                    "mean_delta": np.mean(features["direction_delta"][group], axis=0),
                }
            )

        direction_stats.sort(key=lambda item: item["mean_progress"])
        bad = direction_stats[0]
        good = direction_stats[-1]
        progress_gap = good["mean_progress"] - bad["mean_progress"]
        if progress_gap < args.min_progress_gap:
            continue

        branches.append(
            {
                "branch_id": branch_id,
                "cell": [int(cell[0]), int(cell[1])],
                "cell_center_xy": np.mean(transitions["xy"][indices], axis=0).astype(float).tolist(),
                "num_visits": len(indices),
                "progress_gap": float(progress_gap),
                "good_direction": int(good["direction"]),
                "bad_direction": int(bad["direction"]),
                "good_direction_name": DIRECTION_NAMES[int(good["direction"])],
                "bad_direction_name": DIRECTION_NAMES[int(bad["direction"])],
                "good_mean_progress": float(good["mean_progress"]),
                "bad_mean_progress": float(bad["mean_progress"]),
                "good_success_rate": float(good["success_rate"]),
                "bad_success_rate": float(bad["success_rate"]),
                "good_future_success_rate": float(good["future_success_rate"]),
                "bad_future_success_rate": float(bad["future_success_rate"]),
                "good_indices": list(map(int, good["indices"])),
                "bad_indices": list(map(int, bad["indices"])),
                "good_direction_vector": np.asarray(good["mean_delta"], dtype=np.float32).tolist(),
                "bad_direction_vector": np.asarray(bad["mean_delta"], dtype=np.float32).tolist(),
            }
        )
        branch_id += 1

    return branches


def clone_occupied(occupied: Mapping[int, Sequence[Tuple[int, int]]]) -> Dict[int, List[Tuple[int, int]]]:
    return {int(ep_id): list(intervals) for ep_id, intervals in occupied.items()}


def interval_for_index(idx: int, step_in_episode: np.ndarray, horizon: int) -> Tuple[int, int]:
    start = int(step_in_episode[idx])
    end = int(step_in_episode[idx] + horizon - 1)
    return (start, end)


def has_overlap(occupied: Mapping[int, Sequence[Tuple[int, int]]], ep_id: int, interval: Tuple[int, int]) -> bool:
    for taken_start, taken_end in occupied.get(ep_id, []):
        if not (interval[1] < taken_start or interval[0] > taken_end):
            return True
    return False


def add_interval(occupied: MutableMapping[int, List[Tuple[int, int]]], ep_id: int, interval: Tuple[int, int]) -> None:
    if ep_id not in occupied:
        occupied[ep_id] = []
    occupied[ep_id].append(interval)


def pop_interval(occupied: MutableMapping[int, List[Tuple[int, int]]], ep_id: int) -> None:
    occupied[ep_id].pop()
    if not occupied[ep_id]:
        del occupied[ep_id]


def next_non_overlapping_index(
    ranked_indices: Sequence[int],
    start_ptr: int,
    occupied: Mapping[int, Sequence[Tuple[int, int]]],
    episode_id: np.ndarray,
    step_in_episode: np.ndarray,
    horizon: int,
) -> Tuple[Optional[int], int, Optional[int], Optional[Tuple[int, int]]]:
    ptr = start_ptr
    while ptr < len(ranked_indices):
        idx = int(ranked_indices[ptr])
        ptr += 1
        ep_id = int(episode_id[idx])
        interval = interval_for_index(idx, step_in_episode, horizon)
        if has_overlap(occupied, ep_id, interval):
            continue
        return idx, ptr, ep_id, interval
    return None, ptr, None, None


def select_balanced_disjoint_candidates(
    good_indices: Sequence[int],
    bad_indices: Sequence[int],
    progress: np.ndarray,
    episode_id: np.ndarray,
    step_in_episode: np.ndarray,
    horizon: int,
    pair_limit: int,
    occupied: Mapping[int, Sequence[Tuple[int, int]]],
) -> Tuple[List[int], List[int], Dict[int, List[Tuple[int, int]]]]:
    ranked_good = sorted(good_indices, key=lambda idx: float(progress[idx]), reverse=True)
    ranked_bad = sorted(bad_indices, key=lambda idx: float(progress[idx]))
    local_occupied = clone_occupied(occupied)

    selected_good: List[int] = []
    selected_bad: List[int] = []
    good_ptr = 0
    bad_ptr = 0

    for _ in range(pair_limit):
        good_idx, good_ptr, good_ep, good_interval = next_non_overlapping_index(
            ranked_good,
            good_ptr,
            local_occupied,
            episode_id,
            step_in_episode,
            horizon,
        )
        if good_idx is None or good_ep is None or good_interval is None:
            break
        add_interval(local_occupied, good_ep, good_interval)

        bad_idx, bad_ptr, bad_ep, bad_interval = next_non_overlapping_index(
            ranked_bad,
            bad_ptr,
            local_occupied,
            episode_id,
            step_in_episode,
            horizon,
        )
        if bad_idx is None or bad_ep is None or bad_interval is None:
            pop_interval(local_occupied, good_ep)
            break

        add_interval(local_occupied, bad_ep, bad_interval)
        selected_good.append(good_idx)
        selected_bad.append(bad_idx)

    return selected_good, selected_bad, local_occupied


def build_branch_bank(
    transitions: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    branches: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
    train_snippet_cap: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    branch_definitions: List[Dict[str, object]] = []
    branch_rows: List[Dict[str, object]] = []
    train_records: List[Dict[str, object]] = []
    probe_records: List[Dict[str, object]] = []
    global_occupied: Dict[int, List[Tuple[int, int]]] = {}

    pair_limit = args.num_probe_per_label + max(train_snippet_cap, 1)
    sorted_branches = sorted(branches, key=lambda branch: float(branch["progress_gap"]), reverse=True)

    for branch in sorted_branches:
        good_selected, bad_selected, candidate_occupied = select_balanced_disjoint_candidates(
            good_indices=list(branch["good_indices"]),
            bad_indices=list(branch["bad_indices"]),
            progress=features["progress"],
            episode_id=transitions["episode_id"],
            step_in_episode=transitions["step_in_episode"],
            horizon=args.snippet_horizon,
            pair_limit=pair_limit,
            occupied=global_occupied,
        )

        if len(good_selected) <= args.num_probe_per_label or len(bad_selected) <= args.num_probe_per_label:
            continue

        probe_good = good_selected[: args.num_probe_per_label]
        probe_bad = bad_selected[: args.num_probe_per_label]
        train_good = good_selected[args.num_probe_per_label :]
        train_bad = bad_selected[args.num_probe_per_label :]

        if not train_good or not train_bad:
            continue

        global_occupied = candidate_occupied
        branch_def = dict(branch)
        branch_def["train_good_count"] = int(len(train_good))
        branch_def["train_bad_count"] = int(len(train_bad))
        branch_def["probe_good_count"] = int(len(probe_good))
        branch_def["probe_bad_count"] = int(len(probe_bad))
        branch_definitions.append(branch_def)

        for split_name, label_name, label_value, indices in (
            ("probe", "good", 1, probe_good),
            ("probe", "bad", 0, probe_bad),
            ("train", "good", 1, train_good),
            ("train", "bad", 0, train_bad),
        ):
            for idx in indices:
                row = {
                    "branch_id": int(branch["branch_id"]),
                    "split": split_name,
                    "label": label_name,
                    "label_value": label_value,
                    "source_index": int(idx),
                    "raw_index": int(transitions["raw_index"][idx]),
                    "episode_id": int(transitions["episode_id"][idx]),
                    "step_in_episode": int(transitions["step_in_episode"][idx]),
                    "progress": float(features["progress"][idx]),
                    "future_success": bool(features["future_success"][idx]),
                    "full_episode_success": bool(features["full_episode_success"][idx]),
                    "start_x": float(transitions["xy"][idx, 0]),
                    "start_y": float(transitions["xy"][idx, 1]),
                    "goal_x": float(transitions["goal_xy"][idx, 0]),
                    "goal_y": float(transitions["goal_xy"][idx, 1]),
                    "direction": int(features["direction"][idx]),
                    "direction_name": DIRECTION_NAMES[int(features["direction"][idx])],
                    "dir_dx": float(features["direction_delta"][idx, 0]),
                    "dir_dy": float(features["direction_delta"][idx, 1]),
                    "cell_row": int(branch["cell"][0]),
                    "cell_col": int(branch["cell"][1]),
                }
                branch_rows.append(row)
                target = probe_records if split_name == "probe" else train_records
                target.append(row)

    return branch_definitions, branch_rows, train_records, probe_records


def transition_indices_for_record(record: Mapping[str, object], horizon: int) -> np.ndarray:
    start_idx = int(record["source_index"])
    return np.arange(start_idx, start_idx + horizon, dtype=np.int32)


def build_variant_datasets(
    transitions: Mapping[str, np.ndarray],
    branch_definitions: Sequence[Mapping[str, object]],
    train_records: Sequence[Mapping[str, object]],
    probe_records: Sequence[Mapping[str, object]],
    rhos: Sequence[float],
    output_dir: Path,
    seed: int,
    compress: bool,
    horizon: int,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    train_by_branch: Dict[int, Dict[int, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in train_records:
        train_by_branch[int(record["branch_id"])][int(record["label_value"])].append(dict(record))

    for branch_id in list(train_by_branch):
        for label in (0, 1):
            rng.shuffle(train_by_branch[branch_id][label])

    reserved_transition_indices = set()
    for record in list(train_records) + list(probe_records):
        reserved_transition_indices.update(map(int, transition_indices_for_record(record, horizon)))

    background_indices = np.array(
        [idx for idx in range(len(transitions["observations"])) if idx not in reserved_transition_indices],
        dtype=np.int32,
    )

    variant_summaries = []
    variants_dir = output_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    expected_num_transitions: Optional[int] = None

    for rho in rhos:
        selected_records: List[Mapping[str, object]] = []
        for branch_def in branch_definitions:
            branch_id = int(branch_def["branch_id"])
            branch_budget = min(len(train_by_branch[branch_id][1]), len(train_by_branch[branch_id][0]))
            num_bad = int(round(rho * branch_budget))
            num_good = branch_budget - num_bad
            selected_records.extend(train_by_branch[branch_id][1][:num_good])
            selected_records.extend(train_by_branch[branch_id][0][:num_bad])

        snippet_indices = []
        transition_branch_id: Dict[int, int] = {}
        transition_branch_label: Dict[int, int] = {}
        for record in selected_records:
            indices = transition_indices_for_record(record, horizon)
            snippet_indices.append(indices)
            for idx in indices:
                transition_branch_id[int(idx)] = int(record["branch_id"])
                transition_branch_label[int(idx)] = int(record["label_value"])

        if snippet_indices:
            branch_indices = np.concatenate(snippet_indices, axis=0)
            branch_indices = np.unique(branch_indices)
        else:
            branch_indices = np.zeros((0,), dtype=np.int32)
        final_indices = np.sort(np.concatenate([background_indices, branch_indices], axis=0))

        if expected_num_transitions is None:
            expected_num_transitions = int(len(final_indices))
        elif len(final_indices) != expected_num_transitions:
            raise RuntimeError(
                f"Variant rho={rho:.2f} has {len(final_indices)} transitions; expected {expected_num_transitions}. "
                "This indicates overlapping branch snippets leaked into the bank."
            )

        dataset = {
            "observations": transitions["observations"][final_indices],
            "actions": transitions["actions"][final_indices],
            "next_observations": transitions["next_observations"][final_indices],
            "rewards": transitions["rewards"][final_indices],
            "terminals": transitions["terminals"][final_indices],
            "masks": transitions["masks"][final_indices],
            "source_index": final_indices.astype(np.int32),
            "raw_index": transitions["raw_index"][final_indices].astype(np.int32),
            "episode_id": transitions["episode_id"][final_indices].astype(np.int32),
            "step_in_episode": transitions["step_in_episode"][final_indices].astype(np.int32),
            "xy": transitions["xy"][final_indices].astype(np.float32),
            "goal_xy": transitions["goal_xy"][final_indices].astype(np.float32),
            "qpos": transitions["qpos"][final_indices].astype(np.float32),
            "qvel": transitions["qvel"][final_indices].astype(np.float32),
            "branch_id": np.asarray([transition_branch_id.get(int(idx), -1) for idx in final_indices], dtype=np.int32),
            "branch_label": np.asarray(
                [transition_branch_label.get(int(idx), -1) for idx in final_indices],
                dtype=np.int8,
            ),
        }

        variant_dir = variants_dir / f"rho_{rho:.2f}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = variant_dir / "dataset.npz"
        if compress:
            np.savez_compressed(dataset_path, **dataset)
        else:
            np.savez(dataset_path, **dataset)

        branch_transition_count = int(np.sum(dataset["branch_id"] >= 0))
        unique_state_bins = {xy_to_grid_bin(xy, 1.0) for xy in dataset["xy"]}
        metadata = {
            "rho": float(rho),
            "num_transitions": int(len(final_indices)),
            "num_branch_transitions": branch_transition_count,
            "num_background_transitions": int(len(final_indices) - branch_transition_count),
            "num_selected_snippets": int(len(selected_records)),
            "num_selected_branches": int(len({int(record["branch_id"]) for record in selected_records})),
            "unique_state_bins_1p0": int(len(unique_state_bins)),
            "dataset_path": str(dataset_path),
        }
        with open(variant_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        variant_summaries.append(metadata)

    return variant_summaries


def save_branch_bank_csv(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No branch rows were generated.")
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_probe_bundle(
    probe_records: Sequence[Mapping[str, object]],
    branch_definitions: Sequence[Mapping[str, object]],
    transitions: Mapping[str, np.ndarray],
    output_path: Path,
    snippet_horizon: int,
    env_name: str,
    task_id: int,
) -> None:
    branch_lookup = {int(branch["branch_id"]): branch for branch in branch_definitions}
    probe_rows = []
    for record in probe_records:
        branch = branch_lookup[int(record["branch_id"])]
        idx = int(record["source_index"])
        row = {
            "env_name": np.asarray(env_name),
            "task_id": np.int32(task_id),
            "branch_id": int(record["branch_id"]),
            "label": int(record["label_value"]),
            "source_index": idx,
            "raw_index": int(record["raw_index"]),
            "episode_id": int(record["episode_id"]),
            "step_in_episode": int(record["step_in_episode"]),
            "observation": transitions["observations"][idx],
            "action": transitions["actions"][idx],
            "qpos": transitions["qpos"][idx],
            "qvel": transitions["qvel"][idx],
            "goal_xy": transitions["goal_xy"][idx],
            "start_xy": transitions["xy"][idx],
            "good_direction_vector": np.asarray(branch["good_direction_vector"], dtype=np.float32),
            "bad_direction_vector": np.asarray(branch["bad_direction_vector"], dtype=np.float32),
            "progress": float(record["progress"]),
            "good_direction": int(branch["good_direction"]),
            "bad_direction": int(branch["bad_direction"]),
            "snippet_horizon": snippet_horizon,
        }
        probe_rows.append(row)

    if not probe_rows:
        raise ValueError("No probe states were generated.")

    bundle = {key: np.asarray([row[key] for row in probe_rows]) for key in probe_rows[0]}
    np.savez_compressed(output_path, **bundle)


def maybe_plot_overview(
    transitions: Mapping[str, np.ndarray],
    branch_definitions: Sequence[Mapping[str, object]],
    probe_records: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    xy = transitions["xy"]
    ax.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.08, color="#4d4d4d", label="dataset")

    if branch_definitions:
        branch_xy = np.asarray([branch["cell_center_xy"] for branch in branch_definitions], dtype=np.float32)
        ax.scatter(branch_xy[:, 0], branch_xy[:, 1], s=50, color="#ff7f0e", label="branch cell")

    if probe_records:
        probe_xy = np.asarray([[record["start_x"], record["start_y"]] for record in probe_records], dtype=np.float32)
        probe_colors = ["#1b9e77" if record["label_value"] == 1 else "#d95f02" for record in probe_records]
        ax.scatter(probe_xy[:, 0], probe_xy[:, 1], s=30, color=probe_colors, marker="x", label="probe states")

    ax.set_title("OGBench AntMaze branch mining overview")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "branch_overview.png", dpi=200)
    plt.close(fig)


def dataset_summary(
    args: argparse.Namespace,
    env_name: str,
    dataset_source: str,
    transitions: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    branch_definitions: Sequence[Mapping[str, object]],
    variant_summaries: Sequence[Mapping[str, object]],
    task_id: int,
    goal_xy: np.ndarray,
    train_snippet_cap: int,
) -> Dict[str, object]:
    direction_counts = Counter(
        DIRECTION_NAMES[int(direction)]
        for direction in features["direction"][features["direction"] >= 0]
    )
    branch_progress_gaps = [float(branch["progress_gap"]) for branch in branch_definitions]
    return {
        "env_name": env_name,
        "dataset_source": dataset_source,
        "task_id": int(task_id),
        "goal_xy": goal_xy.astype(float).tolist(),
        "target_branch_transitions": int(args.target_branch_transitions),
        "effective_train_snippet_cap": int(train_snippet_cap),
        "snippet_horizon": int(args.snippet_horizon),
        "num_transitions": int(len(transitions["observations"])),
        "num_episodes": int(len(transitions["episode_starts"])),
        "num_candidate_branches": int(len(branch_definitions)),
        "mean_progress_gap": float(np.mean(branch_progress_gaps)) if branch_progress_gaps else 0.0,
        "max_progress_gap": float(np.max(branch_progress_gaps)) if branch_progress_gaps else 0.0,
        "direction_counts": dict(direction_counts),
        "variant_summaries": list(variant_summaries),
    }


def main() -> None:
    args = parse_args()
    parsed = parse_ogbench_env_name(args.env_name)
    rhos = parse_rho_values(args.rho_values)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_path_from_env_name(args.env_name, args.dataset_dir, args.dataset_path)
    raw = load_npz_dataset(dataset_path)
    transitions = build_ogbench_transitions(raw)
    goal_xy = task_goal_xy(str(parsed["maze_type"]), int(parsed["task_id"]))
    apply_ogbench_singletask_rewards(transitions, goal_xy, OGBENCH_GOAL_TOL)
    reconstruct_episode_metadata(transitions)

    maze_layout = infer_maze_layout(str(parsed["maze_type"]))
    features = build_transition_features(transitions, args, maze_layout)
    candidate_defs = candidate_branches(transitions, features, args)
    train_snippet_cap = resolve_train_snippet_cap(len(candidate_defs), args)
    branch_definitions, branch_rows, train_records, probe_records = build_branch_bank(
        transitions,
        features,
        candidate_defs,
        args,
        train_snippet_cap=train_snippet_cap,
    )

    if not branch_definitions:
        raise RuntimeError(
            "No usable branch bank was found. Try lowering --min_cell_visits, --min_dir_visits, or --min_progress_gap."
        )

    save_branch_bank_csv(branch_rows, output_dir / "branch_bank.csv")
    with open(output_dir / "branch_definitions.json", "w", encoding="utf-8") as handle:
        json.dump(branch_definitions, handle, indent=2, sort_keys=True)
    save_probe_bundle(
        probe_records=probe_records,
        branch_definitions=branch_definitions,
        transitions=transitions,
        output_path=output_dir / "probe_states.npz",
        snippet_horizon=args.snippet_horizon,
        env_name=args.env_name,
        task_id=int(parsed["task_id"]),
    )

    variant_summaries = build_variant_datasets(
        transitions=transitions,
        branch_definitions=branch_definitions,
        train_records=train_records,
        probe_records=probe_records,
        rhos=rhos,
        output_dir=output_dir,
        seed=args.seed,
        compress=not args.uncompressed,
        horizon=args.snippet_horizon,
    )

    config = vars(args).copy()
    config["rho_values"] = rhos
    config["task_id"] = int(parsed["task_id"])
    config["dataset_name"] = str(parsed["dataset_name"])
    config["dataset_path"] = str(dataset_path)
    config["effective_train_snippet_cap"] = int(train_snippet_cap)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            dataset_summary(
                args=args,
                env_name=args.env_name,
                dataset_source=str(dataset_path),
                transitions=transitions,
                features=features,
                branch_definitions=branch_definitions,
                variant_summaries=variant_summaries,
                task_id=int(parsed["task_id"]),
                goal_xy=goal_xy,
                train_snippet_cap=train_snippet_cap,
            ),
            handle,
            indent=2,
            sort_keys=True,
        )

    if not args.skip_plots:
        maybe_plot_overview(transitions, branch_definitions, probe_records, output_dir)

    print(
        "Resolved train snippet cap:",
        train_snippet_cap,
        f"(target_branch_transitions={args.target_branch_transitions})",
    )
    print(f"Wrote OGBench branch bank and variants to {output_dir}")


if __name__ == "__main__":
    main()
