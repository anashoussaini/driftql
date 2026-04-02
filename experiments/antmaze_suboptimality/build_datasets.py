#!/usr/bin/env python3
"""Build controlled AntMaze suboptimality datasets.

This script mines branch states from a raw D4RL AntMaze dataset, separates
good/bad in-support continuations, and exports dataset variants with different
bad-branch ratios.

The main artifact layout is:

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
        rho_0.25/
          ...

The exported `dataset.npz` files keep the current repo's transition convention:
`observations`, `actions`, `next_observations`, `rewards`, `terminals`, `masks`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import numpy as np


# D4RL AntMaze uses a 4.0 world-unit maze cell size.
MAZE_CELL_SIZE = 4.0

U_MAZE_TEST = (
    (1, 1, 1, 1, 1),
    (1, "R", 0, 0, 1),
    (1, 1, 1, 0, 1),
    (1, "G", 0, 0, 1),
    (1, 1, 1, 1, 1),
)

BIG_MAZE_TEST = (
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, "R", 0, 0, 0, 0, "G", 1),
    (1, 0, 1, 0, 1, 1, 0, 1),
    (1, 0, 0, 0, 0, 1, 0, 1),
    (1, 1, 1, 0, 0, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 1, 1, 0, 0, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
)

HARDEST_MAZE_TEST = (
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, "R", 0, 1, 0, 0, 0, 1, 0, "G", 0, 1),
    (1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1),
    (1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1),
    (1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1),
    (1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1),
    (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
)

MAZE_LAYOUTS = {
    "antmaze-umaze": U_MAZE_TEST,
    "antmaze-medium": BIG_MAZE_TEST,
    "antmaze-large": HARDEST_MAZE_TEST,
}

DIRECTION_NAMES = {
    0: "east",
    1: "north",
    2: "west",
    3: "south",
    4: "stay",
}

DIRECTION_VECTORS = {
    0: np.array([1.0, 0.0], dtype=np.float32),
    1: np.array([0.0, 1.0], dtype=np.float32),
    2: np.array([-1.0, 0.0], dtype=np.float32),
    3: np.array([0.0, -1.0], dtype=np.float32),
    4: np.array([0.0, 0.0], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", type=str, default="antmaze-medium-diverse-v2")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional path to a raw AntMaze HDF5 dataset. If omitted, D4RL env loading is used.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rho_values", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--spatial_bin_size", type=float, default=2.0)
    parser.add_argument("--direction_lookahead", type=int, default=6)
    parser.add_argument("--snippet_horizon", type=int, default=20)
    parser.add_argument("--distance_metric", choices=("euclidean", "geodesic"), default="euclidean")
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for branch subsampling and rho-variant construction.",
    )
    parser.add_argument("--uncompressed", action="store_true", help="Save dataset variants with `np.savez`.")
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
    """Choose a per-branch/per-label snippet cap that can hit the desired branch mass.

    At rho=0 or rho=1, each usable branch contributes one label's snippets only, so
    the branch-controlled transition mass scales approximately as:

        num_branches * per_label_snippets * snippet_horizon

    The current repo's original `max_train_snippets_per_label` acts as a floor; this
    helper increases it when `target_branch_transitions` asks for a larger pool.
    """

    base_cap = max(int(args.max_train_snippets_per_label), 1)
    target_branch_transitions = int(getattr(args, "target_branch_transitions", 0))
    if target_branch_transitions <= 0 or num_candidate_branches <= 0:
        return base_cap

    required_per_branch = math.ceil(
        target_branch_transitions / (num_candidate_branches * max(int(args.snippet_horizon), 1))
    )
    return max(base_cap, required_per_branch)


def infer_maze_layout(env_name: str) -> Optional[Tuple[Tuple[object, ...], ...]]:
    if "antmaze-umaze" in env_name:
        return MAZE_LAYOUTS["antmaze-umaze"]
    if "antmaze-medium" in env_name:
        return MAZE_LAYOUTS["antmaze-medium"]
    if "antmaze-large" in env_name:
        return MAZE_LAYOUTS["antmaze-large"]
    return None


def load_hdf5_dataset(dataset_path: str) -> Dict[str, np.ndarray]:
    dataset: Dict[str, np.ndarray] = {}
    with h5py.File(dataset_path, "r") as handle:
        def visitor(name: str, item: h5py.Dataset) -> None:
            if isinstance(item, h5py.Dataset):
                dataset[name] = item[:]

        handle.visititems(visitor)
    return dataset


def load_d4rl_dataset(env_name: str) -> Dict[str, np.ndarray]:
    # Keep heavy RL imports local so `--help` and `py_compile` do not require the full env.
    import d4rl  # noqa: F401
    import gym

    env = gym.make(env_name)
    return env.get_dataset()


def load_raw_dataset(env_name: str, dataset_path: Optional[str]) -> Tuple[Dict[str, np.ndarray], str]:
    if dataset_path is not None:
        return load_hdf5_dataset(dataset_path), dataset_path
    raw = load_d4rl_dataset(env_name)
    return raw, "<d4rl-env>"


def build_qlearning_transitions(raw: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    rewards = np.asarray(raw["rewards"])
    observations = np.asarray(raw["observations"])
    actions = np.asarray(raw["actions"])
    terminals = np.asarray(raw["terminals"]).astype(bool)
    use_timeouts = "timeouts" in raw
    timeouts = np.asarray(raw["timeouts"]).astype(bool) if use_timeouts else None

    obs_list = []
    next_obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    raw_index_list = []
    next_raw_index_list = []

    episode_step = 0
    num_samples = rewards.shape[0]
    for raw_idx in range(num_samples - 1):
        done_bool = bool(terminals[raw_idx])
        final_timestep = bool(timeouts[raw_idx]) if use_timeouts else False
        if final_timestep:
            episode_step = 0
            continue

        obs_list.append(observations[raw_idx].astype(np.float32))
        next_obs_list.append(observations[raw_idx + 1].astype(np.float32))
        action_list.append(actions[raw_idx].astype(np.float32))
        reward_list.append(np.float32(rewards[raw_idx]))
        done_list.append(done_bool)
        raw_index_list.append(raw_idx)
        next_raw_index_list.append(raw_idx + 1)

        if done_bool:
            episode_step = 0
        else:
            episode_step += 1

    return {
        "observations": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(action_list, dtype=np.float32),
        "next_observations": np.asarray(next_obs_list, dtype=np.float32),
        "raw_rewards": np.asarray(reward_list, dtype=np.float32),
        "done_raw": np.asarray(done_list, dtype=np.float32),
        "raw_index": np.asarray(raw_index_list, dtype=np.int32),
        "next_raw_index": np.asarray(next_raw_index_list, dtype=np.int32),
    }


def apply_repo_antmaze_conventions(transitions: MutableMapping[str, np.ndarray]) -> None:
    next_obs = transitions["next_observations"]
    observations = transitions["observations"]
    done_raw = transitions["done_raw"]
    num_transitions = observations.shape[0]

    terminals = np.zeros(num_transitions, dtype=np.float32)
    masks = np.zeros(num_transitions, dtype=np.float32)
    rewards = transitions["raw_rewards"].copy().astype(np.float32) - 1.0

    if num_transitions == 0:
        transitions["terminals"] = terminals
        transitions["masks"] = masks
        transitions["rewards"] = rewards
        return

    for idx in range(num_transitions - 1):
        terminals[idx] = float(np.linalg.norm(observations[idx + 1] - next_obs[idx]) > 1e-6)
        masks[idx] = 1.0 - done_raw[idx]

    masks[-1] = 1.0 - done_raw[-1]
    terminals[-1] = 1.0

    transitions["terminals"] = terminals
    transitions["masks"] = masks
    transitions["rewards"] = rewards


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


def infer_goal_from_observation(observations: np.ndarray) -> np.ndarray:
    if observations.shape[-1] < 31:
        raise ValueError(
            "Goal coordinates are not present in observations and `infos/goal` is missing from the dataset."
        )
    return observations[:, :2] + observations[:, -2:]


def attach_raw_metadata(transitions: MutableMapping[str, np.ndarray], raw: Mapping[str, np.ndarray]) -> None:
    raw_index = transitions["raw_index"]
    next_raw_index = transitions["next_raw_index"]

    if "infos/qpos" in raw:
        qpos = np.asarray(raw["infos/qpos"], dtype=np.float32)
        transitions["qpos"] = qpos[raw_index]
        transitions["next_qpos"] = qpos[next_raw_index]
        transitions["xy"] = transitions["qpos"][:, :2]
        transitions["next_xy"] = transitions["next_qpos"][:, :2]
    else:
        transitions["xy"] = transitions["observations"][:, :2].astype(np.float32)
        transitions["next_xy"] = transitions["next_observations"][:, :2].astype(np.float32)

    if "infos/qvel" in raw:
        qvel = np.asarray(raw["infos/qvel"], dtype=np.float32)
        transitions["qvel"] = qvel[raw_index]
        transitions["next_qvel"] = qvel[next_raw_index]

    if "infos/goal" in raw:
        goals = np.asarray(raw["infos/goal"], dtype=np.float32)
        transitions["goal_xy"] = goals[raw_index]
    else:
        transitions["goal_xy"] = infer_goal_from_observation(transitions["observations"]).astype(np.float32)


def xy_to_grid_bin(xy: np.ndarray, bin_size: float) -> Tuple[int, int]:
    return (int(math.floor(float(xy[0]) / bin_size)), int(math.floor(float(xy[1]) / bin_size)))


def direction_bin(delta: np.ndarray, min_norm: float) -> int:
    norm = float(np.linalg.norm(delta))
    if norm < min_norm:
        return 4
    if abs(delta[0]) >= abs(delta[1]):
        return 0 if delta[0] >= 0 else 2
    return 1 if delta[1] >= 0 else 3


def maze_xy_to_rowcol(xy: np.ndarray) -> Tuple[int, int]:
    x = max(float(xy[0]), 1e-4)
    y = max(float(xy[1]), 1e-4)
    return (int(1 + y / MAZE_CELL_SIZE), int(1 + x / MAZE_CELL_SIZE))


def is_free_cell(maze_layout: Tuple[Tuple[object, ...], ...], row: int, col: int) -> bool:
    if row < 0 or col < 0 or row >= len(maze_layout) or col >= len(maze_layout[0]):
        return False
    return maze_layout[row][col] != 1


@lru_cache(maxsize=None)
def shortest_path_steps(
    maze_layout: Tuple[Tuple[object, ...], ...],
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
    maze_layout: Optional[Tuple[Tuple[object, ...], ...]],
) -> float:
    if metric == "euclidean" or maze_layout is None:
        return float(np.linalg.norm(goal_xy - start_xy))

    start_cell = maze_xy_to_rowcol(start_xy)
    goal_cell = maze_xy_to_rowcol(goal_xy)
    steps = shortest_path_steps(maze_layout, start_cell, goal_cell)
    if steps is None:
        return float(np.linalg.norm(goal_xy - start_xy))
    return float(steps * MAZE_CELL_SIZE)


def build_transition_features(
    transitions: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    maze_layout: Optional[Tuple[Tuple[object, ...], ...]],
) -> Dict[str, np.ndarray]:
    num_transitions = transitions["observations"].shape[0]
    episode_ends = transitions["episode_ends"]
    episode_id = transitions["episode_id"]
    step_in_episode = transitions["step_in_episode"]

    progress = np.full(num_transitions, np.nan, dtype=np.float32)
    direction = np.full(num_transitions, -1, dtype=np.int32)
    direction_delta = np.zeros((num_transitions, 2), dtype=np.float32)
    valid = np.zeros(num_transitions, dtype=bool)
    goal_distance = np.zeros(num_transitions, dtype=np.float32)
    future_success = np.zeros(num_transitions, dtype=bool)
    full_episode_success = np.zeros(num_transitions, dtype=bool)
    spatial_bins = np.zeros((num_transitions, 2), dtype=np.int32)

    episode_end_by_id = {int(episode_id[end_idx]): int(end_idx) for end_idx in episode_ends}
    done_raw = transitions["done_raw"] > 0

    episode_success = {}
    for ep_id, end_idx in episode_end_by_id.items():
        start_idx = int(np.searchsorted(transitions["episode_id"], ep_id, side="left"))
        episode_success[ep_id] = bool(done_raw[start_idx : end_idx + 1].any())

    for idx in range(num_transitions):
        ep_id = int(episode_id[idx])
        full_episode_success[idx] = episode_success[ep_id]
        spatial_bins[idx] = xy_to_grid_bin(transitions["xy"][idx], args.spatial_bin_size)
        goal_distance[idx] = point_distance(
            transitions["xy"][idx],
            transitions["goal_xy"][idx],
            args.distance_metric,
            maze_layout,
        )

        future_idx = idx + args.direction_lookahead
        snippet_end_idx = idx + args.snippet_horizon
        episode_end = episode_end_by_id[ep_id]
        if future_idx > episode_end or snippet_end_idx > episode_end:
            continue

        if goal_distance[idx] < args.min_branch_distance_to_goal:
            continue

        future_xy = transitions["xy"][future_idx]
        snippet_xy = transitions["xy"][snippet_end_idx]
        delta = future_xy - transitions["xy"][idx]
        direction[idx] = direction_bin(delta, args.min_direction_norm)
        direction_delta[idx] = delta.astype(np.float32)
        progress[idx] = (
            point_distance(transitions["xy"][idx], transitions["goal_xy"][idx], args.distance_metric, maze_layout)
            - point_distance(snippet_xy, transitions["goal_xy"][idx], args.distance_metric, maze_layout)
        )
        future_success[idx] = bool(done_raw[idx : snippet_end_idx + 1].any())
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


def select_disjoint_candidates(
    candidate_indices: Sequence[int],
    scores: Sequence[float],
    episode_id: np.ndarray,
    step_in_episode: np.ndarray,
    horizon: int,
    limit: Optional[int],
) -> List[int]:
    ranked = sorted(zip(candidate_indices, scores), key=lambda item: item[1], reverse=True)
    occupied: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    selected: List[int] = []

    for idx, _ in ranked:
        ep_id = int(episode_id[idx])
        interval = (int(step_in_episode[idx]), int(step_in_episode[idx] + horizon - 1))
        overlaps = False
        for taken_start, taken_end in occupied[ep_id]:
            if not (interval[1] < taken_start or interval[0] > taken_end):
                overlaps = True
                break
        if overlaps:
            continue

        occupied[ep_id].append(interval)
        selected.append(int(idx))
        if limit is not None and len(selected) >= limit:
            break

    return selected


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

    for branch in branches:
        good_indices = list(branch["good_indices"])
        bad_indices = list(branch["bad_indices"])

        max_candidates = args.num_probe_per_label + max(train_snippet_cap, 1)
        good_selected = select_disjoint_candidates(
            good_indices,
            scores=list(features["progress"][good_indices]),
            episode_id=transitions["episode_id"],
            step_in_episode=transitions["step_in_episode"],
            horizon=args.snippet_horizon,
            limit=max_candidates,
        )
        bad_selected = select_disjoint_candidates(
            bad_indices,
            scores=list(-features["progress"][bad_indices]),
            episode_id=transitions["episode_id"],
            step_in_episode=transitions["step_in_episode"],
            horizon=args.snippet_horizon,
            limit=max_candidates,
        )

        if len(good_selected) <= args.num_probe_per_label or len(bad_selected) <= args.num_probe_per_label:
            continue

        probe_good = good_selected[: args.num_probe_per_label]
        probe_bad = bad_selected[: args.num_probe_per_label]
        train_good = good_selected[args.num_probe_per_label :]
        train_bad = bad_selected[args.num_probe_per_label :]

        usable_train = min(len(train_good), len(train_bad), train_snippet_cap)
        if usable_train == 0:
            continue

        train_good = train_good[:usable_train]
        train_bad = train_bad[:usable_train]

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

    for rho in rhos:
        selected_records: List[Mapping[str, object]] = []
        for branch_def in branch_definitions:
            branch_id = int(branch_def["branch_id"])
            branch_budget = min(len(train_by_branch[branch_id][1]), len(train_by_branch[branch_id][0]))
            num_bad = int(round(rho * branch_budget))
            num_good = branch_budget - num_bad
            selected_records.extend(train_by_branch[branch_id][1][:num_good])
            selected_records.extend(train_by_branch[branch_id][0][:num_bad])

        selected_transition_indices = set(map(int, background_indices.tolist()))
        transition_branch_id: Dict[int, int] = {}
        transition_branch_label: Dict[int, int] = {}

        for record in selected_records:
            indices = transition_indices_for_record(record, horizon)
            for idx in indices:
                selected_transition_indices.add(int(idx))
                transition_branch_id[int(idx)] = int(record["branch_id"])
                transition_branch_label[int(idx)] = int(record["label_value"])

        final_indices = np.array(sorted(selected_transition_indices), dtype=np.int32)
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
        unique_state_bins = {
            xy_to_grid_bin(xy, 1.0)
            for xy in dataset["xy"]
        }
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
) -> None:
    branch_lookup = {int(branch["branch_id"]): branch for branch in branch_definitions}
    probe_rows = []
    for record in probe_records:
        branch = branch_lookup[int(record["branch_id"])]
        idx = int(record["source_index"])
        row = {
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

    ax.set_title("AntMaze branch mining overview")
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
    rhos = parse_rho_values(args.rho_values)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, dataset_source = load_raw_dataset(args.env_name, args.dataset_path)
    transitions = build_qlearning_transitions(raw)
    apply_repo_antmaze_conventions(transitions)
    reconstruct_episode_metadata(transitions)
    attach_raw_metadata(transitions, raw)

    if "qpos" not in transitions or "qvel" not in transitions:
        raise ValueError(
            "Raw dataset is missing `infos/qpos`/`infos/qvel`. The probe script requires them for state restoration."
        )

    maze_layout = infer_maze_layout(args.env_name)
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
            "No usable branch bank was found. Try lowering `--min_cell_visits`, `--min_dir_visits`, "
            "or `--min_progress_gap`."
        )

    save_branch_bank_csv(branch_rows, output_dir / "branch_bank.csv")
    with open(output_dir / "branch_definitions.json", "w", encoding="utf-8") as handle:
        json.dump(branch_definitions, handle, indent=2, sort_keys=True)
    save_probe_bundle(
        probe_records,
        branch_definitions,
        transitions,
        output_dir / "probe_states.npz",
        args.snippet_horizon,
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
    config["effective_train_snippet_cap"] = int(train_snippet_cap)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            dataset_summary(
                args,
                args.env_name,
                dataset_source,
                transitions,
                features,
                branch_definitions,
                variant_summaries,
                train_snippet_cap,
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
    print(f"Wrote branch bank and variants to {output_dir}")


if __name__ == "__main__":
    main()
