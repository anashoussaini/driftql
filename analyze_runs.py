"""
Analyze DriftQL experiment runs to find best hyperparameters.

Usage:
    python analyze_runs.py --base_dir exp/fql/DriftQLScan_Fine

Filters runs matching:
    drift_temps : 0.2, 0.5, 0.8
    alpha       : 1, 5, 10, 20
    discount    : 0.99
    q_agg       : mean
    q_agg_actor : mean  (optional field, ignored if missing)
    seeds       : 10, 11, 12
"""

import os
import json
import csv
import argparse
import math
import shutil
from collections import defaultdict

# ── Filters ──────────────────────────────────────────────────────────────────
INTEREST = {
    "drift_temps": {0.2, 0.5, 0.8},
    "alpha":       {1, 5, 10, 20},
    "discount":    {0.99},
    "q_agg":       {"mean"},
    "q_agg_actor": {"mean"},
    "seeds":       {13, 14, 15, 16, 17},
}

METRIC_PRIORITY = ["evaluation/success", "evaluation/episode.normalized_return"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_flags(folder):
    path = os.path.join(folder, "flags.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] Failed to read flags.json in {folder}: {e}")

def get_agent_param(flags, key, default=None):
    """Read from flags['agent'][key] or flags[key]."""
    agent = flags.get("agent", {})
    return agent.get(key, flags.get(key, default))


def best_metric_from_csv(folder):
    """
    Return (metric_name, best_value) using the first available metric in
    METRIC_PRIORITY.  best = max over all eval rows.
    Returns (None, None) if file is unreadable or metric absent.
    """
    path = os.path.join(folder, "eval.csv")
    if not os.path.exists(path):
        return None, None

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None, None

    # Pick the first metric that exists in this CSV
    metric = None
    for m in METRIC_PRIORITY:
        if m in rows[0]:
            metric = m
            break
    if metric is None:
        return None, None

    best = max(float(r[metric]) for r in rows if r[metric] not in ("", "nan"))
    return metric, best


def passes_filter(flags):
    """Return True if this run matches the interest filters."""
    agent = flags.get("agent", {})

    seed = flags.get("seed")
    if seed not in INTEREST["seeds"]:
        return False

    drift_temps = agent.get("drift_temps")
    if drift_temps not in INTEREST["drift_temps"]:
        return False

    alpha = agent.get("alpha")
    if alpha not in INTEREST["alpha"]:
        return False

    discount = agent.get("discount", flags.get("discount"))
    if discount not in INTEREST["discount"]:
        return False

    q_agg = agent.get("q_agg", flags.get("q_agg"))
    if q_agg not in INTEREST["q_agg"]:
        return False

    q_agg_actor = agent.get("q_agg_actor", flags.get("q_agg_actor"))
    if q_agg_actor not in INTEREST["q_agg_actor"]:
        return False

    return True


def config_key(flags):
    agent = flags.get("agent", {})
    return (
        agent.get("drift_temps"),
        agent.get("alpha"),
        agent.get("discount", flags.get("discount")),
        agent.get("q_agg", flags.get("q_agg")),
        agent.get("q_agg_actor", flags.get("q_agg_actor")),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main(base_dir):
    # data[env][config_key][seed] = best_value
    data = defaultdict(lambda: defaultdict(dict))
    metric_names = {}   # env -> metric name (for display)
    skipped = 0
    processed = 0

    entries = sorted(os.listdir(base_dir))
    for entry in entries:
        folder = os.path.join(base_dir, entry)
        if not os.path.isdir(folder):
            continue

        flags = load_flags(folder)
        if flags is None:
            skipped += 1
            continue

        if not passes_filter(flags):
            skipped += 1
            continue

        env   = flags.get("env_name", "unknown")
        seed  = flags.get("seed")
        cfg   = config_key(flags)

        metric, best_val = best_metric_from_csv(folder)
        if metric is None:
            skipped += 1
            continue

        # Keep best across duplicate runs with same (env, cfg, seed)
        prev = data[env][cfg].get(seed, -math.inf)
        data[env][cfg][seed] = max(prev, best_val)
        metric_names[env] = metric
        processed += 1

    print(f"\nScanned {len(entries)} folders — {processed} matched, {skipped} skipped.\n")
    print("=" * 80)

    for env in sorted(data):
        metric = metric_names.get(env, "?")
        print(f"\nENV : {env}  (metric: {metric})")
        print("-" * 70)

        configs = data[env]

        # ── Per-config summary ────────────────────────────────────────────────
        config_rows = []
        for cfg, seed_dict in configs.items():
            drift_temps, alpha, discount, q_agg, q_agg_actor = cfg
            vals = list(seed_dict.values())
            avg = sum(vals) / len(vals)
            std = math.sqrt(sum((v - avg) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
            config_rows.append((avg, std, cfg, seed_dict))

        config_rows.sort(key=lambda x: -x[0])  # sort by avg desc

        # ── Best config ───────────────────────────────────────────────────────
        best_avg, best_std, best_cfg, best_seeds = config_rows[0]
        drift_temps, alpha, discount, q_agg, q_agg_actor = best_cfg
        print(f"  ★ BEST CONFIG  drift_temps={drift_temps}  alpha={alpha}  "
              f"discount={discount}  q_agg={q_agg}, q_agg_actor={q_agg_actor}")
        print(f"    avg ± std  :  {best_avg:.4f} ± {best_std:.4f}  "
              f"(over {len(best_seeds)} seed(s))")
        # for s in sorted(best_seeds):
        #     print(f"    seed {s:>3d}   :  {best_seeds[s]:.4f}")

        # # ── Full table ────────────────────────────────────────────────────────
        # print()
        # hdr = f"  {'drift_temps':>12}  {'alpha':>6}  {'avg':>8}  {'std':>8}"
        # all_seeds = sorted({s for _, _, _, sd in config_rows for s in sd})
        # for s in all_seeds:
        #     hdr += f"  {'sd'+str(s):>8}"
        # print(hdr)
        # print("  " + "-" * (len(hdr) - 2))
        #
        # for avg, std, cfg, seed_dict in config_rows:
        #     drift_temps, alpha, *_ = cfg
        #     row = f"  {str(drift_temps):>12}  {str(alpha):>6}  {avg:8.4f}  {std:8.4f}"
        #     for s in all_seeds:
        #         v = seed_dict.get(s)
        #         row += f"  {v:8.4f}" if v is not None else f"  {'—':>8}"
        #     print(row)

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="exp/fql/DriftQLScan_Fine",
        help="Path to the scan directory",
    )
    args = parser.parse_args()
    main(args.base_dir)
