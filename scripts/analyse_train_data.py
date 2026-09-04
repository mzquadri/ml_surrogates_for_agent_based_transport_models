"""Summarise the published training dataloaders.

Reproduces every number in ``docs/DATASET.md`` from the batch files themselves, so the
documentation can be re-derived rather than trusted. One batch is held in memory at a time and
the statistics are accumulated as running sums, which keeps the peak around 1 GB instead of the
2.44 GiB the full corpus would need.

Usage::

    python scripts/analyse_train_data.py
    python scripts/analyse_train_data.py --data-dir <dir> --json summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "code" / "data" / "train_data" / "dist_not_connected_10k_1pct"

# Column order written by scripts/data_preprocessing/process_simulations_for_gnn.py.
# Only the first five reach the network; HIGHWAY is dropped by the ablation feature set.
FEATURES = (
    "VOL_BASE_CASE",
    "CAPACITY_BASE_CASE",
    "CAPACITY_REDUCTION",
    "FREESPEED",
    "HIGHWAY",
    "LENGTH",
)

# Enough distinct values to separate a category code from a continuous quantity.
CARDINALITY_CAP = 60


def batch_paths(data_dir: Path) -> list[Path]:
    """Batch files in numeric order, so batch_2 does not sort after batch_10."""
    paths = list(data_dir.glob("datalist_batch_*.pt"))
    if not paths:
        raise SystemExit(
            f"No datalist_batch_*.pt under {data_dir}.\n"
            "Fetch them first:\n"
            "  gh release download train-data-v1 --repo mzquadri/ml_surrogates_for_agent_based_transport_models \\\n"
            "    --pattern 'datalist_batch_*.pt' --dir <dir>"
        )
    return sorted(paths, key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def summarise(data_dir: Path) -> dict[str, Any]:
    paths = batch_paths(data_dir)
    n_features = len(FEATURES)

    nodes = 0
    graphs = 0
    node_counts: set[int] = set()
    edge_counts: set[int] = set()

    f_sum = np.zeros(n_features)
    f_sq = np.zeros(n_features)
    f_min = np.full(n_features, np.inf)
    f_max = np.full(n_features, -np.inf)
    f_zero = np.zeros(n_features, dtype=np.int64)
    f_nan = np.zeros(n_features, dtype=np.int64)
    f_seen: list[set[float]] = [set() for _ in range(n_features)]

    y_sum = 0.0
    y_sq = 0.0
    y_min, y_max = np.inf, -np.inf
    y_zero = 0
    y_nan = 0
    y_sample: list[np.ndarray] = []

    pos_min = np.full(2, np.inf)
    pos_max = np.full(2, -np.inf)

    # Reference copies from the first graph, to test what actually varies per scenario.
    ref_x: np.ndarray | None = None
    ref_edges: np.ndarray | None = None
    constant = np.ones(n_features, dtype=bool)
    max_delta = np.zeros(n_features)
    topology_constant = True

    for index, path in enumerate(paths, start=1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = torch.load(path, map_location="cpu", weights_only=False)

        graphs += len(batch)
        for data in batch:
            x = data.x.numpy().astype(np.float64)
            y = data.y.numpy().astype(np.float64).ravel()
            pos = data.pos.numpy().astype(np.float64)
            edges = data.edge_index.numpy()

            node_counts.add(x.shape[0])
            edge_counts.add(edges.shape[1])
            nodes += x.shape[0]

            f_sum += x.sum(axis=0)
            f_sq += (x**2).sum(axis=0)
            f_min = np.minimum(f_min, x.min(axis=0))
            f_max = np.maximum(f_max, x.max(axis=0))
            f_zero += (x == 0).sum(axis=0)
            f_nan += np.isnan(x).sum(axis=0)

            y_sum += float(y.sum())
            y_sq += float((y**2).sum())
            y_min = min(y_min, float(y.min()))
            y_max = max(y_max, float(y.max()))
            y_zero += int((y == 0).sum())
            y_nan += int(np.isnan(y).sum())
            if len(y_sample) < 200:
                y_sample.append(y[:20000].copy())

            flat = pos.reshape(-1, 2)
            pos_min = np.minimum(pos_min, flat.min(axis=0))
            pos_max = np.maximum(pos_max, flat.max(axis=0))

            for column in range(n_features):
                if len(f_seen[column]) <= CARDINALITY_CAP:
                    f_seen[column].update(np.unique(x[:4000, column]).tolist())

            if ref_x is None:
                ref_x, ref_edges = x.copy(), edges.copy()
            else:
                assert ref_edges is not None
                if x.shape == ref_x.shape:
                    delta = np.abs(x - ref_x).max(axis=0)
                    max_delta = np.maximum(max_delta, delta)
                    constant &= delta == 0
                if edges.shape != ref_edges.shape or not np.array_equal(edges, ref_edges):
                    topology_constant = False

        print(f"  read {index}/{len(paths)}: {path.name}", file=sys.stderr, flush=True)

    f_mean = f_sum / nodes
    f_std = np.sqrt(np.maximum(f_sq / nodes - f_mean**2, 0.0))
    y_mean = y_sum / nodes
    y_std = float(np.sqrt(max(y_sq / nodes - y_mean**2, 0.0)))
    percentiles = np.percentile(np.concatenate(y_sample), [1, 5, 25, 50, 75, 95, 99])

    return {
        "corpus": {
            "batch_files": len(paths),
            "graphs": graphs,
            "nodes_per_graph": sorted(node_counts),
            "edges_per_graph": sorted(edge_counts),
            "node_observations": nodes,
            "topology_constant": topology_constant,
        },
        "features": {
            name: {
                "min": float(f_min[i]),
                "max": float(f_max[i]),
                "mean": float(f_mean[i]),
                "std": float(f_std[i]),
                "pct_zero": float(100 * f_zero[i] / nodes),
                "distinct_at_least": len(f_seen[i]),
                "constant_across_scenarios": bool(constant[i]),
                "max_delta_across_scenarios": float(max_delta[i]),
            }
            for i, name in enumerate(FEATURES)
        },
        "target": {
            "mean": y_mean,
            "std": y_std,
            "min": float(y_min),
            "max": float(y_max),
            "pct_zero": float(100 * y_zero / nodes),
            "nan": y_nan,
            "percentiles": {
                f"p{p}": float(v) for p, v in zip((1, 5, 25, 50, 75, 95, 99), percentiles)
            },
        },
        "position": {
            "axis_0": [float(pos_min[0]), float(pos_max[0])],
            "axis_1": [float(pos_min[1]), float(pos_max[1])],
        },
        "nan_in_features": int(f_nan.sum()),
    }


def report(summary: dict[str, Any]) -> None:
    corpus = summary["corpus"]
    print("\nCorpus")
    print(f"  batch files       {corpus['batch_files']}")
    print(f"  graphs            {corpus['graphs']}")
    print(f"  nodes per graph   {corpus['nodes_per_graph']}")
    print(f"  edges per graph   {corpus['edges_per_graph']}")
    print(f"  node observations {corpus['node_observations']:,}")
    print(f"  topology constant {corpus['topology_constant']}")

    print("\nNode features")
    header = f"  {'feature':<21}{'min':>12}{'max':>13}{'mean':>12}{'std':>12}{'%zero':>9}{'varies':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, stats in summary["features"].items():
        varies = "no" if stats["constant_across_scenarios"] else "YES"
        print(
            f"  {name:<21}{stats['min']:>12.2f}{stats['max']:>13.2f}{stats['mean']:>12.2f}"
            f"{stats['std']:>12.2f}{stats['pct_zero']:>8.2f}%{varies:>8}"
        )
    print(f"\n  NaNs in x: {summary['nan_in_features']}")

    target = summary["target"]
    print("\nTarget (Delta v, veh/h)")
    print(f"  mean {target['mean']:.4f}  std {target['std']:.4f}")
    print(f"  min {target['min']:.2f}  max {target['max']:.2f}")
    print(f"  exact zeros {target['pct_zero']:.2f}%   NaNs {target['nan']}")
    print("  " + "  ".join(f"{k}={v:.2f}" for k, v in target["percentiles"].items()))

    position = summary["position"]
    print("\nPosition")
    print(f"  axis 0 {position['axis_0'][0]:.3f} .. {position['axis_0'][1]:.3f}")
    print(f"  axis 1 {position['axis_1'][0]:.3f} .. {position['axis_1'][1]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--json", type=Path, help="also write the summary here")
    args = parser.parse_args()

    summary = summarise(args.data_dir)
    report(summary)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
