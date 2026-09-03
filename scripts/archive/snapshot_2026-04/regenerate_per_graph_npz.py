"""
regenerate_per_graph_npz.py
===========================
Regenerate per-graph NPZ files from the full aggregated NPZ.
Each graph has 31,635 nodes, and there are 100 graphs total.

Usage:
    python regenerate_per_graph_npz.py
"""

import os
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# T8 data
T8_DIR = os.path.join(
    REPO_ROOT, "data", "TR-C_Benchmarks", "point_net_transf_gat_8th_trial_lower_dropout"
)
T8_FULL_NPZ = os.path.join(T8_DIR, "uq_results", "mc_dropout_full_100graphs_mc30.npz")
T8_PER_GRAPH_DIR = os.path.join(T8_DIR, "uq_results", "checkpoints_mc30")

# T7 data
T7_DIR = os.path.join(
    REPO_ROOT,
    "data",
    "TR-C_Benchmarks",
    "point_net_transf_gat_7th_trial_80_10_10_split",
)
T7_FULL_NPZ = os.path.join(T7_DIR, "uq_results", "mc_dropout_full_100graphs_mc30.npz")
T7_PER_GRAPH_DIR = os.path.join(T7_DIR, "uq_results", "checkpoints_mc30")

NODES_PER_GRAPH = 31635
N_GRAPHS = 100


def split_npz(full_npz_path, out_dir, label):
    """Split a full NPZ into per-graph NPZ files."""
    print(f"\n  Loading {label} full NPZ: {full_npz_path}")
    if not os.path.exists(full_npz_path):
        print(f"  WARNING: {full_npz_path} not found, skipping.")
        return False

    data = np.load(full_npz_path)
    predictions = data["predictions"]
    uncertainties = data["uncertainties"]
    targets = data["targets"]

    total_nodes = len(predictions)
    expected = NODES_PER_GRAPH * N_GRAPHS
    print(f"  Total nodes: {total_nodes:,} (expected {expected:,})")

    if total_nodes != expected:
        print(f"  ERROR: Node count mismatch! Cannot split.")
        return False

    os.makedirs(out_dir, exist_ok=True)

    for g in range(N_GRAPHS):
        start = g * NODES_PER_GRAPH
        end = start + NODES_PER_GRAPH
        out_path = os.path.join(out_dir, f"graph_{g:04d}.npz")
        np.savez(
            out_path,
            predictions=predictions[start:end],
            uncertainties=uncertainties[start:end],
            targets=targets[start:end],
        )
        if (g + 1) % 20 == 0:
            print(f"    Saved graph {g + 1}/{N_GRAPHS}")

    print(f"  Done: {N_GRAPHS} per-graph NPZ files in {out_dir}")
    return True


if __name__ == "__main__":
    print("Regenerating per-graph NPZ files from full aggregated NPZ...")
    split_npz(T8_FULL_NPZ, T8_PER_GRAPH_DIR, "T8")
    split_npz(T7_FULL_NPZ, T7_PER_GRAPH_DIR, "T7")
    print("\nAll done.")
