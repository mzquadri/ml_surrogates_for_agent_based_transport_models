"""Compute bootstrap confidence intervals for Spearman rho.

Uses the per-graph rho values (100 test graphs) for a graph-level bootstrap,
and also a block bootstrap (resample graphs, average per-graph rhos).

Usage: conda run -n thesis-env python scripts/compute_bootstrap_ci.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
PER_GRAPH = REPO / "docs/verified/phase3_results/per_graph_variation_t8.json"
T8_NPZ = (
    REPO
    / "data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz"
)
OUTPUT = REPO / "docs/verified/phase3_results/bootstrap_ci_results.json"

np.random.seed(42)
B = 10000  # bootstrap replicates


def bootstrap_ci(values, B=10000, alpha=0.05):
    """Percentile bootstrap CI for the mean of values."""
    n = len(values)
    means = np.empty(B)
    for b in range(B):
        idx = np.random.randint(0, n, size=n)
        means[b] = np.mean(values[idx])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi), float(np.std(means))


def graph_level_bootstrap_rho(per_graph_rhos, B=10000, alpha=0.05):
    """Bootstrap CI for the mean of per-graph Spearman rhos."""
    rhos = np.array(per_graph_rhos)
    lo, hi, se = bootstrap_ci(rhos, B=B, alpha=alpha)
    return {
        "mean": round(float(np.mean(rhos)), 4),
        "std": round(float(np.std(rhos)), 4),
        "ci_95_lo": round(lo, 4),
        "ci_95_hi": round(hi, 4),
        "bootstrap_se": round(se, 4),
        "n_graphs": len(rhos),
        "n_bootstrap": B,
    }


def block_bootstrap_aggregate_rho(
    npz_path, n_nodes_per_graph=31635, B=10000, alpha=0.05
):
    """Block bootstrap: resample graphs, use pre-computed per-graph rhos.

    This avoids recomputing Spearman rho on 3M nodes for each replicate.
    Instead, we pre-compute per-graph rho, then bootstrap over graphs.
    """
    data = np.load(npz_path)
    targets = data["targets"].astype(np.float64)
    predictions = data["predictions"].astype(np.float64)
    uncertainties = data["uncertainties"].astype(np.float64)
    abs_errors = np.abs(targets - predictions)

    n_total = len(targets)
    n_graphs = n_total // n_nodes_per_graph
    print(
        f"  Block bootstrap: {n_graphs} graphs x {n_nodes_per_graph} nodes = {n_total}"
    )

    # Pre-compute per-graph rho values
    per_graph_rhos = np.empty(n_graphs)
    for g in range(n_graphs):
        sl = slice(g * n_nodes_per_graph, (g + 1) * n_nodes_per_graph)
        per_graph_rhos[g] = stats.spearmanr(
            uncertainties[sl], abs_errors[sl]
        ).correlation

    print(
        f"  Per-graph rhos computed: mean={np.mean(per_graph_rhos):.4f}, std={np.std(per_graph_rhos):.4f}"
    )

    # Bootstrap: resample graphs, compute mean rho
    boot_means = np.empty(B)
    for b in range(B):
        idx = np.random.randint(0, n_graphs, size=n_graphs)
        boot_means[b] = np.mean(per_graph_rhos[idx])

    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))

    # Also compute the true aggregate rho (on all 3M nodes)
    agg_rho = stats.spearmanr(uncertainties, abs_errors).correlation

    return {
        "aggregate_rho": round(float(agg_rho), 4),
        "mean_of_per_graph_rhos": round(float(np.mean(per_graph_rhos)), 4),
        "ci_95_lo": round(float(lo), 4),
        "ci_95_hi": round(float(hi), 4),
        "bootstrap_se": round(float(np.std(boot_means)), 4),
        "n_bootstrap": B,
        "method": "block_bootstrap_by_graph",
    }


def main():
    results = {}

    # --- Graph-level bootstrap ---
    print("Graph-level bootstrap for per-graph rho...")
    with open(PER_GRAPH) as f:
        pg = json.load(f)

    per_graph_rhos = pg["spearman_rho"]["all_values"]
    print(f"  Loaded {len(per_graph_rhos)} per-graph rho values")
    results["graph_level_mean_rho"] = graph_level_bootstrap_rho(per_graph_rhos, B=B)
    print(
        f"  95% CI for mean rho: [{results['graph_level_mean_rho']['ci_95_lo']}, {results['graph_level_mean_rho']['ci_95_hi']}]"
    )

    # --- Block bootstrap for aggregate rho ---
    print("\nBlock bootstrap for aggregate Spearman rho...")
    results["aggregate_rho_block_bootstrap"] = block_bootstrap_aggregate_rho(
        T8_NPZ, B=B
    )
    print(
        f"  Aggregate rho: {results['aggregate_rho_block_bootstrap']['aggregate_rho']}"
    )
    print(
        f"  95% CI: [{results['aggregate_rho_block_bootstrap']['ci_95_lo']}, {results['aggregate_rho_block_bootstrap']['ci_95_hi']}]"
    )

    # --- Save ---
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
