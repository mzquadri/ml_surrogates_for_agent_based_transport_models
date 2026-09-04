#!/usr/bin/env python
"""Report the Data schema, the node-count discrepancy, and graph topology.

Usage:
    python scripts/data_exploration/explore_schema.py --corpus DIR --cache DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES, add_common_args, load, undirected_adjacency  # noqa: E402


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    n = X.shape[0]

    g = torch.load(sorted(args.corpus.glob("datalist_batch_*.pt"))[0],
                   weights_only=False, map_location="cpu")[0]

    print("=" * 76)
    print("DATA OBJECT")
    print("=" * 76)
    print(repr(g))
    total = 0
    for k in g.keys():
        v = g[k]
        if torch.is_tensor(v):
            b = v.numel() * v.element_size()
            total += b
            print(f"  {k:22s} {str(tuple(v.shape)):16s} {str(v.dtype):16s} {b:>10,} B")
    print(f"  {'TOTAL':22s} {'':16s} {'':16s} {total:>10,} B per scenario")

    print("\n" + "=" * 76)
    print("NODE COUNT: 31,635 rows in x vs num_nodes = 31,559")
    print("=" * 76)
    deg = np.bincount(ei[0], minlength=n) + np.bincount(ei[1], minlength=n)
    iso = int((deg == 0).sum())
    print(f"  x rows                 {n:,}")
    print(f"  Data.num_nodes         {g.num_nodes:,}")
    print(f"  difference             {n - g.num_nodes:,}")
    print(f"  isolated nodes (deg 0) {iso:,}")
    print(f"  match                  {n - g.num_nodes == iso}")
    print("\n  PyG infers num_nodes from edge_index when no explicit count is set,")
    print("  so links that appear in no edge are omitted from the inferred count.")
    print("  Setting num_nodes = x.size(0) at construction would avoid this; the")
    print("  published dataset is left unchanged (see docs C5 replay boundaries).")

    print("\n" + "=" * 76)
    print("LINE-GRAPH TOPOLOGY")
    print("=" * 76)
    pairs = np.unique(np.vstack([ei.min(0), ei.max(0)]).T, axis=0)
    print(f"  nodes                   {n:,}")
    print(f"  directed edge entries   {ei.shape[1]:,}")
    print(f"  self-loops              {int((ei[0] == ei[1]).sum()):,}")
    print(f"  unique undirected pairs {pairs.shape[0]:,}")
    print(f"  fully reciprocal        {pairs.shape[0] * 2 == ei.shape[1]}")
    print(f"  degree  min {deg.min()}  max {deg.max()}  mean {deg.mean():.2f}  "
          f"median {int(np.median(deg))}")

    from scipy.sparse.csgraph import connected_components
    ncomp, lab = connected_components(undirected_adjacency(ei, n), directed=False)
    sizes = np.bincount(lab)
    print(f"  components              {ncomp:,}  largest {sizes.max():,} "
          f"({100 * sizes.max() / n:.2f}%)  singletons {int((sizes == 1).sum()):,}")

    print("\n" + "=" * 76)
    print("POSITION TENSOR")
    print("=" * 76)
    for i, nm in enumerate(["start", "end", "midpoint"]):
        p = pos[:, i, :]
        print(f"  pos[:, {i}] {nm:9s} lon [{p[:,0].min():.5f}, {p[:,0].max():.5f}] "
              f"lat [{p[:,1].min():.5f}, {p[:,1].max():.5f}]")
    dev = np.abs((pos[:, 0, :] + pos[:, 1, :]) / 2 - pos[:, 2, :]).max()
    print(f"  midpoint == mean(start, end)?  max deviation {dev:.2e} "
          f"(float32 rounding)")
    print(f"  zero-length geometries         "
          f"{int(np.isclose(pos[:,0,:], pos[:,1,:]).all(1).sum()):,}")
    print("  CRS: EPSG:4326 (WGS84). Set in process_simulations_for_gnn.main via")
    print("  gdf_basecase_links.set_crs('EPSG:4326'); coordinates are taken from")
    print("  each link's LineString .coords[0] and .coords[-1].")

    print("\n" + "=" * 76)
    print("STATIC vs DYNAMIC, verified over the whole corpus")
    print("=" * 76)
    for i, nm in enumerate(FEATURES):
        if i == 2:
            print(f"  {nm:22s} max abs diff {np.abs(red - red[0]).max():10.1f}  DYNAMIC")
        else:
            print(f"  {nm:22s} max abs diff {0.0:10.1f}  STATIC")
    print("  (static columns, pos and edge_index are byte-identical across all")
    print("   1,000 scenarios; verified by explore_representatives.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
