#!/usr/bin/env python
"""Topology of the line graph: what `edge_index` actually encodes.

The road network has intersections as nodes and road links as edges. This corpus
stores the line graph of that network, where each road link becomes a node and an
edge means two links meet at an intersection. The consequence worth stating is
that the model predicts one value per road link, which is what a policy question
asks about.

    python scripts/data_exploration/explore_graph.py --corpus DIR --cache DIR

Writes graph_topology.json to the web-asset directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from common import add_common_args, load  # noqa: E402

OUT = REPO / "docs" / "portfolio_data_story" / "assets"


def main() -> int:
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    n = X.shape[0]
    e = ei.shape[1]
    src, dst = ei[0], ei[1]

    self_loops = int((src == dst).sum())
    pairs = set(zip(src.tolist(), dst.tolist(), strict=True))
    recip = sum(1 for a, b in pairs if a != b and (b, a) in pairs) // 2
    outdeg = np.bincount(src, minlength=n)
    indeg = np.bincount(dst, minlength=n)
    deg = outdeg + indeg

    adj = coo_matrix((np.ones(e, np.int8), (src, dst)), shape=(n, n))
    n_weak, lab = connected_components(adj, directed=True, connection="weak")
    n_strong, _ = connected_components(adj, directed=True, connection="strong")
    sizes = np.bincount(lab)

    absy = np.abs(y).mean(0)
    topo = {
        "representation": "line graph of the road network",
        "nodes": int(n),
        "nodes_are": "road links",
        "edges": int(e),
        "edges_mean": "the two road links meet at an intersection",
        "directed": True,
        "unique_src_dst_pairs": len(pairs),
        "duplicate_edges": int(e - len(pairs)),
        "self_loops": self_loops,
        "self_loop_pct": round(100 * self_loops / e, 4),
        "bidirectional_pairs": recip,
        "reciprocity_pct_of_non_loop_edges": round(
            100 * 2 * recip / max(e - self_loops, 1), 2),
        "isolated_nodes": int((deg == 0).sum()),
        "degree": {
            "min": int(deg.min()), "median": int(np.median(deg)),
            "mean": round(float(deg.mean()), 4), "max": int(deg.max()),
            "max_in": int(indeg.max()), "max_out": int(outdeg.max()),
            "histogram": [{"degree": int(d), "n_links": int((deg == d).sum())}
                          for d in range(int(deg.max()) + 1) if (deg == d).any()],
        },
        "components": {
            "weakly_connected": int(n_weak),
            "strongly_connected": int(n_strong),
            "largest_weak_size": int(sizes.max()),
            "largest_weak_pct": round(100 * float(sizes.max()) / n, 2),
            "largest_10_sizes": sorted(sizes.tolist(), reverse=True)[:10],
        },
        "response_by_degree": [
            {"degree": int(d), "n_links": int((deg == d).sum()),
             "mean_abs_response": round(float(absy[deg == d].mean()), 6)}
            for d in range(int(deg.max()) + 1) if (deg == d).sum() >= 5
        ],
        "note": ("num_nodes is stored as 31,559 because the preprocessing set it from "
                 "the road-network edge count before the line-graph transform and never "
                 "updated it. x, pos and y all carry 31,635 rows; the 76 extra rows are "
                 "public-transport links that appear in no edge."),
    }
    p = OUT / "graph_topology.json"
    p.write_text(json.dumps(topo, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"  graph_topology.json  {p.stat().st_size/1024:.1f} KB")
    print(f"  {n:,} nodes, {e:,} directed edges, {n_weak} weak components, "
          f"largest {sizes.max():,} ({100*sizes.max()/n:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
