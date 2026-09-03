"""Shared loading and caching for the dataset exploration scripts.

The published corpus is 20 `.pt` files totalling 2.44 GiB. Reading all of it takes
a couple of minutes, so the first script to run materialises the few arrays every
other script needs and caches them as `.npy` under `--cache`. Nothing here writes
into the repository; the cache is a scratch directory the caller chooses.

The corpus itself is never copied into the repository. See
docs/portfolio_data_story/README.md for how to obtain it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: Column order of `Data.x`, from EdgeFeatures in
#: scripts/data_preprocessing/process_simulations_for_gnn.py
FEATURES = [
    "VOL_BASE_CASE",
    "CAPACITY_BASE_CASE",
    "CAPACITY_REDUCTION",
    "FREESPEED",
    "HIGHWAY",
    "LENGTH",
]
#: The one column that varies between scenarios.
DYNAMIC_COL = 2
#: Columns the model consumes (HIGHWAY is excluded: ordinal encoding of a
#: nominal category).
MODEL_COLS = [0, 1, 2, 3, 5]

#: OSM class -> integer, from highway_mapping in
#: scripts/data_preprocessing/help_functions.py. -1 is both the explicit code
#: for "pt" and the fallback for any value absent from the mapping.
HIGHWAY_CLASSES = {
    -1: "pt (public transport) or unmapped",
    0: "trunk / trunk_link / motorway_link",
    1: "primary / primary_link",
    2: "secondary / secondary_link",
    3: "tertiary / tertiary_link",
    4: "residential",
    5: "living_street",
    6: "pedestrian",
    7: "service",
    8: "construction",
    9: "unclassified",
}

NODES_PER_GRAPH = 31_635
N_SCENARIOS = 1_000


def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--corpus", type=Path, required=True,
                    help="directory holding datalist_batch_1.pt ... _20.pt")
    ap.add_argument("--cache", type=Path, required=True,
                    help="scratch directory for cached .npy arrays")
    return ap


def _batch_files(corpus: Path) -> list[Path]:
    files = sorted(corpus.glob("datalist_batch_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise SystemExit(f"no datalist_batch_*.pt under {corpus}")
    return files


def build_cache(corpus: Path, cache: Path) -> None:
    """Stream the corpus once and cache the arrays every script needs.

    Cached:
      red.npy    [S, N] float32  CAPACITY_REDUCTION, the only dynamic column
      y.npy      [S, N] float32  target
      static.npy [N, 6] float64  x from scenario 0 (proven identical elsewhere)
      pos.npy    [N, 3, 2] float32
      edge_index.npy [2, E] int64
    """
    import torch

    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "red.npy").exists():
        return

    red, y, static, pos, ei = [], [], None, None, None
    for f in _batch_files(corpus):
        for g in torch.load(f, weights_only=False, map_location="cpu"):
            red.append(g.x.numpy()[:, DYNAMIC_COL].astype(np.float32))
            y.append(g.y.numpy().ravel().astype(np.float32))
            if static is None:
                static = g.x.numpy().copy()
                pos = g.pos.numpy().copy()
                ei = g.edge_index.numpy().copy()
    np.save(cache / "red.npy", np.stack(red))
    np.save(cache / "y.npy", np.stack(y))
    np.save(cache / "static.npy", static)
    np.save(cache / "pos.npy", pos)
    np.save(cache / "edge_index.npy", ei)


def load(corpus: Path, cache: Path):
    """Return (red, y, static, pos, edge_index), building the cache if needed."""
    build_cache(corpus, cache)
    return (
        np.load(cache / "red.npy"),
        np.load(cache / "y.npy"),
        np.load(cache / "static.npy"),
        np.load(cache / "pos.npy"),
        np.load(cache / "edge_index.npy"),
    )


def undirected_adjacency(edge_index: np.ndarray, n: int):
    """Symmetric boolean adjacency of the line graph."""
    from scipy.sparse import coo_matrix

    a = coo_matrix((np.ones(edge_index.shape[1], np.int8),
                    (edge_index[0], edge_index[1])), shape=(n, n)).tocsr()
    return ((a + a.T) > 0).astype(np.int8)


def directed_adjacency(edge_index: np.ndarray, n: int):
    """Adjacency following the stored edge direction (traffic direction)."""
    from scipy.sparse import coo_matrix

    a = coo_matrix((np.ones(edge_index.shape[1], np.int8),
                    (edge_index[0], edge_index[1])), shape=(n, n)).tocsr()
    return (a > 0).astype(np.int8)


def hop_distance(adj, seed_mask: np.ndarray, max_hops: int = 8) -> np.ndarray:
    """Multi-source BFS. Returns hop count per node, -1 where unreachable."""
    n = seed_mask.size
    dist = np.full(n, -1, np.int16)
    dist[seed_mask] = 0
    frontier = seed_mask.copy()
    for k in range(1, max_hops + 1):
        nxt = ((adj @ frontier.astype(np.int8)) > 0) & (dist < 0)
        if not nxt.any():
            break
        dist[nxt] = k
        frontier = nxt
    return dist
