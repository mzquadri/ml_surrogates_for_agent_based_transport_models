#!/usr/bin/env python
"""Select representative scenarios and links by rule, and check identity.

Every selection is a stated extremum or quantile of a measured quantity. No
scenario or link is chosen because it looks good.

Also verifies the two identity claims the web assets depend on:
  - row order is the permanent link identifier within this corpus
  - the narrative link (row 18785) is uniquely identifiable

Usage:
    python scripts/data_exploration/explore_representatives.py --corpus DIR --cache DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES, add_common_args, load  # noqa: E402

NARRATIVE_LINK = 18785


def selection_rules(red, y):
    """name -> (scenario index, one-line reason). Rules only, no hand-picking."""
    M = red != 0
    touched = M.sum(1)
    cap = np.abs(red).sum(1)
    resp = np.abs(y).sum(1)
    off = np.array([np.abs(y[i][~M[i]]).sum() / np.abs(y[i]).sum()
                    for i in range(len(y))])
    order = np.argsort(touched)
    return {
        "smallest_footprint": (int(np.argmin(touched)),
                               "fewest links with reduced capacity"),
        "largest_footprint": (int(np.argmax(touched)),
                              "most links with reduced capacity"),
        "median_footprint": (int(order[len(order) // 2]),
                             "median number of links intervened"),
        "q25_footprint": (int(order[len(order) // 4]),
                          "25th percentile footprint"),
        "q75_footprint": (int(order[3 * len(order) // 4]),
                          "75th percentile footprint"),
        "highest_offsite_share": (int(np.argmax(off)),
                                  "largest share of response away from the intervention"),
        "lowest_offsite_share": (int(np.argmin(off)),
                                 "smallest share of response away from the intervention"),
        "largest_total_response": (int(np.argmax(resp)),
                                   "largest total absolute response"),
        "smallest_total_response": (int(np.argmin(resp)),
                                    "smallest total absolute response"),
        "most_response_per_capacity": (int(np.argmax(resp / np.maximum(cap, 1))),
                                       "most response per unit capacity removed"),
        "least_response_per_capacity": (int(np.argmin(resp / np.maximum(cap, 1))),
                                        "least response per unit capacity removed"),
        "most_capacity_removed": (int(np.argmax(cap)),
                                  "largest total capacity reduction"),
    }


def link_rules(red, y, X):
    M = red != 0
    per_link = M.sum(0)
    absY = np.abs(y).mean(0)
    return {
        "most_often_intervened": (int(np.argmax(per_link)),
                                  "intervened in the most scenarios"),
        "most_volatile_response": (int(np.argmax(y.std(0))),
                                   "largest standard deviation of response"),
        "busiest_base_volume": (int(np.argmax(X[:, 0])),
                                "highest base-case car volume"),
        "most_reactive_never_intervened": (
            int(np.argmax(np.where(per_link == 0, absY, -1))),
            "largest mean response among links never intervened"),
        "longest_link": (int(np.argmax(X[:, 5])), "greatest length"),
        "highest_capacity": (int(np.argmax(X[:, 1])), "highest base capacity"),
    }


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    M = red != 0

    print("=" * 76)
    print("IDENTITY: is row order the permanent link identifier?")
    print("=" * 76)
    files = sorted(args.corpus.glob("datalist_batch_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    ok_pos = ok_ei = ok_static = True
    static_cols = [0, 1, 3, 4, 5]
    n_seen = 0
    for f in files:
        for g in torch.load(f, weights_only=False, map_location="cpu"):
            ok_pos &= np.array_equal(g.pos.numpy(), pos)
            ok_ei &= np.array_equal(g.edge_index.numpy(), ei)
            ok_static &= np.array_equal(g.x.numpy()[:, static_cols], X[:, static_cols])
            n_seen += 1
    print(f"  scenarios checked                  {n_seen:,}")
    print(f"  pos byte-identical                 {ok_pos}")
    print(f"  edge_index byte-identical          {ok_ei}")
    print(f"  static feature columns identical   {ok_static}")
    se = np.concatenate([pos[:, 0, :], pos[:, 1, :]], axis=1)
    print(f"  unique (start,end) geometries      {len(np.unique(se, axis=0)):,} "
          f"of {X.shape[0]:,}")
    print("  => geometry alone is NOT unique, so row index is the identifier.")
    print("     No MATSim/OSM link id is present in the published tensors; the")
    print("     base-network file that carried it is not part of the release.")

    print("\n" + "=" * 76)
    print(f"NARRATIVE LINK row {NARRATIVE_LINK}")
    print("=" * 76)
    i = NARRATIVE_LINK
    print(f"  highway {int(X[i,4])}  volume {X[i,0]:.0f}  capacity {X[i,1]:.0f}  "
          f"freespeed {X[i,3]:.2f} m/s  length {X[i,5]:.1f} m")
    print(f"  start ({pos[i,0,0]:.6f}, {pos[i,0,1]:.6f})  "
          f"end ({pos[i,1,0]:.6f}, {pos[i,1,1]:.6f})")
    print(f"  is the unique maximum base volume  "
          f"{X[i,0] == X[:,0].max() and int((X[:,0] == X[:,0].max()).sum()) == 1}")
    print(f"  rows sharing its geometry          "
          f"{np.where((se == se[i]).all(1))[0].tolist()}")
    print(f"  ever intervened                    {bool(M[:, i].any())}")
    print(f"  mean |y| {np.abs(y[:, i]).mean():.2f}   std(y) {y[:, i].std():.2f}")

    print("\n" + "=" * 76)
    print("REPRESENTATIVE SCENARIOS")
    print("=" * 76)
    touched = M.sum(1); cap = np.abs(red).sum(1); resp = np.abs(y).sum(1)
    off = np.array([np.abs(y[k][~M[k]]).sum() / np.abs(y[k]).sum()
                    for k in range(len(y))])
    print(f"  {'rule':32s} {'idx':>5} {'links':>7} {'capRemoved':>12} "
          f"{'total|y|':>10} {'offsite':>8}")
    for name, (idx, _) in selection_rules(red, y).items():
        print(f"  {name:32s} {idx:5d} {touched[idx]:7,} {cap[idx]:12,.0f} "
              f"{resp[idx]:10,.0f} {100*off[idx]:7.1f}%")

    print("\n" + "=" * 76)
    print("REPRESENTATIVE LINKS")
    print("=" * 76)
    per_link = M.sum(0); absY = np.abs(y).mean(0)
    print(f"  {'rule':34s} {'row':>6} {'hw':>3} {'timesInt':>9} "
          f"{'mean|y|':>8} {'std(y)':>8}")
    for name, (idx, _) in link_rules(red, y, X).items():
        print(f"  {name:34s} {idx:6d} {int(X[idx,4]):3d} {per_link[idx]:9,} "
              f"{absY[idx]:8.2f} {y[:,idx].std():8.2f}")

    print("\n" + "=" * 76)
    print("FEATURE -> RESPONSE (Spearman on per-link mean |y|)")
    print("=" * 76)
    for i2 in [0, 1, 3, 4, 5]:
        print(f"  {FEATURES[i2]:22s} {spearmanr(X[:, i2], absY).statistic:+.3f}")
    car = X[:, 1] > 0
    print(f"  restricted to car-capable links ({int(car.sum()):,}):")
    for i2 in [0, 1, 3, 5]:
        print(f"    {FEATURES[i2]:20s} {spearmanr(X[car, i2], absY[car]).statistic:+.3f}")

    print("\n" + "=" * 76)
    print("ANOMALIES")
    print("=" * 76)
    deg = np.bincount(ei[0], minlength=X.shape[0]) + np.bincount(ei[1], minlength=X.shape[0])
    never = np.abs(y).max(0) == 0
    inert = (X[:, 0] == 0) & (X[:, 1] == 0)
    print(f"  target exactly 0 in all scenarios   {int(never.sum()):,} "
          f"({100*never.mean():.2f}%)")
    print(f"  of those, car-capable               {int((never & ~inert).sum()):,}")
    print(f"  zero volume and zero capacity       {int(inert.sum()):,}")
    print(f"  zero-length geometries              "
          f"{int(np.isclose(pos[:,0,:], pos[:,1,:]).all(1).sum()):,}")
    print(f"  isolated nodes                      {int((deg == 0).sum()):,}")
    print(f"  self-loops                          {int((ei[0] == ei[1]).sum()):,}")
    print(f"  duplicated feature rows             "
          f"{X.shape[0] - len(np.unique(X, axis=0)):,}")
    print(f"  NaN / inf in x                      "
          f"{int(np.isnan(X).sum())} / {int(np.isinf(X).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
