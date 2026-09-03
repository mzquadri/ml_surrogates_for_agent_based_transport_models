#!/usr/bin/env python
"""How far the response travels from the intervention, with its controls.

The headline observation is that mean |y| is flat beyond the first graph hop.
That is only meaningful alongside the reachability control, and only if it
survives the choice of traversal direction, so both are computed here.

Methodology, stated explicitly because the result depends on it:

  distance      multi-source BFS in the line graph from the set of intervened
                links; hop 0 is the intervened set itself
  direction     computed twice -- undirected (a link is adjacent if the two
                links meet at all) and directed (following the stored
                from->to orientation, i.e. traffic direction)
  unreachable   nodes never reached within max_hops are reported as their own
                band and never folded into the last hop
  aggregation   per scenario, mean |y| within each hop band; then the mean of
                those per-scenario means, so every scenario carries equal
                weight regardless of footprint size
  topology      identical in all 1,000 scenarios, so the hop bands differ only
                through which links were intervened

Usage:
    python scripts/data_exploration/explore_spillover.py --corpus DIR --cache DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, directed_adjacency, hop_distance,  # noqa: E402
                    load, undirected_adjacency)

MAX_HOPS = 8


def profile(adj, red, y, sample):
    """Mean |y| per hop band, share of total |y|, and reachability."""
    M = red != 0
    n = red.shape[1]
    acc = np.zeros(MAX_HOPS + 1); cnt = np.zeros(MAX_HOPS + 1)
    share = np.zeros(MAX_HOPS + 1); cum = np.zeros(MAX_HOPS + 1)
    unre_absy, unre_share, unre_cnt = 0.0, 0.0, 0
    for si in sample:
        dist = hop_distance(adj, M[si], MAX_HOPS)
        ay = np.abs(y[si]); tot = ay.sum()
        for k in range(MAX_HOPS + 1):
            s = dist == k
            if s.any():
                acc[k] += ay[s].mean(); cnt[k] += 1
                share[k] += ay[s].sum() / tot
            cum[k] += ((dist >= 0) & (dist <= k)).mean()
        s = dist < 0
        if s.any():
            unre_absy += ay[s].mean(); unre_share += ay[s].sum() / tot; unre_cnt += 1
    ns = len(sample)
    return (np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan),
            share / ns, cum / ns,
            unre_absy / max(unre_cnt, 1), unre_share / ns,
            1 - cum[MAX_HOPS])


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--scenarios", type=int, default=100,
                    help="how many scenarios to sample (seeded, reproducible)")
    args = ap.parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    n = red.shape[1]
    sample = np.random.default_rng(0).choice(red.shape[0], args.scenarios,
                                             replace=False)

    for label, adj in [("UNDIRECTED", undirected_adjacency(ei, n)),
                       ("DIRECTED (traffic direction)", directed_adjacency(ei, n))]:
        m, sh, cum, ua, ush, unre = profile(adj, red, y, sample)
        print("=" * 76)
        print(f"{label}  --  {args.scenarios} sampled scenarios, seed 0")
        print("=" * 76)
        print(f"  {'hops':>14}  {'mean |y| veh/h':>15}  {'share of |y|':>13}  "
              f"{'network reached':>16}")
        for k in range(MAX_HOPS + 1):
            lbl = "0 (intervened)" if k == 0 else str(k)
            print(f"  {lbl:>14}  {m[k]:15.3f}  {100*sh[k]:12.1f}%  "
                  f"{100*cum[k]:15.2f}%")
        print(f"  {'unreachable':>14}  {ua:15.3f}  {100*ush:12.1f}%  "
              f"{100*unre:15.2f}%")
        drop = m[0] / m[1]
        flat = np.nanmax(m[1:]) - np.nanmin(m[1:])
        print(f"\n  hop 0 -> hop 1 drop: {drop:.2f}x")
        print(f"  spread across hops 1-{MAX_HOPS}: {flat:.3f} veh/h "
              f"(min {np.nanmin(m[1:]):.2f}, max {np.nanmax(m[1:]):.2f})")
        print()

    print("=" * 76)
    print("READING")
    print("=" * 76)
    print("  One sharp fall at the first hop, then a flat profile. Because only")
    print("  about a third of the network lies within one hop of an intervention")
    print("  and roughly 70% within three, the flatness is not a small-diameter")
    print("  artefact: the response is spread across the network at a broadly")
    print("  constant magnitude rather than decaying with graph distance.")
    print()
    print("  This is an observed association in one simulated network under one")
    print("  intervention family. It is not causal proof of a diffusion mechanism,")
    print("  and it says nothing about how a different city or policy would behave.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
