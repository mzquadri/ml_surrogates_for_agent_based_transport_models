#!/usr/bin/env python
"""Report the intervention design: footprints, magnitudes, road classes targeted.

Usage:
    python scripts/data_exploration/explore_intervention.py --corpus DIR --cache DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HIGHWAY_CLASSES, add_common_args, load  # noqa: E402


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    M = red != 0
    n_links = red.shape[1]
    touched = M.sum(1)

    print("=" * 76)
    print("FOOTPRINT PER SCENARIO")
    print("=" * 76)
    print(f"  scenarios              {red.shape[0]:,}")
    print(f"  links touched   min {touched.min():,}  median "
          f"{int(np.median(touched)):,}  mean {touched.mean():,.0f}  "
          f"max {touched.max():,}")
    print(f"  as share of network    {100*touched.min()/n_links:.2f}% .. "
          f"{100*touched.max()/n_links:.2f}%  (mean {100*touched.mean()/n_links:.2f}%)")
    print(f"  unique intervention vectors {len({r.tobytes() for r in red}):,}")
    print(f"  unique intervention masks   {len({m.tobytes() for m in M}):,}")

    print("\n" + "=" * 76)
    print("PER-LINK EXPOSURE")
    print("=" * 76)
    per_link = M.sum(0)
    print(f"  links never intervened  {int((per_link == 0).sum()):,} "
          f"({100*(per_link == 0).mean():.2f}%)")
    print(f"  links ever intervened   {int((per_link > 0).sum()):,}")
    print(f"  of those: mean {per_link[per_link>0].mean():.1f} scenarios, "
          f"max {per_link.max():,}")

    print("\n" + "=" * 76)
    print("MAGNITUDES")
    print("=" * 76)
    nz = red[M]
    vals, cnt = np.unique(nz, return_counts=True)
    print(f"  distinct non-zero values {len(vals)}   all negative: {bool((nz<0).all())}")
    for v, c in sorted(zip(vals.tolist(), cnt.tolist(), strict=True),
                       key=lambda t: -t[1])[:10]:
        print(f"    {v:9.0f} veh/h  {c:10,}  ({100*c/nz.size:5.2f}%)")

    print("\n" + "=" * 76)
    print("WHICH ROAD CLASSES ARE TARGETED?")
    print("=" * 76)
    hw = X[:, 4].astype(int)
    ever = per_link > 0
    absY = np.abs(y).mean(0)
    print(f"  {'code':>4}  {'class':36s} {'links':>7} {'%ever':>7} {'mean|y|':>8}")
    for c in sorted(set(hw.tolist())):
        s = hw == c
        print(f"  {c:>4}  {HIGHWAY_CLASSES.get(c,'?'):36s} {int(s.sum()):7,} "
              f"{100*ever[s].mean():6.1f}% {absY[s].mean():8.3f}")
    tgt = sorted({int(c) for c in hw[ever]})
    print(f"\n  classes ever intervened: {tgt} "
          f"({', '.join(HIGHWAY_CLASSES[c].split(' /')[0] for c in tgt)})")
    print("  every other class is untouched in all 1,000 scenarios.")

    print("\n" + "=" * 76)
    print("RESPONSE vs INTERVENTION")
    print("=" * 76)
    on = np.array([np.abs(y[i][M[i]]).mean() for i in range(len(y))])
    off = np.array([np.abs(y[i][~M[i]]).mean() for i in range(len(y))])
    share = np.array([np.abs(y[i][~M[i]]).sum() / np.abs(y[i]).sum()
                      for i in range(len(y))])
    print(f"  mean |y| on intervened links   {on.mean():.3f} veh/h")
    print(f"  mean |y| on untouched links    {off.mean():.3f} veh/h  "
          f"({on.mean()/off.mean():.2f}x lower)")
    print(f"  share of total |y| off-site    {100*share.mean():.1f}% "
          f"(min {100*share.min():.1f}%, max {100*share.max():.1f}%)")
    print(f"  corr(links touched, total |y|)     "
          f"{np.corrcoef(touched, np.abs(y).sum(1))[0,1]:+.3f}")
    print(f"  corr(capacity removed, total |y|)  "
          f"{np.corrcoef(np.abs(red).sum(1), np.abs(y).sum(1))[0,1]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
