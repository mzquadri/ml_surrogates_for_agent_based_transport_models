#!/usr/bin/env python
"""Join the road links to the 20 Paris arrondissements and summarise by district.

Why this join matters: the preprocessing names every scenario after the
arrondissements its policy touches --

    create_policy_key() -> "Policy introduced in Arrondissement(s) 5, 12"

-- so the arrondissement is the unit the experiment was designed around. The
published tensors keep the resulting capacity reductions but not the district
labels, and this script recovers the geographic frame by locating each link's
midpoint inside the district polygons that ship with the repository.

The polygons are administrative boundaries, not the road network. The road
geometry comes only from `pos`.

Usage:
    python scripts/data_exploration/explore_arrondissements.py --corpus DIR --cache DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import add_common_args, load  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
DISTRICTS = REPO / "data" / "visualisation" / "districts_paris.geojson"


def assign_districts(mid_lon: np.ndarray, mid_lat: np.ndarray):
    """Point-in-polygon of every link midpoint against the 20 arrondissements.

    Returns (codes, polygons) where codes[i] is the arrondissement number of
    link i, or 0 for links outside every polygon -- the network extends beyond
    the city boundary into Ile-de-France, so that group is expected and large.
    """
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree

    gj = json.loads(DISTRICTS.read_text(encoding="utf-8"))
    polys, codes = [], []
    for f in gj["features"]:
        polys.append(shape(f["geometry"]))
        codes.append(int(f["properties"]["c_ar"]))
    tree = STRtree(polys)
    out = np.zeros(mid_lon.size, dtype=np.int16)
    for i in range(mid_lon.size):
        p = Point(float(mid_lon[i]), float(mid_lat[i]))
        for j in tree.query(p):
            if polys[j].contains(p):
                out[i] = codes[j]
                break
    return out, polys, codes


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    mid = pos[:, 2, :]
    M = red != 0

    print("Assigning 31,635 link midpoints to arrondissements ...")
    ar, polys, codes = assign_districts(mid[:, 0], mid[:, 1])

    inside = ar > 0
    print(f"  inside the 20 arrondissements : {inside.sum():,} "
          f"({100*inside.mean():.1f}%)")
    print(f"  outside (wider Ile-de-France) : {(~inside).sum():,} "
          f"({100*(~inside).mean():.1f}%)")

    absY = np.abs(y).mean(0)
    per_link_touch = M.sum(0)
    sev = np.where(M, np.abs(red), np.nan)
    with np.errstate(invalid="ignore"):
        mean_sev = np.nanmean(sev, axis=0)
    mean_sev = np.nan_to_num(mean_sev)

    print(f"\n{'arr':>4}{'links':>8}{'%intervened':>13}{'meanSeverity':>14}"
          f"{'mean|y|':>10}{'meanVol':>10}")
    rows = []
    for c in [0] + sorted(codes):
        s = ar == c
        if not s.any():
            continue
        ever = (per_link_touch[s] > 0).mean()
        row = {
            "arrondissement": int(c),
            "links": int(s.sum()),
            "share_links_ever_intervened": round(float(ever), 5),
            "mean_intervention_severity_vehh": round(float(mean_sev[s].mean()), 3),
            "mean_abs_response_vehh": round(float(absY[s].mean()), 4),
            "mean_vol_base_case": round(float(X[s, 0].mean()), 2),
            "mean_capacity_base_case": round(float(X[s, 1].mean()), 2),
            "total_intervention_events": int(per_link_touch[s].sum()),
        }
        rows.append(row)
        label = "out" if c == 0 else str(c)
        print(f"{label:>4}{s.sum():8,}{100*ever:12.1f}%{mean_sev[s].mean():14.1f}"
              f"{absY[s].mean():10.3f}{X[s,0].mean():10.1f}")

    print("\n=== response concentration ===")
    tot = absY.sum()
    for c in sorted(codes):
        s = ar == c
        if s.any() and absY[s].sum() / tot > 0.04:
            print(f"  arrondissement {c:2d}: {100*absY[s].sum()/tot:5.2f}% of total "
                  f"mean|response| across {100*s.mean():.1f}% of links")

    out = REPO / "docs" / "portfolio_data_story" / "assets" / "arrondissements.json"
    out.write_text(json.dumps({
        "source": "data/visualisation/districts_paris.geojson (20 polygons, CRS84)",
        "join": "point-in-polygon of each link midpoint (pos[:,2]) against the "
                "arrondissement polygons",
        "note": "arrondissement 0 means the link midpoint lies outside all twenty "
                "polygons; the modelled network extends beyond the city boundary",
        "caveat": "the polygons are administrative boundaries, not the road network",
        "links_inside": int(inside.sum()),
        "links_outside": int((~inside).sum()),
        "arrondissements": rows,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out.relative_to(REPO)}")

    np.save(args.cache / "arrondissement_of_link.npy", ar)
    print(f"cached per-link arrondissement codes to {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
