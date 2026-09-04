#!/usr/bin/env python
"""Build the lightweight, derived web assets for the data story.

Everything written here is an aggregate. The 31,635,000-row observation table is
never exported, and the .pt corpus is never copied into the repository.

Deterministic: all sampling is seeded, all row orders are sorted or fixed, floats
are rounded before serialisation, and every file is written with LF endings
regardless of platform, so re-running produces byte-identical files.

Usage:
    python scripts/data_exploration/build_web_assets.py --corpus DIR --cache DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURES,
    HIGHWAY_CLASSES,
    add_common_args,
    directed_adjacency,
    hop_distance,
    load,
    undirected_adjacency,
)
from explore_representatives import NARRATIVE_LINK, link_rules, selection_rules  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "docs" / "portfolio_data_story" / "assets"
R = 5  # coordinate rounding: ~1 m at this latitude, ample for a map


def jdump(obj, path):
    # newline="\n" is required: write_text() would otherwise translate to CRLF on
    # Windows, so a rebuild would not match the committed file byte for byte even
    # though the content is identical. links.csv already writes LF explicitly.
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"  {path.name:28s} {path.stat().st_size/1024:8.1f} KB")


def binned(xv, yv, edges):
    """Median and IQR of yv within bins of xv. Shape is what a chart needs."""
    idx = np.clip(np.digitize(xv, edges) - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        s = idx == b
        if s.sum() < 20:
            continue
        out.append({
            "bin_lo": round(float(edges[b]), 4),
            "bin_hi": round(float(edges[b + 1]), 4),
            "n_links": int(s.sum()),
            "median_abs_y": round(float(np.median(yv[s])), 4),
            "p25_abs_y": round(float(np.percentile(yv[s], 25)), 4),
            "p75_abs_y": round(float(np.percentile(yv[s], 75)), 4),
            "mean_abs_y": round(float(yv[s].mean()), 4),
        })
    return out


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    OUT.mkdir(parents=True, exist_ok=True)
    M = red != 0
    n = X.shape[0]
    per_link = M.sum(0)
    absY = np.abs(y).mean(0)
    stdY = y.std(0)
    deg = np.bincount(ei[0], minlength=n) + np.bincount(ei[1], minlength=n)
    hw = X[:, 4].astype(int)

    print("writing assets to", OUT)

    # ---- links.csv -----------------------------------------------------------
    p = OUT / "links.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["link_row", "start_lon", "start_lat", "end_lon", "end_lat",
                    "mid_lon", "mid_lat", "highway_code", "vol_base_case",
                    "capacity_base_case", "freespeed_ms", "length_m", "degree",
                    "times_intervened", "mean_abs_response", "std_response"])
        for i in range(n):
            w.writerow([
                i,
                round(float(pos[i, 0, 0]), R), round(float(pos[i, 0, 1]), R),
                round(float(pos[i, 1, 0]), R), round(float(pos[i, 1, 1]), R),
                round(float(pos[i, 2, 0]), R), round(float(pos[i, 2, 1]), R),
                int(hw[i]), round(float(X[i, 0]), 1), round(float(X[i, 1]), 1),
                round(float(X[i, 3]), 2), round(float(X[i, 5]), 1),
                int(deg[i]), int(per_link[i]),
                round(float(absY[i]), 3), round(float(stdY[i]), 3),
            ])
    print(f"  {p.name:28s} {p.stat().st_size/1024:8.1f} KB")

    # ---- scenarios.json ------------------------------------------------------
    touched = M.sum(1)
    cap = np.abs(red).sum(1)
    resp = np.abs(y).sum(1)
    off = np.array([np.abs(y[i][~M[i]]).sum() / np.abs(y[i]).sum() for i in range(len(y))])
    jdump({
        "n_scenarios": int(red.shape[0]),
        "n_links": int(n),
        "fields": ["scenario_id", "links_intervened", "capacity_removed_vehh",
                   "total_abs_response_vehh", "offsite_response_share",
                   "mean_abs_response_vehh"],
        "scenarios": [[int(i), int(touched[i]), round(float(cap[i]), 1),
                       round(float(resp[i]), 1), round(float(off[i]), 5),
                       round(float(np.abs(y[i]).mean()), 4)]
                      for i in range(red.shape[0])],
    }, OUT / "scenarios.json")

    # ---- representative scenarios -------------------------------------------
    rules = selection_rules(red, y)
    magnitudes = sorted({float(v) for v in np.unique(red[M])})
    mag_ix = {v: k for k, v in enumerate(magnitudes)}
    index, written = [], {}
    for name, (idx, reason) in rules.items():
        index.append({"scenario_id": idx, "selection_rule": name,
                      "selection_reason": reason,
                      "links_intervened": int(touched[idx]),
                      "offsite_response_share": round(float(off[idx]), 5),
                      "asset": f"scenario_{idx}.json"})
        written.setdefault(idx, []).append((name, reason))
    for idx, rls in written.items():
        links = np.where(M[idx])[0]
        yy = y[idx]
        jdump({
            "scenario_id": int(idx),
            "selection_rules": [{"rule": a, "reason": b} for a, b in rls],
            "links_intervened": int(touched[idx]),
            "capacity_removed_vehh": round(float(cap[idx]), 1),
            "total_abs_response_vehh": round(float(resp[idx]), 1),
            "offsite_response_share": round(float(off[idx]), 5),
            "intervened_link_rows": [int(v) for v in links],
            "reduction_magnitudes_vehh": magnitudes,
            "intervened_reduction_index": [mag_ix[float(v)] for v in red[idx][links]],
            "response_percentiles_vehh": {
                q: round(float(np.percentile(yy, int(q))), 3)
                for q in ["1", "5", "25", "50", "75", "95", "99"]},
            "top_50_responding_link_rows": [int(v) for v in
                                            np.argsort(-np.abs(yy))[:50]],
            "top_50_responding_values_vehh": [round(float(v), 3) for v in
                                              yy[np.argsort(-np.abs(yy))[:50]]],
        }, OUT / f"scenario_{idx}.json")
    jdump({"n_representatives": len(index), "selection": "rule-based extrema and "
           "quantiles of measured quantities; no manual choice", "items": index},
          OUT / "representative_scenarios.json")

    # ---- spillover_decay.json ------------------------------------------------
    sample = np.random.default_rng(0).choice(red.shape[0], 100, replace=False)
    out_dec = {}
    for label, adj in [("undirected", undirected_adjacency(ei, n)),
                       ("directed", directed_adjacency(ei, n))]:
        acc = np.zeros(9); cnt = np.zeros(9); sh = np.zeros(9); cum = np.zeros(9)
        ua = us = 0.0; uc = 0
        for si in sample:
            dist = hop_distance(adj, M[si], 8)
            ay = np.abs(y[si]); tot = ay.sum()
            for k in range(9):
                s = dist == k
                if s.any():
                    acc[k] += ay[s].mean(); cnt[k] += 1; sh[k] += ay[s].sum() / tot
                cum[k] += ((dist >= 0) & (dist <= k)).mean()
            s = dist < 0
            if s.any():
                ua += ay[s].mean(); us += ay[s].sum() / tot; uc += 1
        out_dec[label] = {
            "hops": list(range(9)),
            "mean_abs_response_vehh": [round(float(acc[k] / cnt[k]), 4)
                                       if cnt[k] else None for k in range(9)],
            "share_of_total_abs_response": [round(float(sh[k] / len(sample)), 5)
                                            for k in range(9)],
            "cumulative_network_reached": [round(float(cum[k] / len(sample)), 5)
                                           for k in range(9)],
            "unreachable_mean_abs_response_vehh": round(float(ua / max(uc, 1)), 4),
            "unreachable_share_of_total": round(float(us / len(sample)), 5),
        }
    out_dec["method"] = {
        "distance": "multi-source BFS from the intervened link set; hop 0 is that set",
        "aggregation": "per-scenario mean within each hop band, then unweighted "
                       "mean across scenarios",
        "sample": {"n_scenarios": int(len(sample)), "seed": 0},
        "topology": "identical in all 1,000 scenarios; only the intervened set varies",
        "caveat": "observed association in one simulated network under one "
                  "intervention family; not causal evidence of a mechanism",
    }
    jdump(out_dec, OUT / "spillover_decay.json")

    # ---- highway_classes.json ------------------------------------------------
    ever = per_link > 0
    jdump({"source": "highway_mapping in scripts/data_preprocessing/help_functions.py",
           "note": "-1 is both the explicit code for 'pt' and the fallback for any "
                   "OSM value absent from the mapping",
           "classes": [{
               "code": int(c), "osm_classes": HIGHWAY_CLASSES[int(c)],
               "n_links": int((hw == c).sum()),
               "share_of_network": round(float((hw == c).mean()), 5),
               "share_ever_intervened": round(float(ever[hw == c].mean()), 5),
               "mean_abs_response_vehh": round(float(absY[hw == c].mean()), 4),
               "mean_vol_base_case": round(float(X[hw == c, 0].mean()), 2),
               "mean_capacity_base_case": round(float(X[hw == c, 1].mean()), 2),
           } for c in sorted(set(hw.tolist()))]}, OUT / "highway_classes.json")

    # ---- feature_summary.json (with binned conditional shape) ----------------
    feats = []
    car = X[:, 1] > 0
    for i, nm in enumerate(FEATURES):
        col = X[:, i]
        if i == 2:
            col = red.ravel()
        entry = {
            "index": i, "name": nm,
            "dynamic": i == 2,
            "used_by_model": i != 4,
            "min": round(float(col.min()), 4), "max": round(float(col.max()), 4),
            "mean": round(float(col.mean()), 4), "std": round(float(col.std()), 4),
            "share_zero": round(float((col == 0).mean()), 5),
            "n_distinct": int(len(np.unique(col))),
        }
        if i == 4:
            # HIGHWAY is a nominal category with an ordinal encoding. Binning it
            # numerically, or correlating against it, would treat road-class codes
            # as an ordered quantity, which they are not. Per-class summaries are
            # the only honest presentation; see highway_classes.json.
            entry["per_class_vs_abs_response"] = [
                {"code": int(c), "osm_classes": HIGHWAY_CLASSES[int(c)],
                 "n_links": int((hw == c).sum()),
                 "median_abs_y": round(float(np.median(absY[hw == c])), 4),
                 "mean_abs_y": round(float(absY[hw == c].mean()), 4)}
                for c in sorted(set(hw.tolist()))]
            entry["binning_note"] = ("nominal category: summarised per class, not "
                                     "binned numerically and not correlated")
        elif i != 2:
            lo, hi = np.percentile(X[car, i], [1, 99])
            edges = (np.unique(np.round(np.linspace(lo, hi, 15), 6))
                     if hi > lo else np.array([lo, lo + 1]))
            entry["binned_vs_abs_response"] = binned(X[car, i], absY[car], edges)
        else:
            mag = np.abs(red[M])
            ry = np.abs(y)[M]
            edges = np.unique(np.percentile(mag, np.linspace(0, 100, 13)))
            entry["binned_vs_abs_response"] = binned(mag, ry, edges)
            entry["binning_note"] = ("node-level over intervened links only, "
                                     "not per-link averages")
        feats.append(entry)
    jdump({"n_links": int(n), "n_scenarios": int(red.shape[0]),
           "binning_note": "binned summaries use car-capable links "
                           "(capacity_base_case > 0) unless stated otherwise; "
                           "y is the per-link mean absolute response over scenarios",
           "features": feats}, OUT / "feature_summary.json")

    # ---- narrative link ------------------------------------------------------
    i = NARRATIVE_LINK
    jdump({
        "link_row": int(i),
        "why": "highest base-case car volume in the network, a trunk road, never "
               "intervened in any of the 1,000 scenarios, yet responds strongly",
        "identity_check": {
            "unique_max_vol_base_case":
                bool(X[i, 0] == X[:, 0].max()
                     and int((X[:, 0] == X[:, 0].max()).sum()) == 1),
            "rows_sharing_geometry": [int(v) for v in np.where(
                (np.concatenate([pos[:, 0, :], pos[:, 1, :]], 1) ==
                 np.concatenate([pos[i, 0, :], pos[i, 1, :]])).all(1))[0]],
            "no_matsim_or_osm_id_available": True,
        },
        "start_lon": round(float(pos[i, 0, 0]), R), "start_lat": round(float(pos[i, 0, 1]), R),
        "end_lon": round(float(pos[i, 1, 0]), R), "end_lat": round(float(pos[i, 1, 1]), R),
        "highway_code": int(hw[i]), "highway_class": HIGHWAY_CLASSES[int(hw[i])],
        "vol_base_case": round(float(X[i, 0]), 2),
        "capacity_base_case": round(float(X[i, 1]), 2),
        "length_m": round(float(X[i, 5]), 2),
        "times_intervened": int(per_link[i]),
        "mean_abs_response_vehh": round(float(absY[i]), 4),
        "std_response_vehh": round(float(stdY[i]), 4),
        "response_sorted_vehh": [round(float(v), 3) for v in np.sort(y[:, i])],
    }, OUT / "narrative_link.json")

    # ---- representative links ------------------------------------------------
    jdump({"selection": "rule-based extrema of measured quantities",
           "links": [{"link_row": idx, "selection_rule": nm,
                      "selection_reason": reason,
                      "highway_code": int(hw[idx]),
                      "times_intervened": int(per_link[idx]),
                      "mean_abs_response_vehh": round(float(absY[idx]), 4),
                      "std_response_vehh": round(float(stdY[idx]), 4),
                      "mid_lon": round(float(pos[idx, 2, 0]), R),
                      "mid_lat": round(float(pos[idx, 2, 1]), R)}
                     for nm, (idx, reason) in link_rules(red, y, X, ei).items()]},
          OUT / "representative_links.json")

    total = sum(f.stat().st_size for f in OUT.glob("*") if f.is_file())
    print(f"\n  total {total/1024/1024:.2f} MB across "
          f"{len(list(OUT.glob('*')))} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
