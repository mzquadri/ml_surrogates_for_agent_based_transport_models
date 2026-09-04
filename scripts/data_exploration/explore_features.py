#!/usr/bin/env python
"""Full-corpus statistics for the six columns of `x`, and what the model consumes.

Continuous columns get distributional statistics and a binned response curve
against the target. HIGHWAY gets categorical treatment only: its codes are labels
for road classes, and a mean of label codes would be meaningless.

The response curve is the part worth reading. Correlation alone would report that
volume and response are weakly related; binning by volume shows the relationship
is not monotonic, which is a different and more useful statement.

    python scripts/data_exploration/explore_features.py --corpus DIR --cache DIR

Writes feature_statistics.json and model_inputs.json to the web-asset directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from common import FEATURES, HIGHWAY_CLASSES, MODEL_COLS, add_common_args, load  # noqa: E402

OUT = REPO / "docs" / "portfolio_data_story" / "assets"

#: Column-by-column provenance, traced to the preprocessing source rather than
#: inferred from the values. Line references are to the file named in `source`.
META = {
    "VOL_BASE_CASE": dict(
        unit="vehicles per hour",
        meaning="Car volume on the link in the untouched base-case simulation.",
        source="process_simulations_for_gnn.py: links_base_case['vol_car'].values",
        static=True),
    "CAPACITY_BASE_CASE": dict(
        unit="vehicles per hour",
        meaning="Link capacity in the base case; zero where the link carries no car mode.",
        source="help_functions.py get_basic_edge_attributes: "
               "np.where(modes.str.contains('car'), capacity, 0)",
        static=True),
    "CAPACITY_REDUCTION": dict(
        unit="vehicles per hour (signed, <= 0)",
        meaning="Capacity removed by the policy: scenario capacity minus base capacity. "
                "The only column that differs between scenarios.",
        source="help_functions.py get_basic_edge_attributes: "
               "capacities_new - capacity_base_case",
        static=False),
    "FREESPEED": dict(
        unit="metres per second",
        meaning="Free-flow speed permitted on the link; zero where no car mode is allowed.",
        source="help_functions.py get_basic_edge_attributes: "
               "np.where(modes.str.contains('car'), freespeed, 0)",
        static=True),
    "HIGHWAY": dict(
        unit="categorical code (nominal)",
        meaning="OSM road class mapped to an integer. -1 means public transport or a "
                "class absent from the mapping.",
        source="help_functions.py get_basic_edge_attributes: "
               "gdf['highway'].apply(highway_mapping.get, -1)",
        static=True),
    "LENGTH": dict(
        unit="metres",
        meaning="Physical length of the road link.",
        source="process_simulations_for_gnn.py edge feature dictionary",
        static=True),
}

#: Rows the stored num_nodes excludes: public-transport links with no car access.
N_NODES_STORED = 31_559


def jdump(obj, name):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"  {name:32} {p.stat().st_size/1024:6.1f} KB")


def describe(v: np.ndarray) -> dict:
    """Distributional summary for a continuous column."""
    finite = np.isfinite(v)
    q = np.percentile(v[finite], [0, 1, 25, 50, 75, 99, 100])
    return {
        "count": int(v.size),
        "min": round(float(q[0]), 6), "p01": round(float(q[1]), 6),
        "q1": round(float(q[2]), 6), "median": round(float(q[3]), 6),
        "q3": round(float(q[4]), 6), "p99": round(float(q[5]), 6),
        "max": round(float(q[6]), 6),
        "mean": round(float(v[finite].mean()), 6),
        "std": round(float(v[finite].std()), 6),
        "iqr": round(float(q[4] - q[2]), 6),
        "n_unique": int(np.unique(v).size),
        "zero_pct": round(float(100 * (v == 0).mean()), 4),
        "nan_pct": round(float(100 * (~finite).mean()), 6),
        "negative_pct": round(float(100 * (v < 0).mean()), 4),
    }


def tail_curve(feature: np.ndarray, resp: np.ndarray, width: float,
               min_n: int = 100) -> list:
    """Equal-width bins, merged rightwards until each holds at least `min_n` links.

    Quantile bins cannot resolve the top of a heavily skewed feature: with twelve
    of them the busiest bin spans 167 to 1,596 veh/h and averages away everything
    inside it. Equal-width bins can resolve it but run out of links, so bins are
    merged until the mean is worth reporting, and the count and standard error are
    returned so a reader can see how much each point is worth.
    """
    out = []
    lo = 0.0
    hi_max = float(feature.max())
    while lo < hi_max:
        hi = lo + width
        m = (feature >= lo) & (feature < hi)
        while m.sum() < min_n and hi < hi_max:
            hi += width
            m = (feature >= lo) & (feature < hi)
        if m.sum() == 0:
            break
        v = resp[m]
        out.append({
            "low": round(float(lo), 2), "high": round(float(hi), 2),
            "n_links": int(m.sum()),
            "mean_abs_response": round(float(v.mean()), 4),
            "median_abs_response": round(float(np.median(v)), 4),
            "sem": round(float(v.std() / np.sqrt(m.sum())), 4),
        })
        lo = hi
    return out


def response_curve(feature: np.ndarray, resp: np.ndarray, n_bins: int = 12) -> list:
    """Median |response| per feature bin, using quantile edges so each bin is populated.

    Quantile bins rather than equal-width: the features are heavily skewed, and
    equal-width bins would put almost every link in the first bin and report the
    tail from a handful of points.
    """
    pos = feature > 0
    if pos.sum() < n_bins * 10:
        return []
    edges = np.unique(np.percentile(feature[pos], np.linspace(0, 100, n_bins + 1)))
    if edges.size < 3:
        return []
    idx = np.clip(np.digitize(feature, edges) - 1, 0, edges.size - 2)
    out = []
    for b in range(edges.size - 1):
        m = (idx == b) & pos
        if m.sum() < 5:
            continue
        out.append({
            "bin": b,
            "feature_low": round(float(edges[b]), 4),
            "feature_high": round(float(edges[b + 1]), 4),
            "n_links": int(m.sum()),
            "median_abs_response": round(float(np.median(resp[m])), 6),
            "mean_abs_response": round(float(resp[m].mean()), 6),
            "p90_abs_response": round(float(np.percentile(resp[m], 90)), 6),
        })
    return out


def main() -> int:
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    n_scen, n_links = y.shape
    print(f"corpus: {n_scen:,} scenarios x {n_links:,} links "
          f"= {n_scen*n_links:,} node observations\n")

    absy = np.abs(y).mean(0)             # mean |response| per link, over all scenarios
    ever = (red != 0).any(0)             # link intervened in at least one scenario
    times = (red != 0).sum(0)            # how often each link was intervened
    hw = X[:, 4].astype(int)
    deg = (np.bincount(ei[0], minlength=n_links)
           + np.bincount(ei[1], minlength=n_links))

    feats = {}
    for col, name in enumerate(FEATURES):
        meta = META[name]
        used = col in MODEL_COLS
        entry = {
            "index": col, "name": name, **meta,
            "dtype": str(X.dtype),
            "used_by_model": used,
            "exclusion_reason": (None if used else
                                 "Nominal road class encoded as an integer. The codes "
                                 "carry no order or distance, so feeding them to a "
                                 "network that computes weighted sums would invent "
                                 "arithmetic relationships that do not exist."),
        }
        if name == "HIGHWAY":
            # Categorical only: no mean, no quantiles, no correlation.
            classes = []
            for code in sorted(set(hw.tolist())):
                m = hw == code
                classes.append({
                    "code": int(code),
                    "road_class": HIGHWAY_CLASSES.get(int(code), "unmapped"),
                    "n_links": int(m.sum()),
                    "pct_of_network": round(float(100 * m.mean()), 4),
                    "ever_intervened": int(ever[m].sum()),
                    "pct_of_class_intervened": round(float(100 * ever[m].mean()), 4),
                    "mean_abs_response": round(float(absy[m].mean()), 6),
                    "median_abs_response": round(float(np.median(absy[m])), 6),
                    "directly_intervened": bool(ever[m].any()),
                })
            entry["treatment"] = "categorical"
            entry["classes"] = classes
            entry["n_classes"] = len(classes)
        elif name == "CAPACITY_REDUCTION":
            flat = red.ravel()
            nz = flat[flat != 0]
            entry["treatment"] = "continuous, dynamic"
            entry["stats_over_all_scenarios"] = describe(flat)
            entry["nonzero_only"] = describe(nz)
            entry["n_distinct_magnitudes"] = int(np.unique(np.round(nz, 4)).size)
            entry["links_intervened_per_scenario"] = {
                "min": int((red != 0).sum(1).min()),
                "median": int(np.median((red != 0).sum(1))),
                "max": int((red != 0).sum(1).max()),
                "mean": round(float((red != 0).sum(1).mean()), 2),
            }
            entry["links_ever_intervened"] = int(ever.sum())
            entry["times_intervened_per_link"] = {
                "max": int(times.max()),
                "mean_over_eligible": round(float(times[ever].mean()), 2),
            }
            entry["response_curve_vs_abs_target"] = response_curve(
                np.abs(red).mean(0), absy)
        else:
            v = X[:, col]
            entry["treatment"] = "continuous, static"
            entry["stats"] = describe(v)
            entry["spearman_vs_mean_abs_response"] = round(
                float(sps.spearmanr(v, absy).statistic), 4)
            entry["response_curve_vs_abs_target"] = response_curve(v, absy)
            if name == "VOL_BASE_CASE":
                # The quantile curve above cannot see inside its own top bin, which
                # spans 167 to 1,596 veh/h. This one can, and it shows the response
                # peaking near 500 veh/h and falling for the busiest links.
                entry["response_curve_fine"] = tail_curve(v, absy, width=67.0)
                entry["shape"] = ("inverted U: rises to a peak near 500 veh/h, then "
                                  "falls for the busiest links")
        feats[name] = entry

    stat_dyn = []
    for col, name in enumerate(FEATURES):
        stat_dyn.append({
            "field": f"x[:, {col}]  {name}",
            "static_or_dynamic": "dynamic" if col == 2 else "static",
            "what_changes": ("capacity removed by the policy, per scenario"
                             if col == 2 else "nothing; identical in all 1,000 scenarios"),
            "model_usage": "input feature" if col in MODEL_COLS else "not used",
        })
    for nm, dyn, what, use in [
        ("pos", "static", "nothing; identical in all 1,000 scenarios",
         "pos[:,0] and pos[:,1] feed the two PointNetConv layers; pos[:,2] is unused"),
        ("y", "dynamic", "the per-link change in car volume the policy caused",
         "training target"),
        ("edge_index", "static", "nothing; identical in all 1,000 scenarios",
         "graph connectivity for all four message-passing layers"),
        ("mode_stats_diff", "dynamic", "per-mode travel-time, distance and trip-count "
         "differences", "not read by any training or evaluation code"),
        ("mode_stats_diff_perc", "dynamic", "the same differences as percentages",
         "not read by any training or evaluation code"),
    ]:
        stat_dyn.append({"field": nm, "static_or_dynamic": dyn,
                         "what_changes": what, "model_usage": use})

    jdump({
        "corpus": {"n_scenarios": int(n_scen), "n_links": int(n_links),
                   "n_node_observations": int(n_scen * n_links)},
        "features": feats,
        "static_vs_dynamic": stat_dyn,
    }, "feature_statistics.json")

    # --- what actually enters the model -------------------------------------------
    jdump({
        "verified_from": [
            "scripts/training/help_functions.py (node_features branch)",
            "scripts/gnn/models/point_net_transf_gat.py (forward)",
            "models/point_net_transf_gat_8th_trial_lower_dropout/trained_model/model.pth",
        ],
        "checkpoint_evidence": {
            "layer": "point_net_conv_1.local_nn.0.weight",
            "shape": [256, 7],
            "reading": "PointNetConv concatenates node features with a 2-D relative "
                       "coordinate, so 7 - 2 = 5 node feature channels.",
        },
        "node_features_used": [FEATURES[c] for c in MODEL_COLS],
        "node_features_excluded": [f for i, f in enumerate(FEATURES)
                                   if i not in MODEL_COLS],
        "also_consumed_by_the_architecture": {
            "pos[:, 0]": "start coordinate, first PointNetConv",
            "pos[:, 1]": "end coordinate, second PointNetConv",
            "edge_index": "connectivity for both PointNetConv, both TransformerConv "
                          "and both GATConv layers",
        },
        "not_consumed": {
            "pos[:, 2]": "midpoint; used only for plotting",
            "mode_stats_diff": "no code path reads it",
            "mode_stats_diff_perc": "no code path reads it",
        },
        "precise_statement": (
            "Five of the six node-attribute columns in x were used as node features. "
            "The model also consumes graph connectivity (edge_index) and two of the "
            "three stored coordinate pairs (pos[:,0], pos[:,1]). It is not true that "
            "only five pieces of information enter the model."),
        "isolated_public_transport_links": {
            "n": int(n_links - N_NODES_STORED),
            "stored_num_nodes": N_NODES_STORED,
            "rows_in_x": int(n_links),
            "description": ("Links with no car mode: zero volume, capacity and "
                            "freespeed, HIGHWAY -1, never intervened, target exactly "
                            "zero in every scenario, and absent from edge_index."),
        },
    }, "model_inputs.json")

    print(f"\nlinks ever intervened: {int(ever.sum()):,} of {n_links:,}")
    print(f"mean |response| per link: {absy.mean():.4f} veh/h")
    print(f"degree: min {deg.min()} median {int(np.median(deg))} max {deg.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
