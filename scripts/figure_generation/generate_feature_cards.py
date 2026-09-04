#!/usr/bin/env python
"""One diagram per feature: what it is, where it is, and how it relates to the target.

Six cards, one for each column of `x`. Each answers the same four questions in the
same four places, so the set can be read side by side:

    what it is        the header, in plain words
    where it is       the feature drawn on the real Paris network
    how it spreads    its distribution
    what it does      its relationship with the response

Numbers in the stat strip are read from feature_statistics.json rather than
recomputed, so a card can never disagree with the published asset.

    python scripts/figure_generation/generate_feature_cards.py \
        --corpus DIR --cache DIR

Output: docs/figures/features/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "data_exploration"))

import matplotlib.pyplot as plt  # noqa: E402
import portfolio_style as ps  # noqa: E402
from common import FEATURES, HIGHWAY_CLASSES, MODEL_COLS, add_common_args, load  # noqa: E402

OUT = REPO / "docs" / "figures" / "features"
STATS = REPO / "docs" / "portfolio_data_story" / "assets" / "feature_statistics.json"

#: Per-feature copy. `plain` is the one-line answer to "what is this column".
COPY = {
    "VOL_BASE_CASE": dict(
        plain="How much car traffic the street already carries, before any policy",
        unit="vehicles per hour", cmap=ps.FLOW,
        source="links_base_case['vol_car'] in process_simulations_for_gnn.py",
        insight="The strongest single predictor of where traffic moves. Response rises "
                "to a peak near 500 veh/h and then falls: the busiest roads absorb "
                "a diversion without changing much."),
    "CAPACITY_BASE_CASE": dict(
        plain="How much traffic the street could carry if it filled up",
        unit="vehicles per hour", cmap=ps.FLOW,
        source="np.where(modes.contains('car'), capacity, 0)",
        insight="Only 36 distinct values, and half the network sits at exactly "
                "480 veh/h. Its response curve peaks near 2,500 veh/h and falls "
                "away again, like base volume."),
    "CAPACITY_REDUCTION": dict(
        plain="How much capacity this scenario's policy takes away",
        unit="vehicles per hour, negative", cmap=ps.HEAT_LIGHT,
        source="capacities_new - capacity_base_case",
        insight="The only column that changes between scenarios. Capacity is only "
                "ever removed, never added, and only on three road classes."),
    "FREESPEED": dict(
        plain="How fast traffic is allowed to flow when the street is empty",
        unit="metres per second", cmap=ps.FLOW,
        source="np.where(modes.contains('car'), freespeed, 0)",
        insight="Sixteen discrete values. 8.33 m/s is 30 km/h, and it is the median, "
                "the lower quartile and the upper quartile at once."),
    "HIGHWAY": dict(
        plain="What kind of road it is, as an OSM class code",
        unit="nominal class code", cmap=None,
        source="gdf['highway'].apply(highway_mapping.get, -1)",
        insight="The one column the model never sees. The integers are names, not "
                "amounts, so arithmetic on them would be meaningless."),
    "LENGTH": dict(
        plain="How long the street is",
        unit="metres", cmap=ps.FLOW,
        source="edge feature dictionary in process_simulations_for_gnn.py",
        insight="The only feature with essentially no rank relationship to the "
                "response on its own (Spearman -0.08)."),
}

def class_name(code):
    """Short, readable road-class name for a HIGHWAY code.

    The codes mean nothing to a reader on their own, and this chart exists to make
    the point that they are names rather than numbers.
    """
    full = HIGHWAY_CLASSES.get(int(code), "unmapped")
    return {-1: "public transp.", 0: "trunk"}.get(int(code), full.split(" /")[0])


CLASS_COLOURS = {
    -1: "#94A3B8", 0: ps.PURPLE, 1: ps.RED, 2: ps.AMBER, 3: ps.GREEN,
    4: ps.BLUE_SOFT, 5: "#A5B4FC", 6: "#F0ABFC", 7: "#FDBA74", 8: "#CBD5E1",
    9: "#67E8F9",
}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png")


def stat_strip(fig, items, y=0.115):
    """A row of number/label pairs across the foot of the card."""
    n = len(items)
    x0, span = 0.055, 0.90
    for i, (value, label) in enumerate(items):
        x = x0 + span * (i / n)
        fig.text(x, y, value, fontsize=17.5, color=ps.INK, fontweight="600",
                 ha="left", va="baseline")
        fig.text(x, y - 0.042, label, fontsize=9.8, color=ps.FAINT, ha="left",
                 va="baseline")


def card(name, col, X, red, absy, pos, stats):
    meta = COPY[name]
    used = col in MODEL_COLS
    hw = X[:, 4].astype(int)
    is_dyn = name == "CAPACITY_REDUCTION"
    values = np.abs(red).mean(0) if is_dyn else X[:, col]

    fig = plt.figure(figsize=(13.2, 8.0))
    axM = fig.add_axes([0.045, 0.215, 0.40, 0.545])
    if name == "HIGHWAY":
        # Eleven rotated class names need room beneath each panel, so the two
        # right-hand charts are shorter and pushed apart.
        axD = fig.add_axes([0.545, 0.605, 0.415, 0.160])
        axR = fig.add_axes([0.545, 0.310, 0.415, 0.150])
    else:
        axD = fig.add_axes([0.545, 0.520, 0.415, 0.245])
        axR = fig.add_axes([0.545, 0.215, 0.415, 0.215])

    # --- where it is -----------------------------------------------------------
    if name == "HIGHWAY":
        ps.network(axM, pos, color="#E9EDF3", lw=0.30)
        for code in sorted(set(hw.tolist()), key=lambda c: (hw == c).sum(), reverse=True):
            m = hw == code
            ps.network(axM, pos[m], color=CLASS_COLOURS.get(code, ps.FAINT), lw=0.55)
        ps.focus_on(axM, pos)
    else:
        shown = values.copy()
        if is_dyn:
            ps.network(axM, pos, color="#E9EDF3", lw=0.30)
            m = shown > 0
            ps.network(axM, pos[m], values=shown[m], cmap=meta["cmap"], lw=0.62,
                       vmax=float(np.percentile(shown[m], 99)))
            ps.focus_on(axM, pos)
        else:
            ps.network(axM, pos, values=np.log10(shown + 1) if name != "FREESPEED"
                       else shown, cmap=meta["cmap"], lw=0.55, background="#0B1220")
    axM.text(0, 1.035, "Where it is", transform=axM.transAxes, fontsize=12.4,
             color=ps.INK, fontweight="600", va="bottom")

    # --- how it spreads --------------------------------------------------------
    if name == "HIGHWAY":
        codes = sorted(set(hw.tolist()))
        axD.bar(range(len(codes)), [int((hw == c).sum()) for c in codes],
                color=[CLASS_COLOURS.get(c, ps.FAINT) for c in codes], width=0.72)
        axD.set_xticks(range(len(codes)))
        axD.set_xticklabels([class_name(c) for c in codes], fontsize=7.6, rotation=38,
                            ha="right")
        axD.set_yscale("log")
    else:
        v = red.ravel()[red.ravel() != 0] if is_dyn else X[:, col]
        v = v if is_dyn else v[v > 0]
        axD.hist(v, bins=60, color=ps.AMBER if is_dyn else ps.BLUE, log=True,
                 edgecolor="none")
    ps.clean(axD, grid_axis="y")
    axD.tick_params(labelsize=9.2)
    axD.set_xlabel(meta["unit"], fontsize=9.6)
    axD.text(0, 1.10, "How it spreads", transform=axD.transAxes, fontsize=12.4,
             color=ps.INK, fontweight="600", va="bottom")

    # --- what it does ----------------------------------------------------------
    if name == "HIGHWAY":
        codes = sorted(set(hw.tolist()), key=lambda c: -float(absy[hw == c].mean()))
        axR.bar(range(len(codes)), [float(absy[hw == c].mean()) for c in codes],
                color=[CLASS_COLOURS.get(c, ps.FAINT) for c in codes], width=0.72)
        axR.set_xticks(range(len(codes)))
        axR.set_xticklabels([class_name(c) for c in codes], fontsize=7.6, rotation=38,
                            ha="right")
        axR.set_xlabel("ordered by response, not by code", fontsize=9.6)
    else:
        # Equal-width bands merged rightwards to at least 100 links, not quantile
        # bins. These features are heavily skewed, and a quantile top bin swallows
        # the whole sparse tail -- which is exactly where VOL_BASE_CASE turns over.
        cx, cy = [], []
        lo, hi_max = 0.0, float(values.max())
        width = hi_max / 24.0
        while lo < hi_max:
            hi = lo + width
            m = (values >= lo) & (values < hi)
            while m.sum() < 100 and hi < hi_max:
                hi += width
                m = (values >= lo) & (values < hi)
            if m.sum() == 0:
                break
            cx.append(0.5 * (lo + hi)); cy.append(float(absy[m].mean()))
            lo = hi
        axR.plot(cx, cy, "-", color=ps.RED, lw=2.2)
        axR.plot(cx, cy, "o", color=ps.RED, ms=5, mec=ps.PAPER, mew=1.2)
        axR.set_xlabel(f"{meta['unit']} (bands of at least 100 links)", fontsize=9.6)
    ps.clean(axR, grid_axis="y")
    axR.tick_params(labelsize=9.2)
    axR.set_ylabel("mean change\nin volume (veh/h)", fontsize=9.4)
    axR.text(0, 1.16 if name == "HIGHWAY" else 1.11,
             "What it does to the traffic", transform=axR.transAxes,
             fontsize=12.4, color=ps.INK, fontweight="600", va="bottom")

    # --- header, stats, footnote ------------------------------------------------
    badge = "used as a model feature" if used else "never fed to the model"
    ps.title_block(fig, f"{col}  ·  {name}", meta["plain"], y=0.965, size=23)
    fig.text(0.945, 0.965, badge, fontsize=11.4, ha="right", va="top",
             color=ps.GREEN if used else ps.RED, fontweight="600")

    if name == "HIGHWAY":
        strip = [(f"{len(set(hw.tolist()))}", "distinct classes"),
                 ("11,305", "links ever intervened"),
                 (f"{100*(hw==4).mean():.0f}%", "residential, the largest class"),
                 ("static", "identical in every scenario"),
                 ("excluded", "not a model input")]
    else:
        s = stats["stats_over_all_scenarios"] if is_dyn else stats["stats"]
        strip = [(f"{s['min']:,.0f}", "minimum"), (f"{s['median']:,.1f}", "median"),
                 (f"{s['max']:,.0f}", "maximum"),
                 (f"{s['n_unique']:,}", "distinct values"),
                 (f"{s['zero_pct']:.1f}%", "exactly zero"),
                 ("dynamic" if is_dyn else "static",
                  "changes per scenario" if is_dyn else "same in every scenario")]
    stat_strip(fig, strip)

    ps.footnote(fig, [meta["insight"], f"Computed in preprocessing as: {meta['source']}"],
                y=0.055)
    save(fig, f"{col}_{name.lower()}")


def main() -> int:
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    ps.apply()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    absy = np.abs(y).mean(0)
    allstats = json.loads(STATS.read_text(encoding="utf-8"))["features"]
    print(f"corpus {y.shape[0]:,} scenarios x {y.shape[1]:,} links\n")
    for col, name in enumerate(FEATURES):
        card(name, col, X, red, absy, pos, allstats[name])
    print(f"\ncards written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
