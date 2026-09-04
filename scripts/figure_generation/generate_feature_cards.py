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


def header(fig, idx, name, plain, used_text, used):
    """Shared header for the cards that are whole tensors rather than x columns."""
    ps.title_block(fig, f"{idx}  ·  {name}", plain, y=0.965, size=23)
    fig.text(0.945, 0.965, used_text, fontsize=11.4, ha="right", va="top",
             color=ps.GREEN if used else ps.RED, fontweight="600")


def card_pos(pos, X, absy):
    """pos [N, 3, 2] -- start, end and midpoint, in WGS84 degrees."""
    fig = plt.figure(figsize=(13.2, 8.0))
    axM = fig.add_axes([0.045, 0.215, 0.40, 0.545])
    axA = fig.add_axes([0.545, 0.520, 0.415, 0.245])
    axB = fig.add_axes([0.545, 0.215, 0.415, 0.215])

    ps.network(axM, pos, values=np.log10(X[:, 0] + 1), cmap=ps.FLOW, lw=0.55,
               background="#0B1220")
    axM.text(0, 1.035, "Where it is", transform=axM.transAxes, fontsize=12.4,
             color=ps.INK, fontweight="600", va="bottom")
    ps.note(axM, 0.5, -0.055, "every line is drawn start -> end from this tensor",
            transform=axM.transAxes, ha="center", size=9.8)

    # Geometric length against the stored LENGTH column: the two disagree because
    # pos keeps only the endpoints while LENGTH follows the road.
    lat0 = float(pos[:, 2, 1].mean())
    dx = (pos[:, 1, 0] - pos[:, 0, 0]) * np.cos(np.deg2rad(lat0)) * 111_320.0
    dy = (pos[:, 1, 1] - pos[:, 0, 1]) * 110_574.0
    straight = np.hypot(dx, dy)
    keep = X[:, 5] > 0
    axA.scatter(X[keep, 5], straight[keep], s=1.4, color=ps.BLUE, alpha=0.18,
                linewidths=0)
    lim = float(np.percentile(X[keep, 5], 99.5))
    axA.plot([0, lim], [0, lim], color=ps.RED, lw=1.6, ls="--")
    axA.set_xlim(0, lim); axA.set_ylim(0, lim)
    ps.clean(axA, grid_axis="y")
    axA.tick_params(labelsize=9.2)
    axA.set_xlabel("stored LENGTH (m)", fontsize=9.6)
    axA.set_ylabel("straight-line\nstart to end (m)", fontsize=9.4)
    axA.text(0, 1.10, "Endpoints only, not the full shape", transform=axA.transAxes,
             fontsize=12.4, color=ps.INK, fontweight="600", va="bottom")
    ps.note(axA, 0.97, 0.10, "points below the line are curved roads",
            transform=axA.transAxes, ha="right", size=9.6)

    slices = ["pos[:, 0]\nstart", "pos[:, 1]\nend", "pos[:, 2]\nmidpoint"]
    use = [True, True, False]
    axB.bar(range(3), [1, 1, 1], color=[ps.GREEN if u else ps.RED for u in use],
            width=0.55)
    axB.set_xticks(range(3)); axB.set_xticklabels(slices, fontsize=9.6)
    axB.set_yticks([])
    ps.clean(axB, left=False)
    for i, u in enumerate(use):
        axB.text(i, 0.5, "read by\nthe model" if u else "never\nread", ha="center",
                 va="center", fontsize=10.2, color=ps.PAPER, fontweight="600")
    axB.text(0, 1.11, "Which of the three the model uses", transform=axB.transAxes,
             fontsize=12.4, color=ps.INK, fontweight="600", va="bottom")

    header(fig, 6, "pos", "Where each road link starts, ends and sits, in WGS84 degrees",
           "two of three slices used", True)
    stat_strip(fig, [
        (f"{pos.shape[0]:,}", "links"), ("3 × 2", "coordinates each"), ("float32", "dtype"),
        ("2.15–2.49", "longitude range"), ("48.76–48.93", "latitude range"),
        ("static", "same in every scenario")])
    ps.footnote(fig, [
        "The midpoint is exactly the mean of start and end, checked numerically: the "
        "largest deviation is 3.8e-06 degrees, which is float32 rounding.",
        "Built in get_link_geometries as torch.stack([start, end, midpoint], dim=1). "
        "The model reads pos[:, 0] and pos[:, 1]; pos[:, 2] is only ever plotted."],
        y=0.055)
    save(fig, "6_pos")


def card_y(y, pos, absy):
    """y [N, 1] -- the target: change in car volume per link."""
    fig = plt.figure(figsize=(13.2, 8.0))
    axM = fig.add_axes([0.045, 0.215, 0.40, 0.545])
    axA = fig.add_axes([0.545, 0.520, 0.415, 0.245])
    axB = fig.add_axes([0.545, 0.215, 0.415, 0.215])

    signed = y.mean(0)
    lim = float(np.percentile(np.abs(signed), 99))
    ps.network(axM, pos, values=np.clip(signed, -lim, lim), cmap=ps.DIVERGE,
               lw=0.55, vmin=-lim, vmax=lim)
    axM.text(0, 1.035, "Who gains and who loses", transform=axM.transAxes,
             fontsize=12.4, color=ps.INK, fontweight="600", va="bottom")
    ps.note(axM, 0.5, -0.055, "blue loses traffic · red gains it",
            transform=axM.transAxes, ha="center", size=9.8)

    flat = y.ravel()
    axA.hist(flat[np.abs(flat) > 0], bins=180, range=(-60, 60), color=ps.GREEN,
             log=True, edgecolor="none")
    ps.clean(axA, grid_axis="y")
    axA.tick_params(labelsize=9.2)
    axA.set_xlabel("change in link volume (veh/h)", fontsize=9.6)
    axA.text(0, 1.10, "Symmetric and heavy-tailed", transform=axA.transAxes,
             fontsize=12.4, color=ps.INK, fontweight="600", va="bottom")

    share = [100 * float((flat > 0).mean()), 100 * float((flat == 0).mean()),
             100 * float((flat < 0).mean())]
    axB.barh([0], [share[0]], color=ps.RED, height=0.5)
    axB.barh([0], [share[1]], left=[share[0]], color=ps.HAIR, height=0.5)
    axB.barh([0], [share[2]], left=[share[0] + share[1]], color=ps.BLUE, height=0.5)
    axB.set_xlim(0, 100); axB.set_yticks([]); axB.set_ylim(-0.6, 0.9)
    ps.clean(axB, left=False)
    axB.set_xlabel("share of all 31,635,000 node observations (%)", fontsize=9.6)
    axB.tick_params(labelsize=9.2)
    for x0, val, lbl, col in ((share[0] / 2, share[0], "gain", ps.PAPER),
                              (share[0] + share[1] / 2, share[1], "no change", ps.BODY),
                              (share[0] + share[1] + share[2] / 2, share[2], "lose",
                               ps.PAPER)):
        axB.text(x0, 0, f"{val:.1f}%", ha="center", va="center", fontsize=10.4,
                 color=col, fontweight="600")
        axB.text(x0, 0.45, lbl, ha="center", va="center", fontsize=9.6, color=ps.MUTED)
    axB.text(0, 1.11, "Traffic is moved, not removed", transform=axB.transAxes,
             fontsize=12.4, color=ps.INK, fontweight="600", va="bottom")

    header(fig, 7, "y", "The answer the model is trained to predict, one number per link",
           "the training target", True)
    stat_strip(fig, [
        (f"{flat.mean():+.3f}", "mean (veh/h)"), (f"{flat.min():.0f}", "largest loss"),
        (f"{flat.max():.0f}", "largest gain"), (f"{absy.mean():.2f}", "mean |change|"),
        (f"{share[1]:.1f}%", "exactly zero"), ("dynamic", "differs per scenario")])
    ps.footnote(fig, [
        "y = vol_car(scenario) - vol_car(base case), per link, from "
        "compute_target_tensor_only_edge_features in the preprocessing.",
        "Gains and losses very nearly cancel, which is what redistribution looks like: "
        "capacity removed in one place pushes traffic elsewhere."], y=0.055)
    save(fig, "7_y")


def card_edge_index(ei, pos, absy, X):
    """edge_index [2, E] -- the line-graph connectivity."""
    n = X.shape[0]
    deg = (np.bincount(ei[0], minlength=n) + np.bincount(ei[1], minlength=n))
    fig = plt.figure(figsize=(13.2, 8.0))
    axM = fig.add_axes([0.045, 0.215, 0.40, 0.545])
    axA = fig.add_axes([0.545, 0.520, 0.415, 0.245])
    axB = fig.add_axes([0.545, 0.215, 0.415, 0.215])

    ps.network(axM, pos, values=deg.astype(float), cmap=ps.HEAT_LIGHT, lw=0.55,
               vmin=0, vmax=8)
    iso = deg == 0
    axM.scatter(pos[iso, 2, 0], pos[iso, 2, 1], s=16, color=ps.PURPLE, zorder=5,
                linewidths=0)
    axM.text(0, 1.035, "How connected each link is", transform=axM.transAxes,
             fontsize=12.4, color=ps.INK, fontweight="600", va="bottom")
    ps.note(axM, 0.5, -0.055,
            f"purple: the {int(iso.sum())} links joined to nothing",
            transform=axM.transAxes, ha="center", size=9.8)

    ds = np.arange(deg.max() + 1)
    axA.bar(ds, [int((deg == d).sum()) for d in ds], color=ps.BLUE, width=0.72)
    ps.clean(axA, grid_axis="y")
    axA.tick_params(labelsize=9.2)
    axA.set_xlabel("degree: how many other links it meets", fontsize=9.6)
    axA.text(0, 1.10, "Most links meet four others", transform=axA.transAxes,
             fontsize=12.4, color=ps.INK, fontweight="600", va="bottom")

    ds2 = [d for d in ds if (deg == d).sum() >= 5]
    axB.plot(ds2, [float(absy[deg == d].mean()) for d in ds2], "-", color=ps.RED, lw=2.2)
    axB.plot(ds2, [float(absy[deg == d].mean()) for d in ds2], "o", color=ps.RED, ms=5,
             mec=ps.PAPER, mew=1.2)
    ps.clean(axB, grid_axis="y")
    axB.tick_params(labelsize=9.2)
    axB.set_xlabel("degree", fontsize=9.6)
    axB.set_ylabel("mean change\nin volume (veh/h)", fontsize=9.4)
    axB.text(0, 1.11, "Being connected matters; how much does not",
             transform=axB.transAxes, fontsize=12.4, color=ps.INK, fontweight="600",
             va="bottom")

    header(fig, 8, "edge_index",
           "Which road links meet at an intersection — the graph the GNN walks",
           "read by all six layers", True)
    stat_strip(fig, [
        (f"{ei.shape[1]:,}", "directed edges"), (f"{int(np.median(deg))}", "median degree"),
        (f"{int(deg.max())}", "maximum degree"), ("766", "self-loops"),
        ("121", "components"), ("static", "same in every scenario")])
    ps.footnote(fig, [
        "This is the line graph of the road network: a road link is a node, and an edge "
        "means two links meet. That is why the model predicts one value per street.",
        "92.7% of links sit in one connected component; the other 120 components hold "
        "between 1 and 319 links each."], y=0.055)
    save(fig, "8_edge_index")


def card_mode_stats(msd, msdp, which):
    """The two auxiliary tensors, which nothing reads."""
    is_perc = which == "perc"
    arr = msdp if is_perc else msd
    name = "mode_stats_diff_perc" if is_perc else "mode_stats_diff"
    idx = 10 if is_perc else 9

    fig = plt.figure(figsize=(13.2, 8.0))
    axH = fig.add_axes([0.055, 0.315, 0.40, 0.400])
    axR = fig.add_axes([0.560, 0.315, 0.395, 0.400])

    cols = ["travel\ntime", "routed\ndistance", "trip\ncount"]
    mean = arr.mean(0)
    show = mean.copy()
    im = axH.imshow(show, cmap="RdBu_r" if is_perc else "PuOr",
                    vmin=-100 if is_perc else -np.abs(show).max(),
                    vmax=100 if is_perc else np.abs(show).max(), aspect="auto")
    axH.set_xticks(range(3)); axH.set_xticklabels(cols, fontsize=9.6)
    axH.set_yticks(range(6))
    axH.set_yticklabels([f"mode {i}" for i in range(6)], fontsize=9.6)
    for i in range(6):
        for j in range(3):
            v = show[i, j]
            txt = f"{v:,.3f}%" if is_perc else (f"{v:,.0f}" if abs(v) > 100 else f"{v:,.1f}")
            axH.text(j, i, txt, ha="center", va="center", fontsize=9.0,
                     color=ps.PAPER if abs(v) > (55 if is_perc else np.abs(show).max()*0.5)
                     else ps.INK)
    ps.bare(axH)
    axH.set_xticks(range(3)); axH.set_xticklabels(cols, fontsize=9.6)
    axH.set_yticks(range(6)); axH.set_yticklabels([f"mode {i}" for i in range(6)],
                                                  fontsize=9.6)
    axH.tick_params(length=0)
    axH.text(0, 1.06, "Six transport modes × three quantities", transform=axH.transAxes,
             fontsize=12.4, color=ps.INK, fontweight="600", va="bottom")
    fig.colorbar(im, ax=axH, fraction=0.045, pad=0.03)

    if is_perc:
        # The reconstruction that explains the near -100%: base = diff / (perc/100),
        # and scenario/base then turns out to be 1/trip_count.
        with np.errstate(divide="ignore", invalid="ignore"):
            base = msd / (msdp / 100.0)
            scen = base + msd
        ratio, inv_tc = [], []
        for s in range(arr.shape[0]):
            for i in range(6):
                tc = scen[s, i, 2]
                if not np.isfinite(tc) or tc <= 0:
                    continue
                for j in (0, 1):
                    r = scen[s, i, j] / base[s, i, j]
                    if np.isfinite(r) and r > 0:
                        ratio.append(r); inv_tc.append(1.0 / tc)
        ratio, inv_tc = np.array(ratio), np.array(inv_tc)
        axR.scatter(inv_tc, ratio, s=5, color=ps.RED, alpha=0.25, linewidths=0)
        lo, hi = float(inv_tc.min()), float(inv_tc.max())
        axR.plot([lo, hi], [lo, hi], color=ps.INK, lw=1.5, ls="--")
        axR.set_xscale("log"); axR.set_yscale("log")
        ps.clean(axR, grid_axis="y")
        axR.tick_params(labelsize=9.2)
        axR.set_xlabel("1 / trip count", fontsize=9.6)
        axR.set_ylabel("scenario ÷ base", fontsize=9.6)
        axR.text(0, 1.06, "Why it reads −99.99%", transform=axR.transAxes,
                 fontsize=12.4, color=ps.INK, fontweight="600", va="bottom")
        err = np.abs(ratio - inv_tc) / inv_tc
        ps.note(axR, 0.04, 0.94,
                f"the ratio is 1/trip_count\nmedian error {100*np.median(err):.2f}%",
                transform=axR.transAxes, va="top", size=10.6, color=ps.INK)
    else:
        spread = arr.std(0) / np.maximum(np.abs(arr.mean(0)), 1e-9) * 100
        axR.imshow(spread, cmap="Blues", aspect="auto")
        axR.set_xticks(range(3)); axR.set_xticklabels(cols, fontsize=9.6)
        axR.set_yticks(range(6))
        axR.set_yticklabels([f"mode {i}" for i in range(6)], fontsize=9.6)
        for i in range(6):
            for j in range(3):
                axR.text(j, i, f"{spread[i, j]:.2f}%", ha="center", va="center",
                         fontsize=9.0, color=ps.INK)
        axR.tick_params(length=0)
        for sp in axR.spines.values():
            sp.set_visible(False)
        axR.text(0, 1.06, "How much each cell moves between scenarios",
                 transform=axR.transAxes, fontsize=12.4, color=ps.INK,
                 fontweight="600", va="bottom")

    plain = ("Per-mode differences expressed as percentages — and the one field with a "
             "known defect" if is_perc else
             "Per-mode travel time, distance and trip count, scenario minus base case")
    header(fig, idx, name, plain, "never read by any code", False)
    stat_strip(fig, [
        ("6 × 3", "shape"), ("float64" if is_perc else "float32", "dtype"),
        (f"{arr.shape[0]:,}", "scenarios"),
        ("0" if is_perc else "—", "cells exactly −100" if is_perc else "unused"),
        ("dynamic", "differs per scenario"), ("unused", "no code path reads it")])
    if is_perc:
        ps.footnote(fig, [
            "All six rows sit at about -99.99% in the first two columns, because a "
            "per-mode sum is being subtracted from a per-mode mean and then divided "
            "by the sum. The third column is unaffected: both sides are counts.",
            "Proved by reconstruction rather than guessed: base = diff / (perc/100) "
            "implies scenario ÷ base = 1 / trip_count, which holds to a third of a "
            "percent. See CORRIGENDUM C12.",
            "Nothing downstream depends on it: no training or evaluation code reads "
            "this tensor."], y=0.055)
    else:
        ps.footnote(fig, [
            "Computed as df_mode_stats[numeric] - basecase[numeric] in "
            "process_simulations_for_eign.py. The six rows are transport modes; the "
            "three columns are travel time, routed distance and trip count.",
            "Stored in every scenario and read by nothing. It is documented here rather "
            "than dropped, because an unexplained stored field is worse than an "
            "unused one."], y=0.055)
    save(fig, f"{idx}_{name}")


def main() -> int:
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    ps.apply()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    absy = np.abs(y).mean(0)
    allstats = json.loads(STATS.read_text(encoding="utf-8"))["features"]
    print(f"corpus {y.shape[0]:,} scenarios x {y.shape[1]:,} links\n")

    print("the six columns of x")
    for col, name in enumerate(FEATURES):
        card(name, col, X, red, absy, pos, allstats[name])

    print("\nthe five other stored tensors")
    card_pos(pos, X, absy)
    card_y(y, pos, absy)
    card_edge_index(ei, pos, absy, X)

    msd_path = args.cache / "mode_stats_diff.npy"
    if msd_path.exists():
        msd = np.load(msd_path)
        msdp = np.load(args.cache / "mode_stats_diff_perc.npy")
        card_mode_stats(msd, msdp, "abs")
        card_mode_stats(msd, msdp, "perc")
    else:
        print("  skipped the two mode_stats cards: run explore_tensors.py first to "
              "cache them")

    print(f"\ncards written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
