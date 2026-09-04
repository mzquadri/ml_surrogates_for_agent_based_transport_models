#!/usr/bin/env python
"""The showcase figure set: the dataset, drawn to be looked at.

These are the figures for the README and the website. They cover the same
verified numbers as docs/figures/data_exploration/, but each one carries a single
idea at a size where it can be read, with the takeaway written on the chart.

The maps draw the real street geometry: every link is a segment from pos[:, 0] to
pos[:, 1], so what you see is the actual road network of Paris, not a scatter of
midpoints.

    python scripts/figure_generation/generate_portfolio_figures.py \
        --corpus DIR --cache DIR

Output: docs/figures/portfolio/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "data_exploration"))

import matplotlib.pyplot as plt  # noqa: E402
import portfolio_style as ps  # noqa: E402
from common import HIGHWAY_CLASSES, MODEL_COLS, add_common_args, load  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = REPO / "docs" / "figures" / "portfolio"
NIGHT = "#0B1220"


def save(fig, name, face=ps.PAPER):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", facecolor=face)
    plt.close(fig)
    print(f"  wrote {name}.png")


# ---------------------------------------------------------------------------
def fig_hero(pos, X, polys):
    """The network itself, coloured by how much traffic it already carries."""
    vol = X[:, 0]
    fig = plt.figure(figsize=(13.2, 10.4))
    fig.patch.set_facecolor(NIGHT)
    ax = fig.add_axes([0.03, 0.075, 0.94, 0.755])
    ps.network(ax, pos, values=np.log10(vol + 1), cmap=ps.FLOW, lw=0.6,
               background=NIGHT)
    ps.districts(ax, polys, colour="#93A4B8", lw=1.0, alpha=0.62, label=True,
                 label_size=7.6, label_colour="#E8EEF6", label_bg="#16233A")

    fig.text(0.055, 0.972, "31,635 streets, one city", ha="left", va="top",
             fontsize=27, color="#F8FAFC", fontweight="600")
    fig.text(0.055, 0.930,
             "Every line is one road link in the MATSim model of Paris, drawn from its own "
             "stored start and end coordinates\nand shaded by the car volume it carries "
             "before any policy is applied.",
             ha="left", va="top", fontsize=12.6, color="#94A3B8", linespacing=1.6)

    ps.gradient_key(fig, [0.055, 0.862, 0.20, 0.011], ps.FLOW,
                    "quiet", "1,596 veh/h")
    fig.text(0.055, 0.052,
             "The ring is the Périphérique; the dark band through the middle is the Seine. "
             "23.9% of links carry no car traffic at all.\n"
             "Source: train-data-v1 release, 1,000 scenarios × 31,635 links.",
             ha="left", va="top", fontsize=9.8, color="#64748B", linespacing=1.7)
    save(fig, "01_the_network", face=NIGHT)


def fig_features(X, red, absy):
    """The six columns, as six small charts that each say what the column is."""
    fig = plt.figure(figsize=(13.2, 9.4))
    # Each panel carries a three-line header above its axes, so the rows need far
    # more vertical separation than a default grid gives them.
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.975, top=0.700, bottom=0.170,
                          hspace=0.90, wspace=0.28)
    hw = X[:, 4].astype(int)

    specs = [
        (0, "VOL_BASE_CASE", "vehicles per hour", ps.BLUE,
         "How busy the street already is"),
        (1, "CAPACITY_BASE_CASE", "vehicles per hour", ps.BLUE,
         "How much it can carry"),
        (2, "CAPACITY_REDUCTION", "vehicles per hour", ps.AMBER,
         "What the policy takes away"),
        (3, "FREESPEED", "metres per second", ps.BLUE,
         "How fast traffic may flow"),
        (4, "HIGHWAY", "road class code", ps.RED,
         "What kind of road it is - eleven labels, no order"),
        (5, "LENGTH", "metres", ps.BLUE, "How long the street is"),
    ]
    for k, (col, name, unit, colour, plain) in enumerate(specs):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        used = col in MODEL_COLS
        if name == "HIGHWAY":
            codes = sorted(set(hw.tolist()))
            counts = [int((hw == c).sum()) for c in codes]
            ax.bar(range(len(codes)), counts, color=colour, width=0.72)
            ax.set_xticks(range(len(codes)))
            ax.set_xticklabels([str(c) for c in codes], fontsize=8.6)
            ax.set_yscale("log")
            ps.clean(ax, grid_axis="y")
        else:
            v = red.ravel() if name == "CAPACITY_REDUCTION" else X[:, col]
            v = v[v != 0] if name == "CAPACITY_REDUCTION" else v[v > 0]
            ax.hist(v, bins=64, color=colour, log=True, edgecolor="none")
            ps.clean(ax, grid_axis="y")
        ax.tick_params(labelsize=9.2)
        ax.set_xlabel(unit, fontsize=9.6)

        badge = "used by the model" if used else "not a model feature"
        bcol = ps.GREEN if used else ps.RED
        ax.text(0, 1.235, f"{col}   {name}", transform=ax.transAxes, fontsize=12.2,
                color=ps.INK, fontweight="600", va="bottom")
        ax.text(0, 1.105, plain, transform=ax.transAxes, fontsize=10.4,
                color=ps.MUTED, va="bottom")
        ax.text(1.0, 1.235, badge, transform=ax.transAxes, fontsize=9.6,
                color=bcol, fontweight="600", va="bottom", ha="right")

    ps.title_block(fig, "Six columns describe every street",
                   "Five of them go into the model. The sixth is a road-class label, and "
                   "feeding label numbers to a network that adds\nand multiplies would "
                   "invent an order that does not exist.", y=0.955)
    ps.footnote(fig, ["Counts on a log scale over 31,635 links; CAPACITY_REDUCTION over "
                      "all 31,635,000 node observations with zeros omitted.",
                      "Only CAPACITY_REDUCTION changes between scenarios — everything "
                      "else about the network is fixed."], y=0.072)
    save(fig, "02_six_features")


def fig_inverted_u(X, absy):
    """The single most interesting shape in the dataset."""
    vol = X[:, 0]
    xs, ys, es, ns = [], [], [], []
    lo, hi_max, width = 0.0, float(vol.max()), 67.0
    while lo < hi_max:
        hi = lo + width
        m = (vol >= lo) & (vol < hi)
        while m.sum() < 100 and hi < hi_max:
            hi += width
            m = (vol >= lo) & (vol < hi)
        if m.sum() == 0:
            break
        xs.append(0.5 * (lo + hi)); ys.append(float(absy[m].mean()))
        es.append(float(absy[m].std() / np.sqrt(m.sum()))); ns.append(int(m.sum()))
        lo = hi

    fig = plt.figure(figsize=(13.2, 8.2))
    ax = fig.add_axes([0.075, 0.155, 0.885, 0.655])
    peak = int(np.argmax(ys))

    ax.fill_between(xs, np.array(ys) - np.array(es), np.array(ys) + np.array(es),
                    color=ps.RED_SOFT, alpha=0.35, lw=0)
    ax.plot(xs, ys, "-", color=ps.RED, lw=2.6, zorder=3)
    ax.plot(xs, ys, "o", color=ps.RED, ms=6.5, mec=ps.PAPER, mew=1.4, zorder=4)
    ax.set_xscale("log")
    ps.clean(ax, grid_axis="y")
    ax.set_xlabel("base-case car volume on the street  (veh/h, log scale)", fontsize=11)
    ax.set_ylabel("mean change in volume  (veh/h)", fontsize=11)

    ax.plot([xs[peak]], [ys[peak]], "o", color=ps.INK, ms=9, zorder=5)
    ps.note(ax, xs[peak] * 1.15, ys[peak] + 1.2,
            f"peak {ys[peak]:.0f} veh/h\naround {xs[peak]:.0f} veh/h",
            color=ps.INK, size=11.5)
    ps.note(ax, xs[-1] * 0.72, ys[-1] - 7.5,
            "the busiest roads\nbarely move", color=ps.MUTED, size=11.5, ha="center")
    ps.note(ax, xs[1] * 1.05, ys[1] - 3.4,
            "quiet streets\nhave little to give", color=ps.MUTED, size=11.5, va="top")

    ps.title_block(fig, "The busiest roads are the steady ones",
                   "How much a street's traffic changes, against how busy it already was. "
                   "Rising, then falling: the streets that swing\nhardest are the merely "
                   "busy ones, not the arteries.", y=0.955)
    ps.footnote(fig, [
        f"Equal-width bands of 67 veh/h merged rightwards until each holds at least 100 "
        f"links ({min(ns):,}–{max(ns):,} per band); shaded band is ± one standard error.",
        "Grouped coarsely the fall is unambiguous: 400–600 veh/h averages 36.1 ± 1.7 "
        "(n = 216), 900–1,600 averages 13.5 ± 0.4 (n = 306).",
        "Mean over all 1,000 scenarios of the absolute change in each link's volume."],
        y=0.088)
    save(fig, "03_inverted_u")


def fig_policy_vs_response(pos, red, y, absy, polys):
    """Where the policy lands, and where the traffic actually moves."""
    times = (red != 0).sum(0)
    fig = plt.figure(figsize=(13.6, 7.9))
    fig.patch.set_facecolor(NIGHT)
    axL = fig.add_axes([0.025, 0.115, 0.465, 0.685])
    axR = fig.add_axes([0.508, 0.115, 0.465, 0.685])

    ps.network(axL, pos, color="#1E293B", lw=0.35, background=NIGHT)
    m = times > 0
    ps.network(axL, pos[m], values=times[m], cmap=ps.HEAT, lw=0.75, background=NIGHT)
    ps.districts(axL, polys, colour="#8595AA", lw=0.8, alpha=0.60)
    ps.focus_on(axL, pos)

    ps.network(axR, pos, values=np.log10(absy + 0.05), cmap=ps.HEAT, lw=0.6,
               background=NIGHT)
    ps.districts(axR, polys, colour="#8595AA", lw=0.8, alpha=0.60, label=True,
                 label_size=6.9, label_colour="#E8EEF6", label_bg="#16233A")

    for ax, head, sub in (
        (axL, "Where the policy is applied",
         "11,305 of 31,635 links are ever eligible — primary, secondary and tertiary only"),
        (axR, "Where the traffic actually moves",
         "every link responds, including the 20,330 no policy ever touches")):
        ax.text(0, 1.075, head, transform=ax.transAxes, fontsize=15.5,
                color="#F8FAFC", fontweight="600", va="bottom")
        ax.text(0, 1.028, sub, transform=ax.transAxes, fontsize=10.6,
                color="#94A3B8", va="bottom")

    ps.gradient_key(fig, [0.025, 0.072, 0.16, 0.010], ps.HEAT, "once", "660 scenarios")
    ps.gradient_key(fig, [0.508, 0.072, 0.16, 0.010], ps.HEAT, "still", "large change")

    fig.text(0.055, 0.955, "The effect is not where the policy is",
             ha="left", va="top", fontsize=25, color="#F8FAFC", fontweight="600")
    fig.text(0.055, 0.912,
             "Capacity is only ever removed from three road classes. The response spreads "
             "across the whole network — which is why a\nsurrogate has to model the graph "
             "rather than each street on its own.",
             ha="left", va="top", fontsize=12.4, color="#94A3B8", linespacing=1.6)
    fig.text(0.025, 0.040,
             "Left: how often each link had capacity removed across the 1,000 scenarios. "
             "Right: mean absolute change in link volume, log-shaded.\n"
             "Trunk roads are never intervened and still carry the second-highest mean "
             "response of any road class.",
             ha="left", va="top", fontsize=9.8, color="#64748B", linespacing=1.7)
    save(fig, "04_policy_vs_response", face=NIGHT)


def fig_road_classes(X, red, absy):
    """Categorical, and built around the trunk-road result."""
    #: Two class names are too long to sit in the label gutter.
    SHORT = {-1: "public transport", 0: "trunk / motorway link"}
    hw = X[:, 4].astype(int)
    ever = (red != 0).any(0)
    rows = []
    for c in sorted(set(hw.tolist())):
        m = hw == c
        rows.append((c, SHORT.get(c, HIGHWAY_CLASSES.get(c, "unmapped").split(" /")[0]),
                     int(m.sum()), float(absy[m].mean()), bool(ever[m].any())))
    rows.sort(key=lambda r: -r[3])

    fig = plt.figure(figsize=(13.2, 8.0))
    ax = fig.add_axes([0.235, 0.135, 0.575, 0.665])
    ypos = np.arange(len(rows))[::-1]
    for (_c, nm, n, resp, hit), yy in zip(rows, ypos, strict=True):
        colour = ps.AMBER if hit else ps.BLUE
        ax.barh(yy, resp, color=colour, height=0.66,
                alpha=1.0 if hit else 0.82)
        ax.text(resp + 0.16, yy, f"{resp:.2f}", va="center", fontsize=10.4,
                color=ps.BODY)
        ax.text(-0.22, yy + 0.13, nm, va="center", ha="right", fontsize=11.2,
                color=ps.INK if hit else ps.BODY,
                fontweight="600" if hit else "400")
        ax.text(-0.22, yy - 0.24, f"{n:,} links", va="center", ha="right",
                fontsize=8.8, color=ps.FAINT)
    ax.set_yticks([]); ax.set_ylim(-0.9, len(rows) - 0.1)
    ps.clean(ax, left=False, grid_axis="x")
    ax.set_xlabel("mean change in volume  (veh/h)", fontsize=11)

    trunk_y = [yy for (c, *_), yy in zip(rows, ypos, strict=True) if c == 0][0]
    ax.annotate("never touched by any policy —\nall of this is spillover",
                xy=(rows[[r[0] for r in rows].index(0)][3], trunk_y),
                xytext=(6.0, trunk_y - 2.6), fontsize=11.4, color=ps.INK,
                linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=ps.MUTED, lw=1.3,
                                connectionstyle="arc3,rad=0.22"))
    ax.text(0.99, 1.045, "amber = the policy can touch it", transform=ax.transAxes,
            ha="right", fontsize=10.2, color=ps.AMBER, fontweight="600")
    ax.text(0.99, 1.005, "blue = it never does", transform=ax.transAxes,
            ha="right", fontsize=10.2, color=ps.BLUE, fontweight="600")

    ps.title_block(fig, "Traffic moves on roads the policy never touches",
                   "Mean change in volume by road class. Only primary, secondary and "
                   "tertiary roads ever have capacity removed —\nyet trunk roads respond "
                   "second most strongly of all.", y=0.955)
    ps.footnote(fig, "Averaged over 1,000 scenarios and every link of each class. "
                     "Road classes come from the OSM tag mapped in preprocessing.",
                y=0.075)
    save(fig, "05_road_classes")


def fig_model_inputs(X, red, absy):
    """What the model is given, stated exactly."""
    fig = plt.figure(figsize=(13.2, 7.4))
    ax = fig.add_axes([0, 0, 1, 1]); ps.bare(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def card(x, w, colour, wash, head, items, foot):
        ax.add_patch(Rectangle((x, 0.235), w, 0.475, transform=ax.transData,
                               facecolor=wash, edgecolor=colour, lw=1.0,
                               alpha=0.95, joinstyle="round", zorder=1))
        ax.text(x + 0.022, 0.660, head, fontsize=12.6, color=ps.INK,
                fontweight="600", va="top")
        for i, (mark, mono, plain) in enumerate(items):
            yy = 0.600 - i * 0.056
            # Segoe UI has neither the check nor the ballot X; DejaVu Sans has both,
            # so these two glyphs are rendered in it rather than substituted silently.
            ax.text(x + 0.022, yy, mark, fontsize=13, color=colour,
                    fontweight="700", va="center", family="DejaVu Sans")
            ax.text(x + 0.052, yy, mono, fontsize=11.0, color=ps.INK,
                    family="monospace", va="center")
            if plain:
                ax.text(x + 0.052, yy - 0.030, plain, fontsize=9.5,
                        color=ps.MUTED, va="center")
        ax.text(x + 0.022, 0.272, foot, fontsize=9.6, color=ps.MUTED, va="bottom")

    card(0.055, 0.30, ps.GREEN, "#F0FDF4", "Node features",
         [("✓", "VOL_BASE_CASE", None), ("✓", "CAPACITY_BASE_CASE", None),
          ("✓", "CAPACITY_REDUCTION", None), ("✓", "FREESPEED", None),
          ("✓", "LENGTH", None)],
         "five of the six columns of x")
    card(0.375, 0.24, ps.RED, "#FEF2F2", "Left out",
         [("✗", "HIGHWAY", "a road-class label, not a quantity")],
         "kept in the data, never fed to the model")
    card(0.645, 0.30, ps.BLUE, "#EFF6FF", "Also consumed",
         [("→", "pos[:, 0]", "where the street starts"),
          ("→", "pos[:, 1]", "where it ends"),
          ("→", "edge_index", "which streets meet")],
         "pos[:, 2], the midpoint, is stored but never read")

    ax.text(0.055, 0.185,
            "Five of the six node-attribute columns were used as model features; the GNN "
            "additionally consumed graph connectivity\nand the start/end link coordinates.",
            fontsize=13.2, color=ps.INK, va="top", linespacing=1.6)

    ps.title_block(fig, "What the model is actually given",
                   "Read from the trained checkpoint, not from the configuration: the "
                   "first layer is Linear(7 → 256), and PointNetConv\nappends a 2-D "
                   "relative coordinate, so seven minus two is five feature channels.",
                   y=0.955)
    ps.footnote(fig, "Sources: scripts/training/help_functions.py, "
                     "scripts/gnn/models/point_net_transf_gat.py, and the Trial 8 "
                     "checkpoint.", y=0.085)
    save(fig, "06_model_inputs")


def fig_arrondissements(pos, red, y, absy, cache, polys):
    """The policy units, and the mismatch between treatment and effect."""
    import json
    ar = np.load(cache / "arrondissement_of_link.npy")
    summ = {r["arrondissement"]: r for r in json.loads(
        (REPO / "docs/portfolio_data_story/assets/arrondissements.json")
        .read_text(encoding="utf-8"))["arrondissements"]}
    codes = sorted(c for c in summ if c > 0)
    sev = np.array([summ[c]["mean_intervention_severity_vehh"] for c in codes])
    rsp = np.array([summ[c]["mean_abs_response_vehh"] for c in codes])

    fig = plt.figure(figsize=(13.2, 8.2))
    axM = fig.add_axes([0.045, 0.125, 0.44, 0.635])
    axS = fig.add_axes([0.585, 0.175, 0.375, 0.545])

    inside = ar > 0
    ps.network(axM, pos, color="#E9EDF3", lw=0.30)
    ps.network(axM, pos[inside], values=absy[inside], cmap=ps.HEAT_LIGHT, lw=0.62,
               vmax=float(np.percentile(absy[inside], 99)))
    ps.districts(axM, polys, colour=ps.MUTED, lw=0.9, alpha=0.85, label=True,
                 label_size=7.2, label_colour=ps.INK, label_bg=ps.PAPER)
    ps.focus_on(axM, pos)
    axM.text(0, 1.045, "Response inside the city", transform=axM.transAxes,
             fontsize=13.4, color=ps.INK, fontweight="600", va="bottom")
    axM.text(0, 1.005, "27,958 of 31,635 links fall inside an arrondissement",
             transform=axM.transAxes, fontsize=10.2, color=ps.MUTED, va="bottom")

    r = float(np.corrcoef(sev, rsp)[0, 1])
    axS.scatter(sev, rsp, s=118, color=ps.BLUE, alpha=0.82, edgecolors=ps.PAPER,
                linewidths=1.5, zorder=3)
    for c, a, b in zip(codes, sev, rsp, strict=True):
        axS.annotate(str(c), (a, b), fontsize=8.6, color=ps.PAPER, ha="center",
                     va="center", zorder=4, fontweight="600")
    ps.clean(axS, grid_axis="y")
    axS.set_xlabel("mean capacity removed  (veh/h)", fontsize=10.6)
    axS.set_ylabel("mean change in volume  (veh/h)", fontsize=10.6)
    axS.tick_params(labelsize=9.4)
    axS.text(0, 1.045, "Treatment does not predict effect", transform=axS.transAxes,
             fontsize=13.4, color=ps.INK, fontweight="600", va="bottom")
    axS.text(0, 1.005, f"each point is one arrondissement · correlation r = {r:+.2f}",
             transform=axS.transAxes, fontsize=10.2, color=ps.MUTED, va="bottom")

    top_s = codes[int(np.argmax(sev))]
    top_r = codes[int(np.argmax(rsp))]
    ps.note(axS, sev.max() * 0.99, rsp[codes.index(top_s)] - 0.30,
            f"{top_s}: most heavily treated", ha="right", va="top", size=10.4)
    ps.note(axS, sev[codes.index(top_r)] * 1.02, rsp.max() - 0.16,
            f"{top_r}: largest response", va="top", size=10.4)

    ps.title_block(fig, "Paris decides by arrondissement; traffic does not",
                   "Interventions are drawn per district, so treatment is uneven across "
                   "the city. Where the traffic ends up moving is\nalmost unrelated to "
                   "where the capacity was taken away.", y=0.955)
    ps.footnote(fig, ["Severity is the mean capacity reduction on links that were "
                      "intervened; response is the mean absolute change in link volume, "
                      "including untouched links.",
                      "District 17 absorbs 12.0% of the network's total response while "
                      "holding 6.9% of its links."], y=0.075)
    save(fig, "07_arrondissements")


def main() -> int:
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    ps.apply()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    absy = np.abs(y).mean(0)
    polys = ps.load_districts(REPO)
    print(f"corpus {y.shape[0]:,} scenarios x {y.shape[1]:,} links\n")
    fig_hero(pos, X, polys)
    fig_features(X, red, absy)
    fig_inverted_u(X, absy)
    fig_policy_vs_response(pos, red, y, absy, polys)
    fig_road_classes(X, red, absy)
    fig_model_inputs(X, red, absy)
    fig_arrondissements(pos, red, y, absy, args.cache, polys)
    print(f"\nfigures written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
