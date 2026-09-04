#!/usr/bin/env python
"""Figures for the deep data exploration: the six x columns, the target and the graph.

Every number drawn here comes from the published corpus. HIGHWAY is drawn as a
categorical chart throughout; treating its codes as a continuous axis would imply
an ordering the encoding does not have.

    python scripts/figure_generation/generate_data_exploration_figures.py \
        --corpus DIR --cache DIR

Output: docs/figures/data_exploration/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "data_exploration"))
from common import FEATURES, MODEL_COLS, add_common_args, load  # noqa: E402
from thesis_style import COLORS  # noqa: E402

OUT = REPO / "docs" / "figures" / "data_exploration"
N_NODES_STORED = 31_559

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 190, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "legend.frameon": False,
})


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png -- {caption}")


def aspect(lat):
    return 1.0 / np.cos(np.deg2rad(float(lat)))


def bare_map(ax, lat_mean):
    """Strip a panel down to the geometry: no ticks, no grid, no frame.

    Longitude and latitude tick labels tell a reader nothing about Paris that the
    shape of the network does not already say, and the frame competes with it.
    """
    ax.set_aspect(aspect(lat_mean))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_six_features(X, red, absy):
    """One row per feature: distribution on the left, response relationship right."""
    fig, axes = plt.subplots(6, 2, figsize=(13.0, 19.0))
    hw = X[:, 4].astype(int)
    for row, name in enumerate(FEATURES):
        used = row in MODEL_COLS
        ax, ax2 = axes[row]
        tag = "model input" if used else "NOT a model input"
        col = COLORS["blue"] if used else COLORS["coral"]

        if name == "HIGHWAY":
            codes = sorted(set(hw.tolist()))
            counts = [int((hw == c).sum()) for c in codes]
            resp = [float(absy[hw == c].mean()) for c in codes]
            xs = np.arange(len(codes))
            ax.bar(xs, counts, color=col, width=0.68)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(c) for c in codes], fontsize=8)
            ax.set_yscale("log")
            ax.set_xlabel("HIGHWAY code (nominal label, not a quantity)")
            ax.set_ylabel("links (log)")
            ax2.bar(xs, resp, color=COLORS["purple"], width=0.68)
            ax2.set_xticks(xs)
            ax2.set_xticklabels([str(c) for c in codes], fontsize=8)
            ax2.set_xlabel("HIGHWAY code")
            ax2.set_ylabel("mean |response| (veh/h)")
            ax2.set_title("Mean response by class, not by code order",
                          fontweight="600", color=COLORS["dgray"])
        else:
            v = red.ravel() if name == "CAPACITY_REDUCTION" else X[:, row]
            plot_v = v[v != 0] if name == "CAPACITY_REDUCTION" else v
            ax.hist(plot_v, bins=70, color=col, log=True)
            ax.set_xlabel(f"{name}")
            ax.set_ylabel("count (log)")
            fv = np.abs(red).mean(0) if name == "CAPACITY_REDUCTION" else X[:, row]
            pos = fv > 0
            if pos.sum() > 200:
                edges = np.unique(np.percentile(fv[pos], np.linspace(0, 100, 13)))
                idx = np.clip(np.digitize(fv, edges) - 1, 0, edges.size - 2)
                cx, cy = [], []
                for b in range(edges.size - 1):
                    m = (idx == b) & pos
                    if m.sum() >= 5:
                        cx.append(0.5 * (edges[b] + edges[b + 1]))
                        cy.append(float(absy[m].mean()))
                ax2.plot(cx, cy, "o-", color=COLORS["purple"], ms=4.5, lw=1.7)
                ax2.set_xlabel(f"{name} (quantile bin centre)")
                ax2.set_ylabel("mean |response| (veh/h)")
                mono = all(b >= a - 1e-9 for a, b in zip(cy[:-1], cy[1:], strict=True))
                ax2.set_title("monotone rise" if mono else "not monotone",
                              fontweight="600", color=COLORS["dgray"])
        ax.set_title(f"{row}  {name}   [{tag}]", fontweight="600",
                     color=COLORS["dgray"] if used else COLORS["coral"])
    fig.suptitle("The six columns of x: what is in each, and how it relates to the response",
                 fontsize=13.5, fontweight="600", color=COLORS["dgray"], y=0.999)
    fig.text(0.5, 0.005,
             "Left: distribution over 31,635 links (CAPACITY_REDUCTION over all 31,635,000 "
             "node observations, zeros omitted). Right: mean |response| per quantile bin.\n"
             "HIGHWAY is drawn categorically because its integers are labels for road "
             "classes and carry no order.",
             ha="center", fontsize=8.4, color=COLORS["mgray"])
    fig.tight_layout(rect=(0, 0.012, 1, 0.995))
    save(fig, "01_six_x_features", "distribution and response relationship per feature")


def fig_volume(X, red, y, pos, absy):
    """VOL_BASE_CASE in depth: an inverted U, peaking near 500 veh/h."""
    vol = X[:, 0]
    ever = (red != 0).any(0)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.4))

    ax = axes[0, 0]
    ax.hist(vol[vol > 0], bins=80, color=COLORS["blue"], log=True)
    ax.set_xlabel("VOL_BASE_CASE (veh/h)"); ax.set_ylabel("links (log)")
    ax.set_title(f"Base-case car volume\n{100*(vol==0).mean():.1f}% of links carry no car "
                 f"traffic at all", fontweight="600", color=COLORS["dgray"])

    ax = axes[0, 1]
    edges = np.unique(np.percentile(vol[vol > 0], np.linspace(0, 100, 13)))
    idx = np.clip(np.digitize(vol, edges) - 1, 0, edges.size - 2)
    cx, ab, rel = [], [], []
    for b in range(edges.size - 1):
        m = (idx == b) & (vol > 0)
        if m.sum() < 5:
            continue
        cx.append(0.5 * (edges[b] + edges[b + 1]))
        ab.append(float(absy[m].mean()))
        rel.append(float((absy[m] / vol[m]).mean()))
    # Quantile bins cannot resolve the busy tail: the top one spans 167-1,596 veh/h.
    # Equal-width bands, merged rightwards to at least 100 links, can.
    fine_x, fine_y, fine_e = [], [], []
    lo, hi_max, width = 0.0, float(vol.max()), 67.0
    while lo < hi_max:
        hi = lo + width
        m = (vol >= lo) & (vol < hi)
        while m.sum() < 100 and hi < hi_max:
            hi += width
            m = (vol >= lo) & (vol < hi)
        if m.sum() == 0:
            break
        fine_x.append(0.5 * (lo + hi)); fine_y.append(float(absy[m].mean()))
        fine_e.append(float(absy[m].std() / np.sqrt(m.sum())))
        lo = hi
    ax.errorbar(fine_x, fine_y, yerr=fine_e, fmt="o-", color=COLORS["coral"], ms=5,
                lw=2, capsize=3, label="mean |response| (bands of >=100 links)")
    ax.plot(cx, ab, "s--", color=COLORS["lgray"], ms=4, lw=1.4,
            label="12 quantile bins (hides the fall)")
    peak = int(np.argmax(fine_y))
    ax.axvline(fine_x[peak], color=COLORS["slate"], ls=":", lw=1.2)
    ax.annotate(f"peak {fine_y[peak]:.0f} veh/h", (fine_x[peak], fine_y[peak]),
                textcoords="offset points", xytext=(10, 6), fontsize=8.5,
                color=COLORS["slate"])
    ax.set_xscale("log"); ax.set_xlabel("VOL_BASE_CASE (veh/h, log)")
    ax.set_ylabel("mean |response| (veh/h)")
    ax.set_title("An inverted U: response peaks near 500 veh/h, then falls\n"
                 "the busiest roads are the steady ones",
                 fontweight="600", color=COLORS["dgray"])
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 0]
    o = np.argsort(vol)
    sc = ax.scatter(pos[o, 2, 0], pos[o, 2, 1], s=0.6, c=np.log10(vol[o] + 1),
                    cmap="viridis", linewidths=0)
    bare_map(ax, pos[:, 2, 1].mean())
    plt.colorbar(sc, ax=ax, fraction=0.037, pad=0.02, label="log10(volume + 1)")
    ax.set_title("Where the traffic is", fontweight="600", color=COLORS["dgray"])

    ax = axes[1, 1]
    for m, lbl, c in ((~ever, "never intervened", COLORS["blue"]),
                      (ever, "intervened at least once", COLORS["coral"])):
        cy = []
        for b in range(edges.size - 1):
            s = (idx == b) & (vol > 0) & m
            cy.append(float(absy[s].mean()) if s.sum() >= 5 else np.nan)
        ax.plot(cx, cy, "o-", color=c, ms=4.5, lw=1.8, label=lbl)
    ax.set_xscale("log"); ax.set_xlabel("VOL_BASE_CASE (veh/h, log)")
    ax.set_ylabel("mean |response| (veh/h)"); ax.legend(fontsize=8.5)
    ax.set_title("Links never touched by any policy still respond\n"
                 "across most of the volume range they respond as much as touched links",
                 fontweight="600", color=COLORS["dgray"])
    fig.suptitle("VOL_BASE_CASE: the strongest single predictor of where traffic moves",
                 fontsize=13, fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.035,
             "Top right: equal-width bands merged rightwards to at least 100 links, with "
             "standard errors. The dashed line is the 12-quantile view, whose top bin spans "
             "167-1,596 veh/h and averages the fall away.\n"
             "Bottom right uses those quantile bins. Response is the mean |change in link "
             "volume| over all 1,000 scenarios; Spearman rho against base volume is +0.885 "
             "over 31,635 links.",
             ha="center", fontsize=8.4, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "02_volume_deep_dive", "base volume against response")


def fig_intervention(X, red, pos, absy):
    """CAPACITY_REDUCTION: the experimental knob."""
    ever = (red != 0).any(0)
    times = (red != 0).sum(0)
    per_scen = (red != 0).sum(1)
    nz = red[red != 0]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.0))

    ax = axes[0, 0]
    vals, counts = np.unique(np.round(nz, 3), return_counts=True)
    ax.bar(np.arange(vals.size), counts, color=COLORS["coral"], width=0.7)
    ax.set_xticks(np.arange(vals.size))
    ax.set_xticklabels([f"{v:.0f}" for v in vals], rotation=90, fontsize=6.6)
    ax.set_yscale("log")
    ax.set_xlabel("capacity removed (veh/h)"); ax.set_ylabel("occurrences (log)")
    ax.set_title(f"{vals.size} distinct intervention magnitudes\n"
                 f"all negative: capacity is only ever removed",
                 fontweight="600", color=COLORS["dgray"])

    ax = axes[0, 1]
    ax.hist(per_scen, bins=60, color=COLORS["blue"])
    ax.axvline(np.median(per_scen), color=COLORS["slate"], ls="--", lw=1.2)
    ax.set_xlabel("links intervened in a scenario"); ax.set_ylabel("scenarios")
    ax.set_title(f"Footprint per scenario\nmedian {int(np.median(per_scen)):,}, "
                 f"range {per_scen.min():,}-{per_scen.max():,}",
                 fontweight="600", color=COLORS["dgray"])

    ax = axes[1, 0]
    ax.scatter(pos[~ever, 2, 0], pos[~ever, 2, 1], s=0.35, c="#e2e8f0", linewidths=0)
    o = np.argsort(times)
    m = o[times[o] > 0]
    sc = ax.scatter(pos[m, 2, 0], pos[m, 2, 1], s=1.4, c=times[m], cmap="OrRd", linewidths=0)
    bare_map(ax, pos[:, 2, 1].mean())
    plt.colorbar(sc, ax=ax, fraction=0.037, pad=0.02, label="scenarios intervened")
    ax.set_title(f"How often each link was intervened\n"
                 f"{int(ever.sum()):,} of {ever.size:,} links are ever eligible",
                 fontweight="600", color=COLORS["dgray"])

    ax = axes[1, 1]
    sev = np.abs(red).sum(0) / np.maximum(times, 1)
    m = ever
    ax.scatter(sev[m], absy[m], s=1.6, c=COLORS["purple"], alpha=0.28, linewidths=0)
    ax.set_xlabel("mean capacity removed when intervened (veh/h)")
    ax.set_ylabel("mean |response| (veh/h)")
    ax.set_yscale("log"); ax.set_xscale("log")
    r = np.corrcoef(np.log10(sev[m] + 1), np.log10(absy[m] + 1e-3))[0, 1]
    ax.set_title(f"Severity does not determine response\n"
                 f"Pearson r on log scales = {r:+.2f}",
                 fontweight="600", color=COLORS["dgray"])
    fig.suptitle("CAPACITY_REDUCTION: the only column that changes between scenarios",
                 fontsize=13, fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.035,
             "The policy only ever removes capacity, and only on primary, secondary and "
             "tertiary roads. 87.9% of all node observations have no intervention.",
             ha="center", fontsize=8.4, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "03_intervention", "the experimental knob")


def fig_network_maps(X, pos, absy):
    """The four static continuous features drawn on the real geometry."""
    panels = [
        ("VOL_BASE_CASE", X[:, 0], "viridis", True, "veh/h"),
        ("CAPACITY_BASE_CASE", X[:, 1], "cividis", False, "veh/h"),
        ("FREESPEED", X[:, 3], "magma", False, "m/s"),
        ("LENGTH", X[:, 5], "YlGnBu", True, "m"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.6))
    for ax, (name, v, cmap, logscale, unit) in zip(axes.ravel(), panels, strict=True):
        c = np.log10(v + 1) if logscale else v
        o = np.argsort(c)
        sc = ax.scatter(pos[o, 2, 0], pos[o, 2, 1], s=0.6, c=c[o], cmap=cmap, linewidths=0)
        bare_map(ax, pos[:, 2, 1].mean())
        plt.colorbar(sc, ax=ax, fraction=0.037, pad=0.02,
                     label=f"log10({unit}+1)" if logscale else unit)
        ax.set_title(f"{name}", fontweight="600", color=COLORS["dgray"])
    fig.suptitle("The static features on the real network geometry",
                 fontsize=13, fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.02,
             "Each point is one road link at its stored midpoint pos[:, 2]. Coordinates are "
             "WGS84 longitude and latitude.",
             ha="center", fontsize=8.4, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "04_network_feature_maps", "static features in space")


def fig_target(y, pos, absy):
    """The target: what the model is asked to predict."""
    flat = y.ravel()
    mean_signed = y.mean(0)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.2))

    ax = axes[0, 0]
    ax.hist(flat[np.abs(flat) > 0], bins=200, range=(-60, 60), color=COLORS["green"], log=True)
    ax.set_xlabel("y = change in link car volume (veh/h)"); ax.set_ylabel("count (log)")
    ax.set_title(f"Target distribution\nmean {flat.mean():+.4f}, "
                 f"{100*(flat==0).mean():.1f}% exactly zero, heavy tails both ways",
                 fontweight="600", color=COLORS["dgray"])

    ax = axes[0, 1]
    pos_share = (y > 0).mean(0); neg_share = (y < 0).mean(0)
    ax.hist(pos_share[(pos_share + neg_share) > 0], bins=60, color=COLORS["blue"], alpha=0.85)
    ax.set_xlabel("share of scenarios where the link gains traffic")
    ax.set_ylabel("links")
    ax.set_title(f"Gains and losses are balanced\n"
                 f"{100*(flat>0).mean():.1f}% of observations positive, "
                 f"{100*(flat<0).mean():.1f}% negative",
                 fontweight="600", color=COLORS["dgray"])

    ax = axes[1, 0]
    lim = np.percentile(np.abs(mean_signed), 99)
    o = np.argsort(np.abs(mean_signed))
    sc = ax.scatter(pos[o, 2, 0], pos[o, 2, 1], s=0.7, c=np.clip(mean_signed[o], -lim, lim),
                    cmap="coolwarm", linewidths=0, vmin=-lim, vmax=lim)
    bare_map(ax, pos[:, 2, 1].mean())
    plt.colorbar(sc, ax=ax, fraction=0.037, pad=0.02, label="mean signed y (veh/h)")
    ax.set_title("Who gains and who loses, averaged over 1,000 scenarios",
                 fontweight="600", color=COLORS["dgray"])

    ax = axes[1, 1]
    o = np.argsort(absy)
    sc = ax.scatter(pos[o, 2, 0], pos[o, 2, 1], s=0.7, c=np.log10(absy[o] + 0.01),
                    cmap="inferno", linewidths=0)
    bare_map(ax, pos[:, 2, 1].mean())
    plt.colorbar(sc, ax=ax, fraction=0.037, pad=0.02, label="log10(mean |y| + 0.01)")
    ax.set_title("Where the network responds at all", fontweight="600", color=COLORS["dgray"])
    fig.suptitle("y: the change in car volume the policy caused on each link",
                 fontsize=13, fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.03,
             "y = vol_car(scenario) - vol_car(base case), computed per link in "
             "compute_target_tensor_only_edge_features. The signed map is near-symmetric "
             "because capacity removed in one place\npushes traffic elsewhere: gains and "
             "losses largely cancel across the network.",
             ha="center", fontsize=8.4, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "05_target", "the target in distribution and space")


def fig_graph(X, ei, absy, pos):
    """Line-graph topology."""
    n = X.shape[0]
    deg = (np.bincount(ei[0], minlength=n) + np.bincount(ei[1], minlength=n))
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8))

    ax = axes[0]
    ds = np.arange(deg.max() + 1)
    cnt = [int((deg == d).sum()) for d in ds]
    ax.bar(ds, cnt, color=COLORS["blue"], width=0.7)
    ax.set_xlabel("degree (in + out)"); ax.set_ylabel("links")
    ax.set_title(f"Degree distribution\nmedian {int(np.median(deg))}, max {deg.max()}",
                 fontweight="600", color=COLORS["dgray"])

    ax = axes[1]
    ds2 = [d for d in ds if (deg == d).sum() >= 5]
    rs = [float(absy[deg == d].mean()) for d in ds2]
    ax.plot(ds2, rs, "o-", color=COLORS["purple"], ms=5, lw=1.8)
    ax.set_xlabel("degree"); ax.set_ylabel("mean |response| (veh/h)")
    ax.set_title("Being connected matters; how connected does not\n"
                 "flat from degree 2 upward (3.73-4.52 veh/h)",
                 fontweight="600", color=COLORS["dgray"])

    ax = axes[2]
    iso = deg == 0
    ax.scatter(pos[~iso, 2, 0], pos[~iso, 2, 1], s=0.35, c="#dbe3ea", linewidths=0)
    ax.scatter(pos[iso, 2, 0], pos[iso, 2, 1], s=14, c=COLORS["coral"], linewidths=0,
               label=f"{int(iso.sum())} isolated links")
    bare_map(ax, pos[:, 2, 1].mean())
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title("The isolated public-transport links",
                 fontweight="600", color=COLORS["dgray"])
    fig.suptitle("Line-graph topology: 31,635 road links, 59,851 directed edges",
                 fontsize=13, fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.09,
             "An edge means two road links meet at an intersection. 121 weakly connected "
             "components, the largest holding 92.7% of links.\nThe 76 isolated links carry "
             "no car mode: zero volume, capacity and freespeed, and a target of exactly zero "
             "in every scenario.",
             ha="center", fontsize=8.4, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "06_graph_topology", "degree, response and isolated links")


def main() -> int:
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    absy = np.abs(y).mean(0)
    print(f"corpus {y.shape[0]:,} scenarios x {y.shape[1]:,} links\n")
    fig_six_features(X, red, absy)
    fig_volume(X, red, y, pos, absy)
    fig_intervention(X, red, pos, absy)
    fig_network_maps(X, pos, absy)
    fig_target(y, pos, absy)
    fig_graph(X, ei, absy, pos)
    print(f"\nfigures written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
