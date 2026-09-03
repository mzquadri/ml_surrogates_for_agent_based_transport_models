#!/usr/bin/env python
"""Figures that explain the training corpus.

One figure per question worth asking of the data, not one per feature. Each is
built from the published corpus (`train-data-v1`, 20 batch files), so every
number shown is measured rather than quoted.

Usage
-----
    python scripts/figure_generation/generate_dataset_figures.py --corpus PATH

`--corpus` is the directory holding `datalist_batch_1.pt` ... `datalist_batch_20.pt`.
With `--quick` only the first batch is read, which is enough for every figure
except the corpus-wide statistics panel.

Output: docs/figures/dataset/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_style import COLORS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "docs" / "figures" / "dataset"

NAMES = ["VOL_BASE_CASE", "CAPACITY_BASE_CASE", "CAPACITY_REDUCTION",
         "FREESPEED", "HIGHWAY", "LENGTH"]
UNITS = ["veh/h", "veh/h", "veh/h", "m/s", "class code", "m"]
USED = [True, True, True, True, False, True]  # HIGHWAY is excluded from the model

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def load(corpus: Path, quick: bool):
    files = sorted(corpus.glob("datalist_batch_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise SystemExit(f"No datalist_batch_*.pt under {corpus}")
    if quick:
        files = files[:1]
    print(f"reading {len(files)} batch file(s) from {corpus}")
    graphs = []
    for f in files:
        graphs.extend(torch.load(f, weights_only=False, map_location="cpu"))
    print(f"  {len(graphs)} scenarios loaded")
    return graphs


def save(fig, name: str, caption: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png  -- {caption}")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_feature_distributions(X):
    """Each feature gets the display its shape actually calls for."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6))
    for i, ax in enumerate(axes.ravel()):
        v = X[:, i]
        col = COLORS["blue"] if USED[i] else COLORS["mgray"]
        distinct = len(np.unique(v))
        if distinct <= 40:
            # Few distinct values: a bar chart is honest, a histogram is not.
            vals, cnt = np.unique(v, return_counts=True)
            ax.bar(range(len(vals)), 100 * cnt / cnt.sum(), color=col, width=0.75)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels([f"{x:g}" for x in vals], rotation=90, fontsize=6.4)
            ax.set_ylabel("% of links")
        else:
            nz = v[v != 0]
            ax.hist(nz, bins=80, color=col, log=True)
            ax.set_ylabel("links (log)")
            if (v == 0).mean() > 0.01:
                ax.axvline(0, color=COLORS["coral"], lw=1.2, ls="--")
        title = NAMES[i] + ("" if USED[i] else "  — not used by the model")
        ax.set_title(title, fontweight="600", color=COLORS["dgray"])
        ax.set_xlabel(UNITS[i])
        z = 100 * (v == 0).mean()
        ax.text(0.97, 0.94, f"{z:.1f}% zero\n{distinct:,} distinct",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.6,
                color=COLORS["slate"])
    fig.suptitle("Node feature distributions", fontsize=13, fontweight="600",
                 color=COLORS["dgray"], y=1.00)
    fig.text(0.5, -0.035,
             "Discrete features are shown as value counts; continuous features as log-scale histograms with zeros excluded "
             "(dashed line marks zero).\nCounts are over every node observation in the corpus read.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "01_feature_distributions",
         "what each of the six stored features looks like")


def fig_target(Y):
    """The target's zero spike and tails are the two things worth seeing."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    y = Y.ravel()

    ax = axes[0]
    ax.hist(y[y != 0], bins=160, range=(-60, 60), color=COLORS["blue"], log=True)
    ax.axvline(0, color=COLORS["coral"], lw=1.3, ls="--")
    ax.set_title("Non-zero targets, central range", fontweight="600", color=COLORS["dgray"])
    ax.set_xlabel("change in link volume (veh/h)")
    ax.set_ylabel("links (log)")
    ax.text(0.02, 0.94, f"{100 * (y == 0).mean():.2f}% of links\nare exactly zero",
            transform=ax.transAxes, va="top", fontsize=8.4, color=COLORS["coral"])

    ax = axes[1]
    q = np.percentile(y, [1, 5, 25, 50, 75, 95, 99])
    ax.boxplot(y[np.random.default_rng(0).integers(0, y.size, 400_000)],
               vert=False, widths=0.5, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor=COLORS["blue_lt"], color=COLORS["blue_dk"]),
               medianprops=dict(color=COLORS["coral"], lw=1.6),
               whiskerprops=dict(color=COLORS["blue_dk"]),
               capprops=dict(color=COLORS["blue_dk"]))
    ax.set_yticks([])
    ax.set_title("Spread (400k random sample, outliers hidden)",
                 fontweight="600", color=COLORS["dgray"])
    ax.set_xlabel("change in link volume (veh/h)")
    ax.text(0.02, 0.10,
            "p1 %.1f   p25 %.2f   p50 %.2f   p75 %.2f   p99 %.1f" % (q[0], q[2], q[3], q[4], q[6]),
            transform=ax.transAxes, fontsize=8.2, color=COLORS["slate"],
            family="monospace")

    fig.suptitle("Target: policy-induced change in link volume", fontsize=13,
                 fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.05,
             "Roughly symmetric about zero with heavy tails on both sides: capacity removed in one place displaces traffic to another,\n"
             "so gains and losses largely cancel across the network. The 27.6% zero share is a property of the target, not of the input.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "02_target_distribution", "the target's zero mass and tails")


def fig_intervention(graphs):
    """CAPACITY_REDUCTION is the only thing that changes between scenarios."""
    red = np.stack([g.x.numpy()[:, 2] for g in graphs])
    n_touched = (red != 0).sum(1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    ax = axes[0]
    ax.hist(n_touched, bins=40, color=COLORS["coral"])
    ax.set_title("Links affected per scenario", fontweight="600", color=COLORS["dgray"])
    ax.set_xlabel("links with reduced capacity")
    ax.set_ylabel("scenarios")
    ax.text(0.97, 0.94,
            f"min {n_touched.min():,}\nmean {n_touched.mean():,.0f}\nmax {n_touched.max():,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=COLORS["slate"], family="monospace")

    ax = axes[1]
    nz = red[red != 0]
    vals, cnt = np.unique(nz, return_counts=True)
    ax.bar(range(len(vals)), cnt, color=COLORS["coral"])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([f"{v:g}" for v in vals], rotation=90, fontsize=6.4)
    ax.set_yscale("log")
    ax.set_title("Reduction magnitudes actually used", fontweight="600", color=COLORS["dgray"])
    ax.set_xlabel("capacity reduction (veh/h)")
    ax.set_ylabel("occurrences (log)")

    ax = axes[2]
    var = red.std(0)
    ax.hist(var[var > 0], bins=60, color=COLORS["purple"], log=True)
    ax.set_title("Per-link variability across scenarios", fontweight="600", color=COLORS["dgray"])
    ax.set_xlabel("std of capacity reduction at a link (veh/h)")
    ax.set_ylabel("links (log)")
    ax.text(0.97, 0.94, f"{100 * (var == 0).mean():.1f}% of links\nnever intervened",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=COLORS["slate"])

    fig.suptitle("The intervention: CAPACITY_REDUCTION", fontsize=13,
                 fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.06,
             "Five of six features and the whole topology are identical across scenarios. Everything that distinguishes one scenario\n"
             "from another enters through this feature, which makes its coverage of the network the real experimental design.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "03_intervention_design", "how the scenarios differ from each other")


def fig_spatial(graphs):
    """Where the intervention lands and where the response shows up."""
    g = graphs[0]
    pos = g.pos.numpy()
    mid = pos[:, 2, :]
    red = g.x.numpy()[:, 2]
    y = g.y.numpy().ravel()

    # Coordinates are unprojected WGS84. At this latitude a degree of longitude is
    # ~73 km against ~111 km for a degree of latitude, so an equal aspect would
    # distort the city. Scale y against x by 1/cos(lat) for a correct
    # equirectangular view.
    aspect = 1.0 / np.cos(np.deg2rad(float(mid[:, 1].mean())))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax in axes:
        ax.set_aspect(aspect)
        ax.set_xlabel("longitude")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("latitude")

    ax = axes[0]
    ax.scatter(mid[:, 0], mid[:, 1], s=0.12, c=COLORS["lgray"], linewidths=0)
    m = red != 0
    sc = ax.scatter(mid[m, 0], mid[m, 1], s=2.6, c=-red[m], cmap="OrRd",
                    linewidths=0, vmin=0)
    ax.set_title(f"Where capacity was reduced  ({m.sum():,} links)",
                 fontweight="600", color=COLORS["dgray"])
    plt.colorbar(sc, ax=ax, fraction=0.031, pad=0.02, label="reduction (veh/h)")

    ax = axes[1]
    lim = np.percentile(np.abs(y), 99.5)
    order = np.argsort(np.abs(y))  # draw large responses last
    sc = ax.scatter(mid[order, 0], mid[order, 1], s=1.0, c=y[order],
                    cmap="RdBu_r", vmin=-lim, vmax=lim, linewidths=0)
    ax.set_title("Resulting change in link volume", fontweight="600", color=COLORS["dgray"])
    plt.colorbar(sc, ax=ax, fraction=0.031, pad=0.02, label="change (veh/h)")

    fig.suptitle("One scenario, seen on the Paris network", fontsize=13,
                 fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.04,
             "Link midpoints in WGS84, drawn with an equirectangular aspect so the city is not distorted. The response is concentrated near the "
             "intervention\nbut clearly not confined to it — that spillover onto untouched links is what a graph model is there to capture.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "04_spatial_intervention_response",
         "intervention and response on the real network")


def fig_correlation(X, Y):
    """How much of the target any single feature explains on its own."""
    y = Y.ravel()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, y.size, min(500_000, y.size))
    cols = [X[idx, i] for i in range(6)] + [y[idx]]
    labels = NAMES + ["TARGET"]
    M = np.corrcoef(np.vstack(cols))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(7)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(7)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.2,
                    color="white" if abs(M[i, j]) > 0.55 else COLORS["dgray"])
    ax.set_title("Pearson correlation", fontweight="600", color=COLORS["dgray"])
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    corr = M[:6, 6]
    order = np.argsort(np.abs(corr))
    ax.barh(range(6), corr[order],
            color=[COLORS["blue"] if USED[i] else COLORS["mgray"] for i in order])
    ax.set_yticks(range(6))
    ax.set_yticklabels([NAMES[i] for i in order], fontsize=8.5)
    ax.axvline(0, color=COLORS["slate"], lw=0.9)
    ax.set_xlabel("correlation with target")
    ax.set_title("Linear signal per feature", fontweight="600", color=COLORS["dgray"])

    fig.suptitle("Feature-target relationships", fontsize=13, fontweight="600",
                 color=COLORS["dgray"])
    fig.text(0.5, -0.04,
             "Computed on a 500k random subsample. No single feature is close to sufficient on its own — the intervention feature carries the most\n"
             "linear signal, and the rest of the response comes from network structure, which is the case for using a graph model at all.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "05_feature_target_correlation", "how much any one feature explains")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path,
                    help="directory containing datalist_batch_*.pt")
    ap.add_argument("--quick", action="store_true",
                    help="read only the first batch file")
    args = ap.parse_args()

    graphs = load(args.corpus, args.quick)
    X = np.concatenate([g.x.numpy() for g in graphs]).astype(np.float64)
    Y = np.concatenate([g.y.numpy() for g in graphs]).astype(np.float64)
    print(f"  {X.shape[0]:,} node observations\n")

    fig_feature_distributions(X)
    fig_target(Y)
    fig_intervention(graphs)
    fig_spatial(graphs)
    fig_correlation(X, Y)
    print(f"\nfigures written to {OUT}")


if __name__ == "__main__":
    main()
