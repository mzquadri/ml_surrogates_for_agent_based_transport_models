#!/usr/bin/env python
"""Maps of the real network against the 20 Paris arrondissements.

The polygons are administrative boundaries; the road geometry comes from `pos`.
Both are drawn together because the experiment was designed per arrondissement.

Usage:
    python scripts/figure_generation/generate_arrondissement_figures.py \
        --corpus DIR --cache DIR

Output: docs/figures/geography/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "data_exploration"))
from common import HIGHWAY_CLASSES, add_common_args, load  # noqa: E402
from thesis_style import COLORS  # noqa: E402

OUT = REPO / "docs" / "figures" / "geography"
DISTRICTS = REPO / "data" / "visualisation" / "districts_paris.geojson"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False,
})


def load_polys():
    gj = json.loads(DISTRICTS.read_text(encoding="utf-8"))
    out = []
    for f in gj["features"]:
        c = int(f["properties"]["c_ar"])
        g = f["geometry"]
        rings = g["coordinates"] if g["type"] == "Polygon" else \
            [r for poly in g["coordinates"] for r in poly]
        out.append((c, [np.asarray(r) for r in rings]))
    return out


def draw_boundaries(ax, polys, lw=0.7, color="#64748b", label=True):
    for c, rings in polys:
        for r in rings:
            ax.plot(r[:, 0], r[:, 1], color=color, lw=lw, zorder=5, alpha=0.85)
        if label:
            cx, cy = rings[0][:, 0].mean(), rings[0][:, 1].mean()
            ax.text(cx, cy, str(c), fontsize=6.5, ha="center", va="center",
                    color="#0f172a", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.16", fc="white",
                              ec="#cbd5e1", lw=0.4, alpha=0.85))


def aspect(lat):
    return 1.0 / np.cos(np.deg2rad(float(lat)))


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png -- {caption}")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    mid = pos[:, 2, :]
    polys = load_polys()
    A = aspect(mid[:, 1].mean())
    absY = np.abs(y).mean(0)
    ar = np.load(args.cache / "arrondissement_of_link.npy")
    summ = {r["arrondissement"]: r for r in json.loads(
        (REPO / "docs/portfolio_data_story/assets/arrondissements.json")
        .read_text(encoding="utf-8"))["arrondissements"]}

    # ---- 1. the network on the districts -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))
    for ax in axes:
        ax.set_aspect(A); ax.set_xlabel("longitude")
    axes[0].set_ylabel("latitude")

    ax = axes[0]
    ax.scatter(mid[:, 0], mid[:, 1], s=0.10, c=COLORS["lgray"], linewidths=0)
    inside = ar > 0
    ax.scatter(mid[inside, 0], mid[inside, 1], s=0.10, c=COLORS["blue"], linewidths=0)
    draw_boundaries(ax, polys)
    ax.set_title(f"Modelled network vs city boundary\n"
                 f"{inside.sum():,} links inside ({100*inside.mean():.1f}%), "
                 f"{(~inside).sum():,} outside",
                 fontweight="600", color=COLORS["dgray"])

    ax = axes[1]
    hw = X[:, 4].astype(int)
    # Trunk is drawn because it is the other major-road class, but no scenario ever
    # touches it. The legend says so per class rather than leaving the reader to infer
    # from the panel title that one of the four colours behaves differently.
    ever = (red != 0).any(0)
    pal = {1: COLORS["coral"], 2: COLORS["amber"], 3: COLORS["green"], 0: COLORS["purple"]}
    ax.scatter(mid[:, 0], mid[:, 1], s=0.10, c="#e2e8f0", linewidths=0)
    for code, col in pal.items():
        s = hw == code
        n_ev = int(ever[s].sum())
        tag = f"{n_ev:,} intervened" if n_ev else "never intervened"
        ax.scatter(mid[s, 0], mid[s, 1], s=0.9, c=col, linewidths=0,
                   label=f"{HIGHWAY_CLASSES[code].split(' /')[0]} "
                         f"({s.sum():,} links, {tag})")
    draw_boundaries(ax, polys, label=False)
    ax.legend(fontsize=7.5, markerscale=6, loc="lower left")
    ax.set_title("Road classes the policy can touch\n"
                 f"only primary, secondary and tertiary are ever intervened "
                 f"({int(ever.sum()):,} of {ever.size:,} links)",
                 fontweight="600", color=COLORS["dgray"])
    fig.suptitle("The network and the twenty arrondissements", fontsize=13,
                 fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.03, "Arrondissement polygons are administrative boundaries, not the road network. "
             "Link positions are the midpoints stored in pos[:, 2].",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "01_network_and_districts", "network extent against the city boundary")

    # ---- 2. choropleths ------------------------------------------------------
    metrics = [
        ("share_links_ever_intervened", "Share of links ever intervened", "Blues", 100),
        ("mean_intervention_severity_vehh", "Mean intervention severity (veh/h)", "OrRd", 1),
        ("mean_abs_response_vehh", "Mean |response| (veh/h)", "PuRd", 1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    for ax, (key, title, cmap, scale) in zip(axes, metrics, strict=True):
        vals = {c: summ[c][key] * scale for c in summ if c > 0}
        lo, hi = min(vals.values()), max(vals.values())
        cm = plt.get_cmap(cmap)
        for c, rings in polys:
            v = vals.get(c, lo)
            col = cm(0.15 + 0.85 * (v - lo) / max(hi - lo, 1e-9))
            for r in rings:
                ax.fill(r[:, 0], r[:, 1], color=col, ec="white", lw=0.8, zorder=2)
        draw_boundaries(ax, polys, lw=0.5, color="#94a3b8")
        ax.set_aspect(A); ax.set_title(title, fontweight="600", color=COLORS["dgray"])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(lo, hi))
        plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        top = max(vals, key=vals.get)
        ax.text(0.02, 0.02, f"highest: {top}  ({vals[top]:.1f})", transform=ax.transAxes,
                fontsize=8, color=COLORS["slate"])
    fig.suptitle("Where the policy lands, and where the traffic moves",
                 fontsize=13, fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.02,
             "Per-arrondissement aggregates over all 1,000 scenarios. Severity is the mean capacity reduction on links that were "
             "intervened.\nResponse is the mean absolute change in link volume, which includes links the policy never touched.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "02_arrondissement_choropleths", "intervention and response by district")

    # ---- 3. concentration ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    tot = absY.sum()
    codes = sorted(c for c in summ if c > 0)
    share_resp = np.array([absY[ar == c].sum() / tot for c in codes])
    share_link = np.array([(ar == c).mean() for c in codes])
    ratio = share_resp / share_link
    order = np.argsort(-ratio)

    ax = axes[0]
    cols = [COLORS["coral"] if ratio[i] > 1 else COLORS["blue"] for i in order]
    ax.barh(range(len(codes)), ratio[order], color=cols)
    ax.axvline(1, color=COLORS["slate"], lw=1.1, ls="--")
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([codes[i] for i in order], fontsize=8)
    ax.set_xlabel("share of total response ÷ share of links")
    ax.set_ylabel("arrondissement")
    ax.set_title("Which districts absorb more than their size",
                 fontweight="600", color=COLORS["dgray"])
    ax.text(0.97, 0.04, "dashed line = proportional", transform=ax.transAxes,
            ha="right", fontsize=8, color=COLORS["slate"])

    ax = axes[1]
    sev = [summ[c]["mean_intervention_severity_vehh"] for c in codes]
    rsp = [summ[c]["mean_abs_response_vehh"] for c in codes]
    ax.scatter(sev, rsp, s=[3 + 40 * share_link[i] * 10 for i in range(len(codes))],
               c=COLORS["blue"], alpha=0.75, edgecolors="white", linewidths=0.8)
    for i, c in enumerate(codes):
        ax.annotate(str(c), (sev[i], rsp[i]), fontsize=7.2,
                    textcoords="offset points", xytext=(5, 3), color=COLORS["slate"])
    ax.set_xlabel("mean intervention severity (veh/h)")
    ax.set_ylabel("mean |response| (veh/h)")
    ax.set_title("Severity does not determine local response",
                 fontweight="600", color=COLORS["dgray"])
    r = np.corrcoef(sev, rsp)[0, 1]
    ax.text(0.03, 0.94, f"Pearson r = {r:+.2f}", transform=ax.transAxes, va="top",
            fontsize=9, family="monospace", color=COLORS["slate"])
    fig.suptitle("Response is not proportional to intervention",
                 fontsize=13, fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.06,
             "Marker area is the district's share of network links. A district can be heavily intervened and respond little, or barely "
             "touched\nand absorb a great deal — consistent with the network-wide redistribution reported in the data story.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "03_response_concentration", "disproportionate districts")
    print(f"\nfigures written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
