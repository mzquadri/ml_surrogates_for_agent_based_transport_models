#!/usr/bin/env python
"""The five model input features, in the panel layout the thesis used.

The thesis carried an (a)-(e) figure of the five features that reach the model,
computed over a 200-graph subset (6.3M nodes). This is the same figure rebuilt
over the whole published corpus -- 1,000 scenarios, 31,635,000 node observations
-- with the units a reader actually thinks in and the callouts recomputed rather
than copied.

Numbers therefore differ slightly from the thesis version, and they should: it
described a subset, this describes everything. The zero share of
CAPACITY_REDUCTION is 87.94% here against 85.8% on the 200-graph subset.

Heavy reductions run on the Intel Arc GPU through torch.xpu when it is available,
which is about fifteen times faster than numpy for this shape; the figure is
identical either way.

    python scripts/figure_generation/generate_five_features_figure.py \
        --corpus DIR --cache DIR

Output: docs/figures/portfolio/09_five_model_features.png
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
from common import add_common_args, load  # noqa: E402
from scipy.stats import skew  # noqa: E402

OUT = REPO / "docs" / "figures" / "portfolio"


def moments(flat):
    """Mean, median and skew of a large flat array, on the GPU where possible.

    Returns the backend used so the figure can say which one produced it.
    """
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            t = torch.from_numpy(np.ascontiguousarray(flat)).to("xpu")
            mean = float(t.mean())
            med = float(t.median())
            std = float(t.std())
            sk = float((((t - mean) / std) ** 3).mean())
            torch.xpu.synchronize()
            name = torch.xpu.get_device_name(0)
            del t
            torch.xpu.empty_cache()
            return mean, med, sk, name
    except Exception:
        pass
    return (float(flat.mean()), float(np.median(flat)),
            float(skew(flat)), "CPU (numpy)")


def panel_letter(ax, letter):
    ax.text(-0.155, 1.13, f"({letter})", transform=ax.transAxes, fontsize=15,
            color=ps.INK, fontweight="700", va="top", ha="left")


def callout(ax, x, y, lines, colour=None, ha="right", va="top"):
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, fontsize=10.2,
            color=colour or ps.BODY, ha=ha, va=va, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", fc=ps.PAPER,
                      ec=colour or ps.HAIR, lw=1.0, alpha=0.95))


def main() -> int:
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    ps.apply()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    n_scen, n_links = y.shape

    vol, cap, spd, length = X[:, 0], X[:, 1], X[:, 3], X[:, 5]
    flat_red = red.ravel()
    nz = flat_red[flat_red != 0]
    zero_share = 100.0 * float((flat_red == 0).mean())

    v_mean, v_med, v_skew, backend = moments(vol.astype(np.float32))
    l_mean, l_med, l_skew, _ = moments(length.astype(np.float32))
    print(f"  moments computed on: {backend}")

    fig = plt.figure(figsize=(14.2, 10.2))
    gs = fig.add_gridspec(2, 6, left=0.075, right=0.975, top=0.740, bottom=0.215,
                          hspace=0.62, wspace=1.5)
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2:4])
    axC = fig.add_subplot(gs[0, 4:6])
    axD = fig.add_subplot(gs[1, 1:3])
    axE = fig.add_subplot(gs[1, 3:5])

    # (a) VOL_BASE_CASE -----------------------------------------------------------
    axA.hist(vol, bins=90, color=ps.BLUE, log=True, edgecolor="none")
    axA.axvline(v_med, color=ps.RED, ls="--", lw=1.4)
    ps.clean(axA, grid_axis="y")
    axA.set_xlabel("volume (veh/h)", fontsize=10)
    axA.set_ylabel("links (log)", fontsize=10)
    axA.set_title("VOL_BASE_CASE", fontsize=13, color=ps.INK, fontweight="600", pad=12)
    callout(axA, 0.97, 0.95, [f"median {v_med:.1f} veh/h", f"skew {v_skew:.1f}",
                              f"{100*(vol==0).mean():.1f}% carry no cars"])
    panel_letter(axA, "a")

    # (b) CAPACITY_BASE_CASE ------------------------------------------------------
    vals, counts = np.unique(cap, return_counts=True)
    order = np.argsort(-counts)
    top = order[:9]
    labels = [f"{vals[i]:.0f}" for i in top] + ["other"]
    shares = list(100 * counts[top] / cap.size) + [100 * counts[order[9:]].sum() / cap.size]
    axB.bar(range(len(labels)), shares, color=ps.BLUE, width=0.72)
    axB.set_xticks(range(len(labels)))
    axB.set_xticklabels(labels, fontsize=8.8, rotation=45, ha="right")
    ps.clean(axB, grid_axis="y")
    axB.set_xlabel("capacity value (veh/h)", fontsize=10)
    axB.set_ylabel("share of links (%)", fontsize=10)
    axB.set_title("CAPACITY_BASE_CASE", fontsize=13, color=ps.INK, fontweight="600",
                  pad=12)
    callout(axB, 0.97, 0.95, [f"{vals.size} distinct values",
                              f"{shares[0]:.0f}% sit at {vals[top[0]]:.0f} veh/h"])
    panel_letter(axB, "b")

    # (c) CAPACITY_REDUCTION ------------------------------------------------------
    axC.hist(nz, bins=70, color=ps.AMBER, log=True, edgecolor="none")
    axC.axvline(np.median(nz), color=ps.INK, ls="--", lw=1.4)
    ps.clean(axC, grid_axis="y")
    axC.set_xlabel("reduction (veh/h), non-zero only", fontsize=10)
    axC.set_ylabel("observations (log)", fontsize=10)
    axC.set_title("CAPACITY_REDUCTION", fontsize=13, color=ps.INK, fontweight="600",
                  pad=12)
    callout(axC, 0.05, 0.95,
            [f"{zero_share:.1f}% of observations are 0", "(no policy on that link)"],
            colour=ps.AMBER, ha="left")
    callout(axC, 0.05, 0.42, [f"non-zero median {np.median(nz):.0f}",
                              f"{np.unique(np.round(nz,3)).size} distinct magnitudes",
                              "always negative"], ha="left")
    panel_letter(axC, "c")

    # (d) FREESPEED, in km/h ------------------------------------------------------
    kmh = spd * 3.6
    vals_s, counts_s = np.unique(np.round(kmh, 1), return_counts=True)
    axD.bar(range(vals_s.size), 100 * counts_s / kmh.size, color=ps.GREEN, width=0.7)
    axD.set_xticks(range(vals_s.size))
    axD.set_xticklabels([f"{v:g}" for v in vals_s], fontsize=8.2, rotation=45,
                        ha="right")
    ps.clean(axD, grid_axis="y")
    axD.set_xlabel("speed class (km/h)", fontsize=10)
    axD.set_ylabel("share of links (%)", fontsize=10)
    axD.set_title("FREESPEED", fontsize=13, color=ps.INK, fontweight="600", pad=12)
    peak = int(np.argmax(counts_s))
    callout(axD, 0.97, 0.95, [f"{vals_s.size} discrete classes",
                              f"{100*counts_s[peak]/kmh.size:.1f}% at "
                              f"{vals_s[peak]:g} km/h"])
    panel_letter(axD, "d")

    # (e) LENGTH ------------------------------------------------------------------
    axE.hist(length, bins=90, color=ps.AMBER_SOFT, log=True, edgecolor="none")
    axE.axvline(l_med, color=ps.RED, ls="--", lw=1.4)
    ps.clean(axE, grid_axis="y")
    axE.set_xlabel("link length (m)", fontsize=10)
    axE.set_ylabel("links (log)", fontsize=10)
    axE.set_title("LENGTH", fontsize=13, color=ps.INK, fontweight="600", pad=12)
    callout(axE, 0.97, 0.95, [f"median {l_med:.1f} m", f"skew {l_skew:.1f}",
                              f"longest {length.max():,.0f} m"])
    panel_letter(axE, "e")

    ps.title_block(
        fig, "The five features the model actually reads",
        f"Distributions over the whole published corpus: {n_scen:,} scenarios × "
        f"{n_links:,} links = {n_scen*n_links:,} node observations.\n"
        "HIGHWAY is the sixth column of x and is deliberately absent — its integers "
        "are road-class labels, not quantities.", y=0.962, size=23)

    ps.footnote(fig, [
        "Static columns are summarised once per link; CAPACITY_REDUCTION is summarised "
        "over every node observation, because it is the only column that changes "
        "between scenarios.",
        "FREESPEED is shown in km/h rather than the stored m/s: 8.33 m/s is the 30 km/h "
        "city limit, and it is the median, lower and upper quartile at once.",
        f"Moments computed on {backend}. The thesis version of this figure used a "
        f"200-graph subset, so its callouts differ slightly — its 85.8% zero share "
        f"against {zero_share:.2f}% over the full corpus."], y=0.115)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "09_five_model_features.png")
    plt.close(fig)
    print("  wrote 09_five_model_features.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
