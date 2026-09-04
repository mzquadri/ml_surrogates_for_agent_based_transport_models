"""Design system for the portfolio figure set.

These figures are meant to be looked at, on a GitHub README and later on a
website, so they are built to different rules from the analysis plots:

  * one idea per figure, large enough to read at ~950px wide;
  * the takeaway is written on the chart, not left in a caption;
  * series are labelled where they sit, so the eye never travels to a legend;
  * no chart chrome that carries no information -- no boxes, no heavy grids, no
    tick marks, no colourbar where a direct label will do.

The road network is drawn as real segments from `pos[:, 0]` to `pos[:, 1]`
rather than as scattered midpoints. The corpus stores both endpoints, so the maps
can show the actual streets of Paris instead of a point cloud.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgba  # noqa: E402

#: A restrained palette. One blue for "what is", one amber for "what the policy
#: does", one red for "what moves", greys for everything structural.
INK = "#111827"
BODY = "#374151"
MUTED = "#6B7280"
FAINT = "#9CA3AF"
HAIR = "#E5E7EB"
PAPER = "#FFFFFF"
WASH = "#F9FAFB"

BLUE = "#2563EB"
BLUE_SOFT = "#93C5FD"
AMBER = "#D97706"
AMBER_SOFT = "#FCD34D"
RED = "#DC2626"
RED_SOFT = "#FCA5A5"
GREEN = "#059669"
GREEN_SOFT = "#6EE7B7"
PURPLE = "#7C3AED"
PURPLE_SOFT = "#C4B5FD"

#: Sequential ramps built for dark map backgrounds: they start near the
#: background so quiet links recede, and end bright so busy ones carry.
FLOW = LinearSegmentedColormap.from_list(
    "flow", ["#1E293B", "#1D4ED8", "#0EA5E9", "#22D3EE", "#A7F3D0", "#FEF3C7"])
HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#1E293B", "#4C1D95", "#BE123C", "#F97316", "#FDE68A"])
DIVERGE = LinearSegmentedColormap.from_list(
    "diverge", ["#1D4ED8", "#60A5FA", "#E5E7EB", "#F87171", "#B91C1C"])

#: The same idea as HEAT, for maps drawn on white. FLOW and HEAT begin near black
#: so quiet links sink into a dark ground; on paper that reads as scribble, so
#: this one begins near the paper instead and darkens as the value rises.
HEAT_LIGHT = LinearSegmentedColormap.from_list(
    "heat_light", ["#EEF2F7", "#C7D2FE", "#F59E0B", "#DC2626", "#7F1D1D"])

FAMILY = ["Segoe UI", "Corbel", "DejaVu Sans"]


def apply():
    """Global rcParams. Called once per script."""
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.dpi": 200,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.35,
        "font.family": "sans-serif",
        "font.sans-serif": FAMILY,
        "text.color": BODY,
        "axes.edgecolor": HAIR,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": FAINT,
        "ytick.color": FAINT,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "figure.autolayout": False,
    })


def title_block(fig, title, subtitle=None, x=0.055, y=0.965, size=21):
    """Left-aligned title and subtitle at the top of a figure.

    Left-aligned rather than centred: it reads as a headline over the chart, and
    it lines up with the left edge of the plotting area beneath it.

    The gap between the two is computed in points and converted to a figure
    fraction, so it stays visually constant across figures of different heights.
    A fixed fraction shrinks as the figure gets shorter, which is how a title ends
    up sitting on its own subtitle.
    """
    fig.text(x, y, title, ha="left", va="top", fontsize=size, color=INK,
             fontweight="600")
    if subtitle:
        gap = (size * 1.42 / 72.0) / fig.get_size_inches()[1]
        fig.text(x, y - gap, subtitle, ha="left", va="top", fontsize=12.0,
                 color=MUTED, linespacing=1.55)


def footnote(fig, lines, x=0.055, y=0.035):
    """Source note at the foot, small and grey. Accepts a string or a list."""
    if isinstance(lines, str):
        lines = [lines]
    fig.text(x, y, "\n".join(lines), ha="left", va="top", fontsize=9.4,
             color=FAINT, linespacing=1.6)


def clean(ax, bottom=True, left=True, grid_axis=None):
    """Strip an axes to the minimum, optionally leaving one faint grid direction."""
    for side, keep in (("bottom", bottom), ("left", left)):
        ax.spines[side].set_visible(keep)
        if keep:
            ax.spines[side].set_color(HAIR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.set_axisbelow(True)
        ax.grid(True, axis=grid_axis, color=HAIR, lw=0.9, alpha=0.9)
    ax.tick_params(length=0, pad=6)
    return ax


def label_at(ax, x, y, text, color, size=11.5, weight="600", **kw):
    """Label a series where it sits, so no legend is needed."""
    return ax.text(x, y, text, color=color, fontsize=size, fontweight=weight,
                   clip_on=False, **kw)


def note(ax, x, y, text, color=MUTED, size=10.2, **kw):
    return ax.text(x, y, text, color=color, fontsize=size, clip_on=False,
                   linespacing=1.5, **kw)


def aspect_for(lat):
    """Degrees are not square. Scale longitude so Paris is not stretched."""
    return 1.0 / np.cos(np.deg2rad(float(lat)))


def network(ax, pos, values=None, cmap=FLOW, lw=0.45, color=None, alpha=1.0,
            vmin=None, vmax=None, order_by_value=True, background=None):
    """Draw the road network as real segments from start to end.

    `pos` is the stored `[N, 3, 2]` tensor. Segments use `pos[:, 0]` and
    `pos[:, 1]`; the midpoint slice is never needed here. Passing `values`
    colours each link and draws the strongest links last so they sit on top
    rather than being buried under the quiet majority.
    """
    if background:
        ax.set_facecolor(background)
    seg = np.stack([pos[:, 0, :], pos[:, 1, :]], axis=1)
    if values is not None:
        idx = np.argsort(values) if order_by_value else np.arange(len(values))
        seg, v = seg[idx], np.asarray(values)[idx]
        lc = LineCollection(seg, cmap=cmap, linewidths=lw, alpha=alpha,
                            capstyle="round")
        lc.set_array(v)
        lc.set_clim(vmin if vmin is not None else float(np.nanmin(v)),
                    vmax if vmax is not None else float(np.nanmax(v)))
    else:
        lc = LineCollection(seg, colors=to_rgba(color or FAINT, alpha),
                            linewidths=lw, capstyle="round")
    ax.add_collection(lc)
    ax.set_aspect(aspect_for(pos[:, 2, 1].mean()))
    focus_on(ax, pos)
    bare(ax)
    return lc


def focus_on(ax, pos, q=0.6, pad=0.006):
    """Frame the dense city rather than the full extent.

    A handful of links run far south and west of Paris. Framing on the raw extent
    spends most of the canvas on empty space and shrinks the city to a smudge, so
    the limits come from a percentile of the endpoints instead. Those outlying
    links are still drawn -- they are simply allowed to run off the edge.
    """
    xs = np.concatenate([pos[:, 0, 0], pos[:, 1, 0]])
    ys = np.concatenate([pos[:, 0, 1], pos[:, 1, 1]])
    x0, x1 = np.percentile(xs, [q, 100 - q])
    y0, y1 = np.percentile(ys, [q, 100 - q])
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)


def bare(ax):
    """A map panel: geometry only."""
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def gradient_key(fig, rect, cmap, lo_label, hi_label, title=None, vmin=0, vmax=1):
    """A slim horizontal colour key with its ends labelled in words.

    A conventional colourbar spends a lot of space on numbers a reader does not
    need. Two words at the ends usually say more.
    """
    cax = fig.add_axes(rect)
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cax.imshow(grad, aspect="auto", cmap=cmap, extent=(0, 1, 0, 1))
    bare(cax)
    cax.text(-0.02, 0.5, lo_label, ha="right", va="center", fontsize=9.8,
             color=MUTED, transform=cax.transAxes)
    cax.text(1.02, 0.5, hi_label, ha="left", va="center", fontsize=9.8,
             color=MUTED, transform=cax.transAxes)
    if title:
        cax.text(0, 1.9, title, ha="left", va="bottom", fontsize=9.8,
                 color=FAINT, transform=cax.transAxes)
    return cax


def thousands(x, _pos=None):
    return f"{int(x):,}"
