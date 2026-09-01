"""
thesis_style.py
===============
Unified academic style for all 32 thesis figures.
Import this module at the top of every figure-generation script.

Usage:
    from thesis_style import *
    # Then use COLORS dict, panel_label(), save_fig() as needed.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Academic color palette (slightly more saturated than old pastel for print)
# ---------------------------------------------------------------------------
COLORS = {
    "blue": "#4A90B8",  # primary blue
    "blue_lt": "#8BBAD8",  # light blue
    "blue_dk": "#2A6894",  # dark blue for contrast
    "coral": "#D4694A",  # coral / highlight
    "coral_lt": "#E8A08A",  # light coral
    "green": "#5A9A7C",  # sage green
    "green_lt": "#A0C8B0",  # light green
    "purple": "#7E6CB3",  # lavender
    "amber": "#D89A3C",  # amber / gold
    "slate": "#4C5B6A",  # dark slate for text/arrows
    "dgray": "#2A3A4A",  # near-black for titles
    "mgray": "#6A7A8A",  # medium gray
    "lgray": "#C0C8D0",  # light gray for borders/grid
    "xlgray": "#E0E4E8",  # extra light gray panels
    "white": "#FFFFFF",
}

# Convenience aliases matching old variable names
BG = "#FFFFFF"  # pure white background (academic standard)
PANEL = "#F4F6F8"  # subtle panel fill for schematics
P_BLUE = COLORS["blue"]
P_BLUE_LT = COLORS["blue_lt"]
P_BLUE_DK = COLORS["blue_dk"]
P_CORAL = COLORS["coral"]
P_CORAL_LT = COLORS["coral_lt"]
P_GREEN = COLORS["green"]
P_GREEN_LT = COLORS["green_lt"]
P_PURPLE = COLORS["purple"]
P_AMBER = COLORS["amber"]
P_SLATE = COLORS["slate"]
P_DGRAY = COLORS["dgray"]
P_MGRAY = COLORS["mgray"]
P_LGRAY = COLORS["lgray"]
P_XLGRAY = COLORS["xlgray"]
WHITE = COLORS["white"]

# ---------------------------------------------------------------------------
# Unified rcParams (academic style)
# ---------------------------------------------------------------------------
RCPARAMS = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titlepad": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "grid.color": "#C0C0C0",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "figure.subplot.top": 0.90,
    "figure.subplot.bottom": 0.12,
    "figure.subplot.hspace": 0.35,
    "figure.subplot.wspace": 0.30,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#C0C0C0",
}

plt.rcParams.update(RCPARAMS)


# ---------------------------------------------------------------------------
# Helper: add panel label  e.g. "(a)", "(b)" to a subplot
# ---------------------------------------------------------------------------
def panel_label(ax, label, x=-0.08, y=1.03, fontsize=12, fontweight="bold"):
    """Add a panel label like (a), (b) to the top-left of an axes.

    Positioned just above the axes area to avoid collision with suptitle.
    """
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va="top",
        ha="right",
    )


# ---------------------------------------------------------------------------
# Helper: save figure as PDF + PNG
# ---------------------------------------------------------------------------
def save_fig(fig, name, out_dir=None, bg=None, skip_tight=False):
    """Save figure as both PDF and PNG at 300 DPI.

    Applies tight_layout with generous padding before saving unless
    skip_tight=True (for manually-positioned schematic figures).
    """
    fc = bg if bg is not None else "#FFFFFF"
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    # Auto-apply tight_layout with safe padding
    if not skip_tight:
        try:
            fig.tight_layout(pad=1.8)
        except Exception:
            pass  # Some figure layouts (e.g. axes off) may not support this

    pdf = os.path.join(out_dir, name + ".pdf")
    png = os.path.join(out_dir, name + ".png")
    fig.savefig(pdf, bbox_inches="tight", facecolor=fc, dpi=300)
    fig.savefig(png, bbox_inches="tight", facecolor=fc, dpi=300)
    plt.close(fig)
    print(f"  Saved: {name}.pdf + .png")


# ---------------------------------------------------------------------------
# Standard annotation style
# ---------------------------------------------------------------------------
ANNOT_STYLE = dict(fontsize=10, fontweight="normal", color=COLORS["dgray"])
TITLE_STYLE = dict(fontsize=13, fontweight="bold", color=COLORS["dgray"])
SUBTITLE_STYLE = dict(fontsize=11, fontweight="bold", color=COLORS["dgray"])
NOTE_STYLE = dict(fontsize=9, fontstyle="italic", color=COLORS["mgray"])
