"""
generate_network_intro_figure.py  [v2 — redesigned for TUM thesis]
===================================================================
Professional redesign of fig_network_intro.

Design:
  - Two-panel layout: directed graph (left) | feature panel (right)
  - No title baked into the figure
  - No floating Δv badges on nodes
  - Full feature names, no abbreviations
  - Clean horizontal connector from focus node to feature panel
  - Minimal legend strip at bottom
  - Pastel palette matching all other thesis figures

Run from figures/ directory:
    python generate_network_intro_figure.py

Outputs:
    fig_network_intro.pdf
    fig_network_intro.png
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Palette (matches thesis palette exactly) ─────────────────────────────────
BG = "#FAFBFC"
P_BLUE = "#5B8DB8"
P_BLUE_LT = "#A8C8E8"
P_BLUE_DK = "#2E6494"
P_CORAL = "#E07A5F"
P_CORAL_LT = "#F2B5A0"
P_SLATE = "#5C6B7A"
P_DGRAY = "#3A4A5A"
P_MGRAY = "#7A8A9A"
P_LGRAY = "#D0D8E0"
P_XLGRAY = "#E8EDF2"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 150,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "text.usetex": False,
    }
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def save(fig, name):
    pdf = os.path.join(OUT, name + ".pdf")
    png = os.path.join(OUT, name + ".png")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor=BG)
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {name}.pdf  +  {name}.png")


def _arrow(ax, x0, y0, x1, y1, r=0.28, color=P_LGRAY, lw=1.0, ms=10):
    """Clipped directional arrow between two node centres."""
    d = np.hypot(x1 - x0, y1 - y0)
    if d < 1e-9:
        return
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    ax.annotate(
        "",
        xy=(x1 - ux * r, y1 - uy * r),
        xytext=(x0 + ux * r, y0 + uy * r),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms),
        zorder=2,
    )


# ── Main figure ───────────────────────────────────────────────────────────────


def fig_network_intro():

    # Canvas — 9.6 × 5.4 in gives breathing room for the two-panel layout
    FW, FH = 9.6, 5.4
    DIV_X = 5.20  # x-coordinate of left|right panel divider
    NODE_R = 0.28  # node circle radius (data units = inches here)

    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")

    # =========================================================================
    # LEFT PANEL — directed graph
    # Six nodes; two paths converge through central focus node i.
    # Topology suggests a road-network fragment without being cartoonish.
    # =========================================================================
    nodes = {
        "i": (2.25, 2.80),  # focus node  (blue, thick border)
        "j": (0.75, 4.10),  # regular
        "k": (3.80, 4.25),  # policy-affected
        "l": (0.80, 1.35),  # regular
        "m": (3.50, 1.30),  # policy-affected
        "n": (4.85, 3.22),  # regular
    }
    policy_nodes = {"k", "m"}
    focus_node = "i"

    # Edges encode directed connectivity (src → dst)
    edges = [
        ("j", "i"),
        ("j", "k"),
        ("i", "k"),
        ("i", "m"),
        ("l", "i"),
        ("l", "m"),
        ("k", "n"),
        ("m", "n"),
    ]

    # Draw edges (edges touching the focus node are slightly darker/heavier)
    for src, dst in edges:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        is_f = src == focus_node or dst == focus_node
        _arrow(
            ax,
            x0,
            y0,
            x1,
            y1,
            NODE_R,
            color=P_SLATE if is_f else P_LGRAY,
            lw=1.4 if is_f else 0.9,
            ms=12 if is_f else 10,
        )

    # Draw node circles and labels
    for nid, (nx, ny) in nodes.items():
        if nid == focus_node:
            fc, ec, lw = P_BLUE_LT, P_BLUE_DK, 2.2
        elif nid in policy_nodes:
            fc, ec, lw = P_CORAL_LT, P_CORAL, 1.6
        else:
            fc, ec, lw = P_XLGRAY, P_LGRAY, 1.1

        ax.add_patch(
            mpatches.Circle(
                (nx, ny), NODE_R, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
            )
        )
        ax.text(
            nx,
            ny,
            nid,
            ha="center",
            va="center",
            fontsize=10.5,
            fontstyle="italic",
            color=P_DGRAY,
            zorder=4,
        )

    # Panel divider — dashed, very subtle
    ax.plot([DIV_X, DIV_X], [0.28, FH - 0.22], color=P_LGRAY, lw=0.7, ls="--", zorder=1)

    # =========================================================================
    # RIGHT PANEL — feature card
    # =========================================================================
    PAD = 0.22  # inner card padding
    CL = DIV_X + 0.25  # card left  x
    CR = FW - 0.18  # card right x
    CW = CR - CL  # card width  ≈ 4.17
    CX = (CL + CR) / 2
    CB = 0.35  # card bottom y
    CT = FH - 0.25  # card top    y
    CH = CT - CB  # card height

    # Card outline
    ax.add_patch(
        FancyBboxPatch(
            (CL, CB),
            CW,
            CH,
            boxstyle="round,pad=0.09",
            linewidth=0.9,
            edgecolor=P_LGRAY,
            facecolor=WHITE,
            zorder=5,
        )
    )

    # ── Header ──────────────────────────────────────────────────────────────
    cur_y = CT - PAD  # start from top, moving down

    # Tiny node icon echoes the focus node in the graph
    ICON_R = 0.145
    icx = CL + PAD + ICON_R
    ax.add_patch(
        mpatches.Circle(
            (icx, cur_y),
            ICON_R,
            facecolor=P_BLUE_LT,
            edgecolor=P_BLUE_DK,
            linewidth=1.9,
            zorder=7,
        )
    )
    ax.text(
        icx,
        cur_y,
        "i",
        ha="center",
        va="center",
        fontsize=8.5,
        fontstyle="italic",
        color=P_DGRAY,
        zorder=8,
    )

    # Header title
    ax.text(
        icx + ICON_R + 0.17,
        cur_y,
        "Node i  —  road segment",
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=P_DGRAY,
        zorder=7,
    )

    # Subtitle
    ax.text(
        icx + ICON_R + 0.17,
        cur_y - 0.34,
        "after line-graph transformation of the road network",
        ha="left",
        va="center",
        fontsize=8.0,
        fontstyle="italic",
        color=P_MGRAY,
        zorder=7,
    )

    cur_y -= 0.65

    # Separator 1
    ax.plot([CL + PAD, CR - PAD], [cur_y, cur_y], color=P_LGRAY, lw=0.6, zorder=6)
    cur_y -= 0.26

    # ── Input features section ───────────────────────────────────────────────
    ax.text(
        CL + PAD,
        cur_y,
        "INPUT FEATURES",
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=P_MGRAY,
        zorder=6,
    )
    cur_y -= 0.35

    features = [
        ("VOL_BASE_CASE", "veh/h"),
        ("CAPACITY_BASE_CASE", "veh/h"),
        ("CAPACITY_REDUCTION", "veh/h"),
        ("FREESPEED", "m/s"),
        ("LENGTH", "metres"),
    ]

    for fname, funit in features:
        # Filled circle bullet
        ax.plot(
            CL + PAD + 0.10,
            cur_y,
            "o",
            ms=3.8,
            color=P_BLUE,
            markeredgewidth=0,
            zorder=7,
        )
        ax.text(
            CL + PAD + 0.27,
            cur_y,
            fname,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=P_SLATE,
            zorder=7,
        )
        # Unit label right-aligned inside card
        ax.text(
            CR - PAD - 0.05,
            cur_y,
            f"[{funit}]",
            ha="right",
            va="center",
            fontsize=8.5,
            color=P_MGRAY,
            zorder=7,
        )
        cur_y -= 0.38

    cur_y += 0.09  # tighten before separator

    # Separator 2
    ax.plot([CL + PAD, CR - PAD], [cur_y, cur_y], color=P_LGRAY, lw=0.6, zorder=6)
    cur_y -= 0.22

    # ── Prediction target section ────────────────────────────────────────────
    ax.text(
        CL + PAD,
        cur_y,
        "PREDICTION TARGET",
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=P_MGRAY,
        zorder=6,
    )
    cur_y -= 0.32

    # Tinted formula box
    FORM_H = 0.58
    ax.add_patch(
        FancyBboxPatch(
            (CL + PAD, cur_y - FORM_H),
            CW - 2 * PAD,
            FORM_H,
            boxstyle="round,pad=0.07",
            linewidth=0.7,
            edgecolor=P_CORAL_LT,
            facecolor="#FEF5F2",
            zorder=6,
        )
    )

    ax.text(
        CX,
        cur_y - FORM_H / 2 + 0.08,
        r"$\Delta v = v_{\mathrm{policy}} - v_{\mathrm{baseline}}$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=P_CORAL,
        zorder=7,
    )

    ax.text(
        CX,
        cur_y - FORM_H / 2 - 0.15,
        "[vehicles / hour, per node]",
        ha="center",
        va="center",
        fontsize=8.0,
        color=P_MGRAY,
        zorder=7,
    )

    # ── Connector: focus node → feature panel ───────────────────────────────
    # Straight horizontal dashed line at the y-level of node i.
    # The line enters the card between the CAPACITY_REDUCTION and FREESPEED
    # rows, pointing directly at the feature list — intentional.
    fx, fy = nodes[focus_node]

    ax.plot(
        [fx + NODE_R, CL - 0.09],
        [fy, fy],
        color=P_BLUE_DK,
        lw=0.85,
        ls=(0, (5, 4)),
        alpha=0.75,
        zorder=2,
    )

    # Filled arrowhead at the card edge
    ax.annotate(
        "",
        xy=(CL - 0.04, fy),
        xytext=(CL - 0.16, fy),
        arrowprops=dict(
            arrowstyle="-|>",
            color=P_BLUE_DK,
            lw=0,
            mutation_scale=8,
            fc=P_BLUE_DK,
        ),
        zorder=3,
    )

    # Small dot at the node end of the connector
    ax.plot(fx + NODE_R, fy, "o", ms=3.5, color=P_BLUE_DK, zorder=3)

    # =========================================================================
    # LEGEND — bottom strip
    # =========================================================================
    LY = 0.08
    LR = 0.085

    def _leg_circle(x0, fc, ec, label):
        ax.add_patch(
            mpatches.Circle(
                (x0 + LR, LY + LR),
                LR,
                facecolor=fc,
                edgecolor=ec,
                linewidth=0.8,
                zorder=5,
            )
        )
        ax.text(
            x0 + LR * 2 + 0.13,
            LY + LR,
            label,
            ha="left",
            va="center",
            fontsize=8.0,
            color=P_SLATE,
        )
        # return x position after the label (approximate)
        return x0 + LR * 2 + 0.13 + len(label) * 0.097

    lx = 0.22
    lx = _leg_circle(lx, P_XLGRAY, P_LGRAY, "road segment")
    lx += 0.35
    lx = _leg_circle(lx, P_CORAL_LT, P_CORAL, "policy-affected segment")
    lx += 0.35

    # Arrow swatch
    ax.annotate(
        "",
        xy=(lx + 0.48, LY + LR),
        xytext=(lx, LY + LR),
        arrowprops=dict(arrowstyle="-|>", color=P_LGRAY, lw=0.9, mutation_scale=9),
        zorder=5,
    )
    ax.text(
        lx + 0.60,
        LY + LR,
        "directed edge",
        ha="left",
        va="center",
        fontsize=8.0,
        color=P_SLATE,
    )

    # Network scale note — right-aligned below the feature panel
    ax.text(
        CR,
        LY + LR,
        "Paris network: 31,635 nodes",
        ha="right",
        va="center",
        fontsize=8.0,
        fontstyle="italic",
        color=P_MGRAY,
        zorder=6,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "fig_network_intro")


if __name__ == "__main__":
    print("Generating fig_network_intro (v2 redesign)...")
    fig_network_intro()
    print("Done.")
