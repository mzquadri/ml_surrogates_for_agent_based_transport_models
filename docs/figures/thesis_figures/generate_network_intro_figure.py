"""
generate_network_intro_figure.py  [v6 — Road-to-Graph, clean rewrite]
=====================================================================
Shows the line-graph transformation from physical road network to
graph representation with node features.

Outputs:
    fig_network_intro.pdf
    fig_network_intro.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    save_fig(fig, name, out_dir=OUT, bg=BG, skip_tight=True)


def fig_network_intro():

    fig = plt.figure(figsize=(14, 6.2))

    # ==========================================================
    # 1. LEFT — Physical Road Network (schematic)
    # ==========================================================
    ax_road = fig.add_axes([0.01, 0.06, 0.22, 0.86])
    ax_road.set_xlim(-0.3, 6.3)
    ax_road.set_ylim(-0.3, 6.3)
    ax_road.axis("off")

    # Diamond layout — four intersections
    IX = {
        "A": np.array([0.5, 3.0]),
        "B": np.array([3.0, 5.5]),
        "C": np.array([5.5, 3.0]),
        "D": np.array([3.0, 0.5]),
    }

    SEG_GRAY = "#9AA5B0"
    SEG = {
        "i": ("A", "B", P_BLUE, 4.0),
        "j": ("B", "C", SEG_GRAY, 2.5),
        "k": ("A", "C", P_CORAL_LT, 3.2),
        "l": ("A", "D", SEG_GRAY, 2.5),
        "m": ("C", "D", P_CORAL_LT, 3.2),
        "n": ("B", "D", SEG_GRAY, 2.5),
    }

    # Label positions — pushed well clear of edges, with white bbox
    SEG_LABELS = {
        "i": (0.95, 4.85),
        "j": (5.00, 4.85),
        "k": (3.00, 3.60),
        "l": (0.95, 1.15),
        "m": (5.00, 1.15),
        "n": (3.60, 2.40),
    }

    for sname, (ia, ib, color, lw) in SEG.items():
        p1, p2 = IX[ia], IX[ib]
        ax_road.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            linewidth=lw,
            solid_capstyle="round",
            zorder=2,
        )

    _lbl_bbox = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)
    for sname, (lx, ly) in SEG_LABELS.items():
        fw = "bold" if sname == "i" else "normal"
        ax_road.text(
            lx,
            ly,
            sname,
            fontsize=13,
            fontstyle="italic",
            ha="center",
            va="center",
            color=P_DGRAY,
            fontweight=fw,
            zorder=10,
            bbox=_lbl_bbox,
        )

    for _, pos in IX.items():
        ax_road.plot(
            pos[0],
            pos[1],
            "o",
            ms=9,
            color=P_SLATE,
            markeredgecolor=P_DGRAY,
            markeredgewidth=1.2,
            zorder=5,
        )

    # Title — placed via fig.text for exact alignment with Graph panel
    # (road axes center x = 0.01 + 0.22/2 = 0.12)

    # ==========================================================
    # 2. CENTRE — Transformation arrow + label (figure-level)
    # ==========================================================
    arr_y = 0.52
    arr_x0, arr_x1 = 0.240, 0.315
    fig.patches.append(
        FancyArrowPatch(
            (arr_x0, arr_y),
            (arr_x1, arr_y),
            transform=fig.transFigure,
            arrowstyle="->,head_length=6,head_width=4",
            color=P_SLATE,
            lw=2.2,
            zorder=20,
        )
    )
    arr_cx = (arr_x0 + arr_x1) / 2
    fig.text(
        arr_cx,
        0.43,
        "Line-graph",
        ha="center",
        va="top",
        fontsize=8.5,
        color=P_MGRAY,
        fontstyle="italic",
        zorder=20,
    )
    fig.text(
        arr_cx,
        0.38,
        "transformation",
        ha="center",
        va="top",
        fontsize=8.5,
        color=P_MGRAY,
        fontstyle="italic",
        zorder=20,
    )

    # ==========================================================
    # 3. RIGHT-CENTRE — 3-D Line Graph
    # ==========================================================
    ax_g = fig.add_axes([0.33, 0.06, 0.35, 0.86], projection="3d")

    # ---- Node positions ----
    # Use elev=40, azim=-35.  For this angle the screen mapping is
    # roughly:  screen_x ~ 0.82*x + 0.57*y,  screen_y ~ 0.64*z + ...
    # We design positions so that PROJECTED positions are well-spread.
    #
    # Desired screen layout (roughly):
    #   j  (top)           k (top-right)
    #         i (centre)
    #   l (left)          m (bottom-right)
    #         n (bottom)
    #
    GN = {
        "i": np.array([4.0, 4.0, 1.5]),  # centre
        "j": np.array([0.0, 9.0, 2.2]),  # top-left  (high z → higher on screen)
        "k": np.array([10.0, 5.0, 2.0]),  # right     (high x → right on screen)
        "l": np.array([-2.0, 2.0, 0.9]),  # left      (low x → left)
        "m": np.array([9.0, -2.0, 0.8]),  # bottom-right (high x, low y, low z)
        "n": np.array([2.0, -3.0, 0.6]),  # bottom-left  (low y, low z → bottom)
    }
    policy_set = {"k", "m"}
    focus = "i"

    ax_g.view_init(elev=40, azim=-35)
    z_floor = 0.3

    # Depth pillars + floor shadows
    for name, pos in GN.items():
        if pos[2] > z_floor + 0.2:
            ax_g.plot(
                [pos[0], pos[0]],
                [pos[1], pos[1]],
                [z_floor, pos[2]],
                color="#C0C8D0",
                lw=0.6,
                ls=":",
                alpha=0.45,
                zorder=1,
            )
        ax_g.scatter(
            pos[0],
            pos[1],
            z_floor,
            s=50,
            c="#DDE2E8",
            edgecolors="none",
            alpha=0.35,
            zorder=1,
        )

    # Directed edges — all valid (share an intersection in road network)
    EDGES = [
        ("j", "i"),
        ("l", "i"),
        ("n", "i"),  # toward focus
        ("i", "k"),  # from focus
        ("j", "k"),
        ("k", "m"),  # via C
        ("l", "m"),
        ("m", "n"),  # via D
    ]

    for src, dst in EDGES:
        p0, p1 = GN[src], GN[dst]
        d = p1 - p0
        dist = np.linalg.norm(d)
        u = d / dist
        shrink = 0.45
        start = p0 + u * shrink
        vec = (p1 - u * shrink) - start

        is_f = src == focus or dst == focus
        c = P_SLATE if is_f else "#8A94A0"
        lw = 2.0 if is_f else 1.2
        alr = 0.07 if is_f else 0.05

        ax_g.quiver(
            start[0],
            start[1],
            start[2],
            vec[0],
            vec[1],
            vec[2],
            arrow_length_ratio=alr,
            color=c,
            linewidth=lw,
            zorder=3,
        )

    # Nodes
    for name, pos in GN.items():
        if name == focus:
            fc, ec, sz, lw = P_BLUE, P_BLUE_DK, 550, 3.0
        elif name in policy_set:
            fc, ec, sz, lw = P_CORAL_LT, P_CORAL, 380, 2.2
        else:
            fc, ec, sz, lw = "#B8C2CC", "#6A7A8A", 300, 1.6

        ax_g.scatter(
            *pos,
            s=sz,
            c=fc,
            edgecolors=ec,
            linewidth=lw,
            zorder=5,
            depthshade=False,
            alpha=0.95,
        )

    # Node labels
    NL_OFF = {
        "i": (0.0, 0.0, 0.55),
        "j": (0.0, 0.0, 0.55),
        "k": (0.5, 0.0, 0.40),
        "l": (-0.5, 0.0, 0.40),
        "m": (0.5, 0.0, 0.40),
        "n": (0.0, 0.0, -0.55),
    }
    for name, pos in GN.items():
        off = np.array(NL_OFF[name])
        lp = pos + off
        fw = "bold" if name == focus else "normal"
        ax_g.text(
            lp[0],
            lp[1],
            lp[2],
            name,
            fontsize=14,
            fontstyle="italic",
            ha="center",
            va="bottom",
            color=P_DGRAY,
            fontweight=fw,
            zorder=10,
        )

    # Clean 3D chrome — hide everything
    ax_g.set_xlim([-4.0, 12.0])
    ax_g.set_ylim([-5.0, 11.0])
    ax_g.set_zlim([z_floor - 0.05, 3.0])
    ax_g.set_xticks([])
    ax_g.set_yticks([])
    ax_g.set_zticks([])
    for pane in (ax_g.xaxis.pane, ax_g.yaxis.pane, ax_g.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    for axis in (ax_g.xaxis.line, ax_g.yaxis.line, ax_g.zaxis.line):
        axis.set_color("none")
    ax_g.grid(False)

    # Title — placed via fig.text for exact alignment with Road panel
    # (graph axes center x = 0.33 + 0.35/2 = 0.505)

    # ==========================================================
    # 4. FAR RIGHT — Feature callout card
    # ==========================================================
    ax_c = fig.add_axes([0.67, 0.08, 0.31, 0.84])
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.axis("off")

    ax_c.add_patch(
        FancyBboxPatch(
            (0.03, 0.03),
            0.94,
            0.94,
            boxstyle="round,pad=0.025",
            facecolor=WHITE,
            edgecolor=P_LGRAY,
            linewidth=1.0,
            zorder=1,
        )
    )

    # Header
    cy = 0.91
    ax_c.plot(
        0.10,
        cy,
        "o",
        ms=14,
        color=P_BLUE,
        markeredgecolor=P_BLUE_DK,
        markeredgewidth=1.8,
        zorder=5,
    )
    ax_c.text(
        0.10,
        cy,
        "i",
        ha="center",
        va="center",
        fontsize=9,
        fontstyle="italic",
        color="white",
        fontweight="bold",
        zorder=6,
    )
    ax_c.text(
        0.19,
        cy + 0.005,
        "Node i  \u2014  road segment",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=P_DGRAY,
    )
    cy -= 0.055
    ax_c.text(
        0.19,
        cy,
        "after line-graph transformation of the road network",
        ha="left",
        va="center",
        fontsize=7.5,
        fontstyle="italic",
        color=P_MGRAY,
    )

    cy -= 0.035
    ax_c.plot([0.06, 0.94], [cy, cy], color=P_LGRAY, lw=0.6)

    # Input features
    cy -= 0.04
    ax_c.text(
        0.06,
        cy,
        "INPUT FEATURES",
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=P_MGRAY,
    )
    cy -= 0.05

    features = [
        ("VOL_BASE_CASE", "[veh/h]"),
        ("CAPACITY_BASE_CASE", "[veh/h]"),
        ("CAPACITY_REDUCTION", "[veh/h]"),
        ("FREESPEED", "[m/s]"),
        ("LENGTH", "[metres]"),
    ]
    for fname, funit in features:
        ax_c.plot(0.08, cy, "o", ms=3.5, color=P_BLUE, markeredgewidth=0, zorder=3)
        ax_c.text(
            0.12,
            cy,
            fname,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=P_SLATE,
        )
        ax_c.text(0.92, cy, funit, ha="right", va="center", fontsize=8.5, color=P_MGRAY)
        cy -= 0.065

    cy += 0.015
    ax_c.plot([0.06, 0.94], [cy, cy], color=P_LGRAY, lw=0.6)

    # Prediction target
    cy -= 0.04
    ax_c.text(
        0.06,
        cy,
        "PREDICTION TARGET",
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=P_MGRAY,
    )
    cy -= 0.055
    fh = 0.11
    ax_c.add_patch(
        FancyBboxPatch(
            (0.06, cy - fh),
            0.88,
            fh,
            boxstyle="round,pad=0.015",
            facecolor="#FEF5F2",
            edgecolor=P_CORAL_LT,
            linewidth=0.7,
            zorder=2,
        )
    )
    ax_c.text(
        0.50,
        cy - fh / 2 + 0.015,
        r"$\Delta v = v_{\mathrm{policy}} - v_{\mathrm{baseline}}$",
        ha="center",
        va="center",
        fontsize=13,
        color=P_CORAL,
        zorder=3,
    )
    ax_c.text(
        0.50,
        cy - fh / 2 - 0.022,
        "[vehicles / hour, per node]",
        ha="center",
        va="center",
        fontsize=8,
        color=P_MGRAY,
        zorder=3,
    )

    # ==========================================================
    # 5. CONNECTOR — dashed line from node i to card
    # ==========================================================
    pos_i = GN["i"]
    x2, y2, _ = proj3d.proj_transform(pos_i[0], pos_i[1], pos_i[2], ax_g.get_proj())
    disp = ax_g.transData.transform((x2, y2))
    fig_xy = fig.transFigure.inverted().transform(disp)
    cx_conn, cy_conn = fig_xy

    fig.add_artist(
        Line2D(
            [cx_conn, 0.670],
            [cy_conn, cy_conn],
            transform=fig.transFigure,
            color=P_BLUE_DK,
            lw=1.2,
            ls=(0, (4, 3)),
            alpha=0.65,
            zorder=10,
        )
    )
    fig.add_artist(
        Line2D(
            [cx_conn],
            [cy_conn],
            transform=fig.transFigure,
            marker="o",
            ms=4,
            color=P_BLUE_DK,
            alpha=0.65,
            linestyle="None",
            zorder=10,
        )
    )

    # ==========================================================
    # TITLES — both at exact same figure-level y for alignment
    # ==========================================================
    title_y = 0.96
    fig.text(
        0.12,
        title_y,
        "Road Network",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=P_DGRAY,
    )
    fig.text(
        0.505,
        title_y,
        "Graph Representation",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=P_DGRAY,
    )

    # ==========================================================
    # 6. LEGEND — bottom strip
    # ==========================================================
    ly = 0.030

    fig.text(
        0.05,
        ly,
        "\u25cf",
        fontsize=11,
        color=P_BLUE,
        va="center",
        ha="center",
        fontfamily="serif",
    )
    fig.text(0.07, ly, "focus node", fontsize=8.5, color=P_SLATE, va="center")

    fig.text(
        0.19,
        ly,
        "\u25cf",
        fontsize=11,
        color="#B8C2CC",
        va="center",
        ha="center",
        fontfamily="serif",
    )
    fig.text(0.21, ly, "road segment", fontsize=8.5, color=P_SLATE, va="center")

    fig.text(
        0.36,
        ly,
        "\u25cf",
        fontsize=11,
        color=P_CORAL_LT,
        va="center",
        ha="center",
        fontfamily="serif",
    )
    fig.text(0.38, ly, "policy-affected", fontsize=8.5, color=P_SLATE, va="center")

    fig.text(
        0.54,
        ly,
        "\u25cf",
        fontsize=9,
        color=P_SLATE,
        va="center",
        ha="center",
        fontfamily="serif",
    )
    fig.text(0.56, ly, "intersection", fontsize=8.5, color=P_SLATE, va="center")

    fig.text(
        0.70,
        ly,
        "\u2192",
        fontsize=11,
        color=P_SLATE,
        va="center",
        ha="center",
        fontfamily="serif",
    )
    fig.text(0.72, ly, "directed edge", fontsize=8.5, color=P_SLATE, va="center")

    fig.text(
        0.97,
        ly,
        "Paris network: 31,635 nodes",
        fontsize=8.5,
        fontstyle="italic",
        color=P_MGRAY,
        va="center",
        ha="right",
    )

    save(fig, "fig_network_intro")


if __name__ == "__main__":
    print("Generating fig_network_intro (v6 \u2014 Road-to-Graph 3D)...")
    fig_network_intro()
    print("Done.")
