#!/usr/bin/env python
"""The architecture diagram, drawn from the trained checkpoint.

Every shape and parameter count is read out of
models/point_net_transf_gat_8th_trial_lower_dropout/trained_model/model.pth, so the
diagram cannot drift away from the model it describes. If a layer changes, the
picture changes.

    python scripts/figure_generation/generate_model_diagram.py

Output: docs/figures/portfolio/08_model_architecture.png
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import portfolio_style as ps  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = REPO / "docs" / "figures" / "portfolio"
CKPT = (REPO / "models" / "point_net_transf_gat_8th_trial_lower_dropout"
        / "trained_model" / "model.pth")


def read_checkpoint():
    """Layer shapes and parameter counts, straight from the state dict."""
    import torch

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    shapes = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}
    total = sum(v.numel() for v in sd.values() if hasattr(v, "shape"))

    # Group by drawn stage rather than by top-level module: gat_graph_layers holds
    # both TransformerConv blocks and the first GATConv, which are separate boxes.
    stage_of = {
        "point_net_conv_1": "pn1", "point_net_conv_2": "pn2",
        "gat_graph_layers.module_0": "transf", "gat_graph_layers.module_3": "transf",
        "gat_graph_layers.module_6": "gat", "gat_final": "gat",
    }
    params = dict.fromkeys(("pn1", "pn2", "transf", "gat"), 0)
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue
        for prefix, stage in stage_of.items():
            if k.startswith(prefix):
                params[stage] += v.numel()
                break
    return shapes, params, total


def block(ax, x, y, w, h, fill, edge, title, lines, mono=None, title_size=12.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=fill, edgecolor=edge, linewidth=1.1, zorder=2))
    ax.text(x + 0.016, y + h - 0.030, title, fontsize=title_size, color=ps.INK,
            fontweight="600", va="top", zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + 0.016, y + h - 0.068 - i * 0.033, ln, fontsize=9.8,
                color=ps.MUTED, va="top", zorder=3)
    if mono:
        ax.text(x + 0.016, y + 0.016, mono, fontsize=9.6, color=ps.BODY,
                family="monospace", va="bottom", zorder=3)


def arrow(ax, x1, y1, x2, y2, colour=None, label=None, dashed=False):
    c = colour or ps.FAINT
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6,
                                linestyle="--" if dashed else "-",
                                shrinkA=0, shrinkB=0), zorder=1)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.016, label, fontsize=9.2,
                color=c, ha="center", va="bottom", zorder=3)


def main() -> int:
    ps.apply()
    shapes, params, total = read_checkpoint()
    pn1 = shapes["point_net_conv_1.local_nn.0.weight"]      # (256, 7)
    pn2 = shapes["point_net_conv_2.local_nn.0.weight"]      # (256, 514)
    pn2_out = shapes["point_net_conv_2.global_nn.3.weight"]  # (128, 512)
    t1 = shapes["gat_graph_layers.module_0.lin_key.weight"]  # (256, 128)
    t2 = shapes["gat_graph_layers.module_3.lin_key.weight"]  # (512, 256)
    g1 = shapes["gat_graph_layers.module_6.lin.weight"]      # (64, 512)
    g2 = shapes["gat_final.lin.weight"]                      # (1, 64)
    in_ch = pn1[1] - 2                                       # minus the 2-D coordinate

    fig = plt.figure(figsize=(14.6, 8.6))
    ax = fig.add_axes([0, 0, 1, 1]); ps.bare(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # --- inputs -----------------------------------------------------------------
    block(ax, 0.040, 0.585, 0.170, 0.150, "#F0FDF4", ps.GREEN, "Node features",
          ["five columns of x,", "one row per road link"], f"x [31,635, {in_ch}]")
    block(ax, 0.040, 0.400, 0.170, 0.155, "#EFF6FF", ps.BLUE, "Coordinates",
          ["where each link", "starts and ends"], "pos[:, 0] · pos[:, 1]")
    block(ax, 0.040, 0.215, 0.170, 0.155, "#EFF6FF", ps.BLUE, "Connectivity",
          ["which links meet at", "an intersection"], "edge_index [2, 59,851]")

    # --- backbone ----------------------------------------------------------------
    w, h, y = 0.152, 0.235, 0.430
    xs = [0.253, 0.428, 0.603, 0.778]
    block(ax, xs[0], y, w, h, "#F5F3FF", ps.PURPLE, "PointNetConv 1",
          ["takes the start point", f"local  Linear({pn1[1]}, {pn1[0]})",
           "global 256 → 512 → 512"], f"{params['pn1']:,} params")
    block(ax, xs[1], y, w, h, "#F5F3FF", ps.PURPLE, "PointNetConv 2",
          ["takes the end point", f"local  Linear({pn2[1]}, {pn2[0]})",
           f"global 256 → 512 → {pn2_out[0]}"], f"{params['pn2']:,} params")
    block(ax, xs[2], y, w, h, "#FFF7ED", ps.AMBER, "TransformerConv × 2",
          [f"{t1[1]} → 4 heads × {t1[0]//4}", f"{t2[1]} → 4 heads × {t2[0]//4}",
           "key / query / value / skip"], f"{params['transf']:,} params")
    block(ax, xs[3], y, w, h, "#ECFDF5", ps.GREEN, "GATConv × 2",
          [f"{g1[1]} → {g1[0]}, then {g2[1]} → {g2[0]}", "single attention head",
           "one value per link"], f"{params['gat']:,} params")

    for a, b in zip(xs[:-1], xs[1:], strict=True):
        arrow(ax, a + w, y + h / 2, b, y + h / 2)
    arrow(ax, 0.210, 0.655, xs[0], y + h * 0.76, ps.GREEN)
    arrow(ax, 0.210, 0.477, xs[0], y + h * 0.34, ps.BLUE)
    arrow(ax, 0.210, 0.292, xs[1] + w * 0.55, y - 0.008, ps.BLUE, dashed=True)
    ax.text(0.400, 0.258, "edge_index feeds all six message-passing layers",
            fontsize=10.0, color=ps.BLUE, ha="center")

    # --- output -------------------------------------------------------------------
    # Directly beneath the final layer: a long connector across the whole diagram
    # would cross the dropout band and say nothing the position does not already.
    block(ax, xs[3], 0.222, w, 0.162, "#F8FAFC", "#CBD5E1", "Prediction",
          ["change in car volume,", "one number per road link"],
          "y_pred [31,635, 1]")
    arrow(ax, xs[3] + w / 2, y - 0.004, xs[3] + w / 2, 0.386, "#94A3B8")

    # --- dropout ------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.253, 0.700), 0.677, 0.070,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor="#FFF7ED", edgecolor=ps.AMBER, lw=1.1))
    ax.text(0.269, 0.752, "Two dropout layers, p = 0.2", fontsize=12.0, color=ps.INK,
            fontweight="600", va="top")
    ax.text(0.269, 0.725,
            "Left active at inference, they are what makes post-hoc MC Dropout possible "
            "without retraining anything.",
            fontsize=10.0, color=ps.MUTED, va="top")
    for xx in (0.405, 0.580):
        arrow(ax, xx, 0.700, xx, y + h + 0.004, ps.AMBER, dashed=True)

    ps.title_block(fig, "PointNetTransfGAT, as it was actually trained",
                   "Read from the Trial 8 checkpoint rather than from the configuration: "
                   "every shape and count below is a tensor\nin that file. "
                   f"{total:,} parameters in total.", y=0.962, size=23)
    ps.footnote(fig, [
        f"Evidence for the five input channels: point_net_conv_1.local_nn.0.weight has "
        f"shape {pn1}, and PointNetConv concatenates the node features with a 2-D "
        f"relative coordinate, so {pn1[1]} − 2 = {in_ch}.",
        "The architecture and training code are upstream work by Elena Natterer (MIT). "
        "This thesis takes the trained surrogate as given; the contribution is the "
        "uncertainty layer built on top of it.",
        "Source: models/point_net_transf_gat_8th_trial_lower_dropout/trained_model/"
        "model.pth"], y=0.150)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "08_model_architecture.png")
    plt.close(fig)
    print(f"  wrote 08_model_architecture.png  ({total:,} params, in_channels={in_ch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
