#!/usr/bin/env python
"""Isometric SVG diagrams generated from the real data, not drawn by hand.

Every coordinate in the layered views is a real link midpoint from `pos`,
projected isometrically, so the shapes are the actual Paris network rather than
an illustration of one. That keeps the visual honest: if the data changes, the
diagram changes.

Isometric projection used throughout:

    sx = (x - y) * cos(30 deg)
    sy = (x + y) * sin(30 deg) - z

Usage:
    python scripts/figure_generation/generate_isometric_diagrams.py \
        --corpus DIR --cache DIR

Output: docs/diagrams_isometric/
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "data_exploration"))
from common import add_common_args, load  # noqa: E402

OUT = REPO / "docs" / "diagrams_isometric"
COS30, SIN30 = math.cos(math.radians(30)), math.sin(math.radians(30))

PAL = {
    "ink": "#0f172a", "muted": "#64748b", "line": "#94a3b8",
    "data": "#2563eb", "data_lt": "#bfdbfe",
    "model": "#7c3aed", "model_lt": "#ddd6fe",
    "uq": "#ea580c", "uq_lt": "#fed7aa",
    "eval": "#059669", "eval_lt": "#a7f3d0",
    "plate": "#f8fafc", "plate_edge": "#cbd5e1",
}
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, "
        "Arial, sans-serif")


def iso(x, y, z=0.0):
    return (x - y) * COS30, (x + y) * SIN30 - z


def normalise(mid):
    """Real WGS84 midpoints -> the unit square, without distorting the city.

    Three things matter here:

      * longitude is scaled by cos(lat), so Paris is not stretched sideways;
      * the span comes from the 1st-99th percentile, because a handful of outlying
        links otherwise push the dense core into one corner of the plate;
      * both axes share one scale, so relative distances survive the projection.

    Points outside that frame are clamped to the plate edge rather than dropped. The
    number clamped is returned so the caller can state it instead of hiding it.
    """
    lat0 = float(np.mean(mid[:, 1]))
    xy = np.column_stack([mid[:, 0] * np.cos(np.deg2rad(lat0)), mid[:, 1]])
    q_lo, q_hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
    span = float((q_hi - q_lo).max())
    ctr = (q_hi + q_lo) / 2.0
    p = (xy - ctr) / span + 0.5
    n_out = int(((p < -0.1) | (p > 1.1)).any(1).sum())
    return np.clip(p, -0.1, 1.1), n_out


def plate(cx, cy, w, h, z, fill, edge, opacity=0.92):
    """A flat quad in the ground plane, lifted by z."""
    pts = [iso(cx - w / 2, cy - h / 2, z), iso(cx + w / 2, cy - h / 2, z),
           iso(cx + w / 2, cy + h / 2, z), iso(cx - w / 2, cy + h / 2, z)]
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (f'<polygon points="{d}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{edge}" stroke-width="1.2"/>')


def header(w, h, title, subtitle):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="{FONT}">\n'
            f'  <rect width="{w}" height="{h}" fill="#ffffff"/>\n'
            f'  <style>\n'
            f'    .h {{ fill:{PAL["ink"]}; font-size:19px; font-weight:600; }}\n'
            f'    .s {{ fill:{PAL["muted"]}; font-size:12.5px; }}\n'
            f'    .lbl {{ fill:{PAL["ink"]}; font-size:14px; font-weight:600; }}\n'
            f'    .sub {{ fill:{PAL["muted"]}; font-size:11.5px; }}\n'
            f'    .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'
            f'monospace; font-size:11.5px; fill:#334155; }}\n'
            f'    .cap {{ fill:{PAL["muted"]}; font-size:12px; }}\n'
            f'  </style>\n'
            f'  <text class="h" x="34" y="40">{title}</text>\n'
            f'  <text class="s" x="34" y="62">{subtitle}</text>\n')


def diagram_graph_lifting(mid, ei, hw, out):
    """Geography -> line graph: the same links, seen twice."""
    W, H = 1180, 700
    s = ""
    p, n_out = normalise(mid)
    rng = np.random.default_rng(7)
    sel = rng.choice(len(p), 5200, replace=False)
    Z_TOP = 0.62
    # Centre on the actual projected extent of both plates rather than guessing.
    corners = [iso(u, v, z) for u in (-0.12, 1.12) for v in (-0.12, 1.12)
               for z in (0.0, Z_TOP)]
    cxs = [c[0] for c in corners]; cys = [c[1] for c in corners]
    SCALE = min((W - 380) / (max(cxs) - min(cxs)), (H - 250) / (max(cys) - min(cys)))
    OX = (W + 250) / 2 - (max(cxs) + min(cxs)) / 2 * SCALE
    OY = 96 - min(cys) * SCALE

    def place(u, v, z):
        sx, sy = iso(u, v, z)
        return OX + sx * SCALE, OY + sy * SCALE

    s += header(W, H, "From road geometry to graph nodes",
                "Every point is a real link midpoint from pos[:, 2]: 5,200 sampled links "
                "below, every second one of them redrawn above so the edges stay legible. "
                f"{n_out:,} of 31,635 links fall outside the 1-99 percentile frame and "
                "sit on its edge.")

    # --- lower plane: geography ---
    def quad(z, fill, edge, op):
        pts = [place(-0.1, -0.1, z), place(1.1, -0.1, z),
               place(1.1, 1.1, z), place(-0.1, 1.1, z)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        return (f'<polygon points="{d}" fill="{fill}" fill-opacity="{op}" '
                f'stroke="{edge}" stroke-width="1.3"/>\n')
    s += quad(0.0, "#f1f5f9", PAL["plate_edge"], 1.0)
    for i in sel:
        x, y = place(p[i, 0], p[i, 1], 0.0)
        col = {1: PAL["uq"], 2: "#d97706", 3: PAL["eval"]}.get(int(hw[i]), "#cbd5e1")
        s += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.05" fill="{col}" fill-opacity="0.9"/>\n'
    lx, ly = place(-0.1, 1.1, 0.0)
    s += (f'<text class="lbl" text-anchor="end" x="{lx-26:.0f}" y="{ly-16:.0f}">'
          f'1 · Road network in space</text>\n')
    s += (f'<text class="sub" text-anchor="end" x="{lx-26:.0f}" y="{ly+2:.0f}">'
          f'31,635 links, WGS84 midpoints. Colour = OSM class.</text>\n')

    # --- upper plane: graph ---
    Z = Z_TOP
    s += quad(Z, "#ffffff", PAL["data"], 0.35)
    # draw a subgraph of real edges among the sampled nodes
    pick = set(sel.tolist())
    drawn = 0
    for a, b in zip(ei[0][::37], ei[1][::37], strict=True):
        if drawn > 900:
            break
        if a in pick and b in pick:
            x1, y1 = place(p[a, 0], p[a, 1], Z)
            x2, y2 = place(p[b, 0], p[b, 1], Z)
            s += (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                  f'stroke="{PAL["data"]}" stroke-width="0.45" stroke-opacity="0.5"/>\n')
            drawn += 1
    for i in sel[::2]:
        x, y = place(p[i, 0], p[i, 1], Z)
        s += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.2" fill="{PAL["data"]}" fill-opacity="0.9"/>\n'
    lx, ly = place(-0.1, 1.1, Z)
    s += (f'<text class="lbl" text-anchor="end" x="{lx-26:.0f}" y="{ly-16:.0f}">'
          f'2 · Line graph</text>\n')
    s += (f'<text class="sub" text-anchor="end" x="{lx-26:.0f}" y="{ly+2:.0f}">'
          f'each link becomes a node; an edge means two links meet</text>\n')

    # lifting cues
    for i in sel[:26]:
        x1, y1 = place(p[i, 0], p[i, 1], 0.0)
        x2, y2 = place(p[i, 0], p[i, 1], Z)
        s += (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
              f'stroke="{PAL["line"]}" stroke-width="0.6" stroke-dasharray="3 4" '
              f'stroke-opacity="0.65"/>\n')

    s += (f'<text class="cap" x="34" y="{H-52}">The inversion is the whole point: in the '
          f'road network a junction is a node, here a road link is a node. That is why the '
          f'model predicts</text>\n'
          f'<text class="cap" x="34" y="{H-34}">one value per link, and why 59,851 edges '
          f'connect 31,635 nodes with a maximum degree of 10.</text>\n'
          f'<text class="cap" x="34" y="{H-16}">Generated from the published corpus by '
          f'scripts/figure_generation/generate_isometric_diagrams.py.</text>\n')
    s += "</svg>\n"
    (out / "iso_01_graph_lifting.svg").write_text(s, encoding="utf-8", newline="\n")
    print("  wrote iso_01_graph_lifting.svg")


def diagram_stack(mid, red, y, out):
    """Four stacked planes: features -> intervention -> response -> uncertainty."""
    W, H = 1180, 860
    p, _n_out = normalise(mid)
    rng = np.random.default_rng(11)
    sel = rng.choice(len(p), 4200, replace=False)
    corners = [iso(u, v, z) for u in (-0.12, 1.12) for v in (-0.12, 1.12)
               for z in (0.0, 1.56)]
    cxs = [c[0] for c in corners]; cys = [c[1] for c in corners]
    SCALE = min((W - 400) / (max(cxs) - min(cxs)), (H - 230) / (max(cys) - min(cys)))
    OX = (W + 270) / 2 - (max(cxs) + min(cxs)) / 2 * SCALE
    OY = 92 - min(cys) * SCALE
    M = red != 0
    touch = M.sum(0)
    absY = np.abs(y).mean(0)

    def place(u, v, z):
        sx, sy = iso(u, v, z)
        return OX + sx * SCALE, OY + sy * SCALE

    s = header(W, H, "One network, four layers of the same 31,635 links",
               "Each plane is the same geography carrying a different quantity. "
               "All values are measured, none are illustrative.")

    layers = [
        (0.00, "4 · Uncertainty", "MC Dropout sigma ranks where the model is wrong",
         PAL["uq"], None),
        (0.52, "3 · Response", "change in link volume, the target y",
         PAL["eval"], absY),
        (1.04, "2 · Intervention", "capacity removed, the only column that varies",
         PAL["model"], touch.astype(float)),
        (1.56, "1 · Static features", "volume, capacity, free-speed, length, class",
         PAL["data"], None),
    ]
    for z, title, sub, col, val in layers:
        pts = [place(-0.1, -0.1, z), place(1.1, -0.1, z),
               place(1.1, 1.1, z), place(-0.1, 1.1, z)]
        d = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        s += (f'<polygon points="{d}" fill="#ffffff" fill-opacity="0.75" '
              f'stroke="{PAL["plate_edge"]}" stroke-width="1.1"/>\n')
        if val is None:
            for i in sel[::2]:
                x, yy = place(p[i, 0], p[i, 1], z)
                s += (f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="0.95" fill="{col}" '
                      f'fill-opacity="0.45"/>\n')
        else:
            v = val[sel]
            v = (v - v.min()) / max(v.max() - v.min(), 1e-9)
            for k, i in enumerate(sel):
                if v[k] < 0.04:
                    continue
                x, yy = place(p[i, 0], p[i, 1], z)
                s += (f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="{0.9+2.4*v[k]:.2f}" '
                      f'fill="{col}" fill-opacity="{0.25+0.6*v[k]:.2f}"/>\n')
        lx, ly = place(-0.1, 1.1, z)
        s += (f'<text class="lbl" text-anchor="end" x="{lx-26:.0f}" '
              f'y="{ly-14:.0f}">{title}</text>\n')
        s += (f'<text class="sub" text-anchor="end" x="{lx-26:.0f}" '
              f'y="{ly+4:.0f}">{sub}</text>\n')

    s += (f'<text class="cap" x="34" y="{H-52}">Marker size and opacity encode the '
          f'quantity on that plane. The intervention layer shows how often each link was '
          f'touched across the</text>\n'
          f'<text class="cap" x="34" y="{H-34}">1,000 scenarios; the response layer shows '
          f'mean |change in volume|. They do not coincide — most of the response lands '
          f'away from the intervention.</text>\n'
          f'<text class="cap" x="34" y="{H-16}">Uncertainty is drawn as a plane rather '
          f'than values because sigma is produced per scenario, not per link.</text>\n')
    s += "</svg>\n"
    (out / "iso_02_layer_stack.svg").write_text(s, encoding="utf-8", newline="\n")
    print("  wrote iso_02_layer_stack.svg")


def diagram_pipeline(out):
    """Isometric slab pipeline: simulation -> graph -> model -> UQ -> decision."""
    W, H = 1180, 660
    stages = [
        ("MATSim", "1,000 scenarios", "hours each", PAL["data"], PAL["data_lt"]),
        ("Line graph", "31,635 nodes", "59,851 edges", PAL["data"], PAL["data_lt"]),
        ("GNN surrogate", "1.42 M params", "seconds each", PAL["model"], PAL["model_lt"]),
        ("MC Dropout", "S = 30 passes", "sigma per link", PAL["uq"], PAL["uq_lt"]),
        ("Calibration", "T = 2.702", "conformal", PAL["eval"], PAL["eval_lt"]),
        ("Decision", "-41.2% MAE", "AUROC 0.7585", PAL["eval"], PAL["eval_lt"]),
    ]
    s = header(W, H, "The pipeline as built",
               "Six stages, each labelled with a figure verified by "
               "scripts/verify_headline_results.py.")

    # Six slabs stepping down and to the right span a long diagonal. Fit that diagonal
    # to the canvas instead of assuming a scale: with a fixed one the last two stages
    # fall off the bottom-right edge and collide with the caption.
    MARGIN_X, TOP, BOTTOM = 42, 112, 104
    pts = [iso(k * 1.15 + du, dv, dz)
           for k in range(len(stages))
           for du in (0.0, 0.95) for dv in (0.0, 0.95) for dz in (0.0, 0.30)]
    xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
    SCALE = min((W - 2 * MARGIN_X) / (max(xs) - min(xs)),
                (H - TOP - BOTTOM) / (max(ys) - min(ys)))
    OX = MARGIN_X - min(xs) * SCALE + (W - 2 * MARGIN_X - (max(xs) - min(xs)) * SCALE) / 2
    OY = TOP - min(ys) * SCALE

    for k, (name, a, b, col, lt) in enumerate(stages):
        u = k * 1.15
        # slab with visible thickness
        top = [iso(u, 0, 0.30), iso(u + 0.95, 0, 0.30),
               iso(u + 0.95, 0.95, 0.30), iso(u, 0.95, 0.30)]
        left = [iso(u, 0.95, 0.30), iso(u + 0.95, 0.95, 0.30),
                iso(u + 0.95, 0.95, 0.0), iso(u, 0.95, 0.0)]
        right = [iso(u + 0.95, 0, 0.30), iso(u + 0.95, 0.95, 0.30),
                 iso(u + 0.95, 0.95, 0.0), iso(u + 0.95, 0, 0.0)]
        for face, fill, op in ((left, col, 0.30), (right, col, 0.20), (top, lt, 0.95)):
            d = " ".join(f"{OX+x*SCALE:.1f},{OY+y*SCALE:.1f}" for x, y in face)
            s += (f'<polygon points="{d}" fill="{fill}" fill-opacity="{op}" '
                  f'stroke="{col}" stroke-width="1.1"/>\n')
        cx, cy = iso(u + 0.475, 0.475, 0.30)
        X, Y = OX + cx * SCALE, OY + cy * SCALE
        s += f'<text class="lbl" x="{X:.0f}" y="{Y-6:.0f}" text-anchor="middle">{name}</text>\n'
        s += f'<text class="sub" x="{X:.0f}" y="{Y+11:.0f}" text-anchor="middle">{a}</text>\n'
        s += f'<text class="sub" x="{X:.0f}" y="{Y+27:.0f}" text-anchor="middle">{b}</text>\n'
        if k < len(stages) - 1:
            ax, ay = iso(u + 1.0, 0.475, 0.30)
            bx, by = iso(u + 1.13, 0.475, 0.30)
            s += (f'<line x1="{OX+ax*SCALE:.1f}" y1="{OY+ay*SCALE:.1f}" '
                  f'x2="{OX+bx*SCALE:.1f}" y2="{OY+by*SCALE:.1f}" '
                  f'stroke="{PAL["line"]}" stroke-width="2"/>\n')
    s += (f'<text class="cap" x="34" y="{H-58}">The surrogate replaces the simulator for '
          f'the forward question only. Everything from MC Dropout rightwards is post-hoc: '
          f'the trained</text>\n'
          f'<text class="cap" x="34" y="{H-40}">weights are loaded, frozen, and never '
          f'updated, which is what makes the uncertainty layer cheap enough to be '
          f'practical.</text>\n'
          f'<text class="cap" x="34" y="{H-22}">Calibration figures follow the '
          f'graph20_80_v1 protocol; see docs/CORRIGENDUM.md C3.</text>\n')
    s += "</svg>\n"
    (out / "iso_03_pipeline.svg").write_text(s, encoding="utf-8", newline="\n")
    print("  wrote iso_03_pipeline.svg")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    OUT.mkdir(parents=True, exist_ok=True)
    mid = pos[:, 2, :].astype(np.float64)
    diagram_graph_lifting(mid, ei, X[:, 4].astype(int), OUT)
    diagram_stack(mid, red, y, OUT)
    diagram_pipeline(OUT)
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
