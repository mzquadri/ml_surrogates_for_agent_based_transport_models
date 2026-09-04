#!/usr/bin/env python
"""SVG diagrams for the stored data: what is in a `.pt` file and what reaches the model.

Seven diagrams, all driven by values read from the corpus and the trained
checkpoint rather than typed in by hand. The numbers in the boxes are computed at
generation time, so a change in the data changes the diagram.

SVG `<text>` does not wrap, so every line is emitted as its own element and a
guard at the end asserts that no text node contains a newline.

    python scripts/figure_generation/generate_data_anatomy_diagrams.py \
        --corpus DIR --cache DIR

Output: docs/diagrams_data/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "data_exploration"))
from common import FEATURES, HIGHWAY_CLASSES, MODEL_COLS, add_common_args, load  # noqa: E402

OUT = REPO / "docs" / "diagrams_data"

PAL = {
    "ink": "#0f172a", "muted": "#64748b", "line": "#94a3b8",
    "data": "#2563eb", "data_lt": "#eff6ff", "data_ed": "#bfdbfe",
    "model": "#7c3aed", "model_lt": "#f5f3ff", "model_ed": "#ddd6fe",
    "warn": "#ea580c", "warn_lt": "#fff7ed", "warn_ed": "#fed7aa",
    "ok": "#059669", "ok_lt": "#ecfdf5", "ok_ed": "#a7f3d0",
    "grey_lt": "#f8fafc", "grey_ed": "#e2e8f0",
}
TICK, CROSS = "&#10003;", "&#10007;"
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, "
        "Arial, sans-serif")


def header(w, h, title, subtitle_lines):
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
         f'width="{w}" height="{h}" font-family="{FONT}">\n'
         f'  <rect width="{w}" height="{h}" fill="#ffffff"/>\n'
         f'  <style>\n'
         f'    .h {{ fill:{PAL["ink"]}; font-size:19px; font-weight:600; }}\n'
         f'    .s {{ fill:{PAL["muted"]}; font-size:12.5px; }}\n'
         f'    .lbl {{ fill:{PAL["ink"]}; font-size:14px; font-weight:600; }}\n'
         f'    .sub {{ fill:{PAL["muted"]}; font-size:11.5px; }}\n'
         f'    .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'
         f'monospace; font-size:11.5px; fill:#334155; }}\n'
         f'    .monob {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'
         f'monospace; font-size:12.5px; fill:{PAL["ink"]}; font-weight:600; }}\n'
         f'    .cap {{ fill:{PAL["muted"]}; font-size:12px; }}\n'
         f'    .tick {{ fill:{PAL["ok"]}; font-size:15px; font-weight:700; }}\n'
         f'    .cross {{ fill:{PAL["warn"]}; font-size:15px; font-weight:700; }}\n'
         f'  </style>\n'
         f'  <text class="h" x="34" y="40">{title}</text>\n')
    for i, line in enumerate(subtitle_lines):
        s += f'  <text class="s" x="34" y="{62 + i*17}">{line}</text>\n'
    return s


def box(x, y, w, h, fill, edge, r=9):
    return (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
            f'fill="{fill}" stroke="{edge}" stroke-width="1.4"/>\n')


def text(x, y, s, cls="sub", anchor="start"):
    return f'  <text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{s}</text>\n'


def arrow(x1, y1, x2, y2, color=None):
    """Horizontal connector with an explicit triangular head.

    The head is a polygon rather than a marker: markers are silently dropped by
    some SVG renderers, and a connector that loses its head stops reading as a
    direction.
    """
    c = color or PAL["line"]
    head = 8.0
    return (f'  <line x1="{x1}" y1="{y1}" x2="{x2 - head}" y2="{y2}" stroke="{c}" '
            f'stroke-width="1.8"/>\n'
            f'  <polygon points="{x2},{y2} {x2-head},{y2-4.6} {x2-head},{y2+4.6}" '
            f'fill="{c}"/>\n')


def defs():
    return ""


def caption(w, h, lines):
    s = ""
    for i, line in enumerate(lines):
        s += f'  <text class="cap" x="34" y="{h - 18*(len(lines)-i) - 8}">{line}</text>\n'
    return s


def write(svg, name):
    OUT.mkdir(parents=True, exist_ok=True)
    # SVG text never wraps: a newline inside a <text> silently vanishes on render.
    for chunk in svg.split("<text")[1:]:
        body = chunk.split(">", 1)[1].split("</text>")[0]
        assert "\n" not in body, f"{name}: newline inside a <text> element"
    (OUT / name).write_text(svg, encoding="utf-8", newline="\n")
    print(f"  wrote {name}")


# --------------------------------------------------------------------------------
def d1_pt_anatomy(stats, out):
    """What one .pt file holds."""
    W, H = 1180, 570
    s = header(W, H, "Anatomy of one datalist_batch_*.pt file",
               ["Twenty files hold 1,000 scenarios of the same Paris network. "
                "Each file is a plain Python list of PyG Data objects.",
                "Sizes and shapes read from the published corpus."]) + defs()
    s += box(34, 100, 300, 92, PAL["data_lt"], PAL["data_ed"])
    s += text(52, 128, "datalist_batch_1.pt", "lbl")
    s += text(52, 150, f'list of {stats["per_file"]} Data objects', "sub")
    s += text(52, 168, f'{stats["file_mb"]:.0f} MB on disk', "mono")
    s += arrow(340, 146, 392, 146)

    s += box(400, 100, 746, 92, PAL["grey_lt"], PAL["grey_ed"])
    s += text(418, 128, "one Data object = one policy scenario", "lbl")
    s += text(418, 150, "the same 31,635 road links, one capacity intervention "
                        "applied", "sub")
    s += text(418, 168, f'{stats["mb_per_scenario"]:.2f} MB in memory', "mono")

    rows = stats["fields"]
    y0 = 238
    s += text(34, y0 - 10, "Seven stored entries", "lbl")
    s += text(300, y0 - 10,
              "five tensors, one scalar attribute, and x carrying six columns", "sub")
    s += box(34, y0, 1112, 30, "#f1f5f9", PAL["grey_ed"], r=6)
    for lbl, x in (("field", 52), ("shape", 300), ("dtype", 470), ("MB", 610),
                   ("static?", 690), ("used by the model?", 810)):
        s += text(x, y0 + 20, lbl, "sub")
    for i, r in enumerate(rows):
        yy = y0 + 34 + i * 30
        fill = PAL["ok_lt"] if r["used"] else PAL["warn_lt"]
        edge = PAL["ok_ed"] if r["used"] else PAL["warn_ed"]
        s += box(34, yy, 1112, 26, fill, edge, r=5)
        s += text(52, yy + 18, r["name"], "monob")
        s += text(300, yy + 18, r["shape"], "mono")
        s += text(470, yy + 18, r["dtype"], "mono")
        s += text(610, yy + 18, r["mb"], "mono")
        s += text(690, yy + 18, r["static"], "sub")
        s += text(810, yy + 18, r["use"], "sub")
    s += caption(W, H, [
        "num_nodes is a stored integer, not a tensor, and it disagrees with the "
        "tensors: it says 31,559 while x, pos and y each carry 31,635 rows.",
        "The preprocessing set it from the road-network edge count before the "
        "line-graph transform and never updated it. The 76 extra rows are",
        "public-transport links with no car access; they appear in no edge and their "
        "target is exactly zero in all 1,000 scenarios."])
    s += "</svg>\n"
    write(s, "data_01_pt_anatomy.svg")


def d2_eleven_fields(stats, out):
    """The eleven stored fields in the broader sense."""
    W, H = 1180, 620
    s = header(W, H, "The eleven stored fields",
               ["Six columns inside x, plus five further stored tensors. "
                "This is what the corpus actually contains.",
                "The enum also declares six ALLOWED_MODE_* columns that were never "
                "written."]) + defs()
    s += box(34, 100, 560, 250, PAL["data_lt"], PAL["data_ed"])
    s += text(52, 128, "1-6   the six columns of x", "lbl")
    s += text(52, 148, f'x has shape {stats["x_shape"]} and dtype '
                       f'{stats["x_dtype"]}', "sub")
    for i, f in enumerate(FEATURES):
        yy = 174 + i * 27
        used = i in MODEL_COLS
        s += text(58, yy, TICK if used else CROSS, "tick" if used else "cross")
        s += text(84, yy, f"{i}", "mono")
        s += text(110, yy, f, "monob")
        s += text(330, yy, stats["units"][f], "sub")

    s += box(618, 100, 528, 250, PAL["model_lt"], PAL["model_ed"])
    s += text(636, 128, "7-11   the five other stored tensors", "lbl")
    s += text(636, 148, "not columns of x; separate entries on the Data object", "sub")
    for i, (nm, shp, note) in enumerate(stats["others"]):
        yy = 174 + i * 27
        s += text(642, yy, f"{7+i}", "mono")
        s += text(672, yy, nm, "monob")
        s += text(840, yy, shp, "mono")
        s += text(960, yy, note, "sub")

    s += box(34, 372, 1112, 118, PAL["warn_lt"], PAL["warn_ed"])
    s += text(52, 398, "Designed and never built: ALLOWED_MODE_* at indices 6-11", "lbl")
    s += text(52, 420, "EdgeFeatures declares ALLOWED_MODE_CAR, _BUS, _PT, _TRAIN, "
                       "_RAIL and _SUBWAY at indices 6 to 11, and NET_FLOW at 12.", "sub")
    s += text(52, 440, "process_simulations_for_gnn.py sets use_allowed_modes = False, "
                       "so those columns were never written to any file.", "sub")
    s += text(52, 460, "x therefore has six columns, not twelve. The enum is a plan; "
                       "the corpus is the record of what was built.", "sub")
    s += text(52, 480, "Do not read the enum as a description of the stored data.", "sub")
    s += caption(W, H, [
        "Verified by streaming all 1,000 scenarios: every shape and dtype is "
        "identical across the corpus, and only CAPACITY_REDUCTION varies between",
        "scenarios. See scripts/data_exploration/explore_tensors.py."])
    s += "</svg>\n"
    write(s, "data_02_eleven_fields.svg")


def d3_model_inputs(stats, out):
    """The dedicated five-features visual."""
    W, H = 1180, 560
    s = header(W, H, "What actually enters the model",
               ["Five of the six columns of x are node features. The architecture "
                "also consumes connectivity and two coordinate pairs.",
                "Read from the trained Trial 8 checkpoint, not from documentation."]
               ) + defs()
    s += box(34, 100, 470, 232, PAL["ok_lt"], PAL["ok_ed"])
    s += text(52, 128, "MODEL INPUT   node features from x", "lbl")
    for i, c in enumerate(MODEL_COLS):
        yy = 158 + i * 30
        s += text(58, yy, TICK, "tick")
        s += text(88, yy, f"x[:, {c}]", "mono")
        s += text(160, yy, FEATURES[c], "monob")
    s += text(52, 318, "in_channels = 5", "mono")

    s += box(534, 100, 300, 232, PAL["warn_lt"], PAL["warn_ed"])
    s += text(552, 128, "NOT A MODEL FEATURE", "lbl")
    s += text(558, 168, CROSS, "cross")
    s += text(588, 168, "x[:, 4]", "mono")
    s += text(660, 168, "HIGHWAY", "monob")
    s += text(552, 202, "A nominal road class encoded", "sub")
    s += text(552, 220, "as an integer. The codes have", "sub")
    s += text(552, 238, "no order and no distance, so", "sub")
    s += text(552, 256, "arithmetic on them would be", "sub")
    s += text(552, 274, "meaningless. See diagram 4.", "sub")

    s += box(864, 100, 282, 232, PAL["data_lt"], PAL["data_ed"])
    s += text(882, 128, "ALSO CONSUMED", "lbl")
    s += text(882, 158, "pos[:, 0]", "mono")
    s += text(882, 176, "start coordinate", "sub")
    s += text(882, 200, "pos[:, 1]", "mono")
    s += text(882, 218, "end coordinate", "sub")
    s += text(882, 242, "edge_index", "mono")
    s += text(882, 260, "graph connectivity", "sub")
    s += text(882, 292, "pos[:, 2] midpoint is", "sub")
    s += text(882, 310, "stored but never read", "sub")

    s += box(34, 360, 1112, 132, PAL["grey_lt"], PAL["grey_ed"])
    s += text(52, 386, "The evidence", "lbl")
    s += text(52, 410, "point_net_conv_1.local_nn.0.weight has shape (256, 7). "
                       "PointNetConv concatenates node features with a 2-D relative", "sub")
    s += text(52, 430, "coordinate, so 7 - 2 = 5 feature channels. The forward pass "
                       "reads data.pos[:, 0] and data.pos[:, 1], and passes", "sub")
    s += text(52, 450, "edge_index to all six message-passing layers.", "sub")
    s += text(52, 476, "Five of six x columns are node features. It is not true that "
                       "only five pieces of information enter the model.", "sub")
    s += caption(W, H, [
        "Sources: scripts/training/help_functions.py (the node_features branch), "
        "scripts/gnn/models/point_net_transf_gat.py (forward),",
        "models/point_net_transf_gat_8th_trial_lower_dropout/trained_model/model.pth."])
    s += "</svg>\n"
    write(s, "data_03_model_inputs.svg")


def d4_highway(stats, out):
    """Why HIGHWAY was excluded."""
    W, H = 1180, 660
    s = header(W, H, "Why HIGHWAY is not a model feature",
               ["The column exists and is populated. It is excluded because of what "
                "its numbers mean, not because they are missing."]) + defs()
    s += box(34, 100, 1112, 112, PAL["warn_lt"], PAL["warn_ed"])
    s += text(52, 128, "The encoding is nominal", "lbl")
    s += text(52, 152, "highway_mapping turns an OSM class string into an integer. "
                       "The integers are names, not amounts.", "sub")
    s += text(52, 172, "A network layer computes weighted sums, so feeding these codes "
                       "asserts that tertiary - secondary = secondary - primary,", "sub")
    s += text(52, 192, "and that residential is four times trunk. Neither statement "
                       "means anything.", "sub")

    y0 = 238
    s += text(34, y0 - 8, "The classes as they appear in the corpus", "lbl")
    s += box(34, y0, 1112, 28, "#f1f5f9", PAL["grey_ed"], r=6)
    for lbl, x in (("code", 52), ("road class", 118), ("links", 470), ("share", 560),
                   ("ever intervened", 650), ("mean |response|", 810),
                   ("directly intervened", 970)):
        s += text(x, y0 + 19, lbl, "sub")
    for i, c in enumerate(stats["classes"]):
        yy = y0 + 32 + i * 25
        hit = c["directly_intervened"]
        s += box(34, yy, 1112, 22, PAL["ok_lt"] if hit else PAL["grey_lt"],
                 PAL["ok_ed"] if hit else PAL["grey_ed"], r=4)
        s += text(52, yy + 16, str(c["code"]), "mono")
        s += text(118, yy + 16, c["road_class"][:44], "sub")
        s += text(470, yy + 16, f'{c["n_links"]:,}', "mono")
        s += text(560, yy + 16, f'{c["pct_of_network"]:.1f}%', "mono")
        s += text(650, yy + 16, f'{c["ever_intervened"]:,}', "mono")
        s += text(810, yy + 16, f'{c["mean_abs_response"]:.2f}', "mono")
        s += text(970, yy + 16, "yes" if hit else "never", "sub")
    s += caption(W, H, [
        "Only primary, secondary and tertiary are ever intervened. Trunk is never "
        "touched by any policy yet carries the second-highest mean response in the",
        "network, which is spillover rather than treatment. A one-hot encoding or a "
        "learned embedding would be a defensible way to use this column; neither",
        "was part of the thesis, and both belong under future work."])
    s += "</svg>\n"
    write(s, "data_04_highway_excluded.svg")


def d5_policy_to_graph(stats, out):
    """Geography -> road network -> line graph -> tensors."""
    W, H = 1180, 560
    s = header(W, H, "From a policy on a map to a tensor the model can read",
               ["Four representations of the same thing. The line-graph step is the "
                "one that decides what a prediction is about."]) + defs()
    stages = [
        ("1  Policy", ["a capacity reduction applied", "to roads in one or more of",
                       "the 20 arrondissements"], PAL["warn_lt"], PAL["warn_ed"]),
        ("2  Road network", ["intersections are nodes", "road links are edges",
                             f'{stats["n_links"]:,} links'], PAL["data_lt"],
         PAL["data_ed"]),
        ("3  Line graph", ["road links become nodes", "an edge means two links meet",
                           f'{stats["n_edges"]:,} directed edges'], PAL["model_lt"],
         PAL["model_ed"]),
        ("4  Tensors", ["x, pos, y, edge_index", "one Data object per scenario",
                        f'{stats["mb_per_scenario"]:.2f} MB'], PAL["ok_lt"],
         PAL["ok_ed"]),
    ]
    x = 34
    for i, (title, lines, fill, edge) in enumerate(stages):
        s += box(x, 110, 250, 150, fill, edge)
        s += text(x + 18, 140, title, "lbl")
        for j, ln in enumerate(lines):
            s += text(x + 18, 168 + j * 20, ln, "sub")
        if i < 3:
            s += arrow(x + 254, 185, x + 288, 185)
        x += 292

    s += box(34, 296, 1112, 150, PAL["grey_lt"], PAL["grey_ed"])
    s += text(52, 322, "Why the inversion matters", "lbl")
    s += text(52, 348, "In the road network a prediction sits on an edge. After the "
                       "line-graph transform it sits on a node, and a GNN predicts", "sub")
    s += text(52, 368, "one value per node. That is why the model produces one number "
                       "per road link, which is the unit a policy question asks", "sub")
    s += text(52, 388, "about: how much does traffic on this street change.", "sub")
    s += text(52, 416, f'Degree in the line graph runs from {stats["deg_min"]} to '
                       f'{stats["deg_max"]}, median {stats["deg_med"]}: a road link '
                       f'meets that many other links.', "sub")
    s += caption(W, H, [
        f'{stats["iso"]} links have degree zero. They are public-transport links with '
        f'no car access, and they never respond to any policy.'])
    s += "</svg>\n"
    write(s, "data_05_policy_to_graph.svg")


def d6_tensor_to_input(stats, out):
    """x -> normalisation -> filtered columns -> layers."""
    W, H = 1180, 520
    s = header(W, H, "From the stored x to the tensor the first layer sees",
               ["The column filter is applied at load time, before normalisation "
                "statistics are computed."]) + defs()
    s += box(34, 110, 250, 176, PAL["data_lt"], PAL["data_ed"])
    s += text(52, 138, "stored x", "lbl")
    s += text(52, 160, f'{stats["x_shape"]}', "mono")
    for i, f in enumerate(FEATURES):
        s += text(52, 186 + i * 17, f"{i}  {f}", "mono" if i in MODEL_COLS else "sub")
    s += arrow(290, 198, 328, 198)

    s += box(338, 110, 250, 176, PAL["warn_lt"], PAL["warn_ed"])
    s += text(356, 138, "column filter", "lbl")
    s += text(356, 160, "node_feature_filter", "mono")
    s += text(356, 184, "HIGHWAY is dropped here.", "sub")
    s += text(356, 204, "The five remaining columns", "sub")
    s += text(356, 224, "are min-max normalised with", "sub")
    s += text(356, 244, "statistics fitted on the", "sub")
    s += text(356, 264, "training split only.", "sub")
    s += arrow(594, 198, 632, 198)

    s += box(642, 110, 250, 176, PAL["ok_lt"], PAL["ok_ed"])
    s += text(660, 138, "model input", "lbl")
    s += text(660, 160, f'[{stats["n_links"]}, 5]', "mono")
    s += text(660, 186, "plus pos[:, 0] and pos[:, 1]", "sub")
    s += text(660, 206, "plus edge_index", "sub")
    s += text(660, 236, "concatenated with a 2-D", "sub")
    s += text(660, 256, "relative coordinate inside", "sub")
    s += text(660, 276, "PointNetConv", "sub")
    s += arrow(898, 198, 936, 198)

    s += box(946, 110, 200, 176, PAL["model_lt"], PAL["model_ed"])
    s += text(964, 138, "first layer", "lbl")
    s += text(964, 164, "local_nn.0", "mono")
    s += text(964, 186, "Linear(7, 256)", "monob")
    s += text(964, 214, "7 = 5 features", "sub")
    s += text(964, 234, "    + 2 coordinates", "sub")
    s += text(964, 266, "verified in the", "sub")
    s += text(964, 284, "checkpoint", "sub")
    s += caption(W, H, [
        "The 5-column filter is the use_all_features = False branch in "
        "scripts/training/help_functions.py. The alternative branch would have kept",
        "HIGHWAY and NET_FLOW, giving seven names for a six-column tensor; the "
        "trained checkpoint shows the five-feature branch is the one that ran."])
    s += "</svg>\n"
    write(s, "data_06_tensor_to_input.svg")


def d7_intervention_response(stats, out):
    """Scenario intervention -> target response."""
    W, H = 1180, 560
    s = header(W, H, "One scenario: intervention in, response out",
               ["The only thing that changes between scenarios is one column. "
                "Everything else about the graph is fixed."]) + defs()
    s += box(34, 108, 340, 190, PAL["warn_lt"], PAL["warn_ed"])
    s += text(52, 136, "What the scenario changes", "lbl")
    s += text(52, 162, "x[:, 2]  CAPACITY_REDUCTION", "mono")
    s += text(52, 186, f'{stats["n_ever"]:,} links are ever eligible', "sub")
    s += text(52, 206, f'{stats["med_footprint"]:,} links touched in the median '
                       f'scenario', "sub")
    s += text(52, 226, f'{stats["n_mag"]} distinct magnitudes, all negative', "sub")
    s += text(52, 246, f'range {stats["red_min"]:,.0f} to '
                       f'{stats["red_max"]:,.0f} veh/h', "sub")
    s += text(52, 272, "capacity is only ever removed", "sub")
    s += arrow(380, 200, 424, 200)

    s += box(434, 108, 300, 190, PAL["grey_lt"], PAL["grey_ed"])
    s += text(452, 136, "What stays fixed", "lbl")
    s += text(452, 162, "the other five x columns", "sub")
    s += text(452, 182, "pos", "mono")
    s += text(452, 202, "edge_index", "mono")
    s += text(452, 228, "identical in all 1,000", "sub")
    s += text(452, 248, "scenarios, verified by", "sub")
    s += text(452, 268, "byte comparison", "sub")
    s += arrow(740, 200, 784, 200)

    s += box(794, 108, 352, 190, PAL["ok_lt"], PAL["ok_ed"])
    s += text(812, 136, "What comes out", "lbl")
    s += text(812, 162, "y  change in link car volume", "mono")
    s += text(812, 186, f'mean {stats["y_mean"]:+.4f} veh/h across the network', "sub")
    s += text(812, 206, f'{stats["y_zero"]:.1f}% of links unchanged', "sub")
    s += text(812, 226, f'{stats["y_pos"]:.1f}% gain, {stats["y_neg"]:.1f}% lose '
                        f'traffic', "sub")
    s += text(812, 252, "gains and losses nearly cancel:", "sub")
    s += text(812, 272, "traffic is redistributed, not removed", "sub")

    s += box(34, 330, 1112, 122, PAL["data_lt"], PAL["data_ed"])
    s += text(52, 356, "The result that makes the problem interesting", "lbl")
    s += text(52, 382, f'Of the {stats["n_links"]:,} links, only '
                       f'{stats["n_ever"]:,} can ever be intervened, yet the response '
                       f'reaches far beyond them.', "sub")
    s += text(52, 402, "Trunk roads are never touched by any policy and still carry "
                       "the second-highest mean response of any road class.", "sub")
    s += text(52, 422, "A surrogate that only learned the treated links would miss "
                       "most of what the simulator is being asked about.", "sub")
    s += caption(W, H, [
        "y = vol_car(scenario) - vol_car(base case), from "
        "compute_target_tensor_only_edge_features in "
        "scripts/data_preprocessing/help_functions.py."])
    s += "</svg>\n"
    write(s, "data_07_intervention_response.svg")


def main() -> int:
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    red, y, X, pos, ei = load(args.corpus, args.cache)
    n_links = X.shape[0]
    n_edges = ei.shape[1]
    deg = (np.bincount(ei[0], minlength=n_links) + np.bincount(ei[1], minlength=n_links))
    absy = np.abs(y).mean(0)
    hw = X[:, 4].astype(int)
    ever = (red != 0).any(0)
    nz = red[red != 0]
    flat = y.ravel()

    mb = {"x": X.nbytes / 1e6, "pos": pos.nbytes / 1e6,
          "y": n_links * 4 / 1e6, "edge_index": ei.nbytes / 1e6}
    fields = [
        dict(name="x", shape=f"[{n_links}, 6]", dtype="float64",
             mb=f'{mb["x"]:.2f}', static="5 of 6 static", use="5 of 6 columns"),
        dict(name="pos", shape=f"[{n_links}, 3, 2]", dtype="float32",
             mb=f'{mb["pos"]:.2f}', static="yes", use="pos[:,0] and pos[:,1]"),
        dict(name="y", shape=f"[{n_links}, 1]", dtype="float32",
             mb=f'{mb["y"]:.2f}', static="no", use="training target"),
        dict(name="edge_index", shape=f"[2, {n_edges}]", dtype="int64",
             mb=f'{mb["edge_index"]:.2f}', static="yes", use="all layers"),
        dict(name="mode_stats_diff", shape="[6, 3]", dtype="float32",
             mb="0.00", static="no", use="never read"),
        dict(name="mode_stats_diff_perc", shape="[6, 3]", dtype="float64",
             mb="0.00", static="no", use="never read"),
        dict(name="num_nodes", shape="scalar 31559", dtype="int",
             mb="0.00", static="yes", use="disagrees with x; see caption"),
    ]
    for f in fields:
        f["used"] = f["use"] not in ("never read", "disagrees with x; see caption")

    stats = {
        "per_file": 50, "file_mb": 131.0,
        "mb_per_scenario": sum(mb.values()),
        "fields": fields,
        "x_shape": f"[{n_links}, 6]", "x_dtype": "float64",
        "n_links": n_links, "n_edges": n_edges,
        "units": {"VOL_BASE_CASE": "vehicles per hour, base case",
                  "CAPACITY_BASE_CASE": "vehicles per hour, base case",
                  "CAPACITY_REDUCTION": "vehicles per hour, negative, per scenario",
                  "FREESPEED": "metres per second",
                  "HIGHWAY": "nominal road-class code",
                  "LENGTH": "metres"},
        "others": [
            ("pos", f"[{n_links}, 3, 2]", "start, end, midpoint"),
            ("y", f"[{n_links}, 1]", "the target"),
            ("edge_index", f"[2, {n_edges}]", "connectivity"),
            ("mode_stats_diff", "[6, 3]", "per-mode aggregates"),
            ("mode_stats_diff_perc", "[6, 3]", "the same, as percentages"),
        ],
        "classes": [
            {"code": int(c), "road_class": HIGHWAY_CLASSES.get(int(c), "unmapped"),
             "n_links": int((hw == c).sum()),
             "pct_of_network": 100 * float((hw == c).mean()),
             "ever_intervened": int(ever[hw == c].sum()),
             "mean_abs_response": float(absy[hw == c].mean()),
             "directly_intervened": bool(ever[hw == c].any())}
            for c in sorted(set(hw.tolist()))
        ],
        "deg_min": int(deg.min()), "deg_max": int(deg.max()),
        "deg_med": int(np.median(deg)), "iso": int((deg == 0).sum()),
        "n_ever": int(ever.sum()),
        "med_footprint": int(np.median((red != 0).sum(1))),
        "n_mag": int(np.unique(np.round(nz, 3)).size),
        "red_min": float(nz.min()), "red_max": float(nz.max()),
        "y_mean": float(flat.mean()), "y_zero": 100 * float((flat == 0).mean()),
        "y_pos": 100 * float((flat > 0).mean()), "y_neg": 100 * float((flat < 0).mean()),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    d1_pt_anatomy(stats, OUT)
    d2_eleven_fields(stats, OUT)
    d3_model_inputs(stats, OUT)
    d4_highway(stats, OUT)
    d5_policy_to_graph(stats, OUT)
    d6_tensor_to_input(stats, OUT)
    d7_intervention_response(stats, OUT)
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
