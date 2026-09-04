#!/usr/bin/env python
"""The eleven stored fields as one reference table: shape, dynamic, model input.

The table a reader wants before touching the corpus. Every row is read from
tensor_anatomy.json and feature_statistics.json rather than typed in, so it cannot
drift away from the data it describes.

    python scripts/figure_generation/generate_field_reference.py

Output: docs/figures/portfolio/10_field_reference.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import portfolio_style as ps  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = REPO / "docs" / "figures" / "portfolio"
ASSETS = REPO / "docs" / "portfolio_data_story" / "assets"

TICK, CROSS = "✓", "✗"

#: Why a field is or is not an input, in the fewest words that stay true.
REASON = {
    "x[:, 4]": "nominal codes",
    "pos": "slices 0 and 1",
    "y": "training target",
    "edge_index": "all six layers",
    "mode_stats_diff": "never read",
    "mode_stats_diff_perc": "never read",
}
MEANING = {
    "VOL_BASE_CASE": "car volume before any policy",
    "CAPACITY_BASE_CASE": "what the street could carry",
    "CAPACITY_REDUCTION": "capacity the policy removes",
    "FREESPEED": "free-flow speed",
    "HIGHWAY": "OSM road class",
    "LENGTH": "physical length",
    "pos": "start / end / midpoint, WGS84",
    "y": "change in link volume",
    "edge_index": "which links meet",
    "mode_stats_diff": "per-mode aggregates",
    "mode_stats_diff_perc": "the same, as percentages",
}


def build_rows():
    anat = json.loads((ASSETS / "tensor_anatomy.json").read_text(encoding="utf-8"))
    stats = json.loads((ASSETS / "feature_statistics.json").read_text(encoding="utf-8"))
    shapes = {f["name"]: f["shape"] for f in anat["stored_fields"]}
    dtypes = {f["name"]: f["dtype"].replace("torch.", "") for f in anat["stored_fields"]}

    rows = []
    for r in stats["static_vs_dynamic"]:
        field = r["field"]
        if field.startswith("x[:, "):
            slot, name = field.split("  ", 1)
            shape, dtype = shapes["x"], dtypes["x"]
            group = "x"
        else:
            slot, name = "", field
            shape, dtype = shapes[field], dtypes[field]
            group = "other"
        used = r["model_usage"] not in ("not used", "not read by any training or "
                                        "evaluation code")
        rows.append(dict(slot=slot, name=name, shape=shape, dtype=dtype,
                         dynamic=r["static_or_dynamic"] == "dynamic",
                         used=used, group=group,
                         reason=REASON.get(slot or name, ""),
                         meaning=MEANING.get(name, "")))
    return rows


def main() -> int:
    ps.apply()
    rows = build_rows()

    fig = plt.figure(figsize=(14.0, 10.2))
    ax = fig.add_axes([0, 0, 1, 1]); ps.bare(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    X_SLOT, X_NAME, X_MEAN, X_SHAPE, X_DYN, X_USE = 0.062, 0.128, 0.315, 0.535, 0.737, 0.838
    top, rh = 0.760, 0.0455

    # header ---------------------------------------------------------------------
    for label, x, ha in (("field", X_SLOT, "left"), ("what it holds", X_MEAN, "left"),
                         ("shape", X_SHAPE, "left"), ("dynamic?", X_DYN, "left"),
                         ("model input", X_USE, "left")):
        ax.text(x, top + 0.030, label, fontsize=10.4, color=ps.FAINT, ha=ha,
                va="baseline")
    ax.plot([0.052, 0.952], [top + 0.016, top + 0.016], color=ps.HAIR, lw=1.2)

    y = top
    for i, r in enumerate(rows):
        if i == 6:                      # the boundary between x and everything else
            y -= 0.024
            ax.plot([0.052, 0.952], [y + 0.030, y + 0.030], color=ps.HAIR, lw=1.2)
            ax.text(0.062, y + 0.040, "alongside x — five further tensors",
                    fontsize=10.4, color=ps.FAINT, va="baseline")
            y -= 0.016
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.052, y - 0.016), 0.900, rh - 0.004,
                                   facecolor=ps.WASH, edgecolor="none", zorder=0))
        if r["slot"]:
            ax.text(X_SLOT, y, r["slot"], fontsize=10.6, color=ps.FAINT,
                    family="monospace", va="baseline")
        ax.text(X_NAME, y, r["name"], fontsize=12.0, color=ps.INK,
                family="monospace", va="baseline",
                fontweight="600" if r["used"] else "400")
        ax.text(X_MEAN, y, r["meaning"], fontsize=10.4, color=ps.MUTED, va="baseline")
        ax.text(X_SHAPE, y, "[" + ", ".join(f"{d:,}" for d in r["shape"]) + "]",
                fontsize=10.6, color=ps.BODY, family="monospace", va="baseline")
        ax.text(X_SHAPE + 0.112, y, r["dtype"], fontsize=9.2, color=ps.FAINT,
                family="monospace", va="baseline")

        if r["dynamic"]:
            ax.add_patch(FancyBboxPatch((X_DYN - 0.006, y - 0.011), 0.052, 0.028,
                                        boxstyle="round,pad=0.002,rounding_size=0.006",
                                        facecolor="#FFF7ED", edgecolor=ps.AMBER, lw=0.9))
            ax.text(X_DYN + 0.020, y, "yes", fontsize=10.4, color=ps.AMBER,
                    ha="center", va="baseline", fontweight="600")
        else:
            ax.text(X_DYN + 0.020, y, "no", fontsize=10.4, color=ps.FAINT,
                    ha="center", va="baseline")

        mark, col = (TICK, ps.GREEN) if r["used"] else (CROSS, ps.RED)
        ax.text(X_USE, y, mark, fontsize=13.5, color=col, va="baseline",
                fontweight="700", family="DejaVu Sans")
        if r["reason"]:
            ax.text(X_USE + 0.026, y, r["reason"], fontsize=9.8,
                    color=ps.MUTED if r["used"] else ps.RED, va="baseline")
        y -= rh

    # the bracket marking the six columns that live inside one tensor --------------
    y_top, y_bot = top + 0.014, top - 5 * rh - 0.014
    ax.plot([0.044, 0.044], [y_bot, y_top], color=ps.BLUE, lw=1.6)
    for yy in (y_top, y_bot):
        ax.plot([0.044, 0.052], [yy, yy], color=ps.BLUE, lw=1.6)
    ax.text(0.036, (y_top + y_bot) / 2, "x  [31,635, 6]", fontsize=10.6, color=ps.BLUE,
            rotation=90, ha="center", va="center", family="monospace",
            fontweight="600")

    ps.title_block(
        fig, "Eleven stored fields, and what each one is for",
        "Six columns inside x, plus five further tensors. Only one column changes "
        "between scenarios; everything else about\nthe network is byte-identical "
        "across all 1,000.", y=0.962, size=23)

    ps.footnote(fig, [
        "Shapes and dtypes are read from tensor_anatomy.json, the static/dynamic and "
        "model-usage columns from feature_statistics.json — both produced by streaming "
        "the corpus, so this table cannot drift from the data.",
        "HIGHWAY is populated and correct; it is excluded because its integers are "
        "road-class labels, and a network that adds and multiplies its inputs would "
        "invent an order they do not have.",
        "pos contributes its start and end slices; the midpoint is stored and only ever "
        "plotted. The EdgeFeatures enum also declares six ALLOWED_MODE_* columns at "
        "indices 6-11 that use_allowed_modes = False left unwritten."], y=0.150)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "10_field_reference.png")
    plt.close(fig)
    print(f"  wrote 10_field_reference.png  ({len(rows)} fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
