#!/usr/bin/env python
"""Verify the stored schema of the corpus against the actual `.pt` objects.

Every other exploration script trusts a cache built from scenario 0 on the claim
that the static columns, positions and topology never change. This script is the
one that checks that claim, by streaming all 1,000 scenarios and comparing each
against the first. It also reads the two auxiliary tensors nothing else touches,
and pins down the node-count discrepancy between `x` and the stored `num_nodes`.

    python scripts/data_exploration/explore_tensors.py --corpus DIR --cache DIR

Writes tensor_anatomy.json and auxiliary_tensors.json to the web-asset directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from common import FEATURES, add_common_args  # noqa: E402
from common import _batch_files as batch_files  # noqa: E402

OUT = REPO / "docs" / "portfolio_data_story" / "assets"


def jdump(obj, name):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"  {name:34} {p.stat().st_size/1024:6.1f} KB")


def main() -> int:
    import torch

    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    files = batch_files(args.corpus)
    print(f"streaming {len(files)} batch files from {args.corpus}\n")

    first = None
    n_scen = 0
    per_file = []
    msd, msdp = [], []
    num_nodes_seen: set[int] = set()
    # Claims to falsify, each starts true and is cleared by a counter-example.
    static_constant = True
    pos_constant = True
    ei_constant = True
    tail_y_zero = True
    dyn_col_varies = False
    shapes_constant = True
    dtypes_constant = True
    first_red = None

    for f in files:
        graphs = torch.load(f, weights_only=False, map_location="cpu")
        per_file.append({"file": f.name, "scenarios": len(graphs),
                         "bytes": f.stat().st_size})
        for g in graphs:
            x = g.x.numpy()
            if first is None:
                first = {
                    "x": x.copy(), "pos": g.pos.numpy().copy(),
                    "ei": g.edge_index.numpy().copy(),
                    "shapes": {k: tuple(g[k].shape) for k in g.keys()
                               if torch.is_tensor(g[k])},
                    "dtypes": {k: str(g[k].dtype) for k in g.keys()
                               if torch.is_tensor(g[k])},
                    "num_nodes": int(g.num_nodes),
                    "keys": sorted(k for k in g.keys()),
                }
                first_red = x[:, 2].copy()
            else:
                shapes = {k: tuple(g[k].shape) for k in g.keys() if torch.is_tensor(g[k])}
                dtypes = {k: str(g[k].dtype) for k in g.keys() if torch.is_tensor(g[k])}
                shapes_constant &= shapes == first["shapes"]
                dtypes_constant &= dtypes == first["dtypes"]
                # Every column except CAPACITY_REDUCTION must be byte-identical.
                cols = [c for c in range(x.shape[1]) if c != 2]
                static_constant &= np.array_equal(x[:, cols], first["x"][:, cols])
                pos_constant &= np.array_equal(g.pos.numpy(), first["pos"])
                ei_constant &= np.array_equal(g.edge_index.numpy(), first["ei"])
                dyn_col_varies |= not np.array_equal(x[:, 2], first_red)

            num_nodes_seen.add(int(g.num_nodes))
            nn = int(g.num_nodes)
            tail_y_zero &= bool(np.all(g.y.numpy()[nn:] == 0))
            msd.append(g.mode_stats_diff.numpy().astype(np.float64))
            msdp.append(g.mode_stats_diff_perc.numpy().astype(np.float64))
            n_scen += 1
        del graphs

    msd = np.stack(msd)      # [S, 6, 3]
    msdp = np.stack(msdp)
    x0, ei = first["x"], first["ei"]
    n_rows = x0.shape[0]
    nn = first["num_nodes"]

    deg = (np.bincount(ei[0], minlength=n_rows)
           + np.bincount(ei[1], minlength=n_rows))
    isolated = int((deg == 0).sum())
    self_loops = int((ei[0] == ei[1]).sum())

    print(f"\n{n_scen:,} scenarios across {len(files)} files")
    print(f"  shapes constant across scenarios      : {shapes_constant}")
    print(f"  dtypes constant across scenarios      : {dtypes_constant}")
    print(f"  x static columns identical everywhere : {static_constant}")
    print(f"  pos identical everywhere              : {pos_constant}")
    print(f"  edge_index identical everywhere       : {ei_constant}")
    print(f"  CAPACITY_REDUCTION varies             : {dyn_col_varies}")
    print(f"  distinct stored num_nodes values      : {sorted(num_nodes_seen)}")
    print("\nnode accounting")
    print(f"  rows in x / pos / y : {n_rows:,}")
    print(f"  stored num_nodes    : {nn:,}   (shortfall {n_rows - nn})")
    print(f"  max edge_index + 1  : {int(ei.max()) + 1:,}")
    print(f"  isolated rows       : {isolated:,}")
    print(f"  self-loops          : {self_loops:,} of {ei.shape[1]:,} edges")
    print(f"  tail rows always y=0: {tail_y_zero}")

    anatomy = {
        "source": "train-data-v1 release, datalist_batch_1..20.pt",
        "n_scenario_files": len(files),
        "n_scenarios": n_scen,
        "per_file": per_file,
        "stored_fields": [
            {"name": k, "shape": list(first["shapes"][k]), "dtype": first["dtypes"][k],
             "bytes_per_scenario": int(np.prod(first["shapes"][k])
                                       * (8 if "64" in first["dtypes"][k] else 4))}
            for k in sorted(first["shapes"])
        ],
        "non_tensor_attributes": {"num_nodes": nn},
        "keys": first["keys"],
        "x_columns": FEATURES,
        "invariants_checked_over_all_scenarios": {
            "shapes_constant": bool(shapes_constant),
            "dtypes_constant": bool(dtypes_constant),
            "x_static_columns_identical": bool(static_constant),
            "pos_identical": bool(pos_constant),
            "edge_index_identical": bool(ei_constant),
            "capacity_reduction_varies": bool(dyn_col_varies),
            "distinct_num_nodes_values": sorted(num_nodes_seen),
        },
        "node_accounting": {
            "rows_in_x_pos_y": int(n_rows),
            "stored_num_nodes": nn,
            "shortfall": int(n_rows - nn),
            "max_edge_index_plus_one": int(ei.max()) + 1,
            "isolated_rows": isolated,
            "self_loops": self_loops,
            "edges": int(ei.shape[1]),
            "tail_rows_target_always_zero": bool(tail_y_zero),
            "note": ("The last 76 rows carry real feature and position values but appear "
                     "in no edge and have a target of exactly zero in every scenario. "
                     "num_nodes is stored as 31,559, which excludes them; x, pos and y "
                     "all have 31,635 rows, and the published evaluation counts of "
                     "31,635 per graph include them."),
        },
    }
    jdump(anatomy, "tensor_anatomy.json")

    # --- the two auxiliary tensors ------------------------------------------------
    aux = {
        "mode_stats_diff": {
            "shape": [6, 3], "dtype": first["dtypes"]["mode_stats_diff"],
            "varies_across_scenarios": bool(msd.std(0).max() > 0),
            "per_cell": [[{"mean": round(float(msd[:, i, j].mean()), 6),
                           "std": round(float(msd[:, i, j].std()), 6),
                           "min": round(float(msd[:, i, j].min()), 6),
                           "max": round(float(msd[:, i, j].max()), 6)}
                          for j in range(3)] for i in range(6)],
        },
        "mode_stats_diff_perc": {
            "shape": [6, 3], "dtype": first["dtypes"]["mode_stats_diff_perc"],
            "varies_across_scenarios": bool(msdp.std(0).max() > 0),
            "n_exactly_minus_100": int((msdp == -100).sum()),
            "n_cells_total": int(msdp.size),
            "per_cell": [[{"mean": round(float(msdp[:, i, j].mean()), 6),
                           "std": round(float(msdp[:, i, j].std()), 6),
                           "min": round(float(msdp[:, i, j].min()), 6),
                           "max": round(float(msdp[:, i, j].max()), 6),
                           "share_minus_100": round(
                               float((msdp[:, i, j] == -100).mean()), 6)}
                          for j in range(3)] for i in range(6)],
        },
        "model_usage": ("Neither tensor is read by the training or evaluation code. "
                        "See docs/data_exploration/auxiliary_tensors.md."),
    }
    jdump(aux, "auxiliary_tensors.json")

    print(f"\nmode_stats_diff      varies across scenarios: "
          f"{aux['mode_stats_diff']['varies_across_scenarios']}")
    print(f"mode_stats_diff_perc varies across scenarios: "
          f"{aux['mode_stats_diff_perc']['varies_across_scenarios']}")
    print(f"mode_stats_diff_perc cells exactly -100: "
          f"{aux['mode_stats_diff_perc']['n_exactly_minus_100']:,} "
          f"of {aux['mode_stats_diff_perc']['n_cells_total']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
