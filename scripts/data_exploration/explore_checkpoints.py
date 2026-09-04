#!/usr/bin/env python
"""Recover architecture facts from the checkpoints, and metrics from the trials.

Architecture is derived from tensor shapes alone, so this runs without importing
the model class and without a GPU. Metrics are read from whichever file the trial
actually wrote -- the filename is not consistent across trials, which is itself
worth knowing.

Trials that never recorded test metrics are reported as such. Nothing is inferred.

Usage:
    python scripts/data_exploration/explore_checkpoints.py
    python scripts/data_exploration/explore_checkpoints.py --json out.json
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent

#: Trials wrote their test metrics under different filenames.
METRIC_FILES = ("test_evaluation_complete.json", "test_results.json")

#: Heads that distinguish one architecture variant from another.
BACKBONE = {"point_net_conv_1", "point_net_conv_2", "gat_graph_layers", "gat_final"}


def load_state_dict(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if isinstance(obj, (dict, OrderedDict)) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, (dict, OrderedDict)):
        return {}
    return {k: v for k, v in obj.items() if hasattr(v, "shape")}


def architecture(sd) -> dict:
    """Everything derivable from tensor shapes."""
    if not sd:
        return {}
    out = {"params": sum(int(v.numel()) for v in sd.values()), "tensors": len(sd)}
    first = "point_net_conv_1.local_nn.0.weight"
    if first in sd:
        # PointNetConv concatenates 2 relative coordinates onto the features.
        out["in_channels"] = int(sd[first].shape[1]) - 2
    out["transformer_layers"] = sum(1 for k in sd if "lin_key.weight" in k)
    if "gat_final.lin.weight" in sd:
        out["output_dim"] = int(sd["gat_final.lin.weight"].shape[0])
    heads = sorted({k.split(".")[0] for k in sd} - BACKBONE)
    out["extra_heads"] = heads
    return out


def metrics_for(trial: str) -> tuple[dict | None, str | None]:
    d = REPO / "results" / "trials" / trial
    for name in METRIC_FILES:
        p = d / name
        if p.exists():
            j = json.loads(p.read_text(encoding="utf-8"))
            tm = j.get("test_metrics", j)
            got = {k: tm.get(k) for k in ("r2", "r2_score", "mae", "rmse")
                   if tm.get(k) is not None}
            n = (j.get("num_test_samples")
                 or j.get("statistics", {}).get("n_samples"))
            if n:
                got["n_test_nodes"] = n
                got["n_test_graphs"] = round(n / 31635, 3)
            return (got or None), name
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=REPO / "models")
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--json", type=Path, help="also write the inventory as JSON")
    args = ap.parse_args()

    rows = []
    for d in sorted(p for p in args.models.iterdir() if p.is_dir()):
        cks = sorted(d.rglob("*.pth"))
        arch = architecture(load_state_dict(cks[0])) if cks else {}
        met, src = metrics_for(d.name)
        rows.append({
            "trial": d.name,
            "checkpoints": [str(c.relative_to(REPO)) for c in cks],
            "checkpoint_mb": round(sum(c.stat().st_size for c in cks) / 1e6, 2),
            "architecture": arch,
            "test_metrics": met,
            "metrics_source": src,
        })

    print(f"{'trial':46s}{'MB':>7}{'params':>11}{'in':>4}{'heads':>28}{'R2':>9}{'graphs':>8}")
    for r in rows:
        a, m = r["architecture"], r["test_metrics"] or {}
        r2 = m.get("r2", m.get("r2_score"))
        print(f"{r['trial']:46s}{r['checkpoint_mb']:7.1f}{a.get('params',0):11,}"
              f"{a.get('in_channels','-'):>4}"
              f"{(','.join(a.get('extra_heads',[])) or 'gat_final'):>28}"
              f"{(f'{r2:.4f}' if r2 is not None else 'not rec.'):>9}"
              f"{m.get('n_test_graphs','-'):>8}")

    print("\n=== architecture variants ===")
    var = {}
    for r in rows:
        a = r["architecture"]
        var.setdefault((a.get("params"), tuple(a.get("extra_heads", [])),
                        a.get("output_dim")), []).append(r["trial"])
    for (params, heads, out_dim), trials in sorted(var.items(), key=lambda kv: -len(kv[1])):
        print(f"  {params:,} params | head={','.join(heads) or 'gat_final'} "
              f"| out={out_dim}")
        for t in trials:
            print(f"      {t}")

    missing = [r["trial"] for r in rows if not r["test_metrics"]]
    print(f"\n{len(rows) - len(missing)}/{len(rows)} trials have recorded test metrics.")
    if missing:
        print("checkpoint retained; verified test metrics not recorded:")
        for t in missing:
            print(f"  {t}")

    splits = {}
    for r in rows:
        g = (r["test_metrics"] or {}).get("n_test_graphs")
        if g:
            splits.setdefault(g, []).append(r["trial"])
    if len(splits) > 1:
        print("\nWARNING: trials were scored on different test splits; "
              "R2 is not comparable across them (see CORRIGENDUM C9):")
        for g, ts in sorted(splits.items()):
            print(f"  {g:g} graphs: {', '.join(t.replace('point_net_transf_gat_','') for t in ts)}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8",
                             newline="\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
