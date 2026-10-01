#!/usr/bin/env python
"""Does conformal coverage hold uniformly across the 20 Paris arrondissements, or
does it slip where interventions are heaviest?

Motivation
----------
`docs/CORRIGENDUM.md` C3 already states the caveat every conformal prediction paper
states: the 90%/95% coverage is *marginal* -- averaged over the whole test set --
not a per-scenario, per-link, or per-district guarantee. `results/conformal_conditional_coverage_t8.json`
already checks one conditioning variable (MC Dropout sigma decile) and finds coverage
does drift with it. Nobody has yet checked the geographic axis the experiment itself
was designed around: the README's own data story shows intervention rate and severity
vary sharply by arrondissement (25.9% to 48.1% intervened; the 17th absorbs 12.0% of
the network's total response across 6.9% of its links). If conformal coverage is going
to fail anywhere, a reviewer's first guess would be the districts hit hardest.

Method
------
Same split-conformal recipe as `scripts/evaluation/run_part3_calibration_audit.py`:
first 20% of rows (graphs 1-20, in the graph-major row order every artifact in this
repo uses) calibrate a global absolute-residual quantile at the 90%/95% nominal level;
the remaining 80% (graphs 21-100) are scored against it. This script changes only the
grouping variable at evaluation time -- by arrondissement instead of by sigma decile --
so the two analyses are directly comparable.

No retraining, no new inference: reads the same `trial8_uq_ablation_results.csv` the
calibration audit reads, plus the per-link arrondissement codes recovered by
`scripts/data_exploration/explore_arrondissements.py` (point-in-polygon of each link's
midpoint against `data/visualisation/districts_paris.geojson`).

Usage
-----
    python scripts/analysis/coverage_by_arrondissement.py --corpus DIR --cache DIR

Output: results/conformal_coverage_by_arrondissement.json
        docs/figures/results/06_coverage_by_arrondissement.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "scripts" / "data_exploration"))
sys.path.insert(0, str(REPO / "scripts" / "figure_generation"))
from artifact_paths import RELEASE_HINT, resolve  # noqa: E402
from explore_arrondissements import assign_districts  # noqa: E402
from thesis_style import COLORS  # noqa: E402

NODES_PER_GRAPH = 31_635
N_GRAPHS = 100
OUT_JSON = REPO / "results" / "conformal_coverage_by_arrondissement.json"
OUT_FIG = REPO / "docs" / "figures" / "results"


def load_uq_table() -> pd.DataFrame:
    csv = resolve(
        "point_net_transf_gat_8th_trial_lower_dropout/trial8_uq_ablation_results.csv",
        hint=RELEASE_HINT,
    )
    df = pd.read_csv(csv)
    required = {"target", "pred_mc_mean", "pred_mc_std"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert len(df) == NODES_PER_GRAPH * N_GRAPHS, (
        f"expected {NODES_PER_GRAPH * N_GRAPHS:,} rows (100 graphs), got {len(df):,}"
    )
    return df


def load_arrondissement_codes(corpus: Path, cache: Path) -> np.ndarray:
    """Per-link (31,635,) arrondissement code, 0 = outside all twenty polygons."""
    from common import load as load_corpus

    cache_file = cache / "arrondissement_of_link.npy"
    if cache_file.exists():
        return np.load(cache_file)

    _, _, _, pos, _ = load_corpus(corpus, cache)
    mid = pos[:, 2, :]
    codes, _, _ = assign_districts(mid[:, 0], mid[:, 1])
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, codes)
    return codes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True,
                     help="directory holding datalist_batch_*.pt")
    ap.add_argument("--cache", type=Path, required=True,
                     help="scratch directory for cached .npy arrays")
    args = ap.parse_args()

    print("Loading Trial 8 UQ table (100 test graphs) ...")
    df = load_uq_table()

    print("Resolving per-link arrondissement codes ...")
    codes_per_link = load_arrondissement_codes(args.corpus, args.cache)
    assert codes_per_link.shape == (NODES_PER_GRAPH,)
    codes = np.tile(codes_per_link, N_GRAPHS)
    assert codes.shape[0] == len(df)

    n = len(df)
    split = n // 5
    assert split % NODES_PER_GRAPH == 0, "20% split must land on a graph boundary"
    print(f"  Cal: graphs 1-{split // NODES_PER_GRAPH}   "
          f"Eval: graphs {split // NODES_PER_GRAPH + 1}-{n // NODES_PER_GRAPH}")

    y = df["target"].to_numpy(dtype=np.float64)
    yhat = df["pred_mc_mean"].to_numpy(dtype=np.float64)
    resid_abs = np.abs(y - yhat)

    cal_resid = resid_abs[:split]
    eval_resid = resid_abs[split:]
    eval_codes = codes[split:]

    results = {}
    for nominal in (0.90, 0.95):
        q = float(np.quantile(cal_resid, nominal))
        covered = eval_resid <= q
        rows = []
        for c in sorted(set(eval_codes.tolist())):
            m = eval_codes == c
            rows.append({
                "arrondissement": int(c),
                "n_nodes": int(m.sum()),
                "coverage": round(float(covered[m].mean()), 4),
                "mean_abs_residual": round(float(eval_resid[m].mean()), 4),
            })
        results[f"nominal_{int(nominal * 100)}"] = {
            "global_quantile": round(q, 5),
            "global_coverage": round(float(covered.mean()), 4),
            "by_arrondissement": rows,
        }
        worst = min(rows, key=lambda r: r["coverage"] if r["arrondissement"] != 0 else 1.0)
        print(f"  nominal {int(nominal*100)}%: global coverage "
              f"{results[f'nominal_{int(nominal*100)}']['global_coverage']:.4f}, "
              f"worst arrondissement {worst['arrondissement']} at {worst['coverage']:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "analysis": "conformal_coverage_by_arrondissement",
        "description": "Split-conformal coverage (first 20% of graphs calibrate, "
                        "remaining 80% evaluate), grouped by the arrondissement each "
                        "link's midpoint falls in. 0 = outside all twenty polygons.",
        "trial": "point_net_transf_gat_8th_trial_lower_dropout",
        "n_cal_nodes": int(split),
        "n_eval_nodes": int(n - split),
        **results,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {OUT_JSON.relative_to(REPO)}")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4.6))
    rows90 = [r for r in results["nominal_90"]["by_arrondissement"] if r["arrondissement"] != 0]
    rows90.sort(key=lambda r: r["arrondissement"])
    rows95 = {r["arrondissement"]: r for r in results["nominal_95"]["by_arrondissement"]}
    x = np.arange(len(rows90))
    cov90 = [r["coverage"] for r in rows90]
    cov95 = [rows95[r["arrondissement"]]["coverage"] for r in rows90]
    labels = [str(r["arrondissement"]) for r in rows90]

    ax.bar(x - 0.19, cov90, width=0.38, label="90% nominal", color=COLORS["blue"])
    ax.bar(x + 0.19, cov95, width=0.38, label="95% nominal", color=COLORS["coral"])
    ax.axhline(0.90, color=COLORS["blue"], ls="--", lw=1, alpha=0.6)
    ax.axhline(0.95, color=COLORS["coral"], ls="--", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel("arrondissement")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(min(cov90 + cov95) - 0.03, 1.0)
    ax.set_title("Conformal coverage is marginal, not uniform across the city",
                 fontweight="600", color=COLORS["dgray"])
    ax.legend(loc="lower left", fontsize=8.5)
    fig.text(0.5, -0.05,
              "Dashed lines mark the nominal coverage level. Split-conformal coverage holds on average "
              "over the test set (see README); it is not guaranteed district by district.",
              ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / "06_coverage_by_arrondissement.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {(OUT_FIG / '06_coverage_by_arrondissement.png').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
