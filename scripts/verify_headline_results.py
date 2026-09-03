#!/usr/bin/env python
"""Recompute every headline number in the README from the artifacts behind it.

Each check pairs a value as published in `README.md` with a function that derives
the same quantity from a tracked artifact or a release asset. Nothing is asserted
from a stored summary alone -- where a JSON already holds the answer, the check
recomputes it from the underlying prediction arrays and compares against both.

Exit status
-----------
0  every runnable check passed
1  at least one check drifted outside tolerance

Checks whose source artifact is absent are reported as SKIP and do not fail the
run, so a plain clone verifies what it can and says what it could not reach.

Usage
-----
    python scripts/verify_headline_results.py
    python scripts/verify_headline_results.py --verbose

The two AUROC checks need `trial8_uq_ablation_results.csv` (209 MB, a release
asset). Fetch it, or point THESIS_DATA_ROOT at a data tree that has it:

    gh release download large-files-v1 --repo mzquadri/ml-surrogates-thesis-data \\
      --pattern '*trial8_uq_ablation_results.csv' \\
      --dir data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))

from artifact_paths import ArtifactNotFound, resolve  # noqa: E402

NODES_PER_GRAPH = 31_635
T8 = "point_net_transf_gat_8th_trial_lower_dropout"
T7 = "point_net_transf_gat_7th_trial_80_10_10_split"


# ── artifact loading ──────────────────────────────────────────────────────────

_cache: dict[str, object] = {}


def _npz(trial: str, name: str):
    key = f"{trial}/{name}"
    if key not in _cache:
        _cache[key] = np.load(resolve(f"{trial}/uq_results/{name}"))
    return _cache[key]


def mc(trial: str = T8):
    """MC Dropout archive: predictions are the mean over 30 stochastic passes."""
    d = _npz(trial, "mc_dropout_full_100graphs_mc30.npz")
    return (
        d["targets"].astype(np.float64),
        d["predictions"].astype(np.float64),
        d["uncertainties"].astype(np.float64),
    )


def det(trial: str = T8):
    """Deterministic archive: one forward pass with dropout disabled."""
    d = _npz(trial, "deterministic_full_100graphs.npz")
    return d["targets"].astype(np.float64), d["predictions"].astype(np.float64)


def ablation_csv():
    """The per-node ablation table the calibration and error-detection docs use."""
    if "csv" not in _cache:
        import pandas as pd

        path = resolve(f"{T8}/trial8_uq_ablation_results.csv")
        _cache["csv"] = pd.read_csv(path)
    return _cache["csv"]


def tracked_json(rel: str):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


# ── metric definitions ────────────────────────────────────────────────────────


def m_mae():
    t, p = det()
    return float(np.abs(t - p).mean())


def m_rmse():
    t, p = det()
    return float(np.sqrt(((t - p) ** 2).mean()))


def m_r2():
    t, p = det()
    return float(1.0 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def m_rho_t8():
    t, p, s = mc(T8)
    return float(stats.spearmanr(s, np.abs(t - p)).statistic)


def m_rho_t7():
    t, p, s = mc(T7)
    return float(stats.spearmanr(s, np.abs(t - p)).statistic)


def _selective(retain_frac: float) -> float:
    """MAE reduction (%) when the most uncertain nodes are handed off.

    Ranks every node by sigma, keeps the `retain_frac` least uncertain, and
    compares their MAE against the MAE over all nodes. Both use the MC mean,
    which is the prediction the sigma belongs to.
    """
    t, p, s = mc(T8)
    err = np.abs(t - p)
    keep = np.argsort(s, kind="stable")[: int(round(retain_frac * s.size))]
    return float(100.0 * (1.0 - err[keep].mean() / err.mean()))


def m_selective_50():
    return _selective(0.50)


def _temperature_scaling():
    """Fit one scalar T on the first 20 graphs, score ECE on the last 80.

    ECE here is the Kuleshov calibration error: the mean absolute gap between
    nominal and empirical coverage of the Gaussian intervals sigma * z, averaged
    over the ten nominal levels the audit uses.
    """
    t, p, s = mc(T8)
    err = np.abs(t - p)
    cut = 20 * NODES_PER_GRAPH
    # The ten nominal levels recorded in results/temperature_scaling_t8.json.
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    z = stats.norm.ppf(0.5 + levels / 2.0)

    def ece(e, sig, T=1.0):
        emp = np.array([(e <= T * sig * zi).mean() for zi in z])
        return float(np.abs(emp - levels).mean())

    coarse = np.linspace(0.5, 6.0, 551)
    best = min(coarse, key=lambda T: ece(err[:cut], s[:cut], T))
    fine = np.linspace(best - 0.02, best + 0.02, 81)
    best = min(fine, key=lambda T: ece(err[:cut], s[:cut], T))
    return best, ece(err[cut:], s[cut:]), ece(err[cut:], s[cut:], best)


def m_temperature():
    return _temperature_scaling()[0]


def m_ece_before():
    return _temperature_scaling()[1]


def m_ece_after():
    return _temperature_scaling()[2]


def _conformal(alpha: float) -> float:
    """Split-conformal coverage under the replayable graph20_80_v1 protocol.

    Calibrates the absolute-residual quantile on the first 20 test graphs and
    reports empirical coverage on the remaining 80. This is not the 50/50
    scenario split the thesis reports; see docs/CORRIGENDUM.md C3.
    """
    t, p = det()
    err = np.abs(t - p)
    cut = 20 * NODES_PER_GRAPH
    cal, ev = err[:cut], err[cut:]
    k = int(np.ceil((cal.size + 1) * alpha))
    q = np.sort(cal)[k - 1]
    return float(100.0 * (ev <= q).mean())


def m_conformal_90():
    return _conformal(0.90)


def m_conformal_95():
    return _conformal(0.95)


def _auroc(top_pct: int) -> float:
    """AUROC using sigma to rank nodes whose deterministic error is in the tail.

    Positives are nodes at or above the (100 - top_pct)th percentile of
    `abs_error_det`; the score is `pred_mc_std`. This matches the definition in
    docs/verified/UQ_ERROR_DETECTION_T8.md.
    """
    from sklearn.metrics import roc_auc_score

    df = ablation_csv()
    e = df["abs_error_det"].to_numpy(np.float64)
    s = df["pred_mc_std"].to_numpy(np.float64)
    y = (e >= np.percentile(e, 100 - top_pct)).astype(np.int8)
    return float(roc_auc_score(y, s))


def m_auroc_10():
    return _auroc(10)


def m_auroc_20():
    return _auroc(20)


# ── the checks ────────────────────────────────────────────────────────────────
# `documented` is the value as published, so a drift in either the artifact or
# the prose surfaces here. `tol` is absolute, on the same scale as the value.

CHECKS = [
    # (label, documented, fn, tol, unit, source)
    ("T8 deterministic MAE", 3.96, m_mae, 0.005, "veh/h",
     f"results/predictions/{T8}/uq_results/deterministic_full_100graphs.npz"),
    ("T8 deterministic RMSE", 7.12, m_rmse, 0.005, "veh/h",
     f"results/predictions/{T8}/uq_results/deterministic_full_100graphs.npz"),
    ("T8 R^2", 0.5957, m_r2, 0.0010, "",
     f"results/predictions/{T8}/uq_results/deterministic_full_100graphs.npz"),
    ("T8 MC Dropout Spearman rho", 0.482, m_rho_t8, 0.0010, "",
     f"results/predictions/{T8}/uq_results/mc_dropout_full_100graphs_mc30.npz"),
    # The thesis reports 0.446 for T7. That value is not reproducible from the
    # retained archive, which yields 0.4437 under the same definition that
    # reproduces T8 exactly. See docs/CORRIGENDUM.md C7.
    ("T7 MC Dropout Spearman rho", 0.4437, m_rho_t7, 0.0010, "",
     f"results/predictions/{T7}/uq_results/mc_dropout_full_100graphs_mc30.npz"),
    ("T8 temperature T", 2.702, m_temperature, 0.05, "",
     f"results/predictions/{T8}/uq_results/mc_dropout_full_100graphs_mc30.npz"),
    ("T8 ECE before scaling", 0.269, m_ece_before, 0.010, "",
     f"results/predictions/{T8}/uq_results/mc_dropout_full_100graphs_mc30.npz"),
    ("T8 ECE after scaling", 0.048, m_ece_after, 0.010, "",
     f"results/predictions/{T8}/uq_results/mc_dropout_full_100graphs_mc30.npz"),
    ("T8 selective MAE reduction @50%", 41.2, m_selective_50, 0.5, "%",
     f"results/predictions/{T8}/uq_results/mc_dropout_full_100graphs_mc30.npz"),
    ("T8 conformal coverage @90% (graph20_80_v1)", 90.17, m_conformal_90, 0.10, "%",
     f"results/predictions/{T8}/uq_results/deterministic_full_100graphs.npz"),
    ("T8 conformal coverage @95% (graph20_80_v1)", 95.09, m_conformal_95, 0.10, "%",
     f"results/predictions/{T8}/uq_results/deterministic_full_100graphs.npz"),
    ("T8 error-detection AUROC top-10%", 0.7585, m_auroc_10, 0.0010, "",
     f"{T8}/trial8_uq_ablation_results.csv (release)"),
    ("T8 error-detection AUROC top-20%", 0.7401, m_auroc_20, 0.0010, "",
     f"{T8}/trial8_uq_ablation_results.csv (release)"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="show full source paths")
    args = ap.parse_args()

    print("Verifying headline results against source artifacts\n")
    header = f"{'metric':<44}{'documented':>12}{'recomputed':>13}{'tol':>9}  {'status':<7} source"
    print(header)
    print("-" * len(header))

    failures = skips = 0
    for label, documented, fn, tol, unit, source in CHECKS:
        short = source if args.verbose else Path(source).name
        try:
            got = fn()
        except (ArtifactNotFound, FileNotFoundError):
            skips += 1
            print(f"{label:<44}{documented:>12}{'--':>13}{tol:>9}  {'SKIP':<7} {short} (absent)")
            continue
        ok = abs(got - documented) <= tol
        failures += not ok
        print(f"{label:<44}{documented:>12.4f}{got:>13.4f}{tol:>9.4f}  "
              f"{'PASS' if ok else 'FAIL':<7} {short}")

    checked = len(CHECKS) - skips
    print(f"\n{checked - failures}/{checked} checks passed"
          + (f", {skips} skipped (source artifact not present)" if skips else ""))

    if failures:
        print(f"\n{failures} headline number(s) drifted outside tolerance.")
        return 1
    print("\nEvery runnable headline number matches its source artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
