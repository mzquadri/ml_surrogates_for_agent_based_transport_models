#!/usr/bin/env python
"""Emit the uncertainty, experiment-timeline and calibration web assets.

Everything is read from tracked artifacts, so this runs from a plain clone with no
release downloads. Output is deterministic: fixed ordering, rounded floats, LF
line endings.

Usage:
    python scripts/data_exploration/build_uq_assets.py

Output: docs/portfolio_data_story/assets/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from artifact_paths import ArtifactNotFound, resolve  # noqa: E402

OUT = REPO / "docs" / "portfolio_data_story" / "assets"
T8 = "point_net_transf_gat_8th_trial_lower_dropout"
NPG = 31_635
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def jdump(obj, name):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"  {name:34s} {p.stat().st_size/1024:7.1f} KB")


def load_t8():
    m = np.load(resolve(f"{T8}/uq_results/mc_dropout_full_100graphs_mc30.npz"))
    d = np.load(resolve(f"{T8}/uq_results/deterministic_full_100graphs.npz"))
    return (m["targets"].astype(np.float64), m["predictions"].astype(np.float64),
            m["uncertainties"].astype(np.float64), d["predictions"].astype(np.float64))


def build_calibration(t, pmc, sig):
    """Reliability curve before and after temperature scaling, graph20_80_v1."""
    err = np.abs(pmc - t)
    cut = 20 * NPG
    lv = np.array(LEVELS)
    z = stats.norm.ppf(0.5 + lv / 2)

    def cov(e, s, T=1.0):
        return np.array([(e <= T * s * zi).mean() for zi in z])

    def ece(e, s, T=1.0):
        return float(np.abs(cov(e, s, T) - lv).mean())

    coarse = np.linspace(0.5, 6.0, 551)
    best = min(coarse, key=lambda T: ece(err[:cut], sig[:cut], T))
    best = min(np.linspace(best - 0.02, best + 0.02, 81),
               key=lambda T: ece(err[:cut], sig[:cut], T))
    ev_e, ev_s = err[cut:], sig[cut:]
    return {
        "protocol": "graph20_80_v1",
        "protocol_note": "temperature fitted on the first 20 test graphs, scored on "
                         "the remaining 80. Not the 50/50 scenario split the thesis "
                         "reports; see docs/CORRIGENDUM.md C3.",
        "temperature": round(float(best), 4),
        "nominal_levels": LEVELS,
        "empirical_coverage_raw": [round(float(v), 5) for v in cov(ev_e, ev_s)],
        "empirical_coverage_scaled": [round(float(v), 5) for v in cov(ev_e, ev_s, best)],
        "ece_before": round(ece(ev_e, ev_s), 5),
        "ece_after": round(ece(ev_e, ev_s, best), 5),
        "calibration_nodes": int(cut),
        "evaluation_nodes": int(err.size - cut),
    }


def build_selective(t, pmc, sig):
    """Risk-coverage curve: MAE against the fraction of links retained."""
    err = np.abs(pmc - t)
    order = np.argsort(sig, kind="stable")
    base = float(err.mean())
    fracs = [round(f, 2) for f in np.arange(0.05, 1.01, 0.05)]
    rows = []
    for f in fracs:
        keep = order[: int(round(f * err.size))]
        m = float(err[keep].mean())
        rows.append({"retained": f, "mae": round(m, 4),
                     "mae_reduction_pct": round(100 * (1 - m / base), 3)})
    return {
        "baseline_mae_vehh": round(base, 4),
        "ranking_signal": "MC Dropout sigma, ascending",
        "note": "uses only the ordering of sigma, not its magnitude, which is why it "
                "works despite sigma being uncalibrated",
        "curve": rows,
    }


def build_conformal(t, pdet):
    """Split-conformal coverage and interval width, graph20_80_v1."""
    err = np.abs(pdet - t)
    cut = 20 * NPG
    cal, ev = err[:cut], err[cut:]
    rows = []
    for a in [0.5, 0.7, 0.8, 0.9, 0.95]:
        k = int(np.ceil((cal.size + 1) * a))
        q = float(np.sort(cal)[k - 1])
        rows.append({"nominal": a,
                     "empirical_coverage": round(float((ev <= q).mean()), 5),
                     "half_width_vehh": round(q, 4),
                     "full_width_vehh": round(2 * q, 4)})
    return {"protocol": "graph20_80_v1", "score": "absolute residual of the "
            "deterministic prediction", "levels": rows}


def build_timeline():
    """Trial progression, read from the per-trial result files."""
    stages = [
        ("T1", "pointnet_transf_gat_1st_bs32_5feat_seed42",
         "Linear output head, no dropout", 50,
         "Highest R2 of any trial, but excluded from all UQ work: sigma is zero "
         "everywhere without dropout, and it used a different head and split."),
        ("T2", "point_net_transf_gat_2nd_try", "GATConv output head, dropout on", 50,
         "Accuracy drops sharply against T1; becomes the baseline of the comparable family."),
        ("T3", "point_net_transf_gat_3rd_trial_weighted_loss", "Weighted loss", 50,
         "Worst result recorded."),
        ("T4", "point_net_transf_gat_4th_trial_weighted_loss", "Weighted loss again", 50,
         "Confirms T3; weighted loss abandoned."),
        ("T5", "point_net_transf_gat_5th_try", "Back to unweighted", 50, "Recovers."),
        ("T6", "point_net_transf_gat_6th_trial_lower_lr", "Lower learning rate", 50,
         "No gain; not pursued."),
        ("T7", "point_net_transf_gat_7th_trial_80_10_10_split",
         "80/10/10 split, 100-graph test", 100,
         "Test split doubles here. R2 either side of this point is not comparable."),
        ("T8", "point_net_transf_gat_8th_trial_lower_dropout", "Dropout 0.3 -> 0.2", 100,
         "Best comparable trial and the baseline for every uncertainty method."),
    ]
    rows = []
    for tid, name, change, graphs, lesson in stages:
        met = None
        for fn in ("test_evaluation_complete.json", "test_results.json"):
            p = REPO / "results" / "trials" / name / fn
            if p.exists():
                j = json.loads(p.read_text(encoding="utf-8"))
                tm = j.get("test_metrics", j)
                met = {"r2": tm.get("r2", tm.get("r2_score")), "mae": tm.get("mae"),
                       "rmse": tm.get("rmse")}
                break
        rows.append({"trial": tid, "directory": name, "change": change,
                     "test_graphs": graphs, "metrics": met, "lesson": lesson})
    rows += [
        {"trial": "T9", "directory": "point_net_transf_gat_9th_trial_heteroscedastic",
         "change": "Freeze T8, add a heteroscedastic head", "test_graphs": 100,
         "metrics": None, "best_val_nll": 3.2489, "training_minutes": 873,
         "lesson": "Checkpoint retained; verified test metrics not recorded."},
        {"trial": "T10", "directory": "point_net_transf_gat_10th_trial_cqr",
         "change": "Full CQR retrain from scratch", "test_graphs": 100,
         "metrics": {"r2_midpoint": 0.4057, "mae_midpoint": 4.1305},
         "picp_90": 89.473, "picp_95": 91.779, "training_minutes": 5219,
         "gate_status": "FAIL",
         "lesson": "87 hours of retraining lost most of the backbone's accuracy and "
                   "failed its own acceptance gates."},
        {"trial": "T11", "directory": "point_net_transf_gat_11th_trial_cqr_frozen",
         "change": "Freeze T8, train a quantile head only", "test_graphs": 100,
         "metrics": {"r2_midpoint": 0.5835, "mae_midpoint": 4.3015},
         "picp_90": 89.822, "picp_95": 94.908, "training_minutes": 2385,
         "gate_status": "PASS",
         "lesson": "Same method as T10, opposite outcome, decided by what was allowed "
                   "to move."},
        {"trial": "Ensemble", "directory": "deep_ensemble_results",
         "change": "5 seeds, dropout off at inference", "test_graphs": 100,
         "metrics": {"r2": 0.6841, "mae": 3.4853, "rmse": 6.2927},
         "spearman_rho": 0.3997,
         "lesson": "Most accurate model here, yet its sigma ranks errors worse than MC "
                   "Dropout (0.400 vs 0.482). Better predictions did not mean better "
                   "uncertainty."},
    ]
    return {
        "comparability_warning":
            "Trials T1-T6 were scored on 50 test graphs (1,581,750 nodes) and T7 onward "
            "on 100 (3,163,500). R2 is not comparable across that boundary. See "
            "docs/CORRIGENDUM.md C9.",
        "uq_baseline": "T8",
        "stages": rows,
    }


def build_uq_methods():
    return {
        "note": "Only methods actually implemented. Every value is asserted by "
                "scripts/verify_headline_results.py or read from a tracked artifact.",
        "methods": [
            {"method": "MC Dropout", "question": "Does sigma rank errors?",
             "post_hoc": True, "metric": "Spearman rho",
             "result": {"t8_rho": 0.482, "t7_rho_replayable": 0.4437, "S": 30},
             "limitation": "A ranking signal, not a calibrated scale."},
            {"method": "Temperature scaling", "question": "Is sigma the right size?",
             "post_hoc": True, "metric": "expected calibration error",
             "result": {"temperature_thesis": 2.702,
                        "temperature_recomputed": 2.701,
                        "temperature_note": "The thesis reports the archived optimum "
                                            "2.7025; re-fitting here finds 2.7010. See "
                                            "docs/UQ_SUMMARY.md.",
                        "ece_before": 0.269, "ece_after": 0.048},
             "limitation": "Fixes average width, not per-node width."},
            {"method": "Split conformal", "question": "Can coverage be guaranteed?",
             "post_hoc": True, "metric": "empirical marginal coverage",
             "result": {"coverage_90": 90.17, "coverage_95": 95.09},
             "limitation": "Marginal only; one shared width for every link."},
            {"method": "Adaptive conformal",
             "question": "Can coverage hold across uncertainty levels?",
             "post_hoc": True, "metric": "conditional coverage by sigma decile",
             "result": {"range_adaptive": [83.7, 96.4], "range_standard": [59.0, 98.1]},
             "limitation": "Wider intervals overall."},
            {"method": "Selective prediction", "question": "Does it change a decision?",
             "post_hoc": True, "metric": "MAE reduction at a retention level",
             "result": {"mae_reduction_pct_at_50": 41.2, "at_90": 18.3},
             "limitation": "Needs the ranking only; says nothing about scale."},
            {"method": "Error detection", "question": "Can bad links be flagged?",
             "post_hoc": True, "metric": "AUROC",
             "result": {"auroc_top10": 0.7585, "auroc_top20": 0.7401,
                        "source": "trial8_uq_ablation_results.csv"},
             "limitation": "Ranking quality, not precision at an operating point."},
            {"method": "CQR", "question": "Can the interval itself be learned?",
             "post_hoc": False, "metric": "PICP and interval width",
             "result": {"t10_gate": "FAIL", "t11_gate": "PASS",
                        "t11_picp_90": 89.822, "t11_picp_95": 94.908},
             "limitation": "A full retrain (T10) can lose the backbone's accuracy."},
            {"method": "Deep ensemble", "question": "Does more capacity help?",
             "post_hoc": False, "metric": "R2 and Spearman rho",
             "result": {"r2": 0.6841, "rho": 0.3997, "rho_vs_mc_dropout_pct": -17.07},
             "limitation": "Best accuracy, worse uncertainty ranking than MC Dropout."},
            {"method": "Heteroscedastic regression",
             "question": "Can the model predict its own variance?",
             "post_hoc": False, "metric": "validation NLL",
             "result": {"best_val_nll": 3.2489, "epoch": 290},
             "limitation": "No test metrics were recorded."},
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("building UQ and experiment assets")
    jdump(build_timeline(), "experiment_timeline.json")
    jdump(build_uq_methods(), "uq_methods.json")
    try:
        t, pmc, sig, pdet = load_t8()
    except (ArtifactNotFound, FileNotFoundError) as exc:
        print(f"  skipped calibration/selective/conformal: {exc}")
        return 0
    jdump(build_calibration(t, pmc, sig), "calibration_curve.json")
    jdump(build_selective(t, pmc, sig), "selective_prediction.json")
    jdump(build_conformal(t, pdet), "conformal_coverage.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
