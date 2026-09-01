from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "thesis-intelligence-v1"
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evidence_contract import (
    AUDIT_SOURCE_COMMIT,
    LOCAL_TEST_LOADER_BYTES,
    LOCAL_TEST_LOADER_PATH,
    LOCAL_TEST_LOADER_SHA256,
    SOURCE_ARTIFACTS,
    SUBMITTED_ARTIFACT_COMMIT,
    SUBMITTED_PDF_SHA256,
)
from thesis_dashboard.analytics import (
    array_statistics,
    binned_relationship,
    error_detection_metrics,
    gaussian_reliability,
    load_numeric_npz,
    load_prediction_arrays,
    regression_metrics,
    risk_curve,
    uncertainty_metrics,
)

OUTPUT_ROOT = ROOT / "analysis_outputs"
FIGURE_DIR = OUTPUT_ROOT / "figures"
RETENTIONS = tuple(range(10, 101, 5))
NOMINAL_LEVELS = tuple(np.linspace(0.1, 0.95, 10))
FEATURE_NAMES = (
    "VOL_BASE_CASE",
    "CAPACITY_BASE_CASE",
    "CAPACITY_REDUCTION",
    "FREESPEED",
    "LENGTH",
)

ARTIFACTS = {
    name: Path(relative_path)
    for name, (relative_path, _bytes, _digest) in SOURCE_ARTIFACTS.items()
}

LOCAL_TEST_LOADER = Path(LOCAL_TEST_LOADER_PATH)

DISCREPANCIES = [
    {
        "severity": "high",
        "topic": "target zero-mass claim",
        "finding": (
            "The thesis attributed the CAPACITY_REDUCTION feature's 88.7% zero share to "
            "the test target. Full cached targets contain 872,540 exact zeros out of "
            "3,163,500 rows (27.58%)."
        ),
        "impact": (
            "The stated explanation for selective-prediction and AUROC performance "
            "materially overstated target sparsity."
        ),
        "evidence": (
            "full results/predictions T7/T8/ensemble targets and trusted local T8 test loader"
        ),
        "resolution": (
            "Published as a post-submission corrigendum; submitted thesis files remain "
            "unchanged."
        ),
    },
    {
        "severity": "high",
        "topic": "evaluation normalization",
        "finding": (
            "Base training fits separate feature and position scalers on train, validation, "
            "and test partitions instead of applying training-fitted scalers throughout."
        ),
        "impact": (
            "Evaluation-distribution statistics influence preprocessing; this is not target "
            "leakage, but it weakens deployment and method comparability claims."
        ),
        "evidence": "scripts/training/help_functions.py:180-196,234-265",
        "resolution": "Reported as a limitation; historical artifacts were not rewritten.",
    },
    {
        "severity": "medium",
        "topic": "MC stochastic replay variation",
        "finding": (
            "The trial-specific T8 MC archive and later verification archive are distinct "
            "stochastic runs. They produce slightly different MAE, rho, and k-factor values."
        ),
        "impact": "Rounded T8 UQ values vary slightly with the cached replay used.",
        "evidence": (
            "results/predictions/point_net_transf_gat_8th_trial_lower_dropout/uq_results/"
            "mc_dropout_full_100graphs_mc30.npz and "
            "results/predictions/uq_verification_run/mc_dropout_verified.npz"
        ),
        "resolution": (
            "Dashboard values identify and recompute the trial-specific source; thesis-final "
            "reported values remain linked to the verification replay."
        ),
    },
    {
        "severity": "high",
        "topic": "calibration protocol",
        "finding": (
            "Tracked 20/80 graph-level calibration and final-thesis 30/70 random node-level "
            "calibration produce different temperature and ECE values."
        ),
        "impact": "Calibration claims are protocol-dependent and cannot be pooled.",
        "evidence": (
            "results/temperature_scaling_t8.json; "
            "thesis/latex_tum_official/chapters/04_experiments.tex"
        ),
        "resolution": "Both protocols are named explicitly; the dashboard does not mix them.",
    },
    {
        "severity": "high",
        "topic": "checkpoint loading",
        "finding": "Several evaluation scripts use strict=False without key-coverage checks.",
        "impact": "Incompatible PyG state dictionaries may silently leave random parameters.",
        "evidence": (
            "scripts/evaluation/run_mc_dropout_full.py:77-81; "
            "scripts/evaluation/ensemble_uq_experiments.py:133-137"
        ),
        "resolution": (
            "No checkpoint replay is used in this analysis; cached numeric predictions are "
            "the canonical reproducible input."
        ),
    },
    {
        "severity": "medium",
        "topic": "early stopping",
        "finding": (
            "The thesis says validation R2, while executable base training saves/stops on "
            "validation loss and returns final rather than best values."
        ),
        "impact": "The training narrative does not exactly describe the implementation.",
        "evidence": (
            "scripts/gnn/models/base_gnn.py:202-273; "
            "thesis/latex_tum_official/chapters/04_experiments.tex"
        ),
        "resolution": "Documented; no retraining was performed.",
    },
    {
        "severity": "medium",
        "topic": "scope and sample count",
        "finding": "README says 10,000 simulations; final experiments used 1,000.",
        "impact": "Overstates the experimental training sample count.",
        "evidence": "README.md; thesis/latex_tum_official/chapters/04_experiments.tex",
        "resolution": "README corrected to 1,000.",
    },
    {
        "severity": "medium",
        "topic": "feature semantics",
        "finding": (
            "Documentation called FREESPEED maximum speed and described position order as "
            "start/middle/end; code uses free-flow speed and start/end/midpoint."
        ),
        "impact": "Feature interpretation and geometry provenance were ambiguous.",
        "evidence": "scripts/data_preprocessing/help_functions.py:127-206",
        "resolution": "Canonical feature dictionary records executable meanings.",
    },
]


def read_json(relative_path: Path) -> dict[str, Any]:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {relative_path.as_posix()}")
    return value


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest() -> list[dict[str, Any]]:
    rows = []
    for name, relative_path in ARTIFACTS.items():
        path = ROOT / relative_path
        rows.append(
            {
                "name": name,
                "path": relative_path.as_posix(),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256(path) if path.is_file() else None,
                "trust_boundary": "tracked audited-source artifact",
            }
        )
    loader = ROOT / LOCAL_TEST_LOADER
    rows.append(
        {
            "name": "t8_local_test_loader",
            "path": LOCAL_TEST_LOADER.as_posix(),
            "exists": loader.is_file(),
            "bytes": loader.stat().st_size if loader.is_file() else None,
            "sha256": sha256(loader) if loader.is_file() else None,
            "trust_boundary": "hash-locked local audited-source pickle artifact; never export",
        }
    )
    return rows


def histogram(values: np.ndarray, bins: int = 60) -> dict[str, list[float] | list[int]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    lower, upper = np.quantile(finite, (0.005, 0.995))
    counts, edges = np.histogram(finite, bins=bins, range=(lower, upper))
    return {"counts": counts.astype(int).tolist(), "edges": edges.tolist()}


def analyze_mc_artifact(name: str, relative_path: Path) -> dict[str, Any]:
    predictions, uncertainty, targets = load_prediction_arrays(ROOT / relative_path)
    errors = np.abs(predictions - targets)
    result = {
        "method": "MC Dropout (30 stochastic passes)",
        "source": relative_path.as_posix(),
        "scope": "full cached 100-graph test set",
        "row_count": int(predictions.size),
        "point_metrics": regression_metrics(predictions, targets),
        "uncertainty_metrics": uncertainty_metrics(predictions, uncertainty, targets),
        "selective_risk": risk_curve(predictions, uncertainty, targets, RETENTIONS),
        "gaussian_reliability": gaussian_reliability(
            predictions, uncertainty, targets, NOMINAL_LEVELS
        ),
        "uncertainty_error_bins": binned_relationship(uncertainty, errors, bins=20),
        "quality": {
            "predictions": array_statistics(predictions),
            "uncertainties": array_statistics(uncertainty, plausible_min=0.0),
            "targets": array_statistics(targets),
            "absolute_error": array_statistics(errors, plausible_min=0.0),
        },
        "target_histogram_middle_99pct": histogram(targets),
    }
    if name == "t8_mc":
        deterministic = load_numeric_npz(
            ROOT / ARTIFACTS["t8_deterministic"], ("predictions", "targets")
        )
        if not np.array_equal(deterministic["targets"], targets):
            raise ValueError("T8 deterministic and MC targets are not aligned")
        deterministic_errors = np.abs(deterministic["predictions"] - targets)
        result["error_detection"] = error_detection_metrics(
            uncertainty, deterministic_errors, (90, 95, 99)
        )
        result["error_detection_protocol"] = (
            "Labels use deterministic absolute-error percentiles; scores use MC sigma."
        )
        del deterministic, deterministic_errors
    del predictions, uncertainty, targets, errors
    gc.collect()
    return result


def analyze_ensemble() -> dict[str, Any]:
    relative_path = ARTIFACTS["deep_ensemble"]
    arrays = load_numeric_npz(
        ROOT / relative_path, ("ensemble_mean", "ensemble_std", "targets")
    )
    predictions = arrays["ensemble_mean"]
    uncertainty = arrays["ensemble_std"]
    targets = arrays["targets"]
    errors = np.abs(predictions - targets)
    result = {
        "method": "Deep Ensemble (five deterministic members)",
        "source": relative_path.as_posix(),
        "scope": "full cached 100-graph test set",
        "row_count": int(predictions.size),
        "point_metrics": regression_metrics(predictions, targets),
        "uncertainty_metrics": uncertainty_metrics(predictions, uncertainty, targets),
        "selective_risk": risk_curve(predictions, uncertainty, targets, RETENTIONS),
        "gaussian_reliability": gaussian_reliability(
            predictions, uncertainty, targets, NOMINAL_LEVELS
        ),
        "uncertainty_error_bins": binned_relationship(uncertainty, errors, bins=20),
        "quality": {
            "predictions": array_statistics(predictions),
            "uncertainties": array_statistics(uncertainty, plausible_min=0.0),
            "targets": array_statistics(targets),
            "absolute_error": array_statistics(errors, plausible_min=0.0),
        },
    }
    del arrays, predictions, uncertainty, targets, errors
    gc.collect()
    return result


def analyze_cqr(name: str, relative_path: Path) -> dict[str, Any]:
    arrays = load_numeric_npz(ROOT / relative_path, ("q_lo", "q_hi", "targets"))
    lower, upper, targets = arrays["q_lo"], arrays["q_hi"], arrays["targets"]
    crossing_count = int(np.count_nonzero(lower > upper))
    midpoint = (lower + upper) / 2
    raw_coverage = float(np.mean((targets >= lower) & (targets <= upper)))
    result = {
        "method": name.upper(),
        "source": relative_path.as_posix(),
        "scope": "cached validation predictions; not an independent test archive",
        "row_count": int(targets.size),
        "midpoint_metrics": regression_metrics(midpoint, targets),
        "raw_interval_coverage": raw_coverage,
        "mean_raw_width": float(np.mean(upper - lower)),
        "crossing_count": crossing_count,
        "quality": {
            "lower": array_statistics(lower),
            "upper": array_statistics(upper),
            "targets": array_statistics(targets),
        },
    }
    del arrays, lower, upper, targets, midpoint
    gc.collect()
    return result


def summarize_local_test_loader() -> dict[str, Any]:
    path = ROOT / LOCAL_TEST_LOADER
    if not path.is_file():
        return {
            "available": False,
            "source": LOCAL_TEST_LOADER.as_posix(),
            "limitation": "Local ignored loader is unavailable; no graph-tensor EDA generated.",
        }
    if path.stat().st_size != LOCAL_TEST_LOADER_BYTES or sha256(path) != LOCAL_TEST_LOADER_SHA256:
        raise ValueError("Trusted local test loader does not match its locked audit contract")

    import torch  # pyright: ignore[reportMissingImports]

    # The local PyG graph list is pickle-capable and must come from the trusted audit checkout.
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("Expected the trusted test loader to contain a non-empty graph list")
    feature_chunks: list[list[np.ndarray]] = [[] for _ in FEATURE_NAMES]
    target_chunks: list[np.ndarray] = []
    node_counts = []
    edge_counts = []
    position_shapes: set[tuple[int, ...]] = set()
    for graph in graphs:
        x = graph.x.detach().cpu().numpy()
        y = graph.y.detach().cpu().numpy().reshape(-1)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or x.shape[0] != y.size:
            raise ValueError("Unexpected graph feature/target schema")
        for index in range(len(FEATURE_NAMES)):
            feature_chunks[index].append(x[:, index])
        target_chunks.append(y)
        node_counts.append(int(x.shape[0]))
        edge_counts.append(int(graph.edge_index.shape[1]))
        position_shapes.add(tuple(int(value) for value in graph.pos.shape[1:]))

    feature_quality = {}
    for name, chunks in zip(FEATURE_NAMES, feature_chunks, strict=True):
        feature_quality[name] = array_statistics(np.concatenate(chunks))
    targets = np.concatenate(target_chunks)
    result = {
        "available": True,
        "source": LOCAL_TEST_LOADER.as_posix(),
        "trust_boundary": "trusted local-only pickle-capable artifact; aggregate output only",
        "representation": "scaler-normalized model-ready test tensors",
        "scope": "100 held-out policy graphs, not the raw 1,000-scenario corpus",
        "graph_count": len(graphs),
        "node_count": int(sum(node_counts)),
        "edge_count": int(sum(edge_counts)),
        "nodes_per_graph": array_statistics(np.asarray(node_counts)),
        "edges_per_graph": array_statistics(np.asarray(edge_counts)),
        "position_tail_shapes": [list(shape) for shape in sorted(position_shapes)],
        "feature_order": list(FEATURE_NAMES),
        "features": feature_quality,
        "target": array_statistics(targets),
        "target_zero_fraction": float(np.mean(targets == 0)),
        "limitation": (
            "Physical-unit feature plausibility cannot be reconstructed safely from these "
            "normalized tensors because historical split-specific scalers are ambiguous."
        ),
    }
    del graphs, feature_chunks, target_chunks, targets
    gc.collect()
    return result


def reported_model_comparison() -> list[dict[str, Any]]:
    t7 = read_json(ARTIFACTS["t7_point_metrics"])["test_metrics"]
    t8 = read_json(ARTIFACTS["t8_point_metrics"])["test_metrics"]
    ensemble = read_json(ARTIFACTS["deep_ensemble_metrics"])["point_prediction"]
    t10_document = read_json(ARTIFACTS["t10_metrics"])
    t11_document = read_json(ARTIFACTS["t11_metrics"])
    t10 = t10_document["test_metrics"]
    t11 = t11_document["test_metrics"]
    return [
        {
            "model": "T7 deterministic",
            "r2": t7["r2"],
            "mae": t7["mae"],
            "rmse": t7["rmse"],
            "protocol": "reported held-out test metrics",
        },
        {
            "model": "T8 deterministic",
            "r2": t8["r2"],
            "mae": t8["mae"],
            "rmse": t8["rmse"],
            "protocol": "reported held-out test metrics",
        },
        {
            "model": "Deep Ensemble",
            "r2": ensemble["r2"],
            "mae": ensemble["mae"],
            "rmse": ensemble["rmse"],
            "protocol": "recomputed and reported held-out test metrics",
        },
        {
            "model": "T10 CQR midpoint",
            "r2": t10["r2_midpoint"],
            "mae": t10["mae_midpoint"],
            "rmse": t10["rmse_midpoint"],
            "coverage_90": t10["PICP_90_pct"] / 100,
            "coverage_95": t10["PICP_95_pct"] / 100,
            "gate": t10_document["gate_status"],
            "protocol": "reported CQR test metrics; no cached test arrays",
        },
        {
            "model": "T11 frozen CQR midpoint",
            "r2": t11["r2_midpoint"],
            "mae": t11["mae_midpoint"],
            "rmse": t11["rmse_midpoint"],
            "coverage_90": t11["PICP_90_pct"] / 100,
            "coverage_95": t11["PICP_95_pct"] / 100,
            "gate": t11_document["gate_status"],
            "protocol": "reported CQR test metrics; no cached test arrays",
        },
    ]


def calibration_protocols() -> dict[str, Any]:
    tracked = read_json(ARTIFACTS["temperature"])
    conditional = read_json(ARTIFACTS["conditional_conformal"])
    return {
        "graph20_80_v1": {
            "status": "tracked and dashboard-reproducible",
            "split": tracked["split"],
            "temperature": tracked["optimal_temperature_T"],
            "evaluation_ece_before": tracked["evaluation_set"]["ece_before"],
            "evaluation_ece_after": tracked["evaluation_set"]["ece_after"],
            "evaluation_ece_improvement_pct": tracked["evaluation_set"][
                "ece_improvement_pct"
            ],
            "nominal_levels": tracked["nominal_levels"],
            "coverage_before": tracked["evaluation_set"]["coverage_before"],
            "coverage_after": tracked["evaluation_set"]["coverage_after"],
            "conditional_conformal": {
                "calibration_nodes": conditional["n_cal_nodes"],
                "evaluation_nodes": conditional["n_eval_nodes"],
                "global_quantiles": conditional["global_quantiles"],
                "adaptive_quantiles": conditional["adaptive_quantiles"],
                "sigma_deciles": conditional["sigma_deciles"],
            },
        },
        "node30_70_thesis_final": {
            "status": "reported in final thesis; no canonical cached split indices",
            "split": "random 30% node calibration / 70% node evaluation",
            "temperature_approx": 2.887,
            "evaluation_ece_before_approx": 0.356,
            "evaluation_ece_after_approx": 0.034,
            "warning": (
                "Not directly comparable with graph20_80_v1. Values are displayed as "
                "reported, not recomputed."
            ),
        },
    }


def feature_dictionary() -> list[dict[str, Any]]:
    meanings = (
        ("Base-case road-segment car volume", "veh/h"),
        ("Base-case car capacity; zero where cars are not permitted", "veh/h"),
        ("Policy car capacity minus base-case car capacity; reductions are negative", "veh/h"),
        ("Policy-scenario free-flow speed where cars are permitted", "source network units"),
        ("Base-case road-segment length", "source network units"),
    )
    return [
        {
            "model_index": index,
            "name": name,
            "meaning": meaning,
            "unit": unit,
            "model_representation": "standardized continuous feature",
        }
        for index, (name, (meaning, unit)) in enumerate(zip(FEATURE_NAMES, meanings, strict=True))
    ]


def save_figure(fig: Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "generate_thesis_intelligence.py"},
    )
    svg_path = FIGURE_DIR / f"{stem}.svg"
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Creator": "generate_thesis_intelligence.py", "Date": None},
    )
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8"
    )
    plt.close(fig)


def generate_figures(bundle: Mapping[str, Any]) -> None:
    colors = {"t8_mc": "#0f5b4d", "t7_mc": "#d97732", "deep_ensemble": "#245b8a"}
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for key in ("t8_mc", "t7_mc", "deep_ensemble"):
        rows = bundle["analyses"][key]["selective_risk"]
        ax.plot(
            [row["retention"] for row in rows],
            [row["mae"] for row in rows],
            marker="o",
            ms=3,
            color=colors[key],
            label=key.replace("_", " ").upper(),
        )
    ax.set(xlabel="Predictions retained (%)", ylabel="Accepted-set MAE (veh/h)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "selective_risk_comparison")

    models = bundle["reported_model_comparison"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    labels = [row["model"] for row in models]
    values = [row["r2"] for row in models]
    bars = ax.barh(labels, values, color=["#9ab7ad", "#0f5b4d", "#245b8a", "#d7a24d", "#9b6c8c"])
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(xlabel="Reported held-out R2", xlim=(0, max(values) + 0.1))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "model_r2_comparison")

    protocol = bundle["calibration_protocols"]["graph20_80_v1"]
    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    nominal = protocol["nominal_levels"]
    ax.plot(nominal, nominal, color="#252b2a", linestyle="--", label="Ideal")
    ax.plot(nominal, protocol["coverage_before"], marker="o", label="Raw sigma")
    ax.plot(nominal, protocol["coverage_after"], marker="o", label="Temperature-scaled")
    ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "temperature_reliability_graph20_80")

    deciles = protocol["conditional_conformal"]["sigma_deciles"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x_values = [row["decile"] for row in deciles]
    ax.plot(x_values, [row["global_coverage_90"] for row in deciles], marker="o", label="Global")
    ax.plot(x_values, [row["adaptive_coverage_90"] for row in deciles], marker="o", label="Adaptive")
    ax.axhline(0.9, color="#252b2a", linestyle="--", label="90% nominal")
    ax.set(xlabel="MC uncertainty decile", ylabel="Empirical 90% coverage", ylim=(0.55, 1.01))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "conditional_coverage_by_uncertainty")

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for key in ("t8_mc", "t7_mc", "deep_ensemble"):
        rows = bundle["analyses"][key]["uncertainty_error_bins"]
        ax.plot(
            [row["x_mean"] for row in rows],
            [row["y_mean"] for row in rows],
            marker="o",
            ms=3,
            color=colors[key],
            label=key.replace("_", " ").upper(),
        )
    ax.set(xlabel="Mean uncertainty within quantile bin", ylabel="Mean absolute error (veh/h)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "uncertainty_error_relationship")


def write_csv_outputs(bundle: Mapping[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "artifact_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=bundle["artifact_manifest"][0].keys())
        writer.writeheader()
        writer.writerows(bundle["artifact_manifest"])
    with (OUTPUT_ROOT / "model_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "model",
            "r2",
            "mae",
            "rmse",
            "coverage_90",
            "coverage_95",
            "gate",
            "protocol",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(bundle["reported_model_comparison"])


def report_markdown(bundle: Mapping[str, Any]) -> str:
    analyses = bundle["analyses"]
    t8 = analyses["t8_mc"]
    ensemble = analyses["deep_ensemble"]
    graph_data = bundle["graph_data_quality"]
    local_scope = (
        f"{graph_data['graph_count']} graphs / {graph_data['node_count']:,} nodes"
        if graph_data["available"]
        else "not available"
    )
    return f"""# Thesis Data, Model, and UQ Audit

Generated from trusted local artifacts on {bundle['generated_on']}. All paths are repository-relative and all exported values are aggregates.

Source: audited repository commit `{bundle['source_provenance']['audit_source_commit']}`. This
is a post-submission audit; the immutable submitted PDF remains unchanged.

## Executive findings

- T8 MC Dropout was recomputed over {t8['row_count']:,} cached node predictions: MAE {t8['point_metrics']['mae']:.4f}, R2 {t8['point_metrics']['r2']:.4f}, uncertainty-error Spearman rho {t8['uncertainty_metrics']['spearman_rho']:.4f}.
- The five-member Deep Ensemble improves point prediction to MAE {ensemble['point_metrics']['mae']:.4f} and R2 {ensemble['point_metrics']['r2']:.4f}, but ranks error less strongly (rho {ensemble['uncertainty_metrics']['spearman_rho']:.4f}).
- Raw Gaussian intervals are under-dispersed: T8 95% nominal coverage is {t8['uncertainty_metrics']['raw_gaussian_coverage_95']:.1%}; calibration is required before uncertainty is interpreted as coverage.
- T8 selective prediction cuts accepted-set MAE from {t8['selective_risk'][-1]['mae']:.3f} to {next(row['mae'] for row in t8['selective_risk'] if row['retention'] == 50):.3f} veh/h at 50% retention. This is a review-capacity trade-off, not proof that rejected rows are incorrect.
- Local graph-tensor audit scope: {local_scope}. Raw MATSim-to-graph regeneration remains unavailable in this checkout.
- The strongest methodology risk is split-specific scaling of evaluation data. Historical scores are retained but should be presented with that limitation.

## Data and feature provenance

The model input order is `VOL_BASE_CASE`, `CAPACITY_BASE_CASE`, `CAPACITY_REDUCTION`, `FREESPEED`, `LENGTH`. The target is policy-scenario `vol_car` minus base-case `vol_car`. Position tensors are start point, end point, midpoint; the model consumes start and end. `FREESPEED` means free-flow speed, not maximum speed.

The local loader contains normalized model-ready tensors, so its statistics diagnose schema, missingness, constancy, tails, and distribution shape. They do not recover physical-unit feature plausibility. The confidential ignored `data` junction and all pickle-capable inputs remain outside generated outputs.

## Model and uncertainty interpretation

MC sigma has useful ranking information, especially for routing scarce review capacity, but rho is association rather than causation or calibration. The target has {t8['quality']['targets']['zero_count']:,} exact zeros ({t8['quality']['targets']['zero_count'] / t8['quality']['targets']['count']:.1%}), not the previously reported 88.7%; class imbalance and degradation in the highest-change regime still make policy-critical tail behavior more demanding than pooled summaries.

Deep Ensemble point accuracy is strongest among cached full-test prediction artifacts, while T8 MC Dropout has stronger uncertainty-error ranking. T11 CQR passes its reported joint gate and T10 fails; no cached T10/T11 test arrays exist, so those test scores are reported rather than independently replayed.

## Calibration protocols

`graph20_80_v1` is directly backed by tracked artifacts: first 20 graphs calibrate and the last 80 evaluate. `node30_70_thesis_final` is the final-thesis random node protocol and is reported only. Their temperatures and ECE values must not be compared as if they were repeated estimates of one split.

Global conformal intervals have poor conditional coverage in the highest uncertainty decile. Adaptive intervals improve high-sigma coverage but change interval width across the uncertainty distribution. Coverage claims are marginal unless a conditional stratum is explicitly named.

## Corrective actions

- Use the generated aggregate bundle as the dashboard source of record.
- Label full-data metrics, deterministic 12,000-row plots, reported-only metrics, and local-only graph diagnostics separately.
- Fit preprocessing scalers on training data only in future experiments and persist one versioned scaler/feature schema.
- Replace permissive checkpoint loading with scoped key remapping and explicit key validation before any future replay.
- Preserve old calibration outputs under protocol names rather than overwriting them.

## Limitations

- Raw MATSim scenarios are unavailable, so preprocessing and physical-unit EDA cannot be reproduced end to end.
- Prediction archives support node-level pooled analyses; scenario-level dependence limits naive iid interpretation.
- No T9 prediction cache and no T10/T11 test prediction cache are tracked.
- Spatial/link-level export is intentionally omitted to avoid disclosing row-level confidential research data.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate safe aggregate thesis intelligence outputs."
    )
    parser.add_argument(
        "--include-local-graphs",
        action="store_true",
        help="Load the trusted ignored T8 test loader and export aggregate graph statistics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name, (relative_path, expected_bytes, expected_digest) in SOURCE_ARTIFACTS.items():
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required audited source artifact is missing: {relative_path}")
        if path.stat().st_size != expected_bytes or sha256(path) != expected_digest:
            raise ValueError(
                f"Audited source artifact does not match its locked contract: {name}"
            )

    bundle: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_on": datetime.now(timezone.utc).date().isoformat(),
        "source_provenance": {
            "audit_source_repository": (
                "https://github.com/mzquadri/"
                "ml_surrogates_for_agent_based_transport_models"
            ),
            "audit_source_commit": AUDIT_SOURCE_COMMIT,
            "submitted_artifact_commit": SUBMITTED_ARTIFACT_COMMIT,
            "submitted_pdf_sha256": SUBMITTED_PDF_SHA256,
        },
        "privacy": {
            "classification": "safe aggregate export",
            "contains_row_level_records": False,
            "contains_absolute_paths": False,
            "contains_pickle_payloads": False,
            "source_data_policy": "local processing only; confidential data junction excluded",
        },
        "feature_dictionary": feature_dictionary(),
        "artifact_manifest": artifact_manifest(),
        "analyses": {
            "t8_mc": analyze_mc_artifact("t8_mc", ARTIFACTS["t8_mc"]),
            "t7_mc": analyze_mc_artifact("t7_mc", ARTIFACTS["t7_mc"]),
            "deep_ensemble": analyze_ensemble(),
            "t10_cqr_validation": analyze_cqr("t10_cqr", ARTIFACTS["t10_cqr"]),
            "t11_cqr_validation": analyze_cqr("t11_cqr", ARTIFACTS["t11_cqr"]),
        },
        "reported_model_comparison": reported_model_comparison(),
        "calibration_protocols": calibration_protocols(),
        "discrepancies": DISCREPANCIES,
        "graph_data_quality": (
            summarize_local_test_loader()
            if args.include_local_graphs
            else {
                "available": False,
                "source": LOCAL_TEST_LOADER.as_posix(),
                "limitation": "Skipped; rerun with --include-local-graphs for local-only aggregate EDA.",
            }
        ),
        "scope_notes": {
            "full_data": "All scalar model/UQ metrics and aggregate bins use full cached arrays.",
            "sampled": "Only dashboard scatter rendering uses a deterministic sample (seed 42).",
            "reported_only": "T10/T11 test metrics and final-thesis calibration lack replayable test/split artifacts.",
        },
    }
    write_json(OUTPUT_ROOT / "thesis_intelligence.json", bundle)
    write_csv_outputs(bundle)
    generate_figures(bundle)
    report = report_markdown(bundle)
    (OUTPUT_ROOT / "THESIS_INTELLIGENCE_REPORT.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    print(f"Generated aggregate analysis at {OUTPUT_ROOT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
