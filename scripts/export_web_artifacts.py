"""Derive small, provenance-stamped web artifacts from the thesis evidence.

The portfolio site must never load a checkpoint, a `.pt` dataloader, a raw trial directory or a
large `.npz`. This script is the only bridge: it reads the released training corpus and the
audited result bundle, and writes compact JSON plus rendered figures that a web page can consume
directly.

Two evidence classes are kept apart throughout, and every artifact records which one it is:

``thesis-result``
    Produced by, or audited from, the submitted work. Numbers are copied from
    ``analysis_outputs/thesis_intelligence.json``; nothing is recomputed here.

``post-thesis-data-audit``
    Measured by this script from the released ``train-data-v1`` corpus after submission. These
    describe the published dataset, not the thesis experiments.

Usage::

    python scripts/export_web_artifacts.py
    python scripts/export_web_artifacts.py --out web_exports --skip-corpus
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "code" / "data" / "train_data" / "dist_not_connected_10k_1pct"
AUDIT = ROOT / "analysis_outputs" / "thesis_intelligence.json"
DEFAULT_OUT = ROOT / "web_exports"

SCHEMA_VERSION = 2
DATASET_RELEASE = "train-data-v1"
GENERATED_BY = "scripts/export_web_artifacts.py"

THESIS = "thesis-result"
AUDIT_CLASS = "post-thesis-data-audit"

# Column order written by code/scripts/data_preprocessing/process_simulations_for_gnn.py.
STORED_FEATURES = (
    "VOL_BASE_CASE",
    "CAPACITY_BASE_CASE",
    "CAPACITY_REDUCTION",
    "FREESPEED",
    "HIGHWAY",
    "LENGTH",
)

# Meanings traced to help_functions.get_basic_edge_attributes and the audited feature dictionary.
# `model_index` is None where the selected model does not consume the column.
FEATURE_SEMANTICS: dict[str, dict[str, Any]] = {
    "VOL_BASE_CASE": {
        "meaning": "Base-case road-segment car volume",
        "unit": "veh/h",
        "model_index": 0,
        "role": "static network context",
    },
    "CAPACITY_BASE_CASE": {
        "meaning": "Base-case car capacity; zero where cars are not permitted",
        "unit": "veh/h",
        "model_index": 1,
        "role": "static network context",
    },
    "CAPACITY_REDUCTION": {
        "meaning": "Policy car capacity minus base-case car capacity; reductions are negative",
        "unit": "veh/h",
        "model_index": 2,
        "role": "scenario intervention",
    },
    "FREESPEED": {
        "meaning": "Policy-scenario free-flow speed, zero where cars are not permitted",
        "unit": "m/s",
        "model_index": 3,
        "role": "static network context",
    },
    "HIGHWAY": {
        "meaning": "Label-encoded road class; -1 marks a class absent from the mapping",
        "unit": "category code",
        "model_index": None,
        "role": "stored but not consumed by the selected model",
    },
    "LENGTH": {
        "meaning": "Base-case road-segment length",
        "unit": "m",
        "model_index": 4,
        "role": "static network context",
    },
}

TARGET_DEFINITION = {
    "symbol": "Delta v",
    "name": "policy-induced change in road-segment car volume",
    "expression": "vol_car(policy scenario) - vol_car(base case)",
    "unit": "veh/h",
    "change_type": "absolute",
    "sign_convention": "positive means more car volume under the policy than in the base case",
    "source": "code/scripts/data_preprocessing/help_functions.py:compute_target_tensor_only_edge_features",
}

GRAPH_SEMANTICS = {
    "representation": "dual line graph",
    "graph_node": "one road segment (link) of the Paris network",
    "graph_edge": "an adjacency between two road segments that meet at a junction",
    "x_row": "feature vector of one road segment",
    "y_row": "Delta v for one road segment",
    "pos_0": "segment start point",
    "pos_1": "segment end point",
    "pos_2": "segment midpoint, stored but unused by the forward pass",
    "coordinate_reference_system": "EPSG:4326 (WGS84), set explicitly in process_simulations_for_gnn.py",
    "source": "code/scripts/data_preprocessing/process_simulations_for_gnn.py",
}


def run_git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def provenance(classification: str, source_artifacts: list[str], **extra: Any) -> dict[str, Any]:
    """Stamp every artifact with where it came from. No local paths, no usernames."""
    block = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "source_repository": "https://github.com/mzquadri/ml-surrogates-thesis",
        "source_commit": run_git("rev-parse", "HEAD"),
        "dataset_release": DATASET_RELEASE,
        "generated_by": GENERATED_BY,
        "source_artifacts": source_artifacts,
    }
    block.update(extra)
    return block


def load_audit() -> dict[str, Any]:
    if not AUDIT.exists():
        raise SystemExit(f"Missing audited bundle: {AUDIT.relative_to(ROOT)}")
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def batch_paths() -> list[Path]:
    paths = sorted(
        CORPUS.glob("datalist_batch_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1])
    )
    if not paths:
        raise SystemExit(
            f"No datalist_batch_*.pt under {CORPUS.relative_to(ROOT)}.\n"
            "  gh release download train-data-v1 --repo mzquadri/ml-surrogates-thesis \\\n"
            "    --pattern 'datalist_batch_*.pt' --dir <dir>"
        )
    return paths


# --------------------------------------------------------------------------------------------
# Corpus pass
# --------------------------------------------------------------------------------------------


def scan_corpus() -> dict[str, Any]:
    """One streaming pass over all 1,000 scenarios.

    Accumulates feature statistics, per-scenario intervention statistics, and the response split
    between intervened and non-intervened segments. Holds one batch at a time.
    """
    import warnings

    import torch

    paths = batch_paths()
    n_feat = len(STORED_FEATURES)

    nodes = 0
    node_counts: set[int] = set()
    edge_counts: set[int] = set()
    isolated_counts: set[int] = set()

    f_sum = np.zeros(n_feat)
    f_sq = np.zeros(n_feat)
    f_min = np.full(n_feat, np.inf)
    f_max = np.full(n_feat, -np.inf)
    f_zero = np.zeros(n_feat, dtype=np.int64)

    y_sum = y_sq = 0.0
    y_min, y_max = np.inf, -np.inf
    y_zero = 0

    ref_x: np.ndarray | None = None
    ref_edges: np.ndarray | None = None
    constant = np.ones(n_feat, dtype=bool)
    max_delta = np.zeros(n_feat)
    topology_constant = True

    scenarios: list[dict[str, Any]] = []
    geometry: np.ndarray | None = None

    for path in paths:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = torch.load(path, map_location="cpu", weights_only=False)

        for local_index, data in enumerate(batch):
            x = data.x.numpy().astype(np.float64)
            y = data.y.numpy().astype(np.float64).ravel()
            edges = data.edge_index.numpy()
            index = len(scenarios)

            node_counts.add(x.shape[0])
            edge_counts.add(edges.shape[1])
            nodes += x.shape[0]

            f_sum += x.sum(axis=0)
            f_sq += (x**2).sum(axis=0)
            f_min = np.minimum(f_min, x.min(axis=0))
            f_max = np.maximum(f_max, x.max(axis=0))
            f_zero += (x == 0).sum(axis=0)

            y_sum += float(y.sum())
            y_sq += float((y**2).sum())
            y_min = min(y_min, float(y.min()))
            y_max = max(y_max, float(y.max()))
            y_zero += int((y == 0).sum())

            degree = np.bincount(edges.reshape(-1), minlength=x.shape[0])
            isolated_counts.add(int((degree == 0).sum()))

            if ref_x is None:
                ref_x, ref_edges = x.copy(), edges.copy()
                geometry = data.pos.numpy().astype(np.float64).copy()
            else:
                assert ref_edges is not None
                delta = np.abs(x - ref_x).max(axis=0)
                max_delta = np.maximum(max_delta, delta)
                constant &= delta == 0
                if not np.array_equal(edges, ref_edges):
                    topology_constant = False

            intervened = x[:, STORED_FEATURES.index("CAPACITY_REDUCTION")] != 0
            n_intervened = int(intervened.sum())
            abs_y = np.abs(y)
            scenarios.append(
                {
                    "index": index,
                    "batch_file": path.name,
                    "position_in_batch": local_index,
                    "intervened_links": n_intervened,
                    "intervened_fraction_pct": 100 * n_intervened / x.shape[0],
                    "total_reduction_veh_h": float(
                        x[intervened, STORED_FEATURES.index("CAPACITY_REDUCTION")].sum()
                    ),
                    "median_reduction_veh_h": float(
                        np.median(x[intervened, STORED_FEATURES.index("CAPACITY_REDUCTION")])
                    )
                    if n_intervened
                    else 0.0,
                    "mean_abs_response_intervened": float(abs_y[intervened].mean())
                    if n_intervened
                    else 0.0,
                    "median_abs_response_intervened": float(np.median(abs_y[intervened]))
                    if n_intervened
                    else 0.0,
                    "mean_abs_response_elsewhere": float(abs_y[~intervened].mean()),
                    "median_abs_response_elsewhere": float(np.median(abs_y[~intervened])),
                }
            )

        print(f"  scanned {path.name} ({len(scenarios)}/1000)", file=sys.stderr, flush=True)

    assert ref_x is not None and geometry is not None
    f_mean = f_sum / nodes
    f_std = np.sqrt(np.maximum(f_sq / nodes - f_mean**2, 0.0))
    y_mean = y_sum / nodes

    return {
        "graphs": len(scenarios),
        "node_counts": sorted(node_counts),
        "edge_counts": sorted(edge_counts),
        "isolated_counts": sorted(isolated_counts),
        "node_observations": nodes,
        "topology_constant": topology_constant,
        "feature_stats": {
            name: {
                "min": float(f_min[i]),
                "max": float(f_max[i]),
                "mean": float(f_mean[i]),
                "std": float(f_std[i]),
                "pct_zero": float(100 * f_zero[i] / nodes),
                "constant_across_scenarios": bool(constant[i]),
                "max_delta_across_scenarios": float(max_delta[i]),
            }
            for i, name in enumerate(STORED_FEATURES)
        },
        "target_stats": {
            "mean": y_mean,
            "std": float(np.sqrt(max(y_sq / nodes - y_mean**2, 0.0))),
            "min": float(y_min),
            "max": float(y_max),
            "pct_zero": float(100 * y_zero / nodes),
        },
        "scenarios": scenarios,
        "reference_features": ref_x,
        "geometry": geometry,
    }


def choose_representative(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic pick: nearest to the median intervened-link count, lowest index wins ties.

    Documented rather than eyeballed, so a rerun always selects the same scenario.
    """
    counts = np.array([s["intervened_links"] for s in scenarios])
    median = float(np.median(counts))
    distance = np.abs(counts - median)
    best = int(np.lexsort((np.arange(len(scenarios)), distance))[0])
    chosen = dict(scenarios[best])
    chosen["selection_rule"] = (
        "scenario whose intervened-link count is nearest the corpus median; "
        "ties broken by lowest scenario index"
    )
    chosen["corpus_median_intervened_links"] = median
    return chosen


def load_scenario(index: int) -> Any:
    """Re-read a single scenario without holding the corpus in memory."""
    import warnings

    import torch

    per_batch = 50
    path = CORPUS / f"datalist_batch_{index // per_batch + 1}.pt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = torch.load(path, map_location="cpu", weights_only=False)
    return batch[index % per_batch]


def project(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular projection, for display only.

    A degree of longitude is about cos(latitude) times a degree of latitude, so plotting raw
    WGS84 on equal axes stretches the network east-west. This corrects that for figures. The
    model consumed the raw WGS84 values; nothing here feeds back into it.
    """
    lat0 = float(np.mean(lat))
    return (lon - float(np.mean(lon))) * np.cos(np.radians(lat0)), lat - lat0


# --------------------------------------------------------------------------------------------
# Artifact builders
# --------------------------------------------------------------------------------------------


def build_feature_artifacts(scan: dict[str, Any]) -> dict[str, Any]:
    stats = scan["feature_stats"]
    stored, static, varying = [], [], []
    for name in STORED_FEATURES:
        semantics = FEATURE_SEMANTICS[name]
        entry = {"name": name, **semantics, **stats[name]}
        stored.append(entry)
        (static if stats[name]["constant_across_scenarios"] else varying).append(name)

    return {
        "features": {
            "provenance": provenance(
                AUDIT_CLASS,
                [f"{DATASET_RELEASE}/datalist_batch_1..20.pt"],
                metric_definition=(
                    "Per-column statistics over every node of all 1,000 scenarios "
                    "(31,635,000 node observations)."
                ),
                units="see per-feature unit field",
            ),
            "stored_feature_count": len(STORED_FEATURES),
            "model_feature_count": sum(
                1 for f in FEATURE_SEMANTICS.values() if f["model_index"] is not None
            ),
            "excluded_from_model": [
                n for n, f in FEATURE_SEMANTICS.items() if f["model_index"] is None
            ],
            "positional_input": {
                "note": (
                    "Segment endpoints are consumed separately by the two PointNet stages, "
                    "not as extra feature columns."
                ),
                "pos_0": GRAPH_SEMANTICS["pos_0"],
                "pos_1": GRAPH_SEMANTICS["pos_1"],
            },
            "graph_semantics": GRAPH_SEMANTICS,
            "target": TARGET_DEFINITION,
            "items": stored,
        },
        "feature_variability": {
            "provenance": provenance(
                AUDIT_CLASS,
                [f"{DATASET_RELEASE}/datalist_batch_1..20.pt"],
                metric_definition=(
                    "A column is static when its values are identical in all 1,000 scenarios."
                ),
            ),
            "static_features": static,
            "scenario_varying_features": varying,
            "topology_constant": scan["topology_constant"],
            "max_delta_across_scenarios": {
                n: stats[n]["max_delta_across_scenarios"] for n in STORED_FEATURES
            },
        },
    }


def build_intervention_artifact(scan: dict[str, Any]) -> dict[str, Any]:
    scenarios = scan["scenarios"]
    counts = np.array([s["intervened_links"] for s in scenarios], dtype=float)
    fractions = np.array([s["intervened_fraction_pct"] for s in scenarios])
    on = np.array([s["mean_abs_response_intervened"] for s in scenarios])
    off = np.array([s["mean_abs_response_elsewhere"] for s in scenarios])
    on_med = np.array([s["median_abs_response_intervened"] for s in scenarios])
    off_med = np.array([s["median_abs_response_elsewhere"] for s in scenarios])

    def describe(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(values.min()),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }

    # Guard the ratio: a scenario with a near-zero elsewhere-response would make it explode.
    safe = off > 1e-6
    ratio = on[safe] / off[safe]

    return {
        "provenance": provenance(
            AUDIT_CLASS,
            [f"{DATASET_RELEASE}/datalist_batch_1..20.pt"],
            metric_definition=(
                "Per scenario: segments with CAPACITY_REDUCTION != 0 are 'intervened'. "
                "Response is mean and median |Delta v| within each group. "
                "This is an association measured in the released corpus, not a causal estimate."
            ),
            units="veh/h",
        ),
        "scenarios_analysed": len(scenarios),
        "intervened_links": describe(counts),
        "intervened_fraction_pct": describe(fractions),
        "mean_abs_response_intervened": describe(on),
        "mean_abs_response_elsewhere": describe(off),
        "median_abs_response_intervened": describe(on_med),
        "median_abs_response_elsewhere": describe(off_med),
        "ratio_mean_response": {
            **describe(ratio),
            "scenarios_included": int(safe.sum()),
            "note": "Scenarios with an elsewhere-response at or below 1e-6 veh/h are excluded.",
        },
        "scenarios_where_intervened_exceeds_elsewhere": int((on > off).sum()),
    }


def build_scenario_artifacts(chosen: dict[str, Any], out: Path) -> dict[str, Any]:
    """Representative-scenario summary plus rendered network figures."""
    data = load_scenario(chosen["index"])
    x = data.x.numpy().astype(np.float64)
    y = data.y.numpy().astype(np.float64).ravel()
    pos = data.pos.numpy().astype(np.float64)
    reduction = x[:, STORED_FEATURES.index("CAPACITY_REDUCTION")]
    intervened = reduction != 0

    sx, sy = project(pos[:, 0, 0], pos[:, 0, 1])
    ex, ey = project(pos[:, 1, 0], pos[:, 1, 1])

    render_network(sx, sy, ex, ey, intervened, y, out)

    return {
        "provenance": provenance(
            AUDIT_CLASS,
            [f"{DATASET_RELEASE}/{chosen['batch_file']}"],
            metric_definition=(
                "A single scenario selected by a deterministic rule, summarised for display."
            ),
            units="veh/h",
        ),
        "scenario": {
            k: chosen[k]
            for k in (
                "index",
                "batch_file",
                "position_in_batch",
                "selection_rule",
                "corpus_median_intervened_links",
                "intervened_links",
                "intervened_fraction_pct",
                "total_reduction_veh_h",
                "median_reduction_veh_h",
                "mean_abs_response_intervened",
                "mean_abs_response_elsewhere",
            )
        },
        "segments": int(x.shape[0]),
        "response": {
            "max_abs": float(np.abs(y).max()),
            "pct_zero": float(100 * (y == 0).mean()),
            "mean_abs": float(np.abs(y).mean()),
        },
        "projection": {
            "model_input_crs": "EPSG:4326 (WGS84), raw degrees",
            "display_projection": (
                "local equirectangular about the network centroid, longitude scaled by "
                "cos(mean latitude)"
            ),
            "note": "The projection is applied for figures only and never fed back to the model.",
        },
        "figures": ["network_scenario.webp", "network_response.webp"],
    }


def render_network(
    sx: np.ndarray,
    sy: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    intervened: np.ndarray,
    y: np.ndarray,
    out: Path,
) -> None:
    """Render the network as WebP rasters.

    Rasters rather than vectors: 31,635 separate SVG paths would be megabytes. WebP rather than
    PNG: these are thin coloured lines on transparency, where PNG lands around 450 KB per figure
    and WebP holds the same detail in a fraction of it.

    Only two figures are produced. A plain base-network render is redundant, because the scenario
    figure already draws the untouched network underneath the intervention.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from PIL import Image

    stacked = np.stack([np.column_stack([sx, sy]), np.column_stack([ex, ey])], axis=1)
    span_x = float(sx.max() - sx.min())
    span_y = float(sy.max() - sy.min())
    figsize = (9.0, 9.0 * max(span_y / span_x, 0.25))

    def collection(mask: np.ndarray | None, **kwargs: Any) -> LineCollection:
        chosen = stacked if mask is None else stacked[mask]
        return LineCollection([list(map(tuple, pair)) for pair in chosen], **kwargs)

    def figure(name: str, draw) -> None:
        fig, ax = plt.subplots(figsize=figsize, dpi=118)
        draw(ax)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.margins(0.01)
        fig.tight_layout(pad=0.1)
        png = out / f"{name}.png"
        fig.savefig(png, transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        with Image.open(png) as image:
            image.save(out / f"{name}.webp", "WEBP", quality=72, method=6)
        png.unlink()

    def scenario(ax) -> None:
        ax.add_collection(collection(~intervened, colors="#c2c7d4", linewidths=0.16, alpha=0.55))
        ax.add_collection(collection(intervened, colors="#c2410c", linewidths=0.55, alpha=0.95))
        ax.autoscale_view()

    def response(ax) -> None:
        magnitude = np.abs(y)
        # Percentile clip: a handful of extreme segments would otherwise flatten the whole map.
        ceiling = float(np.percentile(magnitude, 99.5)) or 1.0
        quiet = magnitude <= 0.5
        ax.add_collection(collection(quiet, colors="#d5d9e2", linewidths=0.14, alpha=0.5))
        loud = ~quiet
        norm = np.clip(magnitude[loud] / ceiling, 0, 1)
        order = np.argsort(norm)
        ordered = stacked[loud][order]
        ax.add_collection(
            LineCollection(
                [list(map(tuple, pair)) for pair in ordered],
                colors=plt.get_cmap("magma_r")(0.15 + 0.85 * norm[order]),
                linewidths=0.2 + 0.9 * norm[order],
            )
        )
        ax.autoscale_view()

    figure("network_scenario", scenario)
    figure("network_response", response)


def build_result_artifacts(audit: dict[str, Any]) -> dict[str, Any]:
    """Copy audited thesis results. Nothing is recomputed here."""
    analyses = audit["analyses"]
    t8 = analyses["t8_mc"]
    sources = ["analysis_outputs/thesis_intelligence.json"]
    upstream = audit.get("source_provenance", {})

    selective = {
        "provenance": provenance(
            THESIS,
            sources,
            metric_definition=(
                "Predictions are ranked by MC-dropout sigma. At each retained fraction the "
                "least-uncertain share is kept and MAE is recomputed on it; the remainder is "
                "routed to review."
            ),
            units="veh/h",
            protocol="Trial 8, MC dropout with 30 stochastic passes, 100-graph held-out test set",
            upstream=upstream,
        ),
        "model": t8["method"],
        "row_count": t8["row_count"],
        "scope": t8["scope"],
        "curve": t8["selective_risk"],
    }

    uncertainty = {
        "provenance": provenance(
            THESIS,
            sources,
            metric_definition=(
                "spearman_rho ranks MC-dropout sigma against absolute error. "
                "raw_gaussian_coverage_90/95 is the share of targets inside "
                "prediction +/- z * sigma at nominal 90/95 percent, before any calibration. "
                "k90/k95 are the multipliers sigma would need for that nominal coverage."
            ),
            units="veh/h except correlations and coverages, which are unitless",
            protocol="Trial 8, MC dropout with 30 stochastic passes, 100-graph held-out test set",
            upstream=upstream,
        ),
        "point_metrics": t8["point_metrics"],
        "uncertainty_metrics": t8["uncertainty_metrics"],
        "error_detection": {
            "protocol": t8.get("error_detection_protocol"),
            "definition": (
                "A segment is positive when its deterministic absolute error exceeds the given "
                "percentile of the error distribution. The score being ranked is MC-dropout "
                "sigma. AUROC is therefore the probability that a randomly chosen large-error "
                "segment carries higher sigma than a randomly chosen ordinary one."
            ),
            "levels": t8["error_detection"],
        },
        "uncertainty_error_bins": t8["uncertainty_error_bins"],
    }

    calibration_source = audit["calibration_protocols"]
    calibration = {
        "provenance": provenance(
            THESIS,
            sources,
            metric_definition=(
                "Temperature scaling fits one scalar multiplier for sigma by minimising "
                "expected calibration error. It rescales the uncertainty; it does not change "
                "any point prediction."
            ),
            units="unitless",
            upstream=upstream,
        ),
        "protocols": [
            {
                "id": "graph20_80_v1",
                "label": "Graph-level audit protocol",
                "split": calibration_source["graph20_80_v1"]["split"],
                "status": calibration_source["graph20_80_v1"]["status"],
                "evidence_class": "post-thesis verified; recomputed from full cached arrays",
                "approximate": False,
                "ece_before": calibration_source["graph20_80_v1"]["evaluation_ece_before"],
                "ece_after": calibration_source["graph20_80_v1"]["evaluation_ece_after"],
                "temperature": calibration_source["graph20_80_v1"]["temperature"],
            },
            {
                "id": "node30_70_thesis_final",
                "label": "Final-thesis node protocol",
                "split": calibration_source["node30_70_thesis_final"]["split"],
                "status": calibration_source["node30_70_thesis_final"]["status"],
                "evidence_class": "approximate; canonical split indices unavailable",
                "approximate": True,
                "warning": calibration_source["node30_70_thesis_final"]["warning"],
                "ece_before": calibration_source["node30_70_thesis_final"][
                    "evaluation_ece_before_approx"
                ],
                "ece_after": calibration_source["node30_70_thesis_final"][
                    "evaluation_ece_after_approx"
                ],
                "temperature": calibration_source["node30_70_thesis_final"]["temperature_approx"],
            },
        ],
        "conditional_conformal": calibration_source["graph20_80_v1"]["conditional_conformal"],
    }

    comparison = {
        "provenance": provenance(
            THESIS,
            sources,
            metric_definition=(
                "Held-out test metrics per model. The CQR rows report interval coverage at "
                "nominal 90 and 95 percent; the gate records whether observed coverage met the "
                "nominal level."
            ),
            units="veh/h for mae and rmse; r2 and coverage unitless",
            upstream=upstream,
        ),
        "models": audit["reported_model_comparison"],
        "scope_notes": audit.get("scope_notes"),
    }

    return {
        "selective_risk": selective,
        "uncertainty": uncertainty,
        "calibration": calibration,
        "model_comparison": comparison,
    }


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------


def validate(artifacts: dict[str, Any], scanned: bool) -> list[str]:
    """Assert invariants that are genuinely authoritative, not incidental."""
    problems: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    features = artifacts["features"]
    check(features["stored_feature_count"] == 6, "stored feature count must be 6")
    check(features["model_feature_count"] == 5, "model feature count must be 5")
    check(
        features["excluded_from_model"] == ["HIGHWAY"],
        "HIGHWAY must be the only column excluded from the model",
    )
    check(
        [item["name"] for item in features["items"]] == list(STORED_FEATURES),
        "feature names must match the authoritative column order",
    )

    if scanned:
        variability = artifacts["feature_variability"]
        check(
            variability["scenario_varying_features"] == ["CAPACITY_REDUCTION"],
            "CAPACITY_REDUCTION must be the only scenario-varying column",
        )
        check(variability["topology_constant"] is True, "graph topology must be constant")
        scenario = artifacts["representative_scenario"]["scenario"]
        check(0 <= scenario["index"] < 1000, "representative scenario index out of range")
        check(
            0 < scenario["intervened_fraction_pct"] < 100,
            "intervened fraction must be a valid percentage",
        )

    curve = artifacts["selective_risk"]["curve"]
    check(len(curve) >= 2, "selective-risk curve needs at least two points")
    check(
        all(0 < point["retention"] <= 100 for point in curve),
        "selective-risk retention must be a valid percentage",
    )
    check(
        curve == sorted(curve, key=lambda p: p["retention"]),
        "selective-risk curve must be ordered by retention",
    )

    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            continue
        block = artifact.get("provenance")
        check(bool(block), f"{name} is missing a provenance block")
        if block:
            check(
                block.get("source_commit") not in (None, "", "unknown"),
                f"{name} has no resolved source commit",
            )
            check(
                block.get("classification") in (THESIS, AUDIT_CLASS),
                f"{name} has an invalid classification",
            )

    def finite(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                finite(value, f"{trail}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                finite(value, f"{trail}[{i}]")
        elif isinstance(node, float) and not np.isfinite(node):
            problems.append(f"non-finite value at {trail}")

    finite(artifacts, "artifacts")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="export only audited results; skip the corpus pass and figures",
    )
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    audit = load_audit()
    artifacts: dict[str, Any] = build_result_artifacts(audit)

    if args.skip_corpus:
        artifacts["features"] = build_feature_artifacts(
            {"feature_stats": {n: _empty_stats() for n in STORED_FEATURES}}
        )["features"]
    else:
        scan = scan_corpus()
        artifacts.update(build_feature_artifacts(scan))
        artifacts["intervention"] = build_intervention_artifact(scan)
        chosen = choose_representative(scan["scenarios"])
        artifacts["representative_scenario"] = build_scenario_artifacts(chosen, out)
        artifacts["corpus"] = {
            "provenance": provenance(
                AUDIT_CLASS,
                [f"{DATASET_RELEASE}/datalist_batch_1..20.pt"],
                metric_definition="Shape of the released training corpus.",
            ),
            "graphs": scan["graphs"],
            "segments_per_graph": scan["node_counts"],
            "graph_edges_per_graph": scan["edge_counts"],
            "node_observations": scan["node_observations"],
            "isolated_segments_per_graph": scan["isolated_counts"],
            "isolated_note": (
                "The line-graph transform yields fewer graph nodes than stored feature rows, "
                "leaving a contiguous tail of segments with no incident edge. Their target is "
                "exactly zero in every scanned scenario."
            ),
            "target": {**scan["target_stats"], **TARGET_DEFINITION},
            "graph_semantics": GRAPH_SEMANTICS,
        }

    problems = validate(artifacts, scanned=not args.skip_corpus)

    manifest = {
        "provenance": provenance(
            "manifest",
            ["analysis_outputs/thesis_intelligence.json", f"{DATASET_RELEASE}/*.pt"],
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        "artifacts": {},
    }

    for name, artifact in artifacts.items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["artifacts"][name] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "classification": artifact.get("provenance", {}).get("classification"),
        }

    for figure in sorted(out.glob("*.webp")):
        manifest["artifacts"][figure.stem] = {
            "file": figure.name,
            "bytes": figure.stat().st_size,
            "classification": AUDIT_CLASS,
        }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    total = sum(entry["bytes"] for entry in manifest["artifacts"].values())
    print(f"\nWrote {len(manifest['artifacts'])} artifacts to {out.name}/ ({total / 1024:.1f} KiB)")
    for name, entry in sorted(manifest["artifacts"].items()):
        print(f"  {entry['bytes'] / 1024:8.1f} KiB  {entry['file']:<28} {entry['classification']}")

    if problems:
        print("\nVALIDATION FAILED", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(1)
    print("\nValidation passed.")


def _empty_stats() -> dict[str, Any]:
    return {
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "pct_zero": 0.0,
        "constant_across_scenarios": False,
        "max_delta_across_scenarios": 0.0,
    }


if __name__ == "__main__":
    main()
