from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from analytics import load_prediction_arrays, sample_rows

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "analysis_outputs" / "thesis_intelligence.json"

MODEL_OPTIONS = {
    "Trial 8 MC Dropout": {
        "key": "t8_mc",
        "artifact": ROOT
        / "results/predictions/point_net_transf_gat_8th_trial_lower_dropout/"
        "uq_results/mc_dropout_full_100graphs_mc30.npz",
        "role": "Primary single-model policy surrogate",
    },
    "Trial 7 MC Dropout": {
        "key": "t7_mc",
        "artifact": ROOT
        / "results/predictions/point_net_transf_gat_7th_trial_80_10_10_split/"
        "uq_results/mc_dropout_full_100graphs_mc30.npz",
        "role": "Independent architecture-matched cross-check",
    },
    "Deep Ensemble": {
        "key": "deep_ensemble",
        "artifact": None,
        "role": "Five-model point-accuracy benchmark",
    },
}

st.set_page_config(
    page_title="Traffic Policy Confidence Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #172622;
      --forest: #0f5b4d;
      --forest-soft: #dcebe5;
      --amber: #d97732;
      --paper: #f7f2e7;
      --line: #d7d2c5;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #183e36; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #f5f0e5 !important; }
    [data-testid="stMetric"] {
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--line);
      border-radius: 2px;
      padding: 0.75rem 0.9rem;
      box-shadow: 0 8px 24px rgba(23, 38, 34, 0.06);
    }
    [data-testid="stMetricValue"] { color: var(--forest); }
    .eyebrow {
      color: var(--amber);
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    .hero {
      border-top: 4px solid var(--forest);
      border-bottom: 1px solid var(--line);
      padding: 1.2rem 0 1.35rem 0;
      margin-bottom: 1rem;
    }
    .hero h1 { color: var(--ink); font-size: clamp(2.1rem, 5vw, 4.5rem); line-height: 0.95; }
    .hero p { color: #4d5d58; font-size: 1.05rem; max-width: 68rem; }
    .evidence-note {
      border-left: 4px solid var(--amber);
      background: rgba(255, 255, 255, 0.65);
      padding: 0.85rem 1rem;
      margin: 0.5rem 0 1rem;
    }
    .status-chip {
      display: inline-block;
      background: var(--forest-soft);
      color: var(--forest);
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      padding: 0.3rem 0.55rem;
      text-transform: uppercase;
    }
    div[data-baseweb="tab-list"] { border-bottom: 1px solid var(--line); gap: 0.5rem; }
    button[data-baseweb="tab"] { font-weight: 650; }
    @media (max-width: 720px) {
      .hero h1 { font-size: 2.35rem; }
      [data-testid="stMetric"] { padding: 0.55rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_bundle(path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    with Path(path).open(encoding="utf-8") as handle:
        bundle = json.load(handle)
    if bundle.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported analysis bundle schema")
    return bundle


@st.cache_data(show_spinner="Preparing a deterministic evidence sample...")
def load_plot_sample(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    predictions, uncertainties, targets = load_prediction_arrays(path)
    sample = sample_rows(predictions, uncertainties, targets, sample_size=12_000, seed=42)
    return pd.DataFrame(sample)


def curve_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame.rename(
        columns={
            "retention": "Retention (%)",
            "mae": "MAE (veh/h)",
            "rmse": "RMSE (veh/h)",
            "accepted": "Accepted",
            "reviewed": "Review queue",
            "reduction_pct": "MAE reduction (%)",
            "uncertainty_threshold": "Sigma threshold",
        }
    )


if not BUNDLE_PATH.is_file():
    st.error(
        "The aggregate analysis bundle is missing. Run "
        "`python scripts/analysis/generate_thesis_intelligence.py` first."
    )
    st.stop()

bundle = load_bundle(str(BUNDLE_PATH), BUNDLE_PATH.stat().st_mtime_ns)

with st.sidebar:
    st.markdown("## Policy controls")
    selected_name = st.selectbox("Evidence source", tuple(MODEL_OPTIONS))
    retention = st.slider("Automatic acceptance", 10, 100, 50, 5, format="%d%%")
    st.caption(MODEL_OPTIONS[selected_name]["role"])
    st.markdown("---")
    st.markdown("### Reading rule")
    st.caption(
        "Low-sigma predictions are accepted first. Remaining rows enter a review queue; "
        "they are not discarded or declared wrong."
    )
    st.markdown("---")
    st.markdown(
        '<span class="status-chip">Local-only evidence</span>', unsafe_allow_html=True
    )
    st.caption("Telemetry off · 127.0.0.1 · aggregate export only")

selected = MODEL_OPTIONS[selected_name]
analysis = bundle["analyses"][selected["key"]]
curve = curve_frame(analysis["selective_risk"])
policy_row = curve.loc[curve["Retention (%)"] == retention].iloc[0]
point = analysis["point_metrics"]
uq = analysis["uncertainty_metrics"]

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Uncertainty quantification · operational evidence</div>
      <h1>Traffic Policy<br>Confidence Lab</h1>
      <p>Translate held-out GNN evidence into review capacity, calibrated intervals, and
      defensible model limitations. Every headline value is linked to a local aggregate
      bundle; no source data leaves this machine.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(5)
metric_columns[0].metric("Full-data R2", f"{point['r2']:.3f}")
metric_columns[1].metric("Full-data MAE", f"{point['mae']:.2f} veh/h")
metric_columns[2].metric("UQ rank rho", f"{uq['spearman_rho']:.3f}")
metric_columns[3].metric("Accepted rows", f"{int(policy_row['Accepted']):,}")
metric_columns[4].metric(
    "Accepted-set MAE",
    f"{policy_row['MAE (veh/h)']:.2f} veh/h",
    f"-{policy_row['MAE reduction (%)']:.1f}% vs full set",
)

st.markdown(
    f"""
    <div class="evidence-note"><strong>Decision at {retention}% retention.</strong>
    Automatically accept the {int(policy_row['Accepted']):,} least-uncertain node predictions
    and route {int(policy_row['Review queue']):,} to review. The retrospective held-out MAE
    among accepted rows is {policy_row['MAE (veh/h)']:.3f} veh/h. This threshold is evidence
    for workload planning, not a guaranteed deployment error bound.</div>
    """,
    unsafe_allow_html=True,
)

overview_tab, data_tab, models_tab, uq_tab, calibration_tab, audit_tab = st.tabs(
    [
        "Policy desk",
        "Data evidence",
        "Model bench",
        "UQ diagnostics",
        "Calibration",
        "Audit trail",
    ]
)

with overview_tab:
    left, right = st.columns((1.55, 1))
    with left:
        st.subheader("Risk versus review capacity")
        st.line_chart(
            curve.set_index("Retention (%)")[["MAE (veh/h)", "RMSE (veh/h)"]],
            color=["#0f5b4d", "#d97732"],
        )
    with right:
        st.subheader("Operational threshold")
        st.metric("Maximum accepted sigma", f"{policy_row['Sigma threshold']:.3f}")
        st.metric("Review share", f"{100 - retention}%")
        st.metric("Retrospective MAE reduction", f"{policy_row['MAE reduction (%)']:.1f}%")
        st.caption(
            "The sigma threshold is model- and archive-specific. Recalibrate it after any "
            "model, preprocessing, network, or policy-distribution change."
        )
    with st.expander("Full retention schedule"):
        st.dataframe(curve, hide_index=True, use_container_width=True)

with data_tab:
    graph_quality = bundle["graph_data_quality"]
    st.subheader("Held-out graph tensor quality")
    if graph_quality["available"]:
        qcols = st.columns(4)
        qcols[0].metric("Graphs", f"{graph_quality['graph_count']:,}")
        qcols[1].metric("Nodes", f"{graph_quality['node_count']:,}")
        qcols[2].metric("Edges", f"{graph_quality['edge_count']:,}")
        qcols[3].metric("Exact-zero targets", f"{graph_quality['target_zero_fraction']:.1%}")
        feature_rows = []
        for feature in bundle["feature_dictionary"]:
            stats = graph_quality["features"][feature["name"]]
            feature_rows.append(
                {
                    "Index": feature["model_index"],
                    "Feature": feature["name"],
                    "Executable meaning": feature["meaning"],
                    "Mean (normalized)": stats["mean"],
                    "Std (normalized)": stats["std"],
                    "Missing": stats["non_finite_count"],
                    "Unique": stats["unique_count"],
                    "IQR outliers": stats["outlier_count_iqr"],
                }
            )
        st.dataframe(pd.DataFrame(feature_rows), hide_index=True, use_container_width=True)
        st.warning(graph_quality["limitation"])
    else:
        st.info(graph_quality["limitation"])

    st.subheader("Prediction artifact quality")
    quality_rows = []
    for variable, stats in analysis["quality"].items():
        quality_rows.append(
            {
                "Variable": variable,
                "Rows": stats["count"],
                "Missing/non-finite": stats["non_finite_count"],
                "Minimum": stats["min"],
                "Median": stats["median"],
                "Maximum": stats["max"],
                "Skewness": stats["skewness"],
                "IQR outliers": stats["outlier_count_iqr"],
                "Range failures": stats["plausible_range_failure_count"],
            }
        )
    st.dataframe(pd.DataFrame(quality_rows), hide_index=True, use_container_width=True)
    st.caption("Quality metrics use the full cached arrays, not the plotting sample.")

with models_tab:
    st.subheader("Point prediction and interval-head comparison")
    comparison = pd.DataFrame(bundle["reported_model_comparison"])
    display_columns = [
        "model",
        "r2",
        "mae",
        "rmse",
        "coverage_90",
        "coverage_95",
        "gate",
        "protocol",
    ]
    st.dataframe(comparison.reindex(columns=display_columns), hide_index=True, use_container_width=True)
    st.image(
        str(ROOT / "analysis_outputs/figures/model_r2_comparison.svg"),
        caption="Protocol labels matter: CQR test metrics are reported-only because test arrays are absent.",
    )
    st.markdown(
        """
        **Readout.** The Deep Ensemble has the strongest cached full-test point accuracy.
        T8 MC Dropout ranks errors more strongly than the ensemble and is cheaper to retain as
        a single-model deployment candidate. T11 passes its reported CQR gate; T10 does not.
        These methods answer different questions, so no single score establishes dominance.
        """
    )

with uq_tab:
    left, right = st.columns((1.35, 1))
    with left:
        st.subheader("Uncertainty versus realized error")
        bins = pd.DataFrame(analysis["uncertainty_error_bins"])
        st.line_chart(
            bins.set_index("x_mean")[["y_mean", "y_median"]],
            color=["#0f5b4d", "#d97732"],
            x_label="Mean sigma in quantile bin",
            y_label="Absolute error (veh/h)",
        )
        st.caption("20 equal-frequency bins over the full cached artifact.")
    with right:
        st.subheader("Raw interval warning")
        st.metric("90% nominal raw coverage", f"{uq['raw_gaussian_coverage_90']:.1%}")
        st.metric("95% nominal raw coverage", f"{uq['raw_gaussian_coverage_95']:.1%}")
        st.metric("Empirical k95", f"{uq['k95']:.2f} x sigma")
        st.warning(
            "Raw MC or ensemble spread is not a calibrated predictive interval. Use a named "
            "held-out calibration protocol before making coverage claims."
        )

    artifact = selected["artifact"]
    if artifact is not None and artifact.is_file():
        with st.expander("Deterministic 12,000-row visual sample"):
            sample = load_plot_sample(str(artifact), artifact.stat().st_mtime_ns)
            st.scatter_chart(
                sample,
                x="uncertainty",
                y="absolute_error",
                color="#d97732",
                size=8,
                x_label="MC sigma",
                y_label="Absolute error (veh/h)",
            )
            st.caption(
                "Plot only: sample without replacement, seed 42. Headline and binned metrics "
                "remain full-data calculations."
            )
    else:
        st.caption("Row-level ensemble plotting is intentionally disabled; aggregates are shown.")

    if "error_detection" in analysis:
        st.subheader("Large-error detection")
        st.dataframe(
            pd.DataFrame(analysis["error_detection"]),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(analysis["error_detection_protocol"])

with calibration_tab:
    st.subheader("Calibration protocols must stay separate")
    tracked = bundle["calibration_protocols"]["graph20_80_v1"]
    final_thesis = bundle["calibration_protocols"]["node30_70_thesis_final"]
    if selected["key"] != "t8_mc":
        st.warning(
            f"Calibration below belongs to Trial 8, not {selected_name}. No model-matched "
            "calibration artifact is available for the selected source."
        )
    protocol_columns = st.columns(2)
    with protocol_columns[0]:
        st.markdown("#### `graph20_80_v1` · tracked")
        st.metric("Temperature", f"{tracked['temperature']:.3f}")
        st.metric(
            "Evaluation ECE",
            f"{tracked['evaluation_ece_after']:.3f}",
            f"-{tracked['evaluation_ece_improvement_pct']:.1f}%",
        )
        st.caption(tracked["split"])
    with protocol_columns[1]:
        st.markdown("#### `node30_70_thesis_final` · reported")
        st.metric("Temperature", f"~{final_thesis['temperature_approx']:.3f}")
        st.metric(
            "Evaluation ECE",
            f"~{final_thesis['evaluation_ece_after_approx']:.3f}",
            f"from ~{final_thesis['evaluation_ece_before_approx']:.3f}",
        )
        st.caption(final_thesis["split"])
    st.image(str(ROOT / "analysis_outputs/figures/temperature_reliability_graph20_80.svg"))

    st.subheader("Conditional 90% coverage by uncertainty decile")
    deciles = pd.DataFrame(tracked["conditional_conformal"]["sigma_deciles"])
    coverage = deciles.rename(
        columns={
            "decile": "Sigma decile",
            "global_coverage_90": "Global interval",
            "adaptive_coverage_90": "Adaptive interval",
        }
    ).set_index("Sigma decile")
    st.line_chart(
        coverage[["Global interval", "Adaptive interval"]],
        color=["#d97732", "#0f5b4d"],
        y_label="Empirical coverage",
    )
    st.caption(
        "Global intervals under-cover the highest-sigma decile. Coverage is marginal unless "
        "the evaluated stratum and split are named."
    )

with audit_tab:
    st.subheader("Methodology discrepancy ledger")
    discrepancies = pd.DataFrame(bundle["discrepancies"])
    st.dataframe(discrepancies, hide_index=True, use_container_width=True)

    st.subheader("Artifact provenance")
    st.caption(
        "Availability and hashes describe the audited source checkout, not files shipped in "
        "this canonical repository. Row-level source artifacts remain excluded."
    )
    manifest = pd.DataFrame(bundle["artifact_manifest"])
    st.dataframe(
        manifest[["name", "path", "exists", "bytes", "sha256", "trust_boundary"]],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Safe export")
    st.download_button(
        "Download aggregate evidence bundle",
        data=json.dumps(bundle, indent=2, sort_keys=True),
        file_name="traffic_policy_confidence_aggregates.json",
        mime="application/json",
        help="Contains aggregate statistics, provenance, and report metadata only.",
    )
    st.caption(
        "No node-level rows, graph topology, samples, pickle payloads, absolute paths, or "
        "confidential junction contents are included."
    )

st.markdown("---")
st.caption(
    f"Aggregate schema {bundle['schema_version']} · generated {bundle['generated_on']} · "
    "correlation supports ranking, not causation or calibration"
)
