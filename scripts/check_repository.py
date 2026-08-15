"""Validate the minimum artifact set required to reproduce reported analyses."""

import json
import py_compile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REQUIRED_PATHS = (
    "README.md",
    "CITATION.cff",
    "environment-minimal.yml",
    "docs/verified/VERIFIED_RESULTS_MASTER.csv",
    "docs/verified/REPRODUCIBILITY_GAP_SUMMARY.md",
    "docs/verified/AUDIT_SUMMARY.md",
    "thesis/latex_tum_official/main.tex",
    "thesis/latex_tum_official/main.pdf",
    "results/phase3/pit_t8.json",
    "results/phase3/temperature_scaling_t8.json",
    "models/deep_ensemble_seed42/trained_model/model.pth",
    "models/point_net_transf_gat_8th_trial_lower_dropout/trained_model/model.pth",
    "results/predictions/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz",
    "scripts/evaluation/run_part2_uq_analyses.py",
    "scripts/evaluation/run_part3_calibration_audit.py",
    "scripts/evaluation/run_part4_t7_crosscheck.py",
    "scripts/analysis/generate_thesis_intelligence.py",
    "analysis_outputs/thesis_intelligence.json",
    "analysis_outputs/THESIS_INTELLIGENCE_REPORT.md",
    "thesis_dashboard/app.py",
    "thesis_dashboard/analytics.py",
    ".streamlit/config.toml",
)


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not (REPO / path).is_file()]
    if missing:
        raise SystemExit("Missing required artifacts:\n- " + "\n- ".join(missing))

    for path in REQUIRED_PATHS:
        if path.endswith(".py"):
            py_compile.compile(str(REPO / path), doraise=True)

    with (REPO / "analysis_outputs/thesis_intelligence.json").open(
        encoding="utf-8"
    ) as handle:
        bundle = json.load(handle)
    if bundle.get("schema_version") != "1.0.0":
        raise SystemExit("Unsupported thesis intelligence schema")
    privacy = bundle.get("privacy", {})
    if privacy.get("contains_row_level_records") is not False:
        raise SystemExit("Aggregate bundle privacy contract is invalid")
    target = bundle["analyses"]["t8_mc"]["quality"]["targets"]
    if target["count"] != 3_163_500 or target["zero_count"] != 872_540:
        raise SystemExit("T8 target audit values do not match the validated artifact")

    print(f"Repository check passed: {len(REQUIRED_PATHS)} required artifacts available.")


if __name__ == "__main__":
    main()
