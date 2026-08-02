"""Validate the minimum artifact set required to reproduce reported analyses."""

from pathlib import Path
import py_compile


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
)


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not (REPO / path).is_file()]
    if missing:
        raise SystemExit("Missing required artifacts:\n- " + "\n- ".join(missing))

    for path in REQUIRED_PATHS:
        if path.endswith(".py"):
            py_compile.compile(REPO / path, doraise=True)

    print(f"Repository check passed: {len(REQUIRED_PATHS)} required artifacts available.")


if __name__ == "__main__":
    main()
