# Diagrams

Two sets live here.

## Current set — `01_` … `07_`

Referenced from the top-level `README.md` and written in September 2026. Every tensor shape, layer
width and number in them was read from the artifacts rather than from prose:

| File | Shows |
| --- | --- |
| `01_research_problem.svg` | Simulator → graph → surrogate → prediction, and where uncertainty enters |
| `02_dataset_pipeline.svg` | Scenario generation to model input, marking which stages were retained |
| `03_feature_representation.svg` | What a node is; all six features with measured statistics |
| `04_model_architecture.svg` | Layer widths read from the Trial 8 checkpoint state dict |
| `05_training_evaluation.svg` | Training loop and the frozen test evaluation |
| `06_uncertainty_pipeline.svg` | MC Dropout, and the two questions asked of sigma |
| `07_evaluation_framework.svg` | How ranking, calibration, coverage and utility relate |

Layer widths in `04` come from
`models/point_net_transf_gat_8th_trial_lower_dropout/trained_model/model.pth`; dataset numbers in
`02` and `03` from `docs/DATASET.md`; result numbers from the artifacts that
`scripts/verify_headline_results.py` checks.

## Earlier set — unprefixed filenames

`architecture.svg`, `pipeline.svg`, `calibration.svg`, `conformal_coverage.svg`,
`results_overview.svg`, `selective_prediction.svg`, `stratified_uq.svg`.

These predate the current set and are no longer referenced from any document. They are kept because
they are part of the project's history, not because they are maintained. Where they disagree with the
current set or with [`../CORRIGENDUM.md`](../CORRIGENDUM.md), the current set is correct.
