# Thesis Data, Model, and UQ Audit

Generated from trusted local artifacts on 2026-08-15. All paths are repository-relative and all exported values are aggregates.

Source: audited repository commit `fdb4ef0c9c736576ae34d5e331d8b66a7a6d877a`. This is a
post-submission audit; the immutable submitted PDF remains unchanged.

## Executive findings

- T8 MC Dropout was recomputed over 3,163,500 cached node predictions: MAE 3.9483, R2 0.5855, uncertainty-error Spearman rho 0.4818.
- The five-member Deep Ensemble improves point prediction to MAE 3.4853 and R2 0.6841, but ranks error less strongly (rho 0.3997).
- Raw Gaussian intervals are under-dispersed: T8 95% nominal coverage is 54.8%; calibration is required before uncertainty is interpreted as coverage.
- T8 selective prediction cuts accepted-set MAE from 3.948 to 2.321 veh/h at 50% retention. This is a review-capacity trade-off, not proof that rejected rows are incorrect.
- Local graph-tensor audit scope: 100 graphs / 3,163,500 nodes. Raw MATSim-to-graph regeneration remains unavailable in this checkout.
- The strongest methodology risk is split-specific scaling of evaluation data. Historical scores are retained but should be presented with that limitation.

## Data and feature provenance

The model input order is `VOL_BASE_CASE`, `CAPACITY_BASE_CASE`, `CAPACITY_REDUCTION`, `FREESPEED`, `LENGTH`. The target is policy-scenario `vol_car` minus base-case `vol_car`. Position tensors are start point, end point, midpoint; the model consumes start and end. `FREESPEED` means free-flow speed, not maximum speed.

The local loader contains normalized model-ready tensors, so its statistics diagnose schema, missingness, constancy, tails, and distribution shape. They do not recover physical-unit feature plausibility. The confidential ignored `data` junction and all pickle-capable inputs remain outside generated outputs.

## Model and uncertainty interpretation

MC sigma has useful ranking information, especially for routing scarce review capacity, but rho is association rather than causation or calibration. The target has 872,540 exact zeros (27.6%), not the previously reported 88.7%; class imbalance and degradation in the highest-change regime still make policy-critical tail behavior more demanding than pooled summaries.

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
