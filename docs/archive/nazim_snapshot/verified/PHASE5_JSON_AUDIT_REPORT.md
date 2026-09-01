# Phase 5: Visualization JSON Audit Report

**Date**: 2026-03-26
**Scope**: All 68 JSON files in the project
**Focus**: 10 Tier 1+2 visualization JSON files — full provenance and consistency audit
**Status**: COMPLETE — ALL PASS

---

## Executive Summary

| Metric | Result |
|--------|--------|
| Total JSON files in project | 68 |
| Visualization JSONs (Tier 1+2) | 10 |
| Additional visualization-adjacent JSONs (Tier 1 extended) | 7 |
| Producer/consumer mappings verified | 10/10 |
| JSON-to-figure consistency checks | 10/10 PASS |
| JSON-to-thesis-text cross-checks | Already verified in Phase 3 (130+ claims) |
| Files classified | 68/68 |
| Files needing manual review | 0 |
| HIGH-severity issues | 0 |
| LOW-severity issues | 3 (all portability/style, no correctness impact) |
| INFO notes | 8 (all by-design patterns) |
| Outdated/superseded files | 2 (experiment_a_results.json, experiment_b_results.json) |

---

## Phase 5 Sub-Task Results

### 5a: JSON File Inventory (COMPLETE)
- 68 JSON files identified across the project
- Tiered into 5 levels: Tier 1 (6 direct figure inputs), Tier 2 (4 indirect/compute inputs), Tier 3 (thesis numeric sources), Tier 4 (verification/audit), Tier 5 (infrastructure/non-visualization)

### 5b: JSON Schema Inspection (COMPLETE)
- All 10 Tier 1+2 files inspected for internal coherence
- All are script-generated summary files with machine-precision floats
- 6 early ambiguities documented (all INFO or LOW):
  - A1: pit_after_tempscaling duplicates pit_t8 "before" block (intentional, self-contained)
  - A2: t7_error_detection uses rho=0.4460 (100K subsample) vs 0.4437 (full population) — known Phase 3 note
  - A3: t7_error_detection t8_comparison block has rounded values (human reference only)
  - A4: conformal_conditional uses 20/80 split (expected, differs from main conformal)
  - A5: s_convergence uses 10 graphs (thesis states this explicitly)
  - A6: 5 files contain absolute Windows paths (provenance metadata only)

### 5c: Producer/Consumer Mapping (COMPLETE)
Full mapping saved to: `docs/verified/PHASE5C_JSON_PRODUCER_CONSUMER_MAP.md`

**Key findings:**
- Each of the 10 JSONs has exactly ONE producer script — no conflicting writers
- 5 produced by `generate_phase3_figures.py`, 4 by dedicated scripts, 1 by `regenerate_fig58_s30.py`
- 6 Tier 1 JSONs each have exactly ONE authoritative figure consumer (`run_fig*.py`)
- `pit_after_tempscaling_t8.json` is output-only (figure generated from raw NPZ, JSON is audit artifact)
- `run_fig518.py` uses 100% hardcoded values (does NOT read t7_error_detection.json)
- Absolute Windows paths in 5 JSONs are provenance-only (no runtime use)

**Producer scripts:**
| Producer Script | JSONs Produced |
|----------------|---------------|
| `figures/generate_phase3_figures.py` | conformal_conditional_coverage_t8, reliability_diagram_t8, temperature_scaling_t8, per_graph_variation_t8, stratified_uq_t8, t7_error_detection |
| `scripts/compute_pit.py` | pit_t8 |
| `scripts/compute_pit_after_tempscaling.py` | pit_after_tempscaling_t8 |
| `scripts/regenerate_fig58_s30.py` | selective_prediction_s30 |
| `scripts/run_s_convergence.py` | s_convergence_results |

### 5d: Consistency Verification (COMPLETE — 10/10 PASS)

| # | JSON | Consumer Script | Keys Correct | Assertions Pass | Output Filename Match | Verdict |
|---|------|----------------|-------------|----------------|----------------------|---------|
| 1 | conformal_conditional_coverage_t8.json | run_fig56.py | YES | N/A (no assertions) | YES (t8_conformal_conditional.pdf) | PASS |
| 2 | selective_prediction_s30.json | run_fig58.py | YES | N/A (no assertions) | YES (t8_selective_prediction_curve.pdf) | PASS |
| 3 | reliability_diagram_t8.json | run_fig511.py | YES | 2/2 pass | YES (t8_reliability_diagram.pdf) | PASS |
| 4 | temperature_scaling_t8.json | run_fig512.py | YES | 3/3 pass | YES (t8_temperature_scaling.pdf) | PASS |
| 5 | pit_t8.json | run_fig513.py | YES | 4/4 pass | YES (t8_pit_histogram.pdf) | PASS |
| 6 | s_convergence_results.json | run_fig61.py | YES | 9/9 pass | YES (t8_s_convergence.pdf) | PASS |
| 7 | per_graph_variation_t8.json | generate_phase3_figures.py (self) | YES | In-memory identity | YES (t8_per_graph_variation.pdf) | PASS |
| 8 | stratified_uq_t8.json | generate_phase3_figures.py (self) | YES | In-memory identity | YES (t8_stratified_uq.pdf) | PASS |
| 9 | pit_after_tempscaling_t8.json | NONE (output-only) | N/A | Figure from raw NPZ | YES (t8_pit_after_tempscaling.pdf) | PASS |
| 10 | t7_error_detection.json | run_fig515.py (cross-ref) | YES (3 keys) | 3/3 pass | YES (t7_selective_prediction_curve.pdf) | PASS |

**Total assertions verified: 21/21 PASS**

### 5e: Classification (COMPLETE — 68/68 classified)

| Category | Count | Description |
|----------|-------|-------------|
| CURRENT/FINAL — Visualization | 17 | Directly consumed by figure generators |
| CURRENT/FINAL — Thesis Numeric Source | 12 | Numbers cited in thesis text |
| CURRENT/FINAL — Verification/Audit | 10 | Integrity/audit check results |
| CURRENT/FINAL — Training Infrastructure | 8 | Config files, training params |
| SUPPLEMENTARY | 20 | Earlier-trial metrics, intermediate analyses |
| OUTDATED/SUPERSEDED | 2 | Pre-fix ensemble experiments (known GATConv bug) |
| NEEDS REVIEW | 0 | None |

---

## Issues Register

### LOW Severity (3 items — portability/style only)

| ID | JSON File | Issue | Impact |
|----|-----------|-------|--------|
| L1 | s_convergence_results.json | 2 consumer scripts use hardcoded absolute Windows paths | No correctness impact; thesis built on same machine |
| L2 | generate_all_hd_plots.py consumers | Script uses hardcoded absolute REPO path | HD plots are supplementary, not thesis-critical |
| L3 | temperature_scaling_t8.json | T_OPTIMAL=2.7025 hardcoded in compute_pit_after_tempscaling.py | Matches JSON value; documented in code comment |

### INFO Notes (8 items — by-design patterns)

| ID | Note |
|----|------|
| I1 | pit_after_tempscaling_t8.json is output-only (no script reads it for figure generation) |
| I2 | run_fig518.py uses 100% hardcoded values with 18 inline assertions |
| I3 | per_graph_variation_t8.json and stratified_uq_t8.json are produced and consumed in same function |
| I4 | 5 JSON files contain absolute Windows paths in "data_source" metadata (provenance only) |
| I5 | t7_error_detection.json rho=0.4460 is 100K subsample (full population: 0.4437) — documented Phase 3 note |
| I6 | s_convergence uses 10 graphs (subset analysis); thesis states this explicitly |
| I7 | conformal_conditional uses 20/80 cal/eval split (expected, differs from main conformal) |
| I8 | run_fig61.py double tight_layout call (minor visual, no data impact) |

### OUTDATED Files (2 items — recommend archive in Phase 9)

| File | Reason |
|------|--------|
| experiment_a_results.json | Pre-fix: 2 runs, S=10, 3 graphs. Replaced by experiment_a_fixed_results.json |
| experiment_b_results.json | Pre-fix: GATConv strict=False bug produced negative R2. Replaced by experiment_b_fixed_results.json |

---

## Conclusion

Phase 5 is **COMPLETE**. All 68 JSON files have been inventoried, schema-inspected, producer/consumer-mapped, consistency-verified, and classified. The 10 visualization JSONs that feed thesis figures are all correct, internally coherent, and properly consumed by their authoritative scripts.

**Zero correctness issues found. Zero thesis-text mismatches. The JSON data pipeline is verified end-to-end.**
