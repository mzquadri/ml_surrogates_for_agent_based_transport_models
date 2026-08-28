# Archived one-off scripts

Superseded single-use scripts kept for provenance. Nothing here is part of the
reproduction path — `scripts/evaluation/`, `scripts/figure_generation/`, and
`scripts/training/` are the maintained entry points.

These were moved out of `scripts/misc/` so that directory matches the upstream
repository (`enatterer/ml_surrogates_for_agent_based_transport_models`) again.
They are archived rather than deleted because several produced figures that
appear in the thesis, and that trail is worth keeping.

## What is here

| Group | Files | Purpose |
| --- | --- | --- |
| Batch figure generation | `gen_batch1.py` … `gen_batch7.py`, `gen_clean_v2.py` | Successive passes over the thesis figure set |
| Single-figure regeneration | `regen_fig08.py`, `regen_fig12.py`, `regen_fig12_single.py`, `regen_fig22.py`, `regen_fig28.py`, `gen_fig17_34.py`, `gen_fig35.py` | Targeted re-renders of individual figures |
| Figure post-processing | `soften_figs.py`, `unify_figs.py`, `final_figs.py` | Style passes applied across finished figures |
| Presentation | `generate_presentation.py`, `slide_non_gnn_compare.py`, `slide_non_gnn_compare_v2.py`, `gen_schematics.py` | Built `presentation/thesis_presentation_final.pptx` and its slides |
| Submission packaging | `build_submission_final.py`, `create_cross_check_package.py` | Assembled hand-in bundles |
| Reporting | `gen_summary.py`, `sanity_check_table.py`, `get_timeline.py` | Ad-hoc summary tables and timelines |
| Shared helper | `plot_style.py` | Matplotlib style imported by the scripts above |

## Running them

`plot_style.py` was moved here **with** its dependents — all 18 scripts that
import it are in this directory, so `from plot_style import ...` still resolves
when a script is run from here:

```bash
cd scripts/archive
python gen_batch1.py
```

Do not move `plot_style.py` back on its own; the imports break if it and its
dependents are split across directories.

`scripts/figure_generation/thesis_style.py` is a different module and was
deliberately **not** archived — 26 files outside this directory import it,
including live evaluation scripts.

Paths inside these scripts were written for the layout at the time they ran and
may need adjusting.
