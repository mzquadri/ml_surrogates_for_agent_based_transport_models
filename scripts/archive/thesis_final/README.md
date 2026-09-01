# Archived one-off scripts

Superseded single-use scripts kept for provenance. Nothing here is part of the
reproduction path — `code/scripts/evaluation/`, `code/scripts/figure_generation/`,
and `code/scripts/training/` are the maintained entry points.

They were moved out of `code/scripts/misc/` so that directory matches the
upstream repository (`enatterer/ml_surrogates_for_agent_based_transport_models`)
again. Archived rather than deleted because several produced figures that appear
in the thesis, and that trail is worth keeping.

## What is here

| Group | Files | Purpose |
| --- | --- | --- |
| Batch figure generation | `gen_batch1.py` … `gen_batch7.py`, `gen_clean_v2.py` | Successive passes over the thesis figure set |
| Single-figure regeneration | `regen_fig08.py`, `regen_fig10.py`, `regen_fig12.py`, `regen_fig12_single.py`, `regen_fig22.py`, `regen_fig28.py`, `regen_affected_figures.py`, `gen_fig17_34.py`, `gen_fig35.py`, `fig29_vertical.py`, `fig34_correct.py`, `fig36_winkler_comparison.py` | Targeted re-renders of individual figures |
| Layout and style passes | `soften_figs.py`, `unify_figs.py`, `final_figs.py`, `final_layout_figs.py`, `last_layout_pass.py` | Style passes applied across finished figures |
| Presentation | `slide_architecture.py`, `slide_non_gnn_compare.py`, `slide_non_gnn_compare_v2.py`, `gen_schematics.py` | Slide and schematic generation |
| LaTeX text fixes | `fix_alg_syntax.py`, `force_H.py`, `replace_emdashes.py` | One-off rewrites over the thesis `.tex` sources |
| Reporting | `gen_summary.py`, `sanity_check_table.py` | Ad-hoc summary tables |
| Shared helper | `plot_style.py` | Matplotlib style imported by 18 of the scripts above |

## Running them

`plot_style.py` was moved here **with** its dependents — every script that
imports it is in this directory, so `from plot_style import ...` still resolves
when run from here:

```bash
cd code/scripts/archive
python gen_batch1.py
```

Do not move `plot_style.py` back on its own; the imports break if it and its
dependents are split across directories.

## Deliberately not archived

These stayed in `code/scripts/misc/` because they are maintained QA tools, not
one-offs, and none of them import `plot_style`:

- `verify_bib.py`, `verify_figures.py`, `consistency_check.py`
- `feature_importance.py` and the two analysis notebooks (also present upstream)

Paths inside the archived scripts were written for the layout at the time they
ran and may need adjusting.
