# REORGANIZATION PLAN (Phase 5/6)

**Created:** 2026-03-19
**Principle:** Move clutter into `_archive/`, never delete. Do NOT rename or move any file referenced by code or LaTeX.

---

## Current Issues

1. **31 loose PNGs + 1 HTML** in `thesis/` — old visualization drafts, not referenced by LaTeX
2. **11 scripts** in `thesis/ARCHIVED_OLD_SCRIPTS/` — already marked as archived but not in `_archive/`
3. **2 stray PNGs** in `scripts/` — analysis outputs, not referenced
4. **`docs/MEETING_PREPARATION.md`** — 103 KB, contains known incorrect hyperparameters
5. **`evaluation_scripts/`** — 6 legacy scripts, all with hardcoded paths, not used by thesis pipeline
6. **`docs/` root** has 14 files mixing verified docs with informal summaries and HTML dashboards

---

## Proposed Moves (safe — nothing deleted, nothing renamed)

### Move 1: thesis/ loose figures → _archive/
```
MOVE: thesis/*.png (30 files)  →  _archive/thesis_draft_figures/
MOVE: thesis/DEFENSE_PRESENTATION.html  →  _archive/thesis_draft_figures/
```
**Why:** These are numbered draft figures (01–30) for early presentations. None are referenced by `\includegraphics` in the LaTeX source (all LaTeX figures are in `thesis/latex_tum_official/figures/`).

### Move 2: thesis/ARCHIVED_OLD_SCRIPTS/ → _archive/
```
MOVE: thesis/ARCHIVED_OLD_SCRIPTS/ (11 files)  →  _archive/archived_old_scripts/
```
**Why:** Already labeled "archived" — should live in the canonical `_archive/` directory.

### Move 3: scripts/ stray PNGs → _archive/
```
MOVE: scripts/trial8_complete_mc_dropout_analysis.png  →  _archive/stray_outputs/
MOVE: scripts/trial8_detailed_metrics_dashboard.png    →  _archive/stray_outputs/
```
**Why:** Generated outputs that don't belong in a source code directory.

### Move 4: MEETING_PREPARATION.md → _archive/
```
MOVE: docs/MEETING_PREPARATION.md  →  _archive/docs_deprecated/
```
**Why:** Contains known incorrect hyperparameters (documented in `docs/verified/OLD_CLAIMS_AUDIT.md`). Keeping it in `docs/` risks a professor reading wrong numbers.

### Move 5: evaluation_scripts/ → _archive/
```
MOVE: evaluation_scripts/ (10 files)  →  _archive/evaluation_scripts_legacy/
```
**Why:** All 6 scripts have hardcoded `C:\Users\zamin\...` paths. They are legacy analysis scripts from early trials (model1 comparison, Elena comparison, Colab validation). None are referenced by the thesis pipeline or UQ runner scripts.

### Move 6 (OPTIONAL): docs/ HTML dashboards → _archive/
```
MOVE: docs/THESIS_COMPLETE_DASHBOARD.html      →  _archive/docs_dashboards/
MOVE: docs/THESIS_COMPLETE_DASHBOARD_V2.html   →  _archive/docs_dashboards/
MOVE: docs/UQ_FINDINGS_DASHBOARD.html          →  _archive/docs_dashboards/
MOVE: docs/UQ_FINDINGS_DASHBOARD_v2.html       →  _archive/docs_dashboards/
```
**Why:** Large HTML files (33–198 KB) are working dashboards, not submission docs. Optional because they're harmless.

---

## What NOT to move

| Item | Reason to keep in place |
|---|---|
| `data/TR-C_Benchmarks/*/` | Referenced by runner scripts via relative paths |
| `data/train_data/` | Training data batches |
| `scripts/evaluation/` | Active UQ code, referenced by runners |
| `scripts/gnn/` | Model architecture + `help_functions.py` (MC dropout) |
| `thesis/latex_tum_official/` | Active LaTeX source + figures |
| `docs/verified/` | Ground-truth verification documents |
| `docs/*.md` (non-meeting) | Research summaries, may be useful for examiner |
| `run_part*.py` | Main entry points |
| `environment-minimal.yml` | Cross-platform env |
| `.github/copilot-instructions.md` | IDE config |

---

## Resulting Clean Structure (after moves)

```
repo/
├── README_SUBMISSION.md              # Submission overview
├── environment-minimal.yml           # Cross-platform conda env
├── traffic-gnn.yml                   # Full conda env (Linux)
├── .gitignore
├── run_part2_uq_analyses.py          # UQ entry point: selective + AUROC
├── run_part3_calibration_audit.py    # UQ entry point: conformal + calibration
├── run_part4_t7_crosscheck.py        # UQ entry point: T7 cross-check
│
├── scripts/                          # Source code (CLEAN — no stray files)
│   ├── data_preprocessing/           #   Data pipeline
│   ├── evaluation/                   #   UQ evaluation scripts
│   ├── gnn/                          #   Model architecture + helpers
│   ├── misc/                         #   Utility scripts
│   └── training/                     #   Training scripts
│
├── data/                             # Data + trained models
│   ├── TR-C_Benchmarks/              #   8 trial directories with models + results
│   ├── train_data/                   #   Graph data batches
│   └── visualisation/                #   Network visualization data
│
├── thesis/                           # Thesis (CLEAN — no loose files)
│   └── latex_tum_official/           #   LaTeX source + figures + PDF
│
├── docs/                             # Documentation (CLEAN — no deprecated files)
│   ├── verified/                     #   Ground-truth results + audit docs
│   ├── visuals/                      #   Generated doc visuals
│   ├── ml_surrogates_.../            #   Project docs
│   ├── COMPLETE_RESEARCH_SUMMARY.md
│   ├── COMPLETE_VERIFICATION_REPORT.md
│   ├── ENSEMBLE_UQ_*.md
│   ├── THESIS_EXPLANATION_COMPLETE.md
│   ├── THESIS_COMPLETE_SUMMARY_URDU.md
│   ├── data_preprocessing.md
│   ├── gnn.md
│   └── training.md
│
├── _archive/                         # Archived (excluded from zip)
│   ├── archived_old_figures/         #   (from prior cleanup)
│   ├── epoch_checkpoints/            #   (empty checkpoint dirs)
│   ├── orphan_figures/
│   ├── per_graph_npz/
│   ├── root_pdfs/
│   ├── wandb/
│   ├── thesis_draft_figures/         #   NEW: 31 files from thesis/
│   ├── archived_old_scripts/         #   NEW: 11 files from thesis/ARCHIVED_OLD_SCRIPTS/
│   ├── stray_outputs/                #   NEW: 2 PNGs from scripts/
│   ├── docs_deprecated/              #   NEW: MEETING_PREPARATION.md
│   ├── evaluation_scripts_legacy/    #   NEW: 10 files from evaluation_scripts/
│   └── docs_dashboards/              #   NEW (optional): 4 HTML dashboards
│
└── .github/
    └── copilot-instructions.md
```

---

## Execution Order

1. Create `_archive/thesis_draft_figures/` and move 31 thesis/ loose files
2. Move `thesis/ARCHIVED_OLD_SCRIPTS/` → `_archive/archived_old_scripts/`
3. Create `_archive/stray_outputs/` and move 2 scripts/ PNGs
4. Move `docs/MEETING_PREPARATION.md` → `_archive/docs_deprecated/`
5. Move `evaluation_scripts/` → `_archive/evaluation_scripts_legacy/`
6. (Optional) Move 4 HTML dashboards → `_archive/docs_dashboards/`
7. Rebuild `thesis_upload.zip` (since `_archive/` is excluded, zip gets smaller)
8. Verify LaTeX still compiles (no moved file was referenced)
9. Verify runner scripts still pass syntax check

---

## Risk Assessment

**ZERO RISK** — every move targets files that are:
- Not referenced by any `\includegraphics`, `import`, or `open()` call
- Not referenced by any runner script
- Already marked as archived/deprecated, or are stray outputs

The `_archive/` directory is already excluded from `thesis_upload.zip`, so these files won't confuse an examiner.
