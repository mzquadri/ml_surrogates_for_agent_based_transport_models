# Phase 8b: Bibliography Audit Report

**Date**: 2026-03-26
**Scope**: `bibliography.bib` — cross-check with 48 cited keys, entry quality, consistency
**Verdict**: PASS with advisory findings (0 missing citations, 0 duplicates, some formatting inconsistencies)

---

## 1. Citation Key Cross-Check

| Check | Result |
|-------|--------|
| Keys cited in .tex but missing from .bib | **0** (PASS) |
| Keys in .bib but never cited in .tex | **0** (PASS) |
| Total unique citation keys | **48** |
| Total bib entries | **48** |
| **Perfect 1:1 match** | **YES** |

No "undefined citation" LaTeX warnings expected. No orphaned bib entries.

---

## 2. Entry Type Distribution

| Type | Count |
|------|-------|
| `@article` | 20 |
| `@inproceedings` | 23 |
| `@book` | 4 |
| `@phdthesis` | 1 |
| **Total** | **48** |

All entry types are appropriate for their referenced works.

---

## 3. Issues Found

### HIGH Severity

| # | Entry Key | Issue | Description |
|---|-----------|-------|-------------|
| 1 | `wang2023uncertainty` | `and others` truncation | Only 1 named author: `Wang, Qingyi and others`. Standard practice is at least 2-3 named authors before truncation. |

### MEDIUM Severity

| # | Entry Key | Issue | Description |
|---|-----------|-------|-------------|
| 2 | `hasanzadeh2020bayesian` | Title casing | Bare `Bayesian` without `{B}ayesian` braces. 4 other entries correctly use `{B}ayesian`. |
| 3 | `zhang2019bayesian` | Title casing | Same: bare `Bayesian` without protective braces. |
| 4 | `li2018diffusion` | Booktitle format | `International Conference on Learning Representations (ICLR)` — missing "Proceedings of" prefix used in other ICLR entries. |
| 5 | `kingma2015adam` | Booktitle format | Same inconsistency as `li2018diffusion`. |

### LOW Severity

| # | Entry Key | Issue | Description |
|---|-----------|-------|-------------|
| 6 | `fuchsgruber2024energy` | Booktitle format | Includes edition "37" in NeurIPS booktitle; 5 other NeurIPS entries omit edition number. |
| 7 | `naeini2015obtaining` | Booktitle format | Includes "29th" in AAAI booktitle; `zhang2019bayesian` omits edition number for AAAI. |
| 8 | `gawlikowski2023survey` | `and others` | Only 2 named authors before truncation (borderline low). |
| 9 | Multiple entries | Missing pages | ~8 `@inproceedings` entries lack page numbers. Optional but recommended. |

### INFO

| # | Note |
|---|------|
| 10 | `gal2016dropout` and `gal2016thesis` are distinct works (ICML paper vs PhD thesis) — NOT duplicates. |
| 11 | 5 entries total use `and others`: `abdar2021review` (3 named), `gawlikowski2023survey` (2), `zhao2020tgcn` (3), `paszke2019pytorch` (3), `wang2023uncertainty` (1). |
| 12 | `wu2020comprehensive` year=2020 may refer to online-first; TNNLS Vol 32(1) was published Jan 2021. Most citations use 2020. |

---

## 4. Duplicate Check

**0 duplicates found.** All 48 entries are distinct works.

---

## 5. Required Fields Check

All entries have their required fields:
- `@article`: title, author, journal, year — all present (volume/pages present for most)
- `@inproceedings`: title, author, booktitle, year — all present
- `@book`: title, author/editor, publisher, year — all present
- `@phdthesis`: title, author, school, year — all present

---

## 6. Recommendations for Phase 8d/8e

| Priority | Action | Entry |
|----------|--------|-------|
| HIGH | Add more author names to `wang2023uncertainty` (at least 2-3 before `and others`) | `wang2023uncertainty` |
| MEDIUM | Add braces: `Bayesian` -> `{B}ayesian` in title | `hasanzadeh2020bayesian`, `zhang2019bayesian` |
| MEDIUM | Standardize ICLR booktitle format (add "Proceedings of" prefix or remove from all) | `li2018diffusion`, `kingma2015adam` |
| LOW | Remove edition "37" from NeurIPS booktitle for consistency | `fuchsgruber2024energy` |
| LOW | Standardize AAAI edition numbering | `naeini2015obtaining` or `zhang2019bayesian` |

---

## 7. Overall Assessment

The bibliography is **functionally complete and correct**:
- All 48 cited works are present
- No missing or orphaned entries
- All entry types are appropriate
- No duplicates
- All required fields present

The issues found are **formatting/editorial** (not correctness), affecting rendered bibliography appearance. The only HIGH item (`wang2023uncertainty` author truncation) should be fixed before final submission.

---

*Report generated as part of Phase 8 thesis verification audit.*
