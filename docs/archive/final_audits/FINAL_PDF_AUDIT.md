# Final PDF Audit — Blank Page Analysis

**Thesis:** Uncertainty Quantification for Graph Neural Network Surrogates of Agent-Based Transport Models
**Author:** Mohd Zamin Quadri
**Date of audit:** March 2026 (post-revision)

---

## 1. Document Class and Chapter Opening Behaviour

| Check | Evidence | Result |
|---|---|---|
| Document class | `\documentclass[...]{scrbook}` in `main.tex` | `scrbook` used |
| `openany` option | `\documentclass[openany,...]{scrbook}` | Chapters do NOT force right-page (odd-page) openings |
| Consequence | No blank left-hand pages inserted before chapter starts | No blank pages from chapter breaks |

With `openany`, `scrbook` places each new chapter on the very next available page, whether odd or even. This eliminates the primary source of blank pages in double-sided book-class documents.

---

## 2. `\cleardoublepage` Override

| Check | Evidence | Result |
|---|---|---|
| Override present | `\let\cleardoublepage\clearpage` in `main.tex` preamble | All `\cleardoublepage` calls silently become `\clearpage` |
| Scope | Affects front matter (ToC, list of figures, abstracts) | No blank pages before or after front matter sections |

This override is the standard TUM thesis template mechanism for suppressing blank pages in the front matter when `scrbook` is used without `openany` — here it is redundant but harmless.

---

## 3. Manual Page Break Commands in Chapter Files

Searched all chapter files (`01_introduction.tex` through `07_conclusion.tex`) for `\clearpage`, `\newpage`, and `\cleardoublepage`:

| File | `\clearpage` | `\newpage` | `\cleardoublepage` |
|---|---|---|---|
| 01_introduction.tex | 0 | 0 | 0 |
| 02_background.tex | 0 | 0 | 0 |
| 03_methodology.tex | 0 | 0 | 0 |
| 04_experiments.tex | 0 | 0 | 0 |
| 05_results.tex | 0 | 0 | 0 |
| 06_discussion.tex | 0 | 0 | 0 |
| 07_conclusion.tex | 0 | 0 | 0 |

**No manual page breaks found in any chapter file.** No risk of forced blank pages within chapters.

---

## 4. Float Placement and White-Space Risk

All figures in all chapters use `[htbp]` placement specifiers. LaTeX's float algorithm will attempt:
1. **h** — placement at the current position
2. **t** — top of the current page
3. **b** — bottom of the current page
4. **p** — a dedicated float page only as last resort

With `[htbp]` and the default `\floatpagefraction` (0.5), a float page will only be created if a figure is too large to fit on any non-float page. No figure in this thesis is large enough to trigger this consistently.

**Revision-specific improvements:**
- New §5.5 Coverage Calibration Analysis (Task 6) adds a full-page table and two paragraphs of prose to Chapter 5, reducing the ratio of floats to text and lowering the risk of float-only pages in the results chapter.
- New architecture table in §3.2.2 (Task 3) adds dense tabular content to Chapter 3, similarly improving float/text balance.

---

## 5. New Content Added This Revision (summary for audit)

| Task | File | Change |
|---|---|---|
| T1+2 | `01_introduction.tex` | `fig_network_intro.pdf` replaces legacy PNG; caption updated |
| T3 | `03_methodology.tex` | §3.2 rewritten; verified architecture table added |
| T4 | `04_experiments.tex` | T1 dropout footnote added to Table 4.1 |
| T5 | `05_results.tex` | Pearson r column added to Table 5.1 |
| T6 | `05_results.tex` | New §5.5 Coverage Calibration Analysis inserted |
| T7 | `06_discussion.tex` | Spearman ρ misinterpretation corrected |
| T8 | `06_discussion.tex` | T1 vs T2–T8 GATConv hypothesis expanded |
| T9 | `02_background.tex` | §2.4 Related Work expanded (all 3 subsections) |
| T10 | `02_background.tex` | §2.2.4 line graph paragraph added |

None of these changes introduce `\clearpage`, `\newpage`, or new float-only environments.

---

## 6. Conclusion

**Expected blank pages in compiled PDF: 0**

All structural safeguards are in place:
- `openany` class option eliminates chapter-break blank pages
- `\let\cleardoublepage\clearpage` eliminates front-matter blank pages
- No manual page breaks in any chapter file
- `[htbp]` float placement minimizes float-only pages
- Revision adds prose content that improves float/text balance
