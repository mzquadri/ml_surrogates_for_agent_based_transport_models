"""Replace em-dash punctuation across all chapter and page .tex files.

Strategy:
  - Both literal Unicode em-dash '—' and the LaTeX `---` are flattened to ", " or " (parenthetical) " or "; " or ". " depending on context.
  - We use a simple deterministic rule: replace `---` (or '—') with ", " when it
    sits between two clauses, with ": " when it introduces a definition/list, and
    leave "T1--T8" / "0.5--0.7" / page ranges untouched (those use `--`, en-dash).
  - For simplicity here we apply: `---` -> ", "  (most common case in this thesis).
    Any pair of ", " around a parenthetical aside still reads naturally.
  - Curly quotes are converted to straight ASCII quotes.
"""
import re, glob, os

DOC = "C:/Users/zamin/Downloads/ml_surrogates_thesis_final/ml_surrogates_thesis_final/document"

EM_LATEX = "---"
EM_UNI   = "—"
LDQ, RDQ = "“", "”"
LSQ, RSQ = "‘", "’"

total_emdash = 0
total_curly  = 0
files_changed = []

for f in sorted(glob.glob(f"{DOC}/chapters/*.tex") + glob.glob(f"{DOC}/pages/*.tex")):
    s = open(f, encoding="utf-8").read()
    orig = s

    # Count BEFORE replacement
    n_em_latex = s.count(EM_LATEX)
    n_em_uni   = s.count(EM_UNI)
    n_ldq = s.count(LDQ); n_rdq = s.count(RDQ)
    n_lsq = s.count(LSQ); n_rsq = s.count(RSQ)

    # Replace em-dashes with comma + space (most natural default for this thesis)
    s = s.replace(EM_LATEX, ", ")
    s = s.replace(EM_UNI,    ", ")

    # Tidy double spaces / leading-comma artefacts
    s = re.sub(r"  +", " ", s)
    s = re.sub(r" +,", ",", s)
    s = re.sub(r",,", ",", s)
    s = re.sub(r":,", ":", s)
    # ", . " -> ". "
    s = re.sub(r", \. ", ". ", s)
    s = re.sub(r", \. ", ". ", s)

    # Curly quotes -> straight
    s = s.replace(LDQ, '"').replace(RDQ, '"')
    s = s.replace(LSQ, "'").replace(RSQ, "'")

    if s != orig:
        open(f, "w", encoding="utf-8").write(s)
        files_changed.append(os.path.basename(f))
        total_emdash += n_em_latex + n_em_uni
        total_curly  += n_ldq + n_rdq + n_lsq + n_rsq
        print(f"  {os.path.basename(f):40s}  em-dashes: {n_em_latex + n_em_uni},  curly quotes: {n_ldq+n_rdq+n_lsq+n_rsq}")

print()
print(f"Total em-dash punctuation removed: {total_emdash}")
print(f"Total curly quotes flattened:     {total_curly}")
print(f"Files changed: {len(files_changed)}")
