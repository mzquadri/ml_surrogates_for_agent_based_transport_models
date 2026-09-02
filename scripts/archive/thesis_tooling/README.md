# Thesis tooling (archived)

One-off scripts written while preparing the thesis document, recovered from a working
folder during the cleanup on 2 Sep 2026. They are not part of the analysis pipeline and
nothing imports them.

| Script | What it does |
| ------ | ------------ |
| `bib_check.py`, `bib_check2.py`, `bib_check3.py` | Successive passes at reconciling `bibliography.bib` against the citations in the `.tex` sources |
| `c5.py` | A shorter bibliography check over the same files |
| `rebuild_zip.py` | Rebuilds the submission ZIP from `thesis/latex_tum_official/`, including everything needed to recompile plus the final PDF |
| `run_models_variant_2026-03.py` | A March 2026 variant of `scripts/training/run_models.py`, differing by about 360 lines. Kept for provenance; the maintained version is the one under `scripts/training/` |

## Changed on the way in

All five bibliography and packaging scripts carried an absolute path to one developer
machine (`.../OneDrive/Desktop/...`). Those are now derived from the script's own location,
which both keeps a private path out of a public repository and makes the scripts run from
any checkout. Verified: each still compiles, and `rebuild_zip.py` resolves to the real
`thesis/latex_tum_official/`.

`run_models_variant_2026-03.py` arrived as `Hashirbhaichanges.py.py` with a stray `0 `
prefix on the first line, which made it invalid Python. The prefix is removed and the file
now parses. It is named for its date rather than for the person who touched it.
