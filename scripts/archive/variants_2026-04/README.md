# Superseded variants (April 2026)

These files come from the retired `ml-surrogates-thesis` repository and were recovered
during the consolidation of 1 Sep 2026. They are **older snapshots**, kept only so the
earlier state remains inspectable.

**Do not import from this directory.** The live versions are:

| Archived here | Live version |
| ------------- | ------------ |
| `gnn/losses/heteroscedastic_loss.py` | `scripts/gnn/losses/heteroscedastic_loss.py` |
| `gnn/models/point_net_transf_gat_heteroscedastic.py` | `scripts/gnn/models/point_net_transf_gat_heteroscedastic.py` |
| `gnn/help_functions.py` | `scripts/gnn/help_functions.py` |
| `check_repository.py` | `scripts/check_repository.py` |
| `workflows/repository-check.yml` | `.github/workflows/` |

The live copies are dated August 2026 and are the ones `scripts/training/train_heteroscedastic.py`
imports. The archived copies are dated 11 April 2026.

The differences are real, not just line endings — for example the live
`gnn/help_functions.py` adds `mc_dropout_predict_hetero`, which decomposes predictive
uncertainty into aleatoric and epistemic components and is absent from the April copy.

`check_repository.py` is the exception worth noting: the archived version is
substantially longer (11.5 KB vs 1.5 KB) and is a different tool rather than an older
draft of the same one. Nothing imports either.
