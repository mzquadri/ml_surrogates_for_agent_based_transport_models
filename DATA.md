# Dataset

The artifacts backing this thesis total **7.8 GB across 1,307 files** — too large for
GitHub, which rejects any single file over 100 MB. The data is therefore archived
separately and this repository carries only the manifest needed to verify it.

## Contents

| Directory | Files | Size | What it holds |
| --------- | ----: | ---: | ------------- |
| `TR-C_Benchmarks/` | 1,282 | 5.3 GB | Per-trial outputs: test dataloaders, trained weights, prediction archives (`.npz`), metrics, and plots for all 16 model trials |
| `train_data/` | 22 | 2.5 GB | Training corpus `dist_not_connected_10k_1pct` — 20 × `datalist_batch_*.pt` at ~125 MB each |
| `misc/` | 1 | 121 MB | `feature_data.npz`, the pooled feature array used by the analysis scripts |
| `visualisation/` | 1 | 216 KB | `districts_paris.geojson`, the Paris Île-de-France district layer |

By file type: 733 `.npz` prediction archives, 275 `.png` plots, 102 `.pkl` scalers,
80 `.json` metric files, 40 `.pt` dataloaders, and 16 `.pth` model checkpoints.

The underlying simulations are 10,000 MATSim runs over the Paris Île-de-France road
network (31,635 road segments).

## Where to get it

<!-- Replace this block once the Zenodo record is published. -->
> **Not yet published.** Upload the archive to Zenodo (or the TUM data repository),
> then replace this block with the DOI and direct download link, e.g.
>
> ```
> DOI: 10.5281/zenodo.XXXXXXX
> https://doi.org/10.5281/zenodo.XXXXXXX
> ```

Zenodo accepts records up to 50 GB and mints a citable DOI, which is why it suits this
dataset better than Git LFS.

## Verifying a download

`data/MANIFEST.sha256` lists a SHA-256 for every one of the 1,307 files. After placing
the extracted tree at `data/` in the repository root:

```bash
cd data && sha256sum -c MANIFEST.sha256
```

Every line should report `OK`. The manifest is generated with paths relative to `data/`,
so run the check from inside that directory.

## Why it is not in git

`data/` is listed in `.gitignore`. Thirty-nine files exceed GitHub's 100 MB hard limit
(6.5 GB in total), the largest being the 297 MB test dataloaders. Committing them would
fail outright; committing them through Git LFS would require a paid data pack and make
every clone of this repository multi-gigabyte. Keeping the data in an archive with a DOI
is both cheaper and more citable.

The evaluation scripts read from a local `data/` tree, so download and extract the
archive before running anything in `Reproducing Results` in the [README](README.md).
