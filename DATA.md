# Dataset

The artifacts backing this thesis total **7.8 GB across 1,307 files** — too large to track
in git, since GitHub rejects any single file over 100 MB. They are published as
**[Releases on this repository](../../releases)**, in the same directory layout the
training scripts wrote. There is no companion repository: this one is the single
canonical home for the code, the thesis and the data.

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

`data/` is gitignored, so the tree is rebuilt from the releases. Three downloads give you
everything the evaluation scripts read.

**1. The per-trial tree** — every scaler, checkpoint, metric file, prediction archive and
plot, in the original layout:

```bash
gh release download benchmarks-v1 --repo mzquadri/ml_surrogates_for_agent_based_transport_models --dir /tmp/bench
mkdir -p data/TR-C_Benchmarks && for t in /tmp/bench/*.tar; do tar -xf "$t" -C data/TR-C_Benchmarks; done
```

**2. The oversized files** — fifteen `test_dl.pt` test dataloaders at 148–296 MB each,
`feature_data.npz`, `experiment_a_fixed_data.npz`, and the 200 MB Trial 8 ablation CSV.
These fill the `data_created_during_training/` directories the tars leave empty. Asset
names encode the destination path with `__` in place of `/`, because release assets are a
flat list and fifteen of them are called `test_dl.pt`:

```bash
gh release download thesis-data-v1   --repo mzquadri/ml_surrogates_for_agent_based_transport_models --dir /tmp/large
python scripts/restore_large_files.py /tmp/large
```

`thesis-data-v1` also carries `ml-surrogates-thesis-data-tracked-tree.tar`, the complete
tree as a single archive, if you would rather unpack it in one step than assemble it from
the per-trial tars.

**The twenty training batches** (2.44 GiB) stay on this repository's release, where they
were first published:

```bash
gh release download train-data-v1 \
  --repo mzquadri/ml_surrogates_for_agent_based_transport_models \
  --pattern 'datalist_batch_*.pt' \
  --dir data/train_data/dist_not_connected_10k_1pct
```

`visualisation/districts_paris.geojson` (212 KB) is tracked in this repository directly,
so it needs no download.

### Also available as tar archives

[`benchmarks-v1`](../../releases/tag/benchmarks-v1) on this repository holds the same
`TR-C_Benchmarks` content as 23 tars, one per trial, plus `feature_data.npz`. It predates
the data repository and is kept so existing links keep working. Prefer the clone above —
it gives you the directory structure without unpacking anything, and lets you browse it on
GitHub first.

```bash
gh release download benchmarks-v1 \
  --repo mzquadri/ml_surrogates_for_agent_based_transport_models --dir /tmp/benchmarks
for f in /tmp/benchmarks/*.tar; do tar -xf "$f" -C data/TR-C_Benchmarks/; done
```

## Verifying a download

`data/MANIFEST.sha256` lists a SHA-256 for every one of the 1,307 files:

```bash
cd data && sha256sum -c MANIFEST.sha256
```

Every line should report `OK`. The manifest uses paths relative to `data/`, so run the
check from inside that directory. The data repository carries its own manifest over the
1,267 files it tracks, and `SHA256SUMS.txt` on `large-files-v1` covers the nineteen
restored files.

## Why it is not tracked here

Thirty-nine of these files exceed GitHub's 100 MB hard limit — 6.5 GB in total, the
largest being the 297 MB test dataloaders. Committing them would fail outright, and Git
LFS would need a paid data pack while making every clone of this repository
multi-gigabyte.

Splitting the data into its own repository keeps this one small enough to clone quickly
while leaving the artifacts browsable at their real paths, rather than sealed inside
archives. The layout there follows the convention in the upstream
[`docs/training.md`](https://github.com/enatterer/ml_surrogates_for_agent_based_transport_models/blob/main/docs/training.md):
`data_created_during_training/` for the split scalers, the test set and loader parameters,
and `trained_model/` for `model.pth`.
