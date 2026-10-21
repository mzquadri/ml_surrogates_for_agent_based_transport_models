<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/header-dark.png">
  <img alt="M.Sc. thesis, TUM 2026, Uncertainty for a traffic surrogate: a graph network predicts how Paris traffic shifts when a lane closes, in milliseconds instead of hours, with a line chart showing calibration error falling from 0.269 on raw sigma to 0.048 after one scalar, 82% less calibration error" src="docs/assets/header-light.png">
</picture>

# Uncertainty Quantification for Machine Learning Models in Transportation Policy Analysis

A graph neural network trained on MATSim output predicts how Paris traffic shifts under a
road-capacity policy in milliseconds, where the simulator needs hours, but it returns a
bare number, with nothing to say which of the 31,635 road links it is guessing at. This
repository adds a post-hoc uncertainty layer on top of that already-trained surrogate, MC
Dropout, temperature scaling and conformal prediction, none of which touch the weights,
and then measures how far the resulting sigma can actually be trusted. It is for transport
modellers and ML practitioners who have a working surrogate and need to know when not to
believe it.

![What the thesis is about: MATSim simulation to graph representation to GNN surrogate to prediction, and the post-hoc uncertainty layer added on top](docs/diagrams/01_research_problem.svg)

## The result

Every number here is recomputed from its source artifact by
[`scripts/verify_headline_results.py`](scripts/verify_headline_results.py), which exits
non-zero on drift.

| What was measured | Value |
| --- | --- |
| Calibration error (ECE), raw sigma to one fitted scalar | **0.269 to 0.048**, a **-82%** change (T = 2.702) |
| Error detection, AUROC at the top-10% / top-20% errors | **0.7585** / 0.7401 [^auroc] |
| Does sigma rank errors? Spearman rho against absolute error | **0.482** |
| Selective prediction, MAE change at 50% of links retained | **-41.2%** |
| Split conformal coverage, 90% / 95% nominal | 90.17% / 95.09% [^prot] |
| Conditional coverage across sigma deciles, standard to adaptive conformal | **59.0%-98.1% to 83.7%-96.4%** [^cond] |
| The surrogate it sits on (Trial 8), R2 / MAE / RMSE | 0.5957 / 3.96 / 7.12 veh/h |

Two findings, and they point in different directions. **sigma ranks errors well enough to act
on**: rho = 0.482, error rises monotonically across sigma deciles, and handing off the least
reliable half of the links cuts MAE by 41%. **Raw sigma is not a usable scale**: it covers only
48.6% of errors at the nominal 90% level, so it has to be calibrated before it can be read
as an interval at all. One scalar does most of that work.

A third finding sits underneath both, and it is the one a marginal number hides. **Coverage
that is right on average is not right anywhere in particular.** A single constant conformal
width lands at 88.4% overall while running from 59.0% on the links the model is least sure
about to 98.1% on the ones it is most sure about: too narrow exactly where it matters, and
wastefully wide where it does not. Scaling the width by sigma pulls that band to 83.7%-96.4%
and moves the global figure from 88.4% to 89.9%, nearer the 90% it targets. Reporting only
the marginal number would have shown none of this.

The target the model predicts is the *change* in traffic flow,
`vol_car - vol_base_case`, the flow under the policy minus the flow in the base case. That
is why a prediction goes negative: a capacity reduction pushes cars off the link it touches.

**The PointNetTransfGAT surrogate, the MATSim preprocessing and the base training pipeline
are upstream work by Elena Natterer et al.; the uncertainty layer on top of them is mine.**
Full split in [What is mine and what is inherited](#what-is-mine-and-what-is-inherited).

## Run it

The repository pins **Python 3.10**, `environment-minimal.yml` declares `python=3.10` and
the CI workflow installs the same. There is no `pyproject.toml`; nothing here is
pip-installable, and conda is the supported path.

```bash
git clone https://github.com/mzquadri/ml_surrogates_for_agent_based_transport_models.git
cd ml_surrogates_for_agent_based_transport_models

conda env create -f environment-minimal.yml      # python=3.10, analysis deps only
conda activate traffic-gnn

python scripts/verify_headline_results.py        # recompute every headline number
```

That runs against tracked artifacts and needs no downloads. Two of the thirteen checks,
the AUROC pair, report `SKIP` until the 209 MB ablation CSV is fetched:

```bash
gh release download thesis-data-v1 \
  --repo mzquadri/ml_surrogates_for_agent_based_transport_models \
  --pattern '*trial8_uq_ablation_results.csv' --dir /tmp/large
python scripts/restore_large_files.py /tmp/large
```

The rest of what CI runs:

```bash
python scripts/check_repository.py      # 16 required artifacts exist and compile
python scripts/check_docs.py            # documentation links and SVG validity
python scripts/check_web_assets.py      # web assets parse and match their schema
python -m pytest -q                     # 13 of 15 tests; 2 need Streamlit
```

For the full training environment, including PyTorch Geometric and CUDA, use
`traffic-gnn.yml` instead. Training itself runs on a SLURM cluster via `run_models.sbatch`
and is not reproducible from a clone, see [Reproducing the analyses](#reproducing-the-analyses).

---

## The thesis

**M.Sc. thesis**, Technical University of Munich, School of Computation, Information and Technology

|               |                                                   |
| ------------- | ------------------------------------------------- |
| **Author**    | Mohd Zamin Quadri                                  |
| **Programme** | M.Sc. Mathematics in Science and Engineering       |
| **Supervisor**| Prof. Dr. Stephan Günnemann                        |
| **Advisors**  | Dominik Fuchsgruber, M.Sc., Elena Natterer, M.Sc. |
| **Submitted** | 15 May 2026                                        |

**[Read the thesis (PDF)](thesis/submission_2026-05-15/)**, **[Full results](#results-in-detail)**, **[Data story](docs/portfolio_data_story/)**, **[Corrigendum](docs/CORRIGENDUM.md)**

![The Paris network](docs/figures/portfolio/01_the_network.png)

---

## The problem

Agent-based transport simulators such as MATSim answer policy questions well but slowly,
hours per scenario. A graph neural network trained on their output answers the same
question in seconds, which is what makes large policy sweeps feasible at all.

The catch is that the surrogate returns a bare number. Nothing in it says which of the
31,635 road links it is confident about and which it is guessing at. For a model whose
output is meant to inform a decision about closing a road, that gap matters more than
another point of R2.

**This thesis asks: can a useful, calibrated uncertainty estimate be attached to an
already-trained GNN surrogate, without retraining it?**

---

## What is mine and what is inherited

The PointNetTransfGAT surrogate, the MATSim preprocessing and the base training pipeline
are upstream work by Elena Natterer et al.; the uncertainty layer on top of them is mine.

| Origin | What it covers |
| --- | --- |
| **Inherited** (upstream) | The GNN architectures (`scripts/gnn/`), the MATSim-to-graph preprocessing (`scripts/data_preprocessing/`), the base training pipeline (`scripts/training/`), `help_functions.py` / `plot_functions.py`, and `traffic-gnn.yml`. |
| **Mine** (this thesis) | The entire uncertainty layer, MC Dropout evaluation, temperature scaling, split and adaptive conformal prediction, CQR, deep ensembles, heteroscedastic regression, selective prediction, error detection, the calibration audits, every result artifact under `results/`, the thesis document, and the verification tooling. |

Upstream is **[`enatterer/ml_surrogates_for_agent_based_transport_models`](https://github.com/enatterer/ml_surrogates_for_agent_based_transport_models)**
by **Elena Natterer**, with contributions from **Saini Rohan Rao** and **Thua Duc Nguyen**.
Their full commit history is preserved unmodified on the [`zamin_uq`](../../tree/zamin_uq)
branch, 266 commits, none of them mine. The `main` branch carries this thesis's work.

The trained surrogate itself is taken as given from that work. This thesis does not claim
the architecture, the preprocessing, or the base training procedure.

> Natterer et al. (2025). *Machine Learning Surrogates for Agent-Based Models in
> Transportation Policy Analysis.* Transportation Research Part C, 180, 105360.

---

## The experiment: Paris, by arrondissement

1,000 MATSim scenarios of the Paris Île-de-France network at 1% population sampling. Each
scenario is the **same** road network under a different capacity-reduction policy, and the
preprocessing names each one after the districts its policy touches,
`create_policy_key()` returns *"Policy introduced in Arrondissement(s) 5, 12"*. **The
arrondissement is the unit the experiment was designed around.**

![Paris by arrondissement](docs/figures/portfolio/07_arrondissements.png)

88.4% of links fall inside the twenty arrondissements; the rest extend into the wider
Île-de-France. The policy only ever touches three OSM road classes, primary, secondary and
tertiary. Motorways are **never** intervened in any of the 1,000 scenarios.

![Arrondissement choropleths](docs/figures/geography/02_arrondissement_choropleths.png)

Intervention rates span 25.9% (18th) to 48.1% (16th) and severity 190 to 472 veh/h (8th),
but **local severity does not determine local response**. The 16th is intervened most and
responds least; the 17th absorbs **12.0% of the network's total response across 6.9% of its
links**, the 14th 10.3% across 5.3%.

---

## The dataset: 11 stored fields

Each scenario is one PyTorch Geometric `Data` object holding **six feature columns in `x`
plus five further tensors, eleven stored fields in total**.

![Feature representation](docs/diagrams/03_feature_representation.svg)

| Field | Shape | Dynamic? | Model input |
| --- | --- | :-: | :-: |
| `x[:, 0]` `VOL_BASE_CASE` | `[31635, 6]` | no | ✅ |
| `x[:, 1]` `CAPACITY_BASE_CASE` | ″ | no | ✅ |
| `x[:, 2]` **`CAPACITY_REDUCTION`** | ″ | **yes** | ✅ |
| `x[:, 3]` `FREESPEED` | ″ | no | ✅ |
| `x[:, 4]` `HIGHWAY` | ″ | no | ❌ nominal codes |
| `x[:, 5]` `LENGTH` | ″ | no | ✅ |
| `pos`, start / end / midpoint, WGS84 | `[31635, 3, 2]` | no | ✅ |
| `y`, change in link volume, `vol_car - vol_base_case` | `[31635, 1]` | **yes** | target |
| `edge_index` | `[2, 59851]` | no | ✅ |
| `mode_stats_diff` | `[6, 3]` | yes | ❌ unused |
| `mode_stats_diff_perc` | `[6, 3]` | yes | ❌ unused |

**One column carries the whole experiment.** Five of six features, the positions and the
topology are byte-identical in every scenario; only `CAPACITY_REDUCTION` moves.

The `EdgeFeatures` enum also declares `ALLOWED_MODE_CAR` ... `ALLOWED_MODE_SUBWAY` at indices
6 to 11, but `use_allowed_modes = False` in the preprocessing, so those six columns were
**designed and never materialised**. `x` has six columns, not twelve.

![Field reference](docs/figures/portfolio/10_field_reference.png)

![The five model features](docs/figures/portfolio/09_five_model_features.png)

The same panel layout the thesis used, rebuilt over the whole corpus rather than a
200-graph subset. Every one of the eleven stored fields has its own card,
distribution, spatial map and relationship with the response, in
[`docs/figures/features/`](docs/figures/features/README.md).

![The busiest roads are the steady ones](docs/figures/portfolio/03_inverted_u.png)

A separate check on the stored schema found that `num_nodes` is recorded as 31,559 while
`x`, `pos` and `y` all carry 31,635 rows. The 76 extra rows are public-transport links with
no car access, no edges, and a target of exactly zero in every scenario, see
[CORRIGENDUM C11](docs/CORRIGENDUM.md).

Per-field detail: [`docs/DATASET.md`](docs/DATASET.md),
[`02_features.md`](docs/portfolio_data_story/02_features.md), and the full corpus
exploration in [`docs/data_exploration/`](docs/data_exploration/README.md).

---

## Graph representation

The network is inverted before the model sees it: **a road link is a node**, and an edge
means two links meet. 31,635 nodes, 59,851 directed edges, maximum degree 10.

![Graph lifting](docs/diagrams_isometric/iso_01_graph_lifting.svg)

Both planes hold the same real links, drawn from `pos`. Directedness, degree distribution
and the 121 components:
[`03_graph_topology.md`](docs/portfolio_data_story/03_graph_topology.md) and
[`graph_topology.md`](docs/data_exploration/graph_topology.md).

The model reads five of the six `x` columns as node features, and also consumes
`pos[:, 0]`, `pos[:, 1]` and `edge_index`. `HIGHWAY` is excluded because its integers are
nominal road-class labels, not quantities.

![What actually enters the model](docs/diagrams_data/data_03_model_inputs.svg)

Traced to the code and confirmed against the trained checkpoint:
[`model_inputs.md`](docs/data_exploration/model_inputs.md).

---

## Model

`PointNetTransfGAT`, two PointNet convolutions folding in link geometry, two
graph-transformer layers, two attention layers reducing to one value per link.
**1,416,835 parameters**, read from the Trial 8 checkpoint rather than code defaults.

![Model architecture](docs/figures/portfolio/08_model_architecture.png)

Every shape and count above is a tensor in the Trial 8 checkpoint. There is a card
for each of the eleven stored fields, the six columns of `x` and the five other
tensors, in [`docs/figures/features/`](docs/figures/features/README.md).

The two dropout layers are the only stochastic elements, which is exactly what makes
post-hoc MC Dropout possible without touching the weights.

---

## Experiment progression

Sixteen checkpoints across eleven trials plus a five-seed ensemble.

| Stage | Change | Outcome |
| --- | --- | --- |
| **T1-T6** | output head, weighted loss, learning rate | T1 R2 0.786 (Linear head, no dropout); weighted loss failed twice (0.225, 0.243); T5 recovered to 0.555 |
| **T7-T8** | 80/10/10 split, then dropout 0.3 to **0.2** | **T8: R2 0.5957, MAE 3.96**, the UQ baseline |
| **T9** | freeze T8, add heteroscedastic head | val NLL 3.249; no test metrics recorded |
| **T10** | full CQR retrain, 87 hours | **gate FAIL**, midpoint R2 0.406, PICP95 91.8% |
| **T11** | freeze T8, quantile head only, 40 hours | **gate PASS**, midpoint R2 0.584, PICP95 94.9% |
| **Ensemble** | 5 seeds, dropout off at inference | R2 **0.684**, best accuracy, but rho 0.400 |

> **⚠ Trials were not all scored on the same test split.** T1-T6 used **50 test graphs**
> (1,581,750 nodes); T7-T11 and the ensembles used **100** (3,163,500). R2 is not comparable
> across that boundary, see [CORRIGENDUM C9](docs/CORRIGENDUM.md). All uncertainty work
> uses Trial 8 alone, so no UQ result rests on a cross-split comparison.

**Freezing beat retraining.** T10 retrained everything for 87 hours and failed its own
acceptance gates, losing most of the backbone's accuracy; T11 froze the backbone, trained
one quantile head for 40 hours, passed every gate and kept it. Same method, opposite
outcome, decided by what was allowed to move.

**The ensemble bought accuracy and lost uncertainty quality.** Most accurate model here at
R2 0.684, yet its sigma ranks errors *worse* than MC Dropout, rho 0.400 against 0.482, a 17.1%
drop recorded in its own comparison block. Better predictions did not mean better
uncertainty.

Full inventory: [`08_models_and_experiments.md`](docs/portfolio_data_story/08_models_and_experiments.md).

---

## Uncertainty quantification

Everything here is **post-hoc**: the checkpoint is loaded, frozen, never updated.

![Uncertainty pipeline](docs/diagrams/06_uncertainty_pipeline.svg)

Two questions with different answers. **Does sigma rank errors?** Yes, rho = 0.482, and error
rises monotonically across sigma deciles. **Is sigma a calibrated scale?** No, raw sigma covers only
48.6% of errors at nominal 90%.

![Uncertainty vs error](docs/figures/results/02_uncertainty_vs_error.png)

So sigma is corrected two independent ways: **temperature scaling** (one scalar, keeps an
interpretable per-link sigma) and **split conformal prediction** (exact marginal coverage by
construction, at the cost of one shared width).

![Calibration](docs/figures/results/03_calibration.png)

Four questions, four evaluations, conflating them is the usual way to overstate an
uncertainty result:

![Evaluation framework](docs/diagrams/07_evaluation_framework.svg)

---

## Results in detail

### Trial 8, 100 held-out scenarios, 3,163,500 links

| Metric | Value |
| --- | --- |
| R2 | **0.5957** |
| MAE / RMSE | **3.96 / 7.12** veh/h |

### Uncertainty quality

| Analysis | Trial 8 | Trial 7 |
| --- | --- | --- |
| MC Dropout Spearman rho (sigma vs abs. error) | **0.482** | 0.4437 [^t7] |
| ECE, before to after temperature scaling | 0.269 to 0.048 (T = 2.702) |, |
| Selective prediction, MAE reduction at 50% retained | **-41.2%** | -38.3% |
| Error detection AUROC, top-10% / top-20% | **0.7585 / 0.7401** [^auroc] | 0.7416 /, |
| Split conformal coverage, 90% / 95% | 90.17% / 95.09% [^prot] | 90.18% / 95.11% |

![Selective prediction and conformal coverage](docs/figures/results/04_selective_and_conformal.png)

### Conditional coverage, and why the marginal number is not enough

Marginal coverage is an average over the evaluated split. It can be exactly right while the
guarantee fails on every subgroup a reader would actually ask about. It does here.

Nominal 90%, 10 sigma deciles, 1,581,750 nodes
(`results/phase3/adaptive_conformal_decile.json`):

| Width | Global | Lowest decile | Highest decile | Span |
| --- | --- | --- | --- | --- |
| Standard conformal, one constant width | 88.39% | 59.04% | 98.08% | **39.0 points** |
| Adaptive conformal, width scaled by sigma | 89.89% | 83.69% | 96.44% | **12.8 points** |

The constant width is set by the whole calibration set, so it is too narrow on the links
whose sigma is large, which are the links the model is already least sure about, and too
wide on the links whose sigma is small. The failure is concentrated exactly where a user
would be relying on the interval most. Scaling the width by sigma closes 26 points of that
spread and does not pay for it globally: overall coverage moves from 88.39% to 89.89%,
nearer the 90% it is aiming at, not further from it.

This is a different split from the marginal coverage row in
[The result](#the-result) and the two must not be pooled; see
[CORRIGENDUM C3](docs/CORRIGENDUM.md) on protocols. The same analysis over the
2,530,800-node evaluation set in `results/conformal_conditional_coverage_t8.json` reproduces
the effect: 62.92%-98.57% narrowing to 84.25%-96.24%.

### Where the traffic actually goes

**About 65% of the response lands on links the policy never touched.** Mean |response| is
13.06 veh/h on intervened links against 2.91 elsewhere, but the untouched links are so
numerous they carry two-thirds of the total effect. The graph-distance profile drops 4.4x
at the first hop and is then **flat out to eight hops**, an observed network-wide
redistribution rather than a gradient that fades with distance.

It survives its reachability control (one hop reaches only 32% of the network, three reach
70%) and holds under both directed and undirected traversal. It is an association, not
evidence of a mechanism. See [`05_spillover.md`](docs/portfolio_data_story/05_spillover.md).

![Where the policy lands and where the traffic moves](docs/figures/portfolio/04_policy_vs_response.png)

[^t7]: The thesis reports 0.446 for Trial 7. That value is not reproducible from the
    retained archive, which yields 0.4437 under the definition that reproduces Trial 8
    exactly. See [CORRIGENDUM C7b](docs/CORRIGENDUM.md).

[^auroc]: **Corrected after submission.** The submitted thesis reports a lower pair for
    this metric, citing a file absent from this repository. Recomputation from the cited
    artifact gives 0.7585 and 0.7401. A results figure in this README computes the same
    metric from the tracked NPZ archive and lands slightly lower again, a different
    stochastic MC Dropout replay of the same model, not a different result. Each value is
    correct for the artifact it names; the superseded figures and the full derivation are
    in [CORRIGENDUM C4 and C7a](docs/CORRIGENDUM.md).

[^prot]: Protocol `graph20_80_v1`, calibrate on the first 20 test graphs, evaluate on the
    remaining 80. The thesis reports 90.02% / 95.01% under a 50/50 scenario split whose
    indices were not retained. The two must not be pooled; see
    [CORRIGENDUM C3](docs/CORRIGENDUM.md).

[^cond]: From `results/phase3/adaptive_conformal_decile.json`, recomputed for this table:
    1,581,750 nodes, the last 50 graphs, 10 sigma deciles, nominal 90%. Standard conformal
    (one constant width) spans 59.04% to 98.08% across the deciles at 88.39% globally;
    adaptive conformal (width scaled by sigma, `q_adapt` = 7.709) spans 83.69% to 96.44% at
    89.89% globally. **This is a different split from the marginal coverage row above**,
    which uses `graph20_80_v1` over 80 graphs, so the two rows are not two readings of one
    number and must not be pooled. The same analysis over the 2,530,800-node evaluation set
    in `results/conformal_conditional_coverage_t8.json` gives 62.92%-98.57% narrowing to
    84.25%-96.24%, the same effect at the same size.

### Inference cost

The opening claim of this README is that the simulator needs hours per scenario and the
surrogate answers in milliseconds. That is the reason to build a surrogate at all, and it
was never measured here. It is now, by
[`scripts/benchmark_inference.py`](scripts/benchmark_inference.py).

One forward pass of T8 over the full Paris graph, 31,635 nodes and 59,851 edges, median of
50 timed passes after 10 warm-up passes:

| Device | Hardware | Median | Range |
| --- | --- | --- | --- |
| CPU | Intel Core Ultra 9 285H, 16 cores | 610 ms | 517 - 691 ms |
| GPU | Intel Arc Pro 140T, 50.7 GB, torch 2.11 XPU | **141 ms** | 135 - 151 ms |

The two devices agree to 3.15e-03 on an output spanning roughly 290 vehicles/hour, which is
1.1e-05 of the range and is float32 ordering, not a different computation. The accelerator
is 4.3x faster.

This is a property of the machine it ran on, not a result of the thesis. It is kept out of
[`scripts/verify_headline_results.py`](scripts/verify_headline_results.py), which pins
numbers that have to reproduce on any machine, and it changes no accuracy figure: the
thesis numbers come from the recorded runs and the benchmark only times a forward pass.

Reproducing it needs a scenario file from the `thesis-data-v1` release, so it does not run
in CI. Any scenario will do, because the topology and five of the six feature columns are
byte-identical across scenarios.

### What this does not establish

One Paris network, one capacity-reduction intervention family, a 1,000-scenario subset, one
model family. Conformal coverage is marginal over the evaluated split, not a guarantee for
any individual scenario, link, city, or policy. Checked directly: coverage breaks out by
arrondissement at 80.7%-92.4% against a 90% nominal level, not a flat 90% everywhere, see
[`docs/COVERAGE_BY_ARRONDISSEMENT.md`](docs/COVERAGE_BY_ARRONDISSEMENT.md).

---

## Reproducing the analyses

Beyond the verification run in [Run it](#run-it), the analyses, figures and data
exploration:

```bash
python scripts/evaluation/run_part4_t7_crosscheck.py          # no downloads needed
python scripts/figure_generation/generate_results_figures.py  # no downloads needed
python scripts/data_exploration/explore_checkpoints.py        # all 16 checkpoints
python scripts/data_exploration/explore_arrondissements.py --corpus <dir> --cache <dir>
```

These replay the analyses from cached prediction arrays. They do not retrain the models and
do not rerun MATSim, the raw simulation outputs were not retained.
[`docs/CORRIGENDUM.md`](docs/CORRIGENDUM.md) C5 states the replay boundaries;
[`DATA.md`](DATA.md) covers artifact availability.

---

## The figures

Every figure is generated from the published corpus by a tracked script and rebuilds
byte-identically. Nothing is drawn by hand.

| | What it shows | Where |
|---|---|---|
| **The network** | 31,635 real street segments, shaded by traffic | [gallery](docs/figures/portfolio/README.md) |
| **Five model features** | the (a)-(e) panel figure, full corpus | [gallery](docs/figures/portfolio/README.md) |
| **Field reference** | all eleven fields: shape, dynamic, model input | [gallery](docs/figures/portfolio/README.md) |
| **Eleven field cards** | one card per stored field | [features](docs/figures/features/README.md) |
| **Policy vs response** | where the policy lands against where traffic moves | [gallery](docs/figures/portfolio/README.md) |
| **Model architecture** | read from the Trial 8 checkpoint | [gallery](docs/figures/portfolio/README.md) |
| **Arrondissements** | districts from the GeoJSON, overlaid on the network | [gallery](docs/figures/portfolio/README.md) |
| **Structural diagrams** | problem, features, model, uncertainty, evaluation | [`docs/diagrams/`](docs/diagrams/) |
| **Data anatomy** | `.pt` schema, the eleven fields, why HIGHWAY is excluded | [`docs/diagrams_data/`](docs/diagrams_data/) |

```bash
CORPUS=/path/to/corpus       # gh release download train-data-v1
CACHE=/path/to/scratch
python scripts/data_exploration/explore_tensors.py          --corpus $CORPUS --cache $CACHE
python scripts/figure_generation/generate_portfolio_figures.py  --corpus $CORPUS --cache $CACHE
python scripts/figure_generation/generate_feature_cards.py      --corpus $CORPUS --cache $CACHE
python scripts/figure_generation/generate_five_features_figure.py --corpus $CORPUS --cache $CACHE
python scripts/figure_generation/generate_model_diagram.py
python scripts/figure_generation/generate_field_reference.py
```

Maps draw the real street geometry from `pos[:, 0]` to `pos[:, 1]`, and district
boundaries come from the arrondissement GeoJSON, the same CRS84 longitude and latitude,
so the two overlay without reprojection. Heavy reductions use the Intel Arc GPU through
`torch.xpu` where it helps, which is about fifteen times faster than numpy for this shape.

Three figures are recorded as defective in [`docs/CORRIGENDUM.md`](docs/CORRIGENDUM.md)
C13, `docs/diagrams/calibration.svg`, `docs/diagrams/stratified_uq.svg` and
`thesis/latex_tum_official/figures/new/fig12_sigma_scaling_ece.png`. They are kept as part
of the record and are not embedded anywhere in this README. Do not reuse them.

---

## Repository layout

```
scripts/
  verify_headline_results.py   Recomputes every published number; non-zero exit on drift
  check_docs.py                Link and SVG gate
  benchmark_inference.py       Times one forward pass on CPU and on an XPU/CUDA device
  restore_large_files.py       Rebuilds the data tree from the releases
  data_exploration/            Dataset, arrondissement and checkpoint analysis
  evaluation/  figure_generation/
  gnn/  training/  data_preprocessing/     Upstream model, training and preprocessing
  archive/                     Historical one-offs, provenance only, not runnable
docs/
  DATASET.md  CORRIGENDUM.md  ARTIFACT_PROVENANCE.md
  portfolio_data_story/        Deep corpus analysis + derived web assets
  data_exploration/            Stored fields, model inputs, topology, auxiliary tensors
  diagrams/  diagrams_isometric/  diagrams_data/  figures/
  migration/                   Repository-consolidation record
models/     16 trial checkpoints
results/    Result JSONs, prediction archives, per-trial metrics
thesis/     submission_2026-05-15/ (examined) and latex_tum_official/ (working)
tests/
```

All large artifacts are published as **[Releases on this repository](../../releases)**,
the training corpus, per-trial evaluation outputs, oversized prediction archives, and the
migrated per-trial data. Everything the thesis needs is here or on those releases; there is
no companion repository. See [`DATA.md`](DATA.md).

---

## Licence and attribution

Upstream code is under the **MIT Licence, © 2024 Elena Natterer**, reproduced in
[`LICENSE`](LICENSE). Those terms govern `scripts/gnn/`, `scripts/data_preprocessing/`,
`scripts/training/`, `scripts/evaluation/help_functions.py`,
`scripts/evaluation/plot_functions.py`, the upstream notebooks, and `traffic-gnn.yml`, and
anything derived from them.

Material added by this fork, the thesis text and figures, the trained checkpoints, and the
result artifacts, is **not** covered by that grant. Reuse requires prior permission from
the author and, where applicable, the original data and model owners.

Citation metadata: [`CITATION.cff`](CITATION.cff).

## Tooling

AI coding assistants were used for code review, refactoring, and repository maintenance.
All research design, modelling decisions, analysis, and written content are the author's
own, and every reported number is verified against the artifacts in this repository by a
script anyone can run.
