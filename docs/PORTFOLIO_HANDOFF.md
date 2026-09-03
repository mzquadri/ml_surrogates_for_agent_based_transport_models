# Portfolio handoff — source material

Structured source material for a future portfolio page. No website code, no marketing copy.
Every claim below is traceable to an artifact in this repository and is asserted by
`scripts/verify_headline_results.py`.

---

## Project title

**Uncertainty Quantification for GNN Surrogates of Agent-Based Transport Models**

Short form for cards and navigation: *Uncertainty Quantification for Traffic GNN Surrogates*

## One-sentence summary

A post-hoc uncertainty layer that tells an already-trained graph neural network surrogate of a
Paris traffic simulation which of its 31,635 per-road-link predictions can be trusted.

## Technical summary (3–5 sentences)

Agent-based transport simulators like MATSim answer policy questions accurately but take hours per
scenario, so a graph neural network is trained on their output to answer the same question in
seconds. That surrogate returns a bare number with no indication of where it is wrong, which is a
problem when the output informs a decision about closing a road. This thesis attaches uncertainty
to the trained surrogate without retraining it: MC Dropout over 30 stochastic forward passes
produces a per-link sigma, which is then corrected by temperature scaling and split conformal
prediction. The resulting estimate ranks errors well enough to be operationally useful — handing
off the least reliable half of the links cuts MAE by 41% — while the analysis is explicit that raw
MC Dropout sigma is badly miscalibrated as a scale and must be corrected before it can be read as
an interval.

## My contribution

The uncertainty layer in full: MC Dropout evaluation, temperature scaling, split and adaptive
conformal prediction, selective prediction, error detection, calibration audits, the deep-ensemble
and conformalised-quantile-regression trials, every result artifact, the thesis document, and the
verification tooling that recomputes each published number from its source artifact.

**Not mine, and must be labelled as such on any page:** the GNN architecture, the MATSim-to-graph
preprocessing, and the base training pipeline. Those come from
`enatterer/ml_surrogates_for_agent_based_transport_models` (MIT, © 2024 Elena Natterer), with
contributions from Saini Rohan Rao and Thua Duc Nguyen. The trained surrogate is taken as given.

## Technologies

PyTorch · PyTorch Geometric · MC Dropout · conformal prediction · temperature scaling ·
NumPy / SciPy / scikit-learn · pandas · matplotlib · MATSim (upstream data source) · pytest

## Dataset summary

1,000 MATSim scenarios of the Paris Île-de-France road network at 1% population sampling,
published as a release (2.44 GiB, 20 files). The network is a line graph: each road link is a node,
31,635 nodes and 59,851 edges per scenario, topology byte-identical across all scenarios. Six node
features are stored and five are used; `HIGHWAY` is excluded because it is an ordinally-encoded
nominal category. Only `CAPACITY_REDUCTION` varies between scenarios, so it carries the entire
scenario-discriminating signal. Target is the policy-induced change in link volume in veh/h
(mean 0.42, std 10.71, 27.62% exact zeros). Measured in full in `docs/DATASET.md`.

## Model summary

`PointNetTransfGAT`, 1,416,835 parameters, read from the Trial 8 checkpoint:
two PointNet convolutions folding in link geometry (start and end coordinates), two graph
transformer layers (128 → 4 heads × 64 → 256, then 256 → 4 heads × 128 → 512), a GATConv
512 → 64, and a GATConv 64 → 1 output head. Dropout p = 0.2 sits between the transformer blocks and
is the only stochastic element — which is exactly what makes post-hoc MC Dropout possible without
touching the weights. Trained 80/10/10 by scenario; 16 trials retained.

## Uncertainty-quantification summary

MC Dropout with S = 30 (on the convergence plateau: S = 5 → 30 gains 10.8% in rho, S = 30 → 50
gains 1.0%) produces a per-link sigma. Two questions are asked of it and answered separately.
Does sigma rank errors? Yes. Is sigma a calibrated scale? No — raw sigma covers 48.6% of errors at
the nominal 90% level. Two independent post-hoc corrections are applied: temperature scaling
(one scalar, keeps an interpretable per-node sigma) and split conformal prediction (exact marginal
coverage by construction, at the cost of one shared interval width). Evaluation is deliberately
split into four questions — ranking quality, calibration, coverage, and operational utility —
because conflating them is the usual way to overstate an uncertainty result.

## Strongest verified results

All Trial 8 unless stated; 3,163,500 link-level predictions across 100 held-out scenarios.

| # | Result | Value | Why it matters |
| --- | --- | --- | --- |
| 1 | Selective prediction, MAE reduction at 50% of links retained | **−41.2%** | The headline operational result: hand off the least reliable half, and error on what remains drops by 41%. |
| 2 | Error detection AUROC, top-10% worst-predicted links | **0.7585** | Sigma finds the links most likely to be badly wrong, well above the 0.500 random baseline. |
| 3 | Calibration, ECE before → after temperature scaling | **0.269 → 0.048** | One scalar (T = 2.702) removes most of the miscalibration. |
| 4 | Raw MC Dropout coverage at nominal 90% | **48.6%** | The negative result that motivates the whole calibration step; stated as prominently as the positive ones. |
| 5 | Split conformal coverage, 90% / 95% nominal | **90.17% / 95.09%** | Distribution-free marginal coverage holds empirically (protocol `graph20_80_v1`). |
| 6 | Surrogate accuracy — R² / MAE / RMSE | **0.5957 / 3.96 / 7.12 veh/h** | The baseline the uncertainty layer sits on top of. |

Reproduce all of them: `python scripts/verify_headline_results.py` (13/13 checks).

## Recommended hero image

`docs/figures/dataset/04_spatial_intervention_response.png` — the Paris network with the capacity
intervention on the left and the resulting volume change on the right. It is immediately legible as
a real city, shows the spillover onto untouched links that motivates using a graph model, and needs
no domain knowledge to read.

Alternative hero, if a diagram is preferred over data:
`docs/diagrams/01_research_problem.svg` — the whole argument on one screen.

## Recommended supporting figures

In narrative order:

1. `docs/diagrams/01_research_problem.svg` — problem framing.
2. `docs/diagrams/03_feature_representation.svg` — what a node is, all six features with statistics.
3. `docs/diagrams/04_model_architecture.svg` — architecture, dimensions read from the checkpoint.
4. `docs/figures/results/02_uncertainty_vs_error.png` — sigma ranks error (the core positive claim).
5. `docs/figures/results/03_calibration.png` — raw sigma undercovers; one scalar fixes it.
6. `docs/figures/results/04_selective_and_conformal.png` — the 41% result and conformal coverage.
7. `docs/figures/results/01_accuracy.png` — surrogate accuracy, if a baseline panel is wanted.

## Recommended tags

`uncertainty-quantification` · `graph-neural-networks` · `conformal-prediction` · `mc-dropout` ·
`model-calibration` · `surrogate-models` · `pytorch-geometric` · `transportation` ·
`agent-based-modeling` · `matsim` · `masters-thesis`

## Repository references

| Field | Value |
| --- | --- |
| GitHub | `https://github.com/mzquadri/ml_surrogates_for_agent_based_transport_models` |
| Companion data | `https://github.com/mzquadri/ml-surrogates-thesis-data` |
| Gitea | *(placeholder — not yet published)* |
| Upstream | `https://github.com/enatterer/ml_surrogates_for_agent_based_transport_models` |

## Thesis reference

Mohd Zamin Quadri (2026). *Uncertainty Quantification for GNN Surrogates of Agent-Based Transport
Models.* M.Sc. thesis, M.Sc. Mathematics in Science and Engineering, Technical University of Munich,
School of Computation, Information and Technology. Supervisor: Prof. Dr. Stephan Günnemann.
Advisors: Dominik Fuchsgruber, M.Sc. and Elena Natterer, M.Sc. Submitted 15 May 2026.

Builds on: Natterer et al. (2025). *Machine Learning Surrogates for Agent-Based Models in
Transportation Policy Analysis.* Transportation Research Part C, 180, 105360.

## Provenance note — must appear on the page

Any portfolio page for this project has to make two things explicit.

**First, the split.** The GNN architecture, the MATSim-to-graph preprocessing, and the base
training pipeline are upstream work by Elena Natterer and collaborators, released under MIT. This
project takes the trained surrogate as given and contributes the uncertainty layer on top of it.
A page that shows the architecture diagram without saying whose architecture it is would be
misleading.

**Second, the scope.** The evidence covers one Paris network, one capacity-reduction intervention
family, a 1,000-scenario subset, and one model family. Conformal coverage is marginal over the
evaluated split, not a guarantee for any individual scenario, link, city, or policy. Do not write
"production-ready" or imply the method is validated beyond this setting.

Two published numbers were corrected after submission (Trial 8 AUROC) and one is reported but not
reproducible from the retained archive (Trial 7 rho). Use the values in this file, which are the
verified ones, and link to `docs/CORRIGENDUM.md` rather than restating the corrections on the page.
