# Traffic Policy Confidence Desk

A standalone local dashboard built from audited aggregate results of Trial 8 in the thesis repository. It supports a practical accept/review decision: retain only the least-uncertain surrogate predictions, then route the remainder to MATSim or expert review.

## Run locally

Open `index.html` in any modern browser. No server, dependency installation, raw data, model weights, or network access is required.

## Data basis

- `../results/selective_prediction_s30.json`
- `../results/temperature_scaling_t8.json`
- `../results/conformal_conditional_coverage_t8.json`

The page embeds only aggregate audited metrics. It neither reads nor transmits raw MATSim data.
