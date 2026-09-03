# Web asset schema

The contract for anything built on these files. Every asset is a **derived
aggregate**; the 31,635,000-row observation table is never exported and the
2.44 GiB `.pt` corpus is never copied into this repository.

Regenerate with:

```bash
python scripts/data_exploration/build_web_assets.py --corpus <corpus dir> --cache <scratch dir>
```

Output is deterministic: sampling is seeded, ordering is fixed, and floats are
rounded before serialisation. Rebuilding produces byte-identical files.

## Conventions used everywhere

| Convention | Value |
| --- | --- |
| Coordinate system | **EPSG:4326 (WGS84)**, decimal degrees, rounded to 5 dp (~1 m at 48.8°N) |
| Coordinate naming | always `*_lon` / `*_lat` — never bare `x` / `y` |
| Response unit | veh/h, the change in car volume against the base case |
| Link identifier | `link_row`, the row index in the published tensors — see below |
| "response" | always the **target** `y`, never a model prediction. No asset here contains model output |

### `link_row` is the identifier, and why

The published tensors carry no MATSim or OSM link id — the base-network file that
held it is not part of the release. Row order is used instead, and it is safe to
do so because it was **verified stable across all 1,000 scenarios**: `pos`,
`edge_index`, and all five static feature columns are byte-identical in every
scenario. Row *i* is therefore the same physical road link everywhere in the
corpus.

Geometry is **not** a substitute key: only 31,504 of 31,635 links have a unique
`(start, end)` pair, because 1,756 links have zero-length geometry.

`link_row` is stable **within this published corpus**. It is not a durable
external identifier, and it must not be presented to a visitor as one. If the
corpus is ever regenerated from source, these indices may change.

---

## `links.csv` — 31,635 rows, one per road link

| Field | Type | Unit | Static? | Source | Interpretation | Public? |
| --- | --- | --- | --- | --- | --- | --- |
| `link_row` | int | — | static | row index | The identifier. Join key for every other asset | yes |
| `start_lon` | float | ° | static | `pos[:,0,0]` | Longitude of the link's start point | yes |
| `start_lat` | float | ° | static | `pos[:,0,1]` | Latitude of the start point | yes |
| `end_lon` | float | ° | static | `pos[:,1,0]` | Longitude of the end point | yes |
| `end_lat` | float | ° | static | `pos[:,1,1]` | Latitude of the end point | yes |
| `mid_lon` | float | ° | static | `pos[:,2,0]` | Longitude of the midpoint. **Exactly the mean of start and end** (verified, max deviation 3.8e-6 = float32 rounding) — derivable, included for convenience | yes |
| `mid_lat` | float | ° | static | `pos[:,2,1]` | Latitude of the midpoint, same note | yes |
| `highway_code` | int | code | static | `x[:,4]` | OSM road class, −1..9. See `highway_classes.json` | yes |
| `vol_base_case` | float | vehicles | static | `x[:,0]` | Car volume on this link in the *unmodified* network. The strongest single predictor of response (ρ = +0.885) | yes |
| `capacity_base_case` | float | veh/h | static | `x[:,1]` | Base capacity, **zeroed for non-car links** | yes |
| `freespeed_ms` | float | m/s | static | `x[:,3]` | Free-flow speed, zeroed for non-car links. 33.33 m/s = 120 km/h | yes |
| `length_m` | float | m | static | `x[:,5]` | Link length | yes |
| `degree` | int | — | static | `edge_index` | Number of adjacent links in the line graph (0–10) | yes |
| `times_intervened` | int | scenarios | derived | `x[:,2] != 0` | In how many of the 1,000 scenarios this link had capacity reduced. 0 for 64.26% of links | yes |
| `mean_abs_response` | float | veh/h | derived | `mean(abs(y))` | Mean absolute response over all 1,000 scenarios | yes |
| `std_response` | float | veh/h | derived | `std(y)` | Standard deviation of the signed response | yes |

`CAPACITY_REDUCTION` (`x[:,2]`) is deliberately **absent** — it is the only
per-scenario column and belongs in the scenario assets, not here.

## `scenarios.json` — 1,000 rows

Array-of-arrays under `scenarios`, column names in `fields`:

| Field | Type | Unit | Interpretation |
| --- | --- | --- | --- |
| `scenario_id` | int | — | Index into the corpus (batch order) |
| `links_intervened` | int | links | How many links had capacity reduced (306–11,305) |
| `capacity_removed_vehh` | float | veh/h | Total capacity removed, summed over links |
| `total_abs_response_vehh` | float | veh/h | Sum of `abs(y)` across the network |
| `offsite_response_share` | float | 0–1 | Share of total response landing on links that were **not** intervened |
| `mean_abs_response_vehh` | float | veh/h | Mean `abs(y)` per link |

## `scenario_<id>.json` — representative scenarios only

| Field | Type | Interpretation |
| --- | --- | --- |
| `scenario_id` | int | Corpus index |
| `selection_rules` | list | Every rule that selected this scenario, each with `rule` and `reason`. A scenario may satisfy several |
| `links_intervened`, `capacity_removed_vehh`, `total_abs_response_vehh`, `offsite_response_share` | | As in `scenarios.json` |
| `intervened_link_rows` | int[] | `link_row` values with reduced capacity |
| `reduction_magnitudes_vehh` | float[] | Lookup table of the 28 distinct magnitudes |
| `intervened_reduction_index` | int[] | Index into the lookup, parallel to `intervened_link_rows`. Stored this way to keep the files small |
| `response_percentiles_vehh` | object | 1/5/25/50/75/95/99th percentiles of the signed response |
| `top_50_responding_link_rows` | int[] | The 50 links with the largest `abs(y)` |
| `top_50_responding_values_vehh` | float[] | Their signed responses, parallel array |

## `representative_scenarios.json` / `representative_links.json`

Index of which scenario or link each rule selected, with the rule name and a
one-line reason. **Every selection is a stated extremum or quantile of a measured
quantity — nothing was chosen because it looked good.**

## `spillover_decay.json`

Two blocks, `undirected` and `directed`, each with parallel arrays indexed by
hop 0–8, plus a `method` block recording the BFS definition, the aggregation, the
seed and sample size, and an explicit caveat. Fields: `mean_abs_response_vehh`,
`share_of_total_abs_response`, `cumulative_network_reached`, and the two
`unreachable_*` scalars.

**Read the response profile and the reachability curve together.** The profile
alone does not establish that distance stops mattering.

## `highway_classes.json`

One entry per OSM class code with `n_links`, `share_of_network`,
`share_ever_intervened`, `mean_abs_response_vehh`, and base volume/capacity means.
Carries a note that `-1` means *`pt` **or** unmapped*.

## `feature_summary.json`

Per-feature statistics plus `binned_vs_abs_response`: median, IQR and mean of the
response within bins of the feature, so a chart can show the **shape** of the
relationship rather than a single correlation number. Bins with fewer than 20
links are dropped. Static features are binned over car-capable links only
(`capacity_base_case > 0`); `CAPACITY_REDUCTION` is binned node-wise over
intervened links, which is noted in the file.

## `narrative_link.json`

Link 18785 — the highest-volume link in the network, a trunk road, never
intervened in any scenario, yet strongly affected. Includes an `identity_check`
block recording that it is the unique maximum, that no other row shares its
geometry, and that **no MATSim/OSM id is available**. `response_sorted_vehh`
holds its 1,000 responses sorted ascending, ready to plot directly.

## Display guidance

All fields are safe to publish: this is simulation output over a public road
network, with no personal data.

Two things must accompany any public use:

1. **The response is simulated, not measured.** It comes from MATSim runs on a 1%
   population sample of Paris, not from traffic counts.
2. **Scope.** One city, one capacity-reduction intervention family, 1,000
   scenarios. Nothing here generalises to other cities or policies.

None of these assets contain model predictions or uncertainty estimates. They
describe the *data*. Anything about model performance belongs to
`scripts/verify_headline_results.py` and the figures under `docs/figures/results/`.
