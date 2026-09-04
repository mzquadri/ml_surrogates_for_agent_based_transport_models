# The six features

Each answered the same ten ways. Statistics are over all 31,635,000 node
observations; `ρ` is Spearman correlation between the feature and a link's **mean
absolute response** across the 1,000 scenarios.

| # | Feature | Unit | Range | % zero | Distinct | Dynamic | Model | ρ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `VOL_BASE_CASE` | vehicles | 0 – 1,596 | 23.86% | 5,694 | no | ✅ | **+0.885** |
| 1 | `CAPACITY_BASE_CASE` | veh/h | 0 – 14,400 | 10.79% | 36 | no | ✅ | +0.476 |
| 2 | `CAPACITY_REDUCTION` | veh/h | −7,200 – 0 | 87.94% | 29 | **yes** | ✅ | +0.336 |
| 3 | `FREESPEED` | m/s | 0 – 33.33 | 10.79% | 16 | no | ✅ | +0.411 |
| 4 | `HIGHWAY` | class code | −1 – 9 | 2.95% | 11 | no | ❌ | n/a [^hw] |
| 5 | `LENGTH` | m | 4.17 – 2,568.58 | 0.00% | 23,257 | no | ✅ | −0.075 |

[^hw]: No correlation is quoted for `HIGHWAY`. It is a nominal category with an
    ordinal encoding, so a rank correlation against it would treat road-class codes
    as an ordered quantity. The per-class table below is the honest presentation.

---

## 0 · `VOL_BASE_CASE`

**What** — car volume on the link in the unmodified network.
**From** — `links_base_case['vol_car']`, straight off the base network.
**Unit** — vehicles over the simulated period; float64.
**Values** — 0 to 1,596; 23.86% are zero (links carrying no car traffic at all).
**Dynamic** — no. Identical in all 1,000 scenarios.
**Model** — yes, column 0.
**High/low** — high means a busy road in the untouched city.
**Target** — the strongest single relationship in the dataset, ρ = +0.885. But the
correlation hides the shape:

| base volume | links | median \|response\| |
| --- | --- | --- |
| 0 – 67 | 22,739 | 1.84 |
| 134 – 202 | 1,062 | 11.45 |
| 336 – 403 | 153 | 19.75 |
| **471 – 538** | **65** | **44.77** ← peak |
| 605 – 672 | 34 | 28.44 |
| 874 – 941 | 313 | 12.58 |

**The relationship is an inverted U, not a straight line.** Sensitivity climbs to
roughly 500 veh and then *falls*: the very busiest links — motorway-class, high
capacity — are more stable than moderately busy ones. A single correlation
coefficient would have hidden this entirely.

**Best visualisation** — binned median with an IQR band, x on a linear scale, with
the peak marked.
**For a visitor** — "How busy a road already is predicts how much it changes better
than anything else. But the very biggest roads are the steady ones; it is the
merely-busy roads that swing hardest."

---

## 1 · `CAPACITY_BASE_CASE`

**What** — how many vehicles per hour the link can carry before the policy.
**From** — `np.where(modes.contains('car'), capacity, 0)`. The mask is the reason
10.79% are zero: **links no car may use are set to zero**, not missing.
**Unit** — veh/h; only 36 distinct values, so it is effectively categorical.
**Dynamic** — no.
**Model** — yes, column 1.
**High/low** — high means a wide, fast road. Zero means "not a car road".
**Target** — ρ = +0.476, dropping to +0.311 among car-capable links only. Much of
the apparent strength is the zero/non-zero split rather than a gradient.
**Best visualisation** — bar chart over the 36 distinct values, not a histogram.
**For a visitor** — "Capacity is how much traffic a road can take. Roads closed to
cars are recorded as zero rather than left blank."

---

## 2 · `CAPACITY_REDUCTION` — the intervention

**What** — how much capacity the policy removed from this link.
**From** — `capacities_new − capacity_base_case`, recomputed per scenario.
**Unit** — veh/h, always ≤ 0. 28 distinct non-zero values, all round numbers
(−240, −400, −480, −600, −720, −800, −1200, −1600, −1800, −2400, −3000, −3600 …),
consistent with removing whole lanes.
**Dynamic** — **yes, and it is the only one.** Max difference across scenarios
7,200.
**Model** — yes, column 2.
**High/low** — more negative means more capacity taken away.
**Target** — ρ = +0.336 node-wise, and the binned view is cleanly monotonic:

| capacity removed | node observations | mean \|response\| |
| --- | --- | --- |
| 240 – 400 | 942,656 | 6.27 |
| 800 – 1,200 | 361,238 | 10.15 |
| 1,800 – 7,200 | 387,364 | 20.56 |

**Best visualisation** — the intervention drawn on the map, with a monotone bar
chart of magnitude against response beside it.
**For a visitor** — "This is the policy itself. Everything else about the city
stays the same; this one number is what changes from scenario to scenario."

---

## 3 · `FREESPEED`

**What** — free-flow speed limit.
**From** — `np.where(modes.contains('car'), freespeed, 0)`, same car mask as
capacity, which is why the zero sets are **identical, not merely equal in size**.
**Unit** — m/s. 33.33 m/s = 120 km/h; 8.33 m/s = 30 km/h, the most common value.
**Dynamic** — no.
**Model** — yes, column 3.
**Target** — ρ = +0.411 overall but only +0.157 among car-capable links: again
mostly the car/non-car split.
**Best visualisation** — bar chart over the 16 values, labelled in km/h for
readability.
**For a visitor** — "The speed limit, in metres per second. Convert by ×3.6 for
km/h."

---

## 4 · `HIGHWAY` — stored, not used

**What** — OSM road class, label-encoded.
**From** — `highway_mapping.get(x, -1)` in `help_functions.py:17`.

| Code | OSM class | Links | % ever intervened | mean \|response\| |
| --- | --- | --- | --- | --- |
| −1 | **`pt` (public transport) or unmapped** | 3,173 | 0.0% | 0.35 |
| 0 | trunk / motorway_link | 933 | **0.0%** | **9.15** |
| 1 | primary | 5,295 | **85.0%** | 9.69 |
| 2 | secondary | 4,328 | **79.1%** | 5.31 |
| 3 | tertiary | 3,792 | **89.1%** | 4.11 |
| 4 | residential | 11,796 | 0.0% | 2.27 |
| 5 | living_street | 732 | 0.0% | 1.07 |
| 6 | pedestrian | 29 | 0.0% | 0.00 |
| 7 | service | 471 | 0.0% | 0.00 |
| 8 | construction | 8 | 0.0% | 0.00 |
| 9 | unclassified | 1,078 | 0.0% | 2.26 |

Two things stand out. **The policy only ever touches classes 1, 2 and 3.** And
**motorways carry the second-highest response despite never being intervened** —
they absorb what the primary roads shed.

**Code −1 means `pt`, not "unclassified"** (which is 9), and because the encoding
uses `.get(x, -1)` it also catches any OSM value missing from the table. It is
*not* the same set as the zero-capacity links: 3,173 vs 3,412, with 285 and 524
links respectively in only one of the two. Corrected in `CORRIGENDUM.md` C8a.

**Model** — **no.** Excluded because the codes are nominal but the encoding is
ordinal: nothing makes `residential = 4` twice `secondary = 2`. For the same
reason no correlation or numeric binning is reported for this feature anywhere in
this analysis — only per-class summaries.
**Best visualisation** — the ladder above: "% ever intervened" against "mean
response", one row per class.
**For a visitor** — "Road type. The model ignores it, because numbering road types
1–9 would imply an order that does not exist."

---

## 5 · `LENGTH`

**What** — link length.
**From** — `links_base_case['length']`.
**Unit** — metres, never zero, 23,257 distinct values — the only near-continuous
feature.
**Dynamic** — no.
**Model** — yes, column 5.
**Target** — ρ = −0.075. Essentially unrelated; the binned view is flat.
**Note** — 1,756 links have identical start and end coordinates yet a non-zero
length, so `LENGTH` is the true road length while the geometry is degenerate. See
[06_anomalies.md](06_anomalies.md).
**Best visualisation** — log-scale histogram; skip any length-vs-response chart,
there is nothing to show.
**For a visitor** — "How long the road segment is. It turns out not to predict
much on its own."

---

## What this means together

The model receives five numbers per link, of which **four never change**. They
describe the city; the fifth describes the policy. Everything the surrogate
learns about *which* scenario it is looking at arrives through
`CAPACITY_REDUCTION` — and everything it knows about how the city will absorb
that change comes from the four static ones, above all `VOL_BASE_CASE`.

---

# The other five stored fields

`x` is six of the eleven fields each `Data` object stores. The remaining five are
tensors in their own right, and two of them are not features at all.

## 7 · `pos` — link geometry

**What** — three points per link: start, end, and midpoint.
**From** — `help_functions.get_link_geometries`, stacking `geom.coords[0]`,
`geom.coords[-1]`, and their arithmetic mean.
**Shape / type** — `[31635, 3, 2]` float32. Axis 1 is start / end / midpoint,
axis 2 is `[longitude, latitude]`.
**Unit** — decimal degrees, **EPSG:4326 (WGS84)**, set explicitly in
`process_simulations_for_gnn.main`. Never a projected metric system.
**Values** — longitude 2.15293–2.49007, latitude 48.75772–48.92620.
**Dynamic** — no. Byte-identical in all 1,000 scenarios.
**Model** — yes. `PointNetConv` consumes `pos[:, 0]` (start) in the first layer
and `pos[:, 1]` (end) in the second, appending the relative offset `pos_j − pos_i`
to the features — which is why the first Linear takes 7 inputs, not 5.
**Data quality** — the midpoint is the straight-line midpoint, verified to a
maximum deviation of 3.8e-6 degrees from `mean(start, end)`; for a curved link it
does not lie on the road. 1,756 links have identical start and end, so their
relative offset is exactly zero and the geometric signal is empty for them.
**Best visualisation** — the network drawn on a map with an equirectangular
aspect; at 48.8°N a degree of longitude is ~73 km against ~111 km for latitude.
**For a visitor** — "Where each road segment starts and ends, in ordinary GPS
coordinates."

## 8 · `y` — the target

**What** — the policy-induced change in car volume on each link.
**From** — `compute_target_tensor_only_edge_features`:
`gdf["vol_car"] − vol_base_case`.
**Shape / type** — `[31635, 1]` float32. Unit veh/h.
**Values** — mean 0.4189, std 10.7099, min −237.38, max 180.00, **27.62% exactly
zero**, no NaNs.
**Dynamic** — yes, by construction.
**Model** — this is what is predicted.
**Interpretation** — positive means more traffic than the base case, negative
less. Near-symmetric with heavy tails, because capacity removed in one place
displaces traffic to another and the two largely cancel.
**Data quality** — 7,471 links (23.62%) are exactly zero in *every* scenario, and
only 3,412 of those are structurally inert. Any metric averaged over all nodes is
flattered by that fixed subset.
**Best visualisation** — log-scale histogram of non-zero values beside a box plot;
the zero share stated separately rather than drawn.
**For a visitor** — "How much the traffic on this road changed because of the
policy."

## 9 · `edge_index` — the graph

**What** — which links are adjacent in the line graph.
**From** — `get_link_geometries` builds `from_idx → to_idx` pairs; PyG's
`LineGraph()` transform inverts them.
**Shape / type** — `[2, 59851]` int64, source row over target row.
**Dynamic** — no. Byte-identical across all 1,000 scenarios, so it can be loaded
once and reused.
**Model** — yes; every convolution reads it.
**Structure** — **directed and not reciprocal**: 53,955 unique undirected pairs
against 59,851 entries. Direction encodes that traffic can pass from one link into
the other. 766 self-loops, 76 isolated nodes, 121 components with the largest
holding 92.70%.
**Data quality** — the 76 isolated nodes are exactly why `Data.num_nodes` reads
31,559 while `x` has 31,635 rows.
**Best visualisation** — degree histogram, plus the isometric lifting diagram
showing geography and graph as two planes.
**For a visitor** — "Which roads feed into which. The model passes information
along these connections."

## 10 · `mode_stats_diff` — transport-mode statistics

**What** — how the scenario's trips differ from the base case, per transport mode.
**From** — `calculate_avg_mode_stats`, grouping the trip table by mode.
**Shape / type** — `[6, 3]` float32. **Rows** are the six transport modes in
pandas `groupby("mode")` order, which is alphabetical; the mode labels themselves
are not stored, so the row-to-mode mapping cannot be recovered from the published
tensors alone. **Columns** are average total travel time, average total routed
distance, average trip count.
**Dynamic** — yes, per scenario.
**Model** — **no.** Consumed only when `predict_mode_stats=True`, which defaults
to `False`; no retained trial configuration enables it, and the architecture
docstring records that mode-statistics prediction was never finetuned.
**Best visualisation** — none recommended while the row-to-mode mapping is
unrecoverable.
**For a visitor** — omit; it is carried by the preprocessing and never used.

## 11 · `mode_stats_diff_perc` — the same, as a percentage

**What** — `mode_stats_diff` divided by the base case, times 100.
**Shape / type** — `[6, 3]` float64 — the only float64 tensor besides `x`.
**Dynamic** — yes.
**Model** — no.
**Data quality — do not use.** Columns 0 and 1 sit between **−100.00% and
−99.97%** in every scenario, implying travel time and routed distance collapsed to
nearly zero under every policy, which is not physically possible for a capacity
reduction. The construction subtracts two independently-derived column lists
**positionally**, so any difference in column order, count or aggregation scale
between the scenario frame and the base-case frame misaligns silently. That is a
plausible explanation consistent with the output, not a confirmed diagnosis — the
base-case table is not part of the release, so it cannot be checked.
**Impact** — none. No reported number depends on it.

---

## The six columns that were designed and never built

`EdgeFeatures` declares twelve members, indices 0–11:

```python
VOL_BASE_CASE = 0   CAPACITY_BASE_CASE = 1   CAPACITY_REDUCTION = 2
FREESPEED     = 3   HIGHWAY            = 4   LENGTH             = 5
ALLOWED_MODE_CAR = 6   ALLOWED_MODE_BUS   = 7   ALLOWED_MODE_PT     = 8
ALLOWED_MODE_TRAIN = 9 ALLOWED_MODE_RAIL = 10   ALLOWED_MODE_SUBWAY = 11
```

Indices 6–11 would have one-hot encoded which transport modes each link permits.
They are **never written**: `use_allowed_modes = False` at module scope in both
`process_simulations_for_gnn.py` and `process_simulations_for_eign.py`, and the
tensor is assembled only from the keys present in `edge_feature_dict`.

Measured on the published corpus, `x` has **six columns**. It has never had twelve.
The mode information is not entirely absent from the data, though — the car mask
survives indirectly, because `CAPACITY_BASE_CASE` and `FREESPEED` are both zeroed
on exactly the 3,412 links no car may use.
