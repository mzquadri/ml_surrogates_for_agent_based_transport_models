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
