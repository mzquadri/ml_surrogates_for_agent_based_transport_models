# Data-quality observations

Eight things found while reading the corpus. Most are properties of the network
rather than defects. None changes a reported result; the two that matter for
interpretation are marked.

| # | Observation | Count | Affects results? |
| --- | --- | --- | --- |
| 1 | Target exactly zero in all 1,000 scenarios | 7,471 (23.62%) | interpretation |
| 2 | Zero base volume **and** zero capacity | 3,412 (10.79%) | no |
| 3 | Zero-length geometries (start == end) | 1,756 | interpretation |
| 4 | Duplicated feature rows | 3,108 | no |
| 5 | Self-loops in `edge_index` | 766 | no |
| 6 | Isolated nodes (degree 0) | 76 | no |
| 7 | `mode_stats_diff_perc` ≈ −100% | all scenarios | no |
| 8 | `x` stored as float64 | every scenario | no |

There are **no NaNs and no infinities** anywhere in `x` or `y`.

---

## 1 · A quarter of the network never moves

7,471 links have a target of **exactly** 0.0 in every one of the 1,000 scenarios.

Only 3,412 of those are structurally inert (observation 2). That leaves **4,059
car-capable links that never register any change**, in any scenario, under any
policy.

This matters for reading the headline statistics. The dataset's "27.62% exact
zeros" is not evenly spread noise — a large part of it is a fixed subset of links
that are simply never affected. Any model scores well on those by predicting zero,
and metrics averaged over all links are flattered accordingly. The thesis's
selective-prediction and error-detection results are computed on the same full
node set, so they inherit this too.

**Marked for interpretation.** Not an error in the data; a property to state when
quoting node-level averages.

## 2 · Links no car can use

3,412 links have `VOL_BASE_CASE == 0` and `CAPACITY_BASE_CASE == 0`. These are
produced deliberately: `np.where(modes.contains('car'), value, 0)` zeroes capacity
and free-flow speed for links cars may not use — rail, subway, bus-only.

Expected and documented. Note this set is *not* the same as `HIGHWAY == -1`; see
[02_features.md](02_features.md) and `CORRIGENDUM.md` C8a.

## 3 · 1,756 zero-length geometries

These links have identical start and end coordinates, while `LENGTH` reports a
real non-zero length. So the geometry is degenerate but the attribute is not.

Two consequences. `PointNetConv` consumes relative positions `pos_j − pos_i`; for
these links the start-to-end offset is exactly zero, so the geometric signal the
two PointNet layers extract is empty and the layers fall back on features alone.
And any map rendering must draw them as points, not lines.

**Marked for interpretation.** Roughly 5.6% of links give the geometric part of the
architecture nothing to work with.

## 4 · 3,108 duplicated feature rows

3,108 rows are not unique across all six feature columns. These are genuinely
different roads that happen to share every attribute — same class, same capacity,
same speed, same length, same base volume. Common in a regular street grid.

They are distinguishable by position and by graph neighbourhood, which is what the
model actually uses, so this is not degeneracy in the learning problem.

## 5 · 766 self-loops

Links whose start and end junction coincide — roundabout carriageways and loop
roads — which the line-graph transform renders as a node adjacent to itself.
Legitimate geometry. Traversal code must tolerate them; the BFS in
[05_spillover.md](05_spillover.md) does.

## 6 · 76 isolated nodes

Links that touch no other link in the line graph. They carry features and a target
and receive a prediction, but no neighbour information can reach them, so their
prediction rests entirely on their own features.

They are also the exact cause of the `num_nodes` discrepancy — see
[01_schema_and_provenance.md](01_schema_and_provenance.md).

## 7 · The percentage mode-statistics tensor is not usable

`mode_stats_diff_perc` columns 0 and 1 sit between −100.00% and −99.97% in every
scenario, implying travel time and routed distance fell to nearly zero under every
policy. That is not physically possible for a capacity reduction.

The construction subtracts two independently-derived column lists **positionally**,
so any difference in column order, count or aggregation scale between the scenario
frame and the base-case frame would misalign silently. That is a plausible
explanation consistent with the output, not a confirmed diagnosis — the base-case
mode-statistics table is not part of the release, so it cannot be checked.

**No result depends on it.** `predict_mode_stats` defaults to `False`, no retained
trial configuration enables it, and the architecture docstring records that mode
statistics prediction was never finetuned. Full detail in
[01_schema_and_provenance.md](01_schema_and_provenance.md).

## 8 · `x` is float64

The feature tensor is the only float64 array in the corpus, at 1.45 MB per scenario
against 0.74 MB for `pos` and 0.13 MB for `y`. Roughly 1.4 GiB of the 2.44 GiB
release is this one choice, for values that are small integers, round capacity
figures, and lengths — none of which need double precision.

Casting to float32 would roughly halve the dataset. Noted for any future rebuild;
the published corpus is a fixed artifact and is not being changed.

---

## Nothing here invalidates a result

Observations 1 and 3 change how node-level averages should be *read*. The rest are
either intentional encodings, ordinary network geometry, or storage choices. The
corpus contains no missing values, no corrupt entries, and no duplicated scenarios.
