# The auxiliary tensors

`mode_stats_diff` and `mode_stats_diff_perc` are stored in every scenario and read
by nothing. They are documented here because "the model does not use it" is not a
reason to leave a stored field unexplained — and because the percentages look
alarming until you work out why.

## What they are

Both are `[6, 3]`: six transport modes by three quantities. The columns come from
`calculate_avg_mode_stats` in `scripts/data_preprocessing/help_functions.py`:

```python
mode_stats = (df.groupby("mode")
                .agg({"travel_time": ["mean", "count"], "routed_distance": "mean"})
                .reset_index())
mode_stats.columns = ["mode", "avg_travel_time", "trip_count", "avg_routed_distance"]
```

after a second aggregation across seeds, giving columns named
`avg_total_travel_time`, `avg_total_routed_distance` and `avg_trip_count`.

The difference itself is computed in `process_simulations_for_eign.py`:

```python
numeric_cols_base_case = gdf_basecase_mean_mode_stats.select_dtypes(include=[np.number]).columns
numeric_cols = df_mode_stats.select_dtypes(include=[np.number]).columns
mode_stats_diff = (df_mode_stats[numeric_cols].values
                   - gdf_basecase_mean_mode_stats[numeric_cols_base_case].values)
data.mode_stats_diff = torch.tensor(mode_stats_diff, dtype=torch.float)
mode_stats_diff_perc = (mode_stats_tensor
                        / gdf_basecase_mean_mode_stats[numeric_cols_base_case].values * 100)
data.mode_stats_diff_perc = mode_stats_diff_perc
```

So `diff = scenario − base` and `perc = diff / base × 100`, both element-wise on a
6 × 3 grid.

Both vary between scenarios. Neither contains a NaN, checked across all 1,000.

## The values look wrong

Columns 0 and 1 sit at about **−99.99%** for every one of the six modes, in every
scenario, while column 2 stays near zero:

| Row | col 0 mean % | col 1 mean % | col 2 mean % |
|---:|---:|---:|---:|
| 0 | −99.9664 | −99.9664 | −0.2366 |
| 1 | −99.9965 | −99.9972 | −0.0856 |
| 2 | −99.9869 | −99.9880 | −0.0001 |
| 3 | −99.9960 | −99.9960 | +0.0060 |
| 4 | −99.9975 | −99.9975 | +0.1748 |
| 5 | −99.9968 | −99.9968 | −0.0650 |

Read at face value this says every policy scenario eliminates 99.99% of all travel
time and all distance travelled across the whole region, for every mode at once,
while leaving the number of trips unchanged. That is not a plausible simulation
result.

No cell is *exactly* −100: the count over the whole corpus is 0 of 18,000.

## The cause, established rather than guessed

The base-case CSV is not in the repository or on any release, so the two sides
cannot be compared directly. They can be **reconstructed**, because the two stored
tensors over-determine them:

```text
perc = diff / base × 100    →    base = diff / (perc / 100)    →    scenario = base + diff
```

Doing that for scenario 0:

| Row | column | implied base | implied scenario | ratio |
|---:|---|---:|---:|---:|
| 0 | avg_total_travel_time | 3,203,165.24 | 1,077.74 | 0.000336 |
| 0 | avg_total_routed_distance | 9,934,483.17 | 3,343.17 | 0.000337 |
| 0 | avg_trip_count | 2,980.21 | 2,978.00 | 0.999257 |
| 1 | avg_total_travel_time | 16,448,936.05 | 507.05 | 0.000031 |
| 1 | avg_total_routed_distance | 166,505,928.55 | 4,712.55 | 0.000028 |
| 1 | avg_trip_count | 35,050.17 | 35,020.00 | 0.999139 |

The implied **scenario** values are ordinary per-trip averages: 1,078 s ≈ 18 min
of travel time, 3,343 m of routed distance. The implied **base** values are those
same quantities multiplied by roughly the trip count — that is, sums.

The hypothesis "the base side stores per-mode sums while the scenario side stores
per-mode means" predicts that the ratio must equal `1 / trip_count`. Testing that
across 100 scenarios from four different batches:

| Column | median relative error | 90th percentile |
|---|---:|---:|
| 0, travel time | **0.34%** | 11.41% |
| 1, routed distance | **0.30%** | 1.81% |

The ratio *is* `1 / trip_count`, to within a third of a percent. The small residual
is expected, because the policy changes the trip count slightly, so the scenario
and base counts are not identical.

Column 2 is unaffected because both sides hold a count, on the same scale — which
is exactly why its percentages are small and sensible.

**Conclusion.** The two sides of the subtraction are on different scales: a sum is
being subtracted from a mean, and then divided by the sum. The near −100% is
arithmetic, not a simulation result. The column order is *not* misaligned — the
reconstruction shows column 1 of the scenario side is a distance (3,343 m), not the
trip count — so this is a scale mismatch alone.

This is recorded as CORRIGENDUM C12. It cannot be repaired from the published
artifacts, because the base-case CSV that would let the sums be rebuilt is not part
of any release.

## Why it does not affect any result

Neither tensor is read anywhere in the training or evaluation code. The model's
only mode-stats path reads an attribute called `data.mode_stats`, which does not
exist in this corpus, and it is disabled (`predict_mode_stats=False`) with no
corresponding parameters in any trained checkpoint.

Every headline number in this repository is computed from `x`, `pos`, `edge_index`
and `y`. The auxiliary tensors are inert.

## Should they be visualised?

No. A chart of these values would be a chart of the scale mismatch, and would
invite a reader to interpret −99.99% as a finding. The table above is the honest
presentation: the numbers, the reconstruction that explains them, and the
statement that nothing downstream depends on them.
