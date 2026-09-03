# Schema and provenance

Every attribute of the published `Data` objects, where it comes from, and what it
costs. Line references are to the code that produced the corpus.

## The stored object

Each `.pt` file is `torch.save` of a Python `list` of 50
`torch_geometric.data.Data` objects. Twenty files, 1,000 scenarios.

```python
Data(edge_index=[2, 59851], num_nodes=31559, x=[31635, 6],
     pos=[31635, 3, 2], y=[31635, 1],
     mode_stats_diff=[6, 3], mode_stats_diff_perc=[6, 3])
```

| Attribute | Shape | dtype | Bytes | Built at |
| --- | --- | --- | --- | --- |
| `x` | [31635, 6] | float64 | 1,518,480 | `process_simulations_for_gnn.py` — `edge_tensor` |
| `pos` | [31635, 3, 2] | float32 | 759,240 | `help_functions.get_link_geometries` |
| `y` | [31635, 1] | float32 | 126,540 | `help_functions.compute_target_tensor_only_edge_features` |
| `edge_index` | [2, 59851] | int64 | 957,616 | `get_link_geometries` + PyG `LineGraph()` |
| `mode_stats_diff` | [6, 3] | float32 | 72 | `help_functions.calculate_avg_mode_stats` |
| `mode_stats_diff_perc` | [6, 3] | float64 | 144 | same, divided by the base case |
| | | | **3,362,092** | per scenario |

`x` is the only float64 tensor. It costs 1.45 MB per scenario, roughly 1.4 GiB
across the corpus, for no precision benefit — every value in it is either a small
integer, a round capacity figure, or a length with millimetre precision. Casting
it to float32 would halve the dominant storage cost. Noted, not changed: the
published corpus is a fixed artifact.

## Feature column order

From the `EdgeFeatures` IntEnum, `process_simulations_for_gnn.py:56`:

```python
VOL_BASE_CASE = 0    CAPACITY_BASE_CASE = 1    CAPACITY_REDUCTION = 2
FREESPEED     = 3    HIGHWAY            = 4    LENGTH             = 5
```

The enum continues to `ALLOWED_MODE_CAR = 6` through `ALLOWED_MODE_SUBWAY = 11`,
but `use_allowed_modes = False` at module scope, so those six columns were never
written. That is why `x` has exactly six columns and not twelve.

## Where each column comes from

Four columns are read straight off the **base network** and are therefore identical
in every scenario:

```python
vol_base_case      = links_base_case['vol_car'].values
capacity_base_case = np.where(links_base_case['modes'].str.contains('car'),
                              links_base_case['capacity'], 0)
length             = links_base_case['length'].values
freespeed          = links_base_case['freespeed'].values
```

Two are recomputed per scenario in `get_basic_edge_attributes`:

```python
capacities_new     = np.where(gdf["modes"].str.contains("car"), gdf["capacity"], 0)
capacity_reduction = capacities_new - capacity_base_case
highway            = gdf["highway"].apply(lambda x: highway_mapping.get(x, -1)).values
```

`highway` is recomputed but measured identical in all 1,000 scenarios — the road
class does not change when capacity is reduced. Only `capacity_reduction` actually
varies.

## The target

```python
def compute_target_tensor_only_edge_features(vol_base_case, gdf):
    edge_car_volume_difference = gdf["vol_car"].values - vol_base_case
    return torch.tensor(edge_car_volume_difference, dtype=torch.float).unsqueeze(1)
```

`y` is scenario car volume minus base-case car volume, per link. Positive means
more traffic than the base case, negative means less. Reported throughout the
thesis in veh/h.

## Positions and the coordinate system

`get_link_geometries` takes each link's `LineString` and stacks three points:

```python
start_points = [geom.coords[0]  for geom in links_gdf_input.geometry]
end_points   = [geom.coords[-1] for geom in links_gdf_input.geometry]
edge_midpoints = ((start.x + end.x) / 2, (start.y + end.y) / 2)

stacked_edge_geometries_tensor = torch.stack(
    [edge_start_point_tensor, edge_end_point_tensor, edge_midpoint_tensor], dim=1)
```

So the axis-1 order is **verified as start, end, midpoint**:

| Slice | Meaning | Longitude range | Latitude range |
| --- | --- | --- | --- |
| `pos[:, 0]` | start point | 2.15293 – 2.49007 | 48.75772 – 48.92620 |
| `pos[:, 1]` | end point | 2.15293 – 2.49007 | 48.75772 – 48.92620 |
| `pos[:, 2]` | midpoint | 2.15302 – 2.49007 | 48.75779 – 48.92620 |

The **CRS is EPSG:4326 (WGS84)**, set in `process_simulations_for_gnn.main`:

```python
gdf_basecase_links = gdf_basecase_links.set_crs("EPSG:4326", allow_override=True)
```

Axis 2 is therefore `[longitude, latitude]` in decimal degrees — never a projected
metric system. The web assets name every coordinate `*_lon` / `*_lat` for exactly
this reason.

**The midpoint is the straight-line midpoint**, not the midpoint along the
polyline: it is the arithmetic mean of the two endpoints, verified to a maximum
deviation of 3.8e-6 degrees, which is float32 rounding. For a curved link it does
not lie on the road.

## The node-count question: 31,635 vs 31,559

| Quantity | Value |
| --- | --- |
| Rows in `x`, `pos`, `y` | 31,635 |
| `Data.num_nodes` | 31,559 |
| Difference | 76 |
| Nodes with degree 0 | **76** |

This is not a bug in PyTorch Geometric. `Data.num_nodes` is a *derived* property:
when no explicit node count is stored, PyG infers it from `edge_index` — it cannot
know about nodes that appear in no edge. The corpus has 76 links that touch no
other link, so they are absent from `edge_index` and fall out of the inferred
count. The feature tensors are correct and complete at 31,635 rows.

The construction code sets `data.num_nodes = edge_index.shape[1]` *before* the
`LineGraph()` transform, after which the transform rewrites the graph; the
explicit count does not survive in a form that covers the isolated links.

**Consideration for any future dataset build:** set `num_nodes = x.size(0)`
explicitly after the transform, so the object is self-consistent and downstream
code cannot silently disagree about how many nodes exist. The published corpus is
deliberately left unchanged — it is the artifact the reported results were
computed on, and altering it would break replay (see `CORRIGENDUM.md` C5).

In practice nothing in this project indexes by `num_nodes`; the models read
`data.x` directly, so the discrepancy never affected a result.

## The mode-statistics tensors

Both are `[6, 3]`. They are built in `calculate_avg_mode_stats`
(`help_functions.py:88`), which groups a scenario's trip table by transport mode
and aggregates:

```python
average_mode_stats.columns = ["mode", "avg_total_travel_time",
                              "avg_total_routed_distance", "avg_trip_count"]
```

- **Rows (6)** — one per transport mode, in pandas `groupby("mode")` order, which
  is alphabetical. The mode labels themselves are *not* stored in the tensor, so
  the row-to-mode mapping cannot be recovered from the published data alone.
- **Columns (3)** — average total travel time, average total routed distance,
  average trip count.

`mode_stats_diff` is the scenario's values minus the base case's;
`mode_stats_diff_perc` is that difference divided by the base case, times 100.

**The percentage tensor is implausible** and should not be used. Its first two
columns sit between −100.00% and −99.97% in every scenario, which would mean travel
time and routed distance collapsed to near zero under every policy — physically
impossible for a capacity reduction. The third column is small and plausible.

The likely cause is visible in the construction:

```python
numeric_cols_base_case = gdf_basecase_mean_mode_stats.select_dtypes(...).columns
numeric_cols           = df_mode_stats.select_dtypes(...).columns
mode_stats_diff = df_mode_stats[numeric_cols].values - \
                  gdf_basecase_mean_mode_stats[numeric_cols_base_case].values
```

The two column lists are derived independently and then subtracted **positionally**.
If the two frames' numeric columns differ in order, count, or aggregation scale,
the subtraction silently misaligns. This is a plausible explanation supported by
the shape of the output, not a diagnosis confirmed against the source frames — the
base-case mode-statistics table is not part of the release, so it cannot be checked
directly.

**Was any of it used?** No evidence that it was. `PointNetTransfGAT.forward`
consumes `data.mode_stats` only when `predict_mode_stats=True`; the flag defaults
to `False` (`base_gnn.py:27`), no retained trial configuration under `results/`
records it being enabled, and the architecture's own docstring states that "mode
statistics prediction is not finetuned". Every reported metric in this thesis is a
node-level target metric. The tensors appear to be carried along by the
preprocessing and never read.
