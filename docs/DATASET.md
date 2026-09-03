# Training corpus and model input

What the published dataloaders actually contain, measured rather than described from the
preprocessing code. Every number below comes from a full pass over all 20 batch files
(`datalist_batch_1.pt` … `datalist_batch_20.pt`, 2.44 GiB, published on the
[`train-data-v1`][release] release).

[release]: https://github.com/mzquadri/ml_surrogates_for_agent_based_transport_models/releases/tag/train-data-v1

## Corpus shape

| Property | Value |
| --- | --- |
| Batch files | 20 |
| Graphs per file | 50 |
| Graphs (scenarios) | 1,000 |
| Nodes per graph | 31,635 |
| Edges per graph | 59,851 |
| Node observations | 31,635,000 |
| Stored object | `torch.save` of a list of `torch_geometric.data.Data` |

Each `Data` object carries `x=[31635, 6]`, `pos=[31635, 3, 2]`, `y=[31635, 1]`,
`edge_index=[2, 59851]`, and two `[6, 3]` mode-statistics tensors.

Nodes are road links, not junctions: the network is expressed as a line graph, so an edge in
`edge_index` means "these two links are connected". Node and edge counts do not vary — every
scenario is the same Paris network under a different capacity intervention.

## Node features

Six features are stored. Statistics are over all 31,635,000 node observations.

| # | Feature | Min | Max | Mean | Std | % zero | Distinct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `VOL_BASE_CASE` | 0 | 1,596 | 50.91 | 135.83 | 23.86% | 5,694 |
| 1 | `CAPACITY_BASE_CASE` | 0 | 14,400 | 1,028.96 | 1,264.45 | 10.79% | 36 |
| 2 | `CAPACITY_REDUCTION` | −7,200 | 0 | −93.33 | 334.03 | 87.94% | 29 |
| 3 | `FREESPEED` | 0 | 33.33 | 8.15 | 4.01 | 10.79% | 16 |
| 4 | `HIGHWAY` | −1 | 9 | 2.73 | 2.13 | 2.95% | 11 |
| 5 | `LENGTH` | 4.17 | 2,568.58 | 91.60 | 109.94 | 0.00% | 23,257 |

`Distinct` counts exact distinct `float64` values over all 31,635,000 node
observations. An earlier version of this table understated three of these counts
and wrote "many" for the other two; the values above were recomputed in a single
streaming pass over all twenty batch files.

There are no NaNs in `x` or `y`.

Notes on individual features:

- **`VOL_BASE_CASE`** is the pre-intervention hourly volume. Its 23.86% zeros are links that
  carry no car traffic in the base simulation.
- **`CAPACITY_BASE_CASE`** and **`FREESPEED`** are both zero on exactly the same 3,412 links per
  graph — the masks are identical, not merely equal in size. Both are produced by
  `np.where(modes.contains("car"), value, 0)`, so these are the links no car may use. This set is
  *not* the same as `HIGHWAY == -1` (3,173 links): 285 links are −1 yet car-capable, and 524 are
  non-car with an ordinary road class. See [`CORRIGENDUM.md`](CORRIGENDUM.md) C8a.
- **`CAPACITY_REDUCTION`** is the intervention, and the only feature that differs between
  scenarios. It is zero or negative, taking 29 distinct values down to −7,200 veh/h.
- **`FREESPEED`** is in m/s; 33.33 m/s is 120 km/h.
- **`HIGHWAY`** is a label-encoded road class. Code **−1 means `pt` (public transport), or an OSM
  class absent from the mapping** — it is 10.03% of links. `unclassified` is a different category,
  code 9. The mapping is in `scripts/data_preprocessing/help_functions.py`. Because the encoding is
  ordinal but the categories are nominal, this feature is excluded from the five-feature set the
  models use. An earlier version of this document described −1 as "unclassified"; corrected in
  [`CORRIGENDUM.md`](CORRIGENDUM.md) C8a.
- **`LENGTH`** is in metres and is never zero.

### Only one feature varies across scenarios

Comparing every graph against the first, feature by feature:

| Feature | Identical across graphs | Max absolute difference |
| --- | --- | --- |
| `VOL_BASE_CASE` | yes | 0 |
| `CAPACITY_BASE_CASE` | yes | 0 |
| `CAPACITY_REDUCTION` | **no** | 7,200 |
| `FREESPEED` | yes | 0 |
| `HIGHWAY` | yes | 0 |
| `LENGTH` | yes | 0 |

`edge_index` is also byte-identical across graphs.

This is a property of the experiment design worth stating plainly: five of the six features and
the whole graph topology are constant context, and the entire scenario-discriminating signal
enters through `CAPACITY_REDUCTION`. Measured over all 1,000 scenarios, a scenario reduces
capacity on **306 to 11,305 links**, a median of 3,317 and a mean of 3,814, which is **12.06% of
the network**. All 1,000 footprints are distinct, 20,330 links (64.26%) are never intervened at
all, and the policy only ever touches OSM classes 1, 2 and 3 (primary, secondary, tertiary).
Earlier versions of this document quoted 873–4,299 and 7.82%, which came from the first batch file
rather than the full corpus; corrected in [`CORRIGENDUM.md`](CORRIGENDUM.md) C8b.

That does not make the other features useless — they tell the model which links are capable of
absorbing displaced traffic — but any claim about generalisation is a claim about one network
under one intervention family, which is the limitation the thesis already records.

### The effect is local but not confined

Mean absolute target on links whose capacity was reduced, against everywhere else:

| Scenario | Reduced links | All other links |
| --- | --- | --- |
| 0 | 15.47 | 3.00 |
| 1 | 20.68 | 2.96 |
| 2 | 17.62 | 3.02 |

The response is roughly five to seven times larger where the intervention lands, and is clearly
non-zero away from it. The spillover onto untouched links is the part a graph model is there to
capture; a per-link regression that ignored topology could not produce it.

## Position

`pos` is `[N, 3, 2]`: start, end, and midpoint of each link. Across the corpus the coordinates
span 2.153–2.490 in the first axis and 48.758–48.926 in the second, which is WGS84 longitude and
latitude over Paris rather than a projected metric system.

Two consequences follow, and neither is currently handled:

- A degree of longitude at this latitude is about 73 km against about 111 km for a degree of
  latitude, so the two axes are on different metric scales.
- `PointNetConv` consumes `pos_j - pos_i`, and those offsets are on the order of 1e−3, while the
  features concatenated beside them reach 1e4 before normalisation.

Projecting to a metric CRS before building the graphs would remove both. This is a note for
future work, not a correction to the submitted results.

## Target

`y` is the policy-induced change in link volume, `Delta v`, in veh/h.

| Statistic | Value |
| --- | --- |
| Mean | 0.4189 |
| Std | 10.7099 |
| Min | −237.38 |
| Max | 180.00 |
| Exact zeros | 27.62% |

Percentiles, over a bounded sample rather than the full array: p1 −36.62, p5 −9.24, p25 −0.52,
p50 0.00, p75 2.52, p95 13.60, p99 28.79.

The distribution is centred on zero, close to symmetric in the middle, and heavy-tailed at both
ends. Reducing capacity somewhere pushes traffic elsewhere, so gains and losses roughly cancel
across the network.

The 27.62% zero share measured here over all 1,000 scenarios agrees with the 27.58% recorded for
the 100-scenario test split in [`CORRIGENDUM.md`](CORRIGENDUM.md), and is clearly distinct from
the 87.94% zero share of the `CAPACITY_REDUCTION` input. Those two numbers being confused is
exactly what the corrigendum corrects.

## Model input

`PointNetTransfGAT` is constructed with `in_channels=5`. Every trained checkpoint in
`code/data/TR-C_Benchmarks/` has a first layer of shape `(256, 7)`, which is the five features
plus the two coordinates of the relative position that `PointNetConv` appends. The `HIGHWAY`
column is the one dropped, following the ablation study.

So three counts are in play and it is worth keeping them apart:

- **6** features stored in `x`,
- **5** consumed by the network,
- **11** per-node input quantities in total, if the six `pos` values are counted alongside the
  five features.

## Architecture

Defined in `code/scripts/gnn/models/point_net_transf_gat.py`.

```
x [N,5]  pos [N,3,2]
   |
PointNetConv 1     local MLP  Linear(5+2 -> 256) ReLU
  uses pos[:,0,:]  global MLP Linear(256 -> 512) ReLU Linear(512 -> 512) ReLU
   |
PointNetConv 2     local MLP  Linear(512+2 -> 256) ReLU
  uses pos[:,1,:]  global MLP Linear(256 -> 512) ReLU Linear(512 -> 128) ReLU
   |
TransformerConv(128 -> 64, heads=4)  -> 256   ReLU
TransformerConv(256 -> 128, heads=4) -> 512   ReLU
GATConv(512 -> 64)
   |
GATConv(64 -> out)
```

The two PointNet layers consume the start and the end of each link respectively; the stored
midpoint is not used in the forward pass.

| Variant | Output units | Parameters |
| --- | --- | --- |
| Point prediction | 1 | 1,416,835 |
| Two-output (CQR, heteroscedastic) | 2 | 1,416,902 |

Both counts were read from the saved checkpoints rather than from a rebuilt model.

## Reproducing these numbers

```bash
gh release download train-data-v1 \
  --repo mzquadri/ml_surrogates_for_agent_based_transport_models \
  --pattern 'datalist_batch_*.pt' \
  --dir data/train_data/dist_not_connected_10k_1pct

python scripts/analyse_train_data.py
```

The script streams one batch at a time and accumulates exact sums, so it needs about 1 GB of
memory rather than the full 2.44 GiB.
