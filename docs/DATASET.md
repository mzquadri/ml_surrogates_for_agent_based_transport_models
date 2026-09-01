# Training corpus and model input

What the published dataloaders actually contain, measured rather than described from the
preprocessing code. Every number below comes from a full pass over all 20 batch files
(`datalist_batch_1.pt` … `datalist_batch_20.pt`, 2.44 GiB, published on the
[`train-data-v1`][release] release).

[release]: https://github.com/mzquadri/ml-surrogates-thesis/releases/tag/train-data-v1

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
| 0 | `VOL_BASE_CASE` | 0 | 1,596 | 50.91 | 135.83 | 23.86% | many |
| 1 | `CAPACITY_BASE_CASE` | 0 | 14,400 | 1,028.96 | 1,264.45 | 10.79% | 26 |
| 2 | `CAPACITY_REDUCTION` | −7,200 | 0 | −93.33 | 334.03 | 87.94% | 27 |
| 3 | `FREESPEED` | 0 | 33.33 | 8.15 | 4.01 | 10.79% | 11 |
| 4 | `HIGHWAY` | −1 | 9 | 2.73 | 2.13 | 2.95% | 11 |
| 5 | `LENGTH` | 4.17 | 2,568.58 | 91.60 | 109.94 | 0.00% | many |

There are no NaNs in `x` or `y`.

Notes on individual features:

- **`VOL_BASE_CASE`** is the pre-intervention hourly volume. Its 23.86% zeros are links that
  carry no car traffic in the base simulation.
- **`CAPACITY_BASE_CASE`** and **`FREESPEED`** are both zero on exactly the same 3,412 links per
  graph — the masks are identical, not merely equal in size. These are the non-car links (rail,
  subway, bus-only), which have no vehicle capacity and no free-flow car speed.
- **`CAPACITY_REDUCTION`** is the intervention, and the only feature that differs between
  scenarios. It is zero or negative, taking 27 distinct values down to −7,200 veh/h.
- **`FREESPEED`** is in m/s; 33.33 m/s is 120 km/h.
- **`HIGHWAY`** is a label-encoded road class. The −1 category is 10.03% of links and marks an
  unclassified type rather than a road below class 0. Because the encoding is ordinal but the
  categories are nominal, this feature is excluded from the five-feature set the models use.
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
enters through `CAPACITY_REDUCTION`. A scenario reduces capacity on 873 to 4,299 links, 2,473 on
average, which is 7.82% of the network.

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
  --repo mzquadri/ml-surrogates-thesis \
  --pattern 'datalist_batch_*.pt' \
  --dir code/data/train_data/dist_not_connected_10k_1pct

python scripts/analyse_train_data.py
```

The script streams one batch at a time and accumulates exact sums, so it needs about 1 GB of
memory rather than the full 2.44 GiB.
