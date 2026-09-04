# One diagram per feature

Six cards, one for each column of `x`. Each answers the same four questions in the
same four places, so they can be read side by side:

| | |
|---|---|
| **what it is** | the header, in plain words |
| **where it is** | the feature drawn on the real Paris street network |
| **how it spreads** | its distribution |
| **what it does** | its relationship with the response |

The numbers in the strip along the foot are read from
[`feature_statistics.json`](../../portfolio_data_story/assets/feature_statistics.json),
not recomputed, so a card can never disagree with the published asset.

```bash
python scripts/figure_generation/generate_feature_cards.py --corpus $CORPUS --cache $CACHE
```

| # | Feature | Model input | The card |
|---|---|---|---|
| 0 | `VOL_BASE_CASE` | yes | [card](0_vol_base_case.png) |
| 1 | `CAPACITY_BASE_CASE` | yes | [card](1_capacity_base_case.png) |
| 2 | `CAPACITY_REDUCTION` | yes | [card](2_capacity_reduction.png) |
| 3 | `FREESPEED` | yes | [card](3_freespeed.png) |
| 4 | `HIGHWAY` | **no** | [card](4_highway.png) |
| 5 | `LENGTH` | yes | [card](5_length.png) |

### 0 · VOL_BASE_CASE
![VOL_BASE_CASE](0_vol_base_case.png)

### 1 · CAPACITY_BASE_CASE
![CAPACITY_BASE_CASE](1_capacity_base_case.png)

### 2 · CAPACITY_REDUCTION
![CAPACITY_REDUCTION](2_capacity_reduction.png)

### 3 · FREESPEED
![FREESPEED](3_freespeed.png)

### 4 · HIGHWAY
![HIGHWAY](4_highway.png)

The only card whose charts are categorical throughout, and the only column the model
never sees. Its integers are names for road classes, so a mean or a correlation would
be meaningless.

### 5 · LENGTH
![LENGTH](5_length.png)

---

The response panel on every card uses equal-width bands merged rightwards until each
holds at least 100 links. Quantile bins were tried first and hid the turn in
`VOL_BASE_CASE` entirely inside their top bin.
