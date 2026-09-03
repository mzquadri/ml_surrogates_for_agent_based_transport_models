# Representative scenarios and links

Nothing on this page was chosen because it looked interesting. Every entry is the
**argmax, argmin, or a stated quantile** of a measured quantity, computed in
`scripts/data_exploration/explore_representatives.py` and emitted with its rule
into the assets. Re-running the script reproduces exactly this list.

That matters for anything built on top of these: they are objectively selected
examples, not anecdotes, and any page using one should say which rule picked it.

## Scenarios

| Scenario | Selection rule | Links intervened | Capacity removed | Total \|response\| | Off-site share |
| --- | --- | --- | --- | --- | --- |
| **199** | smallest footprint · highest off-site share · most response per capacity | 306 | 224,360 | 72,756 | **91.8%** |
| 565 | 25th-percentile footprint | 2,224 | — | — | 75.5% |
| 279 | median footprint | 3,318 | 2,644,200 | 133,550 | 64.5% |
| 130 | 75th-percentile footprint | 4,889 | — | — | 54.4% |
| 409 | least response per capacity removed | 8,144 | — | — | 46.3% |
| **94** | lowest off-site share · largest total response | 11,101 | 8,589,720 | 235,838 | 33.5% |
| **95** | largest footprint · most capacity removed | 11,305 | 8,733,600 | 232,287 | 33.6% |
| 990 | smallest total response | 1,200 | — | — | 87.2% |

Three rules select scenario 199 and two select each of 94 and 95, which is why the
asset index lists twelve rules across eight distinct scenarios. Each
`scenario_<id>.json` carries **every** rule that selected it.

### Scenario 199 — why it is worth showing

Selected by three independent rules: it has the **fewest** intervened links (306,
under 1% of the network), the **highest** off-site response share (91.8%), and the
**most** response per unit of capacity removed.

Those three facts are the same fact seen three ways. A small, concentrated
intervention produces a response that is almost entirely somewhere else. It is the
clearest case in the corpus of the property described in
[04_intervention.md](04_intervention.md) — that the target is mostly displacement —
and it is the natural opening example for a visitor.

Its opposite is scenario 94: 11,101 links intervened, and only a third of the
response lands off-site. Showing the two together is the honest framing, because it
makes clear that the off-site share is a function of how much of the network the
policy already covers.

## Links

| Link row | Selection rule | Class | Times intervened | mean \|response\| | std |
| --- | --- | --- | --- | --- | --- |
| 20093 | intervened in the most scenarios | primary | 660 | 28.86 | 21.76 |
| 9884 | largest standard deviation of response | primary | 334 | 75.63 | 98.78 |
| **18785** | highest base-case volume · largest mean response among never-intervened links | trunk | **0** | **54.43** | 62.83 |
| 30244 | greatest length | pt | 0 | 0.00 | 0.00 |

### Link 18785 — why it is worth showing

Selected by two independent rules: it has the **highest base-case car volume in the
network** (1,596), and among the 20,330 links that are **never intervened in any of
the 1,000 scenarios** it has the **largest mean response**.

It is a trunk road — a class the policy never touches (see
[04_intervention.md](04_intervention.md)) — and yet its traffic changes by 54.4
veh/h on average, with a standard deviation of 62.8. It absorbs what the arterial
network sheds without ever being the target of a policy.

**Identity is verified before use.** It is the unique maximum-volume link (no ties),
no other row shares its geometry, and its start and end coordinates are recorded in
`assets/narrative_link.json` so the row index can be cross-checked. There is **no
MATSim or OSM identifier** available for it — the base-network file that carried
one is not part of the release — so `link_row` is the only identifier, and it is
stable within this corpus only. See [`assets/SCHEMA.md`](assets/SCHEMA.md).

`narrative_link.json` also carries its 1,000 responses sorted ascending, ready to
plot without further processing.

## Reproducing

```bash
python scripts/data_exploration/explore_representatives.py \
    --corpus <corpus dir> --cache <scratch dir>
```

The rules themselves are the `selection_rules` and `link_rules` functions in that
script — the single source of truth for both this page and the assets.
