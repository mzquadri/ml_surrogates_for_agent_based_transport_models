# The intervention

One column, `CAPACITY_REDUCTION`, holds the entire experiment. This is what is in
it.

## Footprints

| Property | Value |
| --- | --- |
| Scenarios | 1,000 |
| Links intervened, min | **306** (0.97% of the network) |
| Links intervened, median | 3,317 |
| Links intervened, mean | 3,814 (**12.06%**) |
| Links intervened, max | **11,305** (35.74%) |
| Unique intervention vectors | **1,000 of 1,000** |
| Unique intervention masks | 1,000 |

Every scenario is distinct — no duplicates, and no two scenarios share even the
same *set* of intervened links. The footprint varies by a factor of 37 between the
smallest and largest policy.

> Earlier documentation quoted 873–4,299 links and 7.82%. Those came from the first
> batch file only. Corrected in [`CORRIGENDUM.md`](../CORRIGENDUM.md) C8b.

## Which links are ever touched

| | Links | Share |
| --- | --- | --- |
| Never intervened in any scenario | **20,330** | 64.26% |
| Intervened at least once | 11,305 | 35.74% |
| Intervened in every scenario | 0 | 0% |

Among links that are ever intervened, the mean is 337 scenarios and the maximum is
660 — so no link is a constant target, but some are heavily reused.

## Magnitudes

28 distinct non-zero values, **all negative** — capacity is only ever removed,
never added.

| Reduction | Occurrences | Share |
| --- | --- | --- |
| −240 | 926,179 | 24.29% |
| −1,200 | 720,789 | 18.90% |
| −400 | 661,083 | 17.33% |
| −600 | 506,252 | 13.27% |
| −800 | 352,508 | 9.24% |
| −1,800 | 236,831 | 6.21% |
| −480 | 159,732 | 4.19% |
| −2,400 | 89,793 | 2.35% |

All are round numbers, and they cluster at multiples of 240 and 400. That is the
signature of removing whole traffic lanes rather than scaling capacity by a
fraction — a lane of an urban street being worth a few hundred vehicles per hour.

## Only three road classes are ever targeted

| Code | Class | Links | % ever intervened |
| --- | --- | --- | --- |
| 1 | primary | 5,295 | **85.0%** |
| 2 | secondary | 4,328 | **79.1%** |
| 3 | tertiary | 3,792 | **89.1%** |
| 0 | trunk / motorway_link | 933 | 0.0% |
| 4 | residential | 11,796 | 0.0% |
| −1, 5, 6, 7, 8, 9 | pt, living_street, pedestrian, service, construction, unclassified | 5,491 | 0.0% |

Every class except primary, secondary and tertiary is untouched in **all 1,000
scenarios**. The policy family is specifically about reallocating road space on
the arterial and distributor network — not motorways, not residential streets.

This is a structural fact about the experiment that was not previously recorded
anywhere, and it bounds what the trained surrogate can be expected to generalise
to.

## How much do scenarios overlap?

Pairwise Jaccard similarity of intervention masks, over 200 sampled scenarios:

| Statistic | Value |
| --- | --- |
| Mean | 0.185 |
| Median | 0.166 |
| 95th percentile | 0.470 |
| Maximum | 0.962 |
| Pairs with J > 0.9 | 1 |
| Pairs with J = 0 (disjoint) | 86 |

Typical pairs share about a sixth of their intervened links. A handful are almost
disjoint, and exactly one pair is nearly identical. The design is a broad sweep
over overlapping subsets of the arterial network rather than a set of cleanly
separated treatments.

## What the intervention produces

| Quantity | Value |
| --- | --- |
| Mean \|response\| on intervened links | 13.06 veh/h |
| Mean \|response\| on untouched links | 2.91 veh/h |
| Ratio | 4.5× |
| **Share of total \|response\| landing off the intervened links** | **64.9%** |
| corr(links intervened, total \|response\|) | +0.873 |
| corr(capacity removed, total \|response\|) | +0.870 |

The intervened links move most per link, but they are a minority of the network,
and **roughly two-thirds of the total traffic effect lands somewhere the policy
never touched**. That is the single most important property of this dataset: the
thing being predicted is mostly *displacement*, not local reduction.

Where that displaced traffic goes is the subject of
[05_spillover.md](05_spillover.md).
