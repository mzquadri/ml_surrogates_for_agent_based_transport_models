# How far the response travels

The observation: **the response drops sharply at the first graph hop and is then
flat**. Because that is the strongest claim in this analysis, the method and its
controls come first.

## Method

| Choice | What was done |
| --- | --- |
| Distance | Multi-source breadth-first search in the line graph, seeded with the set of intervened links. Hop 0 *is* the intervened set |
| Direction | Computed **twice** — undirected (links adjacent if they meet at all) and directed (following the stored `from → to` orientation, i.e. traffic direction) |
| Unreachable | Nodes not reached within 8 hops are reported as their own band. They are never folded into the last hop or dropped |
| Aggregation | Per scenario, mean \|response\| within each hop band; then the unweighted mean of those per-scenario means, so a large-footprint scenario cannot dominate |
| Sample | 100 scenarios, `numpy.random.default_rng(0)`, reproducible |
| Topology | Identical in all 1,000 scenarios, so hop bands differ only through which links were intervened — the graph itself introduces no scenario-to-scenario variation |

## Result — undirected

| Hops | Mean \|response\| (veh/h) | Share of total \|response\| | Network reached (cumulative) |
| --- | --- | --- | --- |
| 0 (intervened) | **13.24** | 32.3% | 11.4% |
| 1 | 3.02 | 15.1% | 32.3% |
| 2 | 2.98 | 16.0% | 53.3% |
| 3 | 2.93 | 13.0% | 69.8% |
| 4 | 2.97 | 9.1% | 80.5% |
| 5 | 2.92 | 5.7% | 86.8% |
| 6 | 2.87 | 3.3% | 90.3% |
| 7 | 2.85 | 2.0% | 92.2% |
| 8 | 2.79 | 1.2% | 93.2% |
| unreachable | 0.80 | 2.5% | 6.8% |

One 4.4× fall at the first hop, then a profile that varies by about 0.2 veh/h
across seven further hops.

## Control 1 — is this just a small-diameter graph?

If two hops already covered the whole city, a flat profile would be trivial. It
does not. The reachability column above is the control: **one hop reaches 32% of
the network, two reach 53%, three reach 70%.** Hops 4–8 still add a further 23% of
links, and those links show the same response magnitude as links adjacent to the
intervention.

So the flatness is not an artifact of everything being nearby. Read the two columns
together: the response is spread across the network at a broadly constant
magnitude, rather than decaying with graph distance.

## Control 2 — does traversal direction change it?

| Hops | Undirected | Directed |
| --- | --- | --- |
| 0 | 13.03 | 13.03 |
| 1 | 3.03 | 3.05 |
| 2 | 3.00 | 2.98 |
| 3 | 2.97 | 2.97 |
| 4 | 3.01 | 3.00 |
| 5 | 2.98 | 3.01 |
| 6 | 2.86 | 2.95 |
| 7 | 3.07 | 2.97 |
| 8 | 3.04 | 3.06 |
| unreachable share | 6.7% | 12.3% |

Identical to within noise. Following traffic direction leaves twice as many links
unreachable, as expected on a partly one-way network, but does not change the
shape.

## Control 3 — aggregation

Per-scenario means are averaged with equal weight per scenario. Pooling all nodes
instead would let the largest-footprint scenarios dominate, and those are exactly
the scenarios with the *lowest* offsite share, which would bias the profile
downward at high hop counts. The equal-weight choice is the conservative one for
this claim.

## What it means

Traffic displaced by a capacity reduction does not seep outward through the street
grid, thinning as it goes. It reappears across the network at broadly the same
magnitude wherever it lands. Drivers reroute over whole journeys, so a road eight
turns away from a closure absorbs about as much as a road one turn away.

This is a good argument for the architecture actually used: two graph-transformer
layers give every link attention over distant parts of the graph, which a purely
local aggregation scheme on a max-degree-10 graph could not match at the same
depth.

## What it does not mean

This is an **observed association** in one simulated network under one
intervention family. Specifically, it is not:

- **causal proof of a mechanism** — nothing here identifies *why* the profile is
  flat, only that it is;
- **a claim about real traffic** — the response comes from MATSim runs on a 1%
  population sample, not from measured counts;
- **generalisable** — one city, one capacity-reduction family, 1,000 scenarios.
  A different network or policy type could behave completely differently;
- **independent of the graph definition** — "hops" are hops in *this* line graph.
  A different adjacency would give different bands.

The 2.5% of response landing on unreachable links is a reminder of the last point:
those links have no path to any intervention within 8 hops, yet still move,
because the coupling is behavioural rather than topological.

## Reproducing

```bash
python scripts/data_exploration/explore_spillover.py \
    --corpus <corpus dir> --cache <scratch dir> --scenarios 100
```

Machine-readable form, including both directions and the method block:
[`assets/spillover_decay.json`](assets/spillover_decay.json).
