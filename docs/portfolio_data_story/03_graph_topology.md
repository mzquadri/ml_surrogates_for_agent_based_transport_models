# The graph

## It is a line graph

The road network is inverted before the model sees it. In the usual road graph,
junctions are nodes and roads are edges. Here it is the other way round:

- **a node is a road link**
- **an edge means two links meet**

Built by PyG's `LineGraph()` transform in `process_simulations_for_gnn.py`, applied
to the directed `from_node → to_node` network. This is why the prediction target
is per-link: the thing being predicted is a property of a road segment, so a road
segment is the natural node.

| Property | Value |
| --- | --- |
| Nodes (rows in `x`) | 31,635 |
| Directed edge entries | 59,851 |
| Unique undirected pairs | 53,955 |
| Self-loops | 766 |
| Degree: min / max / mean / median | 0 / 10 / 3.78 / 4 |
| Isolated nodes | 76 |
| Connected components | 121 |
| Largest component | 29,327 nodes (92.70%) |
| Singleton components | 78 |

## The graph is directed

`edge_index` holds 59,851 entries but only 53,955 unique undirected pairs, so it is
**not** a symmetrised graph. Direction is meaningful: an edge records that traffic
can pass from one link into the other, following the original network's
`from_node → to_node` orientation. Roughly 5,900 entries have a matching reverse
edge; the rest are one-way.

This matters for anything that traverses the graph. Both directions are reported
wherever traversal is involved — see [05_spillover.md](05_spillover.md), where the
result is shown to hold either way.

## Degree distribution

| Degree | Links | Share |
| --- | --- | --- |
| 0 | 76 | 0.24% |
| 1 | 33 | 0.10% |
| 2 | 6,065 | 19.17% |
| 3 | 7,197 | 22.75% |
| **4** | **9,901** | **31.30%** |
| 5 | 4,839 | 15.30% |
| 6 | 2,658 | 8.40% |
| 8 | 128 | 0.40% |
| 10 | 1 | 0.00% |

Degree 4 dominates, which is what a street grid produces: a link running between
two ordinary junctions meets two links at each end. Degree 2 is a link in a chain;
degree 6 and above are the complex intersections.

The maximum is 10 — no super-hub. Every link has a small, bounded neighbourhood,
so message passing has to travel many steps to move information across the city.
That is directly relevant to the architecture: two graph-transformer layers give
the model long-range attention that pure local aggregation over a max-degree-10
graph could not provide in the same depth.

## The network is not one piece

121 connected components. The largest holds 92.70% of links, so the bulk of Paris
is one connected mass, but there are 43 non-trivial fragments plus 78 single
isolated links.

The 76 degree-0 nodes are why `Data.num_nodes` reads 31,559 rather than 31,635 —
see [01_schema_and_provenance.md](01_schema_and_provenance.md).

Isolated and fragmented links still carry features and a target, and the model
still emits a prediction for them. For those links the prediction rests entirely on
their own features, because no neighbour information can reach them.

## 766 self-loops

A self-loop is a link whose start and end junction are the same — a roundabout
carriageway or a loop road, which the line-graph transform records as a node
adjacent to itself. They are legitimate network geometry, not corruption, but any
traversal code has to tolerate them.

## Topology is fixed

`edge_index` is **byte-identical in all 1,000 scenarios**, as is `pos`. Reducing a
road's capacity does not remove it from the network — the road is still there,
just narrower. So the graph is a constant, and the only thing that varies from
scenario to scenario is one column of `x`.

For anyone building on this, that is a useful guarantee: the geometry and the
adjacency can be loaded **once** and reused for every scenario, and `link_row`
means the same physical road everywhere.
