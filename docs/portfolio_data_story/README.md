# Data story — what is actually in the training corpus

A deep read of the published `.pt` dataset: every stored attribute traced to the
code that produced it, every statistic measured over all 1,000 scenarios, and a
set of lightweight derived assets for later interactive use.

This is about the **data**, not the model. Nothing here contains a prediction or
an uncertainty estimate; for those see `scripts/verify_headline_results.py` and
`docs/figures/results/`.

| Document | Covers |
| --- | --- |
| [01_schema_and_provenance.md](01_schema_and_provenance.md) | Every `Data` attribute, its dtype and origin; the 31,635 vs 31,559 node count |
| [02_features.md](02_features.md) | All six features answered the same ten ways |
| [03_graph_topology.md](03_graph_topology.md) | The line graph: directedness, degrees, components |
| [04_intervention.md](04_intervention.md) | Footprints, magnitudes, which road classes are targeted |
| [05_spillover.md](05_spillover.md) | How far the response travels, and the controls behind that claim |
| [06_anomalies.md](06_anomalies.md) | Eight data-quality observations and what each does or does not affect |
| [assets/SCHEMA.md](assets/SCHEMA.md) | Field-by-field contract for the derived web assets |

## The four findings worth knowing

1. **65% of the traffic response lands on links that were never touched.** Mean
   |response| is 13.06 veh/h on intervened links against 2.91 elsewhere, but the
   untouched links are so numerous that they carry roughly two-thirds of the
   total effect.
2. **The response does not fade with graph distance.** It drops 4.4× at the first
   hop and is then flat out to eight hops. This survives its reachability control
   and holds for both directed and undirected traversal — see
   [05_spillover.md](05_spillover.md).
3. **The policy only ever touches three road classes** — primary, secondary and
   tertiary. Motorways are never intervened in any of the 1,000 scenarios, yet
   carry the second-highest mean response in the network.
4. **One column carries the whole experiment.** Five of six features, the
   positions and the topology are byte-identical in every scenario. Only
   `CAPACITY_REDUCTION` moves.

## Reproducing this

The corpus is not in the repository. Fetch the 20 batch files (2.44 GiB) from the
[`train-data-v1`](../../releases/tag/train-data-v1) release, then:

```bash
CORPUS=<where you unpacked the batches>
CACHE=<any scratch directory>

python scripts/data_exploration/explore_schema.py          --corpus $CORPUS --cache $CACHE
python scripts/data_exploration/explore_intervention.py    --corpus $CORPUS --cache $CACHE
python scripts/data_exploration/explore_spillover.py       --corpus $CORPUS --cache $CACHE
python scripts/data_exploration/explore_representatives.py --corpus $CORPUS --cache $CACHE
python scripts/data_exploration/build_web_assets.py        --corpus $CORPUS --cache $CACHE
```

The first script to run caches the arrays the others need, so the corpus is read
once. `build_web_assets.py` is deterministic — rebuilding produces byte-identical
files.

## Two corrections came out of this

Writing these documents turned up two errors in `docs/DATASET.md`, both recorded
in [`CORRIGENDUM.md`](../CORRIGENDUM.md) C8:

- **C8a** — `HIGHWAY == -1` means `pt` (public transport), not "unclassified"
  (which is code 9), and it is not the same set as the zero-capacity non-car links.
- **C8b** — the intervention footprint was quoted as 873–4,299 links (7.82% of the
  network) from a single batch file. Over the full corpus it is 306–11,305
  (12.06%).

Neither affects any model or reported metric. Both concern how the input data was
described.
