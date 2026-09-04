# Artifact Provenance

## Canonical Boundaries

| Role | Repository/commit | Treatment |
|---|---|---|
| **Canonical repository** | `mzquadri/ml_surrogates_for_agent_based_transport_models` | The single public home for the code, thesis, documentation and data releases |
| Submitted artifact baseline | commit `4b95a3d8aca5929bb88b84bb7f7ae86c48e2f428`, formerly in `mzquadri/ml-surrogates-thesis` | *Historical repository, deleted after consolidation on 2026-09-04.* History preserved in the `provenance-v1` release |
| Audited evidence source | `mzquadri/ml_surrogates_for_agent_based_transport_models` at `fdb4ef0c9c736576ae34d5e331d8b66a7a6d877a` | Post-submission audit source |
| Pre-audit source baseline | Source commit `c3c84499029f310df4f05c76afa4e3e0d6b79de3` | Matches submitted PDF blob |

## Repository consolidation, 2026-09-04

Two repositories were retired and deleted after their content was migrated here and
verified. Neither is required to obtain any artifact, and no runnable command in this
repository refers to either.

| Retired repository | Role | What became of it |
|---|---|---|
| `mzquadri/ml-surrogates-thesis-data` | Companion data repository, per-trial artifact tree | Its 20 unique release assets (4.35 GB) are now the `thesis-data-v1` release here. Its tracked tree is archived as `ml-surrogates-thesis-data-tracked-tree.tar` on the same release, and was already duplicated inside `benchmarks-v1`. History bundled to `provenance-v1`. |
| `mzquadri/ml-surrogates-thesis` | Retired predecessor holding the submitted-artifact baseline | Its `train-data-v1` release was a byte-for-byte duplicate of the canonical one, so nothing needed migrating. Its 15-commit history — a lineage disjoint from this repository, including the `backup/pre-cleanup-main` tag at `3d9af642f13c12a8f64a3e97bacf5efce2079fb6` — is bundled to `provenance-v1`. |

Recovering either history:

```bash
gh release download provenance-v1 --repo mzquadri/ml_surrogates_for_agent_based_transport_models
git clone ml-surrogates-thesis-history.bundle recovered
```

The machine-verifiable record, including every asset's SHA256 and the size verification
between source and destination, is
[`migration/thesis_repo_consolidation.json`](migration/thesis_repo_consolidation.json).

Mentions of the two retired repositories elsewhere in this repository identify historical
provenance only. They are labelled as such, and no command depends on them.

## Immutable Submitted PDF

| Property | Value |
|---|---|
| Path | `document/main.pdf` |
| SHA-256 | `0ac5309d060cda53d82a05cc837136fe853e7f9dcbabd2f4fb4b4282a39bc97e` |
| Git blob | `1cb3bfdfb5d3126d8dc3cec361ab63f95de38306` |
| Size | 674,395 bytes |
| Submission date | May 15, 2026 |
| `document/` tree | 40 files; Git tree `f104db730eb1c8d228d913fde6545599da7795d5` |

The same PDF blob exists at the audited source's pre-audit commit `c3c8449`. A later regenerated
PDF in the audit branch has a different blob and is intentionally not migrated. The canonical PDF
and its submission-era LaTeX sources remain unchanged.

## Aggregate Audit Export

`analysis_outputs/` was generated on August 15, 2026 from trusted local artifacts available in
the audited source checkout. It contains aggregate JSON, CSV, Markdown, PNG, and SVG outputs only.
The export contract excludes row-level predictions, graph topology, pickle payloads, confidential
data-junction content, and absolute local paths.

`analysis_outputs/artifact_manifest.csv` describes the source artifacts present when the export
was generated. An `exists=True` value records source-checkout availability at generation time; it
does not claim the row-level artifact is included in this canonical repository. SHA-256 hashes
allow an authorized holder to verify those source artifacts before regenerating the aggregate
bundle.

The repository check compares the full submitted `document/` tree with baseline commit `4b95a3d`,
locks every aggregate export, and validates the path, size, and SHA-256 of every source artifact
required for regeneration. The local graph loader is verified before pickle-capable deserialization.

## Excluded Assets

The following are intentionally not migrated into the canonical public repository:

- raw MATSim outputs and local graph loaders;
- scalers and other pickle-capable preprocessing state;
- row-level `.npz` prediction and target arrays;
- PyTorch/PyG checkpoints;
- the post-audit regenerated thesis PDF;
- the untracked static `policy-dashboard/` prototype;
- local absolute paths or confidential junction content.

The Streamlit dashboard reads the safe committed bundle. It optionally enables a deterministic
scatter sample only when an authorized user restores a matching row-level artifact locally.

## Verification

Run `python scripts/check_repository.py`. The check verifies the submitted PDF hash and size,
aggregate schema and privacy contract, corrected target count, relative manifest paths, and
required documentation and dashboard assets.
