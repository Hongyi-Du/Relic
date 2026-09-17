# Figure data (author assets pending)

This directory is intentionally a tracked placeholder. The repository contains
the canonical PDF-transcribed aggregate snapshot in
`artifacts/paper_results/`, but no author-published figure-data package.

## Missing author assets

For every released paper figure or panel, the authors need to publish the
reviewed machine-readable inputs and a provenance manifest. The manifest must
identify the paper figure/panel, metric definition and units, denominators,
study/model/arm/workload/seed or other allowed unit identifiers, source run or
scoring-ledger binding, transformation or plotting specification, code/config
binding, reviewer, and hashes of the supplied files. Any public release must
remain consistent with the documented evaluator, privacy, and legal boundaries.

## Expected contents and contract

No standalone figure-data schema has been published in this release. When the
authors provide one, it should be versioned and include a manifest that maps
each figure/panel to its exact input files, fields, allowed aggregation level,
and derivation. The package may contain only author-reviewed, releasable data;
it must not expose historical raw trajectories, provider receipts, checkpoints,
private agent state, evaluator workspaces, or the unavailable first-author
scoring ledger.

## Source boundary

Figure inputs must originate from the identified author-reviewed exports or
ledger-derived release package. `artifacts/paper_results/` is a final aggregate
snapshot, not a row-level or figure-input source.

Do not create pseudo-data, infer seed/cell values, reverse-engineer confidence
interval inputs, or rebuild figure tables from the PDF or aggregate snapshot.
Those summaries cannot establish the underlying units, transformations,
provenance, or release review required for figure data.
