# Selected traces (author assets pending)

This directory is intentionally a tracked placeholder, not a trace corpus. No
author-selected, sanitized historical trace is included in this release.

## Missing author assets

For each selected case, the authors need to publish a sanitized
`relic-trace-v1` export plus reviewed provenance metadata: the source run or
cell identity, paper section or figure reference, code and configuration
binding, exporter/redaction version, reviewer, case-selection rationale, and
final trace SHA-256. Historical raw trajectories, provider receipts, evaluator
workspaces, checkpoints, and private agent state remain outside this public
directory.

## Expected contents and contract

When supplied, each trace must be a complete, digest-bound
`relic-trace-v1` JSON file accepted by `relic replay`. In particular, it must
use the public typed allow-lists and declare every privacy flag as `false`.
The provenance record belongs alongside the trace and is evidence about the
author-reviewed export; the in-file digest alone does not establish that a
trace is a historical paper case.

## Source boundary

Only an author-reviewed export/redaction of the identified historical run may
be placed here. A new user run, the local mock trace, and the count-only
`relic-public-trace-v1` runtime sidecar are not selected-paper traces.

Do not add representative traces, reconstruct event histories or object IDs,
or derive a trace from `artifacts/paper_results/`, the PDF, aggregate counts,
or other summary outputs. Those sources cannot recover the required public
objects, chronology, privacy review, or provenance.
