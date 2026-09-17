# Selected protocols (author assets pending)

This directory is intentionally a tracked placeholder. The release does not
include an author-reviewed selected protocol or lifecycle evidence set.

## Missing author assets

The authors need to supply the selected, sanitized public protocol records and
their reviewed lineage evidence: the linked historical run or cell, selected
trace and frame/tick where applicable, source proposal and governance event
identifiers, code/config binding, exporter/redaction version, reviewer,
selection rationale, and final artifact hash. Historical raw trajectories,
private reflections, provider messages, evaluator workspaces, and unreviewed
development logs are not public protocol assets.

## Expected contents and contract

A selected protocol is expected to be a public protocol/lifecycle record from
an author-supplied `relic-trace-v1` export, together with a versioned
provenance manifest. Its public fields and object references must conform to
the strict trace validator's typed protocol, proposal, and governance-event
contracts; this directory does not define a looser standalone replacement
schema. The records should make the proposal → adoption → use/enforcement →
amendment or retirement lineage reviewable without exposing private content.

## Source boundary

Only records projected, redacted, and reviewed from the named historical run
may be added here. A protocol synthesized in a local run or extracted from a
mock trace may be useful development evidence, but it is not a selected-paper
asset.

Do not manufacture protocol examples, lifecycle links, or provenance from
`artifacts/paper_results/`, the PDF, aggregate protocol counts, or count-only
runtime sidecars. Aggregate reporting cannot recover the individual objects,
sequence, selection review, or privacy boundary required for a selected case.
