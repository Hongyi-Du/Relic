# Protocol lifecycle

This document describes the B3 organizational mechanism that is retained in
the paper runtime. It is not a claim that arbitrary natural-language rules are
universally enforceable.

```text
event / episode → reflection / wish → proposal → review and governance
       → adoption → active use → violation / enforcement evidence
       → amendment or retirement
```

## Lifecycle records

- A recurring friction can be represented by an episode, reflection, and wish.
- A proposal records its source, rationale, affected process, and governance
  status.
- Adoption creates an explicit protocol object with scope, trigger, responsible
  roles, rule/enforcement fields, success criteria, and a sunset/review policy.
- Active use, violations, and enforcement are distinct events. Enforcement
  evidence is tied to the governed object/state transition rather than inferred
  from prose alone.
- A later proposal may amend an adopted protocol; revisions preserve lineage.
  A protocol may also be deprecated/retired according to its lifecycle policy.

The registry and proposal manager reside under
`environments/org_env/backend/protocol/` and `environments/org_env/proposals/`.
B0--B2 deliberately disable institutionalization; only B3 enables the full
organization-to-institution path described in `configs/arms/b3.yaml`.

## Observation and interpretation

The Inspector can display public proposal, governance, protocol, and state-diff
records in an author-supplied `relic-trace-v1`. It does not expose private
reflection text or candidate-model reasoning. A temporal sequence such as
proposal → adoption → enforcement is descriptive evidence, not by itself a
causal estimate. Paper-level causal claims depend on the stated experimental
comparisons and limitations.

No author-selected sanitized historical trace is bundled in this release. Do
not manufacture lifecycle examples from aggregate results or count-only
sidecars; see [inspector.md](inspector.md) and
[KNOWN_RELEASE_GAPS.md](KNOWN_RELEASE_GAPS.md).
