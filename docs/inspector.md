# Relic Inspector

Relic Inspector is the paper repository's public organization observatory. It
renders a validated `relic-trace-v1` export and is not an omniscient debugger.
It never opens checkpoints, evaluator workspaces, provider messages, private
memories, private reflection text, or arbitrary files.

## Start a replay

Validate a selected trace first:

```bash
uv run relic replay --trace /path/to/selected-trace.json
```

Then start the Inspector and open `http://127.0.0.1:8765`:

```bash
uv run relic inspect \
  --trace /path/to/selected-trace.json \
  --mode replay
```

The equivalent Bash entrypoint is:

```bash
scripts/bash/start_inspector.sh \
  --trace /path/to/selected-trace.json \
  --mode replay
```

On Windows, keep the repository and trace in WSL2 and invoke the PowerShell
launcher or the Bash command in WSL. Open the same localhost URL in the Windows
browser. The PowerShell file only forwards arguments to the canonical Linux
CLI; it does not implement a second Inspector runtime.

## Available data and current asset boundary

The public code contains the Inspector, strict validator, and packaging assets.
The sanitized selected-paper-trace set has not yet been supplied by the authors.
Relic therefore does not bundle or claim a selected historical case at this
stage, and it does not synthesize one from paper aggregates.

Main-study cells currently write a small `relic-public-trace-v1` sidecar with
checkpoint-level counts. That sidecar supports safe status reporting but lacks
the event-level public objects required by the Inspector. It is intentionally
not accepted as `relic-trace-v1`; object identities and event histories are
never invented to make it displayable.

## Trace and privacy contract

The loader rejects a trace unless all of the following hold:

- the top-level schema is exactly `relic-trace-v1`;
- its SHA-256 covers the complete public payload as a self-consistency check
  (not an author signature or proof that a trace is a historical paper run);
- frame sequence is contiguous and ticks are non-decreasing;
- each frame contains at most one ordinary public event, so every published
  event is paired with one post-event organization snapshot;
- agents, tasks, proposals, and protocols are explicitly published;
- every privacy flag is present and `false`;
- nested public records use typed allowlists rather than arbitrary data
  containers;
- private events, blocked private fields (including camelCase variants),
  credential-like values, absolute local paths, duplicate JSON keys, non-finite
  numbers, inconsistent object references, and unsupported fields are absent;
- public decisions contain only selected action/object identifiers, never
  candidate features, utility scores, prompts, rationale, or evaluator-side
  policy audit.

Opaque reflection and wish identifiers may remain on a public proposal to show
lineage. They do not provide access to private reflection content.

## Interaction model

Timeline selection moves to an event-level post-event snapshot. The Object
Inspector and State Diff update to the same frame. Typed public relations can
link proposal → protocol → lifecycle event and permit reverse navigation from a
task or repository object when the trace publishes those relationships.

The panels cover Overview, Members, Tasks, Timeline, Episodes, privacy-preserving
Reflections, Proposals, Governance, Protocols, Artifacts, Repo / PR / CI,
Evaluation, Decisions, Object Inspector, and added/changed/removed State Diff.
Optional artifact, repository, or evaluation collections display as unavailable
when not published; absence is never interpreted as zero.

A future author-supplied selected trace should also ship with reviewed provenance
metadata outside the trace digest: source run/cell, code and configuration
binding, exporter/redaction version, reviewer, case-selection rationale, and the
final trace hash. The in-file hash alone cannot establish those facts.

The view is descriptive evidence. Recorded sequence, object lineage, and state
differences do not establish causal attribution to a protocol, member, or
mechanism; causal claims require the paper's experimental comparisons and
their stated limitations.

Any HCI-facing replay remains an interface demonstration or formative artifact
unless separately supported by reviewed participant-study evidence. It is not a
powered participant evaluation.

## Replay, live, and network behavior

`replay` validates and caches the file once. Later filesystem changes do not
alter the displayed trace.

`live` revalidates the same public file and accepts only append-only updates for
the same run. A partial write, digest mismatch, shortened trace, changed schema,
or rewritten frame is rejected; the server continues serving the last verified
snapshot and reports degraded health. The paper runtime does not currently
publish event-level live traces, so this mode is for a conforming external
exporter rather than the count-only cell sidecar.

Native and WSL launches bind `127.0.0.1` by default. The fixed HTTP surface is
`/`, `/index.html`, `/app.css`, `/app.js`, `/api/health`, and `/api/trace`, with
no CORS allowance, loopback Host-header validation, and restrictive security
headers. Because there is no authentication, a non-loopback bind is rejected
unless `--allow-remote` is supplied explicitly. Remote mode accepts literal IP
Host headers while continuing to reject arbitrary DNS names, including a
rebinding domain. Put an authenticated reverse proxy in front of the Inspector
if a named remote host is required.
