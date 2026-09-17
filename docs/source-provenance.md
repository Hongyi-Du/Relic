# Source provenance

This release is distilled from the private development repository
[`Hongyi-Du/SocioGenesis`](https://github.com/Hongyi-Du/SocioGenesis) at the
following frozen revision:

```text
hci-human-seat@dda36fb563375060ae8d8850300db01eb4695d29
```

The corresponding Git tree object is:

```text
d7276c13312b13d4a030d82c3accc253349d708d
```

Every compiled cell also records a SHA-256 tree digest of the actual Relic
Python runtime, canonical configs, dependency lock, and benchmark manifest.
This distinguishes upstream origin provenance from the exact curated release
bytes that executed the cell, including local modifications.

The development repository is a source pool, not an authority for reported
experimental facts. The final Relic paper is authoritative for experimental
design, model and workload identities, seeds, metrics, denominators, aggregate
results, and scientific claims. The Relic two-repository release handoff is
authoritative for repository scope and release acceptance.

The controlled source closure for the official paired reproduction path is:

- `tools/run_org_baselines.py`;
- `tools/org_inspector_replay.py`;
- `environments/org_env/runtime_adapter/live.py`;
- `environments/org_env/runtime_adapter/snapshot.py`; and
- `environments/org_env/runtime_adapter/replay_delta.py`.

Those files retain the source runner's four-arm fresh-process isolation,
paired-seed ordering, identity-bound checkpoint resume, LLM blackout/provider
circuit gates, and formal run-record gate. `relic.source_runner` is only a
thin 30-batch expansion of the paper's 120 cells; it delegates each batch back
to `tools/run_org_baselines.py` and does not use the legacy `cell_worker` as a
replacement executor.

The release adaptations are deliberately narrow: source imports of
`society_core` point to the equivalent public `relic.research` utilities; the
source B3 identity is `b3_full_sociogenesis` (with the earlier
`b3_relic_organization` accepted only for reading old local artefacts); the
excluded ProgramBench profile fails closed; and the target evaluator's
`RELIC_EVALUATOR_*` names are bridged from the source runner's frozen
`ORG_EVALUATOR_*` binding. The documented `OPENAI_*` and Relic default-header
variables are temporarily bridged into the source runner's allow-listed
`ORG_LLM_*` child environment without being serialized. The HCI-only persona graph builder is optional: the
source snapshot's existing graph representation is used when that frontend
module is absent. Because the release excludes the source synthetic default,
live/mock sessions bind the public frozen `mini_blobstore_v1` OSS pack instead.

Formal evaluator bindings remain an author-asset dependency. Relic does not
invent a container image digest, evaluator environment hash, or qualification
hash; `run-main` requires an explicit published per-pack mapping before any
formal source child or provider client can start.

The final transfer-specific closure is intentionally narrower and separate
from the core HCI closure. The canonical v2 bundle, closed compiler, six guard
bindings, full target-window landscape lock, and thin B2 Text/Exec adapter are
selectively derived from:

```text
origin/codex/transfer-v4-fixed-protocol@1a49b4821ab207b012d95057d2151c5cfaf1bc38
```

This is a module-level provenance reference, not a replacement for the HCI
core source or an authority for paper facts. No ProgramBench package, harness,
asset, or execution profile was copied from that revision; requests for that
profile continue to fail closed in this release. See [transfer.md](transfer.md)
for the exact canonical-v2 identity and final-paper target design.

For final-record repository provenance, `relic.research.repository_digest` is
the source revision's pure `repo_hash` closure (including the corresponding
repository-path policy and no-follow file reader). Both the source run-record
writer and Relic's final evaluator call it, so a record cannot silently use a
second digest contract. This compact closure intentionally excludes the
unrelated Code-Max, SocietyCore, and ProgramBench execution layers.

The public evaluator build closure is separately ported from the same HCI
revision: `.evaluator_image/Dockerfile`, `.cursor/build_evaluators.sh`,
`tools/print_evaluator_hashes.py`, and
`tools/preflight_organization_evaluator.py`. In Relic these become
`evaluator/Dockerfile`, the `relic evaluator-build`, `evaluator-hashes`, and
`evaluator-preflight` commands, and thin `tools/` compatibility wrappers. The
release adaptation removes source proxy/development-machine assumptions and
does not port the source ProgramBench-specific images or assets. Local image
IDs and local qualification hashes are deliberately marked non-paper; the
source main runner still rejects a local `sha256:...` image ID as an evaluator
binding.

Historical raw runs, private model transcripts, private memories,
developer-machine paths, credentials, ProgramBench reproduction assets,
NatureEnv, and obsolete SocioGenesis components are outside the release scope.

Third-party benchmark snapshots retain their upstream provenance and license
files. Their frozen bytes are not normalized by the outer repository.
