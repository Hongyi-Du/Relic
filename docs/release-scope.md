# Release scope

`relic` is the research artifact for reproducing the paper. Its canonical
release branch contains:

- the OrgEnv research environment and B0--B3 conditions;
- the ten `relic-main-v1` workloads;
- experiment launch, resume, retry, evaluation, and aggregation tooling;
- the paper's canonical transfer experiment;
- machine-readable aggregate values reported by the paper;
- the public Inspector and strict `relic-trace-v1` validation contract;
- validated native Linux, WSL2, and core Docker / Compose setup paths; and
- release-focused tests.

## Branch topology and sync policy

The prepared release refs use the following deliberately narrow roles:

- `hci` is `main` plus the formative P2/P3 interface extension;
- `cooper` is `main` plus the CooperBench adapter and its tests; and
- `full-tests` will be `main` plus relevant historical core regression tests.

The locally prepared `hci` and `cooper` refs are deliverable extensions of the
same `main` core. `full-tests` is deliberately not created before a canonical
release tag and a sanitized, source-compatible historical-suite selection
exist. A release owner must publish the intended refs and tag; this document
does not claim that a local branch is already a remote release.

Generic bug fixes belong in `main` first. HCI-only changes belong in `hci`,
Cooper-only changes in `cooper`, and core regression-only additions in the
future `full-tests`. Extension refs must periodically merge `main` and must not
maintain a divergent copy of core behavior. Root Docker / Compose assets cover
core environment checks, mock smoke, dry-run planning, mounted outputs, and the
Inspector. The formal evaluator image and reviewed nested evaluator integration
remain pending release inputs, so Docker formal execution continues to fail
closed.

ProgramBench may be named only as a paper-reported aggregate result. Its tasks,
adapter, harness, scripts, and reproduction artifacts are not included in this
release. Complete historical trajectories, per-run author data, private agent
state, and development logs are also excluded.

The public source-derived evaluator Dockerfile and local qualification tools are
included to verify the OSS evaluator closure. They accept only `relic-main-v1`
pack IDs and explicitly reject ProgramBench. They do not provide the still
missing author-published evaluator image digest or paper qualification bindings;
formal reproduction remains fail-closed until those assets arrive.

The sanitized selected-paper-trace set remains an author-supplied release
asset and is not yet present. The repository does not reconstruct historical
cases from aggregates or count-only runtime sidecars.

The sibling [`relic-agent`](https://github.com/Hongyi-Du/Relic-Agent)
repository is the separate, benchmark-independent organization runtime.
