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

The experiment extensions keep their existing roles:

- `hci` is `main` plus the formative P2/P3 interface extension;
- `cooper` is `main` plus the CooperBench adapter and its tests.

Ordinary tests remain in each repository. No `full-tests` branch, release tag,
or remote publication ceremony is required to run or validate the code.

Generic bug fixes belong in `main` first. HCI-only changes belong in `hci`,
Cooper-only changes in `cooper`. Extension refs must not
maintain a divergent copy of core behavior. Root Docker / Compose assets cover
core environment checks, mock smoke, dry-run planning, mounted outputs, and the
Inspector. Normal reproduction uses the public host evaluator by default; the
formal evaluator image and reviewed nested evaluator integration remain
optional release inputs for strict container claims.

ProgramBench may be named only as a paper-reported aggregate result. Its tasks,
adapter, harness, scripts, and reproduction artifacts are not included in this
release. Complete historical trajectories, per-run author data, private agent
state, and development logs are also excluded.

The public source-derived evaluator Dockerfile and local qualification tools are
included to verify the OSS evaluator closure. They accept only `relic-main-v1`
pack IDs and explicitly reject ProgramBench. They do not provide the still
missing author-published evaluator image digest or paper qualification bindings.
Operator-supplied digest/platform/hash bindings are checked only when strict
reproducibility is explicitly selected; they need not be historical author assets.

The sanitized selected-paper-trace set remains an author-supplied release
asset and is not yet present. The repository does not reconstruct historical
cases from aggregates or count-only runtime sidecars.

The sibling [`relic-agent`](https://github.com/Hongyi-Du/Relic-Agent)
repository is the separate, benchmark-independent organization runtime.
