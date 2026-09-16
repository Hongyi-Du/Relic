# Release scope

`relic` is the research artifact for reproducing the paper. Its canonical
release branch contains:

- the OrgEnv research environment and B0--B3 conditions;
- the ten `relic-main-v1` workloads;
- experiment launch, resume, retry, evaluation, and aggregation tooling;
- the paper's canonical transfer experiment;
- machine-readable aggregate values reported by the paper;
- a small set of sanitized public traces and the public Inspector;
- Docker, WSL2, and Linux setup paths; and
- release-focused tests.

The long-lived extension branches have deliberately narrow roles:

- `hci` is `main` plus the formative P2/P3 interface extension;
- `cooper` is `main` plus the CooperBench adapter and its tests; and
- `full-tests` is `main` plus relevant historical core regression tests.

ProgramBench may be named only as a paper-reported aggregate result. Its tasks,
adapter, harness, scripts, and reproduction artifacts are not included in this
release. Complete historical trajectories, per-run author data, private agent
state, and development logs are also excluded.

The sibling [`relic-agent`](https://github.com/Hongyi-Du/Relic-Agent)
repository is the separate, benchmark-independent organization runtime.

