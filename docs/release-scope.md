# Release scope

`relic` is the research artifact for reproducing the paper. Its canonical
release branch contains:

- the OrgEnv research environment and B0--B3 conditions;
- the ten `relic-main-v1` workloads;
- experiment launch, resume, retry, evaluation, and aggregation tooling;
- the paper's canonical transfer experiment;
- machine-readable aggregate values reported by the paper;
- the public Inspector and strict `relic-trace-v1` validation contract;
- validated native Linux and WSL2 setup paths; and
- release-focused tests.

The handoff defines the following planned long-lived extension branches with
deliberately narrow roles:

- `hci` is `main` plus the formative P2/P3 interface extension;
- `cooper` is `main` plus the CooperBench adapter and its tests; and
- `full-tests` is `main` plus relevant historical core regression tests.

Only `main` is present in the current local/remote ref snapshot. The extension
branches are therefore release topology still to be published, not deliverables
claimed by this checkout. Likewise, root Docker / Compose assets and the formal
evaluator image remain pending release inputs.

ProgramBench may be named only as a paper-reported aggregate result. Its tasks,
adapter, harness, scripts, and reproduction artifacts are not included in this
release. Complete historical trajectories, per-run author data, private agent
state, and development logs are also excluded.

The sanitized selected-paper-trace set remains an author-supplied release
asset and is not yet present. The repository does not reconstruct historical
cases from aggregates or count-only runtime sidecars.

The sibling [`relic-agent`](https://github.com/Hongyi-Du/Relic-Agent)
repository is the separate, benchmark-independent organization runtime.
