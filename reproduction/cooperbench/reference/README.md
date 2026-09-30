# Frozen reference runtime

This directory is the canonical runtime selected by `relic run-cooper`.
It is isolated from the repository's older core-study runtime by subprocess
working directory and `PYTHONPATH`. Do not mix the two `environments` or SDK
namespaces in the same Python process.

`environments/org_env/cooperbench/` is the adapter and pair lifecycle;
`environments/org_env/runtime_adapter/` is the final organizational execution
layer; `agent_sdk/` and `society_core/` retain their Python import closure.
Some generic compatibility helpers are necessary to import that closure, but
no unrelated benchmark datasets, hidden suites, task repositories, logs,
checkpoints, private configuration, or obsolete launch batches are included.

Runtime Python content matches the preserved frozen source except for replacing
the internal treatment-name string with `relic_cooperbench_b3_two_agent` and
Git line-ending/trailing-whitespace normalization.
Schema names and upstream/library versions are not cosmetic labels and are
retained. The runbook and regression entrypoint are in the parent release.
