# relic-main-v1

This directory contains the ten frozen workloads used by the 240-run main
study. The workload IDs, names, order, and original pack IDs match Table 19 of
the paper.

Each pack contains its starter state, task specification, evaluator-only tests,
reference state, and provenance material. Evaluator-only means that those
assets must never enter an agent's context during a run; it does not mean they
are absent from the research release.

The five upstream transition workloads retain their upstream license files.
Do not normalize or edit any file under `packs/`: the pack tree digests in
`manifest.yaml` cover its exact bytes. A deliberate benchmark revision requires
a new benchmark version and a new qualification record.

Verify the frozen trees without contacting a model provider:

```bash
python -m relic.cli verify-benchmark
```

