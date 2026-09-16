# Local evaluator build context

This directory is the public, source-derived build context for Relic's OSS
time-machine evaluator. It is ported from
`SocioGenesis/hci-human-seat@dda36fb563375060ae8d8850300db01eb4695d29`,
`.evaluator_image/Dockerfile`.

Build and run a network-isolated import smoke test from a checkout:

```bash
uv run relic evaluator-build --smoke
```

The resulting image ID is local build evidence only. It is not a published
registry digest, does not establish the evaluator binding used in the paper,
and cannot be supplied to `relic run-main --evaluator-bindings`. That command
accepts only author-published `...@sha256:<64-hex>` bindings.

The context intentionally contains no benchmark or evaluator assets. At
evaluation time, Relic materializes the selected frozen pack and mounts its
evaluator-owned hidden suite read-only through the existing execution policy.
ProgramBench images and assets are intentionally absent from this release.
