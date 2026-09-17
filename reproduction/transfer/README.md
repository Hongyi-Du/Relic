# Transfer reproduction

Paper scope: the final fresh-B2 Text versus Exec transfer comparison. The
canonical v2 bundle and its six guards are part of the checked-in release;
the entrypoint is `relic run-transfer`.

`relic run-transfer --dry-run` creates the 60 new target plans (10 workloads ×
3 seeds × 2 target arms). Fresh is the existing B2 main-study reference, not a
third newly generated transfer arm. See [`../../docs/transfer.md`](../../docs/transfer.md)
for the locked protocol and provenance contract.

Actual source runs require the same author-published evaluator bindings as the
main study. A dry plan, mock artifact, or local evaluator diagnostic is not a
replacement for historical transfer outputs.
