# Next-stage validation

Scope: the supplied next-stage specification, the local Relic paper PDF, and
this repository. Historical handoff documents do not add release gates. Work
is kept on a local Git branch; this stage does not push or publish a release.

## Preserved experiment

The main study remains two model labels × ten workloads × three seeds × four
conditions: 240 runs of 336 ticks. Arm definitions, paired ordering, workload
content, evaluator contract and scoring semantics are preserved. Transfer Text
and Exec retain their frozen content and do not gain new formation or revision.

## Changed entrypoints

Normal main, transfer and compatible cell execution can use the public local
evaluator without a binding. Local preflight and observed evaluator metadata
remain recorded. Explicit bindings are validated and recorded; only explicit
strict reproducibility requires digest, platform and qualification identities.
Runtime model overrides and operator environment variables use the user's own
gateway/model names. Dry planning requires neither credentials nor a model call.

The README documents source batch receipts (`manifest.json`) and aggregated
experiment records (`experiment_runs.json/jsonl`) separately from the legacy
user-run metric aggregator. Historical paper aggregates are not recomputed.

## Evidence

On 2026-09-17, the final suite passed **576 tests, with four skips** in 278.47
seconds. Three skipped tests require explicit live-provider opt-in; one requires
an explicit Docker evaluator build. The sole warning is a third-party Starlette
deprecation. FastAPI TestClient checks need the local test execution permissions;
the restricted environment could hang in that test, while the same isolated
check passed immediately with the appropriate local permissions.

`ruff check relic environments tests tools scripts` and `git diff --check`
passed. The optional-evaluator/source-runner subset passed 27 tests, including
resolving an unconfigured dry-plan deployment name before its first execution
and retaining model identity after execution begins.

A clean local clone of implementation commit `c1efe48d`, installed with
`uv sync --extra dev --frozen`, passed:

- all ten frozen benchmark integrity checks;
- core environment checks and mock smoke;
- the full Claude main-study dry plan without credentials or bindings;
- the selected W01/1401 Text and Exec transfer dry plan without bindings;
- all 18 optional-evaluator tests in the installed checkout;
- a real local evaluator qualification and scoring of the public
  `mini_blobstore_v1` reference fixture (candidate pass rate 1.0, five causal
  fixes, zero unresolved oracles).

An additional short source-runner probe on the preceding implementation snapshot
`33e42a75` started the actual source process without a binding and wrote native
batch receipts, final-evaluation evidence, and `experiment_runs.json/jsonl`.
Its mock treatment was correctly rejected with `llm_treatment_has_no_calls`;
it is evidence of source/evaluator plumbing, not a completed formal B0 cell.
The reference-fixture score is also an evaluator check, not a model result.

No paid model call, Docker certification, or full 240-run paper study was
performed. Normal local evaluator scores retain `formal_claim_ready: false`
because no historical/container reproducibility claim is being made.
