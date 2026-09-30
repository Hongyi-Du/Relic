# Evaluation method and execution-layer deviations

## Benchmark snapshot and treatment

CooperBench is pinned to **v0.0.29**, commit
`4913c4ebb84d2606cdb5628936b88529f3e181df`; the dataset snapshot is
`b612b1a35af722751454813d9e5a7888f065fc9e` of
[`CooperBench/cooperbench-dataset`](https://huggingface.co/datasets/CooperBench/cooperbench-dataset).
The selection contains 652 feature pairs across 30 task instances and 12 repositories.
Task-specific official Docker images are selected by the pinned upstream image
mapping; there is no single replacement task image. We did not replace the
official task tests or gold patches with agent-generated probes.

RELIC B3-2 is a two-member extension, not the canonical eight-member B3 condition.
The two members receive one public feature brief each, implement in isolated
workspaces, perform reciprocal review, and return an identical joint patch.
Internal probes and peer approvals guide delivery; only the official evaluator
produces official PASS/FAIL. The final reference is available on the
[`cooper` branch](https://github.com/Hongyi-Du/Relic/tree/cooper/reproduction/cooperbench).
Earlier attempts used earlier incremental fixes; publishing the final source
does not claim identical source bytes for every historical trajectory.

## Agent and model attribution

The reported backend family is **Claude Opus 4.6**, based on experiment-owner
confirmation, not independently inferred from gateway aliases. The normalized
release omits private endpoints, routing aliases, keys, and operational repair
labels; raw provenance is retained separately. Some authorized routes added
provider-side system context. Normalizing the model-family label does not imply
identical wire configuration or prompt context across all attempts.

## Changes to the external evaluator's execution layer

The actual [execution patch](../../tools/patches/cooperbench-execution.patch) changes
only two upstream files. The evaluator was not byte-for-byte unmodified.

| File / change | What changed | Evaluation boundary |
|---|---|---|
| `eval/backends/docker.py`: Docker timeout cleanup | Avoid the executor context manager waiting again on a timed-out Docker SDK call; kill only that sandbox container, stop waiting for its worker thread, and reject subsequent reuse of the terminated sandbox. | A timeout is an execution failure, not a PASS. No unrelated container is killed. |
| `eval/sandbox.py`: large-patch transport | Encode the same patch as base64, write in 65,536-character chunks, and decode/append into the same patch file. Quote the target path; handle an empty patch explicitly; fail on write errors. | No candidate patch edit or scoring change. This avoids oversized single shell arguments. |
| `eval/sandbox.py`: configurable timeout | Replace the literal default with `COOPERBENCH_EVAL_TIMEOUT_SECONDS`, defaulting to the upstream 600 seconds when unset. | Time allowances can affect whether evaluation finishes. This is a disclosed execution-setting deviation, not a claim of identical timeout behavior. The default is not evidence of every historical run's configured value. |

**No test suite, task acceptance criterion, or PASS/FAIL decision rule was
relaxed by this patch.** `both_passed` still requires both feature evaluations
to pass. Test-result parsing, patch filtering, merge rules, and merged/solo
scoring-function bodies remain unchanged. The official handling of
byte-identical joint patches was already upstream behavior, not a RELIC override.
Agent-side runtime/probe fixes are separate from these evaluator changes;
internal probe success is never substituted for an official verdict.

## Reporting and internal method failures

Raw evaluator outcomes are 392 PASS / 221 FAIL / 39 without an official verdict.
The owner's analysis is 371 PASS / 100 FAIL / 181 **proposed** BROKEN exclusions.
The 100 failures comprise 98 official evaluator FAIL and 2 internal method
failures without an official verdict. The 181 exclusions have per-pair QA and are submitted
for author review, not represented as official author-approved exclusions.

The two DSPy cases failed to complete an evaluable submission through our own
peer-probe/review workflow. This workflow is part of our method, so both count
as method failures, not benchmark defects. Their analysis status is `FAIL`,
with `failure_type=INTERNAL_PROBE_WORKFLOW_FAILURE` and
`failure_verdict_source=METHOD_ACCOUNTING`. Their raw evaluator status remains
`NO_OFFICIAL_VERDICT`; an official FAIL is not manufactured. This accounting
does not claim that the unsubmitted implementations would have failed official tests.

The primary retained-task method success rate is **371/471 (78.77%)**, including
the two internal failures in its denominator. The evaluator-only subset,
371/469 (79.10%), is supplementary and excludes those two failures. Neither is
an official full-652 score or an author-endorsed exclusion-adjusted score.
LLM-call failure percentages are diagnostic only; no 5% exclusion rule is used.
This correction changes classification and documentation only: no rerun,
synthetic trajectory, substituted patch, or rewritten raw official verdict.
