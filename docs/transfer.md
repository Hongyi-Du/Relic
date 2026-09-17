# Final transfer comparison

The final-paper transfer comparison is a separate, source-backed target
matrix. It does not recreate or relabel a historical result. Its 60 new target
runs are ten frozen workloads × three seeds × two arms, all on the fresh
eight-role B2 backbone with `gpt-5.6-terra`:

| New target arm | What target members receive | What differs |
| --- | --- | --- |
| Text | the six canonical rule summaries in direct prompt state | no executable protocol bindings |
| Exec | the same six rule summaries plus their sealed machine bindings | the bindings authorize mutations |

The Fresh comparator is the existing B2 main-study result for the matching
workload and seed. `run-transfer` therefore never schedules a new “Fresh”
target run.

Both arms use the bundled, validated
`environments/org_env/data/capability_bundles/canonical_v2.json` artifact. Its
semantic SHA-256 is
`a627adfcf6c299d0fabb185345bbc01106ea6901380aa0fbfee37c830ad8cdb6`.
It is a closed `org_capability_bundle_v2` / `org_protocol_bindings_v2` package,
not a source-run export selected at execution time. Check it without contacting
a model:

```bash
uv run python tools/build_canonical_v2_bundle.py --check
```

## Fixed target landscape

The entire target window is fixed after injection. Target code suppresses
protocol proposal, support/opposition/follow/amend candidates; rejects direct
protocol or policy-repair proposal/adoption attempts; and skips automatic
harm-detection and institution synthesis. Existing product delivery work is
not disabled.

Exec binding dispatch is closed to these six guard/action routes:

| Guard | Protected action(s) | Block reason |
| --- | --- | --- |
| `issue_owner_assigned` | `open_pr` | `issue_owner_required` |
| `branch_owned_by_actor` | `open_pr` | `branch_owner_mismatch` |
| `unchanged_failed_ci_not_retried` | `run_ci`, `ci_test` | `unchanged_failed_ci` |
| `current_ci_attested` | `merge_pr` | `current_ci_attestation_required` |
| `independent_review` | `approve_pr`, `review_pr`, `formal_pr_review`, `merge_pr` | `independent_review_required` |
| `release_gate_covered` | `publish_product_release` | `release_gate_coverage_required` |

The central pre-action authorizer is called by agent actions, automatic PR/CI/
review/merge and release paths, and delivery repair. A malformed, altered, or
missing compiled binding fails closed. Text has matching prose only; it does
not receive those executable fields.

## Plan or execute

Dry-run planning writes all source B2 case plans for the requested safe subset
without a provider call or condition subprocess:

```bash
uv run relic run-transfer \
  --dry-run \
  --output-root outputs/transfer-v2 \
  --manifest outputs/transfer-v2/transfer_manifest.json \
  --workload w01 --seed 1401
```

Omit the workload/seed filters to materialize the full 60-target-run plan. The
thin adapter delegates every target to `tools/run_org_baselines.py` with
`--cases b2`; it does not use the legacy per-cell worker and it does not copy a
ProgramBench executor.

Formal execution requires the authors' reviewed per-pack evaluator bindings,
with a digest-pinned container image and the matching environment and
qualification hashes. Once those are available, resume the frozen plan:

```bash
uv run relic run-transfer \
  --manifest outputs/transfer-v2/transfer_manifest.json \
  --resume --max-parallel 1 \
  --evaluator-bindings author-published-evaluator-bindings.json
```

The current release deliberately has no such binding file. A non-dry command
therefore fails before it starts a source child or sends a provider request;
local evaluator hashes or image IDs are not promoted to paper evidence.

New local artifacts produced after a binding becomes available are new
reproduction artifacts. They cannot by themselves establish or replace the
paper's historical raw results.

## Provenance and scope

The B2 backbone and canonical runner come from the release's HCI source
closure. The v2 bundle, compiler, fixed-landscape controls, and compiled-guard
integration are a selective module-level port from
`origin/codex/transfer-v4-fixed-protocol@1a49b4821ab207b012d95057d2151c5cfaf1bc38`.
That revision was used only for these transfer components. ProgramBench
packages, harnesses, assets, private source history, model transcripts, and
historical raw results are not included.
