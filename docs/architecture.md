# Architecture

Relic is a paper-reproduction artifact arranged around one canonical execution
path, explicit public/private boundaries, and narrow extension branches.

```text
canonical YAML + frozen benchmark
            ↓
       relic CLI
            ↓
source-backed paired runner ──→ B0/B1/B2/B3 fresh condition processes
            ↓                              ↓
     user-local manifests             private runtime/checkpoint material
            ↓                              ↓
public status / trace projection ← allow-listed public exporter
            ↓
      Inspector (`relic-trace-v1`)
```

## Core layers

- `relic/` supplies the public CLI, path policy, manifests, paper-result
  renderer, evaluator boundary, trace validation/export, and Inspector server.
- `environments/org_env/` contains the retained paper OrgEnv runtime, source
  paired-runner seams, protocol/governance mechanisms, and research substrate
  integration.
- `configs/` and `benchmarks/relic-main-v1/` freeze the study identity and
  workload bytes.
- `evaluator/` provides a local source-closure build and qualification path;
  it does not manufacture the formal evaluator binding.
- `artifacts/paper_results/` holds the canonical PDF-transcribed aggregate
  snapshot. It is never recomputed from unavailable historical raw runs.

## Public/private boundary

Checkpoints, evaluator workspaces, model messages, and private agent memory are
not Inspector inputs. The Inspector accepts only strict, hashed
`relic-trace-v1` data through typed allow-lists and rejects private fields,
credential-like values, local paths, and arbitrary runtime dumps. See
[inspector.md](inspector.md).

## Extension boundary

`main` remains the core source of truth. The `hci` ref adds the P2/P3 HCI
extension and `cooper` adds the CooperBench adapter; neither should fork core
behavior. The future `full-tests` ref may add sanitized core regression depth
after a canonical release tag, but not a second implementation. See
[release-scope.md](release-scope.md#branch-topology-and-sync-policy).

## Formal evaluation boundary

The source runner refuses a non-dry formal run until it receives an explicit,
per-pack, digest-pinned evaluator binding. This is intentional: a locally
built evaluator image is useful source-closure evidence but is not a paper
binding. See [evaluator.md](evaluator.md) and [KNOWN_RELEASE_GAPS.md](KNOWN_RELEASE_GAPS.md).
