# Known release gaps

Status date: 2026-09-17. These are explicit release boundaries, not values to
infer, substitute, or fabricate locally.

## Author-published formal-run assets

- Per-pack author-published evaluator bindings are absent. Each needs a
  digest-pinned registry image, `linux/amd64` platform, evaluator environment
  hash, and qualification-plan hash. A local image ID and locally observed
  hashes must not be promoted to a paper binding.
- The deployed Claude gateway alias required by
  `RELIC_CLAUDE_OPUS_4_6_MODEL` is absent. Its credential and gateway route
  are operator/author assets and were not invented for a dry plan.
- Sanitized selected-paper `relic-trace-v1` files are not bundled. The local
  mock trace in the reproduction report is new and is not a selected paper
  case.
- Historical raw trajectories, provider receipts, evaluator outputs, and the
  first-author leaf-case / contract scoring ledger are not present. The
  aggregate snapshot cannot reconstruct them.

## CooperBench external boundary

- The required external CooperBench checkout must be exactly
  `4913c4ebb84d2606cdb5628936b88529f3e181df`, with its CLI installed.
- The required external dataset snapshot is
  `CooperBench/cooperbench-dataset@b612b1a35af722751454813d9e5a7888f065fc9e`
  and must contain `subsets/b3_v133_combined48_b001.json`.
- Upstream task images, Redis, the official evaluator, a configured
  OpenAI-compatible Claude gateway, and the historical task-image digest
  ledger are external prerequisites. None was substituted in the final run.
- Historical 29/48 raw Cooper artifacts remain unavailable and are not
  reconstructed from the reported aggregate.

## Legal and release publication

- A root `LICENSE` file is absent in the Cooper release check and remains a
  release/legal gap until an authoritative license is supplied for Relic.
- The handoff's `full-tests` branch is not present in the tested remote-ref
  snapshot. Its historical sanitized regressions therefore were not run.
- No Git tag points at tested main-equivalent head
  `57af578847682ff5e34b03865c33c96106918bb8`. The local
  `main-transfer-release-port` branch is not a published canonical release
  ref; the tested snapshot shows `origin/main` and source remote refs, not a
  remote ref for this local handoff branch. A release owner must publish the
  intended canonical branch/tag and record its immutable ref.

## Consequence

The public artifact supports source closure, no-provider planning and mock
execution, local evaluator diagnostics, HCI smoke, and deliberate boundary
failures. It does not currently support a complete historical-paper
reproduction claim. Formal runs must continue to fail closed until the
authoritative assets above are published and bound.

