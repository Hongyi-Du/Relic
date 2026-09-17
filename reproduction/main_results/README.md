# Main-study reproduction

Paper scope: the 10-workload × 3-seed × 4-arm main study described by
`configs/main-study.yaml`, `configs/arms/`, and
`benchmarks/relic-main-v1/`.

Create a no-provider plan with `relic run-main --model gpt-5.6-terra
--dry-run`; it writes a 120-cell, 30 paired-batch manifest under the chosen
output root. Resume and retry use that same manifest. The runner preserves
paired B0--B3 batches rather than treating a single arm as an independent
paper result.

Formal runs and evaluator scoring require the authors' digest-pinned per-pack
evaluator bindings. Their absence is intentional and documented in
[`../../docs/KNOWN_RELEASE_GAPS.md`](../../docs/KNOWN_RELEASE_GAPS.md). A new
user run must be evaluated and aggregated only as a user result, never relabeled
as the paper snapshot in `artifacts/paper_results/`.
