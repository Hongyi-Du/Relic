# Scope and validation limits

Normal `run-main` and `run-transfer` use the public host evaluator without an
evaluator binding. Configure your own API key, gateway, and runtime model name;
no author gateway alias or historical evaluator identity is required.

Bindings are optional provenance. Explicit `--strict-reproducibility` checks
container digest, platform, environment and qualification hashes. An operator
can provide a valid strict binding; it need not be a newly published author
asset. The paper's workloads, conditions, scoring contract, seeds and pairing
remain fixed in either mode.

Historical raw trajectories, selected paper traces, and the author scoring
ledger are not reconstructed by this stage. Their absence does not prevent
normal runs or publication of the source. New local receipts describe new
runs; they do not recreate historical measurements.

The HCI extension remains part of this checkout. Cooper now has a final frozen
full-652 reference and separately published historical trajectories/exclusion QA
linked from `reproduction/cooperbench/README.md`. Upstream dataset, task images,
Docker, Redis, and provider credentials remain external run prerequisites.
ProgramBench reproduction assets remain outside this repository.

Tests stay in the ordinary repository. A `full-tests` branch, signed release
tag, registry publication, supply-chain certification, and cross-platform byte
identity are not prerequisites for normal use. See the README for the ordinary
workflow and [evaluator.md](evaluator.md) for optional strict checks.
