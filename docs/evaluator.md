# Evaluator build and qualification boundary

Relic's OSS time-machine evaluator has two distinct layers:

1. A public, source-derived Python environment and qualification path, included
   here so a fresh checkout can build and smoke-test the evaluator runtime.
2. An optional evaluator binding: an operator or author may supply an immutable
   image reference and per-pack qualification hashes as reproducibility
   provenance. The default reproduction path runs the public evaluator on the
   host and records the observed evaluator metadata.

The build context and tools are ported from
`SocioGenesis/hci-human-seat@dda36fb563375060ae8d8850300db01eb4695d29`:

- `.evaluator_image/Dockerfile` → [`evaluator/Dockerfile`](../evaluator/Dockerfile)
- `.cursor/build_evaluators.sh` → `relic evaluator-build`
- `tools/print_evaluator_hashes.py` → `relic evaluator-hashes`
- `tools/preflight_organization_evaluator.py` → `relic evaluator-preflight`

The release adaptation deliberately removes the source build proxy argument,
development-machine paths, private registry assumptions, and repository-specific
ProgramBench images. It retains the existing `relic.evaluation` isolated,
network-disabled executor and the source-backed evaluator contract. A strict
reproducibility run can restore the binding gate with
`--strict-reproducibility`.

## Local source-closure check

Docker is required on the host; this command is not supported from the
rootless/restricted controller Compose service because that service intentionally
does not mount the Docker socket.

```bash
uv run relic evaluator-build --smoke
```

The command builds `relic-oss-evaluator:local` for `linux/amd64`, checks its
local image ID and platform, then imports the frozen suite dependencies in a
read-only, non-root, network-disabled container. It contains no benchmark,
hidden tests, API key, endpoint, relay credential, author results, or selected
trace. The frozen evaluator assets are staged only when a pack is evaluated and
are mounted read-only by the existing executor.

The equivalent thin checkout wrapper is:

```bash
uv run python tools/build_evaluator_image.py --smoke
```

To verify the public host evaluator without an API key or model call, run:

```bash
env RELIC_EVALUATOR_MODE=local \
  RELIC_EVALUATOR_BACKEND=local \
  RELIC_EVALUATOR_STRICT_REPRODUCIBILITY=0 \
  RELIC_EVALUATOR_CONTAINER_IMAGE= \
  RELIC_EVALUATOR_CONTAINER_PLATFORM= \
  uv run python -c 'from pathlib import Path; from relic.cell_spec import compile_cell_spec; from relic.cell_worker import _preflight_evaluator; s=compile_cell_spec(model="gpt-5.6-terra", workload="W01", arm="B0", seed=1401, output_root=Path("/tmp/relic-local-evaluator-smoke")); b=_preflight_evaluator(s); print({"backend": b["execution_policy"]["backend"], "dataset": b["dataset_id"], "plan_hash": b["qualification_plan_sha256"]})'
```

The output reports the selected local backend, dataset, and qualification plan
hash. It is an operational smoke check; it does not create historical paper
evidence.

## Local qualification diagnostics

After the local build, retrieve its Docker image ID and run a qualification of
one public pack:

```bash
IMAGE_ID="$(docker image inspect relic-oss-evaluator:local --format '{{.Id}}')"
uv run relic evaluator-hashes \
  --dataset mini_blobstore_v1 \
  --backend docker \
  --container-image "$IMAGE_ID" \
  --container-platform linux/amd64
```

This runs the frozen starter and reference suites in the restricted container
and reports the source-compatible environment and plan hashes as
`local_*` fields. To independently recheck a known local result and persist a
receipt, pass those two reported values to:

```bash
uv run relic evaluator-preflight \
  --repository-id mini_blobstore_v1 \
  --dataset mini_blobstore_v1 \
  --backend docker \
  --container-image "$IMAGE_ID" \
  --container-platform linux/amd64 \
  --expected-environment-hash <local-environment-hash> \
  --expected-qualification-hash <local-plan-hash> \
  --output outputs/evaluator-preflight.json
```

The output redacts the supplied container reference to a SHA-256 identity hash
instead of persisting arbitrary registry text. Its `attestation_hash` excludes
the wall-clock timestamp and is deterministic for identical observed inputs.

## What these commands do not establish

A local Docker image ID (`sha256:...`) and locally calculated hashes are not an
author-published evaluator image digest or paper qualification binding. They
remain useful local provenance, but do not describe them as historical paper
evidence. A non-strict run may use an explicit operator binding with a tag or
omitted platform; strict mode accepts only digest-pinned values.

The following historical provenance assets are not included in the available
source branches:

- the author-published immutable evaluator registry digest(s);
- the author-approved per-pack evaluator-environment and qualification-plan
  hashes that bind the paper runs; and
- the first-author leaf-case / contract scoring ledger needed to expose
  paper-named user aggregate metrics.

These assets are optional for normal local runs and for publishing this source
release. They matter only when making a historical byte-level reproducibility
claim or reconstructing paper-named aggregate evidence.

`check-env --scope formal` and `smoke --mode formal` remain strict diagnostics
for a formally reproducible container claim. They can report a missing binding
even though the normal `run-main`, `run-transfer`, and legacy cell paths are
allowed to use the host evaluator. A successful local build or preflight is
useful source-closure evidence only, not a historical paper claim.

## Release scope

Only the ten pack IDs in `relic-main-v1` are accepted by these tools. Absolute
paths, unlisted packs, and any ProgramBench request are rejected before a
container is invoked. ProgramBench tasks, adapters, assets, images, and
reproduction entrypoints are not part of this release.
