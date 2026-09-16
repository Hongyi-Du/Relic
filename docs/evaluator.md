# Evaluator build and qualification boundary

Relic's OSS time-machine evaluator has two distinct layers:

1. A public, source-derived Python environment and qualification path, included
   here so a fresh checkout can build and smoke-test the evaluator runtime.
2. The paper evaluator binding: an author-published immutable image reference
   and per-pack qualification hashes. That binding is not present in the
   available source assets, so paper runs remain fail-closed.

The build context and tools are ported from
`SocioGenesis/hci-human-seat@dda36fb563375060ae8d8850300db01eb4695d29`:

- `.evaluator_image/Dockerfile` → [`evaluator/Dockerfile`](../evaluator/Dockerfile)
- `.cursor/build_evaluators.sh` → `relic evaluator-build`
- `tools/print_evaluator_hashes.py` → `relic evaluator-hashes`
- `tools/preflight_organization_evaluator.py` → `relic evaluator-preflight`

The release adaptation deliberately removes the source build proxy argument,
development-machine paths, private registry assumptions, and repository-specific
ProgramBench images. It retains the existing `relic.evaluation` isolated,
network-disabled executor and the source-backed `run-main` binding gate.

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

A local Docker image ID (`sha256:...`) and locally calculated hashes are not a
published evaluator image digest or a paper qualification binding. In
particular, do not turn their output into an `--evaluator-bindings` file.
`relic run-main` accepts only digest-pinned registry references of the form
`...@sha256:<64-hex>` and still requires author-supplied per-pack bindings
before it starts any provider process.

The following required release inputs were not found in the available HCI or
other source branches and remain explicit gaps:

- the author-published immutable evaluator registry digest(s);
- the author-approved per-pack evaluator-environment and qualification-plan
  hashes that bind the paper runs; and
- the first-author leaf-case / contract scoring ledger needed to expose
  paper-named user aggregate metrics.

Until those assets are supplied, `check-env --scope formal`, formal smoke, and
non-dry-run paper reproduction must remain fail-closed. A successful local
build or preflight is useful source-closure evidence only, not a reproduction
claim.

## Release scope

Only the ten pack IDs in `relic-main-v1` are accepted by these tools. Absolute
paths, unlisted packs, and any ProgramBench request are rejected before a
container is invoked. ProgramBench tasks, adapters, assets, images, and
reproduction entrypoints are not part of this release.
