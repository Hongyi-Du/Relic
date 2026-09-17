# Installation

Relic's supported runtime is Linux. On Windows, use WSL2 and keep the checkout
in the Linux filesystem; native PowerShell only launches that WSL runtime.
Docker is the supported isolated alternative. See [environment.md](environment.md)
for the complete platform matrix.

## Native Linux or WSL2

Requirements: Python 3.12 or newer and `uv`.

```bash
git clone <YOUR-RELIC-REMOTE> relic
cd relic
cp .env.example .env
uv sync --extra dev --frozen
uv run relic verify-benchmark
uv run relic check-env --scope core
uv run relic smoke --mode mock
```

These commands install the release package, verify frozen benchmark bytes, and
exercise the no-provider core path. They do not establish a formal paper run.
Use `scripts/bash/` for thin shell conveniences; all execution logic remains in
the `relic` Python CLI.

## Docker / Compose

Docker requires a Linux Docker daemon, or Docker Desktop with the WSL2 backend.

```bash
cp .env.example .env
mkdir -p outputs cache traces
docker compose build relic
docker compose run --rm relic check-env --scope core
docker compose run --rm relic smoke --mode mock
```

The image uses the same CLI as native installation. It does not bake in API
keys, local outputs, cache, historical results, or selected traces. Mounted
paths and formal-evaluator constraints are documented in [environment.md](environment.md).

## Next safe command

Create the canonical 120-cell source plan without starting a provider or child
condition process:

```bash
uv run relic run-main --model gpt-5.6-terra \
  --output-root outputs/main-study \
  --manifest outputs/main-study/source_main_manifest.json \
  --max-parallel 1 --dry-run
```

Model-backed `run-main` and `run-transfer` use the public host evaluator by
default, so they can run after you configure your provider. Supply
`--evaluator-bindings` for additional provenance, or add
`--strict-reproducibility` to require pinned container values. The formal smoke
command remains a strict container diagnostic. See [evaluator.md](evaluator.md)
and [KNOWN_RELEASE_GAPS.md](KNOWN_RELEASE_GAPS.md).
