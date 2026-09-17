# Configuration

The release separates canonical paper identity from user-local runtime settings.
Canonical study files are versioned; credentials, output/cache locations, and
the Inspector port are environment-local.

## Canonical paper configuration

| Location | Purpose |
| --- | --- |
| `configs/main-study.yaml` | 10 workloads, 3 seeds, 4 arms, ticks, source-runner ceilings |
| `configs/arms/b0.yaml` through `b3.yaml` | Paper arm invariants |
| `configs/models/*.yaml` | Paper model labels and frozen runtime route contract |
| `configs/workloads/w01.yaml` through `w10.yaml` | Workload-to-frozen-pack mapping |
| `benchmarks/relic-main-v1/manifest.yaml` | Benchmark version and byte-tree hashes |

The fixed precedence order is: **explicit CLI argument → canonical experiment
configuration → documented environment variable → repository default**. A CLI
argument may choose a supported model, output directory, bounded batch/workload
selection, or requested concurrency. It may not redefine the canonical arms,
workloads, seeds, tick budget, evaluator policy, or transfer protocol package.

## Environment-local configuration

Copy `.env.example` to `.env`; never commit it. The definitive variable table
is in [environment.md](environment.md#environment-variables). The commonly
used values are:

- `OPENAI_API_KEY` and optional `OPENAI_BASE_URL` for OpenAI-compatible routes;
- `RELIC_RUNTIME_MODEL` (or `ORG_LLM_RUNTIME_MODEL` / `OPENAI_MODEL`) to map a
  canonical paper model to the deployment name used by your provider;
- `RELIC_OUTPUT_ROOT`, `RELIC_CACHE_ROOT`, and `RELIC_BENCHMARK_ROOT` for
  local paths;
- `RELIC_INSPECTOR_PORT` and `RELIC_TRACE_FILE` for Inspector startup;
- `RELIC_CLAUDE_OPUS_4_6_MODEL` only when an author/operator supplies the
  deployed gateway alias; and
- `RELIC_EVALUATOR_MODE=local` for the default public host evaluator, or
  `RELIC_EVALUATOR_MODE=container` with an explicit evaluator binding.

The Bash wrappers parse only this allow-listed public environment contract; they
do not source arbitrary shell code from `.env`. Direct `uv run relic ...`
commands do not load the file implicitly; pass `--env-file .env` for a
provider-backed invocation, for example `uv run --env-file .env relic run-main`.

## Outputs and manifests

`run-main` writes a source-main manifest plus per-batch source plans beneath the
chosen output root. `run-transfer` writes a distinct transfer manifest. Output
directories are user-local and ignored by Git. A manifest records plan identity,
selection, evaluator mode and observed metadata, execution state, and resource
advice; it is not a substitute for the unreleased historical paper result
matrix.

See [reproduction/](../reproduction/README.md) for command-specific output and
paper-section mapping.
