# Relic

[中文说明](README.zh-CN.md) · [Documentation](docs/README.md) · [Paper results](artifacts/paper_results/paper_results.md)

Relic is the public code and configuration for the Relic agent-organization
paper. This repository defines the paper experiment, frozen benchmark, OrgEnv
integration, evaluator boundary, and reproduction commands. The
benchmark-independent organization runtime is maintained in the
[Relic-Agent repository](https://github.com/Hongyi-Du/Relic-Agent).

The YAML files in `configs/` and the benchmark under
`benchmarks/relic-main-v1/` are the experiment specification. The main study
has 10 workloads, 3 seeds, and four arms for each model (120 cells per model).
The transfer study adds fresh B2 Text and Exec targets; Fresh is the matching
main-study B2 result.

## Experimental arms

| Arm | Members | Decision mode | Profile/capability conditioning | Institutionalization |
| --- | ---: | --- | --- | --- |
| B0 | 1 | direct LLM action selection | Off | Off |
| B1 | 8 persistent roles | direct LLM action selection | Off | Off |
| B2 | 8 persistent roles | SDL/profile policy | On | Off |
| B3 | 8 persistent roles | SDL/profile policy | On | On, including runtime protocol binding |

## Install

Use Python 3.12+, `uv`, and native Linux or WSL2 on a Linux filesystem.

```bash
git clone https://github.com/Hongyi-Du/Relic.git relic
cd relic
uv sync --extra dev --frozen
cp .env.example .env
uv run relic verify-benchmark
uv run relic check-env --scope core
uv run relic smoke --mode mock
```

The benchmark check, core environment check, mock smoke, and default test suite
do not contact a model provider. Run `uv run pytest -q` when you want the
release test suite as well.

## Configure a provider

Edit `.env` with your OpenAI-compatible route. Keep credentials in the local
`.env`; Relic passes them to the provider process without writing them to
plans, manifests, or public traces.

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-gateway.example/v1
RELIC_OPENAI_DEFAULT_HEADERS_JSON={}
RELIC_RUNTIME_MODEL=your-provider-deployment
```

Direct `uv run relic ...` commands inherit only the current process environment;
they do not load `.env` implicitly. For provider-backed commands, pass the file
explicitly with `uv run --env-file .env relic ...`, as shown below. The thin
Bash wrappers under `scripts/bash/` load the allow-listed values from the
repository `.env` automatically.

`--model` always names the canonical paper model. `--runtime-model` selects the
deployment name used by the provider while preserving that canonical ID in
plans and receipts. The runtime-name precedence is the CLI option, then
`RELIC_RUNTIME_MODEL`, `ORG_LLM_RUNTIME_MODEL`, `OPENAI_MODEL`, and
`ORG_LLM_MODEL`. For the Claude paper arm, set
`RELIC_CLAUDE_OPUS_4_6_MODEL` to the gateway deployment alias when no generic
runtime override is present. A missing runtime name reports the variable that
needs to be set and the `--runtime-model` alternative.
An unresolved dry plan can fill this name from the CLI or environment on its
first execution. Once execution starts, resume keeps the recorded model identity.

## Plan and run the main study

Create the 120-cell source-backed plan without provider calls:

```bash
uv run relic run-main \
  --model gpt-5.6-terra \
  --output-root outputs/main-study \
  --manifest outputs/main-study/source_main_manifest.json \
  --max-parallel 1 \
  --dry-run
```

After configuring the provider, run the plan. Keep the paired B0–B3 group when
sampling with `--workload`, `--seed`, or `--batch`.

```bash
uv run --env-file .env relic run-main \
  --manifest outputs/main-study/source_main_manifest.json \
  --resume \
  --max-parallel 1 \
  --workload w01 \
  --seed 1401
```

Remove the selection flags for the complete main study. Each source batch
starts fresh B0–B3 condition processes and writes checkpoints and evaluator
metadata beneath its output directory.

The transfer entrypoint uses the same provider configuration:

```bash
uv run relic run-transfer --dry-run \
  --output-root outputs/transfer-v2 \
  --workload w01 \
  --seed 1401
uv run --env-file .env relic run-transfer \
  --manifest outputs/transfer-v2/transfer_manifest.json \
  --resume \
  --max-parallel 1 \
  --workload w01 \
  --seed 1401
```

## Aggregate local runs

The paired `run-main` path writes `source_main_manifest.json` at the output
root and, under each `batches/<batch-id>/`, a `manifest.json` plus
`experiment_runs.json` and `experiment_runs.jsonl`. These aggregate that batch's
B0–B3 experiment records, including evaluator evidence and the case verdict;
failed or degraded cases remain labeled in the aggregate. The top-level
manifest links the batch results and observed evaluator metadata.

The separate legacy cell scheduler uses `run_manifest.json`. For those
completed user-run receipts, evaluate eligible cells and write its metric
aggregate as follows (these commands do not accept `source_main_manifest.json`):

```bash
uv run relic evaluate \
  --manifest outputs/user-run/run_manifest.json \
  --receipt-directory outputs/user-run/evaluation
uv run relic aggregate-user-runs \
  --evaluation-manifest outputs/user-run/evaluation/evaluation_manifest.json \
  --output-directory outputs/user-run/aggregate \
  --allow-partial
```

The aggregate is explicitly labeled as a local user-run result. The checked-in
paper-results snapshot remains the paper's reported aggregate and is not
recomputed from local runs.

## Advanced workflows

The default main and transfer paths run the public evaluator on the host and
record its observed metadata. An optional `--evaluator-bindings` JSON file
records container provenance for selected packs. Add
`--strict-reproducibility` when every selected pack must provide a
digest-pinned image, `linux/amd64`, and qualification hashes. See
[evaluator.md](docs/evaluator.md), [environment.md](docs/environment.md),
[transfer.md](docs/transfer.md), and [inspector.md](docs/inspector.md) for
container qualification, environment variables, transfer details, and trace
inspection.
