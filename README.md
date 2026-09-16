# Relic

Research artifact for the Relic agent-organization paper.

Scientific facts and reported results follow the final paper. Repository scope
follows the two-repository release handoff. Curated implementation provenance is
frozen exclusively to `SocioGenesis/hci-human-seat` at
`dda36fb563375060ae8d8850300db01eb4695d29`.

## Local setup

Python 3.12+, `uv`, and Docker Desktop with its WSL2 backend are recommended.
Run from a Linux/WSL filesystem, not `/mnt/c`:

```bash
uv sync --dev
uv run relic verify-benchmark
uv run relic check-env --scope core
uv run relic smoke --mode mock
uv run pytest -q
```

`check-env` and mock smoke never contact a model provider. Formal checks are
model-specific and fail closed while the evaluator image is unavailable:

```bash
uv run relic check-env --scope formal --model gpt-5.6-terra
uv run relic smoke --mode formal
```

The formal smoke performs a network-disabled container workspace roundtrip but
does not call a model or claim that a paper experiment succeeded.

The Windows runtime path is WSL2/Docker. Native Windows experiment execution is
not maintained. See [docs/environment.md](docs/environment.md) for the support
matrix, `.env` variables, WSL launchers, and memory guidance.

## Plan or run the canonical main study

Use the scheduler's dry-run mode to freeze the 120-cell plan for one model
without contacting a provider, evaluator, or cell subprocess:

```bash
uv run relic run-main \
  --model gpt-5.6-terra \
  --output-root outputs/main-study \
  --manifest outputs/main-study/run_manifest.json \
  --max-parallel 1 \
  --dry-run
```

After reviewing that manifest, start or continue the plan with:

```bash
uv run relic run-main \
  --manifest outputs/main-study/run_manifest.json \
  --resume --max-parallel 1
```

Each cell runs in a separate subprocess. The scheduler never changes the
parent process's `ORG_*` identity environment, skips completed cells, writes
its manifest atomically, and keeps subprocess logs under
`outputs/main-study/private/scheduler/`. To retry only cells classified as
model, evaluator, or infrastructure failures, add `--retry-failed` to the
resume command. `--cell-id` can safely narrow a run or retry to one frozen
cell. Interrupting the scheduler stops its child process groups and records an
interrupted state for a later `--resume`.

Concurrency defaults to one; budget about 16 GiB for each active cell. Formal
execution can be costly, so always inspect the dry-run manifest first.

## Run one cell

Formal execution fails closed unless the evaluator is an untrusted,
network-disabled Docker or Apptainer image pinned by immutable digest. Configure
the actual environment names consumed by the evaluator:

```bash
export OPENAI_API_KEY='...'
export RELIC_EVALUATOR_BACKEND='docker'
export RELIC_EVALUATOR_CONTAINER_IMAGE='registry.example/relic-evaluator@sha256:<64-hex-digest>'
export RELIC_EVALUATOR_CONTAINER_PLATFORM='linux/amd64'

uv run relic run-cell \
  --model gpt-5.6-terra \
  --workload W01 \
  --arm B3 \
  --seed 1401 \
  --output-root outputs
```

The evaluator is qualified before any model call. A canonical cell always uses
336 ticks, checkpoints every 24 ticks, `semi_auto` approval, and exactly the
`work_rhythm` ablation from `configs/main-study.yaml`; these values are not CLI
overrides.

This command may incur substantial model usage. The full 120-cell experiment is
considerably more expensive.

Each cell separates private continuation/evaluator artifacts from its public
allow-list projection:

```text
outputs/relic-main-v1/<model>/<workload>/<arm>/seed-<seed>/
  cell-spec.json
  private/checkpoints/
  private/evaluator/
  private/execution-binding.json
  public/status.json
  public/trace.json
  run-record.json
```

Check status without deserializing a checkpoint:

```bash
uv run relic status --cell-dir outputs/relic-main-v1/gpt-5.6-terra/w01/b3/seed-1401
```

Resume the same canonical cell after a validated checkpoint:

```bash
uv run relic run-cell \
  --model gpt-5.6-terra --workload W01 --arm B3 --seed 1401 \
  --output-root outputs --resume
```

Re-run only the final evaluator from the completed rollout checkpoint:

```bash
uv run relic evaluate \
  --cell-dir outputs/relic-main-v1/gpt-5.6-terra/w01/b3/seed-1401
```

Checkpoint pickle files are private trusted local continuation artifacts. Do not
run `--resume` or `evaluate` on downloaded or otherwise untrusted cell
directories; `status` reads only verified sidecars and public JSON.

## Current provider boundary

The frozen implementation has a real OpenAI runtime but no Anthropic adapter.
Therefore `gpt-5.6-terra` is wired to the formal runner, while
`claude-opus-4.6` fails before world execution with
`unsupported_model_provider:anthropic`. The release cannot claim complete
two-model/240-cell reproduction until a reviewed Anthropic adapter and tests are
added. It will not be emulated through OpenAI or silently replaced by rules.

ProgramBench is represented only by the aggregate values reported in the paper;
its tasks, adapter, harness, and reproduction entrypoints are intentionally not
part of this repository.
