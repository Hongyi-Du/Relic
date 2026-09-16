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
uv sync --extra dev --frozen
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

Evaluate all eligible cells from a user-created v2 run manifest, or from the
run directory that contains exactly that manifest:

```bash
uv run relic evaluate \
  --manifest outputs/main-study/run_manifest.json \
  --receipt-directory outputs/main-study/evaluation

# Equivalent directory form; this does not recursively discover arbitrary runs.
uv run relic evaluate \
  --output-root outputs/main-study \
  --receipt-directory outputs/main-study/evaluation
```

The batch is serial and writes `evaluation_manifest.json` even when an eligible
cell fails. `--selection failed-evaluation` limits a repair pass to scheduler
cells classified as evaluator failures; `all-eligible` includes those and
completed cells. Add `--dry-run` to inspect selection without loading a
checkpoint. Batch evaluation only accepts manifests marked as user-created by
this release and never changes scheduler attempt history.

Batch evaluation has the same checkpoint trust boundary as one-cell evaluation:
it deserializes local Python checkpoints before the candidate repository enters
the restricted evaluator container. Only use checkpoints produced in this
controlled local run directory. Hash sidecars detect accidental changes; they
do not make a downloaded pickle safe.

Summarize one or two single-model evaluation receipts without reading author
raw runs or the canonical paper snapshot:

```bash
uv run relic aggregate-user-runs \
  --evaluation-manifest outputs/main-study/evaluation/evaluation_manifest.json \
  --output-directory outputs/main-study/aggregate \
  --allow-partial
```

Aggregation accepts only completed, non-dry-run local batches whose scheduler
runtime was unchanged. It validates the frozen repository/branch/commit/tree,
both receipt run-origin declarations, each single-model 120-cell plan, and
every included cell before using a value. The output is deliberately named
`user_run_aggregate.json/md`, begins with a “not paper results” notice, averages
seeds inside each model-by-workload block, then weights applicable blocks
equally. B3−B2 uses paired seeds and 10,000 fixed-block bootstrap draws with
seed 1729. A single 120-cell model run is a partial design, so it requires
`--allow-partial`; only both complete model receipts constitute the 240-cell
design. Evaluator infrastructure or unavailable results are excluded as
unavailable, never treated as low scores.

The current frozen evaluator does not emit a versioned, hash-bound scoring
ledger mapping leaf cases and contracts to exposed versus held-out paper units.
Consequently user aggregates report provider-token and generic evaluator
diagnostics, while paper-named contract/case/confirmed-issue metrics remain
explicitly `NA` instead of being guessed. Completing those metrics requires the
first-author scoring ledger. The values under `artifacts/paper_results/` remain
the PDF-transcribed historical snapshot and are never aggregate input.

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
