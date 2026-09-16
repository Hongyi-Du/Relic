# Relic

Research artifact for the Relic agent-organization paper.

Scientific facts and reported results follow the final paper. Repository scope
follows the two-repository release handoff. Curated implementation provenance is
frozen exclusively to `SocioGenesis/hci-human-seat` at
`dda36fb563375060ae8d8850300db01eb4695d29`.

## Local setup

Python 3.12+ and `uv` are required. The currently validated setup is native
Linux or WSL2 from a Linux filesystem, not `/mnt/c`:

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

Formal smoke first runs the formal environment gate. Only after that gate
passes does it perform a network-disabled container workspace roundtrip; it
does not call a model or claim that a paper experiment succeeded. With the
current missing evaluator image, it exits at the gate before the roundtrip.

The current Windows runtime path is WSL2. Native Windows experiment execution is
not maintained. Docker / Compose supports the release core, mock smoke,
main-study dry-run planning, mounted outputs, and the Inspector. Formal cells
and evaluation remain fail-closed until the authors publish the digest-pinned
evaluator image and its reviewed controller-container integration. See
[docs/environment.md](docs/environment.md) for the support matrix, `.env`
variables, WSL launchers, container boundary, and memory guidance.

## Docker quickstart

The image runs the same `relic` Python CLI as the native path. It does not bake
in credentials, local configuration, outputs, caches, historical results, or
selected traces. Create the bind-mount directories as your host user before the
first Compose run so output ownership is predictable:

```bash
cp .env.example .env
mkdir -p outputs cache traces
docker compose build relic
docker compose run --rm relic check-env --scope core
docker compose run --rm relic smoke --mode mock
```

If your Linux or WSL user is not UID/GID 1000, set `RELIC_UID` and `RELIC_GID`
in `.env` to the values printed by `id -u` and `id -g`. The container uses a
read-only root filesystem and writes only to the mounted `outputs/` and
`cache/` directories.

Generate the canonical single-model 120-cell manifest without provider or
evaluator calls:

```bash
docker compose run --rm relic run-main \
  --model gpt-5.6-terra \
  --output-root /data/outputs/main-study \
  --manifest /data/outputs/main-study/run_manifest.json \
  --max-parallel 1 \
  --dry-run
```

The formal CLI remains available as the canonical entrypoint, but a real
`run-cell`, resumed run, or `evaluate` must not be presented as working in this
image yet. The required evaluator image has not been supplied, and the default
Compose services intentionally do not mount the host Docker socket. Formal
checks therefore fail before any provider request.

To use the Inspector, place an author-supplied, sanitized `relic-trace-v1` file
under `traces/`, set `RELIC_TRACE_FILE` in `.env` to its filename, then run:

```bash
docker compose --profile inspector up relic-inspector
```

Open `http://127.0.0.1:8765`. Compose publishes only the host loopback address;
`--allow-remote` acknowledges the necessary container-internal `0.0.0.0` bind.
The server still rejects arbitrary DNS Host headers. No selected paper trace is
bundled, and the count-only
`relic-public-trace-v1` cell sidecar is not valid Inspector input.

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

Concurrency defaults to one; budget about 16 GiB for each active cell:

| Host memory | Conservative maximum parallel cells |
|---:|---:|
| 16 GiB | 1 |
| 32 GiB | 2 |
| 64 GiB | 4 |
| 100+ GiB | up to 8 |
| 128 GiB | recommended for 8 |

For Windows/WSL2 reproduction, budget approximately 16 GB of RAM per active
parallel cell/container. Use 1 parallel worker on 16 GB, 2 on 32 GB, 4 on 64
GB, and 8 only on machines with more than 100 GB of RAM; 128 GB is recommended
for 8-way parallel execution. Check both `.wslconfig` and Docker Desktop memory
limits, and reduce parallelism when the host is also running memory-heavy tools.
The current controller runs several cell subprocesses inside one container;
scaling the Compose service is not a substitute for the canonical scheduler.

Windows / WSL2 复现时建议按照每个活跃并发 cell / container 约 16 GB
内存预算。16 GB 建议 1 并发，32 GB 建议 2 并发，64 GB 建议 4 并发；
8 并发及以上要求机器拥有 100 GB 以上内存，推荐 128 GB。

Formal execution can be costly, so always inspect the dry-run manifest first.

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

## Inspect a selected public trace

Relic Inspector is a local, read-only organization observatory for the strict,
digest-bound `relic-trace-v1` format. Validate a trace without opening a server,
then start replay mode:

```bash
uv run relic replay --trace /path/to/selected-trace.json
uv run relic inspect --trace /path/to/selected-trace.json
```

Open `http://127.0.0.1:8765`. The Inspector synchronizes the Timeline,
event-level organization snapshot, Object Inspector, and State Diff, and never
opens checkpoints, evaluator directories, model messages, or private agent
memory. Native and WSL launches bind loopback by default; non-loopback binding
requires the explicit `--allow-remote` acknowledgement because the server has
no authentication. Remote mode accepts literal IP Host headers and rejects
arbitrary DNS names to retain the DNS-rebinding boundary.

The current author-asset bundle does not yet include sanitized selected paper
traces. No historical case is reconstructed or invented to fill that gap. The
count-only `public/trace.json` checkpoint-progress sidecar written by the
main-study cell runner uses the separate `relic-public-trace-v1` projection and
is deliberately rejected as Inspector input. See
[docs/inspector.md](docs/inspector.md) for the schema, privacy boundary, WSL
launcher, and replay/live semantics.

Inspector views are descriptive: observed lineage, temporal order, and state
differences do not by themselves establish that one protocol, member, or
mechanism caused an outcome.

## Full regression test suite / 完整回归测试

`main` contains the release-focused tests used by the default local and CI
checks. They are safe by default: `uv run pytest` does not make paid model
calls. No current default-suite test is marked `live`, `llm`, `slow`, or
`docker`; future tests using those markers must remain opt-in because they may
require credentials, substantial runtime, or a Docker daemon.

The handoff reserves a future `full-tests` branch for sanitized historical
core regression tests. That branch is not present in this release snapshot, so
this README does not provide a `git switch full-tests` command that would fail.
When published, it must be based on the corresponding release commit, add test
depth without becoming a second implementation, and exclude obsolete systems,
private fixtures, credentials, and development-machine paths.

`main` 包含默认本地检查与 CI 使用的 release-focused tests；默认执行
`uv run pytest` 不会调用付费模型。当前默认测试集没有标记为 `live`、`llm`、
`slow` 或 `docker` 的测试；未来使用这些 marker 的测试必须保持显式启用，因为
它们可能需要凭据、较长运行时间或 Docker。交接文档规划的
`full-tests` 分支用于保存清理后的历史核心回归测试，但当前 release 快照尚未发布
该分支，因此这里不会给出当前必然失败的切换命令。

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
