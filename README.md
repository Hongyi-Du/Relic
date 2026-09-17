# Relic

[中文说明](README.zh-CN.md) · [Documentation](docs/README.md) · [Release scope](docs/release-scope.md) · [Paper results](artifacts/paper_results/paper_results.md) · [Known gaps](docs/KNOWN_RELEASE_GAPS.md)

Public research artifact and partial reproduction harness for the Relic
agent-organization paper. This checkout is not a complete reproduction of the
historical paper experiments: it can materialize source-backed plans, run
no-provider mock/source-closure checks, and locally qualify the OSS evaluator,
but formal model runs remain fail-closed until the authors publish the required
evaluator bindings and evidence assets. See
[`docs/KNOWN_RELEASE_GAPS.md`](docs/KNOWN_RELEASE_GAPS.md) and
[`docs/REPRODUCTION_RUN_REPORT.md`](docs/REPRODUCTION_RUN_REPORT.md) for the
current, tested release state.

Scientific facts and reported results follow the final paper. Repository scope
follows the two-repository release handoff. Curated implementation provenance is
frozen exclusively to `SocioGenesis/hci-human-seat` at
`dda36fb563375060ae8d8850300db01eb4695d29`.

## What is in this repository

`main` is the clean paper-reproduction branch: the OrgEnv core, B0--B3,
`relic-main-v1`, the paired main-study and transfer launchers, evaluator
boundary, paper-result snapshot, and public Inspector. The prepared extension
branches deliberately remain narrow:

| Branch | Purpose | Start here |
| --- | --- | --- |
| `main` | Canonical paper reproduction | This README |
| `hci` | `main` plus the P2/P3 human-seat extension | After `git switch hci`, read `docs/HCI_GUIDE.md` |
| `cooper` | `main` plus the CooperBench B3-2 adapter and tests | After `git switch cooper`, read `reproduction/cooperbench/README.md` |
| `full-tests` | Future sanitized historical **core** regression suite | Not published yet; see [the branch policy](docs/release-scope.md#branch-topology-and-sync-policy) |

The sibling [Relic-Agent repository](https://github.com/Hongyi-Du/Relic-Agent)
is the benchmark-independent organization runtime. This repository is the
paper artifact. The author-controlled project-web URL has not been supplied,
so no project-page link is invented here.

## Canonical experimental arms

The checked-in YAML files under `configs/arms/` are authoritative. The table is
included here so a reader can identify the paper conditions before running a
command.

| Arm | Members | Decision mode | Profile/capability conditioning | Institutionalization |
| --- | ---: | --- | --- | --- |
| B0 | 1 | direct LLM action selection | Off | Off |
| B1 | 8 persistent roles | direct LLM action selection | Off | Off |
| B2 | 8 persistent roles | SDL/profile policy | On | Off |
| B3 | 8 persistent roles | SDL/profile policy | On | On, including runtime protocol binding |

The main benchmark is [`benchmarks/relic-main-v1/`](benchmarks/relic-main-v1/);
the final aggregate snapshot is [`artifacts/paper_results/`](artifacts/paper_results/).
The reproduction-directory index explains the correspondence among paper
sections, configs, entrypoints, outputs, and current asset boundaries.

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
model-specific and fail closed while the author-published evaluator binding is
unavailable:

```bash
uv run relic check-env --scope formal --model gpt-5.6-terra
uv run relic smoke --mode formal
```

Formal smoke first runs the formal environment gate. Only after that gate
passes does it perform a network-disabled container workspace roundtrip; it
does not call a model or claim that a paper experiment succeeded. With the
current missing author-published evaluator binding, it exits at the gate before
the roundtrip.

The public source-derived evaluator build path is available separately for a
local source-closure check:

```bash
uv run relic evaluator-build --smoke
```

It builds and smoke-tests a local, network-isolated evaluator image but does
not create a paper binding. See [docs/evaluator.md](docs/evaluator.md) for the
qualification commands, the intentionally excluded ProgramBench path, and the
remaining author-supplied digest/hash gap.

The current Windows runtime path is WSL2. Native Windows experiment execution is
not maintained. Docker / Compose supports the release core, mock smoke,
main-study dry-run planning, mounted outputs, and the Inspector. Formal cells
and evaluation remain fail-closed until the authors publish the digest-pinned
evaluator image and its reviewed controller-container integration. See
[docs/environment.md](docs/environment.md) for the support matrix, `.env`
variables, WSL launchers, container boundary, and memory guidance.

## Command cost and configuration precedence

Start with the commands above: benchmark verification, `check-env --scope core`,
and `smoke --mode mock` have no model cost. `replay`, `inspect`, and
`build-paper-results` also have no model cost. A formal `run-main` or
`run-transfer` command can incur provider usage only after its author-supplied
formal gate passes; the complete 120-cell command is high-cost/high-memory.
Do not run it as an installation check.

For a parameter that is configurable at more than one layer, the fixed
precedence is: explicit CLI argument, canonical experiment configuration,
documented environment variable, then repository default. Canonical study
identity (arms, workloads, seeds, ticks, evaluator policy) is intentionally not
overridden by `.env`.

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

Generate the source-backed paired 120-cell dry plan without provider or
condition-process calls:

```bash
docker compose run --rm relic run-main \
  --model gpt-5.6-terra \
  --output-root /data/outputs/main-study \
  --manifest /data/outputs/main-study/source_main_manifest.json \
  --max-parallel 1 \
  --dry-run
```

`run-main` is the canonical entrypoint. It delegates every pack/seed group to
the hci source baseline runner, which isolates its four B0--B3 conditions in
fresh processes. The required author-published evaluator image binding and
qualification hashes have not been supplied, so a real source batch fails
closed before any provider request.

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

`run-main` is source-backed: it expands the paper design into 30 paired source
batches (10 packs x 3 seeds), and each batch invokes
`tools/run_org_baselines.py` for its four B0--B3 conditions. The source runner,
not the old Relic per-cell compatibility worker, owns fresh-process isolation,
deterministic paired ordering, checkpoints, resume identity, LLM blackout
gates, and formal-record validation. The frozen source batch also carries the
paper's `work_rhythm` ablation.

Use dry-run mode to write the outer 120-cell plan and all 30 source case plans
without contacting a provider or starting a condition subprocess:

```bash
uv run relic run-main \
  --model gpt-5.6-terra \
  --output-root outputs/main-study \
  --manifest outputs/main-study/source_main_manifest.json \
  --max-parallel 1 \
  --dry-run
```

The outer manifest is `source_main_manifest.json`; source dry-run artefacts are
under `source-dry-run/`, deliberately separate from eventual real batches.
That separation lets a later formal run bind the as-yet-unpublished evaluator
identity without changing a source case plan that has already been used for a
paid run.

Before a real run, obtain the authors' per-pack bindings in a JSON file. Each
entry must contain `backend`, `container_image` (an immutable
`...@sha256:<64-hex>` reference), `container_platform` (`linux/amd64`),
`environment_hash`, and `qualification_plan_hash`. The release intentionally
does not provide placeholder values. Then start or continue the plan with:

```bash
uv run relic run-main \
  --manifest outputs/main-study/source_main_manifest.json \
  --resume --max-parallel 1 \
  --evaluator-bindings author-published-evaluator-bindings.json
```

`--max-parallel` is passed only to the four condition processes inside each
serial source batch; it is not a cross-batch scheduler. To run a safe bounded
sample, retain the paired group and narrow by `--batch w01__seed1401`,
`--workload w01`, or `--seed 1401` rather than selecting one arm. A missing or
incomplete evaluator binding fails before a source child or provider client is
created. `--retry-failed --resume` re-enters the source runner's own
identity-checked case resume path.

## Final transfer comparison

The final-paper transfer entrypoint is `run-transfer`. It plans only the two
new fresh-B2 target arms, Text and Exec (10 workloads × 3 seeds × 2 = 60
targets); Fresh remains the existing main-study B2 reference and is never
generated as a new transfer run. Both arms use the bundled canonical v2
six-guard package and lock protocol formation, adoption, and revision across
the complete target window.

```bash
uv run relic run-transfer --dry-run \
  --output-root outputs/transfer-v2 \
  --workload w01 --seed 1401
```

As with `run-main`, a non-dry transfer command fails closed before any source
child or provider request unless the authors' digest-pinned evaluator bindings
are supplied. It creates new local reproduction artifacts only; it does not
fabricate or substitute paper raw results. See [docs/transfer.md](docs/transfer.md).

## Legacy single-cell compatibility command

`relic run-cell`, `relic.main_runner`, and `relic.cell_worker` remain only for
backwards-compatible local artefacts and their historical tests. They are not
the official paper reproduction executor, and their output must not be mixed
with the paired source-runner matrix above.

Historical compatibility cells separate private continuation/evaluator artifacts
from their public allow-list projection:

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

Resume the same legacy cell after a validated checkpoint:

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

The remaining `evaluate` and `aggregate-user-runs` commands apply only to
user-created legacy v2 manifests; they do not consume source-backed
`source_main_manifest.json` output or establish paper reproduction results.
Evaluate eligible legacy cells from a user-created v2 run manifest, or from the
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
seed 1729 through the source `environments.org_env.experiments.statistics`
paired-unit and fixed-block functions. A single 120-cell model run is a partial design, so it requires
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

The frozen formal runner accepts the source project's OpenAI-compatible provider
surface. The Claude arm is not a native Anthropic SDK adapter: it uses the HCI
source's gateway route (`provider: openai`, `chat_completions`, and
`prompt_only` JSON transport). Configure the gateway URL and credential through
the documented OpenAI-compatible variables, and bind its deployed Claude model
name with `RELIC_CLAUDE_OPUS_4_6_MODEL`. This remains a real provider route; it
is not silently replaced by rules.

Formal two-model reproduction still requires the separately missing,
author-published digest-pinned evaluator binding and author-supplied evidence
assets described in the handoff.

ProgramBench is represented only by the aggregate values reported in the paper;
its tasks, adapter, harness, and reproduction entrypoints are intentionally not
part of this repository.
