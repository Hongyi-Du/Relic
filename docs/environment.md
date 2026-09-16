# Environment and platform support

Linux is the canonical Relic runtime. Windows users run the same Linux runtime
through WSL2; the PowerShell files are launchers only and never execute Relic
core natively. Docker Desktop on Windows must use its WSL2 backend.

| Platform | Support level | Recommended path |
|---|---|---|
| Linux | Full | Native Bash / Python |
| Windows 11 + WSL2 | Full | WSL2 + Bash / Python |
| Windows-native PowerShell | Launcher only | PowerShell invokes WSL2 |
| Docker on Linux | Core release path | Docker / Compose for check, mock smoke, dry-run planning, and Inspector |
| Docker Desktop on Windows | Core release path | WSL2 backend; same container scope as Linux |
| macOS | Not yet validated | Docker may work but is not a claimed release platform |

The repository should live in the WSL Linux filesystem, for example
`~/relic`, rather than under `/mnt/c`. This avoids permission, symlink, line
ending, file-watcher, and small-file performance differences. Windows paths
may still be used to import or export data.

## Local setup

Relic requires Python 3.12 or newer and `uv`:

```bash
uv sync --extra dev --frozen
cp .env.example .env
uv run relic check-env --scope core
uv run relic smoke --mode mock
```

The mock smoke is an installation/config/runtime check. It makes no provider
request and is not evidence that a formal paper cell or evaluator succeeded.
The Bash wrappers load only the documented, allow-listed assignments from the
repository-root `.env` and reject every other variable; direct
`uv run relic ...` commands read only the current process environment.

Formal execution additionally requires provider credentials and a qualified,
network-disabled evaluator container pinned by immutable digest. The evaluator
image has not yet been supplied, so `check-env --scope formal`, formal smoke,
and real cells must fail closed until that image is available. The current
formal provider route is OpenAI-compatible. The paper's Claude arm uses the
HCI source gateway contract (`provider: openai`, `chat_completions`, and
`prompt_only` JSON transport), not a native Anthropic SDK adapter.

The Apptainer backend additionally requires an active Slurm allocation
(`SLURM_JOB_ID` plus `srun`) and the configured registry digest to already be
present in `apptainer cache list -v`. Environment checks never pull images.

## Docker / Compose environment

The release image uses:

- Python 3.12 on a digest-pinned Debian 12 (Bookworm) slim base;
- `uv` 0.12.15 and the frozen `uv.lock` runtime dependency set;
- Debian's `git` and CA certificate packages;
- no Node runtime, because the Inspector ships prebuilt static assets;
- no GPU requirement.

The formal runner accepts the OpenAI-compatible provider route. The paper's
Claude arm is configured through that same route; supply the gateway's deployed
model alias and base URL rather than an Anthropic SDK credential.

The image uses a non-root account. Compose further selects the host UID/GID,
makes the root filesystem read-only, drops Linux capabilities, enables
`no-new-privileges`, supplies a bounded temporary filesystem, and mounts only
these repository-local paths. Provider/evaluator variables are passed only to
the research service; the Inspector service does not receive them.

| Host path | Container path | Access | Purpose |
|---|---|---|---|
| `outputs/` | `/data/outputs` | read/write | manifests, cells, receipts, and exports |
| `cache/` | `/data/cache` | read/write | disposable user cache |
| `traces/` | `/data/traces` | read-only | explicit Inspector input |

Prepare them before the first run, then use the canonical CLI through Compose:

```bash
cp .env.example .env
mkdir -p outputs cache traces
docker compose build relic
docker compose run --rm relic check-env --scope core
docker compose run --rm relic smoke --mode mock
docker compose run --rm relic run-main \
  --model gpt-5.6-terra \
  --output-root /data/outputs/main-study \
  --manifest /data/outputs/main-study/run_manifest.json \
  --max-parallel 1 --dry-run
```

Set `RELIC_UID` and `RELIC_GID` in `.env` if the host values printed by
`id -u` and `id -g` are not 1000. API keys are injected only at runtime via the
environment. They are not needed for any command above and are never copied
into the image.

`.env` is the host-side user configuration boundary for Compose. Compose
interpolates only the explicitly listed provider/evaluator settings and does
not mount the credential file into either container. Canonical study YAML stays
inside the image by design: an arbitrary user-config overlay could silently
change B0--B3 identity while retaining canonical command names, so no such
mount is offered. Outputs, caches, exports, and traces remain explicit bind
mounts as shown above.

The controller image includes the complete frozen, public `relic-main-v1`
research packs. Their `private_evaluator_only` label means those bytes must not
enter model prompts, product workspaces, search, or normal snapshots; it does
not mean they are absent from the public research release. The controller needs
them to verify pack digests and stage a read-only evaluator bundle. This image
is not itself the evaluator sandbox, and the missing reviewed nested-container
integration is why formal Docker execution remains unsupported. The upstream
Celery pack also contains its public example TLS fixtures; they are frozen pack
bytes, not release credentials, and are covered by the benchmark digest.

Formal execution is a separate, unresolved release boundary. The author has
not supplied the immutable evaluator image, and the default controller image
does not contain a nested container runtime or mount the privileged host Docker
socket. Consequently Docker formal checks, real cells, resume, and evaluation
must fail closed; mock smoke and dry-run output are not evidence of paper
reproduction. Do not add the Docker socket ad hoc and call the result supported:
the evaluator workspace mounts and isolation policy require a separately
reviewed integration and end-to-end acceptance test.

For Inspector use, place an author-supplied sanitized `relic-trace-v1` JSON file
in `traces/`, set `RELIC_TRACE_FILE` in `.env`, and run:

```bash
docker compose --profile inspector up relic-inspector
```

The published host port is loopback-only. The trace directory is read-only,
arbitrary DNS Host headers remain rejected, and no selected paper trace is
bundled in the current author assets.

## Docker acceptance record

On 2026-09-17 the core Docker path was exercised from a fresh clone on WSL2,
Linux x86_64, Docker Engine 29.8.0. Acceptance included image build, non-root
and read-only boundary probes, `check-env --scope core`, mock smoke, a 120-cell
single-model dry-run, bind-mount ownership/modes, and Inspector HTTP health,
static assets, and hostile Host-header rejection. The same fresh clone passed
the complete 164-test suite and built its wheel and source distribution.

This is evidence only for the stated core Docker path. It is not a successful
formal cell, evaluator run, selected-paper-trace replay, Claude gateway run, or
complete paper reproduction. Re-run the documented commands on every release
candidate; the ordinary unit suite intentionally does not assume a Docker
daemon is available.

## Environment variables

CLI arguments and canonical experiment configuration take precedence over
environment defaults. Cell identity, ticks, arms, seeds, retry policy, and
evaluation policy are frozen by repository configuration and cannot be
overridden from `.env`.

| Variable | Required | Default | Purpose / example format |
|---|---|---|---|
| `OPENAI_API_KEY` | Formal OpenAI-compatible runs | none | Provider or gateway credential; secret, never persisted by Relic |
| `OPENAI_BASE_URL` | No | official OpenAI endpoint | Optional OpenAI-compatible HTTPS base URL, including the Claude gateway route |
| `RELIC_OPENAI_DEFAULT_HEADERS_JSON` | No | `{}` | Non-authorization compatibility headers as a JSON object |
| `RELIC_OPENAI_DISABLE_RESPONSE_STORAGE` | No | `true` | Keep Responses API storage disabled |
| `RELIC_EVALUATOR_BACKEND` | Formal evaluation | none | `docker`, or `apptainer` with Slurm and a preloaded digest cache entry |
| `RELIC_EVALUATOR_CONTAINER_IMAGE` | Formal evaluation | none | Digest-pinned image such as `registry.example/evaluator@sha256:<64 hex>` |
| `RELIC_EVALUATOR_CONTAINER_PLATFORM` | Formal evaluation | none | Must be `linux/amd64` for the frozen study |
| `ORG_OSS_QUALIFICATION_TIMEOUT` | No | `180` | Evaluator qualification timeout in seconds |
| `RELIC_OUTPUT_ROOT` | No | `<repo>/outputs` | Linux/WSL output root |
| `RELIC_BENCHMARK_ROOT` | No | `<repo>/benchmarks` | Advanced frozen benchmark root override |
| `RELIC_CACHE_ROOT` | No | `<output-root>/.relic-cache` | Writable cache location checked by `check-env` |
| `RELIC_INSPECTOR_PORT` | No | `8765` | Loopback port used by the Inspector CLI and WSL wrapper |
| `RELIC_UID` | No | `1000` | Compose host user ID for bind-mounted output ownership |
| `RELIC_GID` | No | `1000` | Compose host group ID for bind-mounted output ownership |
| `RELIC_TRACE_FILE` | No | `selected-trace.json` | Inspector trace filename under the read-only Compose `traces/` mount |
| `RELIC_CLAUDE_OPUS_4_6_MODEL` | Formal Claude gateway runs | none | Deployed Claude model alias for the OpenAI-compatible gateway |
| `APPTAINER_CACHEDIR` | Apptainer only | runtime default | Optional Apptainer cache path |
| `APPTAINER_TMPDIR` | Apptainer only | runtime default | Optional Apptainer temporary path |

Internal `ORG_*` identity variables are set per cell by the canonical worker.
Users must not set them to redefine an experiment. `ORG_LLM_API_KEY` and
`ORG_LLM_BASE_URL` are advanced aliases for the documented OpenAI-compatible
credential and endpoint variables. The frozen model YAML, rather than an
environment override, fixes the formal worker's wire API and JSON transport.

`check-env` validates the packaged Inspector assets. Inspector reads only a
strict `relic-trace-v1` file supplied with `--trace`; it never reads private
runtime or evaluator directories. The selected paper trace set is still an
author-asset dependency and is not fabricated from aggregate results.

## Outputs, cache, and concurrency

Outputs default to `outputs/` and are ignored by Git. Scheduler logs are private
local artifacts under `<output-root>/private/scheduler/`; cell checkpoints and
evaluator evidence stay in each cell's `private/` directory. Do not resume or
evaluate checkpoint pickle files obtained from an untrusted source.

Budget approximately 16 GiB of visible RAM per active cell:

| Visible RAM | Conservative maximum |
|---:|---:|
| below 32 GiB | 1 |
| 32–63 GiB | 2 |
| 64–99 GiB | 4 |
| 100+ GiB | up to 8 |
| 128 GiB | recommended for 8 |

On Windows, check both `.wslconfig` and Docker Desktop limits before selecting
four or more workers. The scheduler defaults to one and warns when requested
parallelism exceeds the memory-based recommendation. The controller currently
runs multiple cell subprocesses inside one container rather than one Compose
container per cell, but the same conservative 16 GiB per active cell budget
applies. Docker examples default to one worker; do not scale the Compose
service as a substitute for the canonical scheduler.

## Windows / WSL2

From PowerShell, first confirm that WSL2 and a Linux distribution are present.
Set the Linux repository path rather than a Windows drive path:

```powershell
$env:RELIC_WSL_DISTRIBUTION = "Ubuntu"
$env:RELIC_WSL_REPO = "/home/<user>/relic"
$wrapperRoot = "\\wsl.localhost\Ubuntu\home\<user>\relic\scripts\powershell"
& "$wrapperRoot\check_wsl.ps1"
& "$wrapperRoot\check_env.ps1" -ScriptArguments @('--scope', 'core')
& "$wrapperRoot\smoke.ps1" -ScriptArguments @('--mode', 'mock')
& "$wrapperRoot\start_inspector.ps1" `
  -ScriptArguments @('--trace', '/home/<user>/selected-trace.json')
```

On systems exposing the older UNC alias, replace `\\wsl.localhost\Ubuntu`
with `\\wsl$\Ubuntu`. The scripts themselves still execute all Relic logic
inside the selected WSL2 distribution.

The wrappers invoke the corresponding `scripts/bash/` command inside WSL and
forward arguments. They do not parse benchmarks, create experiment matrices,
or manage evaluator state.
