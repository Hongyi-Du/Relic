# Environment and platform support

Linux is the canonical Relic runtime. Windows users run the same Linux runtime
through WSL2; the PowerShell files are launchers only and never execute Relic
core natively. Docker Desktop on Windows must use its WSL2 backend.

| Platform | Support level | Recommended path |
|---|---|---|
| Linux | Full | Native Bash / Python |
| Windows 11 + WSL2 | Full | WSL2 + Bash / Python |
| Windows-native PowerShell | Launcher only | PowerShell invokes WSL2 |
| Docker on Linux | Pending release acceptance | Docker / Compose after image delivery |
| Docker Desktop on Windows | Pending release acceptance | WSL2 backend after image delivery |
| macOS | Not yet validated | Docker path planned |

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
runtime supports OpenAI only; `claude-opus-4.6` remains in the paper design but
cannot run until a reviewed Anthropic adapter is added.

The Apptainer backend additionally requires an active Slurm allocation
(`SLURM_JOB_ID` plus `srun`) and the configured registry digest to already be
present in `apptainer cache list -v`. Environment checks never pull images.

## Environment variables

CLI arguments and canonical experiment configuration take precedence over
environment defaults. Cell identity, ticks, arms, seeds, retry policy, and
evaluation policy are frozen by repository configuration and cannot be
overridden from `.env`.

| Variable | Required | Default | Purpose / example format |
|---|---|---|---|
| `OPENAI_API_KEY` | Formal OpenAI runs | none | Provider credential; secret, never persisted by Relic |
| `OPENAI_BASE_URL` | No | official OpenAI endpoint | Optional OpenAI-compatible HTTPS base URL |
| `RELIC_OPENAI_DEFAULT_HEADERS_JSON` | No | `{}` | Non-authorization compatibility headers as a JSON object |
| `RELIC_OPENAI_DISABLE_RESPONSE_STORAGE` | No | `true` | Keep Responses API storage disabled |
| `RELIC_EVALUATOR_BACKEND` | Formal evaluation | none | `docker`, or `apptainer` with Slurm and a preloaded digest cache entry |
| `RELIC_EVALUATOR_CONTAINER_IMAGE` | Formal evaluation | none | Digest-pinned image such as `registry.example/evaluator@sha256:<64 hex>` |
| `RELIC_EVALUATOR_CONTAINER_PLATFORM` | Formal evaluation | none | Must be `linux/amd64` for the frozen study |
| `ORG_OSS_QUALIFICATION_TIMEOUT` | No | `180` | Evaluator qualification timeout in seconds |
| `RELIC_OUTPUT_ROOT` | No | `<repo>/outputs` | Linux/WSL output root |
| `RELIC_BENCHMARK_ROOT` | No | `<repo>/benchmarks` | Advanced frozen benchmark root override |
| `RELIC_CACHE_ROOT` | No | `<output-root>/.relic-cache` | Writable cache location checked by `check-env` |
| `RELIC_CLAUDE_OPUS_4_6_MODEL` | Not yet usable | none | Reserved model binding for a future Anthropic adapter |
| `APPTAINER_CACHEDIR` | Apptainer only | runtime default | Optional Apptainer cache path |
| `APPTAINER_TMPDIR` | Apptainer only | runtime default | Optional Apptainer temporary path |

Internal `ORG_*` identity variables are set per cell by the canonical worker.
Users must not set them to redefine an experiment. `ORG_LLM_API_KEY`,
`ORG_LLM_BASE_URL`, and `ORG_LLM_WIRE_API` exist only as advanced compatibility
overrides; prefer the documented OpenAI variables unless integrating a reviewed
endpoint.

The Inspector entrypoint is not part of the current main-study runtime
milestone, so `check-env` reports it as a warning rather than pretending it is
available. Its port and cache settings will be documented when the separate
Inspector/HCI surface is integrated.

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
parallelism exceeds the memory-based recommendation.

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
```

On systems exposing the older UNC alias, replace `\\wsl.localhost\Ubuntu`
with `\\wsl$\Ubuntu`. The scripts themselves still execute all Relic logic
inside the selected WSL2 distribution.

The wrappers invoke the corresponding `scripts/bash/` command inside WSL and
forward arguments. They do not parse benchmarks, create experiment matrices,
or manage evaluator state.
