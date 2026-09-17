# CooperBench B3-2 release path

This is the `cooper` branch boundary for the paper's external CooperBench
extension. It preserves the source B3-2 adapter, configurations, compatibility
patch, focused tests, and the exact fixed 48-pair selection. It does not
redistribute CooperBench, its dataset, task images, hidden tests, or official
evaluator.

The paper reports 29/48 Relic B3-2 successes with Claude Opus 4.6 at high
reasoning effort. That is a paper result in
[`artifacts/paper_results/`](../../artifacts/paper_results/), not a claim that
the historical run artifacts are shipped here.

## Source-fixed paper selection

The selection is not reconstructed from the PDF or substituted with the
652-pair benchmark. It is the exact union used by the dedicated source branch:

- [`b3_v108_new16_b001.json`](../../configs/cooperbench/batches/b3_v108_new16_b001.json):
  16 pairs, source blob `5249c727379fa28fad1ae95e3a5aa121e113b47b`, SHA-256
  `41c82038b34a1e1548e7f401ccd89f7c4cb05effddb67a59bb58043936b7df2a`;
- [`b3_v128_expand32_b001.json`](../../configs/cooperbench/batches/b3_v128_expand32_b001.json):
  32 pairs, source blob `060c1fd67b7382bd134fb17392c0b1fa4f8faeca`, SHA-256
  `78da2c612b63aaf38a16839328d7c44f9f7d7ba471a97df23c8368a226a27fa8`.

They are verbatim copies from
`SocioGenesis/codex/cooperbench-b3-two-agent@bbe7c0ad47ada83a710e90b5436f98745586bd83`.
Their union is exactly 48 distinct feature pairs across 30 task instances and
12 repositories. The source's later combined launcher calls this exact
external dataset subset:

```text
b3_v133_combined48_b001
```

`relic check-cooper`, `preflight-cooper`, `run-cooper`, and
`evaluate-cooper` first verify the two local source-file hashes and then
require that the external subset has exactly the same pair-key set. A 652-pair
fallback is never accepted.

## B3-2 provenance

B3-2 is the paper's external two-member adaptation, not an eight-member main
study cell. Its source-backed treatment uses Victor and Calvin, one initially
owned public feature per member, B3 mechanisms, and a merged joint patch. The
official CooperBench evaluator decides a pair only when
`eval.json::both_passed` is true.

The Cooper-only module is the documented exception to the default HCI-source
provenance:

```text
source branch: codex/cooperbench-b3-two-agent
source head:   bbe7c0ad47ada83a710e90b5436f98745586bd83
implementation baseline: b872386c6f9dc1c96895641cc3b303f6b2569ff2
```

## External upstream setup

Use the external CooperBench release rather than copying it into this
repository. Its v0.0.29 package metadata declares MIT; this branch does not
redistribute that repository or its dataset.

```bash
export RELIC_ROOT="$(pwd)"
export COOPER_ROOT=/absolute/path/CooperBench
export DATASET_ROOT=/absolute/path/cooperbench-dataset
export COOPER_VENV=/absolute/path/.cooper-venv
export COOPER_PY="$COOPER_VENV/bin/python"
export COOPERBENCH="$COOPER_VENV/bin/cooperbench"

git clone --branch v0.0.29 --depth 1 \
  https://github.com/cooperbench/CooperBench.git "$COOPER_ROOT"
test "$(git -C "$COOPER_ROOT" rev-parse HEAD)" = \
  4913c4ebb84d2606cdb5628936b88529f3e181df

uv venv --python 3.12 "$COOPER_VENV"
uv pip install --python "$COOPER_PY" -e "$COOPER_ROOT[dev]"
uv pip install --python "$COOPER_PY" -e "$RELIC_ROOT"

git -C "$COOPER_ROOT" apply --check \
  "$RELIC_ROOT/tools/patches/cooperbench-v0.0.29-docker-eval-timeout.patch"
git -C "$COOPER_ROOT" apply \
  "$RELIC_ROOT/tools/patches/cooperbench-v0.0.29-docker-eval-timeout.patch"
```

The narrow timeout patch is source-backed compatibility material for that
exact tag; do not apply it to another upstream revision. The source runbook
pins the required data snapshot to
`CooperBench/cooperbench-dataset@b612b1a35af722751454813d9e5a7888f065fc9e`:

```bash
"$COOPER_PY" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="CooperBench/cooperbench-dataset",
    repo_type="dataset",
    revision="b612b1a35af722751454813d9e5a7888f065fc9e",
    local_dir="$DATASET_ROOT",
)
PY
test -f "$DATASET_ROOT/subsets/b3_v133_combined48_b001.json"
```

Start the upstream-required Redis service, pull task images through the
upstream dataset flow, and keep data, image cache, and logs outside this
checkout. The adapter requires Docker; Relic's core Compose image deliberately
does not include a nested Docker runtime.

## Gateway boundary

The adapter uses the source's OpenAI-compatible gateway route. It does not
install or implement a native Anthropic SDK adapter. Before a paid run, export
credentials outside Git:

```bash
export ORG_LLM_ENABLED=1
export ORG_LLM_PROVIDER=openai
export ORG_LLM_BASE_URL='https://gateway.example/v1'
export ORG_LLM_API_KEY='<secret>'
export ORG_LLM_WIRE_API=chat_completions
export ORG_LLM_JSON_TRANSPORT=prompt_only
export ORG_LLM_REASONING_EFFORT=high
export ORG_LLM_MODEL='<gateway alias for the reported Claude Opus 4.6 model>'
```

The gateway alias is the exact value passed to upstream `-m` and should be
recorded with a new run. It can differ from the paper's reported model label;
the source runbook explicitly distinguishes the two. `run-cooper` adds only
the source adapter import seam (`PYTHONPATH` and
`COOPERBENCH_EXTERNAL_AGENTS`) to its child process. It never prints or saves
credentials.

## Entry points

All commands are thin boundaries over the source-owned public preflight or the
pinned upstream v0.0.29 CLI.

After the external setup is complete, `scripts/bash/run_cooperbench.sh` is the
copyable Linux/WSL shortcut for `relic run-cooper`; it forwards its arguments
unchanged and does not guess any external path, image, credential, or model
alias. On Windows, use `scripts/powershell/run_cooperbench.ps1` from the WSL
repository share after setting `RELIC_WSL_DISTRIBUTION` and `RELIC_WSL_REPO` as
described in [`docs/environment.md`](../../docs/environment.md#windows--wsl2).
The PowerShell script only delegates to WSL2. Preflight and evaluation remain
their separately named CLI entrypoints below.

```bash
# Read-only source-selection and external-input check.
relic check-cooper

# With external dependencies prepared, verify the external subset and gateway contract.
relic check-cooper \
  --cooperbench-root "$COOPER_ROOT" \
  --cooperbench-bin "$COOPERBENCH" \
  --dataset-dir "$DATASET_ROOT" \
  --model "$ORG_LLM_MODEL" \
  --check-provider

# One source-selected, zero-provider preflight. --image is the upstream task-image
# reference for the selected pair; it is not inferred or substituted by Relic.
relic preflight-cooper \
  --pair-key 'dottxt_ai_outlines_task:1371:1,2' \
  --image '<upstream-task-image-reference>' \
  --dataset-dir "$DATASET_ROOT" \
  --output /absolute/empty/preflight-output

# Delegate all and only the source-fixed 48 pairs to upstream.
relic run-cooper \
  --cooperbench-root "$COOPER_ROOT" \
  --cooperbench-bin "$COOPERBENCH" \
  --dataset-dir "$DATASET_ROOT" \
  --log-dir /durable/cooper-runs \
  --run-name relic-b3-2-paper48-r001 \
  --model "$ORG_LLM_MODEL" \
  --concurrency 1 --eval-concurrency 1

# Resume uses the same run name and does not add --force.
relic run-cooper --resume ...same arguments...

# Delegate official evaluation for that same fixed subset.
relic evaluate-cooper \
  --cooperbench-root "$COOPER_ROOT" \
  --cooperbench-bin "$COOPERBENCH" \
  --dataset-dir "$DATASET_ROOT" \
  --log-dir /durable/cooper-runs \
  --run-name relic-b3-2-paper48-r001

# Print upstream-generated summary.json byte-for-byte; Relic does not score it.
relic cooper-summary --log-dir /durable/cooper-runs --run-name relic-b3-2-paper48-r001
```

Use `--dry-run` on `preflight-cooper`, `run-cooper`, or `evaluate-cooper` to
inspect the exact command boundary. Neither run nor evaluation exposes
`--force`, a best-of-attempt option, or a 652-row fallback.

The source public preflight itself rejects task ID 0, so it cannot preflight
the two selected `openai_tiktoken_task:0` pairs without a source change. That
does not affect upstream run/evaluation, which accepts the source fixed subset;
it is an explicit limitation of the unchanged optional preflight tool.

## Result boundary and genuine release gaps

The upstream runner generates `summary.json`; `relic cooper-summary` validates
only that it is JSON and writes those exact bytes to stdout. It does not parse
per-pair results, calculate a rate, turn a partial denominator into 48, or
replace official `eval.json` verdicts.

The release still lacks, and does not fabricate:

- historical 29-success raw trajectories, patches, receipts, and official
  evaluator outputs;
- the historical task-image digest ledger that bound those prior runs; and
- a root `LICENSE` for Relic itself.

Therefore the public adapter can reproduce a new source-verified 48-pair run,
but this checkout does not claim to contain the historical artifact set behind
the paper's 29/48 result.
