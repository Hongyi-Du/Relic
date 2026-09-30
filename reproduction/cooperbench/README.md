# CooperBench B3-2: frozen full-652 reference

This branch publishes the **final bug-fixed reference implementation** for the
two-member Relic B3-2 extension. The runtime is in [`reference/`](reference/),
with its frozen SDK and organizational-runtime dependency closure.
`relic run-cooper` puts that directory first on the external runner's
`PYTHONPATH`; it does not use the older core-study adapter.

Internal repair-batch numbers and private provider aliases are not public
release names. New requests and outcomes use `relic_cooperbench_b3_two_agent`.
Schema identifiers, upstream revisions, and third-party package versions remain
intact because they are needed for compatibility and reproducibility.

## Scope and artifacts

- Exactly **652 pairs / 30 task instances / 12 repositories** in
  [`full652.json`](../../configs/cooperbench/full652.json). All pairs, including
  documented exclusions, remain in this manifest; there is no result-dependent
  filtering by the launcher.
- Victor and Calvin, one initially assigned public feature each, B3 mechanisms,
  reciprocal review, and the identical merged joint patch returned to both
  external agents.
- The pinned **official CooperBench evaluator** scores the pair. Internal
  probes/reviews are not official verdicts. PASS requires
  `eval.json::both_passed`.
- Historical results, trajectories, and exclusion QA are on
  [Hugging Face](https://huggingface.co/datasets/Horseback-Eridute/CooperBench-B3-2-Full-652).

This is the final code for **new runs**, not a claim that every historical
trajectory used identical source bytes. Earlier runs and continuations used
earlier fixes. The older 48-pair paper snapshot remains historical material in
`artifacts/paper_results/`; it is no longer the default execution scope.
Owner-labeled BROKEN exclusions are not claimed to have been endorsed by the
benchmark maintainers.

The historical release now records 371 retained PASS / 100 analysis FAIL /
181 proposed BROKEN, with a retained-task success rate of 371/471 (78.77%).
The 100 failures comprise 98 official evaluator FAIL and 2 internal method
failures with no official verdict. Internal probe/review delivery is part of
our method, so those two failures remain in its denominator. See
[evaluation methods and execution-layer deviations](EVALUATION_METHOD.md) for
the benchmark pins, actual Docker/patch-transport changes, and score denominator.

The historical backend family is Claude Opus 4.6, based on experiment-owner
attestation. Your runtime model is whatever your provider actually serves under
`ORG_LLM_MODEL`; an alias alone does not establish backend identity. No private
key, gateway address, routing alias, checkpoint, or developer-machine
environment is included or required.

## Setup: Linux / WSL, Python 3.12, Docker, Redis

Use a source checkout; the independent reference is not a substitute for
Relic's core-study runtime:

```bash
git clone --branch cooper https://github.com/Hongyi-Du/Relic.git
cd Relic
uv sync --python 3.12 --extra dev --extra cooper --frozen
export RELIC_ROOT="$PWD"
export COOPER_ROOT=/absolute/path/CooperBench
export DATASET_ROOT=/absolute/path/cooperbench-dataset
export COOPER_VENV=/absolute/path/cooper-venv
export COOPER_PY="$COOPER_VENV/bin/python"
export COOPERBENCH="$COOPER_VENV/bin/cooperbench"

git clone --branch v0.0.29 --depth 1 \
  https://github.com/cooperbench/CooperBench.git "$COOPER_ROOT"
test "$(git -C "$COOPER_ROOT" rev-parse HEAD)" = \
  4913c4ebb84d2606cdb5628936b88529f3e181df
git -C "$COOPER_ROOT" apply --check \
  "$RELIC_ROOT/tools/patches/cooperbench-execution.patch"
git -C "$COOPER_ROOT" apply \
  "$RELIC_ROOT/tools/patches/cooperbench-execution.patch"

uv venv --python 3.12 "$COOPER_VENV"
uv pip install --python "$COOPER_PY" -e "$COOPER_ROOT[dev]"
uv pip install --python "$COOPER_PY" -e "$RELIC_ROOT[cooper]"
```

Apply that one execution-only patch on a clean pinned checkout; do not stack it
with the older timeout patch. Install Hugging Face's `hf` CLI in your operator
environment, download the frozen dataset, and copy the complete selection:

```bash
hf download CooperBench/cooperbench-dataset --repo-type dataset \
  --revision b612b1a35af722751454813d9e5a7888f065fc9e \
  --local-dir "$DATASET_ROOT"
mkdir -p "$DATASET_ROOT/subsets"
cp "$RELIC_ROOT/configs/cooperbench/full652.json" \
  "$DATASET_ROOT/subsets/relic_full652.json"
```

Prepare task images using the upstream instructions and start Redis. Keep
dataset, image cache, and outputs outside this checkout. Docker is required;
Redis defaults to `redis://localhost:6379`.

## Configure your own provider and run

Secrets stay in environment variables or an external secret store:

```bash
export ORG_LLM_ENABLED=1
export ORG_LLM_PROVIDER=openai
export ORG_LLM_BASE_URL='https://your-provider.example/v1'
export ORG_LLM_API_KEY='<your-private-key>'
export ORG_LLM_WIRE_API=chat_completions
export ORG_LLM_JSON_TRANSPORT=prompt_only
export ORG_LLM_REASONING_EFFORT=high
export ORG_LLM_MODEL='<your-provider-model-name>'

uv run relic check-cooper \
  --cooperbench-root "$COOPER_ROOT" --cooperbench-bin "$COOPERBENCH" \
  --dataset-dir "$DATASET_ROOT" --model "$ORG_LLM_MODEL" --check-provider

uv run relic run-cooper \
  --cooperbench-root "$COOPER_ROOT" --cooperbench-bin "$COOPERBENCH" \
  --dataset-dir "$DATASET_ROOT" --log-dir /durable/cooper-runs \
  --run-name relic-full652 --model "$ORG_LLM_MODEL" \
  --concurrency 40 --eval-concurrency 16

uv run relic evaluate-cooper \
  --cooperbench-root "$COOPER_ROOT" --cooperbench-bin "$COOPERBENCH" \
  --dataset-dir "$DATASET_ROOT" --log-dir /durable/cooper-runs \
  --run-name relic-full652 --concurrency 16

uv run relic cooper-summary \
  --log-dir /durable/cooper-runs --run-name relic-full652
```

The OpenAI-compatible transport does not imply an OpenAI backend. The frozen
client has thinking-model output budgeting and bounded transport retries. Any
provider-added context belongs to that route and should be disclosed for a new
comparison; this code does not remove it.

Use a **new run name and empty output namespace** for a fresh run. Reusing a
name allows upstream terminal-result skipping and checkpoint recovery.
`--dry-run` prints commands without provider calls. No command adds `--force`,
best-of-attempt selection, or a PASS override. Defaults are 1,000 ticks, 500
logical calls per member, 900 seconds per provider attempt, and 3,600 seconds
per logical call. Inherited LLM health checks are operational diagnostics, not
an added criterion used to invalidate historical official PASS results.

`cooper-summary` emits upstream `summary.json` unchanged; it does not apply
owner exclusions or recompute an adjusted denominator.

## Fixes, execution patch, and regressions

The final source includes stale-PR delivery-before-governance routing,
public source/anchor grounding, resumed identity checks, public-contract
boundary composition, peer-confirmed defective-probe isolation (genuine failing
probes stay red, with no automatic approval), and thinking-model budgeting.

The external patch contains bounded Docker timeout handling, fail-closed
reuse of a terminated sandbox, configurable execution timeout, and chunked
patch transfer. Test bodies, pass/fail parsing, patch filtering, merge rules,
and official `both_passed` scoring remain upstream-owned.

```bash
uv run python tools/test_cooperbench_reference.py
uv run pytest -q tests/test_cooperbench_release.py
```

The reference suite has 182 provider-free regressions, not 182 benchmark
successes. Docker and paid end-to-end runs are separate. This code uses Relic's
existing [PolyForm Noncommercial license](../../LICENSE); CooperBench retains
its upstream MIT licensing. Historical operation labels remain in Git history,
but they are not names for this frozen public implementation.
