# Final reproduction run report

Date: 2026-09-17

This is one final execution of the public entry points. It records what ran,
what stopped at an intentional boundary, and what it does not establish. No
paid model provider was called.

## Tested heads

| Checkout | Branch / role | Head |
| --- | --- | --- |
| Relic main-equivalent worktree | `main-transfer-release-port` (designated canonical-main equivalent in the handoff) | `57af578847682ff5e34b03865c33c96106918bb8` |
| HCI worktree | `hci` | `25d6fd7d1a954307d8d2608f755ebf8be566ea3b` |
| Cooper worktree | `cooper` | `9e869258f19bed6c66f64d929f6a1e949e13b707` |

The main worktree was clean before the documentation changes. Runtime artifacts
were written outside the checkout at
`/tmp/relic-final-repro-20260917.Ql0mgu` (except the source runner's ignored
`log/` output).

## Results

### Main study design

`configs/main-study.yaml` declares the complete 240-cell design: two models,
ten workloads, three seeds, and four arms (120 cells per model).

    UV_CACHE_DIR=/tmp/relic-final-uv-cache uv run --frozen relic run-main --model gpt-5.6-terra --output-root /tmp/relic-final-repro-20260917.Ql0mgu/main-gpt-5.6-terra --manifest /tmp/relic-final-repro-20260917.Ql0mgu/main-gpt-5.6-terra/source_main_manifest.json --max-parallel 1 --dry-run

Passed. The CLI wrote 30 paired source batch plans / 120 selected cells, with
zero failed batches and no provider or condition subprocess.

    UV_CACHE_DIR=/tmp/relic-final-uv-cache uv run --frozen relic run-main --model claude-opus-4.6 --output-root /tmp/relic-final-repro-20260917.Ql0mgu/main-claude-opus-4.6 --manifest /tmp/relic-final-repro-20260917.Ql0mgu/main-claude-opus-4.6/source_main_manifest.json --max-parallel 1 --dry-run

Stopped as expected before creating a plan:
`runtime_model_binding_missing:RELIC_CLAUDE_OPUS_4_6_MODEL`. The missing
author/operator gateway alias was not replaced with a fabricated value. Thus
the declared design is 240 cells, but this run materialized only the GPT plan;
the Claude plan remains an explicit gap.

### Final transfer v2

    UV_CACHE_DIR=/tmp/relic-final-uv-cache uv run --frozen relic run-transfer --dry-run --output-root /tmp/relic-final-repro-20260917.Ql0mgu/transfer-v2 --manifest /tmp/relic-final-repro-20260917.Ql0mgu/transfer-v2/transfer_manifest.json --max-parallel 1

Passed. The complete 10 workloads x 3 seeds x 2 target arms plan was written:
60 selected target runs across 30 workload/seed batches. No provider or source
condition subprocess started.

    UV_CACHE_DIR=/tmp/relic-final-uv-cache uv run --frozen relic run-transfer --manifest /tmp/relic-final-repro-20260917.Ql0mgu/transfer-v2/transfer_manifest.json --output-root /tmp/relic-final-repro-20260917.Ql0mgu/transfer-v2 --resume --max-parallel 1

Exited 2 with `formal_evaluator_bindings_required`. This is the intended
fail-closed result before any source child or provider request.

### Source-native mock trace, replay, and Inspector

    ORG_LLM=0 ORG_LOG_ZIP=0 ORG_REPLAY_MAX_FRAMES=4 ORG_FIGURE_METRICS=0 UV_CACHE_DIR=/tmp/relic-final-uv-cache uv run --frozen python tools/org_inspector_replay.py release_trace 2 1401

The command exited 0 after a two-tick rule/template run with zero provider
calls. It wrote
`log/20260917-103654_org_release_trace_seed1401_mock/public/relic-trace-v1.json`.
The generated run record correctly has status `failed` with
`final_evaluation_not_produced:run_oss_final_evaluation_disabled`: this is a
mock trace exercise, not a formal scored experiment.

    UV_CACHE_DIR=/tmp/relic-final-uv-cache uv run --frozen relic replay --trace log/20260917-103654_org_release_trace_seed1401_mock/public/relic-trace-v1.json
    UV_CACHE_DIR=/tmp/relic-final-uv-cache uv run --frozen relic export-trace --trace log/20260917-103654_org_release_trace_seed1401_mock/public/relic-trace-v1.json --output /tmp/relic-final-repro-20260917.Ql0mgu/release_trace_copy.json

Both passed: `relic-trace-v1`, 8 public frames, 2 ticks, 8 agents, and SHA-256
`d82a88252e0b3fbfcf110adfd42559f97939e8daa9b9f37332798b93903d7acf`.

    UV_CACHE_DIR=/tmp/relic-final-uv-cache uv run --frozen relic inspect --trace log/20260917-103654_org_release_trace_seed1401_mock/public/relic-trace-v1.json --port 18765
    curl --fail --silent --show-error http://127.0.0.1:18765/api/health
    curl --fail --silent --show-error http://127.0.0.1:18765/api/trace

The local Inspector started in replay mode and both endpoints passed. Health
reported `status: ready`, `trace_schema_version: relic-trace-v1`, run ID
`org_org_default_1401`, and last verified frame 7. The temporary server was
then stopped.

### Local source-derived evaluator

    UV_CACHE_DIR=/tmp/relic-final-uv-cache uv run --frozen relic evaluator-build --smoke
    UV_CACHE_DIR=/tmp/relic-final-uv-cache uv run --frozen relic evaluator-hashes --dataset mini_blobstore_v1 --backend docker --container-image sha256:32f68696ea9781bdc669140125612dbda65eb480343f7eb172b1e047329bc832 --container-platform linux/amd64
    UV_CACHE_DIR=/tmp/relic-final-uv-cache uv run --frozen relic evaluator-preflight --repository-id mini_blobstore_v1 --dataset mini_blobstore_v1 --backend docker --container-image sha256:32f68696ea9781bdc669140125612dbda65eb480343f7eb172b1e047329bc832 --container-platform linux/amd64 --expected-environment-hash 87e40c9831622da917a9a6c2931c4156c46f0dc2c4ae0775015d5c1a7509f1a5 --expected-qualification-hash 7be2b7814648b6c1a2449576581c01e5af5ace1edc640c0eea1d4eca37a5961d --output /tmp/relic-final-repro-20260917.Ql0mgu/evaluator-preflight.json

All three passed on `linux/amd64`. The build created local image ID
`sha256:32f68696ea9781bdc669140125612dbda65eb480343f7eb172b1e047329bc832`;
the one-pack preflight passed with five oracles and attestation hash
`864a6686009d6a929277d33d8e3b8d2fdddc03ddf0f19d42094e1d18ceda5dea`.
These are local source-closure diagnostics only, not a paper evaluator binding.

### HCI minimal smoke

    ORG_LLM=0 .venv/bin/python -c 'import json; from environments.org_env.backend import main; assert main.SESSION is None; packs = main.api_packs(); init = main.api_init_world("mini_blobstore_v1", warmup=0, clock_speed=15.0, start_runtime=False); stepped = main.api_sim_step(1); state = main.api_state(); print(json.dumps({"default_setup_uninitialized": True, "pack_count": len(packs), "selected_pack": init.get("pack"), "init_ok": init.get("ok"), "runtime_running": init.get("runtime", {}).get("running"), "step_tick": stepped.get("tick"), "state_tick": state.get("tick"), "llm_attached": state.get("llm", {}).get("attached")}, sort_keys=True)); main.HUMAN.shutdown()'

Passed on the HCI checkout. The default setup was uninitialized, 10 packs were
listed, `mini_blobstore_v1` was selected from its real OSS time-machine source,
the runtime remained paused, a single tick completed, and no LLM was attached.

### Cooper boundary

Historical check of the earlier 48-pair entrypoint. The current branch now
publishes the final full-652 reference; use `reproduction/cooperbench/README.md`
instead of these old setup commands.

    UV_CACHE_DIR=/tmp/relic-cooper-uv-cache uv run --frozen relic check-cooper
    UV_CACHE_DIR=/tmp/relic-cooper-uv-cache uv run --frozen relic preflight-cooper --pair-key dottxt_ai_outlines_task:1371:1,2 --image unavailable-task-image --dataset-dir /tmp/cooperbench-v0.0.29-inspect/dataset --output /tmp/relic-final-repro-20260917.Ql0mgu/cooper-preflight --dry-run
    UV_CACHE_DIR=/tmp/relic-cooper-uv-cache uv run --frozen relic run-cooper --cooperbench-root /tmp/cooperbench-v0.0.29-inspect --cooperbench-bin cooperbench --dataset-dir /tmp/cooperbench-v0.0.29-inspect/dataset --log-dir /tmp/relic-final-repro-20260917.Ql0mgu/cooper-runs --run-name relic-final-preflight --model unconfigured-gateway --concurrency 1 --eval-concurrency 1 --dry-run

All three correctly stopped without a benchmark run. `check-cooper` verified
the 48-pair local source selection but reported the absent external checkout,
CLI, and dataset directory. The supplied inspected v0.0.29 checkout has the
correct commit but lacks the historical combined-48 subset; both preflight and
run dry-run therefore stopped with `cooperbench_paper_subset_missing`, before
an unavailable task image or gateway could be used.

## Bottom line

This run demonstrates the public no-provider paths, local evaluator closure,
and intentional fail-closed gates. It does not reproduce a formal main/transfer
cell, a Claude gateway run, a selected historical trace, or a CooperBench
result. The remaining inputs and release-publication gaps are listed in
[`KNOWN_RELEASE_GAPS.md`](KNOWN_RELEASE_GAPS.md).
