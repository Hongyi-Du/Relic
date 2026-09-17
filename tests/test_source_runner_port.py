"""Release checks for the hci source-backed paired reproduction path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from environments.org_env.config.baseline_conditions import resolve_condition
from environments.org_env.runtime_adapter.live import OrgInspectorSession
from environments.org_env.runtime_adapter.replay_delta import expand_delta_replay
from relic.source_runner import (
    SourceMainRunnerError,
    _source_invocation_environment,
    build_source_main_manifest,
    run_source_main,
)
from tools.run_org_baselines import build_case_environment, main as baseline_main


def test_source_b3_is_canonical_and_early_relic_id_is_read_compatibility() -> None:
    assert resolve_condition("b3").condition_id == "b3_full_sociogenesis"
    assert resolve_condition("b3_full_sociogenesis").condition_id == "b3_full_sociogenesis"
    assert (
        resolve_condition("b3_relic_organization").condition_id
        == "b3_relic_organization"
    )


def test_source_main_plan_has_30_paired_batches_and_120_source_cells(tmp_path: Path) -> None:
    payload = build_source_main_manifest(
        model="gpt-5.6-terra", output_root=tmp_path, max_parallel=2
    )
    plan = payload["plan"]
    batches = plan["batches"]

    assert plan["source"]["baseline_runner"] == "tools/run_org_baselines.py"
    assert plan["source"]["execution_profile"] == "native"
    assert plan["mechanism_ablations"] == ["work_rhythm"]
    assert len(batches) == 30
    assert sum(len(batch["cases"]) for batch in batches) == 120
    assert {batch["pack"] for batch in batches} == {
        "mini_blobstore_v1",
        "traffic_watch_v1",
        "tg_automation_v1",
        "pdf_reformatter_v1",
        "fastapi_dashboard_v1",
        "boltons_v2400_to_v2610",
        "celery_v560_to_v563",
        "soupsieve_v26_to_v291",
        "cattrs_v2510_to_v2610",
        "tenacity_v823_to_v914",
    }
    assert {batch["seed"] for batch in batches} == {1401, 2711, 4013}
    assert all(
        [case["condition_id"] for case in batch["cases"]]
        == [
            "b0_single_agent_founder",
            "b1_persistent_role_org",
            "b2_policy_conditioned_org",
            "b3_full_sociogenesis",
        ]
        for batch in batches
    )
    assert plan["resource_ceilings"] == {
        "max_llm_calls": 6000,
        "max_llm_requested_tokens": 20_000_000,
        "max_llm_prompt_characters": 100_000_000,
        "max_primary_actions": 2688,
        "max_ticks": 336,
    }


def test_direct_source_runner_dry_run_writes_four_source_case_plans(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "baseline"
    assert (
        baseline_main(
            [
                "--dry-run",
                "--llm",
                "--provider",
                "openai",
                "--model",
                "gpt-5.6-terra",
                "--dataset",
                "mini_blobstore_v1",
                "--repository-id",
                "mini_blobstore_v1",
                "--seed",
                "1401",
                "--ticks",
                "336",
                "--checkpoint-every",
                "24",
                "--sprint-ticks",
                "168",
                "--max-llm-calls",
                "6000",
                "--max-llm-requested-tokens",
                "20000000",
                "--max-llm-prompt-characters",
                "100000000",
                "--max-primary-actions",
                "2688",
                "--max-ticks",
                "336",
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert {row["case"] for row in manifest["cases"]} == {"b0", "b1", "b2", "b3"}
    assert {row["status"] for row in manifest["cases"]} == {"dry_run"}
    assert {row["condition_id"] for row in manifest["cases"]} == {
        "b0_single_agent_founder",
        "b1_persistent_role_org",
        "b2_policy_conditioned_org",
        "b3_full_sociogenesis",
    }
    source_text = (Path(__file__).parents[1] / "tools" / "run_org_baselines.py").read_text(
        encoding="utf-8"
    )
    assert "from society_core" not in source_text


def test_programbench_profile_fails_closed_before_environment_construction() -> None:
    with pytest.raises(
        ValueError, match="^programbench_execution_profile_not_available_in_relic_release$"
    ):
        build_case_environment(
            {},
            condition_id="b0_single_agent_founder",
            dataset="mini_blobstore_v1",
            seed=1401,
            sprint_ticks=168,
            llm=False,
            llm_actions=False,
            run_tag="test",
            execution_profile="programbench_leaderboard_v1",
        )


def test_source_case_environment_bridges_one_explicit_evaluator_binding() -> None:
    environment = build_case_environment(
        {},
        condition_id="b3_full_sociogenesis",
        dataset="mini_blobstore_v1",
        seed=1401,
        sprint_ticks=168,
        llm=True,
        llm_actions=False,
        run_tag="test",
        evaluator_backend="docker",
        evaluator_container_image="registry.example/relic@sha256:" + "a" * 64,
        evaluator_container_platform="linux/amd64",
        expected_evaluator_environment_hash="b" * 64,
        expected_qualification_plan_hash="c" * 64,
    )
    assert environment["ORG_EVALUATOR_BACKEND"] == environment["RELIC_EVALUATOR_BACKEND"]
    assert environment["ORG_EVALUATOR_CONTAINER_IMAGE"] == environment[
        "RELIC_EVALUATOR_CONTAINER_IMAGE"
    ]
    assert environment["ORG_EVALUATOR_CONTAINER_PLATFORM"] == environment[
        "RELIC_EVALUATOR_CONTAINER_PLATFORM"
    ]
    assert environment["ORG_EVALUATOR_EXPECTED_ENVIRONMENT_HASH"] == environment[
        "RELIC_EVALUATOR_EXPECTED_ENVIRONMENT_HASH"
    ]
    assert environment["ORG_EVALUATOR_EXPECTED_QUALIFICATION_HASH"] == environment[
        "RELIC_EVALUATOR_EXPECTED_QUALIFICATION_HASH"
    ]


def test_strict_source_main_requires_bindings_before_source_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("a missing evaluator binding must stop before source invocation")

    monkeypatch.setattr("relic.source_runner._invoke_source_batch", forbidden)
    with pytest.raises(SourceMainRunnerError, match="^formal_evaluator_bindings_required$"):
        run_source_main(
            model="gpt-5.6-terra",
            output_root=tmp_path / "formal",
            strict_reproducibility=True,
            workloads=("w01",),
            seeds=(1401,),
        )


def test_source_main_dry_run_invokes_one_paired_source_batch_and_scopes_runtime_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORG_LLM_WIRE_API", "ambient_should_be_restored")
    result = run_source_main(
        model="gpt-5.6-terra",
        output_root=tmp_path,
        dry_run=True,
        workloads=("w01",),
        seeds=(1401,),
    )
    assert result.status == "planned"
    assert result.selected_batches == result.completed_batches == 1
    assert result.failed_batches == 0
    assert os.environ["ORG_LLM_WIRE_API"] == "ambient_should_be_restored"
    source_manifest = json.loads(
        (tmp_path / "source-dry-run" / "w01__seed1401" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(source_manifest["cases"]) == 4
    assert source_manifest["pack"] == "mini_blobstore_v1"
    assert {
        case["environment"]["ORG_MECHANISM_ABLATIONS"]
        for case in source_manifest["cases"]
    } == {"work_rhythm"}


def test_documented_openai_variables_bridge_only_for_source_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ORG_LLM_API_KEY", raising=False)
    monkeypatch.delenv("ORG_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ORG_LLM_DEFAULT_HEADERS_JSON", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-provider-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("RELIC_OPENAI_DEFAULT_HEADERS_JSON", '{"x-route":"test"}')
    values = _source_invocation_environment(
        {
            "ORG_LLM_PROVIDER": "openai",
            "ORG_LLM_MODEL": "gpt-5.6-terra",
            "ORG_LLM_WIRE_API": "responses",
            "ORG_LLM_JSON_TRANSPORT": "native",
            "ORG_LLM_REASONING_EFFORT": "low",
            "ORG_LLM_REQUEST_TIMEOUT_SECONDS": "180",
            "ORG_LLM_MAX_RETRIES": "6",
            "ORG_LLM_RETRY_BACKOFF_SECONDS": "3",
            "ORG_LLM_STORE_RESPONSES": "0",
        }
    )
    assert values["ORG_LLM_API_KEY"] == "test-provider-secret"
    assert values["ORG_LLM_BASE_URL"] == "https://provider.example/v1"
    assert values["ORG_LLM_DEFAULT_HEADERS_JSON"] == '{"x-route":"test"}'


def test_live_session_source_snapshot_and_delta_replay_round_trip(tmp_path: Path) -> None:
    session = OrgInspectorSession(seed=7, load_llm=False)
    session.step(1)
    full = session.to_replay("source-port")
    assert full["frames"][-1]["scenario"]["experiment_condition"] == "b3_full_sociogenesis"
    assert full["frames"][-1]["graphs"]["persona"]
    path = tmp_path / "replay.json"
    session.save_replay(str(path), name="source-port", delta=True)
    saved = json.loads(path.read_text(encoding="utf-8"))
    # The on-disk replay is JSON. Its graph/log/timeline sections are intentionally
    # keyframed, while all world-state fields round-trip tick-for-tick.
    expected = json.loads(json.dumps(full["frames"], ensure_ascii=False))
    expanded = expand_delta_replay(saved)
    assert [frame["tick"] for frame in expanded] == [0, 1]
    for observed, original in zip(expanded, expected, strict=True):
        assert {
            key: value
            for key, value in observed.items()
            if key not in {"graphs", "logs", "timeline"}
        } == {
            key: value
            for key, value in original.items()
            if key not in {"graphs", "logs", "timeline"}
        }
        assert {"graphs", "logs", "timeline"} <= set(observed)
