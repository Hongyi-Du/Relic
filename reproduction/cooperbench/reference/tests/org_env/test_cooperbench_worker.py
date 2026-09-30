from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from environments.org_env.cooperbench.contract import (
    CONTRACT_SCHEMA_VERSION,
    DELIVERY_MODE,
    TREATMENT_ID,
    PairRequest,
)
from environments.org_env.cooperbench.worker import (
    CooperWorkerError,
    _behavior_probe_capability,
    _bind_feature_owners,
    _configure_worker_environment,
    _export_joint_mainline_patch,
    _merged_feature_issue_ids,
    _minimum_solver_liveness_snapshot,
    _observed_live_llm_preflight,
    _persist_candidate_joint_delivery,
    _PUBLIC_MODULE_PROBE_MARKER,
    _public_runtime_preflight,
    _require_evaluator_runtime_binding,
    _require_all_feature_deliveries_merged,
    _require_live_llm_preflight,
    _require_public_runtime_preflight,
    _save_periodic_checkpoint,
    _require_two_person_sdl_complete,
    _treatment_health,
    _treatment_health_snapshot,
    _terminal_coordination_requirement,
    _transport_payload,
    _usage_delta,
    _verified_public_runtime_preflight,
)
from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.product.objects import ProductArtifact


def _request(tmp_path: Path) -> PairRequest:
    return PairRequest(
        schema_version=CONTRACT_SCHEMA_VERSION,
        treatment_id=TREATMENT_ID,
        delivery_mode=DELIVERY_MODE,
        run_id="run123",
        agents=("agent1", "agent2"),
        tasks={"agent1": "feature one", "agent2": "feature two"},
        image="image",
        model_name="gpt-5",
        backend="docker",
        log_dir=str(tmp_path.resolve()),
        ticks=48,
        seed=6101,
        provider="openai",
        reasoning_effort="high",
        max_surface_files=32,
        max_surface_bytes=1024 * 1024,
    ).validated()


def test_periodic_checkpoint_is_saved_at_each_fifty_tick_boundary(
    tmp_path: Path,
):
    calls = []
    session = SimpleNamespace(
        save_checkpoint=lambda path, meta: calls.append((path, meta))
        or {"path": path, "tick": 50}
    )
    request = _request(tmp_path)

    assert _save_periodic_checkpoint(
        session, tmp_path, request, tick=49
    ) is None
    saved = _save_periodic_checkpoint(
        session, tmp_path, request, tick=50
    )

    assert saved == {
        "path": str(tmp_path / "checkpoints" / "t50.pkl"),
        "tick": 50,
    }
    assert len(calls) == 1
    assert calls[0][1]["schema_version"] == (
        "orgenv_cooperbench_pair_checkpoint_v1"
    )
    assert calls[0][1]["tasks"] == dict(request.tasks)


def test_source_conflict_is_recoverable_but_unresolved_probe_coordination_is_terminal():
    world = SimpleNamespace(
        _cooperbench_coordination_required={"kind": "coordination_conflict"}
    )
    assert _terminal_coordination_requirement(world) is None
    world._cooperbench_coordination_required = {
        "kind": "integrated_probe_failure_requires_coordination"
    }
    assert _terminal_coordination_requirement(world) == (
        world._cooperbench_coordination_required
    )


def test_feature_owners_form_a_bijection_to_the_two_internal_members(tmp_path: Path):
    world = SimpleNamespace(
        agents={
            "victor": SimpleNamespace(active_tasks=["stale"]),
            "calvin": SimpleNamespace(active_tasks=[]),
        },
        tasks={
            "task_oss_cooper_feature_1": SimpleNamespace(owner_id=None, description=""),
            "task_oss_cooper_feature_2": SimpleNamespace(owner_id=None, description=""),
        },
        board=SimpleNamespace(owners={}),
        product_artifacts={
            "cooper_feature_1": SimpleNamespace(owner_agent_id=None),
            "cooper_feature_2": SimpleNamespace(owner_agent_id=None),
        },
        _oss_component_map={
            "cooper_feature_1": ["src/one.py"],
            "cooper_feature_2": ["src/two.py"],
        },
    )

    mapping = _bind_feature_owners(world, _request(tmp_path))

    assert mapping == {"agent1": "victor", "agent2": "calvin"}
    assert world.tasks["task_oss_cooper_feature_1"].owner_id == "victor"
    assert world.tasks["task_oss_cooper_feature_2"].owner_id == "calvin"
    assert world.board.owners == {
        "task_oss_cooper_feature_1": "victor",
        "task_oss_cooper_feature_2": "calvin",
    }
    assert world.agents["victor"].active_tasks == [
        "task_oss_cooper_feature_1"
    ]
    assert world.agents["calvin"].active_tasks == ["task_oss_cooper_feature_2"]
    assert world._cooperbench_feature_issue_by_agent == {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    assert world._cooperbench_issue_predecessor == {}
    assert world._cooperbench_disable_backlog_renewal is True
    assert world._cooperbench_delivery_focus is True
    assert world._cooperbench_main_b3_lifecycle is True
    assert world._institution_wish_cluster_minimum == 2
    assert world._cooperbench_sdl_state["phase"] == "delivery"
    assert world._cooperbench_sdl_state["institutionalization_policy"] == (
        "main_b3_episode_reflection_wish_proposal_pair_approval_adoption"
    )
    assert world._cooperbench_sdl_state["late_protocol_action_policy"] == (
        "model_authored_future_action_with_one_bounded_repair_and_final_merge_hold"
    )
    assert world._cooperbench_sdl_state["wish_cluster_minimum"] == 2
    assert world._cooperbench_sdl_state["overlapping_paths"] == []


def test_pair_focus_runs_the_main_b3_reflection_and_cognition_branches():
    world_source = Path(OrgWorld.__module__.replace(".", "/") + ".py")
    if not world_source.exists():
        world_source = (
            Path(__file__).resolve().parents[2]
            / "environments"
            / "org_env"
            / "backend"
            / "simulation"
            / "world.py"
        )
    source = world_source.read_text(encoding="utf-8")

    reflection_start = source.index("        # -- reflection layer")
    cognition_call = source.index("        self._process_cognition(tick)", reflection_start)
    reflection = source[reflection_start:cognition_call]
    assert '"_cooperbench_main_b3_lifecycle"' in reflection
    assert "self.reflection_batch_manager.maybe_run_batch" in reflection
    cognition_start = source.index("    def _process_cognition(self, tick: int)")
    cognition_end = source.index("    def set_approval_mode", cognition_start)
    cognition = source[cognition_start:cognition_end]
    assert '"_cooperbench_main_b3_lifecycle"' in cognition
    assert "and not suppress_generic_cognition" in cognition
    assert "cluster_wishes(self, self.llm_client)" in cognition
    assert "pm.process_pending_adoptions(self)" in cognition


def test_overlapping_feature_surfaces_use_parallel_actor_desks(tmp_path: Path):
    world = SimpleNamespace(
        agents={
            "victor": SimpleNamespace(active_tasks=[]),
            "calvin": SimpleNamespace(active_tasks=[]),
        },
        tasks={
            "task_oss_cooper_feature_1": SimpleNamespace(owner_id=None, description=""),
            "task_oss_cooper_feature_2": SimpleNamespace(owner_id=None, description=""),
        },
        board=SimpleNamespace(owners={}),
        product_artifacts={
            "cooper_feature_1": SimpleNamespace(owner_agent_id=None),
            "cooper_feature_2": SimpleNamespace(owner_agent_id=None),
        },
        _oss_component_map={
            "cooper_feature_1": ["src/shared.py", "src/one.py"],
            "cooper_feature_2": ["src/shared.py", "src/two.py"],
        },
    )

    _bind_feature_owners(world, _request(tmp_path))

    assert world._cooperbench_issue_predecessor == {}
    assert world._cooperbench_sdl_state["overlapping_paths"] == ["src/shared.py"]
    assert world._cooperbench_sdl_state["serialization"] == (
        "parallel_actor_desks_with_integration_replay"
    )


def test_delivery_gate_requires_a_merged_pr_for_each_feature():
    world = SimpleNamespace(
        repo_system=SimpleNamespace(
            repo=SimpleNamespace(
                pull_requests={
                    "pr_1": SimpleNamespace(
                        status="merged",
                        linked_issue_ids=["cooper_feature_1"],
                        linked_issue=None,
                    ),
                    "pr_2": SimpleNamespace(
                        status="open",
                        linked_issue_ids=["cooper_feature_2"],
                        linked_issue=None,
                    ),
                }
            )
        )
    )

    assert _merged_feature_issue_ids(world) == {"cooper_feature_1"}
    with pytest.raises(
        CooperWorkerError,
        match="b3_two_agent_feature_delivery_incomplete:cooper_feature_2",
    ):
        _require_all_feature_deliveries_merged(world)

    world.repo_system.repo.pull_requests["pr_2"].status = "merged"
    assert _require_all_feature_deliveries_merged(world) == {
        "cooper_feature_1",
        "cooper_feature_2",
    }


def test_final_sdl_gate_rejects_missing_required_path_coverage():
    world = SimpleNamespace(
        _cooperbench_sdl_state={
            "phase": "frozen",
            "feature_owners": {
                "cooper_feature_1": "victor",
                "cooper_feature_2": "calvin",
            },
            "required_feature_paths": {
                "cooper_feature_1": ["src/one.py"],
                "cooper_feature_2": ["src/two.py"],
            },
        },
        product_artifacts={
            "one": SimpleNamespace(linked_file_path="src/one.py"),
            "two": SimpleNamespace(linked_file_path="src/two.py"),
        },
        patches={},
    )

    with pytest.raises(
        CooperWorkerError,
        match=(
            "b3_two_agent_required_path_coverage_incomplete:"
            "cooper_feature_1,cooper_feature_2"
        ),
    ):
        _require_two_person_sdl_complete(world)


def test_final_sdl_exports_technical_delivery_without_claiming_b3_realization(
    monkeypatch,
):
    import environments.org_env.cooperbench.worker as worker
    import environments.org_env.cooperbench.work_schedule as schedule

    state = {
        "phase": "frozen",
        "frozen_digest": "joint-digest",
        "decision_schedule": {"schema_version": "test_schedule"},
    }
    world = SimpleNamespace(
        _cooperbench_sdl_state=state,
        _cooperbench_main_b3_lifecycle=True,
    )
    monkeypatch.setattr(
        worker,
        "feature_delivery_coverage",
        lambda _world: {
            "cooper_feature_1": {"complete": True},
            "cooper_feature_2": {"complete": True},
        },
    )
    monkeypatch.setattr(
        worker,
        "peer_review_coverage",
        lambda _world: {"cooper_feature_1": True, "cooper_feature_2": True},
    )
    monkeypatch.setattr(worker, "joint_delivery_digest", lambda _world: "joint-digest")
    monkeypatch.setattr(
        worker,
        "protocol_realization",
        lambda _world: {
            "protocol_formation_satisfied": False,
            "realized_b3": False,
        },
    )
    monkeypatch.setattr(
        schedule, "compressed_schedule_receipt", lambda _world: state["decision_schedule"]
    )
    monkeypatch.setattr(
        schedule, "model_call_budget_receipt", lambda _world: {"valid": True}
    )

    receipt = _require_two_person_sdl_complete(world)

    assert receipt["technical_delivery_satisfied"] is True
    assert receipt["b3_protocol_formation_satisfied"] is False
    assert receipt["b3_realization_satisfied"] is False
    assert receipt["submission_gate_policy"] == (
        "technical_delivery_independent_of_b3_realization"
    )


def test_cooper_merge_evidence_does_not_complete_shared_file_successor():
    feature_one = SimpleNamespace(
        task_id="task_oss_cooper_feature_1",
        status="in_progress",
        progress_evidence=[],
        linked_artifacts=["shared"],
    )
    feature_two = SimpleNamespace(
        task_id="task_oss_cooper_feature_2",
        status="in_progress",
        progress_evidence=[],
        linked_artifacts=["shared"],
    )
    world = SimpleNamespace(
        tasks={
            feature_one.task_id: feature_one,
            feature_two.task_id: feature_two,
        },
        product_artifacts={
            "shared": SimpleNamespace(
                artifact_id="shared",
                linked_task_ids=[feature_one.task_id, feature_two.task_id],
            )
        },
        _cooperbench_delivery_focus=True,
        _task_requirements_met=lambda _task, _artifact: True,
        _emit_task_transition=lambda *_args, **_kwargs: None,
    )
    pr = SimpleNamespace(
        pr_id="pr_1",
        linked_task_ids=[feature_one.task_id],
        linked_task=None,
        approved_by=["calvin"],
    )

    OrgWorld.apply_merge_task_evidence(
        world, pr, ["shared"], "victor", tick=12
    )

    assert any(
        item["evidence_type"] == "merged_to_mainline"
        for item in feature_one.progress_evidence
    )
    assert feature_two.status == "in_progress"
    assert feature_two.progress_evidence == []


def test_worker_environment_clears_ladder_and_transfer_overrides(tmp_path: Path):
    environ = {
        "ORG_EXPERIMENT_CONDITION": "b0_single_agent_founder",
        "ORG_TRANSFER_ARM": "F_Exec",
        "ORG_TRANSFER_SOURCE_BUNDLE": "should-not-survive",
        "ORG_EXECUTION_PROFILE": "programbench_leaderboard_v1",
    }

    _configure_worker_environment(
        _request(tmp_path),
        tmp_path / "pack",
        "project",
        environ=environ,
        evaluator_image="sha256:" + "a" * 64,
        evaluator_platform="linux/amd64",
    )

    assert "ORG_EXPERIMENT_CONDITION" not in environ
    assert not any(key.startswith("ORG_TRANSFER_") for key in environ)
    assert environ["ORG_EXECUTION_PROFILE"] == "native"
    assert environ["ORG_EXPERIMENT_ARM_ID"] == TREATMENT_ID
    assert Path(environ["ORG_PRODUCT_SMOKE_ROOT"]) == (
        tmp_path / "product_smoke"
    ).resolve()
    assert environ["ORG_OSS_MODE"] == "pilot"
    assert environ["ORG_EVALUATOR_BACKEND"] == "docker"
    assert environ["ORG_EVALUATOR_CONTAINER_IMAGE"] == "sha256:" + "a" * 64
    assert environ["ORG_EVALUATOR_CONTAINER_PLATFORM"] == "linux/amd64"
    assert environ["ORG_EVALUATOR_CLEAR_ENTRYPOINT"] == "1"
    assert environ["ORG_LLM_FORCE_PROVIDER_PROCESS_DEADLINE"] == "1"
    assert environ["ORG_LLM_REQUEST_TIMEOUT_SECONDS"] == "3600"
    assert environ["ORG_LLM_PROVIDER_ATTEMPT_TIMEOUT_SECONDS"] == "900"
    assert environ["ORG_LLM_MAX_RETRIES"] == "7"
    assert environ["ORG_LLM_RETRY_BACKOFF_SECONDS"] == "2"
    assert environ["ORG_MECHANISM_ABLATIONS"] == "work_rhythm"


def test_worker_smoke_workspaces_are_pair_local(tmp_path: Path):
    first = {}
    second = {}

    _configure_worker_environment(
        _request(tmp_path),
        tmp_path / "pair-one" / "public_pack",
        "project-one",
        environ=first,
    )
    _configure_worker_environment(
        _request(tmp_path),
        tmp_path / "pair-two" / "public_pack",
        "project-two",
        environ=second,
    )

    assert first["ORG_PRODUCT_SMOKE_ROOT"] != second["ORG_PRODUCT_SMOKE_ROOT"]
    assert Path(first["ORG_PRODUCT_SMOKE_ROOT"]) == (
        tmp_path / "pair-one" / "product_smoke"
    ).resolve()
    assert Path(second["ORG_PRODUCT_SMOKE_ROOT"]) == (
        tmp_path / "pair-two" / "product_smoke"
    ).resolve()


def test_evaluator_runtime_binding_requires_exact_task_image(monkeypatch, tmp_path):
    import environments.org_env.product.materialize as materialize

    expected_image = "sha256:" + "a" * 64
    policy = SimpleNamespace(
        backend="docker",
        container_image=expected_image,
        container_platform="linux/amd64",
        clear_container_entrypoint=True,
        network_enabled=False,
    )
    monkeypatch.setattr(
        materialize,
        "_formal_product_executor",
        lambda: SimpleNamespace(policy=policy),
    )

    receipt = _require_evaluator_runtime_binding(
        _request(tmp_path),
        evaluator_image=expected_image,
        evaluator_platform="linux/amd64",
    )

    assert receipt["container_image"] == expected_image
    assert receipt["clear_container_entrypoint"] is True
    policy.container_image = "sha256:" + "b" * 64
    with pytest.raises(
        CooperWorkerError,
        match="cooperbench_evaluator_runtime_mismatch:container_image",
    ):
        _require_evaluator_runtime_binding(
            _request(tmp_path),
            evaluator_image=expected_image,
            evaluator_platform="linux/amd64",
        )


def test_public_runtime_preflight_runs_the_untouched_declared_suite(monkeypatch):
    import environments.org_env.product.materialize as materialize

    seen = {}

    def release_smoke(world, *, prefer_mainline, timeout, command):
        seen.update(
            {
                "world": world,
                "prefer_mainline": prefer_mainline,
                "timeout": timeout,
                "command": command,
            }
        )
        return {
            "ok": True,
            "returncode": 0,
            "stdout_tail": "84 passed",
            "stderr_tail": "",
            "error": None,
        }

    monkeypatch.setattr(materialize, "release_smoke", release_smoke)
    world = object()
    receipt = _public_runtime_preflight(
        world, ("python", "-m", "pytest", "-q", "tests/test_termui.py")
    )

    assert receipt["passed"] is True
    assert receipt["summary"] == "84 passed"
    assert seen == {
        "world": world,
        "prefer_mainline": True,
        "timeout": 120,
        "command": [
            "python",
            "-m",
            "pytest",
            "-q",
            "tests/test_termui.py",
        ],
    }
    _require_public_runtime_preflight(receipt)


def test_public_runtime_preflight_fails_before_ticks_on_entrypoint_collision(
    monkeypatch,
):
    import environments.org_env.product.materialize as materialize

    monkeypatch.setattr(
        materialize,
        "release_smoke",
        lambda *_args, **_kwargs: {
            "ok": False,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "error": (
                "workspace_probe_exit_1:/usr/local/bin/runner.sh: "
                "cd: /workspace/repo: No such file or directory"
            ),
        },
    )
    receipt = _public_runtime_preflight(
        object(), ("python", "-m", "pytest", "-q", "tests/test_termui.py")
    )

    assert receipt["passed"] is False
    with pytest.raises(
        CooperWorkerError,
        match="cooperbench_public_runtime_preflight_infrastructure_failed",
    ):
        _require_public_runtime_preflight(receipt)


def test_failed_fixed_suite_recovers_only_from_untouched_green_public_modules(
    monkeypatch,
):
    import environments.org_env.product.materialize as materialize

    calls = []

    def release_smoke(_world, *, prefer_mainline, timeout, command):
        calls.append(list(command))
        if command[:2] == ["python", "-c"]:
            payload = {
                "rows": [
                    {"path": "tests/test_optional.py", "returncode": 2, "passed_tests": 0, "timed_out": False},
                    {"path": "tests/test_core.py", "returncode": 0, "passed_tests": 7, "timed_out": False},
                ],
                "chosen": ["tests/test_core.py"],
            }
            return {"ok": True, "returncode": 0,
                    "stdout_tail": _PUBLIC_MODULE_PROBE_MARKER + json.dumps(payload),
                    "stderr_tail": "", "error": None}
        passed = command == [
            "python", "-m", "pytest", "-q", "tests/test_core.py"
        ]
        return {"ok": passed, "returncode": 0 if passed else 2,
                "stdout_tail": "7 passed" if passed else "1 error",
                "stderr_tail": "", "error": None}

    monkeypatch.setattr(materialize, "release_smoke", release_smoke)
    receipt = _verified_public_runtime_preflight(
        object(),
        ("python", "-m", "pytest", "-q", "tests/test_optional.py", "tests/test_core.py"),
    )

    assert receipt["passed"] is True
    assert receipt["command"] == ["python", "-m", "pytest", "-q", "tests/test_core.py"]
    assert receipt["selection_policy"] == "feature_independent_untouched_green_module_subset"
    assert receipt["module_probe_used"] is True
    assert len(calls) == 3


def test_unavailable_public_suite_degrades_to_explicit_source_integrity(
    monkeypatch,
):
    import environments.org_env.product.materialize as materialize

    calls = []

    def release_smoke(_world, *, prefer_mainline, timeout, command):
        calls.append(list(command))
        if command[:2] == ["python", "-c"]:
            return {
                "ok": True,
                "returncode": 0,
                "stdout_tail": "COOPERBENCH_SOURCE_SNAPSHOT_INTEGRITY files=9 bytes=1200",
                "stderr_tail": "",
                "error": None,
            }
        return {
            "ok": False,
            "returncode": 1,
            "stdout_tail": "baseline dependency unavailable",
            "stderr_tail": "",
            "error": None,
        }

    monkeypatch.setattr(materialize, "release_smoke", release_smoke)
    receipt = _verified_public_runtime_preflight(
        object(), ("go", "test", "./...")
    )

    assert receipt["passed"] is True
    assert receipt["functional_public_regression_available"] is False
    assert receipt["public_validation_strength"] == "source_snapshot_integrity"
    assert receipt["candidate_failure"]["command"] == ["go", "test", "./..."]
    assert len(calls) == 2
    _require_public_runtime_preflight(receipt)


def test_missing_preferred_package_runner_uses_same_script_through_npm(
    monkeypatch,
):
    import environments.org_env.product.materialize as materialize

    calls = []

    def release_smoke(_world, *, prefer_mainline, timeout, command):
        calls.append(list(command))
        if command == ["pnpm", "run", "typecheck"]:
            return {
                "ok": False,
                "returncode": 127,
                "stdout_tail": "",
                "stderr_tail": "pnpm: not found",
                "error": None,
            }
        return {
            "ok": command == ["npm", "run", "typecheck"],
            "returncode": 0,
            "stdout_tail": "typecheck passed",
            "stderr_tail": "",
            "error": None,
        }

    monkeypatch.setattr(materialize, "release_smoke", release_smoke)
    receipt = _verified_public_runtime_preflight(
        object(), ("pnpm", "run", "typecheck")
    )

    assert receipt["passed"] is True
    assert receipt["command"] == ["npm", "run", "typecheck"]
    assert receipt["candidate_command"] == ["pnpm", "run", "typecheck"]
    assert receipt["selection_policy"] == "verified_equivalent_package_script_runner"
    assert receipt["functional_public_regression_available"] is False
    assert receipt["public_validation_strength"] == "typecheck_only"
    assert calls == [
        ["pnpm", "run", "typecheck"],
        ["npm", "run", "typecheck"],
    ]


def test_green_go_public_suite_remains_a_functional_gate(monkeypatch):
    import environments.org_env.product.materialize as materialize

    calls = []

    def release_smoke(_world, *, prefer_mainline, timeout, command):
        calls.append(list(command))
        return {
            "ok": command == ["go", "test", "-p=1", "./..."],
            "returncode": 0,
            "stdout_tail": "ok  example.com/sample  0.01s",
            "stderr_tail": "",
            "error": None,
        }

    monkeypatch.setattr(materialize, "release_smoke", release_smoke)
    receipt = _verified_public_runtime_preflight(
        object(), ("go", "test", "-p=1", "./...")
    )

    assert receipt["passed"] is True
    assert receipt["command"] == ["go", "test", "-p=1", "./..."]
    assert receipt["functional_public_regression_available"] is True
    assert receipt["public_validation_strength"] == "public_regression"
    assert calls == [["go", "test", "-p=1", "./..."]]


def test_green_cargo_no_run_is_compile_evidence_not_functional_test(monkeypatch):
    import environments.org_env.product.materialize as materialize

    monkeypatch.setattr(
        materialize,
        "release_smoke",
        lambda *_args, **kwargs: {
            "ok": kwargs["command"]
            == ["cargo", "test", "--workspace", "--no-run", "-j", "1"],
            "returncode": 0,
            "stdout_tail": "Finished test profile",
            "stderr_tail": "",
            "error": None,
        },
    )
    receipt = _verified_public_runtime_preflight(
        object(), ("cargo", "test", "--workspace", "--no-run", "-j", "1")
    )

    assert receipt["passed"] is True
    assert receipt["functional_public_regression_available"] is False
    assert receipt["public_validation_strength"] == "compile_only"


def test_no_declared_public_suite_still_proves_exact_source_runtime(monkeypatch):
    import environments.org_env.product.materialize as materialize

    monkeypatch.setattr(
        materialize,
        "release_smoke",
        lambda *_args, **kwargs: {
            "ok": kwargs["command"][:2] == ["python", "-c"],
            "returncode": 0,
            "stdout_tail": "COOPERBENCH_SOURCE_SNAPSHOT_INTEGRITY files=3 bytes=400",
            "stderr_tail": "",
            "error": None,
        },
    )
    receipt = _verified_public_runtime_preflight(object(), ())
    assert receipt["passed"] is True
    assert receipt["candidate_command"] == []
    assert receipt["degraded_reason"] == "public_regression_surface_unavailable"


@pytest.mark.parametrize(
    ("paths", "available"),
    [
        ({"cooper_feature_1": ["pkg/a.py"], "cooper_feature_2": ["pkg/b.py"]}, True),
        ({"cooper_feature_1": ["a.go"], "cooper_feature_2": ["b.go"]}, False),
        ({"cooper_feature_1": ["a.ts"], "cooper_feature_2": ["b.ts"]}, False),
    ],
)
def test_peer_probe_capability_matches_selected_source_language(paths, available):
    receipt = _behavior_probe_capability(SimpleNamespace(component_map=paths))
    assert receipt["available"] is available
    assert (receipt["mode"] == "python_executable_public_behavior") is available


def test_green_public_modules_from_different_monorepo_packages_are_not_combined(
    monkeypatch,
):
    import environments.org_env.product.materialize as materialize

    def release_smoke(_world, *, prefer_mainline, timeout, command):
        if command[:2] == ["python", "-c"]:
            payload = {
                "rows": [
                    {"path": "alpha/tests/test_a.py", "returncode": 0, "passed_tests": 2, "timed_out": False},
                    {"path": "beta/tests/test_b.py", "returncode": 0, "passed_tests": 3, "timed_out": False},
                ],
                "chosen": ["alpha/tests/test_a.py", "beta/tests/test_b.py"],
            }
            return {"ok": True, "returncode": 0,
                    "stdout_tail": _PUBLIC_MODULE_PROBE_MARKER + json.dumps(payload),
                    "stderr_tail": "", "error": None}
        passed = command == ["python", "-m", "pytest", "-q", "alpha/tests/test_a.py"]
        return {"ok": passed, "returncode": 0 if passed else 4,
                "stdout_tail": "2 passed" if passed else "ImportPathMismatchError",
                "stderr_tail": "", "error": None}

    monkeypatch.setattr(materialize, "release_smoke", release_smoke)
    receipt = _verified_public_runtime_preflight(
        object(),
        ("python", "-m", "pytest", "-q", "alpha/tests/test_a.py", "beta/tests/test_b.py"),
    )
    assert receipt["passed"] is True
    assert receipt["command"][-1] == "alpha/tests/test_a.py"
    assert receipt["module_probe"]["green_paths_before_package_grouping"] == [
        "alpha/tests/test_a.py", "beta/tests/test_b.py"
    ]


def test_public_module_recovery_probes_past_four_green_modules_before_grouping(
    monkeypatch,
):
    import environments.org_env.product.materialize as materialize

    alpha = [f"alpha{i}/tests/test_a.py" for i in range(4)]
    beta = ["beta/tests/test_b.py", "beta/tests/test_c.py"]

    def release_smoke(_world, *, prefer_mainline, timeout, command):
        if command[:2] == ["python", "-c"]:
            payload = {
                "rows": [
                    {"path": path, "returncode": 0, "passed_tests": 1, "timed_out": False}
                    for path in [*alpha, *beta]
                ],
                "chosen": [*alpha, *beta],
            }
            return {
                "ok": True,
                "returncode": 0,
                "stdout_tail": _PUBLIC_MODULE_PROBE_MARKER + json.dumps(payload),
                "stderr_tail": "",
                "error": None,
            }
        passed = command == ["python", "-m", "pytest", "-q", *beta]
        return {
            "ok": passed,
            "returncode": 0 if passed else 4,
            "stdout_tail": "2 passed" if passed else "ImportPathMismatchError",
            "stderr_tail": "",
            "error": None,
        }

    monkeypatch.setattr(materialize, "release_smoke", release_smoke)
    receipt = _verified_public_runtime_preflight(
        object(), ("python", "-m", "pytest", "-q", *alpha, *beta)
    )

    assert receipt["passed"] is True
    assert receipt["command"] == ["python", "-m", "pytest", "-q", *beta]
    assert receipt["module_probe"]["green_paths_before_package_grouping"] == [
        *alpha, *beta
    ]


def test_treatment_health_requires_real_successful_token_usage():
    passed = _treatment_health(
        {"calls": 20, "failures": 1, "total_tokens": 4000}, tick=24
    )
    assert passed["passed"] is True

    with pytest.raises(CooperWorkerError, match="llm_treatment_health_gate_failed"):
        _treatment_health(
            {"calls": 20, "failures": 2, "total_tokens": 4000}, tick=24
        )
    with pytest.raises(CooperWorkerError, match="llm_treatment_health_gate_failed"):
        _treatment_health(
            {"calls": 20, "failures": 0, "total_tokens": 0}, tick=24
        )


def test_treatment_health_snapshot_keeps_failed_run_diagnostics_available():
    failed = _treatment_health_snapshot(
        {"calls": 52, "failures": 11, "total_tokens": 200_211},
        tick=168,
    )

    assert failed == {
        "tick": 168,
        "calls": 52,
        "failures": 11,
        "success_rate": 0.788461538462,
        "total_tokens": 200_211,
        "passed": False,
        "threshold": {
            "max_failure_rate": 0.05,
            "requires_nonzero_calls": True,
            "requires_nonzero_total_tokens": True,
        },
    }


def test_observed_treatment_health_records_failure_without_rejecting_run():
    from environments.org_env.cooperbench.worker import (
        _observed_treatment_health,
    )

    observed = _observed_treatment_health(
        {"calls": 7, "failures": 4, "total_tokens": 9_920}, tick=24
    )

    assert observed["passed"] is False
    assert observed["enforced"] is False
    assert observed["policy"] == "observe"


def test_solver_usage_delta_excludes_worker_transport_preflight():
    assert _usage_delta(
        {
            "calls": 1,
            "failures": 0,
            "retries": 0,
            "provider_attempts": 1,
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
            "cached_prompt_tokens": 0,
        },
        {
            "calls": 8,
            "failures": 4,
            "retries": 2,
            "provider_attempts": 10,
            "prompt_tokens": 220,
            "completion_tokens": 55,
            "total_tokens": 275,
            "cached_prompt_tokens": 40,
        },
    ) == {
        "calls": 7,
        "failures": 4,
        "retries": 2,
        "provider_attempts": 9,
        "prompt_tokens": 200,
        "completion_tokens": 50,
        "total_tokens": 250,
        "cached_prompt_tokens": 40,
    }


def test_minimum_solver_liveness_allows_degraded_but_progressing_run():
    receipt = _minimum_solver_liveness_snapshot(
        {"calls": 7, "failures": 4, "total_tokens": 9_920}, tick=24
    )

    assert receipt == {
        "tick": 24,
        "calls": 7,
        "failures": 4,
        "usable_calls": 3,
        "total_tokens": 9_920,
        "passed": True,
        "enforced": True,
        "policy": "require_usable_metered_solver_response_by_t24",
        "usage_scope": "solver_queries_after_worker_transport_preflight",
    }


@pytest.mark.parametrize(
    "usage",
    (
        {"calls": 0, "failures": 0, "total_tokens": 0},
        {"calls": 6, "failures": 6, "total_tokens": 0},
        {"calls": 6, "failures": 6, "total_tokens": 1_000},
        {"calls": 6, "failures": 5, "total_tokens": 0},
    ),
)
def test_minimum_solver_liveness_rejects_template_only_run(usage):
    receipt = _minimum_solver_liveness_snapshot(usage, tick=24)

    assert receipt["passed"] is False
    assert receipt["enforced"] is True


def test_llm_preflight_requires_a_metered_successful_provider_call():
    class Client:
        calls = 0
        failures = 0
        retries = 0
        provider_attempts = 0
        usage_totals = {"total_tokens": 0}

        def generate_json(self, *_args, **_kwargs):
            self.calls += 1
            self.provider_attempts += 1
            self.usage_totals = {"total_tokens": 7}
            return {"probe": "transport_ready", "version": 1}

    assert _require_live_llm_preflight(Client()) == {
        "passed": True,
        "calls": 1,
        "failures": 0,
        "total_tokens": 7,
        "provider_attempts": 1,
        "transport": {
            "attempt_semantics": "admitted_send_stage_not_gateway_receipt",
            "provider_attempts": 1,
            "provider_successes": 0,
            "provider_failure_counts": {},
            "provider_exception_counts": {},
            "provider_status_counts": {},
            "empty_json_responses": 0,
            "empty_json_retries": 0,
            "unclassified_attempts": 1,
            "last_provider_failure": None,
        },
    }


def test_llm_preflight_fails_closed_on_provider_or_metering_failure():
    class ProviderFailure:
        calls = 0
        failures = 0
        retries = 0
        provider_attempts = 0
        usage_totals = {"total_tokens": 0}

        def generate_json(self, *_args, **_kwargs):
            self.calls += 1
            self.failures += 1
            raise RuntimeError("rate limited")

    with pytest.raises(CooperWorkerError, match="llm_preflight_failed"):
        _require_live_llm_preflight(ProviderFailure())

    class UnmeteredSuccess(ProviderFailure):
        def generate_json(self, *_args, **_kwargs):
            self.calls += 1
            return {"probe": "transport_ready", "version": 1}

    with pytest.raises(CooperWorkerError, match="unmetered_or_failed_response"):
        _require_live_llm_preflight(UnmeteredSuccess())

    class ChallengeMismatch(UnmeteredSuccess):
        def generate_json(self, *_args, **_kwargs):
            self.calls += 1
            self.provider_attempts += 1
            self.usage_totals = {"total_tokens": 7}
            return {"probe": "wrong"}

    with pytest.raises(CooperWorkerError, match="challenge_mismatch"):
        _require_live_llm_preflight(ChallengeMismatch())


def test_observed_llm_preflight_keeps_provider_failure_nonterminal():
    class Client:
        calls = 0
        failures = 0
        retries = 0
        provider_attempts = 0
        usage_totals = {"total_tokens": 0}

        def generate_json(self, *_args, **_kwargs):
            self.calls += 1
            self.failures += 1
            self.provider_attempts += 1
            raise RuntimeError("provider unavailable")

    receipt = _observed_live_llm_preflight(Client())

    assert receipt["passed"] is False
    assert receipt["enforced"] is False
    assert receipt["policy"] == "observe"
    assert receipt["calls"] == 1
    assert receipt["failures"] == 1
    assert receipt["provider_attempts"] == 1
    assert receipt["total_tokens"] == 0
    assert receipt["transport"]["unclassified_attempts"] == 1
    assert "llm_preflight_failed" in receipt["error"]


def test_transport_payload_keeps_only_bounded_failure_receipt_fields():
    class Client:
        provider_attempts = 4

        @staticmethod
        def stats():
            return {
                "provider_attempts": 4,
                "provider_successes": 1,
                "provider_failure_counts": {"http_error": 3},
                "provider_exception_counts": {"InternalServerError": 3},
                "provider_status_counts": {"524": 3},
                "last_provider_failure": {
                    "attempt_number": 3,
                    "category": "http_error",
                    "exception_type": "InternalServerError",
                    "status_code": 524,
                    "provider_message": "Authorization: Bearer must-not-survive",
                },
            }

    payload = _transport_payload(Client())

    assert payload == {
        "attempt_semantics": "admitted_send_stage_not_gateway_receipt",
        "provider_attempts": 4,
        "provider_successes": 1,
        "provider_failure_counts": {"http_error": 3},
        "provider_exception_counts": {"InternalServerError": 3},
        "provider_status_counts": {"524": 3},
        "empty_json_responses": 0,
        "empty_json_retries": 0,
        "unclassified_attempts": 0,
        "last_provider_failure": {
            "attempt_number": 3,
            "category": "http_error",
            "exception_type": "InternalServerError",
            "status_code": 524,
        },
    }


def _git(repo: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_joint_patch_exports_only_merged_mainline_and_applies_to_image_head(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    merged = ProductArtifact(
        artifact_id="art_module_py",
        artifact_type="repo_file",
        title="module.py",
        status="active",
        linked_file_path="module.py",
        content="VALUE = 3\n",
        mainline_content="VALUE = 2\n",
        revision=2,
        mainline_revision=1,
    )
    world = SimpleNamespace(product_artifacts={merged.artifact_id: merged})

    patch, paths = _export_joint_mainline_patch(
        world, repo, tmp_path / "overlay"
    )

    assert paths == ["module.py"]
    assert "+VALUE = 2" in patch
    assert "+VALUE = 3" not in patch
    assert (repo / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    persisted, persisted_paths = _persist_candidate_joint_delivery(
        tmp_path / "result",
        world,
        repo,
        tmp_path / "overlay",
        {"cooper_feature_1", "cooper_feature_2"},
    )

    assert persisted == patch
    assert persisted_paths == paths
    assert (tmp_path / "result" / "candidate_joint.patch").read_text(
        encoding="utf-8"
    ) == patch
    receipt = json.loads(
        (tmp_path / "result" / "candidate_delivery_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "unsubmitted_candidate"
    assert receipt["submission_gate"] == "two_person_sdl_freeze_pending"
    assert receipt["patch_apply_check"] == "passed_against_image_head"
    assert receipt["changed_paths"] == ["module.py"]
