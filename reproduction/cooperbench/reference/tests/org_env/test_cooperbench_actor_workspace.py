"""Private source transactions and actual native two-person B3 action paths."""
import copy
from types import MappingProxyType, SimpleNamespace

import pytest

from environments.org_env.backend.repo.repo import PRStatus, PullRequest
from environments.org_env.cooperbench.lifecycle import feature_delivery_coverage
from environments.org_env.cooperbench import source_views as sv
from environments.org_env.cooperbench.actor_workspace import (
    actor_patch_guard_failure, actor_public_test_record,
)
from environments.org_env.product.patch_objects import CodePatch
from environments.org_env.runtime_adapter.execution import (
    OrgActionMapper,
    _cooperbench_action_block_reason,
    _cooperbench_coordination_conflict,
    _cooperbench_current_public_test_result,
    _cooperbench_feature_edit_candidates,
    _cooperbench_missing_feature_paths,
    _resolve_cooperbench_coordination_conflict,
)
from test_cooperbench_visibility_profile_loop import (
    _native_pool,
    _profile_select,
    _world,
    F1,
    F2,
)
from test_cooperbench_source_ci import _public_runtime
from test_cooperbench_source_views import _merge_ledger, _publish_private


def _private_world():
    world, _ = _world()
    sv.freeze_baseline(world)
    sv.initialize_actor_desks(world)
    return world


def _result():
    return SimpleNamespace(success=True, failure_reason=None, modified_objects=[], created_objects=[],
                           state_delta={}, events=[], graph_edges=[], messages=[])


def _patch(world, actor, artifact_id, text, *, creates=False):
    patch = CodePatch(f"private_{len(world.patches)}_{world.world_tick}", artifact_id, actor,
                      world.world_tick, new_content=text, change_summary="Implement a public behavior",
                      creates_file=creates, related_issue_ids=[F1 if actor == "victor" else F2],
                      related_task_ids=["task_oss_" + (F1 if actor == "victor" else F2)])
    patch.actor_source_snapshot = sv.actor_desk_snapshot(world, actor).receipt()
    return patch


def _apply(world, actor, artifact_id, text):
    patch = _patch(world, actor, artifact_id, text)
    result = _result()
    assert world.apply_product_patch(patch, result, actor), result.failure_reason
    return patch, result


@pytest.mark.parametrize("guard_brief", ["", "merged behavior regressed; inspect beta_0.py"])
def test_conflict_guard_keeps_already_covered_required_paths_reachable(monkeypatch, guard_brief):
    from environments.org_env.runtime_adapter import execution as ex
    from environments.org_env.cooperbench import actor_workspace as aw

    world = _private_world()
    world._cooperbench_sdl_state["required_feature_paths"][F2] = ["alpha.py", "beta_0.py"]
    monkeypatch.setattr(ex, "_cooperbench_coordination_conflict", lambda *a: {
        "status": "repairing", "feature_id": F2, "conflict_paths": ["beta_0.py"]})
    monkeypatch.setattr(ex, "_cooperbench_conflict_repair_missing_paths", lambda *a: ["beta_0.py"])
    monkeypatch.setattr(aw, "actor_patch_guard_failure", lambda *a: {"brief": guard_brief})
    candidates = ex._cooperbench_feature_edit_candidates(world, "calvin")
    assert {c.parameters["file_path"] for c in candidates} == (
        {"alpha.py", "beta_0.py"} if guard_brief else {"beta_0.py"})
    if guard_brief:
        assert all(guard_brief in c.parameters["edit_goal"] for c in candidates)


@pytest.mark.parametrize("old_ledger", ["committed", "pending"])
def test_unbound_private_desk_never_reuses_an_old_native_ledger(old_ledger):
    from environments.org_env.backend.repo import workflow as wf

    world = _private_world()
    old = world.repo_system.create_branch("victor", tick=0)
    old.linked_task = "task_oss_" + F1
    if old_ledger == "committed":
        old.commit_ids.append("old_commit")
        world.repo_system.repo.commits["old_commit"] = SimpleNamespace(artifact_ids=["art_alpha_py"])
    else:
        world.__dict__.setdefault(wf.PENDING, {})[old.branch_id] = [("old_patch", "art_alpha_py")]
    before = copy.deepcopy(old)
    _apply(world, "victor", "art_alpha_py", "VALUE = 1\n")
    new_id = sv.actor_desk_snapshot(world, "victor").branch_id
    assert new_id and new_id != old.branch_id
    assert old == before
    assert sv.actor_desk_snapshot(world, "victor").branch_id == new_id
    world.world_tick += 1
    _apply(world, "victor", "art_alpha_py", "VALUE = 2\n")
    assert sv.actor_desk_snapshot(world, "victor").branch_id == new_id


def test_unbound_private_desk_does_not_bypass_native_branch_capacity():
    from environments.org_env.backend.repo import workflow as wf

    world = _private_world()
    for _ in range(wf.MAX_ACTIVE_BRANCHES):
        branch = world.repo_system.create_branch("victor", tick=0)
        branch.linked_task = "task_oss_" + F1
        branch.commit_ids.append("old_commit")
    before = copy.deepcopy(world.repo_system.repo)
    patch = _patch(world, "victor", "art_alpha_py", "VALUE = 1\n")
    assert wf.record_patch(world, "victor", patch, world.product_artifacts["art_alpha_py"], 1) is None
    assert world.repo_system.repo == before
    assert sv.actor_desk_snapshot(world, "victor").branch_id is None


def test_native_apply_keeps_peer_desk_and_global_source_unchanged():
    world = _private_world()
    peer_before = sv.actor_desk_snapshot(world, "calvin")
    original = world.product_artifacts["art_alpha_py"].content
    patch, result = _apply(world, "victor", "art_alpha_py", "VICTOR_PRIVATE = 7\n")
    assert sv.actor_file_text(world, "victor", "art_alpha_py") == patch.new_content
    assert sv.actor_desk_snapshot(world, "calvin") == peer_before
    assert world.product_artifacts["art_alpha_py"].content == original
    assert world.product_artifacts["art_alpha_py"].mainline_content == original
    assert result.state_delta["actor_source_snapshot"]["patch_ids"] == [patch.patch_id]


def test_native_apply_rollback_restores_patch_repo_and_private_store(monkeypatch):
    world = _private_world()
    patch = _patch(world, "victor", "art_alpha_py", "VALUE = 9\n")
    repo_before = copy.deepcopy(world.repo_system.repo)
    stores_before = copy.deepcopy((world._cooperbench_source_views, world._cooperbench_actor_desks))
    artifact_before = copy.deepcopy(world.product_artifacts["art_alpha_py"].__dict__)
    real_commit = sv.commit_actor_patch

    def after_commit(*args, **kwargs):
        real_commit(*args, **kwargs)
        raise RuntimeError("synthetic post-publication failure")

    monkeypatch.setattr(sv, "commit_actor_patch", after_commit)
    result = _result()
    assert not world.apply_product_patch(patch, result, "victor")
    assert (world._cooperbench_source_views, world._cooperbench_actor_desks) == stores_before
    assert world.repo_system.repo == repo_before
    assert world.product_artifacts["art_alpha_py"].__dict__ == artifact_before
    assert world.patches == {} and result.modified_objects == []
    assert result.failure_reason == "actor_patch_transaction_failed"


def test_non_python_source_review_repair_does_not_require_a_python_probe_plan():
    world = _private_world()
    world._cooperbench_behavior_probe_capability = {
        "available": False,
        "mode": "three_way_source_review_only",
    }
    world._cooperbench_semantic_reviews = {
        F1: {
            "approved": False,
            "reviewer_id": "calvin",
            "source_snapshot": sv.actor_desk_snapshot(world, "victor").receipt(),
        }
    }

    patch, result = _apply(
        world, "victor", "art_alpha_py", "REPAIRED_SOURCE = 1\n"
    )

    assert result.success is True
    assert patch.validation_status == "accepted"
    assert not actor_patch_guard_failure(world, "victor")


def test_actor_edit_does_not_copy_unrelated_read_only_ci_projection():
    world = _private_world()
    from environments.org_env.backend.repo.repo import CIResult

    world.repo_system.repo.ci_runs["ci_old"] = CIResult(
        ci_id="ci_old",
        pr_id="pr_old",
        checks=[{
            "name": "historical_source_projection",
            "projection": MappingProxyType({"alpha.py": "VALUE = 0\n"}),
        }],
    )

    patch, result = _apply(
        world, "victor", "art_alpha_py", "FRESH_AFTER_OLD_CI = 1\n"
    )

    assert result.success is True
    assert patch.validation_status == "accepted"
    assert world.repo_system.repo.ci_runs["ci_old"].checks[0][
        "projection"
    ]["alpha.py"] == "VALUE = 0\n"


def test_multi_file_review_repair_stages_privately_until_guard_is_green(
    monkeypatch,
):
    world = _private_world()
    world._cooperbench_sdl_state["feature_paths"][F2] = [
        "alpha.py", "beta_0.py",
    ]
    world._cooperbench_sdl_state["required_feature_paths"][F2] = [
        "alpha.py", "beta_0.py",
    ]
    observed_before = []

    def guard(_world, *, actor_id, prepared_patch):
        assert actor_id == "calvin"
        observed_before.append(
            sv.actor_desk_snapshot(_world, actor_id).files["alpha.py"]
        )
        source = sv.actor_patch_candidate_snapshot(
            _world, prepared_patch
        ).receipt()
        pending = prepared_patch.file_path == "alpha.py"
        return {
            "applicable": True,
            "available": True,
            "ok": not pending,
            "error": "current_feature_repair_unresolved" if pending else "",
            "repair_feature": F2,
            "repair_paths": ["alpha.py", "beta_0.py"],
            "repair_probe_ids": [F2 + ":0"],
            "preserve_probe_ids": [F2 + ":1"],
            "protected_features": [],
            "failed_protected_probe_ids": [],
            "failed_repair_probe_ids": [F2 + ":0"] if pending else [],
            "failed_preservation_probe_ids": [F2 + ":1"] if pending else [],
            "source_snapshot": source,
            "behavior_evidence": {},
        }

    monkeypatch.setattr(
        "environments.org_env.cooperbench.joint_probe_replay."
        "guard_actor_patch_against_merged_features",
        guard,
    )
    first, first_result = _apply(
        world, "calvin", "art_alpha_py", "FIRST_HALF = 1\n"
    )
    assert first.validation_status == "accepted"
    assert first_result.state_delta["current_feature_repair_pending"][
        "feature_id"
    ] == F2
    assert next(
        event for event in first_result.events
        if event.get("subtype") == "current_feature_repair_guard_pending"
    )
    assert sv.actor_file_text(world, "calvin", "art_alpha_py") == "FIRST_HALF = 1\n"

    blocked = _result()
    world._loop["execution"]._h_commit_patch(
        world,
        "calvin",
        {"branch_id": first_result.state_delta["actor_source_snapshot"]["branch_id"]},
        blocked,
        world.world_tick,
    )
    assert blocked.success is False
    assert blocked.failure_reason == "current_feature_repair_pending"

    second, _ = _apply(
        world, "calvin", "art_beta_0_py", "SECOND_HALF = 1\n"
    )
    assert second.validation_status == "accepted"
    assert observed_before == ["VALUE = 0\n", "FIRST_HALF = 1\n"]
    assert not actor_patch_guard_failure(world, "calvin")


def test_single_file_review_repair_stages_privately_until_guard_is_green(
    monkeypatch,
):
    world = _private_world()
    observed_before = []

    def guard(_world, *, actor_id, prepared_patch):
        assert actor_id == "calvin"
        observed_before.append(
            sv.actor_desk_snapshot(_world, actor_id).files["alpha.py"]
        )
        source = sv.actor_patch_candidate_snapshot(
            _world, prepared_patch
        ).receipt()
        pending = prepared_patch.new_content == "FIRST_STEP = 1\n"
        return {
            "applicable": True,
            "available": True,
            "ok": not pending,
            "error": "current_feature_repair_unresolved" if pending else "",
            "repair_feature": F2,
            "repair_paths": ["alpha.py"],
            "repair_probe_ids": [F2 + ":0"],
            "preserve_probe_ids": [],
            "protected_features": [],
            "failed_protected_probe_ids": [],
            "failed_repair_probe_ids": [F2 + ":0"] if pending else [],
            "failed_preservation_probe_ids": [],
            "source_snapshot": source,
            "behavior_evidence": {},
        }

    monkeypatch.setattr(
        "environments.org_env.cooperbench.joint_probe_replay."
        "guard_actor_patch_against_merged_features",
        guard,
    )
    first, first_result = _apply(
        world, "calvin", "art_alpha_py", "FIRST_STEP = 1\n"
    )
    assert first.validation_status == "accepted"
    assert first_result.state_delta["current_feature_repair_pending"][
        "feature_id"
    ] == F2
    assert sv.actor_file_text(world, "calvin", "art_alpha_py") == "FIRST_STEP = 1\n"

    blocked = _result()
    world._loop["execution"]._h_commit_patch(
        world,
        "calvin",
        {"branch_id": first_result.state_delta["actor_source_snapshot"]["branch_id"]},
        blocked,
        world.world_tick,
    )
    assert blocked.success is False
    assert blocked.failure_reason == "current_feature_repair_pending"

    second, _ = _apply(
        world, "calvin", "art_alpha_py", "FINAL_STEP = 2\n"
    )
    assert second.validation_status == "accepted"
    assert observed_before == ["VALUE = 0\n", "FIRST_STEP = 1\n"]
    assert not actor_patch_guard_failure(world, "calvin")


def test_declared_absent_path_executes_through_owner_scoped_create_candidate(
    monkeypatch,
):
    world = _private_world()
    from environments.org_env.backend.repo.repo import CIResult

    # Historical CI evidence is not part of a new-file transaction.  Its
    # read-only source projections must not prevent the owner from creating a
    # file later in the run.
    world.repo_system.repo.ci_runs["ci_old"] = CIResult(
        ci_id="ci_old",
        pr_id="pr_old",
        checks=[{
            "name": "historical_source_projection",
            "projection": MappingProxyType({"alpha.py": "VALUE = 0\n"}),
        }],
    )
    world._oss_component_map[F1] = ["alpha.py", "new_helper.py"]
    state = world._cooperbench_sdl_state
    state["feature_paths"][F1] = ["alpha.py", "new_helper.py"]
    state["required_feature_paths"][F1] = ["alpha.py", "new_helper.py"]

    candidates = _cooperbench_feature_edit_candidates(world, "victor")
    created = next(
        item
        for item in candidates
        if item.parameters.get("new_file_path") == "new_helper.py"
    )
    assert created.parameters["_create_repo_file"] is True
    assert created.parameters["_oss_issue"] == F1
    assert created.parameters["file_path"] == "new_helper.py"

    from environments.org_env.llm.code_editor import CodeEditorLLM
    from environments.org_env.product.repo_paths import repo_artifact_id

    created_artifact_id = repo_artifact_id("new_helper.py")

    def generate(self, **kwargs):
        assert kwargs["creates_file"] is True
        assert kwargs["target_object_id"] == created_artifact_id
        patch = _patch(
            world,
            kwargs["actor_id"],
            kwargs["target_object_id"],
            "CREATED_THROUGH_EXECUTOR = 1\n",
            creates=True,
        )
        patch.patch_id = kwargs["patch_id"]
        return patch

    monkeypatch.setattr(CodeEditorLLM, "generate_patch", generate)
    result = world._loop["execution"].execute("victor", created, world)

    assert result.success is True, result.failure_reason
    assert created_artifact_id in result.created_objects
    assert (
        sv.actor_file_text(world, "victor", created_artifact_id)
        == "CREATED_THROUGH_EXECUTOR = 1\n"
    )
    assert world.repo_system.repo.ci_runs["ci_old"].checks[0][
        "projection"
    ]["alpha.py"] == "VALUE = 0\n"


def test_failed_merged_feature_guard_rejects_before_private_or_native_publication(monkeypatch):
    world = _private_world()
    patch = _patch(world, "calvin", "art_alpha_py", "REGRESS_PEER = 0\n")
    desk_before = sv.actor_desk_snapshot(world, "calvin")
    repo_before = copy.deepcopy(world.repo_system.repo)
    pending_before = copy.deepcopy(getattr(world, "_pending_by_branch", {}))
    artifact_before = copy.deepcopy(world.product_artifacts["art_alpha_py"].__dict__)
    receipt = {
        "applicable": True, "available": True, "ok": False,
        "error": "merged_feature_probe_regression",
        "protected_features": [F1],
        "failed_protected_probe_ids": [F1 + ":0"],
        "source_snapshot": {"source_kind": "actor_patch_candidate", "snapshot_id": "candidate"},
        "behavior_evidence": {
            "probes": [{
                "probe_id": F1 + ":0",
                "requirement_ids": ["req_preserve_limit_before_transform"],
                "code": (
                    "resolved = block.resolve_audio()\n"
                    "actual_size = len(resolved.getvalue())\n"
                    "block.resolve_audio(max_bytes=10)\n"
                    "assert actual_size > 10"
                ),
            }],
            "results": [{
                "probe_id": F1 + ":0", "candidate": {
                    "status": "fail", "output": "expected 7",
                },
            }],
        },
    }
    monkeypatch.setattr(
        "environments.org_env.cooperbench.joint_probe_replay.guard_actor_patch_against_merged_features",
        lambda *args, **kwargs: copy.deepcopy(receipt),
    )
    result = _result()
    assert not world.apply_product_patch(patch, result, "calvin")
    assert result.failure_reason == "merged_feature_patch_guard_rejected:merged_feature_probe_regression"
    assert patch.validation_status == "rejected" and world.patches[patch.patch_id] is patch
    assert sv.actor_desk_snapshot(world, "calvin") == desk_before
    assert world.repo_system.repo == repo_before
    assert getattr(world, "_pending_by_branch", {}) == pending_before
    assert world.product_artifacts["art_alpha_py"].__dict__ == artifact_before
    assert result.state_delta["merged_feature_patch_guard"]["ok"] is False
    event = next(event for event in result.events
                 if event.get("subtype") == "merged_feature_patch_guard_rejected")
    assert event["failed_protected_probe_ids"] == [F1 + ":0"]
    failure = actor_patch_guard_failure(world, "calvin")
    assert failure["patch_id"] == patch.patch_id
    assert failure["schema_version"] == "cooperbench_actor_patch_guard_failure_v5"
    assert failure["failed_protected_probe_ids"] == [F1 + ":0"]
    assert "expected 7" in failure["brief"] and F1 in failure["brief"]
    assert "EXACT HOST-EXECUTED PUBLIC CHECK" in failure["brief"]
    assert "req_preserve_limit_before_transform" in failure["brief"]
    assert "block.resolve_audio(max_bytes=10)" in failure["brief"]
    from environments.org_env.cooperbench.actor_workspace import (
        actor_patch_guard_failure_history,
    )
    history = actor_patch_guard_failure_history(world, "calvin", F2)
    assert len(history) == 1
    assert history[0]["fingerprint"] == "candidate"
    assert history[0]["feature_ids"] == [F2]
    repair_candidates = _cooperbench_feature_edit_candidates(world, "calvin")
    assert repair_candidates
    assert all(
        "block.resolve_audio(max_bytes=10)" in item.parameters["edit_goal"]
        for item in repair_candidates
    )
    # Passing a later guard clears the immediate blocker, but an unresolved
    # semantic review must still carry the rejected implementation into the
    # editor prompt so the pair cannot oscillate back to it.
    world._cooperbench_actor_patch_guard_failures.pop("calvin", None)
    monkeypatch.setattr(
        "environments.org_env.runtime_adapter.execution._cooperbench_semantic_review_failure",
        lambda _world, _agent, _feature: "ordered public trim stage is still missing",
    )
    monkeypatch.setattr(
        "environments.org_env.runtime_adapter.execution._cooperbench_missing_feature_paths",
        lambda _world, _agent, _feature: [],
    )
    historical_candidates = _cooperbench_feature_edit_candidates(world, "calvin")
    assert historical_candidates
    historical_goal = historical_candidates[0].parameters["edit_goal"]
    assert "PREVIOUSLY EXECUTED REPAIR APPROACHES THAT FAILED" in historical_goal
    assert "Do not repeat or cosmetically rename them" in historical_goal
    assert "block.resolve_audio(max_bytes=10)" in historical_goal


@pytest.mark.parametrize("other_failures", [0, 5])
def test_incomplete_patch_guard_receipt_reaches_the_next_editor_prompt(monkeypatch, other_failures):
    world = _private_world()
    patch = _patch(world, "calvin", "art_alpha_py", "REFERENCE_MISSING = 1\n")
    receipt = {
        "applicable": True,
        "available": False,
        "ok": False,
        "error": "actor_patch_guard_execution_incomplete",
        "protected_features": [],
        "repair_feature": F2,
        "repair_probe_ids": [F2 + ":0"],
        "preserve_probe_ids": [],
        "failed_protected_probe_ids": [],
        "failed_repair_probe_ids": [],
        "failed_preservation_probe_ids": [],
        "incomplete_probe_ids": [F2 + ":0"],
        "repair_paths": ["alpha.py"],
        "source_snapshot": {
            "source_kind": "actor_patch_candidate",
            "snapshot_id": "incomplete-candidate",
        },
        "behavior_evidence": {
            "probes": [{
                "probe_id": F2 + ":0",
                "requirement_ids": ["req_public_constructor"],
                "paths": ["alpha.py"],
                "compare_baseline": False,
                "code": "import alpha\nassert alpha.make_value() == 2\n",
            }],
            "results": [{
                "probe_id": F2 + ":0",
                "baseline": None,
                "candidate": {
                    "status": "unavailable",
                    "error": "candidate_import_failed",
                    "output": "Traceback: NameError: public dependency is not defined",
                },
            }],
        },
    }
    # The production failure had many failed checks before the unavailable
    # check; the old three-row brief never showed the actual execution blocker.
    for index in range(other_failures):
        probe_id = F2 + f":other_{index}"
        receipt["behavior_evidence"]["probes"].insert(0, {
            "probe_id": probe_id, "code": "assert False\n" * 100,
            "requirement_ids": ["req_other"],
        })
        receipt["behavior_evidence"]["results"].insert(0, {
            "probe_id": probe_id,
            "candidate": {"status": "fail", "output": "ordinary failure\n" * 100},
        })
    monkeypatch.setattr(
        "environments.org_env.cooperbench.joint_probe_replay.guard_actor_patch_against_merged_features",
        lambda *args, **kwargs: copy.deepcopy(receipt),
    )

    result = _result()
    assert not world.apply_product_patch(patch, result, "calvin")
    event = next(event for event in result.events
                 if event.get("subtype") == "current_feature_repair_guard_rejected")
    assert event["incomplete_probe_ids"] == [F2 + ":0"]
    failure = actor_patch_guard_failure(world, "calvin")
    assert failure["incomplete_probe_ids"] == [F2 + ":0"]
    assert "CANDIDATE STATUS: unavailable" in failure["brief"]
    assert "ERROR: candidate_import_failed" in failure["brief"]
    assert "NameError: public dependency is not defined" in failure["brief"]
    assert "assert alpha.make_value() == 2" in failure["brief"]
    assert "Unavailable execution is not a product-code failure verdict" in failure["brief"]

    candidates = _cooperbench_feature_edit_candidates(world, "calvin")
    assert candidates
    assert all(
        "NameError: public dependency is not defined" in item.parameters["edit_goal"]
        for item in candidates
    )


def test_probe_oscillation_requires_a_disjoint_prior_failure(monkeypatch):
    from environments.org_env.cooperbench import joint_probe_replay as replay

    world = _private_world()
    world._cooperbench_actor_patch_guard_failures = {
        "calvin": {
            "feature_ids": [F2],
            "source_snapshot": {"tree_digest": "older-tree"},
            "failed_repair_probe_ids": [F2 + ":0"],
            "failed_preservation_probe_ids": [],
            "receipt": {"behavior_evidence": {"probes": [], "results": []}},
        }
    }
    monkeypatch.setattr(
        replay.sr,
        "_adjudicate_probe_defects",
        lambda *args, **kwargs: pytest.fail("overlapping failures are not oscillation"),
    )
    receipt = replay._adjudicate_oscillating_repair_probes(
        world,
        actor_id="calvin",
        binding={
            "repair_feature": F2,
            "source_snapshot": {"tree_digest": "new-tree"},
            "repair_plan": {"origin_reviewer": "victor"},
        },
        behavior={},
        failed_repair=[F2 + ":0"],
        failed_preservation=[],
    )
    assert receipt == {}


def test_probe_oscillation_needs_every_current_failure_confirmed_defective(monkeypatch):
    from environments.org_env.cooperbench import joint_probe_replay as replay

    world = _private_world()
    world._cooperbench_actor_patch_guard_failures = {
        "calvin": {
            "feature_ids": [F2],
            "source_snapshot": {"tree_digest": "older-tree"},
            "failed_repair_probe_ids": [F2 + ":0"],
            "failed_preservation_probe_ids": [],
            "receipt": {"behavior_evidence": {"probes": [], "results": []}},
        }
    }
    monkeypatch.setattr(replay.bp, "visible_review_contract", lambda *args: {
        "description": "Public behavior",
        "requirements": {"req": "Must support the public behavior"},
    })
    calls = []

    def adjudicate(*args, **kwargs):
        calls.append(kwargs)
        probe_ids = {row["probe_id"] for row in kwargs["defects"]}
        if probe_ids == {F2 + ":1"}:
            return (
                [{"probe_id": F2 + ":1", "adjudication_status": "confirmed"}],
                [], "", {"receipts": [{"probe_id": F2 + ":1", "verdict": "defective"}]},
            )
        assert probe_ids == {F2 + ":0"}
        return (
            [], [{"probe_id": F2 + ":0", "adjudication_status": "confirmed"}],
            "", {"receipts": [{"probe_id": F2 + ":0", "verdict": "valid"}]},
        )

    monkeypatch.setattr(replay.sr, "_adjudicate_probe_defects", adjudicate)
    receipt = replay._adjudicate_oscillating_repair_probes(
        world,
        actor_id="calvin",
        binding={
            "repair_feature": F2,
            "source_snapshot": {"tree_digest": "new-tree"},
            "repair_plan": {"origin_reviewer": "victor"},
        },
        behavior={"probes": [], "results": []},
        failed_repair=[],
        failed_preservation=[F2 + ":1"],
    )
    assert receipt["confirmed"] is True
    assert receipt["allow_patch"] is True
    assert receipt["current_failed_probe_ids"] == [F2 + ":1"]
    assert receipt["defective_probe_ids"] == [F2 + ":1"]
    assert receipt["valid_probe_ids"] == [F2 + ":0"]
    assert receipt["prior_failed_probe_ids"] == [[F2 + ":0"]]
    assert len(calls) == 2 and calls[0]["reviewer_id"] == "victor"

    monkeypatch.setattr(
        replay.sr,
        "_adjudicate_probe_defects",
        lambda *args, **kwargs: (
            ([{"probe_id": F2 + ":0", "adjudication_status": "confirmed"}], [], "",
             {"receipts": [{"probe_id": F2 + ":0", "verdict": "defective"}]})
            if {row["probe_id"] for row in kwargs["defects"]} == {F2 + ":0"}
            else ([], [{"probe_id": F2 + ":1", "adjudication_status": "confirmed"}], "",
                  {"receipts": [{"probe_id": F2 + ":1", "verdict": "valid"}]})
        ),
    )
    rejected = replay._adjudicate_oscillating_repair_probes(
        world,
        actor_id="calvin",
        binding={
            "repair_feature": F2,
            "source_snapshot": {"tree_digest": "newer-tree"},
            "repair_plan": {"origin_reviewer": "victor"},
        },
        behavior={"probes": [], "results": []},
        failed_repair=[],
        failed_preservation=[F2 + ":1"],
    )
    assert rejected["triggered"] is True
    assert rejected["confirmed"] is True
    assert rejected["allow_patch"] is False


def test_confirmed_probe_oscillation_only_routes_to_new_peer_review(monkeypatch):
    world = _private_world()
    probe_id = F2 + ":1"
    world._cooperbench_behavior_plans = {
        F2: {
            "identity": "old-plan",
            "probes": [{"probe_id": probe_id, "requirement_ids": ["req"]}],
        }
    }
    world._cooperbench_semantic_reviews = {
        F2: {
            "available": True,
            "approved": False,
            "reviewer_id": "victor",
            "pr_id": "pr_old",
            "source_snapshot": {"snapshot_id": "reviewed-head"},
            "behavior_evidence": {"plan_hash": "old-plan-hash"},
            "repair_paths": ["alpha.py"],
        }
    }
    receipt = {
        "applicable": True,
        "available": True,
        "ok": True,
        "error": "",
        "classification": "passed_with_peer_adjudicated_probe_correction",
        "protected_features": [],
        "repair_feature": F2,
        "repair_probe_ids": [],
        "preserve_probe_ids": [probe_id],
        "failed_protected_probe_ids": [],
        "failed_repair_probe_ids": [],
        "failed_preservation_probe_ids": [probe_id],
        "incomplete_probe_ids": [],
        "quarantined_probe_ids": [probe_id],
        "repair_paths": ["alpha.py"],
        "source_snapshot": {"source_kind": "actor_patch_candidate"},
        "probe_correction": {
            "triggered": True,
            "confirmed": True,
            "allow_patch": True,
            "feature_id": F2,
            "reviewer_id": "victor",
            "current_failed_probe_ids": [probe_id],
            "defective_probe_ids": [probe_id],
            "valid_probe_ids": [F2 + ":0"],
            "prior_failed_probe_ids": [[F2 + ":0"]],
            "probe_defects": [{
                "probe_id": probe_id,
                "adjudication_status": "confirmed",
            }],
            "probe_validity": {"receipts": [{
                "probe_id": probe_id,
                "verdict": "defective",
            }]},
        },
        "behavior_evidence": {},
    }
    monkeypatch.setattr(
        "environments.org_env.cooperbench.joint_probe_replay.guard_actor_patch_against_merged_features",
        lambda *args, **kwargs: copy.deepcopy(receipt),
    )

    patch, result = _apply(world, "calvin", "art_alpha_py", "CORRECTED = 1\n")

    assert patch.validation_status == "accepted"
    assert result.success is True
    assert not world._cooperbench_semantic_reviews[F2]["available"]
    assert world._cooperbench_semantic_reviews[F2]["error"] == (
        "public_behavior_probe_correction_required"
    )
    assert F2 not in world._cooperbench_behavior_plans
    gap = world._cooperbench_probe_evidence_gaps[F2]
    assert gap["repair_probe_ids"] == [probe_id]
    assert gap["origin"] == "actor_patch_guard_disjoint_failure_oscillation"
    authorization = world._cooperbench_probe_correction_authorizations[F2]
    assert authorization["probe_ids"] == [probe_id]
    assert authorization["source_snapshot"]["snapshot_id"] == (
        result.state_delta["actor_source_snapshot"]["snapshot_id"]
    )
    assert not world.repo_system.repo.pull_requests
    event = next(
        row for row in result.events
        if row.get("subtype") == "current_feature_repair_guard_passed"
    )
    assert event["quarantined_probe_ids"] == [probe_id]
    assert event["probe_correction_triggered"] is True


def test_mixed_validity_oscillation_quarantines_only_defect_and_keeps_patch_red(
    monkeypatch,
):
    world = _private_world()
    valid_id = F2 + ":0"
    defective_id = F2 + ":1"
    world._cooperbench_behavior_plans = {
        F2: {
            "identity": "old-plan",
            "probes": [
                {"probe_id": valid_id, "requirement_ids": ["req"]},
                {"probe_id": defective_id, "requirement_ids": ["req"]},
            ],
        }
    }
    world._cooperbench_semantic_reviews = {
        F2: {
            "available": True,
            "approved": False,
            "reviewer_id": "victor",
            "pr_id": "pr_old",
            "source_snapshot": {"snapshot_id": "reviewed-head"},
            "behavior_evidence": {"plan_hash": "old-plan-hash"},
            "repair_paths": ["alpha.py"],
        }
    }
    receipt = {
        "applicable": True,
        "available": True,
        "ok": False,
        "error": "current_feature_repair_unresolved",
        "classification": "rejected_unresolved_current_feature_repair",
        "protected_features": [],
        "repair_feature": F2,
        "repair_probe_ids": [valid_id, defective_id],
        "preserve_probe_ids": [],
        "failed_protected_probe_ids": [],
        "failed_repair_probe_ids": [valid_id, defective_id],
        "failed_preservation_probe_ids": [],
        "incomplete_probe_ids": [],
        "quarantined_probe_ids": [defective_id],
        "repair_paths": ["alpha.py"],
        "source_snapshot": {
            "source_kind": "actor_patch_candidate",
            "snapshot_id": "mixed-candidate",
        },
        "probe_correction": {
            "triggered": True,
            "confirmed": True,
            "allow_patch": False,
            "feature_id": F2,
            "reviewer_id": "victor",
            "current_failed_probe_ids": [valid_id, defective_id],
            "defective_probe_ids": [defective_id],
            "valid_probe_ids": [valid_id],
            "prior_failed_probe_ids": [[defective_id]],
            "probe_defects": [{
                "probe_id": defective_id,
                "adjudication_status": "confirmed",
            }],
            "probe_validity": {"receipts": [{
                "probe_id": defective_id,
                "verdict": "defective",
            }]},
        },
        "behavior_evidence": {},
    }
    monkeypatch.setattr(
        "environments.org_env.cooperbench.joint_probe_replay.guard_actor_patch_against_merged_features",
        lambda *args, **kwargs: copy.deepcopy(receipt),
    )

    patch, result = _apply(world, "calvin", "art_alpha_py", "MIXED_REPAIR = 1\n")

    assert patch.validation_status == "accepted"
    assert result.success is True
    assert F2 not in world._cooperbench_behavior_plans
    gap = world._cooperbench_probe_evidence_gaps[F2]
    assert gap["repair_probe_ids"] == [defective_id]
    assert [row["probe_id"] for row in gap["retained_probes"]] == [valid_id]
    assert world._cooperbench_semantic_reviews[F2]["error"] == (
        "public_behavior_probe_correction_required"
    )
    assert world._cooperbench_probe_correction_authorizations[F2][
        "source_snapshot"
    ]["snapshot_id"] == "mixed-candidate"
    assert not (world.__dict__.get("_cooperbench_green_repair_desks") or {}).get(
        "calvin", {}
    ).get(F2)
    event = next(
        row for row in result.events
        if row.get("subtype") == "current_feature_repair_guard_pending"
    )
    assert event["quarantined_probe_ids"] == [defective_id]
    assert event["probe_correction_triggered"] is True


def test_actor_public_tests_export_own_desk_and_do_not_authorize_peer(monkeypatch, tmp_path):
    world = _private_world()
    _apply(world, "victor", "art_alpha_py", "VICTOR_PRIVATE = 7\n")
    _apply(world, "calvin", "art_beta_0_py", "CALVIN_PRIVATE = 11\n")
    calls = _public_runtime(monkeypatch, tmp_path, world)
    result = _result()
    world._loop["execution"]._h_run_public_tests(world, "victor", {}, result, world.world_tick)
    assert result.success and result.state_delta["public_tests_passed"]
    assert calls == [dict(sv.actor_desk_snapshot(world, "victor").files)]
    assert calls[0]["beta_0.py"] == "VALUE = 0\n"
    assert _cooperbench_current_public_test_result(world, "victor") is True
    assert _cooperbench_current_public_test_result(world, "calvin") is None
    assert not world.__dict__.get("_public_tests_last")
    _apply(world, "calvin", "art_beta_0_py", "CALVIN_NEW = 12\n")
    assert _cooperbench_current_public_test_result(world, "victor") is True
    _apply(world, "victor", "art_alpha_py", "VICTOR_NEW = 8\n")
    assert _cooperbench_current_public_test_result(world, "victor") is None
    assert OrgActionMapper._public_tests_worth_running(world, "victor") is True


@pytest.mark.parametrize(
    ("green_current_desk", "failed_node", "expected_passed"),
    [
        (True, "FAILED tests/test_beta_0.py::test_old_shape", True),
        (False, "FAILED tests/test_beta_0.py::test_old_shape", False),
        (True, "FAILED tests/test_unrelated.py::test_real_regression", False),
    ],
)
def test_actor_public_tests_only_supersede_direct_stale_contract_tests(
    monkeypatch, green_current_desk, failed_node, expected_passed
):
    world = _private_world()
    _apply(world, "calvin", "art_beta_0_py", "VALUE = 11\n")
    snapshot = sv.actor_desk_snapshot(world, "calvin")
    if green_current_desk:
        world._cooperbench_green_repair_desks = {
            "calvin": {F2: snapshot.snapshot_id}
        }
    world._cooperbench_public_runtime_preflight = {
        "passed": True,
        "public_validation_strength": "public_regression",
        "functional_public_regression_available": True,
    }
    monkeypatch.setattr(
        "environments.org_env.product.materialize.run_public_tests",
        lambda *args, **kwargs: {
            "ok": False,
            "available": True,
            "returncode": 1,
            "error": None,
            "failed_tests": [failed_node],
            "summary": "1 failed, 9 passed",
            "failure_brief": "old starter expectation failed",
            "collected": 10,
        },
    )

    result = _result()
    world._loop["execution"]._h_run_public_tests(
        world, "calvin", {}, result, world.world_tick
    )

    record = actor_public_test_record(world, "calvin")
    assert result.success is True
    assert record["passed"] is expected_passed
    assert result.state_delta["public_tests_passed"] is expected_passed
    if expected_passed:
        compatibility = record["public_contract_compatibility"]
        assert compatibility["feature_id"] == F2
        assert compatibility["source_snapshot_id"] == snapshot.snapshot_id
        assert compatibility["superseded_test_modules"] == [
            "tests/test_beta_0.py"
        ]
        assert compatibility["official_evaluator_required"] is True
        assert record["failed_tests"] == []
        assert record["failure_brief"] == ""
        assert "old starter expectation failed" in record["superseded_failure_brief"]
    else:
        assert "public_contract_compatibility" not in record
        assert record["failed_tests"] == [failed_node]


def test_exact_actor_compatibility_receipt_carries_through_both_pr_ci_trees(
    monkeypatch,
):
    from environments.org_env.cooperbench.source_ci import run_pr_source_ci

    world = _private_world()
    _apply(world, "calvin", "art_beta_0_py", "VALUE = 11\n")
    desk = sv.actor_desk_snapshot(world, "calvin")
    world._cooperbench_green_repair_desks = {"calvin": {F2: desk.snapshot_id}}
    world._cooperbench_public_runtime_preflight = {
        "passed": True,
        "public_validation_strength": "public_regression",
        "functional_public_regression_available": True,
    }
    stale_failure = {
        "ok": False,
        "available": True,
        "returncode": 1,
        "error": None,
        "failed_tests": ["FAILED tests/test_beta_0.py::test_old_shape"],
        "summary": "1 failed, 9 passed",
        "failure_brief": "old starter expectation failed",
        "collected": 10,
    }
    monkeypatch.setattr(
        "environments.org_env.product.materialize.run_public_tests",
        lambda *args, **kwargs: copy.deepcopy(stale_failure),
    )

    public = _result()
    world._loop["execution"]._h_run_public_tests(
        world, "calvin", {}, public, world.world_tick
    )
    assert public.state_delta["public_tests_passed"] is True

    pr = _publish_private(world, "calvin")
    pr.linked_issue_ids = [F2]
    for patch_id in pr.patch_ids:
        world.patches[patch_id].related_issue_ids = [F2]

    verdict = run_pr_source_ci(world, pr)
    assert verdict["ok"] is True
    assert set(verdict["checks"]) == {"pr_head", "merge_candidate"}
    for role, check in verdict["checks"].items():
        assert check["ok"] is False  # raw old starter test is retained
        resolution = check["public_contract_compatibility"]
        assert resolution["feature_id"] == F2
        assert resolution["resolution_scope"] == "exact_committed_source_ci"
        assert resolution["ci_source_snapshot_id"] == check["source_snapshot"][
            "snapshot_id"
        ]


@pytest.mark.parametrize("mismatch", ["failure", "feature", "tree"])
def test_pr_ci_compatibility_receipt_fails_closed_on_identity_drift(
    monkeypatch, mismatch
):
    from environments.org_env.cooperbench.source_ci import run_pr_source_ci

    world = _private_world()
    _apply(world, "calvin", "art_beta_0_py", "VALUE = 11\n")
    desk = sv.actor_desk_snapshot(world, "calvin")
    world._cooperbench_green_repair_desks = {"calvin": {F2: desk.snapshot_id}}
    world._cooperbench_public_runtime_preflight = {"passed": True}
    stale_failure = {
        "ok": False,
        "available": True,
        "returncode": 1,
        "error": None,
        "failed_tests": ["FAILED tests/test_beta_0.py::test_old_shape"],
        "summary": "1 failed, 9 passed",
        "failure_brief": "old starter expectation failed",
        "collected": 10,
    }
    responses = [copy.deepcopy(stale_failure)]

    def public_tests(*args, **kwargs):
        return copy.deepcopy(responses[-1])

    monkeypatch.setattr(
        "environments.org_env.product.materialize.run_public_tests", public_tests
    )
    public = _result()
    world._loop["execution"]._h_run_public_tests(
        world, "calvin", {}, public, world.world_tick
    )
    assert public.state_delta["public_tests_passed"] is True
    pr = _publish_private(world, "calvin")
    pr.linked_issue_ids = [F1 if mismatch == "feature" else F2]
    if mismatch == "failure":
        responses.append(
            {
                **stale_failure,
                "failed_tests": [
                    "FAILED tests/test_beta_0.py::test_new_regression"
                ],
            }
        )
    elif mismatch == "tree":
        head = sv.pr_head_snapshot(world, pr)
        record = world._cooperbench_actor_public_tests["calvin"]
        record["public_contract_compatibility"]["source_tree_digest"] = (
            "not-" + head.tree_digest
        )

    verdict = run_pr_source_ci(world, pr)
    assert verdict["ok"] is False
    assert verdict["kind"] == "contract_break"
    assert "public_contract_compatibility" not in verdict["checks"]["pr_head"]


def test_real_b3_profile_reaches_private_edit_test_commit_and_fixed_pr_ci(monkeypatch, tmp_path):
    world = _private_world()
    calls = _public_runtime(monkeypatch, tmp_path, world)
    from environments.org_env.llm.code_editor import CodeEditorLLM

    def generate(self, **kwargs):
        patch = _patch(world, kwargs["actor_id"], kwargs["target_object_id"], "VALUE = 7\n")
        patch.patch_id = kwargs["patch_id"]
        assert kwargs["source_snapshot"].receipt() == patch.actor_source_snapshot
        return patch

    monkeypatch.setattr(CodeEditorLLM, "generate_patch", generate)
    for action_type in ("edit_repo_file", "run_public_tests", "commit_patch", "open_pr", "run_ci"):
        candidate = _profile_select(world, "victor", action_type)
        result = world._loop["execution"].execute("victor", candidate, world)
        assert result.success, (action_type, result.failure_reason)
        world.world_tick += 1
    pr = next(iter(world.repo_system.repo.pull_requests.values()))
    assert pr.ci_passed and pr.ci_source_snapshot["tree_digest"] == sv.pr_head_snapshot(world, pr).tree_digest
    assert sv.pr_head_snapshot(world, pr).files["alpha.py"] == "VALUE = 7\n"
    assert sv.actor_file_text(world, "calvin", "art_alpha_py") == "VALUE = 0\n"
    assert actor_public_test_record(world, "victor")["passed"]
    assert all(item["beta_0.py"] == "VALUE = 0\n" for item in calls)


def test_real_b3_profile_selects_explicit_sync_after_peer_merge(monkeypatch, tmp_path):
    world = _private_world()
    _public_runtime(monkeypatch, tmp_path, world)
    _apply(world, "victor", "art_alpha_py", "MERGED_PEER = 7\n")
    for action_type in ("run_public_tests", "commit_patch", "open_pr"):
        candidate = _profile_select(world, "victor", action_type)
        result = world._loop["execution"].execute("victor", candidate, world)
        assert result.success, result.failure_reason
        world.world_tick += 1
    pr = next(iter(world.repo_system.repo.pull_requests.values()))
    _merge_ledger(world, pr)
    assert sv.actor_file_text(world, "calvin", "art_alpha_py") == "VALUE = 0\n"
    before = sv.actor_desk_snapshot(world, "calvin")
    candidate = _profile_select(world, "calvin", "sync_actor_workspace")
    assert sv.actor_desk_snapshot(world, "calvin") == before  # selection never syncs
    result = world._loop["execution"].execute("calvin", candidate, world)
    assert result.success, result.failure_reason
    assert sv.actor_file_text(world, "calvin", "art_alpha_py") == "MERGED_PEER = 7\n"
    assert any(event.get("subtype") == "actor_workspace_synced" for event in result.events)


def test_parallel_conflict_restarts_owner_and_surfaces_fresh_feature_edit(
    monkeypatch, tmp_path
):
    world = _private_world()
    _public_runtime(monkeypatch, tmp_path, world)
    monkeypatch.setattr(
        "environments.org_env.cooperbench.joint_probe_replay.guard_actor_patch_against_merged_features",
        lambda *args, **kwargs: {"applicable": False, "ok": True},
    )
    state = world._cooperbench_sdl_state
    state["feature_paths"][F2] = ["alpha.py"]
    state["required_feature_paths"][F2] = ["alpha.py"]

    _apply(world, "victor", "art_alpha_py", "VICTOR_MERGED = 1\n")
    victor_pr = _publish_private(world, "victor")
    victor_pr.linked_issue_ids = [F1]
    for patch_id in victor_pr.patch_ids:
        world.patches[patch_id].related_issue_ids = [F1]

    _apply(world, "calvin", "art_alpha_py", "CALVIN_STALE = 1\n")
    calvin_pr = _publish_private(world, "calvin")
    calvin_pr.linked_issue_ids = [F2]
    for patch_id in calvin_pr.patch_ids:
        world.patches[patch_id].related_issue_ids = [F2]
    _merge_ledger(world, victor_pr)

    from environments.org_env.cooperbench.source_ci import run_pr_source_ci
    from environments.org_env.product.contracts import record_integration_verdict

    verdict = run_pr_source_ci(world, calvin_pr)
    assert verdict["kind"] == "coordination_conflict"
    ci = world.repo_system.run_ci(pr_id=calvin_pr.pr_id, tick=world.world_tick)
    record_integration_verdict(ci, calvin_pr, verdict, world=world)
    conflict = _cooperbench_coordination_conflict(world, "calvin")
    assert conflict is not None and conflict["feature_id"] == F2

    candidate = _profile_select(world, "calvin", "sync_actor_workspace")
    assert candidate.parameters["conflicted_pr_id"] == calvin_pr.pr_id
    result = world._loop["execution"].execute("calvin", candidate, world)
    assert result.success, result.failure_reason
    assert calvin_pr.status.value == "stale"
    assert conflict["status"] == "repairing"
    assert any(
        event.get("subtype") == "actor_workspace_conflict_restarted"
        for event in result.events
    )

    repair = _cooperbench_feature_edit_candidates(world, "calvin")
    assert len(repair) == 1
    assert repair[0].parameters["file_path"] == "alpha.py"
    assert repair[0].parameters["_cooperbench_repair_source"] == (
        "source_merge_conflict"
    )
    assert "peer's feature merged first" in repair[0].parameters["edit_goal"]
    assert sv.actor_file_text(world, "calvin", "art_alpha_py") == (
        "VICTOR_MERGED = 1\n"
    )

    _apply(
        world,
        "calvin",
        "art_alpha_py",
        "VICTOR_MERGED = 1\nCALVIN_REIMPLEMENTED = 1\n",
    )
    assert _cooperbench_feature_edit_candidates(world, "calvin") == []
    replacement_pr = _publish_private(world, "calvin")
    replacement_pr.linked_issue_ids = [F2]
    _merge_ledger(world, replacement_pr)
    _resolve_cooperbench_coordination_conflict(
        world, replacement_pr, world.world_tick
    )
    assert not world.__dict__.get("_cooperbench_coordination_required")
    assert world._cooperbench_coordination_history[-1]["status"] == "resolved"
    assert world._cooperbench_coordination_history[-1]["resolution_pr_id"] == (
        replacement_pr.pr_id
    )


def test_conflict_restart_requires_fresh_coverage_for_every_required_path(
    monkeypatch, tmp_path
):
    world = _private_world()
    _public_runtime(monkeypatch, tmp_path, world)
    monkeypatch.setattr(
        "environments.org_env.cooperbench.joint_probe_replay.guard_actor_patch_against_merged_features",
        lambda *args, **kwargs: {"applicable": False, "ok": True},
    )
    state = world._cooperbench_sdl_state
    state["feature_paths"][F2] = ["alpha.py", "beta_0.py"]
    state["required_feature_paths"][F2] = ["alpha.py", "beta_0.py"]

    _apply(world, "victor", "art_alpha_py", "VICTOR_MERGED = 1\n")
    victor_pr = _publish_private(world, "victor")
    victor_pr.linked_issue_ids = [F1]
    for patch_id in victor_pr.patch_ids:
        world.patches[patch_id].related_issue_ids = [F1]

    _apply(world, "calvin", "art_alpha_py", "CALVIN_STALE = 1\n")
    _apply(world, "calvin", "art_beta_0_py", "CALVIN_OLD_HELPER = 1\n")
    calvin_pr = _publish_private(world, "calvin")
    calvin_pr.linked_issue_ids = [F2]
    for patch_id in calvin_pr.patch_ids:
        world.patches[patch_id].related_issue_ids = [F2]
    _merge_ledger(world, victor_pr)

    from environments.org_env.cooperbench.source_ci import run_pr_source_ci
    from environments.org_env.product.contracts import record_integration_verdict

    verdict = run_pr_source_ci(world, calvin_pr)
    ci = world.repo_system.run_ci(pr_id=calvin_pr.pr_id, tick=world.world_tick)
    record_integration_verdict(ci, calvin_pr, verdict, world=world)
    restart = _profile_select(world, "calvin", "sync_actor_workspace")
    result = world._loop["execution"].execute("calvin", restart, world)
    assert result.success, result.failure_reason

    assert _cooperbench_missing_feature_paths(world, "calvin", F2) == [
        "alpha.py",
        "beta_0.py",
    ]
    assert feature_delivery_coverage(world)[F2]["complete"] is False
    assert _cooperbench_action_block_reason(
        world, "calvin", "run_public_tests", {}
    ) == "assigned_feature_required_paths_incomplete"

    _apply(
        world,
        "calvin",
        "art_alpha_py",
        "VICTOR_MERGED = 1\nCALVIN_REIMPLEMENTED = 1\n",
    )
    assert _cooperbench_missing_feature_paths(world, "calvin", F2) == [
        "beta_0.py"
    ]
    remaining = _cooperbench_feature_edit_candidates(world, "calvin")
    assert [item.parameters["file_path"] for item in remaining] == [
        "beta_0.py"
    ]
    assert _cooperbench_action_block_reason(
        world, "calvin", "run_public_tests", {}
    ) == "assigned_feature_required_paths_incomplete"

    _apply(world, "calvin", "art_beta_0_py", "CALVIN_FRESH_HELPER = 1\n")
    assert _cooperbench_missing_feature_paths(world, "calvin", F2) == []
    assert _cooperbench_feature_edit_candidates(world, "calvin") == []

    monkeypatch.setattr(
        "environments.org_env.runtime_adapter.execution._cooperbench_public_failure_brief",
        lambda _world, _agent: (
            "alpha.py:27: cannot use pattern as string in struct literal"
        ),
    )
    repairs = _cooperbench_feature_edit_candidates(world, "calvin")
    assert [item.parameters["file_path"] for item in repairs] == ["alpha.py"]
    assert repairs[0].parameters["_cooperbench_repair_source"] == "public_tests"
    assert repairs[0].parameters["_cooperbench_repair_pr_id"] == ""
    assert "cannot use pattern as string" in repairs[0].parameters["edit_goal"]


def test_review_request_ignores_stale_pr_and_targets_latest_live_owner_pr():
    world = _private_world()
    world._cooperbench_delivery_focus = False
    repo = world.repo_system.repo
    repo.pull_requests["pr_stale"] = PullRequest(
        pr_id="pr_stale",
        author_id="calvin",
        source_branch="branch_stale",
        linked_issue=F2,
        linked_issue_ids=[F2],
        reviewers=["victor"],
        status=PRStatus.STALE,
    )
    repo.pull_requests["pr_fresh"] = PullRequest(
        pr_id="pr_fresh",
        author_id="calvin",
        source_branch="branch_fresh",
        linked_issue=F2,
        linked_issue_ids=[F2],
        reviewers=["victor"],
        status=PRStatus.REVIEW_REQUESTED,
    )

    _, candidates = _native_pool(world, "calvin")
    requests = [
        item
        for item in candidates
        if item.action_type == "ask_for_review"
    ]
    assert [item.parameters["pr_id"] for item in requests] == ["pr_fresh"]
    assert _cooperbench_action_block_reason(
        world,
        "calvin",
        "ask_for_review",
        {"pr_id": "pr_stale", "target_agent": "victor"},
    ) == "feature_review_request_requires_live_owner_pr"


def test_native_create_create_reuses_path_descriptor_without_peer_source(monkeypatch):
    world = _private_world()
    from environments.org_env.llm.code_editor import CodeEditorLLM

    def generate(self, **kwargs):
        assert kwargs["creates_file"] is True
        assert "new_shared.py" not in kwargs["source_snapshot"].files
        patch = _patch(world, kwargs["actor_id"], kwargs["target_object_id"],
                       f"AUTHOR = '{kwargs['actor_id']}'\n", creates=True)
        patch.patch_id = kwargs["patch_id"]
        return patch

    monkeypatch.setattr(CodeEditorLLM, "generate_patch", generate)
    executor = world._loop["execution"]
    first = _result()
    artifact = executor._create_and_patch_repo_file(
        world, "victor", {"new_file_path": "new_shared.py", "_oss_issue": F1,
                          "edit_goal": "Create a new public module"}, first, world.world_tick)
    assert artifact is not None and first.success, (first.failure_reason, first.state_delta)
    world.world_tick += 1

    # The canonical descriptor now exists globally, but Calvin's private desk
    # still does not contain the path.  This is the exact state after a stale
    # create is abandoned during conflict restart.
    world._oss_component_map[F2] = ["new_shared.py"]
    state = world._cooperbench_sdl_state
    state["feature_paths"][F2] = ["new_shared.py"]
    state["required_feature_paths"][F2] = ["new_shared.py"]
    candidate = SimpleNamespace(
        action_type="edit_repo_file",
        parameters={
            "new_file_path": "new_shared.py",
            "file_path": "new_shared.py",
            "_create_repo_file": True,
            "_oss_issue": F2,
            "_cooperbench_required_path": True,
            "_blocker_fix": True,
            "edit_goal": "Recreate the assigned public module",
        },
    )
    second = executor.execute("calvin", candidate, world)

    assert second.success, (second.failure_reason, second.state_delta)
    assert artifact.content == "" and artifact.mainline_content == ""
    assert sv.actor_desk_snapshot(world, "victor").files["new_shared.py"] == "AUTHOR = 'victor'\n"
    assert sv.actor_desk_snapshot(world, "calvin").files["new_shared.py"] == "AUTHOR = 'calvin'\n"


def test_new_file_ci_conflict_routes_pr_owner_to_fresh_workspace_restart(monkeypatch):
    world = _private_world()
    state = world._cooperbench_sdl_state
    for feature_id in (F1, F2):
        world._oss_component_map[feature_id] = ["new_shared.py"]
        state["feature_paths"][feature_id] = ["new_shared.py"]
        state["required_feature_paths"][feature_id] = ["new_shared.py"]

    from environments.org_env.llm.code_editor import CodeEditorLLM

    def generate(self, **kwargs):
        patch = _patch(
            world,
            kwargs["actor_id"],
            kwargs["target_object_id"],
            f"AUTHOR = '{kwargs['actor_id']}'\n",
            creates=True,
        )
        patch.patch_id = kwargs["patch_id"]
        return patch

    monkeypatch.setattr(CodeEditorLLM, "generate_patch", generate)
    executor = world._loop["execution"]
    artifact = None
    for actor, feature_id in (("victor", F1), ("calvin", F2)):
        result = _result()
        artifact = executor._create_and_patch_repo_file(
            world,
            actor,
            {
                "new_file_path": "new_shared.py",
                "_oss_issue": feature_id,
                "edit_goal": "Create the shared public module",
            },
            result,
            world.world_tick,
        )
        assert artifact is not None and result.success, result.failure_reason

    victor_pr = _publish_private(world, "victor")
    victor_pr.linked_issue_ids = [F1]
    calvin_pr = _publish_private(world, "calvin")
    calvin_pr.linked_issue_ids = [F2]
    _merge_ledger(world, calvin_pr)
    artifact.mainline_revision = 1
    artifact.mainline_content = "AUTHOR = 'calvin'\n"

    ci_result = _result()
    executor._h_run_ci(
        world,
        "calvin",
        {"pr_id": victor_pr.pr_id},
        ci_result,
        world.world_tick,
    )

    assert not ci_result.success
    assert ci_result.failure_reason == "new_file_create_conflict"
    assert victor_pr.status == PRStatus.REVIEW_REQUESTED
    conflict = _cooperbench_coordination_conflict(world, "victor")
    assert conflict is not None
    assert conflict["owner_id"] == "victor"
    assert conflict["feature_id"] == F1
    assert conflict["conflict_paths"] == ["new_shared.py"]
    assert _cooperbench_coordination_conflict(world, "calvin") is None

    restart = _profile_select(world, "victor", "sync_actor_workspace")
    assert restart.parameters["conflicted_pr_id"] == victor_pr.pr_id
    restarted = executor.execute("victor", restart, world)
    assert restarted.success, restarted.failure_reason
    assert victor_pr.status == PRStatus.STALE
    assert conflict["status"] == "repairing"

    repairs = _cooperbench_feature_edit_candidates(world, "victor")
    assert [item.parameters["file_path"] for item in repairs] == [
        "new_shared.py"
    ]
    assert repairs[0].parameters["_cooperbench_repair_source"] == (
        "source_merge_conflict"
    )
    assert sv.actor_file_text(world, "victor", artifact.artifact_id) == (
        "AUTHOR = 'calvin'\n"
    )


def test_actor_outage_retries_without_replaying_global_green(monkeypatch, tmp_path):
    world = _private_world()
    _apply(world, "victor", "art_alpha_py", "VALUE = 7\n")
    calls = _public_runtime(monkeypatch, tmp_path, world,
                            response=lambda files, count: (None, "", "", "temporary outage")
                            if count == 1 else (0, "1 passed", "", None))
    world._public_tests_last = {"passed": True, "status": "passed"}
    first = _result()
    world._loop["execution"]._h_run_public_tests(world, "victor", {}, first, world.world_tick)
    assert not first.success and _cooperbench_current_public_test_result(world, "victor") is None
    second = _result()
    world._loop["execution"]._h_run_public_tests(world, "victor", {}, second, world.world_tick + 4)
    assert second.success and _cooperbench_current_public_test_result(world, "victor") is True
    assert len(calls) == 2 and calls[0] == calls[1]


def test_stale_actor_patch_does_not_erase_intervening_own_work():
    world = _private_world()
    stale = _patch(world, "victor", "art_alpha_py", "STALE = 2\n")
    _apply(world, "victor", "art_alpha_py", "LATEST = 3\n")
    current = sv.actor_desk_snapshot(world, "victor")
    result = _result()
    assert not world.apply_product_patch(stale, result, "victor")
    assert sv.actor_desk_snapshot(world, "victor") == current
    assert result.failure_reason.endswith("actor_desk_changed")
