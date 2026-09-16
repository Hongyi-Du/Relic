from __future__ import annotations

from types import SimpleNamespace

import pytest

from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.cooperbench import actor_workspace, source_views
from environments.org_env.cooperbench.source_views import SourceViewError
from environments.org_env.cooperbench.work_schedule import (
    install_compressed_schedule,
    install_model_call_budget,
)


def test_actor_desk_patch_dispatch_precedes_global_artifact_mutation(monkeypatch):
    """The source-owned private desk is the only patch path when it is active."""

    seen = {}

    def apply(world, patch, result, actor_id):
        seen.update(
            world=world,
            patch=patch,
            result=result,
            actor_id=actor_id,
        )
        return True

    monkeypatch.setattr(source_views, "actor_desks_enabled", lambda _world: True)
    monkeypatch.setattr(actor_workspace, "apply_actor_product_patch", apply)
    world = SimpleNamespace(product_artifacts={})
    patch = SimpleNamespace(target_object_id="not-in-global-artifacts")
    result = SimpleNamespace()

    assert OrgWorld.apply_product_patch(world, patch, result, "victor") is True
    assert seen == {
        "world": world,
        "patch": patch,
        "result": result,
        "actor_id": "victor",
    }


def test_feature_completion_credit_requires_the_assigned_owner_and_delivery(
    monkeypatch,
):
    """A peer's edit to a shared path cannot complete the other Cooper feature."""

    feature_id = "cooper_feature_1"
    task = SimpleNamespace(
        task_id=f"task_oss_{feature_id}",
        owner_id="victor",
        linked_artifacts=["shared.py"],
        completion_requirements=[],
        progress_evidence=[],
        progress_score=0.0,
    )
    artifact = SimpleNamespace(
        revision=1,
        artifact_type="code",
        known_gaps=[],
        linked_proposal_ids=[],
    )
    peer_patch = SimpleNamespace(
        actor_id="calvin",
        related_issue_ids=[feature_id],
        target_object_id="shared.py",
        validation_status="accepted",
    )
    own_patch = SimpleNamespace(
        actor_id="victor",
        related_issue_ids=[feature_id],
        target_object_id="shared.py",
        validation_status="accepted",
    )
    world = SimpleNamespace(
        _cooperbench_sdl_state={"feature_owners": {feature_id: "victor"}},
        patches={"peer": peer_patch, "own": own_patch},
        _WEAK_ACTIONS=set(),
        agents={"victor": object(), "calvin": object()},
    )
    world._cooperbench_task_assignment = lambda candidate: (
        OrgWorld._cooperbench_task_assignment(world, candidate)
    )

    assert (
        OrgWorld._evidence_relevance(
            world, task, "edit_code", "shared.py", "peer"
        )
        == "none"
    )
    assert (
        OrgWorld._evidence_relevance(
            world, task, "edit_code", "shared.py", "own"
        )
        == "substantive"
    )

    import environments.org_env.cooperbench.lifecycle as lifecycle

    monkeypatch.setattr(
        lifecycle,
        "feature_delivery_coverage",
        lambda _world: {feature_id: {"complete": False}},
    )
    assert OrgWorld._task_requirements_met(world, task, artifact) is False
    assert task.progress_score == 0.0

    monkeypatch.setattr(
        lifecycle,
        "feature_delivery_coverage",
        lambda _world: {feature_id: {"complete": True}},
    )
    assert OrgWorld._task_requirements_met(world, task, artifact) is True
    assert task.progress_score == 1.0


def _merge_world(*, patch_present: bool):
    artifact = SimpleNamespace(
        artifact_id="shared.py",
        mainline_revision=0,
        last_reviewed_tick=None,
        awaiting_review=True,
        status="needs_review",
        mainline_content="UNCOMMITTED_GLOBAL_TEXT",
        content="UNCOMMITTED_GLOBAL_TEXT",
        linked_pr_ids=[],
    )
    world = SimpleNamespace(
        product_artifacts={"shared.py": artifact},
        patches=(
            {"patch": SimpleNamespace(new_content="")}
            if patch_present
            else {}
        ),
        repo_system=SimpleNamespace(
            merged_commit_patches=lambda _pr: [("patch", "shared.py")]
        ),
        events=[],
    )
    world.apply_merge_task_evidence = lambda *_args: None
    world._retire_verdicts_made_against_the_old_mainline = lambda *_args: None
    world.reconcile = lambda **_kwargs: None
    return world, artifact


def test_source_view_merge_promotes_exact_committed_text_and_fails_closed(
    monkeypatch,
):
    """An empty committed patch is not replaced by later global desk content."""

    monkeypatch.setattr(source_views, "source_views_enabled", lambda _world: True)
    pr = SimpleNamespace(
        pr_id="pr_1", linked_issue_ids=[], linked_issue=None, approved_by=[]
    )
    world, artifact = _merge_world(patch_present=True)

    assert OrgWorld.apply_merged_pr(world, pr, "calvin", tick=4) == ["shared.py"]
    assert artifact.mainline_content == ""
    assert artifact.mainline_revision == 1

    missing_patch_world, _ = _merge_world(patch_present=False)
    with pytest.raises(SourceViewError, match="merge_patch_full_text_missing"):
        OrgWorld.apply_merged_pr(missing_patch_world, pr, "calvin", tick=4)


def test_delivery_focus_disables_the_generic_repo_workflow_sweep():
    world = SimpleNamespace(_cooperbench_delivery_focus=True)

    assert OrgWorld.process_repo_workflow(world) == []


def test_compressed_cooper_schedule_keeps_hard_gates_and_bypasses_calendar():
    clock = SimpleNamespace(
        current_tick=3,
        phase_of_day="night_sleep",
        is_weekend=False,
        is_after_hours=True,
        is_late_night=False,
    )
    availability = SimpleNamespace(
        current_availability_status="available",
        after_hours_responsiveness=0.0,
        weekend_work_tendency=0.0,
    )
    world = SimpleNamespace(
        time=SimpleNamespace(
            availability={"victor": availability}, rhythm_enabled=True, clock=clock
        )
    )
    agent = SimpleNamespace(id="victor", vitals={"attention": 1.0}, work_state=None)

    assert OrgWorld._agent_action_block_reason(world, agent, clock) == (
        "night_sleep_unavailable"
    )
    install_compressed_schedule(world)
    install_model_call_budget(world, ["victor", "calvin"])
    assert OrgWorld._agent_action_block_reason(world, agent, clock) is None

    world._cooperbench_model_call_budget["logical_calls_by_member"]["victor"] = 500
    assert OrgWorld._agent_action_block_reason(world, agent, clock) == (
        "model_call_budget_exhausted"
    )


def test_delivery_focus_memory_uses_explicit_reads_and_full_message_text():
    class Memory:
        def __init__(self):
            self.turns = []

        def record_turn(self, message_id, text):
            self.turns.append((message_id, text))

    memory = Memory()
    unread = SimpleNamespace(
        message_id="unread",
        created_tick=1,
        text_summary="UNREAD_SUMMARY",
        full_text="UNREAD_FULL_TEXT",
        sender_id="calvin",
        read_by=set(),
    )
    read = SimpleNamespace(
        message_id="read",
        created_tick=2,
        text_summary="READ_SUMMARY",
        full_text="READ_FULL_TEXT",
        sender_id="calvin",
        read_by={"victor"},
    )
    world = SimpleNamespace(
        _cooperbench_delivery_focus=True,
        reflection_manager=SimpleNamespace(_memory=lambda _world, _aid: memory),
        comm=SimpleNamespace(perceivable_messages=lambda _aid: [unread, read]),
        agent_memories={"victor": object()},
    )

    OrgWorld._remember_the_conversation(world, "victor")

    assert memory.turns == [("read", "[t2] calvin: READ_FULL_TEXT")]
