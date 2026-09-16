"""Real-data contracts for the Secretary-integrated Resource Inspector."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.human.api import HumanApi
from environments.org_env.human.execution_team import ExecutionTeam
from environments.org_env.human.liaison import LiaisonFacade
from environments.org_env.human.working_agent import LiaisonTaskContext
from environments.org_env.llm.client import MockOrgLLMClient
from environments.org_env.backend.entities.work import Task
from environments.org_env.product.objects import ProductArtifact
from environments.org_env.product.patch_objects import CodePatch
from environments.org_env.proposals.objects import Proposal
from environments.org_env.runtime_adapter.live import OrgInspectorSession


SEAT = "victor"
REVIEW_SENTINEL = "The parser drops the final evidence record; add a regression test."
FILE_SENTINEL = "def parse_evidence():\n    return 'complete'\n"
DIFF_SENTINEL = (
    "--- a/src/evidence.py\n"
    "+++ b/src/evidence.py\n"
    "@@ -1 +1,2 @@\n"
    "+def parse_evidence():\n"
    "+    return 'complete'\n"
)


def _facade():
    session = OrgInspectorSession(seed=42)
    human = HumanApi(lambda: session, seconds_per_tick=60.0)
    facade = LiaisonFacade(human)
    token = facade.session()["token"]
    return facade, human, session, token


def _seed_real_pr(human: HumanApi):
    runtime = human.runtime()
    world = runtime.world
    artifact = ProductArtifact(
        artifact_id="artifact_inspector",
        artifact_type="code",
        title="Evidence parser",
        owner_agent_id=SEAT,
        linked_file_path="src/evidence.py",
        content=FILE_SENTINEL,
        mainline_content="# old parser\n",
        status="in_review",
    )
    world.product_artifacts[artifact.artifact_id] = artifact
    world.tasks["task_inspector"] = Task(
        task_id="task_inspector",
        title="Repair evidence parser",
        description="Preserve every evidence record and prove the fix in review.",
        owner_id=SEAT,
        linked_artifacts=[artifact.artifact_id],
    )
    patch = CodePatch(
        patch_id="patch_inspector",
        target_object_id=artifact.artifact_id,
        actor_id=SEAT,
        tick=10,
        edit_goal="Preserve every evidence record",
        new_content=FILE_SENTINEL,
        unified_diff=DIFF_SENTINEL,
        files_changed=[artifact.linked_file_path],
        change_summary="Repair truncated evidence parsing",
        validation_status="accepted",
        applied_tick=10,
    )
    world.patches[patch.patch_id] = patch
    branch = world.repo_system.create_branch(SEAT, linked_task="task_inspector", tick=9)
    commit = world.repo_system.commit_changes(
        agent_id=SEAT,
        branch_id=branch.branch_id,
        message="Repair evidence parser",
        changed_files=[artifact.linked_file_path],
        tick=11,
        test_status="passed",
        linked_task_id="task_inspector",
        patch_ids=[patch.patch_id],
        artifact_ids=[artifact.artifact_id],
    )
    assert commit is not None
    pr = world.repo_system.open_pr(
        agent_id=SEAT,
        source_branch=branch.branch_id,
        reviewers=["calvin"],
        linked_task="task_inspector",
    )
    pr.opened_tick = 12
    assert world.repo_system.review_pr(
        reviewer_id="calvin",
        pr_id=pr.pr_id,
        approve=False,
        comment=REVIEW_SENTINEL,
        tick=13,
    )
    ci = world.repo_system.run_ci(pr_id=pr.pr_id, tick=14)
    assert ci is not None
    world.events.append({
        "type": "task_progress_event",
        "subtype": "review_requested",
        "task_id": "task_inspector",
        "agent_id": SEAT,
        "tick": 14,
    })
    runtime.refresh_views()
    return pr, patch, artifact, commit, ci


def test_task_inspector_connects_real_files_diffs_reviews_ci_and_events():
    facade, human, _session, token = _facade()
    try:
        pr, patch, artifact, _commit, ci = _seed_real_pr(human)
        view = human.view(token)["view"]
        task_ref = next(
            ref for ref, row in facade._resource_index(token, view).items()
            if row["kind"] == "task" and row["id"] == "task_inspector"
        )
        team = ExecutionTeam(
            human.runtime(), llm_provider=lambda: MockOrgLLMClient(script=[]))
        human._execution_team = team
        prepared = team.prepare(SEAT, {
            "title": "Inspect the evidence parser",
            "references": [{"object_id": "task_inspector"}],
            "workers": [{"name": "Reviewer", "role": "reviewer",
                         "assignment": "Inspect the task evidence."}],
        })
        human._agent(SEAT)._append(
            "agent", "Grounded review of the evidence parser.", kind="task_review",
            references=[],
            evidence_records=[{
                "evidence_id": "evidence_linked_source",
                "worker_id": "secretary",
                "worker_name": "Secretary",
                "tool": "read_repo",
                "args": {"path": artifact.linked_file_path},
                "source_text": FILE_SENTINEL,
            }, {
                "evidence_id": "evidence_test_source",
                "worker_id": "secretary",
                "worker_name": "Secretary",
                "tool": "read_repo",
                "args": {"path": "tests/test_evidence.py"},
                "source_text": "def test_preserves_every_record():\n    assert True\n",
            }],
        )
        task_context = LiaisonTaskContext("Inspect the task's linked source")
        task_context.begin_step(1)
        task_context.record_observation(
            "read_repo", {"path": artifact.linked_file_path}, FILE_SENTINEL)
        assert human._agent(SEAT)._object_references_from_evidence(
            task_context.evidence_records()) == [{"object_id": "task_inspector"}]
        state = facade.state(token)
        state_job = state["execution_jobs"][0]
        assert state_job["resource_ref"] == task_ref
        assert state_job["resource_refs"] == [{
            "ref": task_ref, "kind": "task", "title": "Repair evidence parser",
        }]
        assert "evidence_records" not in next(
            message for message in state["conversation"]
            if message.get("kind") == "task_review")
        report = next(message for message in state["conversation"]
                      if message.get("kind") == "task_review")
        assert report["resource_ref"] == task_ref

        overview = facade.resource(token, task_ref, "overview")
        assert overview["execution_context"]["jobs"][0]["job_id"] == (
            prepared["job"]["job_id"])
        assert overview["content"]["task"]["description"] == (
            "Preserve every evidence record and prove the fix in review.")
        assert overview["content"]["related_pull_requests"][0]["id"] == pr.pr_id
        assert overview["content"]["related_events"][0]["subtype"] == "review_requested"
        assert overview["content"]["linked_files"][0]["path"] == artifact.linked_file_path
        secretary_evidence = overview["secretary_context"]["evidence_records"]
        test_evidence = next(row for row in secretary_evidence
                             if row["args"]["path"] == "tests/test_evidence.py")
        assert "test_preserves_every_record" in test_evidence["source_text"]

        files = facade.resource(token, task_ref, "files")
        assert files["content"]["files"][0]["content"] == FILE_SENTINEL
        diff = facade.resource(token, task_ref, "diff")
        assert diff["content"]["diffs"][0]["patch_id"] == patch.patch_id
        assert diff["content"]["diffs"][0]["unified_diff"] == DIFF_SENTINEL
        review = facade.resource(token, task_ref, "review")
        assert review["content"]["comments"][0]["comment"] == REVIEW_SENTINEL
        tests = facade.resource(token, task_ref, "ci")
        assert tests["content"]["runs"][0]["id"] == ci.ci_id
        assert tests["content"]["runs"][0]["pull_request_id"] == pr.pr_id
    finally:
        human.shutdown()


def test_task_files_report_a_specific_empty_state_and_private_tasks_stay_hidden():
    _facade_api, human, _session, _token = _facade()
    try:
        world = human.runtime().world
        world.tasks["task_without_files"] = Task(
            task_id="task_without_files", title="Plan evidence review", owner_id=SEAT)
        world.tasks["task_private"] = Task(
            task_id="task_private", title="Private work", owner_id="sean", visibility="private")
        human.runtime().refresh_views()

        empty = human.runtime().seat_resource(SEAT, "task", "task_without_files", "files")
        assert empty["availability"]["files"] == {
            "available": False,
            "reason": "no_visible_task_linked_file_content",
        }
        assert empty["content"] == {"files": []}
        assert human.runtime().seat_resource(
            SEAT, "task", "task_private", "overview")["error"] == "resource_not_visible"
    finally:
        human.shutdown()


def test_pr_inspector_returns_exact_recorded_review_ci_file_and_diff():
    facade, human, _session, _token = _facade()
    try:
        pr, patch, artifact, commit, ci = _seed_real_pr(human)
        runtime = human.runtime()

        review = runtime.seat_resource(SEAT, "pull_request", pr.pr_id, "review")
        recorded = review["content"]["comments"][0]
        assert recorded == {
            "reviewer": "calvin",
            "approve": False,
            "comment": REVIEW_SENTINEL,
            "review_id": "review_1",
            "tick": 13,
        }

        tests = runtime.seat_resource(SEAT, "pull_request", pr.pr_id, "ci")
        assert tests["content"]["runs"][0]["id"] == ci.ci_id
        assert tests["content"]["runs"][0]["checks"] == ci.checks
        assert tests["content"]["runs"][0]["failure_reasons"] == ci.failure_reasons

        files = runtime.seat_resource(SEAT, "pull_request", pr.pr_id, "files")
        assert files["content"]["files"][0]["path"] == artifact.linked_file_path
        assert files["content"]["files"][0]["content"] == FILE_SENTINEL
        assert len(files["content"]["files"][0]["content_hash"]) == 64

        diff = runtime.seat_resource(SEAT, "pull_request", pr.pr_id, "diff")
        assert diff["content"]["diffs"][0]["unified_diff"] == DIFF_SENTINEL
        assert diff["content"]["diffs"][0]["patch_id"] == patch.patch_id

        provenance = runtime.seat_resource(SEAT, "pull_request", pr.pr_id, "provenance")
        assert commit.commit_id in provenance["content"]["commit_ids"]
        assert patch.patch_id in provenance["content"]["patch_ids"]
        assert ci.ci_id in provenance["content"]["ci_run_ids"]
        assert provenance["provenance"]["parent_chain_revalidated"] is True
    finally:
        human.shutdown()


def test_rich_resource_reads_do_not_wait_for_a_model_backed_world_tick():
    _facade_api, human, _session, _token = _facade()
    release = threading.Event()
    entered = threading.Event()
    holder = None
    try:
        pr, _patch, _artifact, _commit, _ci = _seed_real_pr(human)
        runtime = human.runtime()

        def hold_world_lock():
            with runtime.lock:
                entered.set()
                release.wait(1.0)

        holder = threading.Thread(target=hold_world_lock)
        holder.start()
        assert entered.wait(0.2)
        started = time.monotonic()
        review = runtime.seat_resource(SEAT, "pull_request", pr.pr_id, "review")
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
        assert review["content"]["comments"][0]["comment"] == REVIEW_SENTINEL
    finally:
        release.set()
        if holder is not None:
            holder.join(timeout=1.0)
        human.shutdown()


def test_pr_child_material_fails_closed_when_not_linked_through_a_commit():
    facade, human, _session, _token = _facade()
    try:
        pr, _patch, artifact, _commit, _ci = _seed_real_pr(human)
        unrelated = CodePatch(
            patch_id="patch_unrelated",
            target_object_id=artifact.artifact_id,
            actor_id=SEAT,
            tick=15,
            new_content="SECRET_UNRELATED_CONTENT",
            unified_diff="SECRET_UNRELATED_DIFF",
            validation_status="accepted",
        )
        human.runtime().world.patches[unrelated.patch_id] = unrelated
        pr.patch_ids.append(unrelated.patch_id)
        human.runtime().refresh_views()

        files = human.runtime().seat_resource(SEAT, "pull_request", pr.pr_id, "files")
        diff = human.runtime().seat_resource(SEAT, "pull_request", pr.pr_id, "diff")
        assert "SECRET_UNRELATED_CONTENT" not in str(files)
        assert "SECRET_UNRELATED_DIFF" not in str(diff)
    finally:
        human.shutdown()


def test_nonparticipant_cannot_use_a_visible_pr_id_as_a_resource_capability():
    facade, human, _session, _token = _facade()
    try:
        pr, _patch, _artifact, _commit, _ci = _seed_real_pr(human)
        human.claim("sean")
        hidden = human.runtime().seat_resource("sean", "pull_request", pr.pr_id, "review")
        assert hidden["error"] == "resource_not_visible"
        assert REVIEW_SENTINEL not in str(hidden)
    finally:
        human.shutdown()


def test_resource_handle_is_token_bound_and_rechecks_current_seat_visibility():
    """An opaque P3 handle must not turn a visible PR into a transferable capability."""
    facade, human, _session, token = _facade()
    try:
        pr, _patch, _artifact, _commit, _ci = _seed_real_pr(human)
        victor_view = human.view(token)["view"]
        victor_ref = next(
            ref for ref, row in facade._resource_index(token, victor_view).items()
            if row["kind"] == "pull_request" and row["id"] == pr.pr_id
        )
        assert victor_ref.startswith("ri_")
        assert pr.pr_id not in victor_ref
        assert facade.resource(token, victor_ref, "review")["content"]["comments"][0]["comment"] == REVIEW_SENTINEL

        other_token = human.claim("sean")["token"]
        # P3 is fixed to Victor, and the resource adapter independently denies
        # Sean the same PR even if he learns its raw id.
        assert facade.resource(other_token, victor_ref, "review")["error"] == "p3_fixed_seat_required:victor"
        hidden = human.runtime().seat_resource("sean", "pull_request", pr.pr_id, "review")
        assert hidden["error"] == "resource_not_visible"
        assert REVIEW_SENTINEL not in str(hidden)
    finally:
        human.shutdown()


def test_secretary_model_can_open_exact_source_and_the_ref_is_revalidated():
    facade, human, _session, token = _facade()
    try:
        pr, _patch, _artifact, _commit, _ci = _seed_real_pr(human)
        before_prs = len(human.runtime().world.repo_system.repo.pull_requests)
        human._agent(SEAT).llm = MockOrgLLMClient(script=[{
            "route": "open_resource",
            "resource_kind": "pull_request",
            "resource_id": pr.pr_id,
            "resource_section": "review",
            "reply": "I opened the recorded review and its source material.",
        }])

        opened = facade.ask(token, "把这个 PR 的原始 review 打开，我要看具体意见")
        assert opened["handled"] == "open_resource"
        focus = opened["resource_focus"]
        assert focus["ref"].startswith("ri_")
        assert focus["section"] == "review"
        assert len(human.runtime().world.repo_system.repo.pull_requests) == before_prs

        state = facade.state(token)
        message = next(item for item in state["conversation"]
                       if item.get("kind") == "resource_opened")
        assert message["resource_focus"] == focus
        source = facade.resource(token, focus["ref"], "review")
        assert source["content"]["comments"][0]["comment"] == REVIEW_SENTINEL

        assert facade.release(token)["released"] == SEAT
        assert facade.resource(token, focus["ref"], "review")["error"] == "invalid_seat_token"
    finally:
        human.shutdown()


def test_open_resource_failure_never_records_a_false_opening_acknowledgement():
    facade, human, _session, token = _facade()
    try:
        human._agent(SEAT).llm = MockOrgLLMClient(script=[{
            "route": "open_resource",
            "resource_kind": "task",
            "resource_id": "task_no_longer_visible",
            "resource_section": "overview",
            "reply": "正在打开 Dashboard 任务的完整上下文供您查看。",
        }, {
            "route": "open_resource",
            "resource_kind": "task",
            "resource_section": "overview",
            "reply": "正在打开 Dashboard 任务的完整上下文供您查看。",
        }])

        missing_target = facade.ask(token, "打开 Dashboard 任务的原始上下文")
        assert missing_target["handled"] == "resource_clarification"
        assert "resource_focus" not in missing_target
        target_reply = next(
            row for row in facade.state(token)["conversation"]
            if row.get("id") == missing_target["message_id"])
        assert "not currently visible" in target_reply["text"]
        assert "正在打开" not in target_reply["text"]

        missing_identity = facade.ask(token, "只查看那个 Dashboard 任务")
        assert missing_identity["handled"] == "resource_clarification"
        assert "resource_focus" not in missing_identity
        identity_reply = next(
            row for row in facade.state(token)["conversation"]
            if row.get("id") == missing_identity["message_id"])
        assert "resource identity was missing" in identity_reply["text"]
        assert "正在打开" not in identity_reply["text"]

        task = next(row for row in human.view(token)["view"]["objects"]["tasks"])
        human._agent(SEAT).llm = MockOrgLLMClient(script=[{
            "route": "open_resource",
            "resource_kind": "task",
            "resource_id": task["id"],
            "resource_section": "overview",
            "reply": "正在打开 Dashboard 任务的完整上下文供您查看。",
        }])
        runtime = human.runtime()
        original_resolver = runtime.seat_resource
        runtime.seat_resource = lambda *_args, **_kwargs: {"error": "resource_unavailable"}
        try:
            resolver_failure = facade.ask(token, "打开这个任务的原始上下文")
        finally:
            runtime.seat_resource = original_resolver
        assert resolver_failure["handled"] == "resource_clarification"
        assert "resource_focus" not in resolver_failure
        resolver_reply = next(
            row for row in facade.state(token)["conversation"]
            if row.get("id") == resolver_failure["message_id"])
        assert "could not be loaded; nothing was opened" in resolver_reply["text"]
        assert "正在打开" not in resolver_reply["text"]
    finally:
        human.shutdown()


def test_generic_review_like_objects_still_open_through_the_same_inspector():
    facade, human, _session, token = _facade()
    try:
        view = human.view(token)["view"]
        candidate = next(
            row for rows in view["objects"].values() for row in rows
            if row.get("kind") and row.get("id")
            and row.get("kind") not in ("pull_request", "task")
        )
        resource = human.runtime().seat_resource(
            SEAT, candidate["kind"], candidate["id"], "overview")
        assert resource["resource"]["kind"] == candidate["kind"]
        assert resource["content"] == candidate
        unsupported = human.runtime().seat_resource(
            SEAT, candidate["kind"], candidate["id"], "diff")
        assert unsupported["availability"]["diff"]["available"] is False
        assert unsupported["content"] == {}
    finally:
        human.shutdown()


def test_proposal_review_uses_the_same_inspector_with_complete_decision_material():
    facade, human, _session, _token = _facade()
    try:
        proposal = Proposal(
            proposal_id="proposal_inspector",
            proposal_type="workflow_proposal",
            title="Require reproducible review evidence",
            summary="Every review should link its original evidence.",
            proposer_agent_id="calvin",
            target_problem="Summaries cannot currently be audited.",
            proposed_solution="Attach a seat-scoped source inspector.",
            required_actions=["review source", "record verdict"],
            required_artifacts=["review packet"],
            expected_benefits=["traceable decisions"],
            expected_costs=["one lazy read"],
            risks=["visibility leakage"],
            failure_modes=["stale opaque handle"],
            approval_required_from=[SEAT],
            status="under_review",
            created_at_tick=7,
        )
        human.runtime().world.proposal_manager.proposals[proposal.proposal_id] = proposal
        human.runtime().refresh_views()

        packet = human.runtime().seat_resource(
            SEAT, "proposal", proposal.proposal_id, "review")
        assert packet["content"]["target_problem"] == proposal.target_problem
        assert packet["content"]["proposed_solution"] == proposal.proposed_solution
        assert packet["content"]["risks"] == proposal.risks
        assert packet["content"]["failure_modes"] == proposal.failure_modes
        assert packet["content"]["approval_required_from"] == [SEAT]
    finally:
        human.shutdown()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
