"""Focused contract tests for real P3 execution workers.

Run: PYTHONPATH=. python tests/org_env/test_human_execution_team.py
"""
import json
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.human.api import HumanApi
from environments.org_env.human import gateway
from environments.org_env.human.execution_team import ExecutionTeam, WORKER_DECISION_SHAPE
from environments.org_env.human.liaison import LiaisonFacade
from environments.org_env.human.affordances import all_offered_action_specs
from environments.org_env.human.working_agent import SeatTools, WorkingAgentSession
from environments.org_env.backend.entities import Task
from environments.org_env.product.objects import ProductArtifact
from environments.org_env.runtime_adapter.live import OrgInspectorSession
from agent_sdk.lived.core.contracts import ActionCandidate


class ScriptedModel:
    """Records each model context and supplies a small independent-worker trace."""

    def __init__(self, responses, text_responses=None):
        self.responses = list(responses)
        self.calls = []
        self.text_responses = list(text_responses or [])
        self.text_calls = []

    def generate_json(self, system, user, schema, **_kwargs):
        self.calls.append((system, user, schema))
        if not self.responses:
            raise AssertionError("unexpected model call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def generate_text(self, system, user, **_kwargs):
        self.text_calls.append((system, user))
        if not self.text_responses:
            raise AssertionError("unexpected text model call")
        return self.text_responses.pop(0)


class BlockingModel(ScriptedModel):
    """Hold one worker response so cancellation/concurrent start can be tested."""

    def __init__(self, first_response, responses):
        super().__init__(responses)
        self.first_response = first_response
        self.entered = threading.Event()
        self.release = threading.Event()
        self._first = True

    def generate_json(self, system, user, schema, **kwargs):
        if self._first:
            self._first = False
            self.calls.append((system, user, schema))
            self.entered.set()
            assert self.release.wait(2.0)
            return self.first_response
        return super().generate_json(system, user, schema, **kwargs)


def _api(model=None):
    session = OrgInspectorSession(seed=42)
    session.step(1)
    if model is not None:
        session.world.llm_client = model
    return HumanApi(lambda: session, seconds_per_tick=60.0)


def _wait(team, job_id):
    for _ in range(100):
        state = team.public(job_id)
        if state.get("status") in {"completed", "failed", "blocked", "cancelled"}:
            return state
        time.sleep(0.01)
    raise AssertionError(f"execution job did not settle: {team.public(job_id)!r}")


def test_execution_workers_are_independent_model_contexts_and_gateway_backed():
    api = _api()
    try:
        victor = api.claim("victor")
        token = victor["token"]
        rt = api.runtime()
        model = ScriptedModel([
            # Worker A: a consequential action, its separate semantic audit,
            # then its next independent decision.
            {"action": {"action_type": "send_message",
                        "params": {"channel_id": "team_general", "text": "Availability verified."}}},
            {"approve": True, "reason": "Requested and grounded."},
            {"report": "## Files\n\n- Availability action completed."},
            {"complete": True, "reason": "The gateway evidence supports this report."},
            # Worker B's own first decision -- no copied scripted action plan.
            {"report": "Checked the visible work context; no further action needed."},
            {"complete": True, "reason": "The assigned review is complete."},
            {"summary": "Secretary summary: one verified Victor action completed."},
        ])
        team = ExecutionTeam(rt, llm_provider=lambda: model,
                             on_complete=api._execution_complete)
        api._execution_team = team
        agent = api._agent("victor")
        agent._append("human", "Please carry this out.", thread_id="thread_victor")

        prepared = team.prepare("victor", {
            "title": "Verify and update availability",
            "workers": [
                {"name": "Operator", "role": "operator",
                 "assignment": "Set Victor availability only if visible facts support it."},
                {"name": "Reviewer", "role": "reviewer",
                 "assignment": "Review the resulting visible state and report facts."},
            ],
        }, request="Update availability after verification.", thread_id="thread_victor")
        assert prepared["ok"] is True
        job_id = prepared["job"]["job_id"]
        assert set(prepared["job"]) == {
            "job_id", "created_at", "title", "goal", "completion_criteria", "status", "startable",
            "start_block_reason", "cancelable", "workers", "timeline", "final_report", "secretary_model_calls",
            "thread_id", "source", "runtime_epoch", "settled_at", "references",
            "delivery_branch_id",
            "evidence_records", "task_id", "claim_task", "task_allocations",
            "progress_interval_seconds",
        }
        assert prepared["job"]["source"] == "liaison_execution_runtime"
        assert api.execution_start(token, job_id)["ok"] is True
        result = _wait(team, job_id)

        assert result["status"] == "completed"
        assert result["thread_id"] == "thread_victor"
        assert all(worker["model_calls"] > 0 for worker in result["workers"])
        assert any(row["action_type"] == "send_message" and row["ok"]
                   for row in result["timeline"] if "action_type" in row)
        assert rt.world.controller_log[-1]["execution_mode"] == "liaison_subagent"
        assert "verified Victor action" in result["final_report"]
        assert "## Worker reports" in result["final_report"]
        assert "## Files" in result["final_report"]
        assert "Availability action completed." in result["final_report"]
        assert result["workers"][0]["model_call_started_at"] is None
        assert result["workers"][0]["model_call_phase"] == ""
        assert result["secretary_model_calls"] == 1
        assert any(message.thread_id == "thread_victor" and
                   message.kind == "execution_summary" for message in agent.transcript)
        # The model saw distinct worker identifiers and a separate audit call.
        model_payloads = [user for _system, user, _schema in model.calls]
        assert sum("liaison_worker_" in payload for payload in model_payloads) >= 3
        assert any("Audit the proposed consequential action" in system
                   for system, _user, _schema in model.calls)
    finally:
        api.shutdown()


def test_transient_worker_model_failure_retries_the_same_turn_without_losing_job():
    api = _api()
    try:
        victor = api.claim("victor")
        model = ScriptedModel([
            RuntimeError("temporary provider interruption"),
            {"report": "The visible state was inspected after retry."},
            {"complete": True, "reason": "The report is grounded."},
            {"summary": "Secretary delivered the retried result."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Retry one interrupted turn", "goal": "Inspect visible state.",
            "completion_criteria": "Return one grounded report.",
            "workers": [{"name": "Reader", "role": "reviewer",
                         "assignment": "Inspect and report visible state."}],
        })

        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        settled = _wait(team, prepared["job"]["job_id"])

        assert settled["status"] == "completed"
        assert settled["workers"][0]["model_calls"] == 3
        assert len(model.calls) == 4
        assert victor["token"]
    finally:
        api.shutdown()


def test_execution_summary_retains_exact_worker_sources_for_the_inspector():
    api = _api()
    try:
        api.claim("victor")
        api._agent("victor")
        view = api.runtime().seat_view("victor")
        task = (view.get("objects") or {}).get("tasks", [])[0]
        task_id = task["id"]
        evidence = {
            "evidence_id": "evidence_dashboard_source",
            "worker_id": "liaison_worker_review",
            "worker_name": "Reviewer",
            "tool": "read_repo",
            "args": {"path": "src/dashboard.py"},
            "source_text": "def create_dashboard():\n    raise NotImplementedError\n",
            "step": 2,
        }

        api._execution_complete({
            "seat_id": "victor",
            "job_id": "exec_review",
            "status": "completed",
            "final_report": "Dashboard review completed with source evidence.",
            "thread_id": "",
            "references": [{"object_id": task_id, "kind": "task"}],
            "evidence_records": [evidence],
            "timeline": [],
        })

        summary = next(message for message in reversed(api._agent("victor").transcript)
                       if message.kind == "execution_summary")
        assert summary.references[0]["object_id"] == task_id
        assert summary.evidence_records == [evidence]
    finally:
        api.shutdown()


def test_worker_retries_once_after_a_real_gateway_failure():
    api = _api()
    try:
        victor = api.claim("victor")
        model = ScriptedModel([
            {"action": {"action_type": "pick_task",
                        "params": {"task_id": "task_missing"}}},
            {"approve": True, "reason": "Try the requested task."},
            {"action": {"action_type": "send_message",
                        "params": {"channel_id": "team_general", "text": "Recovery verified."}}},
            {"approve": True, "reason": "Corrected using the failure evidence."},
            {"report": "Recovered and updated availability."},
            {"complete": True, "reason": "The corrected gateway action succeeded."},
            {"summary": "Secretary summary: the worker recovered from one failed action."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        api._execution_team = team
        prepared = team.prepare("victor", {
            "title": "Recover one action", "goal": "Complete one valid update.",
            "completion_criteria": "A corrected gateway action succeeds.",
            "workers": [{"name": "Operator", "role": "operator",
                         "assignment": "Try the target, inspect failure, and correct it."}],
        }, request="Complete the valid update.")

        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        final = _wait(team, prepared["job"]["job_id"])

        assert final["status"] == "completed"
        assert any(row.get("kind") == "action_recovery" and row.get("status") == "retrying"
                   for row in final["timeline"])
        actions = [row for row in final["timeline"] if row.get("kind") == "action"]
        assert [row["ok"] for row in actions] == [False, True]
        assert final["workers"][0]["model_calls"] >= 6
        assert api.agent_state(victor["token"])["execution_agents"][0]["status"] == "inactive"
    finally:
        api.shutdown()


def test_worker_gets_one_catalog_schema_correction_instead_of_abandoning_the_job():
    api = _api()
    try:
        api.claim("victor")
        model = ScriptedModel([
            {"kind": "action", "action": {
                "action_type": "commit_patch",
                "params": {"message": "finish dashboard"},
            }},
            {"kind": "action", "action": {
                "action_type": "send_message",
                "params": {"channel_id": "team_general", "text": "Schema corrected."},
            }},
            {"approve": True, "reason": "The corrected action matches the catalog."},
            {"kind": "report", "report": "Corrected the action and completed the update."},
            {"complete": True, "reason": "A valid gateway action succeeded."},
            {"summary": "Secretary: the worker corrected one catalog mismatch."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Correct one malformed action",
            "goal": "Complete one valid organization update.",
            "completion_criteria": "A catalog-valid action succeeds.",
            "workers": [{"name": "Operator", "role": "operator",
                         "assignment": "Correct a malformed action and finish the update."}],
        })

        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        final = _wait(team, prepared["job"]["job_id"])

        assert final["status"] == "completed", final
        recovery = next(row for row in final["timeline"]
                        if row.get("kind") == "action_recovery")
        assert "invalid_action_schema:commit_patch" in recovery["summary"]
        worker_context = json.loads(model.calls[1][1])
        mismatch = worker_context["recent_worker_history"][-1]
        assert mismatch["params"] == {"message": "finish dashboard"}
        assert mismatch["expected_required"] == []
        assert mismatch["expected_optional"] == ["branch_id", "patch_id"]
        assert "edit_repo_file, run_public_tests, commit_patch, open_pr" in model.calls[0][0]
    finally:
        api.shutdown()


def test_completion_cannot_accept_an_edit_left_outside_every_commit():
    api = _api()
    try:
        api.claim("victor")
        model = ScriptedModel([])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        job = {
            "job_id": "exec_delivery_check",
            "status": "running",
            "timeline": [],
        }
        worker = {
            "worker_id": "worker_delivery_check",
            "status": "running",
            "timeline": [{
                "kind": "action",
                "action_type": "edit_repo_file",
                "ok": True,
                "state_delta": {"patch_id": "patch_uncommitted"},
            }, {
                "kind": "action",
                "action_type": "open_pr",
                "ok": True,
                "state_delta": {"pr_id": "pr_incomplete"},
            }],
        }

        team._finish_worker_report(job, worker, "Everything is complete.")

        assert worker["status"] == "blocked"
        assert job["timeline"][-1]["uncommitted_patch_ids"] == ["patch_uncommitted"]
        assert model.calls == [], "an invariant violation is not delegated to prose judgment"
    finally:
        api.shutdown()


def test_worker_gets_one_live_tool_directory_correction_for_unknown_tool():
    api = _api()
    try:
        api.claim("victor")
        model = ScriptedModel([
            {"kind": "tool", "tool": "read_knowledge", "args": {}},
            {"kind": "tool", "tool": "list_repo", "args": {}},
            {"kind": "report", "report": "Recovered and inspected the repository."},
            {"complete": True, "reason": "The corrected repository read is retained."},
            {"summary": "Secretary: the worker corrected one tool mismatch."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Correct one unknown tool",
            "goal": "Inspect the visible repository.",
            "completion_criteria": "Repository evidence exists.",
            "workers": [{"name": "Reader", "role": "engineer",
                         "assignment": "Inspect the repository and report."}],
        })

        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        final = _wait(team, prepared["job"]["job_id"])

        assert final["status"] == "completed", final
        recovery = next(row for row in final["timeline"]
                        if row.get("kind") == "tool_recovery")
        assert recovery["tool"] == "read_knowledge"
        corrected_context = json.loads(model.calls[1][1])
        mismatch = corrected_context["recent_worker_history"][-1]
        assert mismatch["status"] == "invalid_tool"
        assert "list_repo" in mismatch["available_tools"]
        assert any(row.get("tool") == "list_repo" and row.get("status") == "evidence"
                   for row in final["timeline"])
    finally:
        api.shutdown()


def test_worker_can_refresh_a_dynamic_observation_after_visible_state_changes():
    api = _api()
    try:
        api.claim("victor")
        task_id = next(iter(api.runtime().world.tasks))

        class RefreshingModel(ScriptedModel):
            def generate_json(self, system, user, schema, **kwargs):
                if len(self.calls) == 1:
                    api.runtime().world.tasks[task_id].title = "Changed while worker waited"
                    api.runtime().refresh_views()
                return super().generate_json(system, user, schema, **kwargs)

        model = RefreshingModel([
            {"kind": "tool", "tool": "read_object", "args": {"object_id": task_id}},
            {"kind": "tool", "tool": "read_object", "args": {"object_id": task_id}},
            {"kind": "report", "report": "Observed the task before and after its visible change."},
            {"complete": True, "reason": "Both task states are retained as evidence."},
            {"summary": "Secretary: the worker observed the asynchronous state change."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Observe one asynchronous change",
            "goal": "Wait for and observe the task change.",
            "completion_criteria": "The task state change is visible.",
            "workers": [{"name": "Observer", "role": "reviewer",
                         "assignment": "Observe the selected task until its state changes."}],
        })

        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        final = _wait(team, prepared["job"]["job_id"])

        reads = [row for row in final["timeline"]
                 if row.get("kind") == "tool" and row.get("tool") == "read_object"]
        assert final["status"] == "completed", final
        assert len(reads) == 2
        assert reads[0]["evidence_id"] != reads[1]["evidence_id"]
        second_worker_context = json.loads(model.calls[1][1])
        assert second_worker_context["task_anchor"]["worker_completion_criteria"] == (
            "Observe the selected task until its state changes.")
        assert second_worker_context["task_anchor"]["job_completion_criteria"] == (
            "The task state change is visible.")
    finally:
        api.shutdown()


def test_existing_pr_identity_is_preserved_as_successful_open_pr_recovery(monkeypatch):
    api = _api()
    try:
        api.claim("victor")
        model = ScriptedModel([
            {"kind": "action", "action": {
                "action_type": "open_pr", "params": {"branch_id": "branch_victor"},
                "rationale": "Resume the already opened request."}},
            {"approve": True, "reason": "The branch and request are grounded."},
            {"kind": "report", "report": "Continued with existing pull request pr_existing."},
            {"complete": True, "reason": "The existing pull request identity was recovered."},
            {"summary": "Secretary: existing pull request pr_existing was recovered."},
        ])

        class ExistingPrResult:
            success = False
            action_id = "action_open_existing"
            failure_reason = "already_requested"
            created_objects = []
            modified_objects = []
            state_delta = {"pr_id": "pr_existing"}

        monkeypatch.setattr(
            "environments.org_env.human.execution_team.gateway.submit",
            lambda *_args, **_kwargs: ExistingPrResult())
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Resume existing PR", "goal": "Continue a prior delivery.",
            "completion_criteria": "The existing PR is recovered.",
            "workers": [{"name": "Delivery", "role": "engineer",
                         "assignment": "Resume the existing pull request and report its id."}],
        })

        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        final = _wait(team, prepared["job"]["job_id"])

        action = next(row for row in final["timeline"]
                      if row.get("action_type") == "open_pr")
        assert final["status"] == "completed", final
        assert action["ok"] is True
        assert action["created"] == ["pr_existing"]
        assert action["state_delta"] == {"pr_id": "pr_existing"}
        assert action["recovered_from"] == "already_requested"
        report_payload = json.loads(model.calls[3][1])
        assert report_payload["recent_worker_history"][-1]["created"] == ["pr_existing"]
    finally:
        api.shutdown()


def test_execution_agent_persists_inactive_and_is_reused_with_prior_context():
    api = _api()
    try:
        victor = api.claim("victor")
        model = ScriptedModel([
            {"report": "First review completed with the visible evidence."},
            {"complete": True, "reason": "The first review is grounded."},
            {"summary": "First job settled."},
            {"report": "Follow-up review used the earlier verified context."},
            {"complete": True, "reason": "The revision review is grounded."},
            {"summary": "Follow-up job settled."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        api._execution_team = team

        first = team.prepare("victor", {
            "title": "Review the first version", "goal": "Review one task version.",
            "completion_criteria": "Return an evidence-bounded review.",
            "workers": [{"name": "River", "role": "reviewer",
                         "assignment": "Review the current version."}],
        }, request="Review the current version.")
        worker_id = first["job"]["workers"][0]["worker_id"]
        assert team.agents_for("victor") == [], "a pending plan must not create a ghost agent"
        assert team.start("victor", first["job"]["job_id"])["ok"] is True
        first_final = _wait(team, first["job"]["job_id"])
        first_snapshot = json.loads(json.dumps(first_final, sort_keys=True))

        inactive = team.agents_for("victor")
        assert len(inactive) == 1
        assert inactive[0]["worker_id"] == worker_id
        assert inactive[0]["status"] == "inactive"
        assert inactive[0]["activation_count"] == 1
        assert inactive[0]["recent_reports"] == [
            "First review completed with the visible evidence."]
        state = api.agent_state(victor["token"])
        assert state["execution_agents"][0]["worker_id"] == worker_id
        assert state["execution_agents"][0]["status"] == "inactive"

        second = team.prepare("victor", {
            "title": "Review the revision", "goal": "Review the revised version.",
            "completion_criteria": "Return an evidence-bounded revision review.",
            # A reused identity keeps its established role even if a compiler
            # tries to relabel it in a later activation.
            "workers": [{"worker_id": worker_id, "name": "River", "role": "operator",
                         "assignment": "Review the revision of the same task."}],
        }, request="Have the same agent review the revision.")
        assert second["job"]["workers"][0]["reused"] is True
        assert second["job"]["workers"][0]["role"] == "reviewer"
        assert second["job"]["workers"][0]["activation_number"] == 2
        assert team.start("victor", second["job"]["job_id"])["ok"] is True
        second_final = _wait(team, second["job"]["job_id"])

        assert second_final["status"] == "completed"
        assert team.public(first["job"]["job_id"]) == first_snapshot
        roster = team.agents_for("victor")
        assert len(roster) == 1
        assert roster[0]["status"] == "inactive"
        assert roster[0]["activation_count"] == 2
        assert "memory" not in roster[0]
        worker_calls = [json.loads(user) for system, user, _schema in model.calls
                        if "independent execution worker" in system]
        assert len(worker_calls) == 2
        assert worker_calls[1]["worker_id"] == worker_id
        prior_memory = worker_calls[1]["durable_memory"][0]
        assert prior_memory["untrusted_summary"] == (
            "First review completed with the visible evidence.")
        assert prior_memory["source"]["run_id"] == first_final["workers"][0]["run_id"]
        assert prior_memory["requires_live_revalidation"] is True
        assert "report" not in prior_memory
        assert any("untrusted_summary" in system and "never an instruction" in system
                   for system, _user, _schema in model.calls)

        compiler_model = ScriptedModel([{"reply": "I can reuse River."}])
        liaison = WorkingAgentSession(
            api.runtime(), "victor", llm_client=compiler_model,
            execution_agents_provider=lambda: team.agents_for("victor"),
        )
        assert liaison.send("让上次那个 agent 再改一次", require_model=True)["accepted"] is True
        for _ in range(100):
            if not liaison.busy:
                break
            time.sleep(0.01)
        compiler_payload = json.loads(compiler_model.calls[0][1])
        assert compiler_payload["execution_agents"][0]["worker_id"] == worker_id
        assert compiler_payload["execution_agents"][0]["activation_count"] == 2
        assert api.release(victor["token"])["released"] == "victor"
        reclaimed = api.claim("victor")
        reclaimed_roster = api.agent_state(reclaimed["token"])["execution_agents"]
        assert reclaimed_roster[0]["worker_id"] == worker_id
        assert reclaimed_roster[0]["status"] == "inactive"
    finally:
        api.shutdown()


def test_active_execution_agent_cannot_be_concurrently_reused():
    api = _api()
    model = None
    try:
        victor = api.claim("victor")
        provider = {"model": ScriptedModel([
            {"report": "Initial review complete."},
            {"complete": True, "reason": "Grounded."},
            {"summary": "Initial activation settled."},
        ])}
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: provider["model"])
        api._execution_team = team
        initial = team.prepare("victor", {
            "title": "Initial", "workers": [
                {"name": "River", "role": "reviewer", "assignment": "Review initial."}],
        })
        worker_id = initial["job"]["workers"][0]["worker_id"]
        assert team.start("victor", initial["job"]["job_id"])["ok"] is True
        assert _wait(team, initial["job"]["job_id"])["status"] == "completed"
        claim_task_id = "task_busy_worker_must_not_claim"
        api.runtime().world.tasks[claim_task_id] = Task(
            task_id=claim_task_id, title="Busy-worker claim guard",
            description="Remain unclaimed when the requested worker is busy.",
            owner_id=None, visibility="team")
        api.runtime().refresh_views()

        model = BlockingModel(
            {"report": "First job review complete."},
            [{"complete": True, "reason": "Grounded."}, {"summary": "Done."}],
        )
        provider["model"] = model
        first = team.prepare("victor", {
            "title": "First", "workers": [
                {"worker_id": worker_id, "name": "River", "role": "reviewer",
                 "assignment": "Review first."}],
        })
        second = team.prepare("victor", {
            "title": "Second", "task_id": claim_task_id, "claim_task": True,
            "workers": [
                {"worker_id": worker_id, "name": "River", "role": "reviewer",
                 "assignment": "Review second."}],
        })
        assert team.start("victor", first["job"]["job_id"])["ok"] is True
        assert model.entered.wait(1.0)
        active = team.agents_for("victor")[0]
        assert active["status"] == "active"
        assert active["active_job_id"] == first["job"]["job_id"]
        blocked_pending = team.public(second["job"]["job_id"])
        assert blocked_pending["startable"] is False
        assert blocked_pending["start_block_reason"] == f"execution_agent_busy:{worker_id}"
        assert team.start("victor", first["job"]["job_id"]) == {
            "error": "execution_job_not_startable:running"}
        assert team.start("victor", second["job"]["job_id"]) == {
            "error": f"execution_agent_busy:{worker_id}"}
        assert api.runtime().world.tasks[claim_task_id].owner_id is None
        model.release.set()
        assert _wait(team, first["job"]["job_id"])["status"] == "completed"
        assert team.agents_for("victor")[0]["status"] == "inactive"
        assert victor["token"]
    finally:
        if model is not None:
            model.release.set()
        api.shutdown()


def test_world_reset_fences_old_compiler_sink_and_completion_callback():
    first_session = OrgInspectorSession(seed=42)
    first_session.step(1)
    holder = {"session": first_session}
    api = HumanApi(lambda: holder["session"], seconds_per_tick=60.0)
    try:
        assert api.claim("victor")["agent_id"] == "victor"
        old_agent = api._agent("victor")
        old_plan_sink = old_agent._execution_plan_sink
        old_team = api._execution_team
        assert old_plan_sink is not None and old_team is not None
        old_completion = old_team._on_complete
        assert old_completion is not None

        replacement = OrgInspectorSession(seed=99)
        replacement.step(1)
        holder["session"] = replacement
        api.runtime()  # binds a new world and invalidates the old epoch
        victor = api.claim("victor")
        current_agent = api._agent("victor")
        before_messages = len(current_agent.transcript)
        before_logs = len(api.get_hci_logs())

        stale = old_plan_sink("victor", {
            "title": "Stale job",
            "workers": [{"name": "Old", "role": "reviewer",
                         "assignment": "Must never enter the replacement world."}],
        }, "stale request", "thread_old")
        assert stale == {"error": "stale_runtime_discarded"}
        assert api._execution_team.jobs_for("victor") == []

        old_completion({
            "seat_id": "victor", "job_id": "exec_old", "status": "completed",
            "final_report": "A late summary from the discarded world.",
            "thread_id": "thread_old", "timeline": [],
        })
        assert len(current_agent.transcript) == before_messages
        assert len(api.get_hci_logs()) == before_logs
        assert victor["token"]
    finally:
        api.shutdown()


def test_cancelled_late_worker_response_is_fenced_before_gateway():
    api = _api()
    try:
        victor = api.claim("victor")
        model = BlockingModel(
            {"action": {"action_type": "send_message",
                        "params": {"channel_id": "team_general", "text": "late"}}},
            [{"summary": "Cancelled without executing the late response."}],
        )
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        api._execution_team = team
        prepared = team.prepare("victor", {
            "title": "Cancelable", "workers": [
                {"name": "River", "role": "operator", "assignment": "Update availability."}],
        })
        job_id = prepared["job"]["job_id"]
        assert team.start("victor", job_id)["ok"] is True
        assert model.entered.wait(1.0)
        assert team.cancel("victor", job_id)["ok"] is True
        model.release.set()
        final = _wait(team, job_id)
        assert final["status"] == "cancelled"
        assert not api.runtime().world.controller_log
        assert any(row["kind"] == "stale_response_discarded"
                   for row in final["timeline"])
        assert team.agents_for("victor")[0]["status"] == "inactive"
    finally:
        model.release.set()
        api.shutdown()


def test_execution_start_cancel_are_token_scoped_and_pending_cancel_is_real():
    api = _api()
    try:
        victor = api.claim("victor")
        sean = api.claim("sean")
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: ScriptedModel([]))
        api._execution_team = team
        prepared = team.prepare("victor", {
            "title": "Do nothing yet",
            "workers": [{"name": "Worker", "role": "reviewer",
                         "assignment": "Inspect current work and report."}],
        }, request="Inspect.")
        job_id = prepared["job"]["job_id"]

        assert "error" in api.execution_start("not-a-seat-token", job_id)
        # Another claimed human cannot start or cancel Victor's job.
        assert api.execution_start(sean["token"], job_id) == {
            "error": "execution_job_not_found"}
        assert api.execution_cancel(sean["token"], job_id) == {
            "error": "execution_job_not_found"}

        cancelled = api.execution_cancel(victor["token"], job_id)
        assert cancelled["ok"] is True
        assert cancelled["job"]["status"] == "cancelled"
        assert cancelled["job"]["cancelable"] is False
        assert "execution_job_not_startable:cancelled" == api.execution_start(
            victor["token"], job_id)["error"]
    finally:
        api.shutdown()


def test_report_without_evidence_can_only_leave_an_implementation_worker_blocked():
    api = _api()
    try:
        victor = api.claim("victor")
        model = ScriptedModel([
            {"report": "Implementation is done."},
            {"complete": False, "reason": "No visible tool or action evidence supports implementation."},
            {"summary": "Secretary: execution remains blocked because no evidence was produced."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        api._execution_team = team
        prepared = team.prepare("victor", {
            "title": "Implement a change", "goal": "Implement and verify a real change.",
            "completion_criteria": "A real P1/P2 action or inspection evidence exists.",
            "workers": [{"name": "Implementer", "role": "engineer",
                         "assignment": "Implement the requested change."}],
        }, request="Implement a change.")
        job_id = prepared["job"]["job_id"]
        assert api.execution_start(victor["token"], job_id)["ok"] is True
        result = _wait(team, job_id)
        assert result["status"] == "blocked"
        assert result["workers"][0]["status"] == "blocked"
    finally:
        api.shutdown()


def test_releasing_a_human_seat_cancels_that_seats_execution_authority():
    api = _api()
    try:
        victor = api.claim("victor")
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: ScriptedModel([]))
        api._execution_team = team
        prepared = team.prepare("victor", {
            "title": "Pending release test",
            "workers": [{"name": "Worker", "role": "reviewer",
                         "assignment": "Inspect visible work and report."}],
        }, request="Inspect.")
        job_id = prepared["job"]["job_id"]
        assert api.release(victor["token"])["released"] == "victor"
        released = team.public(job_id)
        assert released["status"] == "cancelled"
        assert released["final_report"]
        assert team.agents_for("victor") == []
        reclaimed = api.claim("victor")
        assert api.agent_state(reclaimed["token"])["execution_agents"] == []
    finally:
        api.shutdown()


def test_liaison_natural_language_path_prepares_then_runs_a_real_job():
    model = ScriptedModel([
        {"route": "ordinary", "reply": ""},
        {"execution_plan": {
            "title": "Verify availability", "goal": "Verify and update availability.",
            "completion_criteria": "A gateway action and evidence-backed report exist.",
            "workers": [{"name": "Operator", "role": "operator",
                         "assignment": "Verify and set availability if grounded."}],
        }},
        {"action": {"action_type": "send_message",
                    "params": {"channel_id": "team_general",
                               "text": "Availability verified."}}},
        {"approve": True, "reason": "Grounded in the assigned goal."},
        {"report": "Availability was verified and updated."},
        {"complete": True, "reason": "Gateway evidence confirms the update."},
        {"summary": "Secretary summary: availability update was verified."},
    ])
    api = _api(model)
    facade = LiaisonFacade(api)
    try:
        token = facade.session()["token"]
        accepted = facade.ask(token, "Please have a small team verify and update my availability.")
        assert accepted["accepted"] is True
        for _ in range(100):
            pending = facade.state(token).get("execution_jobs") or []
            if pending:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("compiler did not create an execution job")
        job = pending[0]
        assert job["status"] == "pending_confirmation"
        assert not api.runtime().world.controller_log, "planning must not mutate the world"

        assert facade.execution_start(token, job["job_id"])["ok"] is True
        final = _wait(api._execution_team, job["job_id"])
        assert final["status"] == "completed", final
        worker_call = next((call for call in model.calls
                            if "independent execution worker" in call[0]), None)
        assert worker_call is not None
        worker_payload = __import__("json").loads(worker_call[1])
        actual_catalog = [{key: row[key] for key in
                           ("action_type", "label", "required", "optional", "target_param", "confirm")}
                           for row in worker_payload["action_catalog"]]
        expected_catalog = [spec.schema() for spec in all_offered_action_specs()]
        assert actual_catalog == expected_catalog
        assert sum(row["action_type"] == "run_ci" for row in actual_catalog) == 2
        assert any(row["execution_mode"] == "liaison_subagent"
                   for row in api.runtime().world.controller_log)
        state = facade.state(token)
        assert state["execution_agents"][0]["status"] == "inactive"
        assert state["execution_agents"][0]["activation_count"] == 1
        assert any(row.get("kind") == "execution_summary" and
                   row.get("thread_id") == ""
                   for row in state["conversation"])
    finally:
        api.shutdown()


def test_secretary_closes_task_reference_claim_review_and_worker_code_action():
    task_id = "task_dashboard_closure"
    source_path = "src/dashboard.py"
    model = ScriptedModel([
        # The read-only answer owns the numbered reference set. One invented
        # reference is included deliberately and must be discarded server-side.
        {"route": "grounded_answer",
         "reply": "1. **Dashboard is not implemented**（尚未认领）",
         "references": [{"object_id": task_id}, {"object_id": "task_not_visible"}]},
        {"route": "ordinary", "reply": ""},
        # First compiler pass understands the verb but omits the target.
        {"drafts": [{"action_type": "pick_task", "params": {},
                     "rationale": "Victor asked to claim the first listed task."}]},
        # The bounded claim resolver follows the verified structured reference.
        {"resolution_basis": "conversation_reference", "task_id": task_id,
         "rationale": "The verified first reference is this claimable task.",
         "reply": "我已把“第一个”解析为 **Dashboard is not implemented**，等待你确认认领。",
         "clarification": ""},
        {"route": "ordinary", "reply": ""},
        {"tool": "read_object", "args": {"object_id": task_id},
         "thought": "Read the exact claimed task and acceptance context."},
        {"tool": "list_repo", "args": {},
         "thought": "Find the visible product source files."},
        {"tool": "read_repo", "args": {"path": source_path},
         "thought": "Inspect the dashboard implementation itself."},
        {"reply": (
            "## 任务理解\n"
            "补齐 `src/dashboard.py` 的实时告警输出。\n\n"
            "## 已有实现\n"
            "- 当前只返回静态空列表。\n\n"
            "## 需要补充\n"
            "- 接入违规事件数据。\n"
            "- 增加空数据与单条告警测试。\n\n"
            "```python\n"
            "def alerts(events):\n"
            "    return [event for event in events if event.get('violation')]\n"
            "```\n"
            "Worker 将修改该文件并留下可审计动作证据。"
         ),
         "references": [{"object_id": task_id}],
         "execution_plan": {
             "title": "Implement dashboard alerts",
             "goal": "Implement the claimed dashboard task in src/dashboard.py.",
             "completion_criteria": "A gateway-backed repository edit exists and the worker reports the exact file.",
             "workers": [{"name": "Dashboard Walker", "role": "implementation worker",
                          "assignment": "Modify src/dashboard.py for task_dashboard_closure and report the result."}],
         }},
        {"action": {"action_type": "edit_repo_file",
                    "params": {"file_path": source_path,
                               "edit_goal": "Implement real violation alerts for the claimed dashboard task."}}},
        {"approve": True, "reason": "The exact visible file and claimed task were inspected."},
        {"change_summary": "Implement violation-backed dashboard alerts.",
         "edits": [{
             "search": "def alerts(events):\n    return []",
             "replace": (
                 "def alerts(events):\n"
                 "    return [event for event in events if event.get('violation')]"
             ),
         }],
         "changed_behavior": ["Dashboard alerts now reflect violation events."],
         "related_task_ids": [task_id]},
        {"report": "Updated src/dashboard.py for task_dashboard_closure with gateway evidence."},
        {"complete": True, "reason": "The repository edit action succeeded for the assigned file."},
        {"summary": "Secretary summary: the Dashboard Walker updated src/dashboard.py and the edit is recorded."},
    ])
    api = _api(model)
    facade = LiaisonFacade(api)
    try:
        world = api.runtime().world
        world.tasks[task_id] = Task(
            task_id=task_id, title="Dashboard is not implemented",
            description="Render live traffic violations and cover the data path.",
            owner_id=None, linked_artifacts=["art_dashboard_closure"],
            visibility="team")
        world.product_artifacts["art_dashboard_closure"] = ProductArtifact(
            artifact_id="art_dashboard_closure", artifact_type="repo_file",
            title=source_path, status="active", linked_file_path=source_path,
            content="def alerts(events):\n    return []\n",
            mainline_content="def alerts(events):\n    return []\n",
            linked_task_ids=[task_id])
        api.runtime().refresh_views()
        token = facade.session()["token"]

        assert facade.ask(token, "哪些任务还没有被认领？")["handled"] == "grounded_answer"
        first_answer = next(
            row for row in facade.state(token)["conversation"]
            if row.get("kind") == "grounded_answer")
        assert [row["object_id"] for row in first_answer["references"]] == [task_id]

        accepted = facade.ask(token, "那我认领第一个")
        assert accepted["accepted"] is True
        for _ in range(200):
            drafts = facade.state(token).get("pending_actions") or []
            if drafts:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("the repaired claim draft was not prepared")
        assert drafts[0]["action_type"] == "pick_task"
        assert drafts[0]["params"]["task_id"] == task_id
        repair_payload = json.loads(model.calls[3][1])
        references = repair_payload["recent_dialogue"][-2]["references"]
        assert references[0]["object_id"] == task_id
        assert all(row.get("owner") in (None, "")
                   for row in repair_payload["eligible_tasks"])

        assert facade.confirm(token, drafts[0]["draft_id"])["ok"] is True
        assert world.tasks[task_id].owner_id == "victor"
        focus = facade.state(token)["active_task_focus"]["__main__"]
        assert focus["object_id"] == task_id
        assert focus["claimed_by"] == "victor"

        assert facade.ask(
            token,
            "读一下这个任务和相关代码，准确告诉我要做什么、缺什么，然后交给 Walker Agent 去实现。",
        )["accepted"] is True
        for _ in range(300):
            state = facade.state(token)
            jobs = state.get("execution_jobs") or []
            if jobs:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("the reviewed Worker plan was not prepared")
        report = next(row for row in state["conversation"]
                      if row.get("kind") == "task_review")
        assert "## 需要补充" in report["text"]
        assert "```python" in report["text"]
        assert report["references"][0]["object_id"] == task_id
        assert jobs[0]["status"] == "pending_confirmation"
        assert jobs[0]["workers"][0]["name"] == "Dashboard Walker"
        assert not any(row.get("execution_mode") == "liaison_subagent"
                       for row in world.controller_log)

        assert facade.execution_start(token, jobs[0]["job_id"])["ok"] is True
        final = _wait(api._execution_team, jobs[0]["job_id"])
        assert final["status"] == "completed", final
        assert any(row.get("execution_mode") == "liaison_subagent"
                   for row in world.controller_log)
        assert any(row.get("action_type") == "edit_repo_file"
                   for row in final["timeline"])
        after = facade.state(token)
        assert after["execution_agents"][0]["status"] == "inactive"
        assert after["execution_agents"][0]["name"] == "Dashboard Walker"
        assert any(row.get("kind") == "execution_summary"
                   and "Dashboard Walker" in row.get("text", "")
                   for row in after["conversation"])
    finally:
        api.shutdown()


def test_one_claimable_task_is_compiled_without_an_unnecessary_clarification():
    model = ScriptedModel([
        {"route": "ordinary", "reply": ""},
        {"drafts": [{"action_type": "pick_task", "params": {},
                     "rationale": "Victor asked the secretary to choose one task to claim."}]},
    ])
    api = _api(model)
    facade = LiaisonFacade(api)
    try:
        token = facade.session()["token"]
        world = api.runtime().world
        tasks = list(world.tasks.values())
        assert tasks
        selected = tasks[0]
        for task in tasks:
            task.owner_id = "calvin"
        selected.owner_id = None
        api.runtime().refresh_views()

        assert facade.ask(token, "你来给我认领一个任务吧")["accepted"] is True
        for _ in range(200):
            state = facade.state(token)
            drafts = state.get("pending_actions") or []
            if drafts:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("the unique claim draft was not prepared")

        assert len(drafts) == 1
        assert drafts[0]["action_type"] == "pick_task"
        assert drafts[0]["params"] == {"task_id": selected.task_id}
        assert len(model.calls) == 2
        assert not any(row.get("kind") == "clarification"
                       for row in state["conversation"])
    finally:
        api.shutdown()


def test_secretary_chooses_one_of_multiple_claimable_tasks_when_choice_is_delegated():
    chosen_id = "task_claim_choice_red_light"
    other_id = "task_claim_choice_dashboard"
    model = ScriptedModel([
        {"route": "ordinary", "reply": ""},
        {"drafts": [{"action_type": "pick_task", "params": {},
                     "rationale": "Victor asked the secretary to choose one task."}]},
        {"resolution_basis": "delegated_choice", "task_id": chosen_id,
         "rationale": "The red-light pipeline is the highest-priority eligible task.",
         "reply": "我选择了红灯检测任务，并准备了认领动作，等待你确认。"},
    ])
    api = _api(model)
    facade = LiaisonFacade(api)
    try:
        world = api.runtime().world
        for task in world.tasks.values():
            if task.owner_id is None:
                task.owner_id = "paul"
        world.tasks[chosen_id] = Task(
            task_id=chosen_id, title="Implement pipeline red-light detection",
            description="Implement the missing red-light detector.",
            owner_id=None, priority=9, visibility="team")
        world.tasks[other_id] = Task(
            task_id=other_id, title="Implement traffic dashboard",
            description="Implement the missing dashboard.",
            owner_id=None, priority=3, visibility="team")
        api.runtime().refresh_views()
        token = facade.session()["token"]

        assert facade.ask(token, "你来给我认领一个任务吧")["accepted"] is True
        for _ in range(200):
            state = facade.state(token)
            drafts = state.get("pending_actions") or []
            if drafts:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("the delegated task choice was not prepared")

        assert len(drafts) == 1
        assert drafts[0]["action_type"] == "pick_task"
        assert drafts[0]["params"] == {"task_id": chosen_id}
        assert len(model.calls) == 3
        planner_call = model.calls[2]
        planner_payload = json.loads(planner_call[1])
        assert {row["object_id"] for row in planner_payload["eligible_tasks"]} == {
            chosen_id, other_id,
        }
        assert "task-claim resolver" in planner_call[0]
        assert not any(row.get("kind") == "clarification"
                       for row in state["conversation"])
    finally:
        api.shutdown()


def test_claimed_task_investigation_separates_control_json_from_markdown_report():
    task_id = "task_claimed_dashboard_investigation"
    source_path = "traffic_violation_system/dashboard.py"
    request = (
        "OK既然你都想好了那么现在你去探索一下这个任务让他们给我汇报这个任务需要的内容是什么"
        "然后我需要做什么现在有什么难点在哪里"
    )
    report = (
        "## 任务需要什么\n"
        "完成实时告警、持久化记录以及上传或摄像头输入。\n\n"
        "## 当前实现\n"
        f"`{source_path}` 的 `create_app` 仍直接抛出 `NotImplementedError`。\n\n"
        "## 当前难点\n"
        "需要把上传、实时 feed、存储和报告导出接到同一条可测试的数据流。\n\n"
        "## 你现在需要做什么\n"
        "确认是否启动下面的调查 Worker；目前还没有执行代码修改或测试。"
    )
    model = ScriptedModel([
        {"route": "ordinary", "reply": ""},
        {"kind": "tool", "tool": "read_object", "args": {"object_id": task_id},
         "thought": "Read the claimed task requirements."},
        {"kind": "tool", "tool": "read_repo", "args": {"path": source_path},
         "thought": "Inspect the linked dashboard implementation."},
        {"kind": "synthesize", "thought": "The task and implementation are grounded."},
        {"evidence_status": "needs_source",
         "missing_fact": "The existing storage interface has not been inspected.",
         "next_tool": "read_repo",
         "next_args": {"path": "traffic_violation_system/storage.py"},
         "mode": "report", "references": []},
        {"kind": "synthesize", "thought": "Audit the remaining report dependency."},
        {"evidence_status": "needs_source",
         "missing_fact": "The existing report exporter interface has not been inspected.",
         "next_tool": "read_repo",
         "next_args": {"path": "traffic_violation_system/reports.py"},
         "mode": "report", "references": []},
        {"kind": "synthesize", "thought": "All explicitly required interfaces are grounded."},
        # The model may omit the final control reference even after reading the
        # task.  The exact read_object ledger still anchors the report and job.
        {"evidence_status": "sufficient", "mode": "execution_plan", "references": [],
         "execution_plan": {
             "title": "Investigate the claimed dashboard task",
             "goal": "Report exact implementation gaps and a bounded delivery path.",
             "completion_criteria": "A source-grounded gap and test report is returned.",
             "workers": [
                 {"name": "Dashboard Analyst", "role": "code investigator",
                  "assignment": "Inspect dashboard data flow and list missing implementation."},
                 {"name": "Dashboard Test Analyst", "role": "test investigator",
                  "assignment": "Inspect expected behavior and propose concrete verification."},
             ],
         }},
    ], text_responses=[report])
    api = _api(model)
    facade = LiaisonFacade(api)
    try:
        world = api.runtime().world
        world.tasks[task_id] = Task(
            task_id=task_id, title="Implement dashboard",
            description="Provide live violations, uploads, storage, and report export.",
            owner_id="victor", linked_artifacts=["art_claimed_dashboard"],
            visibility="team")
        world.product_artifacts["art_claimed_dashboard"] = ProductArtifact(
            artifact_id="art_claimed_dashboard", artifact_type="repo_file",
            title=source_path, status="active", linked_file_path=source_path,
            content="def create_app():\n    raise NotImplementedError\n",
            mainline_content="def create_app():\n    raise NotImplementedError\n",
            linked_task_ids=[task_id])
        api.runtime().refresh_views()
        token = facade.session()["token"]
        secretary = api._agent("victor")
        secretary._active_task_focus["__main__"] = {
            "object_id": task_id, "kind": "task", "label": "Implement dashboard",
            "status": "open", "owner": "victor", "claimed_by": "victor",
        }

        assert facade.ask(token, request)["accepted"] is True
        for _ in range(300):
            state = facade.state(token)
            jobs = state.get("execution_jobs") or []
            if jobs:
                break
            time.sleep(0.01)
        else:
            raise AssertionError(f"the investigation plan was not prepared: {state!r}")

        task_report = next(row for row in state["conversation"]
                           if row.get("kind") == "task_review")
        assert task_report["text"] == report
        assert task_report["references"][0]["object_id"] == task_id
        report_record = next(row for row in secretary.transcript
                             if row.kind == "task_review")
        assert [row["args"]["path"] for row in report_record.evidence_records
                if row["tool"] == "read_repo"] == [
            source_path,
            "traffic_violation_system/storage.py",
            "traffic_violation_system/reports.py",
        ]
        assert jobs[0]["status"] == "pending_confirmation"
        assert [row["name"] for row in jobs[0]["workers"]] == [
            "Dashboard Analyst", "Dashboard Test Analyst",
        ]
        assert not any(row.get("execution_mode") == "liaison_subagent"
                       for row in world.controller_log)
        assert len(model.text_calls) == 1
        assert "Do not output JSON" in model.text_calls[0][0]
        control_call = next(call for call in model.calls
                            if "private Human--Organization Liaison" in call[0])
        assert "Never place a final report, Markdown, code block" in control_call[0]
    finally:
        api.shutdown()


def test_grounded_report_is_preserved_when_execution_plan_is_invalid():
    api = _api()
    try:
        api.claim("victor")
        secretary = api._agent("victor")
        secretary._prepare_execution_plan_response(
            {
                "title": "Incomplete investigation plan",
                "goal": "Inspect the task.",
                "completion_criteria": "Return a grounded report.",
                "workers": [{"name": "Analyst", "role": "investigator"}],
            },
            "Explore this task and report back.",
            {"reply": "## 已确认的任务现状\n报告内容仍应对 Victor 可见。"},
        )

        reports = [row for row in secretary.transcript if row.kind == "task_review"]
        errors = [row for row in secretary.transcript if row.kind == "clarification"]
        assert reports[-1].text.startswith("## 已确认的任务现状")
        assert "worker_1_needs_assignment" in errors[-1].text
    finally:
        api.shutdown()


def test_secretary_repair_cannot_replace_an_existing_target_or_drop_actions():
    compiler_error = "The structured plan is incomplete; keep the original targets."
    model = ScriptedModel([
        {"drafts": [{
            "action_type": "assign_task_owner",
            "params": {"task_id": "task_different", "owner_id": "calvin"},
        }, {
            "action_type": "send_message",
            "params": {"channel_id": "channel_keep", "text": "status"},
        }]},
        {"drafts": [{
            "action_type": "assign_task_owner",
            "params": {"task_id": "task_keep", "owner_id": "calvin"},
        }]},
    ])
    api = _api(model)
    try:
        api.claim("victor")
        secretary = api._agent("victor")
        original = [{
                "action_type": "assign_task_owner",
                "params": {"task_id": "task_keep", "owner_id": ""},
            }, {
                "action_type": "send_message",
                "params": {"channel_id": "channel_keep", "text": ""},
            }]
        created, error, _decision = secretary._repair_compiled_drafts(
            original,
            compiler_error,
            "Assign that task and send the requested message.",
        )
        assert created == []
        assert error == compiler_error

        created, error, _decision = secretary._repair_compiled_drafts(
            original,
            compiler_error,
            "Assign that task and send the requested message.",
        )
        assert created == []
        assert error == compiler_error
    finally:
        api.shutdown()


def test_secretary_can_choose_delegated_batch_assignments_from_visible_workload():
    assignments = {
        "task_batch_dashboard": "victor",
        "task_batch_alerts": "scarlett",
        "task_batch_pipeline": "sean",
    }
    model = ScriptedModel([
        {"route": "ordinary", "reply": ""},
        {"drafts": [{"action_type": "assign_task_owner", "params": {}}]},
        {"explicit_delegation": True, "scope": "all_eligible",
         "assignments": [
             {"task_id": task_id, "owner_id": owner_id,
              "rationale": "Explicitly delegated allocation grounded in role and current workload."}
             for task_id, owner_id in assignments.items()
         ],
         "reply": "我按角色和当前负载准备了三个任务分配，其中给 Victor 留了一个。"},
    ])
    api = _api(model)
    facade = LiaisonFacade(api)
    try:
        world = api.runtime().world
        for task in world.tasks.values():
            if task.owner_id is None:
                task.owner_id = "paul"
        for task_id in assignments:
            world.tasks[task_id] = Task(
                task_id=task_id, title=f"Implement {task_id}",
                description="Visible unassigned implementation work.",
                owner_id=None, visibility="team")
        api.runtime().refresh_views()
        token = facade.session()["token"]

        result = facade.ask(
            token, "把这些任务分配下去，给我自己留一个，谁做什么你来决定。")
        assert result["accepted"] is True
        for _ in range(200):
            state = facade.state(token)
            drafts = state.get("pending_actions") or []
            if len(drafts) == 3:
                break
            time.sleep(0.01)
        else:
            raise AssertionError(f"delegated batch assignments were not prepared: {state!r}")

        compiler_call = next(call for call in model.calls
                             if "private Human--Organization Liaison" in call[0])
        compiler_payload = json.loads(compiler_call[1])
        candidates = compiler_payload["context"]["action_candidates"]["assign_task_owner"]
        assert {row["object_id"] for row in candidates["tasks"]} == set(assignments)
        assert {row["agent_id"] for row in candidates["members"]}.issuperset(
            {"victor", "scarlett", "sean"})
        assert "explicitly delegates that choice" in compiler_call[0]
        allocation_call = next(call for call in model.calls
                               if "dedicated delegated task-allocation" in call[0])
        allocation_payload = json.loads(allocation_call[1])
        assert allocation_payload["human_request"] == (
            "把这些任务分配下去，给我自己留一个，谁做什么你来决定。")
        assert len(allocation_payload["eligible_candidates"]["tasks"]) == 3
        assert all(world.tasks[task_id].owner_id is None for task_id in assignments)

        actual = {
            draft["params"]["task_id"]: draft["params"]["owner_id"]
            for draft in drafts
        }
        assert actual == assignments
        for draft in drafts:
            assert facade.confirm(token, draft["draft_id"])["ok"] is True
        assert {task_id: world.tasks[task_id].owner_id for task_id in assignments} == assignments
    finally:
        api.shutdown()


def test_assignment_planner_fails_closed_without_explicit_model_authorization():
    model = ScriptedModel([{
        "explicit_delegation": False,
        "scope": "unclear",
        "assignments": [],
        "clarification": "请明确指定负责人，或者明确授权我根据角色和负载选择。",
    }])
    api = _api(model)
    try:
        api.claim("victor")
        secretary = api._agent("victor")
        created, error, _decision = secretary._repair_compiled_drafts(
            [{"action_type": "assign_task_owner", "params": {}}],
            "missing task and owner",
            "把这个任务分配下去。",
        )
        assert created == []
        assert error == "请明确指定负责人，或者明确授权我根据角色和负载选择。"
        assert not secretary.drafts
    finally:
        api.shutdown()


def test_repeated_incomplete_batch_errors_are_returned_once():
    api = _api(ScriptedModel([]))
    try:
        api.claim("victor")
        secretary = api._agent("victor")
        created, error = secretary._prepare_compiled_drafts(
            [{"action_type": "assign_task_owner", "params": {}} for _ in range(3)],
            request="把任务分配下去。",
        )
        assert created == []
        assert error.count("我理解你要执行") == 1
    finally:
        api.shutdown()


def test_world_failure_is_not_misreported_as_a_submitted_working_agent_draft():
    api = _api()
    try:
        victor = api.claim("victor")
        rt = api.runtime()
        execution = rt.world._loop["execution"]
        branch = execution.execute(
            "victor", ActionCandidate(action_type="create_branch",
                                    parameters={"linked_task": "task_00"}), rt.world,
        ).created_objects[0]
        worker = WorkingAgentSession(rt, "victor")
        draft = worker.add_draft("run_ci", {"branch_id": branch})
        result = worker.confirm(draft.draft_id)
        assert result["ok"] is False
        assert worker.drafts[draft.draft_id].status == "failed"
        assert "did not complete" in worker.transcript[-1].text
        assert "Executed" not in worker.transcript[-1].text
        # Keep the token live through assertion to make the seat provenance explicit.
        assert victor["token"]
    finally:
        api.shutdown()


def test_pending_plan_does_not_create_a_persistent_worker_until_started():
    api = _api()
    try:
        victor = api.claim("victor")
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: ScriptedModel([]))
        api._execution_team = team
        prepared = team.prepare("victor", {
            "title": "Unconfirmed plan", "workers": [{
                "name": "Provisional", "role": "reviewer",
                "assignment": "Review only after Victor starts the job.",
            }],
        })
        assert team.agents_for("victor") == []
        assert team.cancel("victor", prepared["job"]["job_id"])["ok"] is True
        assert team.agents_for("victor") == []
        assert victor["token"]
    finally:
        api.shutdown()


def test_release_drops_late_completion_from_the_released_seat():
    api = _api()
    try:
        victor = api.claim("victor")
        active = api._agent("victor")
        before_messages = len(active.transcript)
        before_logs = len(api.get_hci_logs())
        assert api.release(victor["token"])["released"] == "victor"
        api._execution_complete({
            "seat_id": "victor", "job_id": "exec_late", "status": "cancelled",
            "final_report": "This must not be appended after seat release.",
            "thread_id": "", "timeline": [],
        }, expected_epoch=api._runtime_epoch)
        assert len(active.transcript) == before_messages
        assert len(api.get_hci_logs()) == before_logs
    finally:
        api.shutdown()


def test_duplicate_reads_do_not_consume_the_bounded_productive_step_budget():
    """A worker can recover from repeated reads instead of the old six-step stop."""
    api = _api()
    try:
        victor = api.claim("victor")
        repeated_read = {"tool": "search_org", "args": {"query": "current status"}}
        model = ScriptedModel([
            repeated_read,
            *[repeated_read for _ in range(7)],
            {"report": "The one distinct observation was reviewed."},
            {"complete": True, "reason": "The retained evidence supports this review."},
            {"summary": "Long-horizon worker settled with one distinct read."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Recover from duplicate reads", "goal": "Review current status.",
            "completion_criteria": "Return an evidence-grounded review.",
            "workers": [{"name": "Reader", "role": "reviewer",
                         "assignment": "Inspect the visible organization and report."}],
        })
        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        settled = _wait(team, prepared["job"]["job_id"])

        assert settled["status"] == "completed"
        worker_calls = [call for call in model.calls
                        if "independent execution worker" in call[0]]
        assert len(worker_calls) == 9, "duplicates must not trigger the former six-step cutoff"
        assert sum(event.get("status") == "deduplicated" for event in settled["timeline"]) == 7
        assert not any("step limit" in event.get("summary", "")
                       for event in settled["timeline"])
        assert victor["token"]
    finally:
        api.shutdown()


def test_duplicate_read_feedback_is_an_explicit_next_turn_constraint():
    api = _api()
    try:
        api.claim("victor")
        repeated_read = {"tool": "search_org", "args": {"query": "current status"}}
        model = ScriptedModel([
            repeated_read,
            repeated_read,
            {"report": "The current status was reviewed once."},
            {"complete": True, "reason": "The retained evidence supports the report."},
            {"summary": "Duplicate read recovery completed."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Reject duplicate read", "goal": "Review current status once.",
            "completion_criteria": "Return one evidence-grounded review.",
            "workers": [{"name": "Reader", "role": "reviewer",
                         "assignment": "Read the current status once and report."}],
        })
        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        settled = _wait(team, prepared["job"]["job_id"])

        worker_payloads = [json.loads(user) for system, user, _schema in model.calls
                           if "independent execution worker" in system]
        feedback = worker_payloads[2]["executor_feedback"]
        assert feedback["duplicate_reads_rejected"] == [repeated_read]
        assert "Do not request it again" in feedback["instruction"]
        duplicate = next(row for row in settled["timeline"]
                         if row.get("status") == "deduplicated")
        assert duplicate["tool"] == "search_org"
        assert duplicate["args"] == {"query": "current status"}
    finally:
        api.shutdown()


def test_long_horizon_context_compacts_history_but_retains_old_evidence_sources():
    """Older observations remain inspectable after recent history rolls over."""
    api = _api()
    try:
        victor = api.claim("victor")
        reads = [
            {"tool": "search_org", "args": {"query": query}}
            for query in ["old-evidence-token", *[f"follow-up-{index}" for index in range(11)]]
        ]
        model = ScriptedModel([
            *reads,
            {"report": "Twelve visible observations were reviewed."},
            {"complete": True, "reason": "The source ledger supports the review."},
            {"summary": "Long-horizon evidence review settled."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Long evidence review", "goal": "Review all visible organization evidence.",
            "completion_criteria": "Report from retained live evidence.",
            "workers": [{"name": "Investigator", "role": "reviewer",
                         "assignment": "Read the relevant visible evidence and report."}],
        })
        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        settled = _wait(team, prepared["job"]["job_id"])

        assert settled["status"] == "completed", "12 productive reads must not be cut off at six"
        worker_payloads = [json.loads(user) for system, user, _schema in model.calls
                           if "independent execution worker" in system]
        report_payload = worker_payloads[-1]
        assert report_payload["task_anchor"]["goal"] == "Review all visible organization evidence."
        assert any(entry["args"].get("query") == "old-evidence-token"
                   for entry in report_payload["evidence_ledger"])
        assert any(row.get("tool") == "search_org"
                   for row in report_payload["history_summary"])
        assert len(report_payload["recent_worker_history"]) <= team.MAX_HISTORY_ITEMS
        assert not any("step limit" in event.get("summary", "")
                       for event in settled["timeline"])
        assert victor["token"]
    finally:
        api.shutdown()


def test_next_worker_turn_keeps_the_complete_bounded_source_read(monkeypatch):
    """A worker must see the file body it just read, not a 1,200-char prefix."""
    api = _api()
    try:
        victor = api.claim("victor")
        source = "x" * 5900 + "\nTAIL_IMPLEMENTATION_MARKER"
        monkeypatch.setattr(SeatTools, "read_repo", lambda self, path: source)
        model = ScriptedModel([
            {"tool": "read_repo", "args": {"path": "large_module.py"}},
            {"report": "The complete source was reviewed."},
            {"complete": True, "reason": "The source evidence supports the report."},
            {"summary": "Large source review completed."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Read a full module", "goal": "Inspect the implementation tail.",
            "completion_criteria": "Report after seeing the complete bounded source.",
            "workers": [{"name": "Reader", "role": "reviewer",
                         "assignment": "Read large_module.py through its final marker."}],
        })
        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        settled = _wait(team, prepared["job"]["job_id"])

        assert settled["status"] == "completed"
        worker_payloads = [json.loads(user) for system, user, _schema in model.calls
                           if "independent execution worker" in system]
        assert "TAIL_IMPLEMENTATION_MARKER" in (
            worker_payloads[1]["evidence_ledger"][0]["source_excerpt"]
        )
        assert victor["token"]
    finally:
        api.shutdown()


def test_evidence_context_keeps_recent_source_full_and_compacts_older_bodies():
    api = _api()
    try:
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: None)
        team.MAX_ACTIVE_EVIDENCE_CHARS = 150
        worker = {"evidence_ledger": [
            {"evidence_id": f"e{index}", "kind": "read", "tool": "read_repo",
             "args": {"path": f"module_{index}.py"}, "source_text": char * 100,
             "world_tick": 1}
            for index, char in enumerate(("a", "b", "c"), 1)
        ]}

        context = team._evidence_context(worker)

        assert [row["compacted"] for row in context] == [True, True, False]
        assert context[-1]["source_excerpt"] == "c" * 100
        assert all(row["args"]["path"].startswith("module_") for row in context)
    finally:
        api.shutdown()


def test_rereading_compacted_evidence_promotes_it_back_to_active_context(monkeypatch):
    api = _api()
    try:
        victor = api.claim("victor")
        monkeypatch.setattr(ExecutionTeam, "MAX_ACTIVE_EVIDENCE_CHARS", 200)
        monkeypatch.setattr(
            SeatTools, "search_org",
            lambda self, query: f"{query}:" + (query[0] * 240),
        )
        model = ScriptedModel([
            {"tool": "search_org", "args": {"query": "old"}},
            {"tool": "search_org", "args": {"query": "new"}},
            {"tool": "search_org", "args": {"query": "old"}},
            {"report": "The old source was promoted and reviewed again."},
            {"complete": True, "reason": "The refreshed evidence is active."},
            {"summary": "Evidence refresh completed."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Refresh compacted evidence", "goal": "Revisit an older source.",
            "completion_criteria": "Promote the older source and report.",
            "workers": [{"name": "Reader", "role": "reviewer",
                         "assignment": "Read old, read new, then revisit old."}],
        })
        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        settled = _wait(team, prepared["job"]["job_id"])

        assert settled["status"] == "completed"
        assert any(event.get("status") == "refreshed" for event in settled["timeline"])
        assert sum(row["args"].get("query") == "old"
                   for row in settled["evidence_records"]) == 2
        assert victor["token"]
    finally:
        api.shutdown()


def test_productive_step_ceiling_reserves_final_synthesis_and_completion_audit():
    """Using all work steps still leaves one factual conclusion and its audit."""
    api = _api()
    try:
        victor = api.claim("victor")
        reads = [
            {"tool": "search_org", "args": {"query": f"source-{index}"}}
            for index in range(ExecutionTeam.MAX_STEPS)
        ]
        model = ScriptedModel([
            *reads,
            {"summary": "The bounded investigation found no verified implementation result."},
            {"complete": False,
             "reason": "The evidence supports a report, but not implementation completion."},
            {"summary": "Secretary: the investigation settled as blocked, with evidence."},
        ])
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Exhaust the work budget", "goal": "Investigate before implementing.",
            "completion_criteria": "Implementation evidence must exist.",
            "workers": [{"name": "Investigator", "role": "reviewer",
                         "assignment": "Inspect the visible evidence and report truthfully."}],
        })
        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        settled = _wait(team, prepared["job"]["job_id"])

        assert settled["status"] == "blocked"
        assert settled["workers"][0]["report"] == (
            "The bounded investigation found no verified implementation result.")
        assert any(event.get("kind") == "worker_synthesis"
                   for event in settled["timeline"])
        assert not any("without verified completion" in event.get("summary", "")
                       for event in settled["timeline"])
        assert victor["token"]
    finally:
        api.shutdown()


def test_investigation_worker_receives_exact_tool_protocol_and_returns_source_report():
    task_id = "task_worker_dashboard_analysis"
    source_path = "traffic_violation_system/dashboard.py"
    model = ScriptedModel([
        {"kind": "tool", "tool": "read_object",
         "args": {"object_id": task_id}, "action": {}, "report": ""},
        {"kind": "tool", "tool": "read_repo",
         "args": {"path": source_path}, "action": {}, "report": ""},
        {"kind": "report", "tool": "", "args": {}, "action": {},
         "report": "The task requires a dashboard, and create_app is still unimplemented."},
        {"complete": True, "reason": "The report cites the task and repository evidence."},
        {"summary": "Secretary: the dashboard investigation completed from source evidence."},
    ])
    api = _api(model)
    try:
        world = api.runtime().world
        world.tasks[task_id] = Task(
            task_id=task_id, title="Implement traffic dashboard",
            description="Provide the traffic violation dashboard.",
            owner_id="victor", linked_artifacts=["art_worker_dashboard"],
            visibility="team")
        world.product_artifacts["art_worker_dashboard"] = ProductArtifact(
            artifact_id="art_worker_dashboard", artifact_type="repo_file",
            title=source_path, status="active", linked_file_path=source_path,
            content="def create_app():\n    raise NotImplementedError\n",
            mainline_content="def create_app():\n    raise NotImplementedError\n",
            linked_task_ids=[task_id])
        api.runtime().refresh_views()
        victor = api.claim("victor")
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Dashboard requirements analysis",
            "references": [{"object_id": task_id}],
            "goal": "Inspect the dashboard task and implementation gap.",
            "completion_criteria": "Return a source-grounded requirements report.",
            "workers": [{
                "name": "Requirements Analyst", "role": "code investigator",
                "assignment": (
                    f"Read {task_id} and {source_path}; report the exact missing behavior."),
            }],
        }, request="Explore this task and report what remains to do.")

        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        settled = _wait(team, prepared["job"]["job_id"])

        assert settled["status"] == "completed", settled
        assert [row.get("tool") for row in settled["timeline"]
                if row.get("kind") == "tool"] == ["read_object", "read_repo"]
        assert settled["references"] == [{"object_id": task_id}]
        source = next(row for row in settled["evidence_records"]
                      if row["tool"] == "read_repo")
        assert source["args"] == {"path": source_path}
        assert "NotImplementedError" in source["source_text"]
        timeline_source = next(row for row in settled["timeline"]
                               if row.get("tool") == "read_repo")
        assert source_path in timeline_source["summary"]
        assert timeline_source["args"] == {"path": source_path}
        first_system, first_payload, first_schema = model.calls[0]
        assert first_schema == WORKER_DECISION_SHAPE
        assert "Return exactly one kind per turn" in first_system
        context = json.loads(first_payload)
        tools = {row["name"]: row for row in context["observation_tools"]}
        assert tools["read_object"]["args"] == ["object_id"]
        assert tools["read_repo"]["args"] == ["path"]
        assert any(row["action_type"] == "edit_repo_file"
                   for row in context["action_catalog"])
        assert victor["token"]
    finally:
        api.shutdown()


def test_confirmed_start_claims_task_then_delivers_factual_periodic_progress():
    """The start boundary claims Victor's task before a worker is activated.

    A deliberately blocked first worker turn keeps the job live past one real
    wall-clock interval.  This verifies an emitted report is a running-job
    snapshot with actual worker/evidence counts, rather than a plan string or
    an end-of-job secretary summary.
    """
    task_id = "task_periodic_progress"
    model = BlockingModel(
        {"kind": "tool", "tool": "search_org", "args": {"query": "progress source"}},
        [
            {"kind": "report", "report": "One visible source was inspected."},
            {"complete": True, "reason": "The retained source supports this review."},
            {"summary": "Secretary: task review completed after the progress update."},
        ],
    )
    api = _api(model)
    reports = []
    try:
        world = api.runtime().world
        world.tasks[task_id] = Task(
            task_id=task_id, title="Periodic progress task",
            description="An unclaimed visible task for the execution-start boundary.",
            owner_id=None, visibility="team")
        api.runtime().refresh_views()
        api.claim("victor")
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model,
                             on_progress=lambda job: reports.append(job))
        prepared = team.prepare("victor", {
            "title": "Claim and review", "task_id": task_id, "claim_task": True,
            "progress_interval_seconds": 1,
            "goal": "Claim the selected task and inspect one visible source.",
            "workers": [{"name": "Reviewer", "role": "reviewer",
                         "assignment": "Inspect current evidence and report it."}],
        })
        assert prepared["ok"] is True
        assert team.agents_for("victor") == []

        started = team.start("victor", prepared["job"]["job_id"])
        assert started["ok"] is True
        assert world.tasks[task_id].owner_id == "victor"
        assert model.entered.wait(0.5)
        assert json.loads(model.calls[0][1])["task_anchor"]["task_id"] == task_id

        deadline = time.time() + 1.5
        while not reports and time.time() < deadline:
            time.sleep(0.02)
        assert reports, "a running worker must deliver its chosen periodic update"
        progress = reports[-1]
        event = next(row for row in reversed(progress["timeline"])
                     if row.get("kind") == "progress_report")
        assert progress["status"] == "running"
        assert progress["progress_summary"] == event["summary"]
        assert event["worker_status_counts"] == {"running": 1}
        assert event["evidence_reads"] == 0
        assert event["actions_succeeded"] == 0
        assert any(row.get("kind") == "task_claim" and row.get("ok") is True
                   for row in progress["timeline"])

        model.release.set()
        settled = _wait(team, prepared["job"]["job_id"])
        assert settled["status"] == "completed"
        assert settled["task_id"] == task_id
        assert settled["claim_task"] is True
        assert settled["progress_interval_seconds"] == 1
    finally:
        api.shutdown()


def test_task_claim_is_revalidated_before_any_worker_is_activated():
    """A stale prepared plan may not create a ghost execution worker."""
    task_id = "task_claim_changed_before_start"
    api = _api(ScriptedModel([]))
    try:
        world = api.runtime().world
        world.tasks[task_id] = Task(
            task_id=task_id, title="Claim race", description="Visible work.",
            owner_id=None, visibility="team")
        api.runtime().refresh_views()
        api.claim("victor")
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: ScriptedModel([]))
        prepared = team.prepare("victor", {
            "task_id": task_id, "claim_task": True,
            "workers": [{"assignment": "Only run after a valid task claim."}],
        })
        assert prepared["ok"] is True
        # A different member takes the task after the inert plan was prepared.
        world.tasks[task_id].owner_id = "scarlett"

        started = team.start("victor", prepared["job"]["job_id"])
        assert started["error"] == (
            f"execution_task_claim_not_available:task_already_owned:{task_id}")
        assert team.public(prepared["job"]["job_id"])["status"] == "pending_confirmation"
        assert team.agents_for("victor") == []
    finally:
        api.shutdown()


def test_later_workers_are_bound_to_the_jobs_first_delivery_branch(monkeypatch):
    model = ScriptedModel([
        {"kind": "action", "action": {
            "action_type": "edit_repo_file",
            "params": {"file_path": "traffic_violation_system/dashboard.py"},
        }},
        {"approve": True, "reason": "The first project edit is grounded."},
        {"kind": "report", "report": "Dashboard edit recorded."},
        {"complete": True, "reason": "The assigned edit was recorded."},
        {"kind": "action", "action": {
            "action_type": "edit_repo_file",
            "params": {"file_path": "traffic_violation_system/rules.py"},
        }},
        {"approve": True, "reason": "The second project edit is grounded."},
        {"kind": "report", "report": "Rules edit recorded."},
        {"complete": True, "reason": "The assigned edit was recorded."},
        {"summary": "Both Workers contributed to one delivery branch."},
    ])
    api = _api(model)
    submitted = []

    def fake_submit(_runtime, _seat_id, action_type, params, **_kwargs):
        submitted.append((action_type, dict(params)))
        return types.SimpleNamespace(
            success=True, action_id=f"action_{len(submitted)}",
            failure_reason="", created_objects=[], modified_objects=[],
            state_delta={"patch_id": f"patch_{len(submitted)}",
                         "branch_id": "branch_project"},
        )

    try:
        api.claim("victor")
        monkeypatch.setattr(gateway, "submit", fake_submit)
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "One project delivery",
            "workers": [
                {"name": "Dashboard Worker", "role": "implementation",
                 "assignment": "Implement the dashboard."},
                {"name": "Rules Worker", "role": "implementation",
                 "assignment": "Implement the rule engine."},
            ],
        })

        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        settled = _wait(team, prepared["job"]["job_id"])

        assert settled["status"] == "completed"
        assert settled["delivery_branch_id"] == "branch_project"
        assert "branch_id" not in submitted[0][1]
        assert submitted[1][1]["branch_id"] == "branch_project"
    finally:
        api.shutdown()


def test_one_start_claims_victors_task_and_assigns_the_remaining_tasks():
    model = ScriptedModel([
        {"kind": "report", "report": "The confirmed allocation is visible."},
        {"complete": True, "reason": "The task state proves the allocation."},
        {"summary": "Victor's task was claimed and the other task was assigned."},
    ])
    api = _api(model)
    try:
        world = api.runtime().world
        world.tasks["task_for_victor"] = Task(
            task_id="task_for_victor", title="Victor task",
            description="Claim this task.", owner_id=None, visibility="team")
        world.tasks["task_for_sean"] = Task(
            task_id="task_for_sean", title="Sean task",
            description="Assign this task.", owner_id=None, visibility="team")
        api.runtime().refresh_views()
        api.claim("victor")
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Claim and allocate",
            "task_id": "task_for_victor", "claim_task": True,
            "task_allocations": [{
                "task_id": "task_for_sean", "owner_id": "sean",
                "rationale": "Sean is available.",
            }],
            "workers": [{"name": "Delivery Worker", "role": "implementation",
                         "assignment": "Verify the confirmed task allocation."}],
        })

        assert prepared["ok"] is True
        assert world.tasks["task_for_victor"].owner_id is None
        assert world.tasks["task_for_sean"].owner_id is None
        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        settled = _wait(team, prepared["job"]["job_id"])

        assert settled["status"] == "completed"
        assert world.tasks["task_for_victor"].owner_id == "victor"
        assert world.tasks["task_for_sean"].owner_id == "sean"
        assert any(row.get("kind") == "task_allocation" and row.get("ok") is True
                   for row in settled["timeline"])
    finally:
        api.shutdown()


def test_worker_accepts_an_asynchronously_merged_pr_as_a_satisfied_postcondition(monkeypatch):
    model = ScriptedModel([
        {"kind": "action", "action": {
            "action_type": "run_ci", "params": {"pr_id": "pr_finished"},
        }},
        {"approve": True, "reason": "Refresh the delivery CI state."},
        {"kind": "report", "report": (
            "PR pr_finished was independently reviewed, passed CI, and merged."
        )},
        {"complete": True, "reason": "The merged PR satisfies the delivery criteria."},
        {"summary": "Secretary: PR pr_finished was reviewed, green, and merged."},
    ])
    api = _api(model)

    def already_terminal(_runtime, _seat_id, _action_type, _params, **_kwargs):
        return types.SimpleNamespace(
            success=False, action_id="action_terminal", failure_reason="pr_not_open",
            created_objects=[], modified_objects=[], state_delta={},
        )

    try:
        api.claim("victor")
        api.runtime().world.repo_system.repo.pull_requests["pr_finished"] = types.SimpleNamespace(
            pr_id="pr_finished", status="merged", ci_passed=True,
        )
        monkeypatch.setattr(gateway, "submit", already_terminal)
        team = ExecutionTeam(api.runtime(), llm_provider=lambda: model)
        prepared = team.prepare("victor", {
            "title": "Finish an asynchronously reviewed delivery",
            "completion_criteria": "The pull request is green and merged.",
            "workers": [{
                "name": "Delivery Worker", "role": "implementation",
                "assignment": "Verify CI and finish the already reviewed pull request.",
            }],
        })

        assert team.start("victor", prepared["job"]["job_id"])["ok"] is True
        settled = _wait(team, prepared["job"]["job_id"])

        assert settled["status"] == "completed"
        recovered = next(row for row in settled["timeline"]
                         if row.get("action_type") == "run_ci")
        assert recovered["ok"] is True
        assert recovered["recovered_from"] == "already_merged"
        assert recovered["modified"] == ["pr_finished"]
    finally:
        api.shutdown()


if __name__ == "__main__":
    test_execution_workers_are_independent_model_contexts_and_gateway_backed()
    test_execution_agent_persists_inactive_and_is_reused_with_prior_context()
    test_active_execution_agent_cannot_be_concurrently_reused()
    test_world_reset_fences_old_compiler_sink_and_completion_callback()
    test_cancelled_late_worker_response_is_fenced_before_gateway()
    test_execution_start_cancel_are_token_scoped_and_pending_cancel_is_real()
    test_report_without_evidence_can_only_leave_an_implementation_worker_blocked()
    test_releasing_a_human_seat_cancels_that_seats_execution_authority()
    test_liaison_natural_language_path_prepares_then_runs_a_real_job()
    test_world_failure_is_not_misreported_as_a_submitted_working_agent_draft()
    test_pending_plan_does_not_create_a_persistent_worker_until_started()
    test_release_drops_late_completion_from_the_released_seat()
    test_duplicate_reads_do_not_consume_the_bounded_productive_step_budget()
    test_long_horizon_context_compacts_history_but_retains_old_evidence_sources()
    test_productive_step_ceiling_reserves_final_synthesis_and_completion_audit()
    test_confirmed_start_claims_task_then_delivers_factual_periodic_progress()
    test_task_claim_is_revalidated_before_any_worker_is_activated()
    print("All human execution-team tests passed!")
