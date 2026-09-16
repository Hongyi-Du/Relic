"""Regression for a delegated task choice inside a compound secretary request."""
import json
import os
import time

import pytest

from environments.org_env.backend.entities import Task
from environments.org_env.human.api import HumanApi
from environments.org_env.human.liaison import LiaisonFacade
from environments.org_env.runtime_adapter.live import OrgInspectorSession


REQUEST = "现在给我认领一个任务然后你把任务布置下去让他们开始干每5分钟汇报一次进度"
BROKEN_BATCH = [
    {"action_type": "pick_task", "params": {}},
    {"action_type": "assign_task_owner", "params": {}},
    {"action_type": "send_message", "params": {}},
]


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, system, user, schema, **kwargs):
        self.calls.append((system, json.loads(user), schema))
        assert self.responses, "unexpected extra model call"
        return self.responses.pop(0)


def wait_idle(secretary):
    deadline = time.monotonic() + 4
    while secretary.state()["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not secretary.state()["busy"]


def fixture(model):
    session = OrgInspectorSession(seed=42)
    session.world.llm_client = model
    human = HumanApi(lambda: session, seconds_per_tick=60)
    facade = LiaisonFacade(human)
    token = facade.session()["token"]
    for task_id, title in (("task_choice_dashboard", "Dashboard"),
                           ("task_choice_rules", "Red-light detection")):
        session.world.tasks[task_id] = Task(
            task_id=task_id, title=title, description=f"Implement {title} and verify it",
            status="open", owner_id=None, visibility="team")
    human.runtime().refresh_views()
    return session, human, facade, token


def plan_response():
    return {
        "reply": "我选 Dashboard，由你认领，再安排执行人员实现并验证，每五分钟在这里汇报。请确认启动。",
        "execution_plan": {
            "title": "Dashboard 实现", "goal": "Claim Dashboard for Victor, implement and verify it",
            "task_id": "task_choice_dashboard", "claim_task": True,
            "progress_interval_seconds": 300,
            "completion_criteria": "Source changes and test evidence, followed by a final report",
            "workers": [{"name": "Dashboard worker", "role": "Implementation and verification",
                         "assignment": "Inspect task_choice_dashboard, implement it, test and report the results."}],
        },
    }


def test_mixed_incomplete_actions_become_one_confirmable_claim_and_execution_plan():
    model = ScriptedModel([
        {"route": "ordinary"}, {"drafts": BROKEN_BATCH}, plan_response(),
    ])
    session, human, facade, token = fixture(model)
    try:
        before = len(session.world.action_log)
        result = facade.ask(token, REQUEST)
        assert not result.get("error")
        wait_idle(human._agent("victor"))
        state = facade.state(token)
        assert state["pending_actions"] == []
        assert len(state["execution_jobs"]) == 1
        job = state["execution_jobs"][0]
        assert job["status"] == "pending_confirmation"
        assert job["task_id"] == "task_choice_dashboard"
        assert job["claim_task"] is True
        assert job["progress_interval_seconds"] == 300
        assert job["resource_ref"].startswith("ri_")
        assert state["execution_agents"] == []
        assert session.world.tasks[job["task_id"]].owner_id is None
        assert len(session.world.action_log) == before
        assert not any(row.get("kind") == "clarification" for row in state["conversation"])
        repair = model.calls[-1][1]
        assert repair["human_request"] == REQUEST
        eligible = repair["verified_context"]["action_candidates"]["pick_task"]
        assert "task_choice_dashboard" in {row["object_id"] for row in eligible}
    finally:
        human.shutdown()


@pytest.mark.skipif(os.environ.get("HCI_LIVE_MODEL_TESTS") != "1",
                    reason="Explicit opt-in required for configured provider calls")
@pytest.mark.parametrize("followup", [False, True])
def test_live_model_plans_delegated_traffic_workflow(followup):
    """Exercise real routing and planning, stopping at the confirmation card."""
    session = OrgInspectorSession(
        seed=42, load_llm=True, interaction_profile="human_project_workspace",
        product_substrate={"type": "oss_time_machine", "dataset_id": "traffic_watch_v1",
                           "repository_id": "traffic_watch_v1", "mode": "dev"},
        selected_pack={"id": "traffic_watch_v1", "product_name": "traffic_violation_system"})
    assert session.world.llm_client is not None
    human = HumanApi(lambda: session)
    facade = LiaisonFacade(human)
    token = facade.session()["token"]
    secretary = human._agent("victor")
    try:
        if followup:
            # Reproduce the existing user's failed turn as conversation state;
            # every new routing/selection/planning response still uses the LLM.
            secretary.record_exchange(REQUEST, "当前有多个未认领任务，请选一个。", kind="clarification")
            secretary._set_pending_clarification(REQUEST)
        chosen_request = "你选一个就行" if followup else REQUEST
        print(f"live scenario: {chosen_request}", flush=True)
        result = facade.ask(token, chosen_request)
        assert not result.get("error"), result
        deadline = time.monotonic() + 300
        last_step = None
        while secretary.state()["busy"] and time.monotonic() < deadline:
            state = secretary.state()
            progress = state.get("working") or {}
            stage = (progress.get("step"), progress.get("summary"))
            if stage != last_step:
                print(json.dumps(progress, ensure_ascii=False), flush=True)
                last_step = stage
            time.sleep(0.5)
        assert not secretary.state()["busy"], "live planning did not finish"
        state = facade.state(token)
        jobs = state["execution_jobs"]
        assert len(jobs) == 1, state["conversation"][-4:]
        job = jobs[0]
        assert job["status"] == "pending_confirmation"
        assert job["claim_task"] is True
        assert job["progress_interval_seconds"] == 300
        assert session.world.tasks[job["task_id"]].owner_id is None
        assert state["pending_actions"] == []
        assert job["resource_ref"].startswith("ri_")
        member_names = {row["name"] for row in secretary._compiler_context()["members"]}
        assert not any(worker["name"] in member_names for worker in job["workers"])
        print(json.dumps({key: job[key] for key in
                          ("title", "task_id", "claim_task", "progress_interval_seconds", "workers")},
                         ensure_ascii=False), flush=True)
    finally:
        human.shutdown()


def test_you_choose_followup_keeps_original_claim_execution_and_report_cadence():
    model = ScriptedModel([
        {"route": "ordinary"}, {"drafts": BROKEN_BATCH},
        {"clarification": "先选哪一个任务？"},
        {"route": "ordinary"}, {"drafts": BROKEN_BATCH[:2]}, plan_response(),
    ])
    session, human, facade, token = fixture(model)
    try:
        facade.ask(token, REQUEST)
        wait_idle(human._agent("victor"))
        assert facade.state(token)["execution_jobs"] == []
        facade.ask(token, "你选一个就行")
        wait_idle(human._agent("victor"))
        state = facade.state(token)
        job = state["execution_jobs"][0]
        assert job["task_id"] == "task_choice_dashboard"
        assert job["claim_task"] is True
        assert job["progress_interval_seconds"] == 300
        repair_request = model.calls[-1][1]["human_request"]
        assert REQUEST in repair_request and "你选一个就行" in repair_request
        assert state["pending_actions"] == []
        assert session.world.tasks[job["task_id"]].owner_id is None
        assert state["conversation"][-1]["kind"] != "clarification"
        # Reporting must reach the same conversation with source links, and
        # an old runtime's callback must never appear in the current world.
        callback = {**job, "seat_id": "victor", "progress_summary": "Dashboard worker：正在检查任务代码。"}
        human._execution_progress(callback, expected_epoch=human._runtime_epoch - 1)
        assert not any(row.get("kind") == "execution_progress" for row in facade.state(token)["conversation"])
        human._execution_progress(callback, expected_epoch=human._runtime_epoch)
        progress = facade.state(token)["conversation"][-1]
        assert progress["kind"] == "execution_progress"
        assert progress["resource_ref"] == job["resource_ref"]
    finally:
        human.shutdown()


@pytest.mark.skipif(os.environ.get("HCI_LIVE_MODEL_TESTS") != "1",
                    reason="Explicit opt-in required for configured provider calls")
def test_live_model_allocates_one_task_to_victor_and_all_remaining_to_members():
    session = OrgInspectorSession(
        seed=42, load_llm=True, interaction_profile="human_project_workspace",
        product_substrate={"type": "oss_time_machine", "dataset_id": "traffic_watch_v1",
                           "repository_id": "traffic_watch_v1", "mode": "dev"},
        selected_pack={"id": "traffic_watch_v1", "product_name": "traffic_violation_system"})
    human = HumanApi(lambda: session)
    facade = LiaisonFacade(human)
    token = facade.session()["token"]
    secretary = human._agent("victor")
    initial_owners = {key: task.owner_id for key, task in session.world.tasks.items()}
    eligible = {row["object_id"] for row in
                secretary._compiler_context()["action_candidates"]["assign_task_owner"]["tasks"]}
    assert len(eligible) == 4
    try:
        secretary.record_exchange("当前状况如何？", "当前有7个开放任务，其中4个尚未分配。",
                                  kind="grounded_answer")
        result = facade.ask(token, "给我选一个任务然后把剩下没有分配的任务你也都分配出去")
        assert not result.get("error"), result
        deadline = time.monotonic() + 240
        while secretary.state()["busy"] and time.monotonic() < deadline:
            time.sleep(0.5)
        assert not secretary.state()["busy"]
        state = facade.state(token)
        drafts = state["pending_actions"]
        assert len(drafts) == 4, state["conversation"][-3:]
        assert state["execution_jobs"] == []
        planned = {
            row["params"]["task_id"]: ("victor" if row["action_type"] == "pick_task"
                                        else row["params"]["owner_id"])
            for row in drafts
        }
        assert set(planned) == eligible
        assert sum(owner == "victor" for owner in planned.values()) == 1
        assert {key: task.owner_id for key, task in session.world.tasks.items()} == initial_owners
        print("Live allocation reply:", state["conversation"][-1]["text"], flush=True)
        for row in drafts:
            assert facade.confirm(token, row["draft_id"])["ok"] is True
        assert {key: session.world.tasks[key].owner_id for key in eligible} == planned
        assert all(task.owner_id == initial_owners[key] for key, task in session.world.tasks.items()
                   if key not in eligible)
        print("Confirmed isolated-world owners:", json.dumps(planned, ensure_ascii=False), flush=True)
    finally:
        human.shutdown()
