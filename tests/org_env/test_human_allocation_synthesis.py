"""Regression contracts for synthesized secretary task allocation.

These cases intentionally exercise the ``kind=synthesize`` route.  The
secretary must retain a direct allocation request as provisional P1/P2 cards,
not turn it into a request to read information it already has in its
seat-visible compiler context.

Run: PYTHONPATH=. python -m pytest tests/org_env/test_human_allocation_synthesis.py -q
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.backend.entities import Task
from environments.org_env.human.runtime import HumanModeRuntime
from environments.org_env.human.working_agent import WorkingAgentSession
from environments.org_env.runtime_adapter.live import OrgInspectorSession


REQUEST = "给我选一个任务然后把剩下没有分配的任务你也都分配出去"
SEAT = "victor"
TASK_IDS = ("task_allocate_1", "task_allocate_2", "task_allocate_3", "task_allocate_4")


class ScriptedModel:
    """A deterministic secretary model that records both synthesis stages."""

    def __init__(self, json_responses, text_responses=()):
        self.json_responses = list(json_responses)
        self.text_responses = list(text_responses)
        self.json_calls = []
        self.text_calls = []

    def generate_json(self, system, user, schema, **_kwargs):
        self.json_calls.append((system, json.loads(user), schema))
        assert self.json_responses, "unexpected JSON model call"
        return self.json_responses.pop(0)

    def generate_text(self, system, user, **_kwargs):
        self.text_calls.append((system, json.loads(user)))
        assert self.text_responses, "unexpected text model call"
        return self.text_responses.pop(0)


def _wait_idle(agent, timeout=5.0):
    deadline = time.monotonic() + timeout
    while agent.state()["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not agent.state()["busy"]


def _runtime_with_tasks(model):
    inspector = OrgInspectorSession(seed=42)
    inspector.step(1)
    runtime = HumanModeRuntime(inspector.world, seconds_per_tick=60.0)
    runtime.claim_seat(SEAT)
    # The production inspector has seed-specific board items.  Isolate this
    # contract to exactly four eligible rows so all-eligible coverage is
    # deterministic rather than coupled to fixture evolution.
    inspector.world.tasks.clear()
    for index, task_id in enumerate(TASK_IDS, 1):
        inspector.world.tasks[task_id] = Task(
            task_id=task_id,
            title=f"Allocation task {index}",
            description=f"Visible allocation work {index}",
            status="open",
            owner_id=None,
            visibility="team",
        )
    inspector.world.tasks["task_existing_owner"] = Task(
        task_id="task_existing_owner",
        title="Already owned work",
        description="Must remain outside an all-eligible allocation.",
        status="open",
        owner_id="scarlett",
        visibility="team",
    )
    runtime.refresh_views()
    return runtime, WorkingAgentSession(runtime, SEAT, llm_client=model)


def _all_eligible_drafts():
    return [
        {"action_type": "pick_task", "params": {"task_id": TASK_IDS[0]},
         "rationale": "Victor keeps the highest-priority visible task."},
        *[
            {"action_type": "assign_task_owner",
             "params": {"task_id": task_id, "owner_id": owner_id},
             "rationale": "Assigned from the seat-visible workload."}
            for task_id, owner_id in zip(TASK_IDS[1:], ("calvin", "iris", "paul"))
        ],
    ]


def _synthesis_drafts(drafts):
    return {
        "evidence_status": "sufficient",
        "mode": "drafts",
        "drafts": drafts,
        "reply": "已为 Victor 和其他成员完成全部任务分配。",
        "references": [{"object_id": task_id} for task_id in TASK_IDS],
        "allocation_scope": "all_eligible",
        "self_task_count": 1,
    }


def _allocation_repair(assignments=None):
    rows = assignments or [
        {"task_id": task_id, "owner_id": owner_id,
         "rationale": "Replanned from the verified current candidate set."}
        for task_id, owner_id in zip(
            TASK_IDS, (SEAT, "calvin", "iris", "paul"))
    ]
    return {
        "scope": "all_eligible",
        "self_task_count": 1,
        "assignments": rows,
        "reply": "已根据当前可分配任务和成员修正完整分配。",
    }


def test_synthesis_keeps_full_allocation_as_confirmable_owner_drafts():
    model = ScriptedModel(
        [
            {"kind": "synthesize", "thought": "Current task and member context is sufficient."},
            _synthesis_drafts(_all_eligible_drafts()),
        ],
        ["## Allocation prepared\n\nAll four cards await Victor's confirmation."],
    )
    runtime, agent = _runtime_with_tasks(model)
    try:
        before = len(runtime.world.action_log)
        agent.send(REQUEST, require_model=True)
        _wait_idle(agent)

        state = agent.state()
        drafts = state["drafts"]
        assert len(drafts) == 4
        assert [row["action_type"] for row in drafts] == ["assign_task_owner"] * 4
        assert drafts[0]["params"] == {"task_id": TASK_IDS[0], "owner_id": SEAT}
        assert {row["params"]["task_id"] for row in drafts[1:]} == set(TASK_IDS[1:])
        assert len(runtime.world.action_log) == before, "draft creation must not write"
        assert all(runtime.world.tasks[task_id].owner_id is None for task_id in TASK_IDS)
        assert runtime.world.tasks["task_existing_owner"].owner_id == "scarlett"
        assert not any(row["role"] == "tool" for row in state["messages"])
        assert not any(row["kind"] == "clarification" for row in state["messages"])
        prepared = state["messages"][-1]["text"]
        assert "4 个结构化动作" in prepared
        assert "确认前不会执行" in prepared
        assert "完成全部任务分配" not in prepared

        # The live simulation may claim a task after planning but before the
        # human confirms. An explicit allocation to Victor remains the exact
        # confirmed intent instead of failing as a stale first-come claim.
        runtime.world.tasks[TASK_IDS[0]].owner_id = "scarlett"
        for draft in list(drafts):
            assert agent.confirm(draft["draft_id"])["ok"] is True
        assert runtime.world.tasks[TASK_IDS[0]].owner_id == SEAT
        assert {runtime.world.tasks[task_id].owner_id for task_id in TASK_IDS[1:]} == {
            "calvin", "iris", "paul",
        }
        assert runtime.world.tasks["task_existing_owner"].owner_id == "scarlett"
        assert len(runtime.world.action_log) == before + 4
        recap = agent.state()["messages"][-1]
        assert recap["kind"] == "action_result"
        assert "本次批量操作结果" in recap["text"]
        assert recap["text"].count("- 成功:") == 4
    finally:
        runtime.shutdown()


def test_delegated_allocation_retracts_a_needless_clarification_and_synthesizes():
    proposed = (
        "There are several task and owner candidates. Please choose one before I continue."
    )
    model = ScriptedModel([
        {"kind": "clarification", "clarification": proposed},
        {
            "clarification_needed": False,
            "kind": "synthesize",
            "thought": "Victor explicitly delegated the allocation choices.",
        },
        _synthesis_drafts(_all_eligible_drafts()),
    ])
    runtime, agent = _runtime_with_tasks(model)
    try:
        agent.send(REQUEST + "，谁做什么你来决定", require_model=True)
        _wait_idle(agent)

        state = agent.state()
        assert len(state["drafts"]) == 4
        assert not any(row["kind"] == "clarification" for row in state["messages"])
        assert "确认前不会执行" in state["messages"][-1]["text"]
        audit_system, audit_payload, audit_schema = model.json_calls[1]
        assert "Multiple eligible candidates" in audit_system
        assert audit_payload["proposed_clarification"] == proposed
        assert audit_schema["required"] == ["clarification_needed", "kind"]
    finally:
        runtime.shutdown()


def test_synthesis_report_receives_current_allocation_context_when_ledger_is_empty():
    model = ScriptedModel(
        [
            {"kind": "synthesize", "thought": "The board is already in context."},
            {"evidence_status": "sufficient", "mode": "report", "references": []},
        ],
        ["The current visible board has been summarized without a new read."],
    )
    runtime, agent = _runtime_with_tasks(model)
    try:
        agent.send("总结当前可见任务分配", require_model=True)
        _wait_idle(agent)

        planner_payload = model.json_calls[1][1]
        report_payload = model.text_calls[0][1]
        for payload in (planner_payload, report_payload):
            assert payload["task_state"]["evidence"] == []
            candidates = payload["context"]["action_candidates"]
            assert {row["object_id"] for row in candidates["pick_task"]} == set(TASK_IDS)
            assert {row["object_id"] for row in candidates["assign_task_owner"]["tasks"]} == set(TASK_IDS)
            assert candidates["assign_task_owner"]["members"]
            assert payload["context"]["team_progress"]["people"]
            assert {row["task_id"] for row in
                    payload["context"]["team_progress"]["unassigned_tasks"]} >= set(TASK_IDS)
        assert agent.state()["drafts"] == []
    finally:
        runtime.shutdown()


def test_allocation_validation_survives_autonomous_claims_during_model_latency():
    model = ScriptedModel([])
    runtime, agent = _runtime_with_tasks(model)
    try:
        decision = _synthesis_drafts(_all_eligible_drafts())
        runtime.world.tasks[TASK_IDS[0]].owner_id = "scarlett"
        runtime.world.tasks[TASK_IDS[1]].owner_id = "calvin"
        runtime.refresh_views()

        drafts, returned = agent._validate_or_repair_allocation(
            _all_eligible_drafts(), decision, REQUEST)

        assert returned is decision
        assert model.json_calls == [], "a live ownership race must not trigger model repair"
        assert [row["action_type"] for row in drafts] == ["assign_task_owner"] * 4
        assert drafts[0]["params"] == {"task_id": TASK_IDS[0], "owner_id": SEAT}
    finally:
        runtime.shutdown()


def test_synthesis_needs_source_reads_automatically_then_continues():
    model = ScriptedModel(
        [
            {"kind": "synthesize", "thought": "Need one current progress snapshot."},
            {"evidence_status": "needs_source", "missing_fact": "current visible workload",
             "next_tool": "team_progress", "next_args": {}},
            {"kind": "synthesize", "thought": "The progress snapshot now grounds the report."},
            {"evidence_status": "sufficient", "mode": "report", "references": []},
        ],
        ["The report uses the real current team-progress source."],
    )
    runtime, agent = _runtime_with_tasks(model)
    try:
        agent.send("根据当前团队进度给我报告", require_model=True)
        _wait_idle(agent)

        messages = agent.state()["messages"]
        assert [row["tool"] for row in messages if row["role"] == "tool"] == ["team_progress"]
        assert messages[-1]["text"] == "The report uses the real current team-progress source."
        assert not any(row["kind"] == "clarification" for row in messages)
        assert model.json_calls[3][1]["task_state"]["evidence"][0]["source"]["tool"] == "team_progress"
    finally:
        runtime.shutdown()


def test_invalid_allocation_gets_one_model_correction_before_reaching_the_human():
    model = ScriptedModel([
        {"kind": "synthesize", "thought": "Allocate every remaining task."},
        _synthesis_drafts(_all_eligible_drafts()[:2]),
        _allocation_repair(),
    ])
    runtime, agent = _runtime_with_tasks(model)
    try:
        before = len(runtime.world.action_log)
        agent.send(REQUEST, require_model=True)
        _wait_idle(agent)

        state = agent.state()
        assert len(state["drafts"]) == 4
        assert state["failed_request"] is None
        assert {row["params"]["task_id"] for row in state["drafts"]} == set(TASK_IDS)
        assert all(row["action_type"] == "assign_task_owner" for row in state["drafts"])
        assert sum(row["params"].get("owner_id") == SEAT for row in state["drafts"]) == 1
        assert len(runtime.world.action_log) == before
        assert all(runtime.world.tasks[task_id].owner_id is None for task_id in TASK_IDS)

        repair_system, repair_payload, _schema = model.json_calls[2]
        assert "public_validator_feedback" in repair_payload
        assert "cover all" in repair_payload["public_validator_feedback"].lower()
        assert repair_payload["required_scope"] == "all_eligible"
        assert repair_payload["required_self_task_count"] == 1
        assert "Repair one rejected task-allocation plan" in repair_system
    finally:
        runtime.shutdown()


@pytest.mark.parametrize(
    "drafts,metadata",
    [
        (_all_eligible_drafts()[:2], {"allocation_scope": "all_eligible", "self_task_count": 1}),
        ([
            _all_eligible_drafts()[0],
            _all_eligible_drafts()[1],
            _all_eligible_drafts()[1],
            _all_eligible_drafts()[3],
        ], {"allocation_scope": "all_eligible", "self_task_count": 1}),
        ([
            _all_eligible_drafts()[0],
            {"action_type": "pick_task", "params": {"task_id": TASK_IDS[1]},
             "rationale": "Invalid second self claim."},
            _all_eligible_drafts()[2],
            _all_eligible_drafts()[3],
        ], {"allocation_scope": "all_eligible", "self_task_count": 1}),
    ],
    ids=["partial_coverage", "duplicate_task", "wrong_self_task_count"],
)
def test_synthesis_rejects_malformed_all_eligible_coverage_without_partial_drafts(drafts, metadata):
    conclusion = _synthesis_drafts(drafts)
    conclusion.update(metadata)
    model = ScriptedModel([
        {"kind": "synthesize", "thought": "Allocate every remaining task."},
        conclusion,
        _allocation_repair(assignments=[
            {"task_id": TASK_IDS[0], "owner_id": SEAT, "rationale": "first"},
            {"task_id": TASK_IDS[0], "owner_id": "calvin", "rationale": "duplicate"},
        ]),
    ])
    runtime, agent = _runtime_with_tasks(model)
    try:
        before = len(runtime.world.action_log)
        agent.send(REQUEST, require_model=True)
        _wait_idle(agent)

        state = agent.state()
        assert state["drafts"] == []
        assert state["messages"][-1]["kind"] == "model_error"
        assert "任务分配" in state["messages"][-1]["text"]
        failed = state["failed_request"]
        assert failed["failure_kind"] == "invalid_model_plan"
        assert failed["retry_current_step"] is True
        assert failed["step"] == 1
        assert failed["next_step"] == 1
        assert "ValueError" not in failed["error"]
        assert "ineligible" not in failed["error"]
        assert len(runtime.world.action_log) == before
        assert all(runtime.world.tasks[task_id].owner_id is None for task_id in TASK_IDS)
    finally:
        runtime.shutdown()


def test_invalid_allocation_retry_replans_the_same_step_from_current_state():
    model = ScriptedModel([
        {"kind": "synthesize", "thought": "First allocation attempt."},
        _synthesis_drafts(_all_eligible_drafts()[:2]),
        _allocation_repair(assignments=[
            {"task_id": TASK_IDS[0], "owner_id": SEAT, "rationale": "first"},
            {"task_id": TASK_IDS[0], "owner_id": "calvin", "rationale": "duplicate"},
        ]),
        {"kind": "synthesize", "thought": "Retry the same planning step."},
        _synthesis_drafts(_all_eligible_drafts()),
    ])
    runtime, agent = _runtime_with_tasks(model)
    try:
        agent.send(REQUEST, require_model=True)
        _wait_idle(agent)
        assert agent.state()["failed_request"]["next_step"] == 1

        assert agent.retry_failed() == {
            "accepted": True,
            "resumed": True,
            "thread_id": "",
            "next_step": 1,
            "preserved_sources": 0,
        }
        _wait_idle(agent)

        state = agent.state()
        assert state["failed_request"] is None
        assert len(state["drafts"]) == 4
        assert model.json_calls[3][1]["task_state"]["budget"]["step"] == 1
    finally:
        runtime.shutdown()
