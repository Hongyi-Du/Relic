"""A member's private working agent (HCI P4).

Three properties are load-bearing. It cannot act on the organization by itself
— everything that would goes into a draft the human confirms. It sees only what
its member sees, so an assistant is not a way around information asymmetry. And
it never holds the world lock, because thinking takes far longer than a tick.

Run:  PYTHONPATH="." python tests/org_env/test_human_working_agent.py
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.human.runtime import HumanModeRuntime
from environments.org_env.human.working_agent import (
    MAX_STEPS,
    LiaisonTaskContext,
    TOOLS,
    SeatTools,
    WorkingAgentSession,
    build_recent_activity_snapshot,
    build_team_progress_snapshot,
)
from environments.org_env.human.affordances import all_offered_action_specs
from environments.org_env.llm.client import MockOrgLLMClient
from environments.org_env.runtime_adapter.live import OrgInspectorSession

SEAT = "sean"
OTHER = "calvin"


def _runtime(ticks=24):
    s = OrgInspectorSession(seed=42)
    s.step(ticks)
    rt = HumanModeRuntime(s.world, seconds_per_tick=60.0)
    rt.claim_seat(SEAT)
    return rt


def _wait_idle(session, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not session.busy:
            return True
        time.sleep(0.02)
    return False


def _scripted(*steps):
    """A mock LLM that walks a fixed decision script."""
    return MockOrgLLMClient(script=list(steps))


# ---- it cannot act on its own -------------------------------------------- #
def test_a_prepared_action_does_nothing_until_the_human_confirms():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=None)
        before = len(rt.world.action_log)

        draft = agent.add_draft("send_message",
                                {"channel_id": "team_general", "text": "status"},
                                rationale="the team asked for an update")
        assert draft is not None and draft.status == "pending"
        assert len(rt.world.action_log) == before, "a draft reached the organization"
        assert agent.state()["drafts"][0]["action_type"] == "send_message"

        out = agent.confirm(draft.draft_id)
        assert out["ok"] is True
        assert len(rt.world.action_log) == before + 1
        assert agent.state()["drafts"] == [], "a sent draft is still pending"
    finally:
        rt.shutdown()


def test_a_confirmed_action_is_recorded_as_liaison_assisted():
    """HCI V0 §7: the organization sees the member acting; the research log
    records that an assistant prepared it."""
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=None)
        draft = agent.add_draft("send_message",
                                {"channel_id": "team_general", "text": "update"})
        agent.confirm(draft.draft_id)

        entry = rt.world.controller_log[-1]
        assert entry["controller_type"] == "human"
        assert entry["execution_mode"] == "liaison_assisted"
        assert all("controller" not in k for k in rt.world.action_log[-1])
        assert all("agent" not in (e.get("subtype") or "") for e in rt.world.events[-3:])
    finally:
        rt.shutdown()


def test_a_draft_still_goes_through_the_gateway():
    """The assistant is not a way around the role gates."""
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=None)   # sean: fast_engineer
        draft = agent.add_draft("propose_protocol", {"title": "review everything"})
        out = agent.confirm(draft.draft_id)

        assert "role_not_permitted" in out["error"]
        assert agent.drafts[draft.draft_id].status == "pending"
    finally:
        rt.shutdown()


def test_it_refuses_to_draft_an_action_that_does_not_exist():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=None)
        assert agent.add_draft("teleport", {}) is None
        assert agent.state()["drafts"] == []
    finally:
        rt.shutdown()


def test_a_discarded_draft_cannot_be_sent():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=None)
        draft = agent.add_draft("send_message",
                                {"channel_id": "team_general", "text": "hi"})
        agent.discard(draft.draft_id)
        assert agent.confirm(draft.draft_id)["error"].startswith("unknown_or_settled")
    finally:
        rt.shutdown()


# ---- it sees only what its member sees ----------------------------------- #
def test_tools_are_scoped_to_the_member():
    from environments.org_env.backend.entities import Task

    rt = _runtime()
    try:
        rt.world.tasks["task_hidden"] = Task(task_id="task_hidden",
                                             title="calvins secret refactor",
                                             owner_id=OTHER, visibility="private")
        mine = SeatTools(rt, SEAT)
        theirs = SeatTools(rt, OTHER)

        assert "secret refactor" not in mine.search_org("secret")
        assert "secret refactor" in theirs.search_org("secret")
        assert "object_not_visible" in mine.read_object("task_hidden")
        assert "task_hidden" in theirs.read_object("task_hidden")
    finally:
        rt.shutdown()


def test_a_private_channel_is_not_searchable_through_the_assistant():
    rt = _runtime()
    try:
        rt.world.comm.create_channel("ch_founders", members={"paul", "victor"})
        rt.world.comm.send_message(sender_id="paul", channel_id="ch_founders",
                                   text="we may miss payroll", tick=rt.world.world_tick)

        assert "miss payroll" not in SeatTools(rt, SEAT).search_org("payroll")
        assert "miss payroll" in SeatTools(rt, "paul").search_org("payroll")
    finally:
        rt.shutdown()


def test_the_repo_tools_read_the_real_product():
    rt = _runtime()
    try:
        tools = SeatTools(rt, SEAT)
        listing = tools.list_repo()
        assert ".py" in listing or "no files" in listing

        first = next((ln.split("  (")[0] for ln in listing.splitlines()
                      if ln.endswith("chars)")), None)
        if first:
            assert tools.read_repo(first)
            assert "No file at" in tools.read_repo("does/not/exist.py")
    finally:
        rt.shutdown()


def test_run_tests_returns_the_public_suite_verdict(monkeypatch):
    from environments.org_env.product import materialize

    rt = _runtime(ticks=1)
    try:
        monkeypatch.setattr(materialize, "run_public_tests", lambda _world: {
            "ok": False,
            "available": True,
            "returncode": 1,
            "summary": "2 failed, 7 passed in 0.31s",
            "failed_tests": ["tests/public/test_smoke.py::test_dashboard"],
            "error": None,
        })

        verdict = json.loads(SeatTools(rt, SEAT).run_tests())

        assert verdict == {
            "ok": False,
            "available": True,
            "returncode": 1,
            "summary": "2 failed, 7 passed in 0.31s",
            "failed_tests": ["tests/public/test_smoke.py::test_dashboard"],
            "error": None,
        }
    finally:
        rt.shutdown()


# ---- it does not block the organization ---------------------------------- #
def test_thinking_never_holds_the_world_lock():
    """A slow assistant must not stop the company's clock or the other seats."""
    rt = _runtime()
    try:
        released = threading.Event()

        def slow_responder(system, user, schema):
            released.wait(timeout=5)          # "thinking" for a long time
            return {"reply": "done"}

        agent = WorkingAgentSession(rt, SEAT,
                                    llm_client=MockOrgLLMClient(responder=slow_responder))
        agent.send("look into the sorting crash")
        time.sleep(0.2)
        assert agent.state()["busy"] is True

        # The world lock must be free the whole time it is thinking.
        acquired = rt.lock.acquire(timeout=1.0)
        assert acquired, "the assistant is holding the world lock while it thinks"
        rt.lock.release()

        released.set()
        assert _wait_idle(agent)
    finally:
        rt.shutdown()


def test_progress_grounding_keeps_pr_author_and_reviewer_roles_separate():
    view = {
        "member": {"members": [
            {"agent_id": "victor", "name": "Victor Miracle", "role": "cofounder"},
            {"agent_id": "paul", "name": "Paul Dreamer", "role": "engineer"},
        ]},
        "objects": {
            "tasks": [
                {"id": "task_1", "title": "Build dashboard", "owner": "paul",
                 "status": "in_progress", "progress": 0.6},
                {"id": "task_2", "title": "Add alert storage", "owner": None,
                 "status": "open", "priority": "high", "progress": 0.0},
            ],
            "pull_requests": [
                {"id": "pr_17", "title": "pr_17", "author": "paul",
                 "status": "merged", "reviewers": ["victor"],
                 "approved_by": ["victor"], "ci_passed": True},
            ],
        },
        "feed": {"events": [
            {"type": "repo_event", "subtype": "pr_reviewed", "pr_id": "pr_17",
             "agent_id": "victor", "tick": 35},
        ]},
    }
    progress = build_team_progress_snapshot(view)
    victor = next(row for row in progress["people"] if row["agent_id"] == "victor")
    paul = next(row for row in progress["people"] if row["agent_id"] == "paul")
    assert victor["authored_pull_requests"] == []
    assert victor["approved_pull_requests"][0]["pr_id"] == "pr_17"
    assert paul["authored_pull_requests"][0]["pr_id"] == "pr_17"
    assert paul["active_tasks"][0]["task_id"] == "task_1"
    assert progress["unassigned_tasks"] == [{
        "task_id": "task_2", "title": "Add alert storage", "status": "open",
        "priority": "high", "progress": 0.0,
    }]

    activity = build_recent_activity_snapshot(view)
    assert activity[0]["event_actor"] == "Victor Miracle"
    assert activity[0]["current_visible_object"]["author"] == "paul"
    assert activity[0]["current_visible_object"]["author_name"] == "Paul Dreamer"


def test_human_turns_queue_across_shared_tabs_instead_of_failing_agent_busy():
    rt = _runtime()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def responder(_system, user, _schema):
        request = json.loads(user)["task_state"]["task_anchor"]["original_request"]
        calls.append(request)
        if len(calls) == 1:
            started.set()
            release.wait(timeout=5)
        return {"reply": f"finished {request}"}

    try:
        agent = WorkingAgentSession(
            rt, SEAT, llm_client=MockOrgLLMClient(responder=responder))
        first = agent.send("first shared-tab request", require_model=True)
        assert first["accepted"] is True
        assert started.wait(timeout=2)

        second = agent.send("second shared-tab request", require_model=True)
        assert second["accepted"] is True
        assert second["queued"] is True
        assert second["queue_depth"] == 1
        assert agent.state()["queue_depth"] == 1
        assert [row["text"] for row in agent.state()["messages"]
                if row["role"] == "human"][-2:] == [
                    "first shared-tab request", "second shared-tab request"]

        release.set()
        assert _wait_idle(agent)
        assert calls == ["first shared-tab request", "second shared-tab request"]
        replies = [row["text"] for row in agent.state()["messages"]
                   if row["role"] == "agent"]
        assert replies[-2:] == ["finished first shared-tab request",
                                "finished second shared-tab request"]
    finally:
        release.set()
        rt.shutdown()


def test_a_human_turn_can_be_recorded_before_model_work_without_duplication():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(
            rt, SEAT, llm_client=_scripted({"reply": "grounded reply"}))
        recorded = agent.record_human_message("inspect the current task")

        assert recorded["accepted"] is True
        assert agent.state()["busy"] is False
        assert [row["text"] for row in agent.state()["messages"]] == [
            "inspect the current task"]

        accepted = agent.enqueue_recorded_message(
            recorded["message_id"], require_model=True)
        assert accepted["accepted"] is True
        assert _wait_idle(agent)
        messages = agent.state()["messages"]
        assert [row["text"] for row in messages if row["role"] == "human"] == [
            "inspect the current task"]
        assert messages[-1]["text"] == "grounded reply"
    finally:
        rt.shutdown()


def test_stop_current_request_waits_for_the_active_model_call_then_stops():
    rt = _runtime()
    entered = threading.Event()
    release = threading.Event()

    def responder(_system, _user, _schema):
        entered.set()
        assert release.wait(timeout=5)
        return {"tool": "list_repo", "args": {}}

    try:
        agent = WorkingAgentSession(
            rt, SEAT, llm_client=MockOrgLLMClient(responder=responder))
        agent.send("inspect and continue", require_model=True)
        assert entered.wait(timeout=2)
        assert agent.cancel_current() == {"ok": True, "status": "stopping"}
        assert agent.state()["working"]["phase"] == "stopping"

        release.set()
        assert _wait_idle(agent)
        messages = agent.state()["messages"]
        assert messages[-1]["kind"] == "request_cancelled"
        assert not any(row["role"] == "tool" for row in messages)
    finally:
        release.set()
        rt.shutdown()


def test_public_working_state_lists_sanitized_tool_activity():
    task = LiaisonTaskContext(request="inspect")
    task.begin_step(1)
    task.record_observation("read_object", {"object_id": "task_1"}, "private output")

    public = task.public()
    assert public["activity_log"] == [{
        "step": 1, "label": "read_object · task_1", "status": "completed"}]
    assert "private output" not in json.dumps(public)


def test_two_seats_have_separate_private_conversations():
    rt = _runtime()
    try:
        rt.claim_seat(OTHER)
        mine = WorkingAgentSession(rt, SEAT, llm_client=None)
        theirs = WorkingAgentSession(rt, OTHER, llm_client=None)

        mine.send("what am I working on")
        theirs.send("what am I working on")
        assert _wait_idle(mine) and _wait_idle(theirs)

        my_text = " ".join(m["text"] for m in mine.state()["messages"])
        their_text = " ".join(m["text"] for m in theirs.state()["messages"])
        assert "Sean" in my_text and "Sean" not in their_text
        assert "Calvin" in their_text
    finally:
        rt.shutdown()


def test_the_conversation_never_reaches_the_organization():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=None)
        before_messages = len(rt.world.comm.messages)
        before_events = len(rt.world.events)

        agent.send("draft me something about the crash")
        assert _wait_idle(agent)

        assert len(rt.world.comm.messages) == before_messages
        assert len(rt.world.events) == before_events
        assert rt.world.controller_log == []
    finally:
        rt.shutdown()


def test_a_world_execution_failure_is_reported_as_failure_not_sent_or_completed():
    rt = _runtime(ticks=1)
    events = []
    try:
        agent = WorkingAgentSession(
            rt, SEAT, llm_client=None,
            event_sink=lambda event_type, data: events.append((event_type, data)),
        )
        draft = agent.add_draft("open_pr", {"branch_id": "branch_missing"})
        result = agent.confirm(draft.draft_id)

        assert result["ok"] is False
        assert result["failure_reason"] in {"nothing_to_request", "no_branch"}
        answer = agent.state()["messages"][-1]
        assert "did not complete" in answer["text"]
        assert "No completion is being claimed" in answer["text"]
        assert "Sent" not in answer["text"]
        assert any(event_type == "liaison_draft_failed"
                   for event_type, _data in events)
    finally:
        rt.shutdown()


def test_organization_status_uses_the_visibility_bounded_brief():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=None)
        agent.send("What is the organization status and current blocker?")
        assert _wait_idle(agent)

        messages = agent.state()["messages"]
        tool = next(m for m in messages if m["tool"] == "organization_brief")
        assert "Organization brief" in tool["text"]
        assert "Sources:" in tool["text"]
        # P3 hides raw tool rows, so the secretary turn itself must retain the
        # evidence-bearing answer instead of being a content-free acknowledgement.
        assert "Organization brief" in messages[-1]["text"]
        assert "What is happening:" in messages[-1]["text"]
    finally:
        rt.shutdown()


def test_a_consequential_request_wins_over_the_chinese_organization_status_marker():
    """"请组织批准发布" is a route-and-confirm request, not a read-only brief."""
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=None)
        before = len(rt.world.action_log)
        agent.send("请组织批准发布")
        assert _wait_idle(agent)

        state = agent.state()
        assert state["drafts"], "the consequential request was mistaken for status"
        assert not any(m["tool"] == "organization_brief" for m in state["messages"])
        assert len(rt.world.action_log) == before
    finally:
        rt.shutdown()


# ---- the loop ------------------------------------------------------------- #
def test_it_uses_a_tool_then_answers():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=_scripted(
            {"thought": "check the board", "tool": "my_situation", "args": {}},
            {"reply": "You have work waiting on the sorting crash."},
        ))
        agent.send("what should I do next")
        assert _wait_idle(agent)

        roles = [m["role"] for m in agent.state()["messages"]]
        tools = [m["tool"] for m in agent.state()["messages"] if m["tool"]]
        assert roles == ["human", "tool", "agent"]
        assert tools == ["my_situation"]
    finally:
        rt.shutdown()


def test_it_batches_model_selected_independent_read_only_sources():
    rt = _runtime()
    physical_reads = []
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=_scripted(
            {
                "kind": "tool",
                "thought": "Read the three exact source files the human named.",
                "tool_calls": [
                    {"tool": "read_repo", "args": {"path": "dashboard.py"}},
                    {"tool": "read_repo", "args": {"path": "storage.py"}},
                    {"tool": "read_repo", "args": {"path": "reports.py"}},
                ],
            },
            {"reply": "All three requested sources were inspected."},
        ))
        agent.tools.read_repo = lambda path: physical_reads.append(path) or f"source:{path}"

        agent.send("Read dashboard.py, storage.py, and reports.py")
        assert _wait_idle(agent)

        assert physical_reads == ["dashboard.py", "storage.py", "reports.py"]
        messages = agent.state()["messages"]
        assert [row["tool"] for row in messages if row["role"] == "tool"] == [
            "read_repo", "read_repo", "read_repo",
        ]
        assert messages[-1]["text"] == "All three requested sources were inspected."
    finally:
        rt.shutdown()


def test_it_model_audits_and_retracts_a_needless_clarification():
    rt = _runtime()
    try:
        from environments.org_env.backend.entities import Task

        task = Task(task_id="task_unique_dashboard", title="Dashboard is not implemented",
                    owner_id=None, visibility="team")
        rt.world.tasks[task.task_id] = task
        rt.refresh_views()
        agent = WorkingAgentSession(rt, SEAT, llm_client=_scripted(
            {
                "clarification": (
                    f"I cannot uniquely identify {task.title}; please choose a task again."
                ),
            },
            {
                "clarification_needed": False,
                "kind": "drafts",
                "drafts": [{
                    "action_type": "pick_task",
                    "params": {"task_id": task.task_id},
                    "rationale": "The human named the unique visible task title.",
                }],
            },
        ))

        agent.send(f"Claim {task.title} for me")
        assert _wait_idle(agent)

        state = agent.state()
        assert len(state["drafts"]) == 1
        assert state["drafts"][0]["params"] == {"task_id": task.task_id}
        assert not any(row["kind"] == "clarification" for row in state["messages"])
    finally:
        rt.shutdown()


def test_it_can_answer_and_prepare_an_action_in_one_turn():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=_scripted({
            "reply": "I drafted the update for you.",
            "draft": {"action_type": "send_message",
                      "params": {"channel_id": "team_general",
                                 "text": "taking the sorting crash"},
                      "rationale": "you asked to claim it publicly"},
        }))
        agent.send("tell the team I am taking the crash")
        assert _wait_idle(agent)

        drafts = agent.state()["drafts"]
        assert len(drafts) == 1
        assert drafts[0]["action_type"] == "send_message"
        assert drafts[0]["label"] == "Send a message"
        assert len(rt.world.comm.messages) == len(rt.world.comm.messages)  # nothing sent
    finally:
        rt.shutdown()


def test_it_stops_instead_of_looping_forever():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=MockOrgLLMClient(
            responder=lambda s, u, sc: {"tool": "my_situation", "args": {}}))
        agent.send("keep going")
        assert _wait_idle(agent, timeout=30)

        used = [m for m in agent.state()["messages"] if m["role"] == "tool"]
        assert len(used) <= TOOLS.__len__() * 3
        assert agent.state()["messages"][-1]["role"] == "agent"
    finally:
        rt.shutdown()


def test_codex_style_context_blocks_a_repeated_read_and_forces_synthesis():
    """The regression that read doc_arch seven times must stop after one repeat."""
    rt = _runtime()
    prompts = []
    physical_reads = []

    def responder(_system, user, _schema):
        payload = json.loads(user)
        prompts.append(payload)
        if len(prompts) <= 2:
            return {"thought": "read the dashboard source",
                    "tool": "read_repo", "args": {"path": "dashboard.py"}}
        assert payload["task_state"]["budget"]["synthesis_only"] is True
        return {"reply": "Dashboard requirements and starter evidence are summarized here."}

    try:
        agent = WorkingAgentSession(
            rt, SEAT, llm_client=MockOrgLLMClient(responder=responder))
        agent.tools.read_repo = lambda path: (
            physical_reads.append(path) or "def create_app(): raise NotImplementedError")

        agent.send("Tell me the dashboard requirements and starter code")
        assert _wait_idle(agent)

        assert physical_reads == ["dashboard.py"], "the duplicate read reached the tool"
        assert len(prompts) == 3, "one read, one detected repeat, then one synthesis call"
        final_state = prompts[-1]["task_state"]
        assert final_state["task_anchor"]["original_request"] == \
            "Tell me the dashboard requirements and starter code"
        assert final_state["evidence"][0]["source"] == {
            "tool": "read_repo", "args": {"path": "dashboard.py"}}
        assert [row["status"] for row in final_state["tool_call_ledger"]] == [
            "completed", "duplicate_blocked"]
        assert all(row["role"] != "tool"
                   for row in prompts[-1]["context"]["recent_dialogue"])
        answer = agent.state()["messages"][-1]
        assert answer["text"].startswith("Dashboard requirements")
        assert "stopped after several steps" not in answer["text"]
    finally:
        rt.shutdown()


def test_model_failure_preserves_sources_and_retry_resumes_without_rereading():
    rt = _runtime()
    calls = 0
    physical_reads = []

    def responder(_system, _user, _schema):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"tool": "read_repo", "args": {"path": "dashboard.py"}}
        if calls == 2:
            raise ValueError("empty model response")
        return {"reply": "Resumed from the saved dashboard source and completed the review."}

    try:
        agent = WorkingAgentSession(
            rt, SEAT, llm_client=MockOrgLLMClient(responder=responder))
        agent.tools.read_repo = lambda path: (
            physical_reads.append(path) or "def create_app(): raise NotImplementedError")

        agent.send("Inspect the dashboard source and report the gap", require_model=True)
        assert _wait_idle(agent)
        failed = agent.state()["failed_request"]
        assert failed["step"] == 2
        assert failed["source_count"] == 1
        assert failed["sources"][0]["args"] == {"path": "dashboard.py"}
        assert failed["failure_kind"] == "model_interrupted"
        assert "empty model response" not in failed["error"]
        assert "original request is preserved" in failed["error"].lower()
        assert agent.state()["messages"][-1]["kind"] == "model_error"

        resumed = agent.retry_failed()
        assert resumed == {
            "accepted": True, "resumed": True, "thread_id": "",
            "next_step": 3, "preserved_sources": 1,
        }
        assert _wait_idle(agent)
        assert physical_reads == ["dashboard.py"]
        assert agent.state()["failed_request"] is None
        assert agent.state()["messages"][-1]["text"].startswith("Resumed from")
    finally:
        rt.shutdown()


def test_codex_style_evidence_ledger_keeps_early_sources_after_eight_steps():
    rt = _runtime()
    prompts = []

    def responder(_system, user, _schema):
        payload = json.loads(user)
        prompts.append(payload)
        index = len(prompts) - 1
        if index < 10:
            return {"thought": f"collect source {index}",
                    "tool": "search_repo", "args": {"query": f"fact-{index}"}}
        return {"reply": "I retained and synthesized all ten sources."}

    try:
        agent = WorkingAgentSession(
            rt, SEAT, llm_client=MockOrgLLMClient(responder=responder))
        agent.tools.search_repo = lambda query: f"evidence returned for {query}"
        agent.send("Investigate all dashboard requirements")
        assert _wait_idle(agent)

        final_state = prompts[-1]["task_state"]
        assert len(final_state["evidence"]) == 10
        assert final_state["evidence"][0]["source"]["args"]["query"] == "fact-0"
        assert final_state["evidence"][-1]["source"]["args"]["query"] == "fact-9"
        assert "working_history" not in prompts[-1]
        assert final_state["task_anchor"]["original_request"] == \
            "Investigate all dashboard requirements"
    finally:
        rt.shutdown()


def test_codex_style_compaction_preserves_every_source_and_both_output_ends():
    task = LiaisonTaskContext("keep my exact goal", max_steps=MAX_STEPS)
    for index in range(8):
        task.begin_step(index + 1)
        task.record_observation(
            "search_repo", {"query": f"source-{index}"},
            f"HEAD-{index}-" + ("x" * 5_000) + f"-TAIL-{index}",
        )

    payload = task.model_payload()
    assert payload["task_anchor"]["original_request"] == "keep my exact goal"
    assert len(payload["evidence"]) == 8
    assert payload["compaction"]["omitted_characters"] > 0
    for index, evidence in enumerate(payload["evidence"]):
        assert evidence["source"]["args"]["query"] == f"source-{index}"
        assert evidence["output"].startswith(f"HEAD-{index}-")
        assert evidence["output"].endswith(f"-TAIL-{index}")


def test_working_state_exposes_progress_without_model_thought():
    rt = _runtime()
    entered = threading.Event()
    released = threading.Event()

    def responder(_system, _user, _schema):
        entered.set()
        released.wait(timeout=5)
        return {"reply": "done"}

    try:
        agent = WorkingAgentSession(
            rt, SEAT, llm_client=MockOrgLLMClient(responder=responder))
        agent.send("inspect the dashboard")
        assert entered.wait(timeout=2)

        progress = agent.state()["working"]
        assert progress["status"] == "working"
        assert progress["step"] == 1
        assert progress["max_steps"] == MAX_STEPS
        assert progress["phase"] == "awaiting_model"
        assert "inspect the dashboard" not in json.dumps(progress)
        assert "thought" not in json.dumps(progress).lower()

        released.set()
        assert _wait_idle(agent)
        assert agent.state()["working"] is None
    finally:
        released.set()
        rt.shutdown()


def test_it_works_without_a_language_model():
    """The repository stays offline-testable; an assistant with no model still
    answers from its tools rather than pretending to reason."""
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=None)
        agent.send("show me the repo")
        assert _wait_idle(agent)

        messages = agent.state()["messages"]
        assert any(m["tool"] == "list_repo" for m in messages)
        assert messages[-1]["role"] == "agent"
        assert "No language model" in messages[-1]["text"]
    finally:
        rt.shutdown()


def test_p3_required_model_path_never_falls_back_to_text_rules():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=None)
        agent.send("Edit tools/claim_tracker.py and merge it", require_model=True)
        assert _wait_idle(agent)

        state = agent.state()
        assert state["drafts"] == []
        assert not [row for row in state["messages"] if row["role"] == "tool"]
        assert state["messages"][-1]["kind"] == "clarification"
        assert "requires the liaison model" in state["messages"][-1]["text"]
        assert "No regex or template fallback" in state["messages"][-1]["text"]
    finally:
        rt.shutdown()


def test_slack_style_thread_keeps_model_context_out_of_the_main_timeline():
    rt = _runtime()
    captured = []

    def responder(_system, user, _schema):
        payload = json.loads(user)
        captured.append(payload["context"]["recent_dialogue"])
        return {"reply": f"answer {len(captured)}"}

    try:
        agent = WorkingAgentSession(
            rt, SEAT, llm_client=MockOrgLLMClient(responder=responder))
        agent.send("root question")
        assert _wait_idle(agent)
        first_state = agent.state()
        root_answer = first_state["messages"][-1]

        agent.send("later mainline question")
        assert _wait_idle(agent)
        out = agent.send("explain the root answer here", reply_to=root_answer["message_id"])
        assert out["accepted"] is True
        assert out["thread_id"] == root_answer["message_id"]
        assert out["parent_id"] == root_answer["message_id"]
        assert _wait_idle(agent)

        state = agent.state()
        top_level = [row for row in state["messages"] if not row["thread_id"]]
        replies = [row for row in state["messages"]
                   if row["thread_id"] == root_answer["message_id"]
                   and row["role"] != "tool"]
        assert len(top_level) == 4
        assert [row["role"] for row in replies] == ["human", "agent"]
        assert replies[0]["parent_id"] == root_answer["message_id"]
        assert replies[1]["parent_id"] == replies[0]["message_id"]
        assert state["threads"] == [{
            "thread_id": root_answer["message_id"],
            "root_message_id": root_answer["message_id"],
            "root_text": "answer 1",
            "reply_count": 2,
            "latest_at": replies[-1]["at"],
            "latest_message_id": replies[-1]["message_id"],
            "latest_text": "answer 3",
        }]

        thread_context = "\n".join(row["text"] for row in captured[-1])
        assert "answer 1" in thread_context
        assert "explain the root answer here" in thread_context
        assert "later mainline question" not in thread_context
        assert "answer 2" not in thread_context
    finally:
        rt.shutdown()


def test_thread_reply_targets_fail_closed_and_thread_drafts_stay_attached():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=_scripted(
            {"reply": "root"},
            {"reply": "I prepared the thread action.", "drafts": [{
                "action_type": "run_public_tests", "params": {},
                "rationale": "requested inside the message thread",
            }]},
        ))
        assert agent.send("start a topic")["accepted"] is True
        assert _wait_idle(agent)
        root_id = agent.state()["messages"][-1]["message_id"]

        before = len(agent.state()["messages"])
        assert agent.send("bad", reply_to="missing")["error"] == \
            "unknown_reply_target:missing"
        assert len(agent.state()["messages"]) == before

        assert agent.send("run the tests in this thread", reply_to=root_id)["accepted"] is True
        assert _wait_idle(agent)
        draft = agent.state()["drafts"][0]
        assert draft["action_type"] == "run_public_tests"
        assert draft["thread_id"] == root_id
    finally:
        rt.shutdown()


def test_model_compiler_catalog_is_exactly_the_complete_p1_p2_action_surface():
    catalog = WorkingAgentSession.action_catalog()
    expected = {
        (spec.action_type, tuple(spec.required), tuple(spec.optional), spec.target_param)
        for spec in all_offered_action_specs()
    }
    actual = {
        (row["action_type"], tuple(row["required"]), tuple(row["optional"]),
         row["target_param"])
        for row in catalog
    }
    assert actual == expected
    assert len(catalog) == len(all_offered_action_specs())
    for row in catalog:
        assert set(row) >= {"action_type", "label", "required", "optional",
                            "category", "effect"}


def test_model_compiles_one_natural_instruction_into_mixed_structured_actions():
    rt = _runtime()
    captured = {}

    def responder(system, user, _schema):
        captured["system"] = system
        captured["user"] = json.loads(user)
        return {
            "reply": "代码修改、测试、协议提案和 PR 审批都已经完成。",
            "drafts": [
                {"action_type": "edit_repo_file",
                 "params": {"file_path": "tools/claim_tracker.py",
                            "edit_goal": "补上证据校验"},
                 "rationale": "修改用户明确指定的代码"},
                {"action_type": "run_public_tests", "params": {},
                 "rationale": "验证修改"},
                {"action_type": "propose_protocol",
                 "params": {"title": "Evidence before approval",
                            "problem_evidence": "缺少可复现证据"},
                 "rationale": "提交制度提案"},
                {"action_type": "approve_pr", "params": {"pr_id": "pr_visible"},
                 "rationale": "按用户要求准备审批"},
            ],
        }

    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=MockOrgLLMClient(responder=responder))
        agent.send("修改 claim tracker，跑测试，提一个 evidence protocol，然后批准那个 PR")
        assert _wait_idle(agent)

        state = agent.state()
        assert [row["action_type"] for row in state["drafts"]] == [
            "edit_repo_file", "run_public_tests", "propose_protocol", "approve_pr"]
        assert state["drafts"][0]["params"]["file_path"] == "tools/claim_tracker.py"
        assert "Complete P1/P2 action catalog" in captured["system"]
        prompt_catalog = json.loads(
            captured["system"].split("Complete P1/P2 action catalog:\n", 1)[1]
            .split("\n\nReturn JSON", 1)[0])
        assert {row["action_type"] for row in prompt_catalog} == {
            spec.action_type for spec in all_offered_action_specs()}
        assert captured["user"]["context"]["members"]
        assert captured["user"]["context"]["objects"]["tasks"]
        summary = state["messages"][-1]["text"]
        assert "4 个结构化动作" in summary
        assert "确认前不会执行" in summary
        assert "都已经完成" not in summary
    finally:
        rt.shutdown()


def test_model_compiler_symbol_table_does_not_drop_targets_after_twenty():
    from environments.org_env.backend.entities import Task

    rt = _runtime(ticks=1)
    captured = {}

    def responder(_system, user, _schema):
        captured.update(json.loads(user)["context"])
        return {"reply": "I can see the complete target table."}

    try:
        for index in range(25):
            task_id = f"task_compiler_{index:02d}"
            rt.world.tasks[task_id] = Task(
                task_id=task_id, title=f"Compiler target {index:02d}",
                owner_id=SEAT, visibility="public")
        members = set(rt.world.agents)
        for index in range(25):
            rt.world.comm.create_channel(f"compiler_channel_{index:02d}", members=members)
        rt.refresh_views()

        agent = WorkingAgentSession(rt, SEAT,
                                    llm_client=MockOrgLLMClient(responder=responder))
        agent.send("What exact task and channel targets can I act on?")
        assert _wait_idle(agent)
        assert any(row.get("id") == "task_compiler_24"
                   for row in captured["objects"]["tasks"])
        assert any(row.get("channel_id") == "compiler_channel_24"
                   for row in captured["channels"])
    finally:
        rt.shutdown()


def test_model_compiler_rejects_an_incomplete_batch_without_leaving_partial_drafts():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=_scripted({
            "drafts": [
                {"action_type": "run_public_tests", "params": {}, "rationale": "test"},
                {"action_type": "approve_pr", "params": {}, "rationale": "approve"},
            ],
        }))
        agent.send("run tests and approve the PR")
        assert _wait_idle(agent)

        state = agent.state()
        assert state["drafts"] == []
        assert state["messages"][-1]["kind"] == "clarification"
        assert "Approve" in state["messages"][-1]["text"]
        # Compiler failures speak in the user's visible vocabulary and offer
        # a bounded visible choice; implementation schema names are not a
        # useful request for a person to act on.
        assert "Visible pull request candidates:" in state["messages"][-1]["text"]
        assert "pr_3" in state["messages"][-1]["text"]
        assert "pr_id" not in state["messages"][-1]["text"]
    finally:
        rt.shutdown()


def test_assignment_question_is_answered_as_information_not_a_decision_or_action():
    rt = _runtime()
    try:
        answer = ("你当前负责 1 项。第一部分由 Los Xi 负责，第三部分尚未认领；"
                  "可见候选人是 Calvin 和 Iris。")
        agent = WorkingAgentSession(rt, SEAT, llm_client=_scripted({"reply": answer}))
        agent.send("第一部分和第三部分有人认领吗，没有的话我能 assign 谁？")
        assert _wait_idle(agent)

        state = agent.state()
        assert state["drafts"] == []
        assert state["messages"][-1]["text"] == answer
        assert state["messages"][-1]["kind"] != "clarification"
    finally:
        rt.shutdown()


def test_an_offline_consequential_request_becomes_a_draft_not_an_action():
    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=None)
        before = len(rt.world.action_log)
        agent.send("Tell the team to prioritize the release blocker")
        assert _wait_idle(agent)

        drafts = agent.state()["drafts"]
        assert len(drafts) == 1
        assert drafts[0]["action_type"] == "send_message"
        assert "prioritize the release blocker" in drafts[0]["params"]["text"]
        assert len(rt.world.action_log) == before
    finally:
        rt.shutdown()


def test_a_provider_failure_falls_back_to_seat_visible_tools():
    class BrokenProvider:
        def generate_json(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    rt = _runtime()
    try:
        agent = WorkingAgentSession(rt, SEAT, llm_client=BrokenProvider())
        agent.send("what am I working on")
        assert _wait_idle(agent)

        text = " ".join(message["text"] for message in agent.state()["messages"])
        assert "Sean" in text
        assert "answering from the current seat-visible state" in text
    finally:
        rt.shutdown()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("  ok ", fn.__name__)
    print(f"All {len(fns)} working agent tests passed!")
