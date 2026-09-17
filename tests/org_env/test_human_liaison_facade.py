"""P3 Organization-as-a-Service façade contract.

The façade is intentionally less revealing than P2: it gets its facts only
from P2's filtered seat view, then replaces source ids with token-bound opaque
evidence refs.  These tests exercise the contract without requiring FastAPI.
"""
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.human.api import HumanApi
from environments.org_env.human.affordances import all_offered_action_types
from environments.org_env.human.liaison import LiaisonFacade, P3_FIXED_SEAT_ID
from environments.org_env.llm.client import MockOrgLLMClient
from environments.org_env.runtime_adapter.live import OrgInspectorSession

SEAT = "victor"


def _facade(ticks=24):
    session = OrgInspectorSession(seed=42)
    session.step(ticks)
    human = HumanApi(lambda: session, seconds_per_tick=60.0)
    facade = LiaisonFacade(human)
    token = facade.session()["token"]
    return facade, human, session, token


def test_human_text_is_visible_while_the_semantic_router_is_still_running():
    entered = threading.Event()
    release = threading.Event()

    class BlockingRouter:
        def generate_json(self, _system, _user, _schema, **_kwargs):
            entered.set()
            assert release.wait(timeout=5)
            return {"route": "grounded_answer", "reply": "Current visible answer."}

    facade, human, session, token = _facade(ticks=4)
    session.world.llm_client = BlockingRouter()
    result = {}

    def ask():
        result.update(facade.ask(token, "What is happening now?"))

    thread = threading.Thread(target=ask)
    try:
        thread.start()
        assert entered.wait(timeout=2)
        conversation = facade.state(token)["conversation"]
        assert any(row["role"] == "human" and row["text"] == "What is happening now?"
                   for row in conversation)
        release.set()
        thread.join(timeout=5)
        assert result["handled"] == "grounded_answer"
        conversation = facade.state(token)["conversation"]
        assert sum(row["role"] == "human" and row["text"] == "What is happening now?"
                   for row in conversation) == 1
    finally:
        release.set()
        thread.join(timeout=5)
        human.shutdown()


def test_p3_opens_directly_as_victor_and_rejects_every_other_seat_token():
    facade, human, _session, token = _facade(ticks=4)
    try:
        assert P3_FIXED_SEAT_ID == SEAT
        resumed = facade.session(token)
        assert resumed["agent_id"] == SEAT
        assert resumed["token"] == token
        assert resumed["reused"] is True

        # Codex's embedded browser and Chrome have separate localStorage.
        # A second client without a token attaches to the same fixed seat
        # instead of trying to claim Victor again.
        second_client = facade.session()
        assert second_client["agent_id"] == SEAT
        assert second_client["token"] == token
        assert second_client["reused"] is True
        assert second_client["shared_fixed_session"] is True

        other = human.claim("calvin")["token"]
        assert facade.state(other)["error"] == "p3_fixed_seat_required:victor"
        assert facade.session(other)["error"] == "p3_fixed_seat_required:victor"
        assert facade.session("bad")["error"] == "invalid_seat_token"
    finally:
        human.shutdown()


def test_default_state_hides_the_raw_p2_collections_and_uses_opaque_refs():
    from environments.org_env.backend.entities import Task

    facade, human, _session, token = _facade()
    try:
        world = human.runtime().world
        world.tasks["task_private"] = Task(task_id="task_private", title="private launch plan",
                                            owner_id="calvin", visibility="private")
        human.runtime().refresh_views()

        state = facade.state(token)
        encoded = json.dumps(state, default=str).lower()
        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(row) for row in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(row) for row in value))
            return set()

        forbidden_keys = {"member", "feed", "objects", "roster", "channels",
                          "controller", "policy", "evaluator"}
        assert not (forbidden_keys & keys(state))
        assert "private launch plan" not in encoded
        assert "task_private" not in encoded
        assert "task_private" not in encoded
        assert "private launch plan" not in encoded

        assert set(state) >= {"project_summary", "workstreams", "blocker",
                              "decision_inbox", "organization_available", "conversation"}
        visible_company = human.runtime().seat_view(SEAT)["member"]["company"]
        assert state["project_summary"]["title"] == visible_company["product_name"]
        assert state["project_summary"]["title"] != "LanternForge"
        assert "organization" not in state
        refs = [ref for row in state["workstreams"]
                for ref in row["evidence_refs"]]
        assert refs and all(ref.startswith("ev_") and "task_" not in ref for ref in refs)
    finally:
        human.shutdown()


def test_state_projects_the_working_context_without_private_reasoning():
    facade, human, _session, token = _facade(ticks=4)
    try:
        human._agent(SEAT)._working_state = {
            "status": "working",
            "phase": "investigating",
            "summary": "Read read_object · task_1.",
            "step": 3,
            "max_steps": 24,
            "remaining_steps": 21,
            "tools_completed": 2,
            "unique_sources": 2,
            "duplicate_calls_blocked": 0,
            "started_at": 1.0,
            "thread_id": "",
        }
        state = facade.state(token)
        assert state["working"]["phase"] == "investigating"
        assert state["working"]["unique_sources"] == 2
        assert "thought" not in state["working"]
        assert "request" not in state["working"]
    finally:
        human.shutdown()


def test_decision_inbox_preserves_human_judgment_and_composer_starters():
    facade, human, _session, token = _facade(ticks=4)
    try:
        row = facade._item(token, {
            "id": "proposal_visible", "kind": "proposal",
            "title": "Read-Before-Edit Protocol",
            "what": "Decide proposal: Read-Before-Edit Protocol",
            "why": "You are a designated approver; the liaison cannot approve it for you.",
            "impact": "This determines whether the protocol can be adopted.",
            "context": {"proposal_type": "protocol", "current_status": "under_review"},
            "decision_type": "proposal_approval",
            "refs": [{"kind": "proposal", "id": "proposal_visible",
                      "title": "Read-Before-Edit Protocol"}],
        })
        assert row["summary"] == "Decide proposal: Read-Before-Edit Protocol"
        assert row["requires_human_decision"] is True
        assert row["decision_type"] == "proposal_approval"
        assert row["why"].startswith("You are a designated approver")
        assert row["impact"] == "This determines whether the protocol can be adopted."
        assert row["context"]["current_status"] == "under_review"
        assert [option["id"] for option in row["decision_options"]] == [
            "context", "approve", "changes", "reject"]
        assert all(option["instruction"] for option in row["decision_options"])
        assert all(ref.startswith("ev_") for ref in row["evidence_refs"])
        assert "proposal_visible" not in json.dumps(row["decision_options"])
    finally:
        human.shutdown()


def test_decision_item_has_a_direct_seat_scoped_source_in_addition_to_summary_evidence():
    """A decision card must open its actual visible object without a second NL hop.

    ``evidence_refs`` explain a compact FYI/decision summary.  They are not a
    substitute for the resource inspector's token-bound ``ri_`` handle: a
    human who must review an object needs an explicit, directly usable route
    to that currently visible source.
    """
    from environments.org_env.proposals.objects import Proposal

    facade, human, _session, token = _facade(ticks=4)
    try:
        proposal = Proposal(
            proposal_id="proposal_decision_source",
            proposal_type="workflow_proposal",
            title="Require source-backed release decisions",
            summary="Every decision card should lead to the record it summarizes.",
            proposer_agent_id="calvin",
            target_problem="A summary alone cannot be audited.",
            proposed_solution="Expose a read-only, seat-scoped source handle.",
            approval_required_from=[SEAT],
            status="under_review",
            created_at_tick=4,
        )
        human.runtime().world.proposal_manager.proposals[proposal.proposal_id] = proposal
        human.runtime().refresh_views()

        row = facade._item(token, {
            "id": proposal.proposal_id,
            "kind": "proposal",
            "title": proposal.title,
            "what": f"Decide proposal: {proposal.title}",
            "why": "Victor is the designated approver.",
            "decision_type": "proposal_approval",
            "refs": [{"kind": "proposal", "id": proposal.proposal_id,
                      "title": proposal.title}],
        })

        assert row["resource_ref"].startswith("ri_")
        assert proposal.proposal_id not in row["resource_ref"]
        # The direct handle resolves the source record, not another secretary
        # summary or a guessed raw proposal id.
        source = facade.resource(token, row["resource_ref"], "overview")
        assert source["resource"]["id"] == proposal.proposal_id
        assert source["content"]["title"] == proposal.title
        assert source["content"]["status"] == "under_review"
        assert source["content"]["proposal_type"] == "workflow_proposal"
        assert source["content"]["author"] == "calvin"
    finally:
        human.shutdown()


def test_decision_option_keeps_its_exact_proposal_reference_through_router_and_compiler():
    """A card click must not ask the model to rediscover either target or option."""
    from environments.org_env.proposals.objects import Proposal

    class DecisionAwareModel:
        def __init__(self, proposal_id):
            self.proposal_id = proposal_id
            self.router_payload = None
            self.compiler_payload = None

        def generate_json(self, system, user, _schema, **_kwargs):
            payload = json.loads(user)
            if "semantic router" in system:
                self.router_payload = payload
                return {"route": "ordinary", "reply": ""}
            if "Audit one proposed clarification" in system:
                return {
                    "clarification_needed": True,
                    "kind": "clarification",
                    "clarification": "Please identify the proposal again.",
                }
            self.compiler_payload = payload
            current = payload["context"]["recent_dialogue"][-1]
            assert current["decision_context"]["option_id"] == "approve"
            assert current["references"][0]["object_id"] == self.proposal_id
            anchor = payload["task_state"]["task_anchor"]
            assert anchor["selected_decision"]["option_id"] == "approve"
            assert anchor["selected_references"][0]["object_id"] == self.proposal_id
            # Reproduce the observed model failure.  The server-validated card
            # selection must still bind the exact proposal/action afterwards.
            return {"clarification": "Please identify the proposal again."}

    facade, human, _session, token = _facade(ticks=4)
    try:
        proposal = Proposal(
            proposal_id="proposal_exact_card_reference",
            proposal_type="protocol_proposal",
            title="Founder Approval Authority Delegation Protocol",
            summary="Delegate routine decisions while retaining consequential approvals.",
            proposer_agent_id="paul",
            approval_required_from=[SEAT],
            status="under_review",
            created_at_tick=4,
        )
        human.runtime().world.proposal_manager.proposals[proposal.proposal_id] = proposal
        human.runtime().refresh_views()
        decision = next(row for row in facade.state(token)["decision_inbox"]
                        if row["title"] == proposal.title)
        worker = human._agent(SEAT)
        assert decision["id"].startswith("decision_ev_")
        reply_scope = worker.conversation_context(reply_to=decision["id"])
        assert reply_scope["thread_id"] == decision["id"]
        assert reply_scope["messages"][0]["references"][0]["object_id"] == proposal.proposal_id
        model = DecisionAwareModel(proposal.proposal_id)
        worker.llm = model

        sent = facade.ask(
            token,
            f'I want to approve the proposal “{proposal.title}”. Prepare it for confirmation.',
            decision_context={
                "resource_ref": decision["resource_ref"],
                "option_id": "approve",
                # Deliberately forged display values must be ignored.
                "title": "another proposal",
                "option_label": "amend",
            },
        )
        assert sent["accepted"] is True
        worker._thread.join(timeout=10)

        assert model.router_payload["submitted_decision_context"]["title"] == proposal.title
        assert model.router_payload["submitted_decision_context"]["option_label"] == "Draft approval"
        pending = facade.state(token)["pending_actions"]
        assert [(row["action_type"], row["params"]) for row in pending] == [
            ("approve_proposal", {"proposal_id": proposal.proposal_id})]
        human_turn = next(row for row in reversed(facade.state(token)["conversation"])
                          if row["role"] == "human")
        assert human_turn["decision_context"]["title"] == proposal.title
        assert human_turn["decision_context"]["option_id"] == "approve"
    finally:
        human.shutdown()


def test_decision_reference_is_revalidated_before_the_human_turn_is_recorded():
    facade, human, _session, token = _facade(ticks=4)
    try:
        before = len(facade.state(token)["conversation"])
        result = facade.ask(
            token, "approve this",
            decision_context={"resource_ref": "ri_not_a_visible_decision", "option_id": "approve"},
        )
        assert result["error"] == "stale_or_invisible_decision_reference"
        assert len(facade.state(token)["conversation"]) == before
    finally:
        human.shutdown()


def test_fyi_evidence_adds_event_and_related_object_facts_without_repeating_the_headline():
    """Progressive disclosure must add auditable facts at each layer."""
    facade, human, _session, token = _facade(ticks=4)
    try:
        runtime = human.runtime()
        world = runtime.world
        task = world.tasks["task_onboarding_doc"]
        task.owner_id = "paul"
        world.board.owners[task.task_id] = "paul"
        world.events.extend([
            {"type": "release_event", "subtype": "readiness_check", "agent_id": SEAT,
             "candidate_id": "rc_evidence_ladder", "status": "blocked", "tick": 411,
             "blockers": ["gate_ci", "gate_evidence"]},
            {"type": "release_event", "subtype": "blocker_to_task", "agent_id": SEAT,
             "candidate_id": "rc_evidence_ladder", "task_id": task.task_id, "owner_id": "paul",
             "gate": "gate_ci", "tick": 411},
        ])
        runtime.refresh_views()

        state = facade.state(token)
        notice = next(row for row in state["notifications"]
                      if row["kind"] == "release_readiness")
        headline = notice["summary"]
        evidence = facade.evidence(token, notice["evidence_refs"][0])["evidence"]
        trace = facade.trace(token, notice["evidence_refs"][0])["trace"]

        # Layer 2 makes the event record inspectable; layer 3 adds the related
        # task/owner object.  Neither may be a content-free copy of layer 1.
        assert evidence["event"] == {
            "type": "release_event", "subtype": "readiness_check",
            "agent_id": SEAT, "candidate_id": "rc_evidence_ladder",
            "status": "blocked", "tick": 411,
            "blockers": ["gate_ci", "gate_evidence"],
        }
        assert evidence["event_summary"] == {
            "actor": "Victor Miracle", "action": "readiness check",
            "organization_tick": 411,
        }
        assert "gate ci" in evidence["why"].lower()
        assert "Paul Dreamer" in evidence["why"]
        assert headline != evidence["why"]
        related = evidence["related_current_visible_objects"]
        assert any(row["id"] == task.task_id and row["owner"] == "paul" for row in related)
        assert len(trace["steps"]) >= 3
        assert trace["steps"][0]["organization_tick"] == 411
        assert trace["steps"][0]["event"]["candidate_id"] == "rc_evidence_ladder"
        assert trace["steps"][1]["kind"] == "visible_evidence"
        assert "Create customer-facing onboarding doc" in trace["steps"][1]["summary"]
        assert trace["steps"][1]["summary"] != headline
        related_step = trace["steps"][2]
        assert related_step["kind"] == "organizational_object"
        assert related_step["object"]["id"] == task.task_id
        assert related_step["object"]["owner"] == "paul"
    finally:
        human.shutdown()


def test_evidence_and_trace_resolve_only_current_p2_cited_sources():
    facade, human, _session, token = _facade()
    try:
        state = facade.state(token)
        refs = [ref for row in state["workstreams"]
                for ref in row["evidence_refs"]]
        assert refs
        evidence = facade.evidence(token, refs[0])
        trace = facade.trace(token, refs[0])
        assert evidence["evidence"]["ref"] == refs[0]
        assert trace["trace"]["ref"] == refs[0]
        assert "id" not in evidence["evidence"]["source"]
        raw_source_ids = [ref["id"] for ref in human.brief(token)["brief"]["source_refs"]]
        assert trace["trace"]["source"]["id"] in raw_source_ids
        assert trace["trace"]["steps"][-1]["kind"] in {
            "organizational_object", "visible_message"
        }
        assert facade.evidence(token, "ev_not_a_real_ref")["error"] == "unknown_evidence_ref"

        # The reference is token-bound: a second P2 seat cannot resolve it.
        other = human.claim("calvin")["token"]
        assert facade.evidence(other, refs[0])["error"] == "p3_fixed_seat_required:victor"
    finally:
        human.shutdown()


def test_p3_reads_are_read_only_and_use_the_same_visible_brief():
    facade, human, _session, token = _facade()
    try:
        world = human.runtime().world
        before = (len(world.action_log), len(world.controller_log), len(world.events))
        state = facade.state(token)
        facade.organization(token)
        ref = state["workstreams"][0]["evidence_refs"][0]
        facade.evidence(token, ref)
        facade.trace(token, ref)
        assert (len(world.action_log), len(world.controller_log), len(world.events)) == before
    finally:
        human.shutdown()


def test_show_organization_reveals_only_a_safe_seat_visible_outline():
    from environments.org_env.backend.entities import Task

    facade, human, _session, token = _facade()
    try:
        world = human.runtime().world
        world.tasks["task_private"] = Task(task_id="task_private", title="Calvin private plan",
                                            owner_id="calvin", visibility="private")
        human.runtime().refresh_views()

        shown = facade.organization(token)["organization"]
        encoded = json.dumps(shown, default=str)
        assert shown["agents"], "Show organization should reveal the visible roster"
        assert "recent_activity" in shown
        assert shown["recent_activity"], "The organization outline should retain visible events"
        assert all(item.get("what") and item.get("why") and item.get("evidence_refs")
                   for item in shown["recent_activity"])
        assert "Calvin private plan" not in encoded
        assert "task_private" not in encoded
        assert "agent_id" not in encoded
        assert "controller" not in encoded
    finally:
        human.shutdown()


def test_confirm_is_exactly_once_and_records_liaison_assisted():
    facade, human, _session, token = _facade()
    try:
        draft = human._agent(SEAT).add_draft(
            "send_message", {"channel_id": "team_general", "text": "status"})
        before = len(human.runtime().world.action_log)

        assert facade.confirm(token, draft.draft_id)["ok"] is True
        assert len(human.runtime().world.action_log) == before + 1
        assert human.runtime().world.controller_log[-1]["execution_mode"] == "liaison_assisted"
        assert facade.confirm(token, draft.draft_id)["error"].startswith("unknown_or_settled")
        assert len(human.runtime().world.action_log) == before + 1
    finally:
        human.shutdown()


def test_confirm_revalidates_the_current_world_and_leaves_a_refused_draft_pending():
    facade, human, _session, token = _facade()
    try:
        draft = human._agent(SEAT).add_draft(
            "send_message", {"channel_id": "team_general", "text": "status"})
        world = human.runtime().world
        world.comm.channels["team_general"].members.remove(SEAT)
        before = len(world.action_log)

        out = facade.confirm(token, draft.draft_id)
        assert out["error"] == "not_a_channel_member:team_general"
        assert len(world.action_log) == before
        assert human._agent(SEAT).drafts[draft.draft_id].status == "pending"
    finally:
        human.shutdown()


def test_invalid_token_is_rejected_by_every_p3_read_and_write_route():
    facade, human, _session, _token = _facade()
    try:
        for result in (
            facade.state("bad"), facade.organization("bad"), facade.evidence("bad", "ev_x"),
            facade.trace("bad", "ev_x"), facade.ask("bad", "hello"),
            facade.confirm("bad", "draft_x"), facade.discard("bad", "draft_x"),
        ):
            assert result["error"] == "invalid_seat_token"
    finally:
        human.shutdown()


def test_p3_facade_preserves_slack_thread_identity_and_scoped_drafts():
    facade, human, _session, token = _facade(ticks=4)
    try:
        worker = human._agent(SEAT)
        worker.llm = MockOrgLLMClient(script=[
            {"route": "ordinary", "reply": ""},
            {"reply": "Here is the top-level answer."},
            {"route": "ordinary", "reply": ""},
            {"reply": "Thread-only plan", "drafts": [{
                "action_type": "run_public_tests", "params": {},
                "rationale": "requested in this thread",
            }]},
        ])
        assert facade.ask(token, "Open a topic")["accepted"] is True
        worker._thread.join(timeout=10)
        root = next(row for row in reversed(facade.state(token)["conversation"])
                    if row["role"] == "liaison" and row.get("id"))
        assert root["id"].startswith("liaison_msg_")

        out = facade.ask(token, "Run tests as a reply", reply_to=root["id"])
        assert out["thread_id"] == root["id"]
        worker._thread.join(timeout=10)
        state = facade.state(token)
        replies = [row for row in state["conversation"]
                   if row.get("thread_id") == root["id"]]
        assert [row["role"] for row in replies] == ["human", "liaison"]
        assert replies[0]["parent_id"] == root["id"]
        assert state["conversation_threads"][0]["reply_count"] == 2
        assert state["pending_actions"][0]["thread_id"] == root["id"]
    finally:
        human.shutdown()


def test_review_instruction_with_you_go_language_reaches_the_model_action_path():
    facade, human, _session, token = _facade(ticks=4)
    try:
        worker = human._agent(SEAT)
        worker.llm = MockOrgLLMClient(script=[
            {"route": "ordinary", "reply": ""},
            {"reply": "I will ground Will's visible task before preparing review and merge actions."},
        ])
        out = facade.ask(
            token, "will的那个任务，你去查一下，给我一份review报告，没问题就可以合并到main去")
        assert out["accepted"] is True
        assert "handled" not in out
        worker._thread.join(timeout=10)
        answer = worker.state()["messages"][-1]
        assert "Will's visible task" in answer["text"]
        assert "meeting" not in answer["text"].lower()
    finally:
        human.shutdown()


def test_p3_lifecycle_and_runtime_need_no_human_namespace_calls():
    facade, human, _session, _token = _facade(ticks=4)
    try:
        assert facade.session(_token)["agent_id"] == SEAT
        assert facade.runtime_status()["running"] is False
        assert facade.runtime_start(60)["running"] is True
        assert facade.runtime_pause()["running"] is False
        # P3 delivery memory belongs to its token, never the organization.
        state = facade.state(_token)
        assert state["notifications"]
        assert facade.release(_token)["released"] == SEAT
        assert _token not in facade._notification_history
        assert _token not in facade._notification_seen
    finally:
        human.shutdown()


def test_prepare_has_the_full_p2_action_surface_but_is_zero_mutation_until_confirm():
    """The integrated composer is not a reduced secretarial action set."""
    facade, human, _session, token = _facade()
    try:
        world = human.runtime().world
        before = (len(world.action_log), len(world.controller_log), len(world.events))
        offered = all_offered_action_types()
        drafted = []
        for action_type in offered:
            result = facade.prepare(token, action_type, {})
            assert result["draft"]["action_type"] == action_type
            drafted.append(result["draft"]["draft_id"])
        assert len(set(drafted)) == len(offered)
        # Preparation neither executes nor grants a liaison-only action.
        assert (len(world.action_log), len(world.controller_log), len(world.events)) == before
        assert facade.prepare("bad", "send_message", {})["error"] == "invalid_seat_token"
    finally:
        human.shutdown()


def test_ambiguous_delegation_clarifies_then_prepares_existing_p2_assignment_and_route():
    facade, human, _session, token = _facade()
    try:
        worker = human._agent(SEAT)
        worker.llm = None  # deterministic: do not invoke an external provider in this behavior test
        world = human.runtime().world
        before = len(world.action_log)

        worker._work("Please delegate the tracker improvement to someone")
        assert not worker.state()["drafts"]
        clarification = worker.state()["messages"][-1]["text"]
        assert "Visible tasks, in order:" in clarification
        assert "Visible assignment candidates:" in clarification
        assert len(world.action_log) == before

        worker._work("Why do you need that, and which people and tasks are visible?")
        followup = worker.state()["messages"][-1]
        assert followup["kind"] == "clarification"
        assert "Visible assignment candidates:" in followup["text"]
        assert not worker.state()["drafts"]
        assert len(world.action_log) == before
        assert any(message.get("kind") == "clarification"
                   for message in facade.state(token)["conversation"])

        worker._work("Ask Calvin to implement Enforce evidence links in claim tracker")
        types = {draft["action_type"] for draft in worker.state()["drafts"]}
        assert types == {"assign_task_owner", "send_message"}
        assert len(world.action_log) == before
        message = worker.state()["messages"][-1]["text"].lower()
        assert "none of them says that work has started or completed" in message
    finally:
        human.shutdown()


def test_single_composer_reaches_test_and_repo_edit_through_the_model_compiler():
    facade, human, _session, _token = _facade()
    try:
        worker = human._agent(SEAT)
        worker.llm = MockOrgLLMClient(script=[
            {"reply": "Prepared the requested test.", "drafts": [{
                "action_type": "run_public_tests", "params": {},
                "rationale": "The human explicitly requested the public tests.",
            }]},
            {"reply": "Prepared the requested repository edit.", "drafts": [{
                "action_type": "edit_repo_file",
                "params": {"file_path": "tools/claim_tracker.py",
                           "edit_goal": "add a narrow guard"},
                "rationale": "The human explicitly named this path and edit goal.",
            }]},
        ])
        world = human.runtime().world
        before = len(world.action_log)
        worker._work("Please run public tests")
        first = worker.state()["drafts"][-1]
        assert first["action_type"] == "run_public_tests"
        assert len(world.action_log) == before

        worker._work("Edit tools/claim_tracker.py to add a narrow guard")
        second = worker.state()["drafts"][-1]
        assert second["action_type"] == "edit_repo_file"
        assert second["params"]["file_path"] == "tools/claim_tracker.py"
        assert len(world.action_log) == before
    finally:
        human.shutdown()


def test_model_resolves_visible_ordinal_and_member_without_text_parser_or_execution():
    """The model resolves natural references to exact visible ids itself."""
    class _SemanticModel:
        def __init__(self):
            self.calls = 0

        def generate_json(self, _system, user, _schema, **_kwargs):
            self.calls += 1
            context = json.loads(user)["context"]
            task_id = context["objects"]["tasks"][0]["id"]
            return {
                "drafts": [{
                    "action_type": "assign_task_owner",
                    "params": {"task_id": task_id, "owner_id": "sean"},
                    "rationale": "Assign the task the human just referenced to Sean.",
                }],
                "reply": "",
            }

    facade, human, _session, _token = _facade()
    try:
        worker = human._agent(SEAT)
        model = _SemanticModel()
        worker.llm = model
        visible = human.runtime().seat_view(SEAT)
        first_task = (visible.get("objects") or {}).get("tasks", [])[0]
        before = len(human.runtime().world.action_log)

        worker._work("第一个任务给我分配给sean吧")

        drafts = worker.state()["drafts"]
        assert model.calls == 1, "the model must select the action semantics"
        assert len(drafts) == 1
        assert drafts[0]["action_type"] == "assign_task_owner"
        assert drafts[0]["params"] == {
            "task_id": first_task["id"], "owner_id": "sean",
        }
        assert len(human.runtime().world.action_log) == before
        assert worker.state()["messages"][-1]["kind"] == "action_interpretation"
    finally:
        human.shutdown()


def test_model_resolves_first_person_will_without_a_regex_name_matcher():
    """The model, not a substring matcher, distinguishes 'I will' from Will."""
    class _SemanticModel:
        def generate_json(self, _system, user, _schema, **_kwargs):
            context = json.loads(user)["context"]
            return {
                "drafts": [{
                    "action_type": "assign_task_owner",
                    "params": {"task_id": context["objects"]["tasks"][0]["id"],
                               "owner_id": "sean"},
                    "rationale": "Prepare the requested assignment.",
                }],
            }

    facade, human, _session, _token = _facade()
    try:
        worker = human._agent(SEAT)
        worker.llm = _SemanticModel()
        first_task = human.runtime().seat_view(SEAT)["objects"]["tasks"][0]
        before = len(human.runtime().world.action_log)

        worker._work("I will assign the first task to Sean")

        drafts = worker.state()["drafts"]
        assert len(drafts) == 1
        assert drafts[0]["params"] == {
            "task_id": first_task["id"], "owner_id": "sean",
        }
        assert len(human.runtime().world.action_log) == before
    finally:
        human.shutdown()


def test_missing_model_parameter_is_not_guessed_by_a_text_parser():
    """Incomplete model output fails closed with visible candidates."""
    class _IncompleteModel:
        def generate_json(self, _system, user, _schema, **_kwargs):
            context = json.loads(user)["context"]
            return {
                "drafts": [{
                    "action_type": "assign_task_owner",
                    "params": {"task_id": context["objects"]["tasks"][0]["id"]},
                    "rationale": "The human did not choose an owner.",
                }],
                "clarification": "Need owner_id.",
            }

    facade, human, _session, _token = _facade()
    try:
        worker = human._agent(SEAT)
        worker.llm = _IncompleteModel()
        before = len(human.runtime().world.action_log)

        worker._work("Assign the first task to someone")

        state = worker.state()
        assert state["drafts"] == []
        assert state["messages"][-1]["kind"] == "clarification"
        assert "Visible owner candidates:" in state["messages"][-1]["text"]
        assert "owner_id" not in state["messages"][-1]["text"]
        assert len(human.runtime().world.action_log) == before
    finally:
        human.shutdown()


def test_visible_event_feedback_is_opaque_and_only_terminal_task_status_is_completed():
    from environments.org_env.backend.entities.work import TaskStatus

    facade, human, _session, token = _facade()
    try:
        runtime = human.runtime()
        world = runtime.world
        task = next(iter(world.tasks.values()))
        task.status = TaskStatus.IN_PROGRESS
        world.events.append({"type": "task_progress_event", "subtype": "assigned",
                             "agent_id": SEAT, "task_id": task.task_id})
        world.events.append({"type": "reflection_event", "subtype": "reflected",
                             "agent_id": SEAT, "object_id": "reflection_private_secret"})
        world.events.append({"type": "meeting_event", "subtype": "closed",
                             "participants": [SEAT], "meeting_id": "meeting_internal_secret"})
        runtime.refresh_views()
        activity = facade.state(token)
        encoded = json.dumps(activity, default=str)
        assert str(task.title) in encoded
        assert "in_progress" in encoded
        assert "Completed:" not in encoded
        assert "reflection_private_secret" not in encoded
        assert "meeting_internal_secret" not in encoded

        reflection = next(row for row in activity["notifications"]
                          if row["kind"] == "reflection_cycle")
        trace = facade.trace(token, reflection["evidence_refs"][0])
        assert "reflection_private_secret" not in json.dumps(trace, default=str)
        assert trace["trace"]["steps"][0]["actor"] == "Victor Miracle"
        assert "reflection cycle" in trace["trace"]["steps"][0]["summary"]
        assert "Private reflection content is not exposed" in trace["trace"]["steps"][1]["summary"]

        task.status = TaskStatus.DONE
        world.events.append({"type": "task_progress_event", "subtype": "update",
                             "agent_id": SEAT, "task_id": task.task_id})
        runtime.refresh_views()
        completed = facade.state(token)
        assert any(row["summary"].startswith("Completed:")
                   for row in completed["notifications"])
        assert any(message["text"].startswith("Completed:")
                   for message in completed["conversation"])

        # Later events about a completed task do not replay completion, and
        # the secretary's feedback remains available on a later poll.
        world.events.append({"type": "task_progress_event", "subtype": "observed",
                             "agent_id": SEAT, "task_id": task.task_id})
        runtime.refresh_views()
        later = facade.state(token)
        completions = [row for row in later["notifications"]
                       if row["summary"].startswith("Completed:")]
        assert len(completions) == 1
        assert completions[0]["id"].startswith("nt_")
        assert completions[0]["evidence_refs"][0].startswith("ev_")
        assert isinstance(completions[0]["at"], float)
    finally:
        human.shutdown()


def test_notification_baseline_is_compact_and_semantic_dedup_prevents_reflood():
    facade, human, _session, token = _facade()
    try:
        runtime = human.runtime()
        world = runtime.world
        # A critical release block must survive the compact baseline even when
        # many later, lower-priority events exist.
        world.events.append({"type": "release_event", "subtype": "readiness_check",
                             "agent_id": SEAT, "candidate_id": "rc_priority",
                             "status": "blocked", "tick": 200,
                             "blockers": ["gate_ci", "gate_evidence"]})
        for index in range(12):
            world.events.append({"type": "funding_event", "subtype": "grant_full",
                                 "tranche_id": f"tranche_{index}", "amount": 1000 + index,
                                 "tick": 201 + index})
        runtime.refresh_views()
        # Initial P3 entry summarizes a baseline, it does not turn the whole
        # existing feed into a 60-message secretary transcript.
        initial = facade.state(token)
        assert len(initial["notifications"]) <= 8
        assert any(row["kind"] == "release_readiness"
                   and "blocked by 2 gates" in row["summary"]
                   for row in initial["notifications"])

        task = next(iter(world.tasks.values()))
        for sequence in range(65):
            world.events.append({"type": "task_progress_event", "subtype": "activity",
                                 "agent_id": SEAT, "task_id": task.task_id,
                                 "sequence": sequence})
        runtime.refresh_views()
        after_burst = facade.state(token)
        # Different tick/event ids carrying the same task state are one fact,
        # not 65 cards in the liaison transcript.
        assert len(after_burst["notifications"]) <= len(initial["notifications"]) + 1
        ids = [row["id"] for row in after_burst["notifications"]]

        # The same P2 feed is not replayed just because history reached its
        # bounded length: the separate seen set owns delivery deduplication.
        after_poll = facade.state(token)
        assert [row["id"] for row in after_poll["notifications"]] == ids
        assert len(after_poll["notifications"]) == len(after_burst["notifications"])
    finally:
        human.shutdown()


def test_traffic_watch_style_events_keep_signal_and_filter_tick_and_empty_noise():
    """P3 cards must expose a decision-relevant fact, not event plumbing."""
    facade, human, _session, token = _facade()
    try:
        runtime = human.runtime()
        world = runtime.world
        task = world.tasks["task_onboarding_doc"]
        task.owner_id = "paul"
        world.board.owners[task.task_id] = "paul"
        world.events.extend([
            {"type": "tick", "current_tick": 301},
            {"type": "background_job", "subtype": "finished", "tick": 301},
            {"type": "release_event", "subtype": "readiness_check", "agent_id": SEAT,
             "candidate_id": "rc_traffic", "status": "blocked", "tick": 302,
             "blockers": ["gate_ci", "gate_tests", "gate_evidence", "gate_approval"]},
            {"type": "release_event", "subtype": "blocker_to_task", "agent_id": SEAT,
             "candidate_id": "rc_traffic", "task_id": task.task_id, "owner_id": "paul",
             "gate": "gate_ci", "tick": 302},
            {"type": "product_event", "subtype": "dogfood", "agent_id": SEAT, "tick": 303,
             "query": "find route incidents", "outcome": "broken_output",
             "finding": "response omitted the incident owner"},
            {"type": "funding_event", "subtype": "grant_full", "tranche_id": "tranche_2",
             "amount": 125000, "condition": "traffic reliability gate", "tick": 303},
            {"type": "pull_request_event", "subtype": "public_tests_failed",
             "agent_id": SEAT, "pr_id": "pr_17", "tick": 303},
            {"type": "governance_event", "subtype": "wish_clustering", "open_wishes": 3,
             "themes": 2, "tick": 303},
            {"type": "organization_event", "subtype": "changed", "tick": 304},
        ])
        runtime.refresh_views()

        state = facade.state(token)
        summaries = [row["summary"] for row in state["notifications"]]
        encoded = json.dumps(state, default=str)
        assert any("blocked by 4 gates" in text for text in summaries)
        assert not any(row["kind"] == "release_blocker_task" for row in state["notifications"])
        assert any("dogfooded" in text and "find route incidents" in text and "broken output" in text
                   for text in summaries)
        assert not any("Funding decision" in text for text in summaries)
        assert not any(row["kind"] == "funding_decision" for row in state["notifications"])
        assert any("Public tests failed for pull request pr_17." == text
                   for text in summaries)
        assert not any("visible work; status: activity" in text for text in summaries)
        assert any("clustered 3 open wishes into 2 themes" in text for text in summaries)
        assert '"type": "tick"' not in encoded
        assert "The organization recorded a visible update" not in encoded
        assert not any(row["kind"] == "organization_event" for row in state["notifications"])

        readiness = next(row for row in state["notifications"] if row["kind"] == "release_readiness")
        evidence = facade.evidence(token, readiness["evidence_refs"][0])["evidence"]
        trace = facade.trace(token, readiness["evidence_refs"][0])["trace"]
        assert evidence["event"]["subtype"] == "readiness_check"
        assert evidence["event"]["candidate_id"] == "rc_traffic"
        assert evidence["event_summary"]["actor"] == "Victor Miracle"
        assert evidence["event_summary"]["organization_tick"] == 302
        assert "gate ci" in evidence["why"]
        assert "Paul Dreamer" in evidence["why"]
        assert "Create customer-facing onboarding doc" in evidence["why"]
        assert trace["steps"][0]["organization_tick"] == 302
        assert trace["steps"][1]["summary"] != trace["steps"][0]["summary"]
        assert next(row for row in state["conversation"]
                    if row.get("kind") == "event" and "blocked by 4 gates" in row["text"])["tick"] == 302
        assert facade._actor_name(runtime.seat_view(SEAT), {
            "type": "proposal_event", "subtype": "approved", "agent_id": SEAT,
            "approver": "calvin",
        }) == "Calvin Jacobi"
    finally:
        human.shutdown()


def test_pr_event_projection_never_turns_a_reviewer_into_the_author_or_refloods():
    facade, human, _session, _token = _facade(ticks=2)
    try:
        event = {"type": "repo_event", "subtype": "pr_reviewed",
                 "pr_id": "pr_35", "agent_id": "victor", "tick": 35}
        view = {
            "seat": {"agent_id": "victor"},
            "member": {"members": [
                {"agent_id": "victor", "name": "Victor Miracle"},
                {"agent_id": "paul", "name": "Paul Dreamer"},
            ]},
            "objects": {"pull_requests": [{
                "id": "pr_35", "kind": "pull_request", "title": "pr_35",
                "author": "paul", "status": "approved",
                "reviewers": ["victor"], "approved_by": ["victor"],
            }]},
            "feed": {"events": [event]},
        }
        _source, item = facade._event_source(event, view)
        assert item["title"] == "Victor Miracle reviewed Paul Dreamer's pull request pr_35."
        assert item["status"] == "reviewed"
        assert "PR author: Paul Dreamer" in item["why"]
        assert "event actor: Victor Miracle" in item["why"]

        first = facade._notifications("pr-role-token", view)
        assert len(first) == 1
        view["objects"]["pull_requests"][0]["status"] = "merged"
        second = facade._notifications("pr-role-token", view)
        assert [row["id"] for row in second] == [row["id"] for row in first]
        assert len(second) == 1
        assert second[0]["status"] == "reviewed"
    finally:
        human.shutdown()


def test_replying_to_an_event_card_passes_that_card_and_team_state_to_one_model_call():
    facade, human, _session, token = _facade(ticks=8)
    captured = {}

    def responder(_system, user, _schema):
        captured.update(json.loads(user))
        return {"route": "grounded_answer",
                "reply": "这条卡片记录的是一次组织事件；我已按作者、reviewer 和当前状态分别核对。"}

    try:
        state = facade.state(token)
        card = next(row for row in state["conversation"]
                    if row.get("kind") == "event" and row.get("id"))
        human._agent(SEAT).llm = MockOrgLLMClient(responder=responder)

        out = facade.ask(token, "这是什么？", reply_to=card["id"])
        assert out["handled"] == "grounded_answer"
        assert captured["conversation_scope"] == card["id"]
        assert captured["recent_dialogue"][0]["text"] == card["summary"]
        assert captured["project_context"]["product_name"]
        assert captured["runtime_context"]["world_tick"] == 8
        assert captured["runtime_context"]["running"] is False
        assert captured["runtime_context"]["engine"] == "org_env.OrgWorld"
        assert "pack" in captured["runtime_context"]
        assert captured["team_progress"]["people"]
        assert "recent_activity" in captured

        after = facade.state(token)
        replies = [row for row in after["conversation"]
                   if row.get("thread_id") == card["id"]]
        assert [row["role"] for row in replies] == ["human", "liaison"]
        assert replies[0]["parent_id"] == card["id"]
        thread = next(row for row in after["conversation_threads"]
                      if row["root_message_id"] == card["id"])
        assert thread["root_text"] == card["summary"]
    finally:
        human.shutdown()


def test_router_provider_failure_preserves_the_human_turn_without_a_text_fallback():
    facade, human, _session, token = _facade(ticks=2)

    class FailingModel:
        def generate_json(self, *_args, **_kwargs):
            raise RuntimeError("temporary upstream failure")

    try:
        human._agent(SEAT).llm = FailingModel()
        out = facade.ask(token, "把这条消息保留下来")
        assert out["accepted"] is True
        assert out["handled"] == "model_error"
        assert out["retryable"] is True
        assert "error" not in out

        messages = facade.state(token)["conversation"]
        assert any(row["role"] == "human" and row["text"] == "把这条消息保留下来"
                   for row in messages)
        failure = next(row for row in messages if row.get("kind") == "model_error")
        assert "原消息已经保留" in failure["text"]
        assert "正则或模板" in failure["text"]
        failed_request = facade.state(token)["failed_request"]
        assert failed_request["original_request"] == "把这条消息保留下来"
        assert failed_request["next_step"] == 1

        human._agent(SEAT).llm = MockOrgLLMClient(script=[
            {"route": "grounded_answer", "reply": "已从原消息重新完成解释。"},
        ])
        retried = facade.retry_request(token)
        assert retried["handled"] == "grounded_answer"
        state = facade.state(token)
        assert state["failed_request"] is None
        assert sum(row["role"] == "human" and row["text"] == "把这条消息保留下来"
                   for row in state["conversation"]) == 1
        assert state["conversation"][-1]["text"] == "已从原消息重新完成解释。"
        assert not human._agent(SEAT).state()["drafts"]
    finally:
        human.shutdown()


def test_router_retries_transient_model_failures_before_showing_recovery_ui():
    facade, human, _session, token = _facade(ticks=2)

    class TransientlyFailingModel:
        def __init__(self):
            self.calls = 0

        def generate_json(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporary malformed response")
            return {"route": "grounded_answer", "reply": "第三次模型调用完成了解释。"}

    try:
        model = TransientlyFailingModel()
        human._agent(SEAT).llm = model

        out = facade.ask(token, "现在项目进度怎样？")

        assert model.calls == 3
        assert out["handled"] == "grounded_answer"
        state = facade.state(token)
        assert state["conversation"][-1]["text"] == "第三次模型调用完成了解释。"
        assert state["failed_request"] is None
        assert not any(row.get("kind") == "model_error"
                       for row in state["conversation"])
    finally:
        human.shutdown()


def test_baseline_collapses_visible_proposal_history_and_paired_release_tasks():
    """Initial P3 entry prioritizes current decisions over event-transition spam."""
    facade, human, _session, _token = _facade()
    try:
        view = {
            "member": {"members": [
                {"agent_id": "victor", "name": "Victor Miracle", "role": "founder"},
                {"agent_id": "paul", "name": "Paul Dreamer", "role": "engineer"},
                {"agent_id": "sean", "name": "Sean Lightning", "role": "engineer"},
                {"agent_id": "calvin", "name": "Calvin Jacobi", "role": "reviewer"},
            ]},
            "objects": {
                "tasks": [
                    {"id": "blocker_ci", "title": "Repair CI gate", "owner": "paul", "status": "open"},
                    {"id": "blocker_tests", "title": "Repair test gate", "owner": "sean", "status": "open"},
                ],
                "proposals": [{"id": "proposal_1", "title": "Adopt traffic watch rollout",
                               "status": "approved"}],
                "meetings": [{"id": "meeting_1", "title": "Release review", "status": "closed"}],
            },
            "feed": {"events": [
                {"type": "tick", "current_tick": 400},
                {"type": "release_event", "subtype": "readiness_check", "agent_id": "victor",
                 "candidate_id": "rc_1", "status": "blocked", "tick": 401,
                 "blockers": ["gate_ci", "gate_tests"]},
                {"type": "release_event", "subtype": "blocker_to_task", "agent_id": "victor",
                 "candidate_id": "rc_1", "task_id": "blocker_ci", "owner_id": "paul",
                 "gate": "gate_ci", "tick": 401},
                {"type": "release_event", "subtype": "blocker_to_task", "agent_id": "victor",
                 "candidate_id": "rc_1", "task_id": "blocker_tests", "owner_id": "sean",
                 "gate": "gate_tests", "tick": 401},
                {"type": "release_event", "subtype": "blocker_cleared", "agent_id": "victor",
                 "gate": "gate_docs", "tick": 402},
                {"type": "product_event", "subtype": "dogfood", "agent_id": "victor", "tick": 403,
                 "query": "route incident query", "outcome": "ok", "finding": "working report"},
                {"type": "reflection_event", "subtype": "reflected", "agent_id": "victor", "tick": 404},
                {"type": "proposal_event", "subtype": "created", "agent_id": "victor",
                 "proposal_id": "proposal_1", "tick": 404},
                {"type": "proposal_event", "subtype": "under_review", "agent_id": "victor",
                 "proposal_id": "proposal_1", "tick": 405},
                {"type": "proposal_event", "subtype": "approved", "agent_id": "victor",
                 "approver": "calvin", "proposal_id": "proposal_1", "tick": 406},
                {"type": "meeting_event", "subtype": "closed", "agent_id": "victor",
                 "meeting_id": "meeting_1", "tick": 407},
            ]},
        }
        cards = facade._notifications("baseline-collapse-token", view)
        summaries = [card["summary"] for card in cards]
        kinds = [card["kind"] for card in cards]
        assert len(cards) == 6
        assert {"release_readiness", "release_blocker_cleared", "product_dogfood",
                "reflection_cycle", "proposal_event", "meeting_event"} <= set(kinds)
        assert not any(kind == "release_blocker_task" for kind in kinds)
        assert any("Follow-ups: Paul Dreamer ×1, Sean Lightning ×1" in card["summary"]
                   for card in cards if card["kind"] == "release_readiness")
        proposal = [card for card in cards if card["kind"] == "proposal_event"]
        assert len(proposal) == 1
        assert proposal[0]["summary"].startswith("Calvin Jacobi approved proposal")
        assert not any("created proposal" in text or "under review proposal" in text for text in summaries)
    finally:
        human.shutdown()


def test_secretary_asks_before_a_meeting_and_allows_context_before_a_decision():
    facade, human, _session, token = _facade(ticks=2)
    try:
        runtime = human.runtime()
        world = runtime.world
        meeting = world.meeting_system.schedule_meeting(
            created_by="calvin", meeting_type="planning_meeting",
            title="Release boundary review", participants=[SEAT, "calvin"],
            scheduled_tick=world.world_tick + 1,
            agenda=["decide whether the current evidence is enough to release"],
        )
        runtime.refresh_views()
        human._agent(SEAT).llm = MockOrgLLMClient(script=[{
            "route": "meeting_context", "meeting_id": meeting.meeting_id,
            "reply": ("The agenda asks whether the current evidence is enough to release; "
                      "Calvin and Victor are participants."),
        }])

        state = facade.state(token)
        invitation = next(message for message in state["conversation"]
                          if message.get("kind") == "meeting_invitation")
        assert invitation["status"] == "decision_required"
        assert "before" in invitation["text"].lower()
        assert "Release boundary review" in invitation["text"]
        assert invitation["resource_ref"].startswith("ri_")
        assert "error" not in facade.resource(token, invitation["resource_ref"], "overview")
        root_context = human._agent(SEAT).conversation_context(reply_to=invitation["id"])
        root_reference = root_context["messages"][0]["references"][0]
        assert root_reference["object_id"] == meeting.meeting_id
        assert root_reference["kind"] == "meeting"
        assert state["meeting_invitations"][0]["decision"] == "pending"
        assert state["meeting_invitations"][0]["resource_ref"] == invitation["resource_ref"]
        assert not state["meeting_invitations"][0]["attendance_confirmed"]

        # A question is a question, not a disguised RSVP. The secretary gives
        # visible context and leaves the meeting pending.
        out = facade.ask(token, "这个会议为什么要开，议程和参与者是谁？")
        assert out["handled"] == "meeting_context"
        after_question = facade.state(token)
        answer = next(message for message in reversed(after_question["conversation"])
                      if message.get("kind") == "clarification")
        assert "current evidence is enough" in answer["text"]
        assert not after_question["pending_actions"]
        assert meeting.status.value == "scheduled"
    finally:
        human.shutdown()


def test_natural_language_neither_plan_prepares_a_real_skip_and_no_report_request():
    facade, human, _session, token = _facade(ticks=2)
    try:
        runtime = human.runtime()
        world = runtime.world
        meeting = world.meeting_system.schedule_meeting(
            created_by="calvin", meeting_type="planning_meeting", title="Planning sync",
            participants=[SEAT, "calvin"], scheduled_tick=world.world_tick + 1,
            agenda=["plan the next release"],
        )
        runtime.refresh_views()
        facade.state(token)
        human._agent(SEAT).llm = MockOrgLLMClient(script=[
            {"route": "clarification", "reply": (
                "你不参会时，我还需要知道是否转达观点、带回信息，或者两者都不要。")},
            {"route": "meeting_plan", "meeting_id": meeting.meeting_id,
             "decision": "skip", "viewpoint": "", "return_focus": "",
             "reply": "你不参会，秘书不表达观点，也不准备详细会后汇报。"},
        ])

        first = facade.ask(token, "我不去这个会")
        assert first["handled"] == "meeting_clarification"
        assert not facade.state(token)["pending_actions"]

        second = facade.ask(token, "我不参加，秘书也不用表达观点，不用带回信息，两者都不要")
        assert second["handled"] == "meeting_instruction"
        state = facade.state(token)
        assert not state["pending_actions"]
        assert len(state["meeting_plans"]) == 1
        assert state["meeting_invitations"][0]["decision"] == "skip"
        result = facade.confirm_meeting_plan(token, state["meeting_plans"][0]["plan_id"])
        assert result["ok"] is True
        assert SEAT in meeting.skipped_by
        assert SEAT not in meeting.attendees
    finally:
        human.shutdown()


def test_delegate_plan_relays_only_exact_viewpoint_then_records_requested_return_focus():
    facade, human, _session, token = _facade(ticks=2)
    try:
        runtime = human.runtime()
        world = runtime.world
        meeting = world.meeting_system.schedule_meeting(
            created_by="calvin", meeting_type="planning_meeting", title="Launch review",
            participants=[SEAT, "calvin"], scheduled_tick=world.world_tick + 1,
            agenda=["decide launch readiness"],
        )
        runtime.refresh_views()
        facade.state(token)
        human._agent(SEAT).llm = MockOrgLLMClient(script=[{
            "route": "meeting_plan", "meeting_id": meeting.meeting_id,
            "decision": "delegate", "viewpoint": "证据不够，不能上线",
            "return_focus": "CI 和最终决定",
            "reply": "我会转达你的原话，并只带回 CI 和最终决定。",
        }])

        out = facade.ask(token, "我不去，你替我表达证据不够，不能上线，回来告诉我 CI 和最终决定")
        assert out["handled"] == "meeting_instruction"
        state = facade.state(token)
        assert not state["pending_actions"]
        assert len(state["meeting_plans"]) == 1
        plan = state["meeting_plans"][0]
        assert plan["viewpoint"] == "证据不够，不能上线"
        assert state["meeting_invitations"][0]["return_focus"] == "CI 和最终决定"
        visible_text = "\n".join(message["text"] for message in state["conversation"])
        assert "Prepared:" not in visible_text
        assert "send_message" not in visible_text
        assert "skip_meeting" not in visible_text

        # One semantic confirmation applies the exact viewpoint first and then
        # the absence RSVP; component draft ordering never leaks to P3.
        assert SEAT not in meeting.skipped_by
        confirmed = facade.confirm_meeting_plan(token, plan["plan_id"])
        assert confirmed["ok"] is True
        assert confirmed["viewpoint_relayed"] is True
        relayed = next(message for message in world.comm.messages.values()
                       if message.full_text == "证据不够，不能上线")
        assert relayed.sender_id == SEAT
        assert relayed.full_text == "证据不够，不能上线"
        assert SEAT in meeting.skipped_by
        assert world.controller_log[-1]["execution_mode"] == "liaison_assisted"

        # Close with an ordinary public outcome. The secretary reports only
        # the visible record and the return focus the human asked for.
        world.meeting_system.start_meeting(meeting.meeting_id, world.world_tick + 1)
        world.meeting_system.attend("calvin", meeting.meeting_id)
        world.meeting_system.record_meeting_notes(
            agent_id="calvin", meeting_id=meeting.meeting_id,
            summary="CI passed after rerun.", decisions=["Hold launch for evidence review"],
            unresolved=["Who signs the evidence bundle?"], tick=world.world_tick + 1,
        )
        world.meeting_system.close_meeting(meeting.meeting_id, world.world_tick + 2)
        runtime.refresh_views()
        report = next(message for message in facade.state(token)["conversation"]
                      if message.get("kind") == "meeting_report")
        assert "CI 和最终决定" in report["text"]
        assert "CI passed after rerun" in report["text"]
        assert "Hold launch for evidence review" in report["text"]
        assert "Your confirmed viewpoint was relayed" in report["text"]
        assert report["resource_ref"].startswith("ri_")
    finally:
        human.shutdown()


def test_meeting_plan_preflights_every_component_before_relaying_anything():
    facade, human, _session, token = _facade(ticks=2)
    try:
        runtime = human.runtime()
        world = runtime.world
        meeting = world.meeting_system.schedule_meeting(
            created_by="calvin", meeting_type="planning_meeting", title="Atomic RSVP",
            participants=[SEAT, "calvin"], scheduled_tick=world.world_tick + 1,
            agenda=["review evidence"],
        )
        runtime.refresh_views()
        facade.state(token)
        human._agent(SEAT).llm = MockOrgLLMClient(script=[{
            "route": "meeting_plan", "meeting_id": meeting.meeting_id,
            "decision": "delegate", "viewpoint": "证据不足",
            "return_focus": "最终决定", "reply": "会议安排已准备好。",
        }])
        assert facade.ask(
            token, "我不参会，替我表达证据不足，回来告诉我最终决定"
        )["handled"] == "meeting_instruction"
        plan_id = facade.state(token)["meeting_plans"][0]["plan_id"]
        before_messages = len(world.comm.messages)

        # The RSVP becomes invalid between preparation and confirmation. The
        # viewpoint must not be sent as a partial first step.
        meeting.attendees.append(SEAT)
        runtime.refresh_views()
        result = facade.confirm_meeting_plan(token, plan_id)
        assert result["error"] == "already_attending"
        assert result["completed_steps"] == []
        assert len(world.comm.messages) == before_messages
    finally:
        human.shutdown()


def test_delegate_plan_understands_natural_return_focus_with_only_adverb():
    facade, human, _session, token = _facade(ticks=2)
    try:
        runtime = human.runtime()
        world = runtime.world
        meeting = world.meeting_system.schedule_meeting(
            created_by="paul", meeting_type="release_gate_review",
            title="Release evidence decision", participants=["paul", SEAT],
            scheduled_tick=world.world_tick + 1,
            agenda=["decide whether reproducible evidence is sufficient"],
        )
        runtime.refresh_views()
        facade.state(token)
        human._agent(SEAT).llm = MockOrgLLMClient(script=[{
            "route": "meeting_plan", "meeting_id": meeting.meeting_id,
            "decision": "delegate",
            "viewpoint": "没有可复现证据就不要解除发布门禁",
            "return_focus": "会议最后决定了什么、谁负责补证据",
            "reply": "我会按你的原话转达，并带回指定信息。",
        }])
        out = facade.ask(
            token,
            "我不参会。请替我表达：没有可复现证据就不要解除发布门禁。"
            "回来只告诉我会议最后决定了什么、谁负责补证据。",
        )
        assert out["handled"] == "meeting_instruction"
        state = facade.state(token)
        assert len(state["meeting_invitations"]) == 1
        plan = state["meeting_invitations"][0]
        assert plan["viewpoint"] == "没有可复现证据就不要解除发布门禁"
        assert plan["return_focus"] == "会议最后决定了什么、谁负责补证据"
    finally:
        human.shutdown()


def test_discard_meeting_plan_cancels_all_internal_drafts_as_one_choice():
    facade, human, _session, token = _facade(ticks=2)
    try:
        runtime = human.runtime()
        world = runtime.world
        meeting = world.meeting_system.schedule_meeting(
            created_by="calvin", meeting_type="planning_meeting", title="Evidence sync",
            participants=[SEAT, "calvin"], scheduled_tick=world.world_tick + 1,
            agenda=["review evidence"],
        )
        runtime.refresh_views()
        facade.state(token)
        human._agent(SEAT).llm = MockOrgLLMClient(script=[{
            "route": "meeting_plan", "meeting_id": meeting.meeting_id,
            "decision": "delegate", "viewpoint": "证据不足",
            "return_focus": "最终决定", "reply": "会议安排已准备好。",
        }])
        assert facade.ask(
            token, "我不参会，替我表达证据不足，回来告诉我最终决定"
        )["handled"] == "meeting_instruction"
        state = facade.state(token)
        plan_id = state["meeting_plans"][0]["plan_id"]
        result = facade.discard_meeting_plan(token, plan_id)
        assert result["ok"] is True
        assert result["discarded_count"] == 2
        after = facade.state(token)
        assert not after["meeting_plans"]
        assert after["meeting_invitations"][0]["decision"] == "pending"
        assert SEAT not in meeting.skipped_by
        assert not human._agent(SEAT).state()["drafts"]
    finally:
        human.shutdown()


def test_secretary_triages_direct_messages_and_natural_language_can_show_everything():
    facade, human, _session, token = _facade(ticks=2)
    try:
        runtime = human.runtime()
        world = runtime.world
        ordinary = world.comm.send_message(
            sender_id="calvin", channel_id="team_general", text="Lunch is at noon.",
            tick=world.world_tick, mentions=[SEAT], importance="trivial", urgency="low")
        urgent = world.comm.send_message(
            sender_id="paul", channel_id="team_general",
            text="Release is blocked; Victor must decide the rollback.",
            tick=world.world_tick, mentions=[SEAT], importance="blocker", urgency="urgent")
        runtime.refresh_views()

        triaged = facade.state(token)
        delivered = [message for message in triaged["conversation"]
                     if message.get("kind") == "incoming_message"]
        assert any(urgent.full_text in message["text"] for message in delivered)
        assert not any(ordinary.full_text in message["text"] for message in delivered)
        assert triaged["attention_preferences"]["message_delivery"] == "secretary_triage"
        important = next(message for message in delivered if urgent.full_text in message["text"])
        assert facade.evidence(token, important["evidence_refs"][0])["evidence"]["source"]["kind"] == "message"

        human._agent(SEAT).llm = MockOrgLLMClient(script=[
            {"route": "attention_preference", "message_delivery": "all_messages",
             "reply": "以后所有可见消息都交给你亲自查看。"},
            {"route": "attention_preference", "message_delivery": "secretary_triage",
             "reply": "已恢复秘书筛选，只立即转达重要消息。"},
        ])
        changed = facade.ask(token, "以后我要亲自看所有消息，不要替我筛选")
        assert changed["handled"] == "attention_preference"
        all_state = facade.state(token)
        assert all_state["attention_preferences"]["message_delivery"] == "all_messages"
        assert any(ordinary.full_text in message["text"]
                   for message in all_state["conversation"]
                   if message.get("kind") == "incoming_message")

        restored = facade.ask(token, "恢复筛选，只告诉我重要消息")
        assert restored["handled"] == "attention_preference"
        assert facade.state(token)["attention_preferences"]["message_delivery"] == "secretary_triage"
    finally:
        human.shutdown()


def test_human_related_events_reach_secretary_after_the_live_feed_rolls_over():
    facade, human, _session, token = _facade(ticks=2)
    try:
        runtime = human.runtime()
        world = runtime.world
        world.events.append({
            "type": "ownership_event", "subtype": "assigned",
            "assignee_id": SEAT, "agent_id": "calvin",
            "participants": [SEAT], "title": "Review the release evidence",
            "status": "open", "tick": 500,
        })
        runtime.refresh_views()
        human._capture_visible_delivery()

        # More than the P2 feed limit passes before the browser polls again.
        # P3 reads the bounded, seat-filtered delivery journal rather than
        # widening the public P2 feed or reaching into the world directly.
        world.events.extend({"type": "tick", "current_tick": 501 + index}
                            for index in range(140))
        runtime.refresh_views()
        human._capture_visible_delivery()
        assert not any(event.get("type") == "ownership_event"
                       for event in runtime.seat_view(SEAT)["feed"]["events"])

        state = facade.state(token)
        routed = [row for row in state["notifications"]
                  if row["kind"] == "human_related_event"]
        assert len(routed) == 1
        assert "Calvin Jacobi assigned work involving Victor" in routed[0]["summary"]
        assert routed[0]["status"] == "open"
    finally:
        human.shutdown()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("  ok ", fn.__name__)
    print(f"All {len(fns)} liaison facade tests passed!")
