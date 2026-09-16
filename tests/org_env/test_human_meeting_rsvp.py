"""Human-seat RSVP behaviour for the shared meeting lifecycle."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_sdk.lived.core.contracts import ActionCandidate
from environments.org_env.human import gateway
from environments.org_env.human.runtime import HumanModeRuntime
from environments.org_env.human.seat_view import build_seat_view
from environments.org_env.runtime_adapter.live import OrgInspectorSession


HUMAN = "victor"


def _runtime():
    session = OrgInspectorSession(seed=42)
    session.step(4)
    return HumanModeRuntime(session.world, seconds_per_tick=60.0)


def _scheduled_meeting(world, participants=None, scheduled_tick=None):
    tick = int(world.world_tick) + 1 if scheduled_tick is None else scheduled_tick
    return world.meeting_system.schedule_meeting(
        created_by="paul", meeting_type="planning_meeting", title="release plan",
        participants=participants or ["paul", HUMAN, "calvin"],
        scheduled_tick=tick, agenda=["review release blocker"],
    )


def test_human_participant_holds_only_its_meeting_until_an_rsvp():
    runtime = _runtime()
    try:
        world = runtime.world
        runtime.claim_seat(HUMAN)
        due = int(world.world_tick) + 1
        held = _scheduled_meeting(world, scheduled_tick=due)
        unrelated = _scheduled_meeting(world, participants=["paul", "calvin"], scheduled_tick=due)

        world.world_tick = due
        world.time.clock.current_tick = due
        world._process_meetings(due)

        assert held.status.value == "scheduled"
        assert HUMAN not in held.attendees
        assert HUMAN not in held.skipped_by
        assert not world.agents[HUMAN].work_state.is_busy(due)
        # The RSVP wait is local to this meeting, not a global organization pause.
        assert unrelated.status.value == "active"
    finally:
        runtime.shutdown()


def test_confirmed_human_attend_starts_and_blocks_the_member_once_due():
    runtime = _runtime()
    try:
        world = runtime.world
        runtime.claim_seat(HUMAN)
        due = int(world.world_tick) + 1
        meeting = _scheduled_meeting(world, scheduled_tick=due)

        response = gateway.submit(runtime, HUMAN, "attend_meeting", {"meeting_id": meeting.meeting_id},
                                  execution_mode="liaison_assisted")
        assert response.success, response.failure_reason
        assert HUMAN in meeting.attendees
        assert world.controller_log[-1]["controller_type"] == "human"
        assert world.controller_log[-1]["execution_mode"] == "liaison_assisted"
        assert any(event.get("subtype") == "rsvp_attending" and event.get("meeting_id") == meeting.meeting_id
                   for event in world.events)

        world.world_tick = due
        world.time.clock.current_tick = due
        world._process_meetings(due)

        assert meeting.status.value == "active"
        assert HUMAN in meeting.attendees
        assert world.agents[HUMAN].work_state.is_busy(due)
        # The RSVP action was already logged through the shared action pipeline;
        # starting the meeting must not log a second human RSVP.
        human_attends = [row for row in world.action_log
                         if row["agent_id"] == HUMAN and row["action_type"] == "attend_meeting"]
        assert len(human_attends) == 1
    finally:
        runtime.shutdown()


def test_confirmed_skip_starts_without_the_human_and_never_blocks_them():
    runtime = _runtime()
    try:
        world = runtime.world
        runtime.claim_seat(HUMAN)
        due = int(world.world_tick) + 1
        meeting = _scheduled_meeting(world, scheduled_tick=due)

        response = gateway.submit(runtime, HUMAN, "skip_meeting", {"meeting_id": meeting.meeting_id},
                                  execution_mode="liaison_assisted")
        assert response.success, response.failure_reason
        assert HUMAN in meeting.skipped_by
        assert HUMAN not in meeting.attendees

        world.world_tick = due
        world.time.clock.current_tick = due
        world._process_meetings(due)

        assert meeting.status.value == "active"
        assert HUMAN not in meeting.attendees
        assert not world.agents[HUMAN].work_state.is_busy(due)
        assert any(event.get("subtype") == "rsvp_skipped" and event.get("meeting_id") == meeting.meeting_id
                   for event in world.events)
    finally:
        runtime.shutdown()


def test_human_attendee_does_not_run_autonomous_meeting_subactions():
    runtime = _runtime()
    try:
        world = runtime.world
        runtime.claim_seat(HUMAN)
        due = int(world.world_tick) + 1
        meeting = _scheduled_meeting(world, participants=[HUMAN], scheduled_tick=due)
        assert gateway.submit(runtime, HUMAN, "attend_meeting", {"meeting_id": meeting.meeting_id}).success

        # ``step`` runs both the scheduled-meeting lifecycle and the normal
        # per-agent meeting subaction loop.  Victor is busy in the meeting, so
        # this is the regression path that used to auto-write meeting notes.
        world.step()
        assert meeting.status.value == "active"
        ws = world.agents[HUMAN].work_state
        assert ws.current_meeting_id == meeting.meeting_id
        assert meeting.notes_doc_id is None
    finally:
        runtime.shutdown()


def test_rsvp_is_revalidated_and_seat_view_projects_public_meeting_details():
    runtime = _runtime()
    try:
        world = runtime.world
        runtime.claim_seat(HUMAN)
        meeting = _scheduled_meeting(world)

        assert gateway.submit(runtime, HUMAN, "skip_meeting", {"meeting_id": meeting.meeting_id}).success
        try:
            gateway.validate(world, HUMAN, "attend_meeting", {"meeting_id": meeting.meeting_id})
        except gateway.ActionRefused as exc:
            assert str(exc) == "already_skipped"
        else:
            raise AssertionError("gateway accepted a contradictory RSVP")
        refused = world.apply_action_candidate(
            HUMAN, ActionCandidate(action_type="attend_meeting", parameters={"meeting_id": meeting.meeting_id}),
            controller_type="human", execution_mode="liaison_assisted")
        assert not refused.success
        assert refused.failure_reason == "already_skipped"

        meeting.notes_doc_id = "note_visible"
        note = world.meeting_system.record_meeting_notes(
            agent_id="paul", meeting_id=meeting.meeting_id,
            summary="Keep the release blocked until CI is green.",
            decisions=["Run CI before release."], unresolved=["Who owns the retry?"], tick=world.world_tick)
        action = world.meeting_system.action_items[meeting.action_item_ids[-1]]
        action.assignee_id = "calvin"
        action.linked_task_id = "task_onboarding_doc"
        action.status = "open"
        view = build_seat_view(world, HUMAN)
        projected = next(row for row in view["objects"]["meetings"] if row["id"] == meeting.meeting_id)

        assert projected["declined"] is True
        assert projected["agenda"] == ["review release blocker"]
        assert projected["room_channel"] == meeting.room_channel_id
        assert projected["notes_summary"] == note.summary
        assert projected["decision_summaries"] == ["Run CI before release."]
        assert projected["unresolved_questions"] == ["Who owns the retry?"]
        assert projected["action_items"][-1]["assignee_id"] == "calvin"
        assert projected["action_items"][-1]["linked_task_id"] == "task_onboarding_doc"
    finally:
        runtime.shutdown()


def test_send_message_forwards_targeting_and_safe_message_projection():
    runtime = _runtime()
    try:
        world = runtime.world
        runtime.claim_seat(HUMAN)
        result = gateway.submit(runtime, HUMAN, "send_message", {
            "channel_id": "team_general",
            "text": "@Calvin please review the release evidence.",
            "mentions": ["calvin"],
            "importance": "decision_relevant",
            "urgency": "urgent",
        })
        assert result.success, result.failure_reason
        message = world.comm.messages[result.created_objects[0]]
        assert message.mentions == ["calvin"]
        assert message.importance == "decision_relevant"
        assert message.urgency == "urgent"

        view = build_seat_view(world, HUMAN)
        projected = next(row for row in view["feed"]["threads"]
                         for row in row["messages"] if row["id"] == message.message_id)
        assert projected["recipients"] == message.recipients
        assert projected["dm_id"] is None
        assert projected["mentions"] == ["calvin"]
    finally:
        runtime.shutdown()
