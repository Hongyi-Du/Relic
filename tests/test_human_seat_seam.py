from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from environments.org_env.backend.meetings.meeting import Meeting, MeetingStatus
from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.config.scenarios import oss_time_machine
from environments.org_env.product.substrates import loader
from relic.benchmark import benchmark_directory
from relic.decision.contracts import ActionCandidate


HUMAN = "sean"


def _world(*, interaction_profile: str | None = None, dataset_id: str = "mini_blobstore_v1") -> OrgWorld:
    scenario = oss_time_machine(
        seed=17,
        dataset_id=dataset_id,
        interaction_profile=interaction_profile,
    )
    return OrgWorld(scenario).build()


def _message(world: OrgWorld, agent_id: str, *, controller_type: str = "agent",
             execution_mode: str = "direct"):
    return world.apply_action_candidate(
        agent_id,
        ActionCandidate(
            action_type="send_message",
            parameters={"channel_id": "team_general", "text": "status update"},
        ),
        controller_type=controller_type,
        execution_mode=execution_mode,
    )


def test_human_seat_shares_the_action_pipeline_without_public_controller_state() -> None:
    world = _world()
    before = world.agents[HUMAN]
    seat_shape = (before.role, dict(before.skills), list(before.permissions))

    assert world.human_seat_ids() == []
    assert all(not world.is_human_controlled(agent_id) for agent_id in world.agents)

    world.assign_human_seat(HUMAN)
    assert world.is_human_controlled(HUMAN)
    assert world.human_seat_ids() == [HUMAN]
    assert (before.role, before.skills, before.permissions) == seat_shape

    action_start = len(world.action_log)
    agent_result = _message(world, "calvin")
    human_result = _message(
        world,
        HUMAN,
        controller_type="human",
        execution_mode="working_agent_assisted",
    )

    action_entries = world.action_log[action_start:]
    assert len(action_entries) == 2
    assert set(action_entries[0]) == set(action_entries[1])
    assert all("controller" not in key for entry in action_entries for key in entry)
    assert agent_result.action_type == human_result.action_type == "send_message"
    assert all(
        "controller_type" not in event and "execution_mode" not in event
        for event in world.events
    )
    assert len(world.controller_log) == 1
    controller_entry = world.controller_log[0]
    assert controller_entry["agent_id"] == HUMAN
    assert controller_entry["action_type"] == "send_message"
    assert controller_entry["action_id"] == human_result.action_id
    assert controller_entry["tick"] == world.world_tick
    assert controller_entry["wall_time"] > 0
    assert controller_entry["controller_type"] == "human"
    assert controller_entry["execution_mode"] == "working_agent_assisted"
    assert controller_entry["success"] is human_result.success

    visible_state = json.dumps(asdict(world.get_state()), default=str)
    assert "controller_type" not in visible_state
    assert "controller_log" not in visible_state

    world.release_human_seat(HUMAN)
    assert not world.is_human_controlled(HUMAN)


def test_human_seat_skips_autonomous_inbox_choice_and_meeting_subaction(monkeypatch) -> None:
    world = _world()
    world.assign_human_seat(HUMAN)
    inbox_calls: list[str] = []
    original_inbox = world._process_inbox

    def record_inbox(agent_id: str, tick: int) -> None:
        inbox_calls.append(agent_id)
        original_inbox(agent_id, tick)

    monkeypatch.setattr(world, "_process_inbox", record_inbox)
    monkeypatch.setattr(world, "can_agent_act", lambda _agent, _clock: True)
    world.step()

    assert HUMAN not in inbox_calls
    assert any(agent_id != HUMAN for agent_id in inbox_calls)

    meeting = Meeting(
        meeting_id="meeting_human_seat",
        participants=[HUMAN],
        status=MeetingStatus.ACTIVE,
        start_tick=world.world_tick + 1,
    )
    world.meeting_system.meetings[meeting.meeting_id] = meeting
    work_state = world.agents[HUMAN].work_state
    work_state.current_meeting_id = meeting.meeting_id
    work_state.next_available_tick = world.world_tick + 10
    subactions: list[str] = []
    monkeypatch.setattr(
        world,
        "_run_meeting_subaction",
        lambda agent_id, *_args: subactions.append(agent_id),
    )

    world.step()
    assert subactions == []


def test_human_seat_must_rsvp_before_a_scheduled_meeting_starts() -> None:
    world = _world()
    world.assign_human_seat(HUMAN)
    meeting = Meeting(
        meeting_id="meeting_human_rsvp",
        participants=[HUMAN, "calvin"],
        scheduled_tick=world.world_tick,
    )
    world.meeting_system.meetings[meeting.meeting_id] = meeting

    world._process_meetings(world.world_tick)
    assert meeting.status is MeetingStatus.SCHEDULED
    assert HUMAN not in meeting.attendees
    assert not [
        entry for entry in world.action_log
        if entry["agent_id"] == HUMAN and entry["action_type"] == "attend_meeting"
    ]

    # A human gateway records an RSVP before the ordinary lifecycle starts
    # the meeting. The common world seam must then preserve that decision
    # instead of adding a second automatic attendance action.
    assert world.meeting_system.attend(HUMAN, meeting.meeting_id)
    world._process_meetings(world.world_tick)
    assert meeting.status is MeetingStatus.ACTIVE
    assert HUMAN in meeting.attendees
    assert not [
        entry for entry in world.action_log
        if entry["agent_id"] == HUMAN and entry["action_type"] == "attend_meeting"
    ]


def test_human_owned_ready_branch_is_not_auto_submitted() -> None:
    world = _world()
    repo = world.repo_system
    branch = repo.create_branch(HUMAN, tick=0)
    assert repo.edit_file(HUMAN, branch.branch_id)
    assert repo.commit_changes(
        agent_id=HUMAN,
        branch_id=branch.branch_id,
        message="human work",
        changed_files=["src/example.py"],
        tick=0,
    ) is not None
    world.world_tick = 3

    world.assign_human_seat(HUMAN)
    assert world.process_repo_workflow() == []
    assert repo.repo.pull_requests == {}

    world.release_human_seat(HUMAN)
    opened = world.process_repo_workflow()
    assert len(opened) == 1
    assert len(repo.repo.pull_requests) == 1


def test_project_workspace_uses_pack_identity_and_disables_market_and_funding(monkeypatch) -> None:
    pack_id = "traffic_watch_v1"
    scenario = oss_time_machine(
        seed=23,
        dataset_id=pack_id,
        interaction_profile="human_project_workspace",
    )
    substrate = scenario.params["company_config"]["product_substrate"]
    assert substrate["dataset_id"] == pack_id
    assert not Path(substrate["dataset_id"]).is_absolute()

    spec = loader.load_oss_substrate_spec(pack_id)
    assert Path(spec.dataset_dir) == benchmark_directory() / "packs" / pack_id

    world = OrgWorld(scenario).build()
    assert world.product.substrate_meta["dataset_id"] == pack_id
    assert world.company_config["product_name"] == spec.product_name
    assert world.interaction_profile == "human_project_workspace"
    assert world.funding_simulation_enabled is False
    assert world.customer_market_enabled is False
    assert world.budget_system.funding.tranches == []
    assert world.process_funding_checkpoint(0) == []

    from environments.org_env.backend.market import validation
    from environments.org_env.external_society import bridge

    monkeypatch.setattr(
        validation,
        "product_quality",
        lambda _world: (_ for _ in ()).throw(AssertionError("market must stay disabled")),
    )
    assert validation.run_market_trials(world, tick=1) == []

    world.external_society = SimpleNamespace()
    monkeypatch.setattr(
        bridge,
        "run_external_product_experience",
        lambda *_args: (_ for _ in ()).throw(AssertionError("market must stay disabled")),
    )
    assert bridge.drive_post_release_market(world, tick=1) == []


def test_default_scenario_keeps_the_main_study_profile_implicit() -> None:
    scenario = oss_time_machine(seed=23, dataset_id="traffic_watch_v1")
    assert "interaction_profile" not in scenario.params

    world = OrgWorld(scenario).build()
    assert world.interaction_profile == "organization_simulation"
    assert world.funding_simulation_enabled is True
    assert world.customer_market_enabled is True
    assert world.budget_system.funding.tranches
