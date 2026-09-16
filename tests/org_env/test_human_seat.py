"""Human-controlled member seats (HCI P0) — seat marking, the shared action
pipeline, and the invariant that the organization cannot tell a human seat from
an autonomous one.

Run:  PYTHONPATH="." python tests/org_env/test_human_seat.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_sdk.lived.core.contracts import ActionCandidate
from environments.org_env.runtime_adapter.live import OrgInspectorSession
from environments.org_env.runtime_adapter.snapshot import org_lived_full_snapshot

HUMAN = "sean"


def _session(seed=42, ticks=0):
    s = OrgInspectorSession(seed=seed)
    if ticks:
        s.step(ticks)
    return s


# ---- seat marking -------------------------------------------------------- #
def test_every_seat_is_autonomous_by_default():
    w = _session().world
    assert w.human_seat_ids() == []
    assert all(not w.is_human_controlled(aid) for aid in w.agents)


def test_assigning_a_seat_changes_only_who_decides():
    w = _session().world
    before = w.agents[HUMAN]
    role, skills, perms = before.role, dict(before.skills), list(before.permissions)

    w.assign_human_seat(HUMAN)

    assert w.is_human_controlled(HUMAN)
    assert w.human_seat_ids() == [HUMAN]
    # The member itself is untouched, so the same seat stays comparable across
    # controllers — that is the whole point of taking over an existing member.
    assert w.agents[HUMAN].role == role
    assert w.agents[HUMAN].skills == skills
    assert w.agents[HUMAN].permissions == perms
    assert [aid for aid in w.agents if aid != HUMAN and w.is_human_controlled(aid)] == []

    w.release_human_seat(HUMAN)
    assert not w.is_human_controlled(HUMAN)


def test_assigning_an_unknown_seat_fails_loudly():
    w = _session().world
    try:
        w.assign_human_seat("nobody")
    except KeyError as exc:
        assert "nobody" in str(exc)
    else:
        raise AssertionError("expected KeyError for an unknown agent")


# ---- the autonomous loop leaves a human seat alone ----------------------- #
def test_step_never_acts_for_a_human_seat():
    s = _session()
    s.world.assign_human_seat(HUMAN)
    s.step(24)

    acted = {a["agent_id"] for a in s.world.action_log}
    assert HUMAN not in acted, f"autonomous loop acted for the human seat: {acted}"
    # ...while the rest of the company keeps working, so the world is still live.
    assert len(acted) >= 2, acted


def test_repo_sweep_never_opens_a_pull_request_for_a_human_seat():
    w = _session().world
    w.assign_human_seat(HUMAN)
    human_branch = w.repo_system.create_branch(HUMAN, tick=0)
    agent_branch = w.repo_system.create_branch("calvin", tick=0)
    for owner, branch in ((HUMAN, human_branch), ("calvin", agent_branch)):
        w.repo_system.edit_file(owner, branch.branch_id)
        w.repo_system.commit_changes(
            agent_id=owner,
            branch_id=branch.branch_id,
            message=f"{owner} implementation",
            changed_files=[f"{owner}.py"],
            tick=0,
        )
    w.world_tick = 3

    w.process_repo_workflow()

    requests = list(w.repo_system.repo.pull_requests.values())
    assert all(pr.author_id != HUMAN for pr in requests)
    assert any(pr.author_id == "calvin" and pr.source_branch == agent_branch.branch_id
               for pr in requests)
    assert human_branch.status.value == "ready_for_pr"


def test_a_human_seat_still_receives_organizational_state():
    """Skipping the decision loop must not turn the seat into a ghost: other
    members still see it, and it still has a perceivable task board / inbox."""
    s = _session()
    s.world.assign_human_seat(HUMAN)
    s.step(24)
    w = s.world

    assert HUMAN in w.agents
    perception = w._loop["perception"].build_perception(HUMAN, w, w.world_tick)
    assert perception.agent_id == HUMAN
    assert perception.visible_tasks, "human seat sees no task board"


# ---- one pipeline, two controllers --------------------------------------- #
def _send(world, agent_id, controller_type="agent", execution_mode="direct"):
    action = ActionCandidate(action_type="send_message",
                             parameters={"channel_id": "ch_general",
                                         "text": "status update"})
    return world.apply_action_candidate(agent_id, action,
                                        controller_type=controller_type,
                                        execution_mode=execution_mode)


def test_human_and_agent_actions_are_indistinguishable_in_the_org_record():
    s = _session(ticks=8)
    w = s.world
    w.assign_human_seat(HUMAN)

    before = len(w.action_log)
    agent_result = _send(w, "calvin")
    human_result = _send(w, HUMAN, controller_type="human",
                         execution_mode="working_agent_assisted")

    entries = w.action_log[before:]
    assert len(entries) == 2, entries
    # Same keys, same shape: nothing in the organizational log marks the human.
    assert set(entries[0]) == set(entries[1])
    assert all("controller" not in k for e in entries for k in e), entries
    assert agent_result.action_type == human_result.action_type

    for ev in w.events:
        assert "controller_type" not in ev
        assert "execution_mode" not in ev


def test_controller_provenance_lands_only_in_the_research_log():
    s = _session(ticks=8)
    w = s.world
    w.assign_human_seat(HUMAN)

    _send(w, "calvin")                       # autonomous: nothing recorded
    assert w.controller_log == []

    _send(w, HUMAN, controller_type="human", execution_mode="working_agent_assisted")

    assert len(w.controller_log) == 1
    entry = w.controller_log[0]
    assert entry["agent_id"] == HUMAN
    assert entry["controller_type"] == "human"
    assert entry["execution_mode"] == "working_agent_assisted"
    assert entry["action_type"] == "send_message"
    assert entry["tick"] == w.world_tick
    assert entry["wall_time"] > 0


def _downstream_delta(world, agent_id, **kwargs):
    """Run one action and report what it moved in every downstream system."""
    before = (len(getattr(world.event_graph, "edges", []) or []),
              len(world._growth_signals), len(world.events),
              len(world.memory.get(agent_id, [])), len(world.action_log))
    result = _send(world, agent_id, **kwargs)
    after = (len(getattr(world.event_graph, "edges", []) or []),
             len(world._growth_signals), len(world.events),
             len(world.memory.get(agent_id, [])), len(world.action_log))
    return tuple(b - a for a, b in zip(before, after)), result


def test_a_human_action_feeds_the_same_downstream_systems():
    """The same seat taking the same action must move every downstream system
    identically whether a human or the policy chose it — otherwise the seat is
    observably different from an autonomous one."""
    autonomous, agent_result = _downstream_delta(_session(ticks=8).world, HUMAN)

    human_world = _session(ticks=8).world
    human_world.assign_human_seat(HUMAN)
    human, human_result = _downstream_delta(human_world, HUMAN,
                                            controller_type="human")

    assert human == autonomous, f"human {human} != autonomous {autonomous}"
    assert human[0] > 0, "the action reached no event-graph edge at all"
    assert human[4] == 1, "the action was not written to the organizational log"
    assert human_result.action_type == agent_result.action_type
    assert human_result.success == agent_result.success


# ---- the seat's controller is invisible in-world ------------------------- #
def test_snapshot_never_reveals_which_seat_is_human():
    import json

    s = _session(ticks=12)
    s.world.assign_human_seat(HUMAN)
    s.step(4)
    frame = org_lived_full_snapshot(s.world, mode="live")

    blob = json.dumps(frame, default=str)
    assert "controller_type" not in blob
    assert "controller_log" not in blob
    for agent in frame["agents"].values():
        assert "controller_type" not in agent


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  ok ", fn.__name__)
    print(f"All {len(fns)} human seat tests passed!")
