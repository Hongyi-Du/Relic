"""What a human seat may see (HCI P1).

Two properties matter here. The seat view must show a member exactly what the
autonomous loop would let that member perceive — no more, or the two
controllers are deciding under different information and comparing them is
meaningless. And it must never carry another member's interior, the evaluator's
answers, or which seats are human-driven.

Run:  PYTHONPATH="." python tests/org_env/test_human_seat_view.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.human import visibility as vis
from environments.org_env.human.affordances import OBJECT_ACTIONS
from environments.org_env.human.organization_brief import build_organization_brief
from environments.org_env.human.seat_view import build_seat_view
from environments.org_env.runtime_adapter.live import OrgInspectorSession
from environments.org_env.runtime_adapter.perception import OrgPerceptionAdapter

SEAT = "sean"
OTHER = "calvin"


def _world(ticks=48, seed=42):
    s = OrgInspectorSession(seed=seed)
    s.step(ticks)
    return s.world


def _perception(world, agent_id=SEAT):
    return OrgPerceptionAdapter().build_perception(agent_id, world, world.world_tick)


def _ids(rows, key="id"):
    return {r[key] for r in rows}


# ---- the seat and the loop filter identically ---------------------------- #
def test_seat_view_matches_agent_perception_object_for_object():
    """The one property that keeps the human and the agent comparable. If these
    two ever disagree, the HCI is showing a different organization than the one
    the policy sees."""
    world = _world()
    pkt = _perception(world)
    view = build_seat_view(world, SEAT)
    objects = view["objects"]

    assert _ids(objects["tasks"]) == {t["task_id"] for t in pkt.visible_tasks}
    assert _ids(objects["documents"]) == {d["doc_id"] for d in pkt.visible_docs}
    assert _ids(objects["pull_requests"]) == {p["pr_id"] for p in pkt.visible_prs}
    assert _ids(objects["experiments"]) == {e["experiment_id"] for e in pkt.visible_experiments}
    assert _ids(objects["meetings"]) == {m["meeting_id"] for m in pkt.visible_meetings}
    assert _ids(objects["protocols"]) == {p["protocol_id"] for p in pkt.visible_protocols}
    assert _ids(objects["branches"]) == {b["branch_id"] for b in pkt.repo_branch_summary}
    assert _ids(objects["results"]) == {r["result_id"] for r in pkt.visible_results}

    seen = {m["id"] for t in view["feed"]["threads"] for m in t["messages"]}
    assert seen == {m["message_id"] for m in pkt.visible_messages}

    # ...and the fixture is rich enough for that agreement to mean something.
    assert len(objects["tasks"]) >= 5
    assert len(objects["documents"]) >= 3
    assert len(seen) >= 5


def test_every_object_action_kind_has_a_seat_visible_symbol_collection():
    """The natural-language compiler cannot ground an action whose target
    class is absent from the P1/P2 view, even if the menu knows the verb."""
    world = _world()
    objects = build_seat_view(world, SEAT)["objects"]
    plural = {
        "task": "tasks", "pull_request": "pull_requests", "branch": "branches",
        "document": "documents", "experiment": "experiments", "result": "results",
        "meeting": "meetings", "proposal": "proposals", "protocol": "protocols",
        "release_candidate": "release_candidates", "release": "releases",
        "message": "messages", "issue": "issues",
    }
    assert set(OBJECT_ACTIONS) == set(plural)
    assert all(collection in objects for collection in plural.values())


def test_awaiting_me_matches_what_perception_flags():
    world = _world()
    pkt = _perception(world)
    view = build_seat_view(world, SEAT)

    assert (_ids(view["member"]["awaiting_me"]["reviews"])
            == {p["pr_id"] for p in pkt.prs_awaiting_my_review})
    assert (_ids(view["member"]["my_tasks"])
            == {t["task_id"] for t in pkt.assigned_tasks})


def test_two_seats_see_different_organizations():
    """Information asymmetry has to survive the HCI layer."""
    world = _world()
    mine = build_seat_view(world, SEAT)
    theirs = build_seat_view(world, OTHER)

    assert mine["seat"]["agent_id"] != theirs["seat"]["agent_id"]
    # Private-by-owner classes must not be identical across two members.
    assert (_ids(mine["member"]["my_tasks"]) != _ids(theirs["member"]["my_tasks"])
            or not mine["member"]["my_tasks"])


# ---- private things stay private ----------------------------------------- #
def test_another_members_private_work_is_not_in_the_view():
    from environments.org_env.backend.entities import Document, Task

    world = _world(ticks=12)
    world.tasks["task_secret"] = Task(task_id="task_secret", title="calvin's private task",
                                      owner_id=OTHER, visibility="private")
    world.documents["doc_secret"] = Document(doc_id="doc_secret", title="calvin's private doc",
                                             author_id=OTHER, visibility="private")
    world.documents["doc_secret"].owner_id = OTHER

    view = build_seat_view(world, SEAT)
    assert "task_secret" not in _ids(view["objects"]["tasks"])
    assert "doc_secret" not in _ids(view["objects"]["documents"])

    # ...and its owner does see them, so the filter is discriminating rather
    # than just dropping everything.
    owner_view = build_seat_view(world, OTHER)
    assert "task_secret" in _ids(owner_view["objects"]["tasks"])


def test_a_channel_the_member_is_not_in_is_invisible():
    world = _world(ticks=12)
    world.comm.create_channel("ch_founders_only", channel_type="private",
                              members={"paul", "victor"})
    world.comm.send_message(sender_id="paul", channel_id="ch_founders_only",
                            text="we are running out of money", tick=world.world_tick)

    view = build_seat_view(world, SEAT)
    channels = {c["id"] for c in view["feed"]["channels"]}
    texts = " ".join(m["text"] for t in view["feed"]["threads"] for m in t["messages"])

    assert "ch_founders_only" not in channels
    assert "running out of money" not in texts

    insider = build_seat_view(world, "paul")
    insider_texts = " ".join(m["text"] for t in insider["feed"]["threads"]
                             for m in t["messages"])
    assert "running out of money" in insider_texts


def _all_keys(node, acc=None):
    acc = acc if acc is not None else set()
    if isinstance(node, dict):
        for k, v in node.items():
            acc.add(k)
            _all_keys(v, acc)
    elif isinstance(node, list):
        for item in node:
            _all_keys(item, acc)
    return acc


def test_seat_view_carries_no_agent_interior_or_evaluator_data():
    world = _world()
    world.assign_human_seat(SEAT)
    view = build_seat_view(world, SEAT)

    keys = _all_keys(view)
    leaked = sorted(keys & set(vis.FORBIDDEN_IN_SEAT_VIEW))
    assert not leaked, f"seat view exposes {leaked}"


def test_seat_view_never_says_who_is_human():
    import json

    world = _world(ticks=12)
    world.assign_human_seat(SEAT)
    view = build_seat_view(world, SEAT)

    assert "controller_type" not in json.dumps(view, default=str)
    # The roster is the place it would leak: every member must look the same.
    roster = view["member"]["members"]
    assert len(roster) == len(world.agents)
    shapes = {tuple(sorted(m)) for m in roster}
    assert len(shapes) == 1, f"roster entries differ in shape: {shapes}"


def test_a_human_seat_is_never_shown_a_tick():
    """HCI V0 §5: real time only. A tick counter in the view would leak the
    simulation's clock into a UI that is meant to read as wall-clock."""
    world = _world(ticks=12)
    view = build_seat_view(world, SEAT, wall_clock=lambda t: 1_700_000_000.0 + t * 20)

    assert "tick" not in view["clock"]
    assert set(view["clock"]) == {"now"}
    a_task = view["objects"]["tasks"][0]
    assert "age_hours" in a_task and "at" in a_task


def test_organization_brief_is_a_provenance_bearing_projection_of_the_seat_view():
    """The liaison can make state legible, but cannot see behind the seat view."""
    from environments.org_env.backend.entities import Task

    world = _world(ticks=24)
    world.tasks["task_secret"] = Task(task_id="task_secret", title="private launch plan",
                                      owner_id=OTHER, visibility="private")
    view = build_seat_view(world, SEAT)
    brief = build_organization_brief(view)

    assert brief["visibility"] == "seat_visible_state"
    assert brief["what_is_happening"]
    assert all({"id", "kind", "title", "what", "why", "owner", "refs"} <= set(item)
               for item in brief["what_is_happening"])
    assert all(ref["id"] != "task_secret" for ref in brief["source_refs"])

    visible_ids = {
        (row["kind"], row["id"])
        for rows in view["objects"].values() if isinstance(rows, list)
        for row in rows if isinstance(row, dict) and row.get("kind") and row.get("id")
    }
    assert all((ref["kind"], ref["id"]) in visible_ids for ref in brief["source_refs"])
    assert brief["workstreams"] == brief["what_is_happening"]
    assert brief["pending_decisions"] == brief["needs_your_decision"]


def test_organization_brief_marks_only_the_seats_own_pending_decisions():
    view = {
        "member": {
            "company": {},
            "awaiting_me": {
                "proposals": [{"id": "proposal_visible", "kind": "proposal",
                               "title": "Adopt review rule", "author": "calvin"}],
                "reviews": [], "meetings": [],
            },
        },
        "objects": {"tasks": [], "pull_requests": [], "experiments": [], "protocols": []},
    }
    brief = build_organization_brief(view)
    assert brief["needs_your_decision"] == [{
        "id": "proposal_visible",
        "kind": "proposal",
        "title": "Adopt review rule",
        "what": "Decide proposal: Adopt review rule",
        "why": "You are a designated approver; the liaison cannot approve it for you.",
        "owner": "calvin",
        "refs": [{"kind": "proposal", "id": "proposal_visible", "title": "Adopt review rule"}],
        "decision_type": "proposal_approval",
    }]


def test_unknown_member_is_rejected():
    world = _world(ticks=4)
    try:
        build_seat_view(world, "nobody")
    except KeyError as exc:
        assert "nobody" in str(exc)
    else:
        raise AssertionError("expected KeyError for an unknown member")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  ok ", fn.__name__)
    print(f"All {len(fns)} seat view tests passed!")
