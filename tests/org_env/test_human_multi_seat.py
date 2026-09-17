"""Several people, one organization (HCI P5).

Multiple humans hold different seats in the same world at the same time, each
from their own browser. What has to hold: their views differ according to what
each member may see, their tokens do not cross, their assistants are separate,
and simultaneous writes do not corrupt a world that is not thread-safe.

Run:  PYTHONPATH="." python tests/org_env/test_human_multi_seat.py
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.human.api import HumanApi
from environments.org_env.runtime_adapter.live import OrgInspectorSession

SEATS = ("sean", "calvin", "will")


def _api(ticks=24):
    session = OrgInspectorSession(seed=42)
    session.step(ticks)
    return HumanApi(lambda: session, seconds_per_tick=0.05)


# ---- separate people ------------------------------------------------------ #
def test_three_people_hold_three_seats_at_once():
    api = _api()
    try:
        tokens = {s: api.claim(s)["token"] for s in SEATS}
        assert len(set(tokens.values())) == 3, "tokens collided"
        assert sorted(api.runtime().world.human_seat_ids()) == sorted(SEATS)

        for seat_id, token in tokens.items():
            assert api.view(token)["view"]["seat"]["agent_id"] == seat_id

        # ...and the rest of the company still runs itself.
        autonomous = set(api.runtime().world.agents) - set(SEATS)
        assert len(autonomous) == 5
    finally:
        api.shutdown()


def test_each_seat_sees_its_own_organization():
    from environments.org_env.backend.entities import Task

    api = _api()
    try:
        tokens = {s: api.claim(s)["token"] for s in SEATS}
        world = api.runtime().world
        world.tasks["task_for_calvin"] = Task(task_id="task_for_calvin",
                                              title="calvin only", owner_id="calvin",
                                              visibility="private")
        api.runtime().refresh_views()

        def task_ids(seat):
            return {t["id"] for t in api.view(tokens[seat])["view"]["objects"]["tasks"]}

        assert "task_for_calvin" in task_ids("calvin")
        assert "task_for_calvin" not in task_ids("sean")
        assert "task_for_calvin" not in task_ids("will")
    finally:
        api.shutdown()


def test_a_token_only_ever_acts_as_its_own_seat():
    api = _api()
    try:
        sean = api.claim("sean")["token"]
        api.claim("calvin")

        out = api.act(sean, "send_message",
                      {"channel_id": "team_general", "text": "who am I"})
        assert out["ok"] is True
        assert api.runtime().world.action_log[-1]["agent_id"] == "sean"
        assert api.runtime().world.controller_log[-1]["agent_id"] == "sean"
    finally:
        api.shutdown()


def test_releasing_one_seat_leaves_the_others_alone():
    api = _api()
    try:
        tokens = {s: api.claim(s)["token"] for s in SEATS}
        api.release(tokens["sean"])

        assert api.view(tokens["sean"])["error"] == "invalid_seat_token"
        assert api.view(tokens["calvin"])["view"]["seat"]["agent_id"] == "calvin"
        assert not api.runtime().world.is_human_controlled("sean")
        assert api.runtime().world.is_human_controlled("calvin")
    finally:
        api.shutdown()


def test_assistants_do_not_share_conversations():
    api = _api()
    try:
        tokens = {s: api.claim(s)["token"] for s in SEATS}
        api._agent("sean").add_draft("send_message",
                                     {"channel_id": "team_general", "text": "mine"})

        assert len(api.agent_state(tokens["sean"])["drafts"]) == 1
        assert api.agent_state(tokens["calvin"])["drafts"] == []
        assert api.agent_state(tokens["will"])["drafts"] == []
    finally:
        api.shutdown()


# ---- simultaneous writes -------------------------------------------------- #
def test_three_people_acting_at_once_while_the_clock_runs():
    """The world is not thread-safe. Three browsers plus a background clock is
    the realistic worst case for the runtime lock."""
    api = _api()
    try:
        tokens = {s: api.claim(s)["token"] for s in SEATS}
        api.runtime_start(seconds_per_tick=0.05)
        errors: list = []
        each = 8

        def hammer(seat_id, token):
            for i in range(each):
                out = api.act(token, "send_message",
                              {"channel_id": "team_general", "text": f"{seat_id} {i}"})
                if out.get("error") or out.get("ok") is False:
                    errors.append((seat_id, out))

        threads = [threading.Thread(target=hammer, args=(s, t))
                   for s, t in tokens.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        api.runtime_pause()

        assert not errors, errors[:3]
        world = api.runtime().world
        human = [c for c in world.controller_log if c["controller_type"] == "human"]
        assert len(human) == len(SEATS) * each
        by_seat = {s: sum(1 for c in human if c["agent_id"] == s) for s in SEATS}
        assert all(n == each for n in by_seat.values()), by_seat
        # The organization kept running underneath them.
        assert world.world_tick > 24
    finally:
        api.shutdown()


def test_a_message_from_one_seat_reaches_the_others():
    api = _api()
    try:
        tokens = {s: api.claim(s)["token"] for s in SEATS}
        api.act(tokens["sean"], "send_message",
                {"channel_id": "team_general", "text": "standup in five"})

        for reader in ("calvin", "will"):
            view = api.view(tokens[reader])["view"]
            texts = [m["text"] for t in view["feed"]["threads"] for m in t["messages"]]
            assert any("standup in five" in x for x in texts), reader
    finally:
        api.shutdown()


def test_no_seat_can_tell_which_others_are_human():
    """HCI V0 §6: the roster looks the same for everyone, whoever is behind it."""
    api = _api()
    try:
        tokens = {s: api.claim(s)["token"] for s in SEATS}
        roster = api.view(tokens["sean"])["view"]["member"]["members"]

        assert len(roster) == 8
        assert {tuple(sorted(m)) for m in roster}.__len__() == 1
        assert not any("controller" in k for m in roster for k in m)
    finally:
        api.shutdown()


# ---- reachable from another machine --------------------------------------- #
def test_the_server_binds_where_it_is_told():
    """Remote seats are the reason to bind beyond loopback, so the host has to
    be configurable rather than hardcoded."""
    import inspect

    from environments.org_env.backend import main

    source = inspect.getsource(main.main)
    assert 'os.environ.get("ORG_HOST"' in source
    assert "uvicorn.run(build_app(), host=host, port=port)" in source
    assert 'os.environ.get("ORG_PORT"' in source


def test_the_printed_urls_include_the_member_workspace():
    import inspect

    from environments.org_env.backend import main

    source = inspect.getsource(main.main)
    assert "/org/seat" in source
    # Binding wide is a real exposure; say so rather than doing it quietly.
    assert "anyone who can reach this port" in source


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("  ok ", fn.__name__)
    print(f"All {len(fns)} multi-seat tests passed!")
