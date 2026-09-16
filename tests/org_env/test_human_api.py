"""HTTP surface for human seats (HCI P1).

Route logic is plain functions on ``HumanApi``, so most of this needs no web
server. The FastAPI pass at the end exists for one reason the pure functions
cannot catch: ``/api/org/{section}`` is a catch-all registered last, and it
will silently swallow any human route declared after it.

Run:  PYTHONPATH="." python tests/org_env/test_human_api.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.human.api import HumanApi
from environments.org_env.runtime_adapter.live import OrgInspectorSession

SEAT = "sean"


def _api(ticks=12):
    session = OrgInspectorSession(seed=42)
    if ticks:
        session.step(ticks)
    api = HumanApi(lambda: session, seconds_per_tick=0.05)
    # These are deterministic HTTP/runtime contract tests. A developer's
    # ignored llm.local.yaml must not turn them into paid provider tests.
    api._llm_client = lambda: None
    return api, session


# ---- seats ---------------------------------------------------------------- #
def test_members_are_listed_with_enough_detail_to_choose_one():
    api, _ = _api()
    try:
        members = api.members()["members"]
        assert len(members) == 8
        sean = next(m for m in members if m["agent_id"] == SEAT)
        assert sean["role"] == "fast_engineer" and sean["claimed"] is False
        assert sean["identity"]
    finally:
        api.shutdown()


def test_claim_returns_a_token_and_release_invalidates_it():
    api, _ = _api()
    try:
        claim = api.claim(SEAT)
        assert claim["agent_id"] == SEAT and claim["token"]
        assert api.runtime().world.is_human_controlled(SEAT)

        assert api.release(claim["token"]) == {"released": SEAT}
        assert not api.runtime().world.is_human_controlled(SEAT)
        assert api.view(claim["token"])["error"] == "invalid_seat_token"
    finally:
        api.shutdown()


def test_claiming_a_taken_seat_or_an_unknown_member_is_refused():
    api, _ = _api()
    try:
        api.claim(SEAT)
        assert "already_claimed" in api.claim(SEAT)["error"]
        assert "unknown_member" in api.claim("nobody")["error"]
    finally:
        api.shutdown()


# ---- reads ---------------------------------------------------------------- #
def test_a_view_needs_a_valid_token():
    api, _ = _api()
    try:
        api.claim(SEAT)
        for bad in ("", "not-a-token"):
            assert api.view(bad)["error"] == "invalid_seat_token"
    finally:
        api.shutdown()


def test_one_seats_token_only_opens_that_seats_view():
    """Multi-seat isolation: holding Sean's token must not show Calvin's
    organization, which is a different set of visible objects."""
    api, _ = _api()
    try:
        sean = api.claim(SEAT)
        calvin = api.claim("calvin")

        assert api.view(sean["token"])["view"]["seat"]["agent_id"] == SEAT
        assert api.view(calvin["token"])["view"]["seat"]["agent_id"] == "calvin"
    finally:
        api.shutdown()


def test_view_reports_unchanged_when_the_client_is_current():
    api, _ = _api()
    try:
        token = api.claim(SEAT)["token"]
        first = api.view(token)
        version = first["view"]["version"]

        assert first["unchanged"] is False
        assert api.view(token, since_version=version)["unchanged"] is True
        assert api.view(token, since_version=version - 1)["unchanged"] is False
    finally:
        api.shutdown()


def test_brief_requires_a_token_and_carries_only_seat_visible_sources():
    api, _ = _api(ticks=24)
    try:
        assert api.brief("bad")["error"] == "invalid_seat_token"
        token = api.claim(SEAT)["token"]
        out = api.brief(token)

        assert out["brief"]["visibility"] == "seat_visible_state"
        assert out["version"] > 0
        assert all({"kind", "id", "title"} <= set(ref)
                   for ref in out["brief"]["source_refs"])
    finally:
        api.shutdown()


# ---- acting --------------------------------------------------------------- #
def test_offers_need_a_token_and_come_back_marked_allowed_or_not():
    api, _ = _api(ticks=24)
    try:
        assert api.offers("bad")["error"] == "invalid_seat_token"

        token = api.claim(SEAT)["token"]          # sean is a fast_engineer
        actions = {a["action_type"]: a for a in api.offers(token)["actions"]}
        assert actions["send_message"]["allowed"] is True
        assert actions["propose_protocol"]["allowed"] is False
        assert actions["propose_protocol"]["denied_because"].startswith("role_not_permitted")
        assert actions["send_message"]["required"] == ["channel_id", "text"]
        assert actions["send_message"]["confirm"] is True
    finally:
        api.shutdown()


def test_an_object_menu_is_scoped_to_that_object():
    api, _ = _api(ticks=24)
    try:
        token = api.claim(SEAT)["token"]
        task_id = next(iter(api.runtime().world.tasks))
        menu = api.offers(token, task_id)

        assert menu["kind"] == "task" and menu["object_id"] == task_id
        assert {a["action_type"] for a in menu["actions"]} >= {"pick_task", "work_on_task"}
        assert all(a.get("target") == task_id for a in menu["actions"])
    finally:
        api.shutdown()


def test_act_executes_and_refusals_come_back_as_errors():
    api, _ = _api(ticks=24)
    try:
        token = api.claim(SEAT)["token"]
        out = api.act(token, "send_message",
                      {"channel_id": "team_general", "text": "on the sorting crash"})
        assert out["ok"] is True and out["action_type"] == "send_message"
        assert out["version"] > 0

        assert api.act(token, "propose_protocol",
                       {"title": "x"})["error"].startswith("role_not_permitted")
        assert api.act(token, "nonsense", {})["error"] == "unknown_action:nonsense"
        assert api.act("bad-token", "send_message", {})["error"] == "invalid_seat_token"
    finally:
        api.shutdown()


def test_acting_is_attributed_to_the_human_in_the_research_log_only():
    api, _ = _api(ticks=24)
    try:
        token = api.claim(SEAT)["token"]
        api.act(token, "send_message",
                {"channel_id": "team_general", "text": "hello"},
                execution_mode="working_agent_assisted")

        world = api.runtime().world
        entry = world.controller_log[-1]
        assert entry["controller_type"] == "human"
        assert entry["execution_mode"] == "working_agent_assisted"
        assert all("controller" not in k for k in world.action_log[-1])
    finally:
        api.shutdown()


# ---- working agent -------------------------------------------------------- #
def test_the_working_agent_is_reachable_and_private_to_its_seat():
    import time

    api, _ = _api(ticks=12)
    try:
        mine = api.claim(SEAT)["token"]
        theirs = api.claim("calvin")["token"]

        assert api.agent_send("bad", "hello")["error"] == "invalid_seat_token"
        assert api.agent_send(mine, "  ")["error"] == "empty_message"
        assert api.agent_send(mine, "what am I working on")["accepted"] is True

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and api.agent_state(mine)["busy"]:
            time.sleep(0.05)

        mine_text = " ".join(m["text"] for m in api.agent_state(mine)["messages"])
        assert "Sean" in mine_text
        # The other seat's assistant knows nothing about this conversation.
        assert api.agent_state(theirs)["messages"] == []
    finally:
        api.shutdown()


def test_a_draft_is_only_sent_when_the_human_confirms_it():
    api, _ = _api(ticks=12)
    try:
        token = api.claim(SEAT)["token"]
        session = api._agent(SEAT)
        draft = session.add_draft("send_message",
                                  {"channel_id": "team_general", "text": "on it"})

        world = api.runtime().world
        before = len(world.action_log)
        assert api.agent_state(token)["drafts"][0]["draft_id"] == draft.draft_id
        assert len(world.action_log) == before

        out = api.agent_confirm(token, draft.draft_id)
        assert out["ok"] is True
        assert len(world.action_log) == before + 1
        assert world.controller_log[-1]["execution_mode"] == "liaison_assisted"

        assert api.agent_confirm(token, "nope")["error"].startswith("unknown_or_settled")
    finally:
        api.shutdown()


def test_a_draft_can_be_discarded():
    api, _ = _api(ticks=12)
    try:
        token = api.claim(SEAT)["token"]
        draft = api._agent(SEAT).add_draft("send_message",
                                           {"channel_id": "team_general", "text": "x"})
        assert api.agent_discard(token, draft.draft_id)["discarded"] == draft.draft_id
        assert api.agent_state(token)["drafts"] == []
    finally:
        api.shutdown()


def test_liaison_input_drafts_and_confirmations_are_exportable_hci_events():
    api, _ = _api(ticks=12)
    try:
        token = api.claim(SEAT)["token"]
        assert api.agent_send(token, "Tell the team to prioritize the blocker")["accepted"]

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and api.agent_state(token)["busy"]:
            time.sleep(0.02)
        draft = api.agent_state(token)["drafts"][0]
        assert api.agent_confirm(token, draft["draft_id"])["ok"] is True

        types = [event["type"] for event in api.get_hci_logs()]
        assert "liaison_message" in types
        assert "liaison_draft_created" in types
        assert "liaison_draft_confirmed" in types
    finally:
        api.shutdown()


# ---- clock ---------------------------------------------------------------- #
def test_runtime_can_be_started_paused_and_retuned():
    import time

    api, _ = _api()
    try:
        assert api.runtime_status()["running"] is False
        api.runtime_start(seconds_per_tick=0.05)
        assert api.runtime_status()["running"] is True

        deadline = time.monotonic() + 5
        start_tick = api.runtime().world.world_tick
        while time.monotonic() < deadline and api.runtime().world.world_tick <= start_tick:
            time.sleep(0.02)
        assert api.runtime().world.world_tick > start_tick

        assert api.runtime_pause()["running"] is False
    finally:
        api.shutdown()


def test_the_inspector_keeps_up_with_the_live_clock():
    """The seat UI and the inspector share one world. The human clock steps it
    directly, so without a frame capture per tick /api/org/lived/* would sit
    frozen at the last manual step while the organization moved on."""
    import time

    api, session = _api(ticks=4)
    try:
        before = session.buffer.latest_tick
        api.runtime_start(seconds_per_tick=0.05)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and session.buffer.latest_tick <= before:
            time.sleep(0.02)
        api.runtime_pause()

        assert session.buffer.latest_tick > before, "inspector frames stopped"
        assert session.buffer.latest_tick == api.runtime().world.world_tick
        assert session.full()["tick"] == api.runtime().world.world_tick
    finally:
        api.shutdown()


def test_resetting_the_session_rebinds_the_runtime_to_the_new_world():
    """The inspector can reset or load a checkpoint under us; a clock still
    stepping the discarded world would be a silent, very confusing leak."""
    api, session = _api(ticks=4)
    try:
        api.claim(SEAT)
        old_world = api.runtime().world
        old_runtime = api.runtime()

        session.reset(seed=7)
        new_runtime = api.runtime()

        assert new_runtime is not old_runtime
        assert new_runtime.world is session.world
        assert new_runtime.world is not old_world
        # The seat did not survive the reset; the new world has no human seats.
        assert new_runtime.world.human_seat_ids() == []
    finally:
        api.shutdown()


# ---- HTTP ----------------------------------------------------------------- #
def test_human_routes_are_not_swallowed_by_the_catch_all():
    try:
        from fastapi.testclient import TestClient
    except Exception:
        print("  (skip HTTP test: fastapi not installed)")
        return

    from environments.org_env.backend import main
    previous_session = main.SESSION
    main.SESSION = main.OrgInspectorSession(seed=42)
    try:
        main.SESSION.reset(seed=42)
        main.api_sim_run_ticks(8)
        client = TestClient(main.build_app())
        assert len(client.get("/api/org/human/members").json()["members"]) == 8

        token = client.post("/api/org/human/seats/claim",
                            json={"agent_id": SEAT}).json()["token"]
        body = client.get(f"/api/org/human/view?token={token}").json()
        assert body["view"]["seat"]["agent_id"] == SEAT
        assert body["unchanged"] is False

        brief = client.get(f"/api/org/human/brief?token={token}").json()
        assert brief["brief"]["visibility"] == "seat_visible_state"

        version = body["view"]["version"]
        assert client.get(
            f"/api/org/human/view?token={token}&since_version={version}"
        ).json()["unchanged"] is True

        assert client.get("/api/org/human/runtime").json()["running"] is False
        assert client.post("/api/org/human/runtime/start",
                           json={"seconds_per_tick": 60}).json()["running"] is True
        assert client.post("/api/org/human/runtime/pause").json()["running"] is False

        offers = client.get(f"/api/org/human/offers?token={token}").json()
        assert any(a["action_type"] == "send_message" for a in offers["actions"])

        acted = client.post("/api/org/human/act", json={
            "token": token, "action_type": "send_message",
            "params": {"channel_id": "team_general", "text": "shipping the fix"},
        }).json()
        assert acted["ok"] is True

        # The pre-existing catch-all still answers for real sections.
        assert client.get("/api/org/messages").status_code == 200
        assert client.get("/api/org/human/view?token=bogus").json()["error"]
    finally:
        main.HUMAN.shutdown()
        main.SESSION = previous_session


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  ok ", fn.__name__)
    print(f"All {len(fns)} human API tests passed!")
