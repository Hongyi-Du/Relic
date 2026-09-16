"""OrgEnv Live Inspector — snapshot contract + frame buffer + sim controls + API
route logic (frontend spec §10/§13). No browser/FastAPI needed: route logic is
plain functions over an OrgInspectorSession.

Run:  PYTHONPATH="." python tests/org_env/test_org_inspector.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.runtime_adapter.live import OrgInspectorSession
from environments.org_env.runtime_adapter.snapshot import org_lived_full_snapshot
from environments.org_env.human.seat_view import build_seat_view


def _world(seed=42, ticks=24):
    s = OrgInspectorSession(seed=seed)
    s.step(ticks)
    return s


# ---- §10 snapshot contract ---------------------------------------------- #
def test_snapshot_has_all_top_level_sections():
    f = _world().full()
    for k in ("wired", "mode", "tick", "day", "hour", "phase", "company", "agents",
              "internal", "external", "graphs", "timeline", "logs", "scenario"):
        assert k in f, k


def test_snapshot_has_8_agents_with_full_state():
    f = _world().full()
    assert len(f["agents"]) == 8
    a = f["agents"]["calvin"]
    for k in ("work_state", "org_state", "profile", "skills", "failure_modes",
              "communication_style", "routine_profile", "compensation", "availability",
              "memory", "workspace"):
        assert k in a, k
    # agent-perspective workspace (frontend): private notes/todos/focus + local files w/ content
    ws = a["workspace"]
    for k in ("current_focus", "personal_notes", "private_todos", "open_questions",
              "draft_docs", "local_files", "local_file_count"):
        assert k in ws, k
    assert isinstance(ws["local_files"], list)
    ws = a["work_state"]
    for k in ("attention_remaining_today", "fatigue", "stress", "burnout_risk", "morale",
              "next_available_tick", "aux_speech_slots_remaining", "background_jobs"):
        assert k in ws, k
    for k in ("trust_in_company", "compensation_stress", "retention_risk"):
        assert k in a["org_state"], k


def test_internal_sections_present():
    I = _world().full()["internal"]
    for k in ("tasks", "docs", "files", "messages", "channels", "meetings", "repo",
              "sandbox", "experiments", "results", "budget", "payroll", "protocols",
              "commitments", "disputes", "requests", "detectors"):
        assert k in I, k
    assert isinstance(I["repo"]["pull_requests"], list)


def test_external_sections_present():
    X = _world().full()["external"]
    for k in ("profiles", "posts", "comments", "signals", "offers"):
        assert k in X, k
    assert len(X["profiles"]) >= 3 and len(X["posts"]) >= 3


def test_graphs_present():
    g = _world().full()["graphs"]
    for k in ("persona", "social", "event", "external_network"):
        assert k in g, k
    # persona is now 4 modes per agent (summary default + policy_debug/evidence/raw)
    assert "calvin" in g["persona"]
    pc = g["persona"]["calvin"]
    for mode in ("summary", "policy_debug", "evidence", "raw"):
        assert mode in pc, mode
    assert pc["summary"]["nodes"] and len(pc["summary"]["nodes"]) <= pc["raw"]["raw_node_count"]
    assert isinstance(g["event"]["nodes"], list) and g["event"]["nodes"]


def test_logs_present():
    L = _world().full()["logs"]
    for k in ("actions", "work_sessions", "appraisals", "feedback_decisions",
              "text_generation", "protocol_events"):
        assert k in L, k


# ---- §11 visibility metadata -------------------------------------------- #
def test_visibility_metadata_on_objects():
    f = _world().full()
    docs = f["internal"]["docs"]
    assert docs and all("visible_to_agents" in d and "debug_visible" in d for d in docs)
    msgs = f["internal"]["messages"]
    if msgs:
        assert "read_by" in msgs[0] and "visible_to_agents" in msgs[0]
    results = f["internal"]["results"]
    if results:
        assert "lifecycle" in results[0] and "logged_to_tracker" in results[0]["lifecycle"]


def test_external_post_internal_exposure():
    posts = _world(ticks=48).full()["external"]["posts"]
    assert posts and "internal_exposure" in posts[0]
    ie = posts[0]["internal_exposure"]
    for k in ("read_by_internal_agents", "shared_to_internal_channels", "cited_by_docs"):
        assert k in ie


# ---- §7/§12 frame buffer + sim controls --------------------------------- #
def test_frame_buffer_and_step():
    s = OrgInspectorSession(seed=42)
    assert s.buffer.latest_tick == 0           # tick-0 frame captured at reset
    s.step(5)
    assert s.buffer.latest_tick == 5
    fr = s.frames_since(2)
    assert all(f["tick"] > 2 for f in fr["frames"]) and fr["frames"]


def test_reset_restarts_world():
    s = _world(ticks=10)
    out = s.reset(seed=7)
    assert out["reset"] and s.world.world_tick == 0 and s.seed == 7


def test_determinism_same_seed():
    a = OrgInspectorSession(seed=99); a.step(24)
    b = OrgInspectorSession(seed=99); b.step(24)
    fa, fb = a.full(), b.full()
    assert fa["company"]["tasks_done"] == fb["company"]["tasks_done"]
    assert fa["agents"]["sean"]["work_state"]["fatigue"] == fb["agents"]["sean"]["work_state"]["fatigue"]


def test_to_replay_shape():
    s = _world(ticks=6)
    rep = s.to_replay("t")
    assert rep["meta"]["ticks"] == len(rep["frames"]) and rep["frames"]
    assert sorted(rep["meta"]["agents"]) and "frames" in rep


def test_delta_replay_roundtrip_is_identical_and_smaller():
    """The compact base+delta replay reconstructs byte-identical frames and is
    materially smaller than the full-frame replay (frontend spec — replay size)."""
    import json as _json

    from environments.org_env.runtime_adapter.replay_delta import (
        KEYFRAME_KEYS, expand_delta_replay)
    s = _world(ticks=14)
    full = s.to_replay("t")
    delta = s.to_replay_delta("t")
    assert delta["format"] == "org-delta-v1" and "base" in delta and "deltas" in delta
    rebuilt = expand_delta_replay(delta)
    assert len(rebuilt) == len(full["frames"])
    # per-tick org/agent state is reconstructed EXACTLY; heavy derived/debug sections
    # (graphs/logs/timeline) are keyframed + carried forward, so they're present every
    # frame and exact on keyframe ticks.
    for orig, got in zip(full["frames"], rebuilt):
        for k, v in orig.items():
            if k in KEYFRAME_KEYS:
                assert k in got                            # carried forward (present)
            else:
                assert got[k] == v, k                      # exact per-tick reconstruction
    # legacy {frames:[...]} still expands (back-compat)
    assert expand_delta_replay({"frames": full["frames"]}) == full["frames"]
    # the delta envelope is MUCH smaller than the full frame dump
    assert len(_json.dumps(delta)) < 0.5 * len(_json.dumps(full))


# ---- §9/§13 API route logic (no FastAPI) -------------------------------- #
def test_api_route_logic():
    from environments.org_env.backend import main
    original = main.SESSION
    main.SESSION = OrgInspectorSession(seed=42, load_llm=False)
    try:
        main.api_sim_run_ticks(24)
        assert main.api_full()["tick"] == 24
        assert main.api_state()["day"] >= 0
        assert len(main.api_agents()["agents"]) == 8
        assert main.api_agent("calvin")["agent"]["id"] == "calvin"
        assert "persona_graph" in main.api_agent("calvin")
        assert isinstance(main.api_objects("task")["objects"], list)
        assert main.api_section("event_graph")["nodes"]
        # object lookup by id
        tasks = main.api_objects("task")["objects"]
        if tasks:
            tid = tasks[0]["task_id"]
            got = main.api_object(tid)
            assert got.get("object", {}).get("task_id") == tid and "linked_edges" in got
        fr = main.api_frames(10)
        assert fr["frames"] and fr["latest_tick"] == 24
    finally:
        main.HUMAN.shutdown()
        main.SESSION = original


def test_selected_oss_pack_rebuilds_the_world_and_remains_identifiable(monkeypatch):
    """A setup choice must bind the running substrate, not only decorate the UI."""
    from environments.org_env.backend import main

    original = main.SESSION
    session = OrgInspectorSession(seed=42, load_llm=False)
    monkeypatch.setenv("ORG_LLM", "0")
    main.SESSION = session
    try:
        pack_id = "traffic_watch_v1"
        main._load_pack_into_session(pack_id, main.PACK_DIR / pack_id)

        assert session.world.product.substrate_type == "oss_time_machine"
        assert session.world.product.substrate_meta["dataset_id"] == pack_id
        assert session.world.company_config["product_name"] == "traffic_violation_system"
        assert session.world.company.company_name == "traffic_violation_system"
        assert session.state()["company"]["name"] == "traffic_violation_system"
        assert session.selected_pack == {
            "id": pack_id,
            "source": "oss_time_machine/real",
            "dataset_id": pack_id,
            "product_name": "traffic_violation_system",
        }
        # Reset is a legitimate live-session action; it must not silently
        # discard the selected pack and revert to LanternScout.
        session.reset(seed=7)
        assert session.world.product.substrate_type == "oss_time_machine"
        assert session.world.product.substrate_meta["dataset_id"] == pack_id
        assert session.world.company.company_name == "traffic_violation_system"
        assert session.state()["pack"]["id"] == pack_id
        metadata = session.runtime_metadata()
        assert metadata["engine"] == "org_env.OrgWorld"
        assert metadata["event_source"] == "live_org_world_seat_filtered_projection"
        assert metadata["llm"] == {
            "requested": False,
            "attached": False,
            "mode": "rule_template_only",
            "load_error": False,
        }
        assert metadata["policy_mode"] == "mock"
        assert metadata["action_selection_mode"] != "unknown"
    finally:
        main.SESSION = original


def test_lazy_pack_initialization_honors_the_paused_launcher_mode(monkeypatch):
    """``run_hci.py --paused`` must survive lazy setup and not start a thread."""
    from environments.org_env.backend import main

    original = main.SESSION
    main.HUMAN.shutdown()
    main.SESSION = None
    monkeypatch.setenv("ORG_LLM", "0")
    monkeypatch.setenv("ORG_HCI_START_PAUSED", "1")
    try:
        result = main.api_init_world("traffic_watch_v1", warmup=0, clock_speed=0.05)
        assert result["ok"] is True
        assert result["runtime"]["running"] is False
        assert result["runtime"]["interaction_profile"] == "human_project_workspace"
        assert main.SESSION.world.funding_simulation_enabled is False
        assert main.SESSION.world.customer_market_enabled is False
        assert main.SESSION.world.budget_system.funding.tranches == []
        hci_company = build_seat_view(
            main.SESSION.world, "victor"
        )["member"]["company"]
        assert "runway_days" not in hci_company
        assert "budget_pressure" not in hci_company
        assert "cash_balance" not in hci_company
        # Tick zero used to inject tr1 into every HCI world. The project
        # workspace now keeps resource accounting without investor events.
        main.SESSION.step(1)
        assert not [e for e in main.SESSION.world.events
                    if e.get("type") == "funding_event"]
        tick = main._runtime_status()["world_tick"]
        import time
        time.sleep(0.15)
        assert main._runtime_status()["world_tick"] == tick
    finally:
        main.HUMAN.shutdown()
        main.SESSION = original


def test_plain_organization_simulation_keeps_economy_profile():
    """The HCI gate must not alter experiment/default OrgWorld semantics."""
    session = OrgInspectorSession(seed=42, load_llm=False)
    assert session.world.interaction_profile == "organization_simulation"
    assert session.world.funding_simulation_enabled is True
    assert session.world.customer_market_enabled is True
    simulation_company = build_seat_view(
        session.world, "victor"
    )["member"]["company"]
    assert "runway_days" in simulation_company
    assert "budget_pressure" in simulation_company
    session.step(1)
    assert [e for e in session.world.events if e.get("type") == "funding_event"]


def test_inspector_html_has_two_tabs():
    """The inspector is now a React + Vite + React Flow SPA: index.html mounts #root and
    pulls the hashed bundle under /org/app; the tab labels + API endpoints live in the
    built JS bundle (or the still-served legacy template when no build is present)."""
    from environments.org_env.backend import main
    html = main.inspector_html()
    assert '<div id="root">' in html and "/org/app/assets/" in html
    if main.APP_DIST.is_dir():
        blob = "".join(p.read_text(encoding="utf-8") for p in main.APP_DIST.glob("assets/*.js"))
    else:
        blob = main.legacy_inspector_html()
    assert "Company Internal" in blob and "External Network" in blob
    assert "/api/org/lived/frames" in blob and "/api/org/sim/step" in blob


def test_server_routes_via_testclient():
    """End-to-end via FastAPI TestClient (skips if fastapi not installed)."""
    try:
        from fastapi.testclient import TestClient
    except Exception:
        print("  (skip server test: fastapi not installed)")
        return
    from environments.org_env.backend import main
    original = main.SESSION
    # HCI starts at the pack-picker deliberately.  This inspector-route test
    # exercises a preloaded ordinary world, so it must provide that world
    # explicitly instead of relying on the old eager module-global session.
    main.SESSION = OrgInspectorSession(seed=42, load_llm=False)
    try:
        c = TestClient(main.build_app())
        assert c.get("/org/inspector").status_code == 200
        assert c.get("/api/org/lived/full").json()["tick"] == 0
        assert c.post("/api/org/sim/step", json={"n": 3}).json()["tick"] == 3
        assert len(c.get("/api/org/lived/frames?since=0").json()["frames"]) == 3
        assert c.get("/api/org/agents/calvin").json()["agent"]["id"] == "calvin"
        assert len(c.get("/api/org/objects?type=task").json()["objects"]) == 9
        assert c.get("/api/org/external").json()["posts"]
    finally:
        main.HUMAN.shutdown()
        main.SESSION = original


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  ok ", fn.__name__)
    print(f"All {len(fns)} org inspector tests passed!")
