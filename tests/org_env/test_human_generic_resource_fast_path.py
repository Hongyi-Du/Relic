"""Published generic resource projections must not wait behind a world tick."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.backend.entities.work import Task
from environments.org_env.backend.repo.repo import ReleaseCandidate
from environments.org_env.human.api import HumanApi
from environments.org_env.human.liaison import LiaisonFacade
from environments.org_env.runtime_adapter.live import OrgInspectorSession


SEAT = "victor"


def test_generic_release_resource_uses_published_view_while_world_lock_is_held():
    session = OrgInspectorSession(seed=42)
    human = HumanApi(lambda: session, seconds_per_tick=60.0)
    facade = LiaisonFacade(human)
    token = facade.session()["token"]
    runtime = human.runtime()
    holder = threading.Event()
    release = threading.Event()
    reader = None
    lock_thread = None
    try:
        world = runtime.world
        candidate = ReleaseCandidate(
            candidate_id="rc_fast_resource", version="0.1", created_by="victor",
            created_at_tick=34, status="blocked", blockers=["gate_ci"],
        )
        world.repo_system.repo.release_candidates[candidate.candidate_id] = candidate
        world.tasks["task_fast_followup"] = Task(
            task_id="task_fast_followup", title="Repair CI gate", owner_id="paul")
        world.events.extend([
            {"type": "release_event", "subtype": "readiness_check", "agent_id": SEAT,
             "candidate_id": candidate.candidate_id, "status": "blocked", "tick": 34,
             "blockers": ["gate_ci"]},
            {"type": "release_event", "subtype": "blocker_to_task", "agent_id": SEAT,
             "candidate_id": candidate.candidate_id, "task_id": "task_fast_followup",
             "owner_id": "paul", "gate": "gate_ci", "tick": 34},
        ])
        runtime.refresh_views()

        state = facade.state(token)
        notice = next(row for row in state["notifications"]
                      if row["kind"] == "release_readiness")
        assert "blocked by 1 gates" in notice["summary"]
        assert "Follow-ups: Paul Dreamer ×1" in notice["summary"]
        resource_ref = notice["resource_ref"]

        def hold_world_lock():
            with runtime.lock:
                holder.set()
                assert release.wait(timeout=3)

        lock_thread = threading.Thread(target=hold_world_lock)
        lock_thread.start()
        assert holder.wait(timeout=1)

        result = {}

        def read_resource():
            result.update(facade.resource(token, resource_ref, "overview"))

        started = time.monotonic()
        reader = threading.Thread(target=read_resource)
        reader.start()
        reader.join(timeout=0.5)
        elapsed = time.monotonic() - started
        assert not reader.is_alive(), "generic resource read waited for the world lock"
        assert elapsed < 0.5
        assert result["content"]["blockers"] == ["gate_ci"]
        assert result["observed_at"]["world_tick"] == 0

        release.set()
        lock_thread.join(timeout=1)
        candidate.blockers = ["gate_docs"]
        world.world_tick = 35
        runtime.refresh_views()
        updated = facade.resource(token, resource_ref, "overview")
        assert updated["content"]["blockers"] == ["gate_docs"]
        assert updated["observed_at"]["world_tick"] == 35

        human.release(token)
        denied = facade.resource(token, resource_ref, "overview")
        assert denied.get("error")
    finally:
        release.set()
        if reader is not None:
            reader.join(timeout=1)
        if lock_thread is not None:
            lock_thread.join(timeout=1)
        human.shutdown()
