"""The model-backed P3 API must not block the ASGI event loop."""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_blocked_liaison_model_call_does_not_freeze_runtime_polling(monkeypatch):
    """A slow semantic-router call must leave other P3 tabs responsive."""
    from environments.org_env.backend import main

    entered = threading.Event()
    release = threading.Event()

    def blocked_ask(token, text, *, reply_to="", thread_id="", decision_context=None):
        entered.set()
        assert release.wait(2.0)
        return {"accepted": True, "token": token, "text": text}

    monkeypatch.setattr(main.LIAISON, "ask", blocked_ask)
    monkeypatch.setattr(main.LIAISON, "runtime_status", lambda: {"running": True})

    async def scenario():
        app = main.build_app()
        endpoints = {
            route.path: route.endpoint for route in app.routes
            if getattr(route, "path", None) in {
                "/api/org/liaison/ask", "/api/org/liaison/runtime"}
        }
        started_at = time.monotonic()
        ask_task = asyncio.create_task(endpoints["/api/org/liaison/ask"]({
            "token": "victor-token", "text": "inspect the project",
        }))
        assert await asyncio.to_thread(entered.wait, 0.5)
        runtime = await asyncio.wait_for(
            endpoints["/api/org/liaison/runtime"](), timeout=0.5)
        elapsed = time.monotonic() - started_at
        assert runtime["running"] is True
        assert elapsed < 0.5
        release.set()
        response = await asyncio.wait_for(ask_task, timeout=1.0)
        assert response["accepted"] is True

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_world_locking_liaison_reads_do_not_freeze_runtime_polling(monkeypatch):
    """Session resume and resource inspection may wait for a tick's world lock."""
    from environments.org_env.backend import main

    monkeypatch.setattr(main.LIAISON, "runtime_status", lambda: {"running": True})

    for method_name, path in (
        ("session", "/api/org/liaison/session"),
        ("state", "/api/org/liaison/state"),
        ("resource", "/api/org/liaison/resource"),
    ):
        entered = threading.Event()
        release = threading.Event()

        def blocked_read(*args, **kwargs):
            entered.set()
            assert release.wait(2.0)
            return {"method": method_name}

        monkeypatch.setattr(main.LIAISON, method_name, blocked_read)

        async def scenario():
            app = main.build_app()
            endpoints = {
                route.path: route.endpoint for route in app.routes
                if getattr(route, "path", None) in {
                    path, "/api/org/liaison/runtime"}
            }
            if method_name == "session":
                read_task = asyncio.create_task(endpoints[path]({"token": "victor-token"}))
            elif method_name == "state":
                read_task = asyncio.create_task(endpoints[path](
                    token="victor-token", since=0))
            else:
                read_task = asyncio.create_task(endpoints[path](
                    token="victor-token", ref="ri_visible", section="overview"))
            assert await asyncio.to_thread(entered.wait, 0.5)
            runtime = await asyncio.wait_for(
                endpoints["/api/org/liaison/runtime"](), timeout=0.5)
            assert runtime["running"] is True
            release.set()
            response = await asyncio.wait_for(read_task, timeout=1.0)
            assert response["method"] == method_name

        try:
            asyncio.run(scenario())
        finally:
            release.set()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
