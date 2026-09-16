from __future__ import annotations

import copy
import http.client
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from relic.cli import main
from relic.inspector import InspectorError, create_inspector_server
from relic.replay.hashing import canonical_sha256
from relic.replay.trace import build_trace


def _frame(sequence: int, tick: int, status: str) -> dict:
    events = []
    if sequence:
        events = [
            {
                "event_id": f"event_{sequence}",
                "tick": tick,
                "event_type": "task_status_changed",
                "actor_id": "member_1",
                "object_ids": ["task_1"],
                "payload": {"summary": f"Task is now {status}"},
                "visibility": "public",
            }
        ]
    return {
        "frame_id": f"frame_{sequence:06d}",
        "sequence": sequence,
        "tick": tick,
        "organization": {
            "organization_id": "paper_org",
            "name": "Selected public paper case",
            "tick": tick,
            "agents": [{"agent_id": "member_1", "role": "researcher", "status": "active"}],
            "tasks": [{"task_id": "task_1", "title": "Public task", "status": status}],
            "proposals": [],
            "protocols": [],
        },
        "events": events,
        "episodes": [],
        "decisions": [],
        "governance_events": [],
    }


def _trace() -> dict:
    return build_trace(
        run_id="selected-paper-case-test",
        organization_id="paper_org",
        config_digest="a" * 64,
        frames=[_frame(0, 0, "open"), _frame(1, 4, "done")],
    )


def _resign(payload: dict) -> dict:
    payload.pop("trace_sha256", None)
    payload["trace_sha256"] = canonical_sha256(payload)
    return payload


def _write_trace(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@contextmanager
def _running_server(trace_path: Path, *, mode: str = "replay"):
    server = create_inspector_server(trace_path=trace_path, port=0, mode=mode)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(
    server,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    body = response.read()
    headers = {key.lower(): value for key, value in response.getheaders()}
    status = response.status
    connection.close()
    return status, headers, body


@pytest.mark.release
def test_inspector_serves_only_allowlisted_routes_with_security_headers(tmp_path: Path) -> None:
    trace = _trace()
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, trace)

    with _running_server(trace_path) as server:
        status, _, body = _request(server, "/api/health")
        assert status == 200
        health = json.loads(body)
        assert health["status"] == "ready"
        assert health["run_id"] == trace["run_id"]

        status, headers, body = _request(server, "/api/trace")
        assert status == 200
        assert json.loads(body)["trace_sha256"] == trace["trace_sha256"]
        assert "base-uri 'none'" in headers["content-security-policy"]
        assert "frame-ancestors 'none'" in headers["content-security-policy"]
        assert headers["cache-control"] == "no-store"
        assert headers["x-content-type-options"] == "nosniff"
        assert "access-control-allow-origin" not in headers
        assert headers["server"].strip() == "RelicInspector"

        for route, content_type in (
            ("/", "text/html"),
            ("/app.css", "text/css"),
            ("/app.js", "text/javascript"),
        ):
            status, route_headers, route_body = _request(server, route)
            assert status == 200
            assert route_headers["content-type"].startswith(content_type)
            assert route_body

        assert _request(server, "/../pyproject.toml")[0] == 404
        assert _request(server, "/api/unknown")[0] == 404
        assert _request(server, "/api/trace", headers={"Host": "rebind.example"})[0] == 400
        status, method_headers, _ = _request(server, "/api/trace", method="POST")
        assert status == 405
        assert method_headers["allow"] == "GET, HEAD"
        status, _, head_body = _request(server, "/app.js", method="HEAD")
        assert status == 200
        assert head_body == b""


@pytest.mark.replay
def test_live_inspector_keeps_last_verified_append_only_trace(tmp_path: Path) -> None:
    full = _trace()
    full["frames"][0]["organization"]["tasks"][0]["progress_score"] = 1
    _resign(full)
    trace_path = tmp_path / "trace.json"
    prefix = copy.deepcopy(full)
    prefix["frames"] = prefix["frames"][:1]
    _write_trace(trace_path, _resign(prefix))

    with _running_server(trace_path, mode="live") as server:
        initial = json.loads(_request(server, "/api/trace")[2])
        assert len(initial["frames"]) == 1

        trace_path.write_text("{", encoding="utf-8")
        stale = json.loads(_request(server, "/api/trace")[2])
        health = json.loads(_request(server, "/api/health")[2])
        assert stale["trace_sha256"] == initial["trace_sha256"]
        assert health["status"] == "degraded"

        replacement = trace_path.with_suffix(".replacement")
        _write_trace(replacement, full)
        replacement.replace(trace_path)
        refreshed = json.loads(_request(server, "/api/trace")[2])
        assert len(refreshed["frames"]) == len(full["frames"])
        assert json.loads(_request(server, "/api/health")[2])["status"] == "ready"

        rewritten = copy.deepcopy(full)
        rewritten["frames"][0]["organization"]["name"] = "rewritten history"
        _write_trace(trace_path, _resign(rewritten))
        retained = json.loads(_request(server, "/api/trace")[2])
        assert retained["trace_sha256"] == full["trace_sha256"]
        assert json.loads(_request(server, "/api/health")[2])["status"] == "degraded"

        type_rewritten = copy.deepcopy(full)
        type_rewritten["frames"][0]["organization"]["tasks"][0]["progress_score"] = 1.0
        _write_trace(trace_path, _resign(type_rewritten))
        retained = json.loads(_request(server, "/api/trace")[2])
        assert retained["frames"][0]["organization"]["tasks"][0]["progress_score"] == 1
        assert json.loads(_request(server, "/api/health")[2])["status"] == "degraded"


@pytest.mark.release
def test_inspector_validates_before_binding_and_requires_remote_opt_in(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        create_inspector_server(trace_path=invalid, port=0)

    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, _trace())
    with pytest.raises(InspectorError, match="allow-remote"):
        create_inspector_server(trace_path=trace_path, host="0.0.0.0", port=0)

    server = create_inspector_server(
        trace_path=trace_path,
        host="0.0.0.0",
        port=0,
        allow_remote=True,
    )
    server.server_close()


@pytest.mark.release
def test_replay_and_inspect_cli_use_only_validated_public_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, _trace())

    assert main(["replay", "--trace", str(trace_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema_version"] == "relic-replay-summary-v1"
    assert summary["frames"] == 2
    assert summary["run_id"] == "selected-paper-case-test"

    received = {}

    def fake_serve(**kwargs) -> None:
        received.update(kwargs)

    monkeypatch.setattr("relic.inspector.serve_inspector", fake_serve)
    assert (
        main(
            [
                "inspect",
                "--trace",
                str(trace_path),
                "--host",
                "0.0.0.0",
                "--port",
                "9012",
                "--mode",
                "live",
                "--allow-remote",
                "--verbose",
            ]
        )
        == 0
    )
    assert received == {
        "allow_remote": True,
        "host": "0.0.0.0",
        "mode": "live",
        "open_browser": False,
        "port": 9012,
        "trace_path": trace_path,
        "verbose": True,
    }
