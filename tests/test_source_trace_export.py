from __future__ import annotations

import sys
from pathlib import Path

import pytest

from environments.org_env.runtime_adapter.live import OrgInspectorSession
from relic.inspector.server import TraceSource
from relic.replay import load_trace
from relic.replay.source_export import (
    SourceTraceExportError,
    append_run_record_evidence,
    copy_public_trace_prefix,
    export_source_world_trace,
    public_trace_path,
)


def _context(session: OrgInspectorSession) -> dict[str, object]:
    return {
        "case_id": "source-public-trace-test",
        "condition": session.world.experiment_condition,
        "benchmark": "mini_blobstore_v1",
        "model": "rules",
        "seed": 7,
        "source_commit": "a" * 40,
    }


def _session() -> OrgInspectorSession:
    return OrgInspectorSession(seed=7, load_llm=False)


@pytest.mark.replay
def test_source_projection_is_live_append_only_and_never_reads_private_state(tmp_path: Path) -> None:
    session = _session()
    world = session.world
    secret = "sk-private-canary-never-publish"
    world.messages = [{"body": secret}]
    world.memory = {"paul": [{"secret": secret}]}
    world.events.append(
        {
            "type": "repo_event",
            "subtype": "ci",
            "agent_id": "paul",
            "tick": 0,
            "summary": secret,
        }
    )
    path = public_trace_path(tmp_path / "run")

    first = export_source_world_trace(world, destination=path, context=_context(session))
    original = load_trace(path)
    source = TraceSource(path, "live")
    assert first.frames == 1
    assert secret not in path.read_text(encoding="utf-8")
    assert original["privacy"] == {
        "private_reflections_included": False,
        "private_memories_included": False,
        "provider_messages_included": False,
    }
    assert original["frames"][0]["events"][0]["event_type"] == "repo_event"
    assert original["frames"][0]["events"][0]["payload"] == {}

    session.step(1)
    update = export_source_world_trace(world, destination=path, context=_context(session))
    refreshed = source.read()

    assert update.frames > first.frames
    assert refreshed["frames"][: len(original["frames"])] == original["frames"]
    assert source.degraded is False
    assert refreshed["frames"][-1]["tick"] == 1
    assert all("messages" not in frame for frame in refreshed["frames"])


@pytest.mark.replay
def test_source_projection_rejects_history_gaps_and_missing_resume_prefix(tmp_path: Path) -> None:
    session = _session()
    path = public_trace_path(tmp_path / "missing")
    session.step(1)
    with pytest.raises(SourceTraceExportError, match="public_trace_history_missing"):
        export_source_world_trace(session.world, destination=path, context=_context(session))

    session = _session()
    path = public_trace_path(tmp_path / "gap")
    export_source_world_trace(session.world, destination=path, context=_context(session))
    session.step(2)
    with pytest.raises(SourceTraceExportError, match="public_trace_tick_gap"):
        export_source_world_trace(session.world, destination=path, context=_context(session))


@pytest.mark.replay
def test_source_projection_resume_copy_keeps_the_verified_prefix(tmp_path: Path) -> None:
    session = _session()
    context = _context(session)
    original_path = public_trace_path(tmp_path / "original")
    export_source_world_trace(session.world, destination=original_path, context=context)
    session.step(1)
    export_source_world_trace(session.world, destination=original_path, context=context)
    original = load_trace(original_path)

    resumed_path = public_trace_path(tmp_path / "resumed")
    copy_public_trace_prefix(original_path, resumed_path)
    session.step(1)
    export_source_world_trace(session.world, destination=resumed_path, context=context)
    resumed = load_trace(resumed_path)

    assert resumed["frames"][: len(original["frames"])] == original["frames"]
    assert len(resumed["frames"]) > len(original["frames"])
    assert resumed["frames"][-1]["tick"] == 2


@pytest.mark.replay
def test_run_record_contributes_only_allowlisted_public_evidence(tmp_path: Path) -> None:
    session = _session()
    path = public_trace_path(tmp_path / "run")
    export_source_world_trace(session.world, destination=path, context=_context(session))
    secret = "sk-private-canary-never-publish"
    record = {
        "status": "failed",
        "failure_reason": secret,
        "provenance": {"private": secret},
        "metrics": {
            "merged_pr_count": 2.0,
            "unreleased_metric": 99.0,
        },
        "final_evaluation": {
            "status": "failed",
            "formal_claim_ready": False,
            "raw_output": secret,
        },
    }

    export_source_world_trace(
        session.world,
        destination=path,
        context=_context(session),
        run_record=record,
    )
    append_run_record_evidence(path, {**record, "status": "blocked"})
    trace = load_trace(path)
    text = path.read_text(encoding="utf-8")
    evidence = [
        item
        for frame in trace["frames"]
        for item in frame.get("evaluation_annotations", [])
    ]

    assert secret not in text
    assert {item["evidence_id"] for item in evidence} >= {
        "source_run_status_failed",
        "source_run_status_blocked",
        "source_final_evaluation_failed",
        "source_metric_merged_pr_count",
    }
    assert "source_metric_unreleased_metric" not in {item["evidence_id"] for item in evidence}


@pytest.mark.integration
def test_source_runner_writes_valid_public_trace_per_new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import org_inspector_replay

    monkeypatch.setattr(org_inspector_replay, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("ORG_LOG_ZIP", "0")
    monkeypatch.setenv("ORG_FIGURE_METRICS", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        ["org_inspector_replay.py", "public-export", "1", "7"],
    )

    org_inspector_replay.main()

    traces = list(tmp_path.glob("log/*/public/relic-trace-v1.json"))
    assert len(traces) == 1
    trace = load_trace(traces[0])
    assert trace["schema_version"] == "relic-trace-v1"
    assert trace["run_metadata"]["exporter_profile"] == (
        "source-runner-explicit-public-state-v1"
    )
    assert trace["frames"][-1]["tick"] == 1
