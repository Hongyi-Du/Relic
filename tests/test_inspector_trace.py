from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from relic.replay import TraceError, build_trace, load_trace, validate_trace
from relic.replay.hashing import canonical_sha256


def _frame(sequence: int, tick: int, *, status: str) -> dict:
    event = []
    if sequence:
        event = [
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
            "agents": [
                {
                    "agent_id": "member_1",
                    "display_name": "Member One",
                    "role": "researcher",
                    "active_task_ids": ["task_1"],
                    "status": "active",
                }
            ],
            "tasks": [
                {
                    "task_id": "task_1",
                    "title": "Public task",
                    "status": status,
                    "owner_id": "member_1",
                }
            ],
            "proposals": [],
            "protocols": [],
        },
        "events": event,
        "episodes": [],
        "decisions": [],
        "governance_events": [],
    }


def _trace() -> dict:
    return build_trace(
        run_id="selected-paper-case-test",
        organization_id="paper_org",
        config_digest="a" * 64,
        frames=[_frame(0, 0, status="open"), _frame(1, 4, status="done")],
    )


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _resign(payload: dict) -> dict:
    payload.pop("trace_sha256", None)
    payload["trace_sha256"] = canonical_sha256(payload)
    return payload


@pytest.mark.replay
def test_relic_trace_v1_round_trips_with_digest_and_event_snapshots(tmp_path: Path) -> None:
    trace = _trace()
    path = tmp_path / "trace.json"
    _write(path, trace)

    loaded = load_trace(path)

    assert loaded == trace
    assert loaded["frames"][0]["organization"]["tasks"][0]["status"] == "open"
    assert loaded["frames"][1]["organization"]["tasks"][0]["status"] == "done"
    assert loaded["privacy"] == {
        "private_reflections_included": False,
        "private_memories_included": False,
        "provider_messages_included": False,
    }


@pytest.mark.replay
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda trace: trace["privacy"].__setitem__("private_memories_included", True),
            "must be false",
        ),
        (
            lambda trace: trace["frames"][0].__setitem__("private_memory", {"text": "hidden"}),
            "blocked field",
        ),
        (
            lambda trace: trace["frames"][-1]["organization"]["tasks"][0].__setitem__(
                "description", "Read /home/alice/private.txt"
            ),
            "local filesystem path",
        ),
        (
            lambda trace: trace["frames"][-1]["events"][0]["payload"].__setitem__(
                "summary", "Read /tmp/private.txt"
            ),
            "local filesystem path",
        ),
        (
            lambda trace: trace["frames"][-1]["events"][0]["payload"].__setitem__(
                "summary", r"Read C:\Users\alice\private.txt"
            ),
            "local filesystem path",
        ),
        (
            lambda trace: trace["frames"][-1]["events"][0]["payload"].__setitem__(
                "summary", r"Read \\server\share\private.txt"
            ),
            "local filesystem path",
        ),
        (
            lambda trace: trace["frames"][-1]["organization"]["tasks"][0].__setitem__(
                "description", "Use sk-abcdefghijklmnopqrstuvwxyz"
            ),
            "credential-like",
        ),
        (
            lambda trace: trace["frames"][-1]["decisions"].append(
                {
                    "decision_id": "decision_1",
                    "tick": 4,
                    "agent_id": "member_1",
                    "chosen_action_id": "work",
                    "chosen_object_id": None,
                    "candidates": [{"utility": 1.0}],
                }
            ),
            "blocked field",
        ),
        (
            lambda trace: trace["frames"][-1]["events"][0]["payload"].__setitem__(
                "messages", [{"role": "assistant", "content": "private transcript"}]
            ),
            "blocked field",
        ),
        (
            lambda trace: trace["frames"][-1]["events"][0]["payload"].__setitem__(
                "policy_trace", {"candidate_scores": {"work": 0.9}}
            ),
            "blocked field",
        ),
        (
            lambda trace: trace.__setitem__(
                "evaluation_annotations", {"hidden_tests": ["private evaluator case"]}
            ),
            "blocked field",
        ),
        (
            lambda trace: trace["frames"][-1]["organization"]["agents"][0].__setitem__(
                "local_state", {"private_state": "memory contents"}
            ),
            "blocked field",
        ),
        (
            lambda trace: trace["frames"][-1]["organization"]["agents"][0].__setitem__(
                "local_state", {"mode": "focused"}
            ),
            "must be a string",
        ),
        (
            lambda trace: trace["frames"][-1]["organization"]["agents"][0].__setitem__(
                "local_state", {"apiKey": "test-only-unredacted-value"}
            ),
            "blocked field",
        ),
        (
            lambda trace: trace["frames"][-1]["events"][0]["payload"].__setitem__(
                "metadata", {"label": "untyped container"}
            ),
            "unsupported fields",
        ),
    ],
)
def test_digest_valid_private_sensitive_or_policy_audit_content_is_rejected(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    trace = _trace()
    mutate(trace)
    path = tmp_path / "trace.json"
    _write(path, _resign(trace))

    with pytest.raises(TraceError, match=message):
        load_trace(path)


@pytest.mark.replay
def test_private_events_non_finite_numbers_and_duplicate_keys_are_rejected(
    tmp_path: Path,
) -> None:
    trace = _trace()
    trace["frames"][1]["events"][0]["visibility"] = "private"
    with pytest.raises(TraceError, match="not a public event"):
        validate_trace(_resign(trace))

    trace = _trace()
    trace["frames"][1]["organization"]["tasks"][0]["progress_score"] = float("nan")
    with pytest.raises(TraceError, match="non-finite"):
        validate_trace(trace, verify_digest=False)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"relic-trace-v1","schema_version":"other"}', encoding="utf-8"
    )
    with pytest.raises(TraceError, match="duplicate key"):
        load_trace(duplicate)


@pytest.mark.replay
def test_public_frames_are_single_event_post_event_snapshots() -> None:
    trace = _trace()
    trace["frames"][1]["events"].append(
        {
            "event_id": "event_2",
            "tick": 4,
            "event_type": "task_reopened",
            "actor_id": "member_1",
            "object_ids": ["task_1"],
            "payload": {"summary": "Task was reopened"},
            "visibility": "public",
        }
    )

    with pytest.raises(TraceError, match="at most one post-event snapshot event"):
        validate_trace(_resign(trace))


@pytest.mark.replay
def test_optional_public_records_use_typed_allowlists() -> None:
    trace = _trace()
    frame = trace["frames"][1]
    trace["run_metadata"] = {
        "case_id": "case_public_1",
        "exporter_profile": "public-projection-v1",
        "redaction_reviewed": True,
    }
    frame["organization"]["tasks"][0]["history"] = [
        {"tick": 4, "event": "completed", "agent_id": "member_1"}
    ]
    frame["organization"]["tasks"][0]["progress_evidence"] = ["evidence_1"]
    frame["events"][0]["payload"]["progress"] = 1.0
    frame["organization"]["protocols"] = [
        {
            "protocol_id": "protocol_1",
            "protocol_type": "peer_review",
            "scope": "organization",
            "supporters": ["member_1"],
            "responsible_roles": {"reviewer": ["approve"]},
            "revisions": [
                {
                    "event_id": "revision_1",
                    "tick": 4,
                    "revision_kind": "clarify",
                }
            ],
            "impact_metrics": {"completed_tasks": 1.0},
            "status": "active",
        }
    ]
    frame["episodes"] = [
        {
            "episode_id": "episode_1",
            "episode_type": "work",
            "title": "Public work episode",
            "status": "resolved",
            "participants": ["member_1"],
            "linked_event_ids": ["event_1"],
            "linked_task_ids": ["task_1"],
            "timeline": [
                {
                    "tick": 4,
                    "event_id": "event_1",
                    "event_type": "task_status_changed",
                    "actor_id": "member_1",
                }
            ],
        }
    ]
    frame["governance_events"] = [
        {
            "event_id": "governance_1",
            "event_type": "use",
            "protocol_id": "protocol_1",
            "actor_id": "member_1",
            "tick": 4,
            "data": {"context_id": "task-completion:task_1", "task_id": "task_1"},
        }
    ]
    frame["artifacts"] = [
        {
            "artifact_id": "artifact_1",
            "artifact_type": "document",
            "title": "Public report",
            "related_task_ids": ["task_1"],
        }
    ]
    frame["repo_state"] = {
        "repository_id": "repo_1",
        "branch": "release",
        "pull_requests": [{"pr_id": "pr_1", "status": "open"}],
    }
    frame["evaluation_annotations"] = [
        {
            "evidence_id": "evidence_1",
            "evidence_type": "test",
            "status": "passed",
            "passed": True,
            "related_object_ids": ["task_1"],
        }
    ]

    assert validate_trace(_resign(trace))["run_metadata"]["redaction_reviewed"] is True


@pytest.mark.replay
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda trace: trace["frames"][1]["events"][0].__setitem__("actor_id", "ghost_agent"),
            "references an unpublished object",
        ),
        (
            lambda trace: trace["frames"][1]["events"][0].__setitem__("object_ids", ["ghost_task"]),
            "references an unpublished object",
        ),
        (
            lambda trace: trace["frames"][1]["decisions"].append(
                {
                    "decision_id": "decision_1",
                    "tick": 4,
                    "agent_id": "ghost_agent",
                    "chosen_action_id": "work",
                    "chosen_object_id": None,
                }
            ),
            "references an unpublished object",
        ),
        (
            lambda trace: trace["frames"][1]["organization"]["tasks"][0].__setitem__(
                "task_id", "member_1"
            ),
            "reuses public object id",
        ),
        (
            lambda trace: trace["frames"][1]["organization"]["tasks"][0].__setitem__(
                "owner_id", "ghost_agent"
            ),
            "references an unpublished object",
        ),
    ],
)
def test_public_object_ids_and_typed_references_are_consistent(mutate, message: str) -> None:
    trace = _trace()
    mutate(trace)

    with pytest.raises(TraceError, match=message):
        validate_trace(_resign(trace))


@pytest.mark.replay
def test_count_only_cell_sidecar_is_not_misrepresented_as_an_inspector_trace() -> None:
    count_only = {
        "schema_version": "relic-public-trace-v1",
        "projection_profile": "relic-public-allowlist-v1",
        "cell_id": "cell",
        "frames": [{"tick": 24, "counts": {}}],
    }

    with pytest.raises(TraceError, match="relic-trace-v1"):
        validate_trace(copy.deepcopy(count_only), verify_digest=False)
