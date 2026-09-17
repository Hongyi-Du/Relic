"""The fail-closed public ``relic-trace-v1`` contract."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from relic.replay.hashing import canonical_sha256

TRACE_SCHEMA_VERSION = "relic-trace-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CREDENTIAL_VALUE_RE = re.compile(
    r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]{12,}|\bsk-[A-Za-z0-9_-]{12,}|"
    r"\bgh[opusr]_[A-Za-z0-9_]{12,}|\bgithub_pat_[A-Za-z0-9_]{12,}|"
    r"\bAKIA[0-9A-Z]{16}\b)",
    re.IGNORECASE,
)
_LOCAL_PATH_RE = re.compile(
    r"(?:^|[\s\"'=:(])(?:/(?!/)[^\s\"'<>]+|[A-Za-z]:[\\/][^\s\"'<>]+|"
    r"\\\\[^\\\s]+\\[^\\\s]+|file://[^\s\"'<>]+)",
    re.IGNORECASE,
)
_BLOCKED_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "candidate",
        "candidate_scores",
        "candidates",
        "checkpoint",
        "checkpoint_path",
        "cookie",
        "credential",
        "credentials",
        "feature",
        "feature_vector",
        "feature_vectors",
        "features",
        "heldout_test",
        "heldout_tests",
        "hidden_test",
        "hidden_tests",
        "memories",
        "memory",
        "message",
        "messages",
        "model_message",
        "model_messages",
        "password",
        "policy",
        "policy_audit",
        "policy_trace",
        "private_memory",
        "private_memories",
        "private_context",
        "private_state",
        "private_workspace",
        "prompt",
        "prompts",
        "provider_message",
        "provider_messages",
        "provider_response",
        "provider_responses",
        "rationale",
        "raw_reflection",
        "raw_reflection_excerpt",
        "raw_response",
        "raw_responses",
        "raw_state",
        "reasoning",
        "reflection_text",
        "runtime_dump",
        "secret",
        "system_prompt",
        "transcript",
        "transcripts",
        "utilities",
        "utility",
    }
)
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "organization_id",
        "config_sha256",
        "privacy",
        "frames",
        "evaluation_annotations",
        "run_metadata",
        "trace_sha256",
    }
)
_PRIVACY_KEYS = frozenset(
    {
        "private_reflections_included",
        "private_memories_included",
        "provider_messages_included",
    }
)
_FRAME_KEYS = frozenset(
    {
        "frame_id",
        "sequence",
        "tick",
        "organization",
        "events",
        "episodes",
        "decisions",
        "governance_events",
        "artifacts",
        "repo_state",
        "evaluation_annotations",
    }
)
_ORGANIZATION_KEYS = frozenset(
    {
        "organization_id",
        "name",
        "tick",
        "agents",
        "tasks",
        "proposals",
        "protocols",
        "artifacts",
        "repo_state",
        "evidence",
    }
)
_EVENT_KEYS = frozenset(
    {"event_id", "tick", "event_type", "actor_id", "object_ids", "payload", "visibility"}
)
_DECISION_KEYS = frozenset(
    {"decision_id", "tick", "agent_id", "chosen_action_id", "chosen_object_id"}
)
_GOVERNANCE_EVENT_KEYS = frozenset(
    {"event_id", "event_type", "protocol_id", "actor_id", "tick", "data"}
)
_PRIVATE_EVENT_TYPES = frozenset({"reflection_completed", "wish_created"})

_AGENT_FIELD_KINDS = {
    "agent_id": "string",
    "display_name": "string",
    "role": "string",
    "tools": "strings",
    "active_task_ids": "strings",
    "current_task_id": "string",
    "status": "string",
    "local_state": "string",
    "known_context": "strings",
    "recent_decisions": "strings",
    "obligations": "strings",
    "relevant_protocol_ids": "strings",
}
_TASK_FIELD_KINDS = {
    "task_id": "string",
    "title": "string",
    "description": "string",
    "status": "string",
    "priority": "scalar",
    "owner_id": "string",
    "required_skills": "strings",
    "dependencies": "strings",
    "deadline_tick": "integer",
    "estimated_effort": "number",
    "actual_effort": "number",
    "visibility": "string",
    "progress_score": "number",
    "progress_evidence": "strings",
    "history": "task_history",
    "blockers": "strings",
    "related_artifact_ids": "strings",
    "related_protocol_ids": "strings",
}
_PROPOSAL_FIELD_KINDS = {
    "proposal_id": "string",
    "proposal_type": "string",
    "title": "string",
    "summary": "string",
    "proposer_agent_id": "string",
    "source_wish_id": "string",
    "source_wish_ids": "strings",
    "source_reflection_id": "string",
    "source_episode_id": "string",
    "source_episode_ids": "strings",
    "source_event_ids": "strings",
    "target_problem": "string",
    "proposed_solution": "string",
    "required_actions": "strings",
    "required_capabilities": "strings",
    "required_artifacts": "strings",
    "required_participants": "strings",
    "affected_agents": "strings",
    "affected_objects": "strings",
    "affected_protocols": "strings",
    "expected_benefits": "strings",
    "expected_costs": "strings",
    "risks": "strings",
    "failure_modes": "strings",
    "feasibility_score": "number",
    "usefulness_score": "number",
    "risk_score": "number",
    "adoption_score": "number",
    "suggested_revision": "string",
    "family": "string",
    "status": "string",
    "approval_required_from": "strings",
    "approved_by": "strings",
    "supporters": "strings",
    "rejected_by": "strings",
    "opposers": "strings",
    "rejection_reason": "string",
    "object_created_id": "string",
    "adopted_tick": "integer",
    "impact": "number_mapping",
    "created_at_tick": "integer",
    "updated_at_tick": "integer",
    "amends_protocol_id": "string",
    "repair_kind": "string",
    "repair_target_protocol_id": "string",
}
_PROTOCOL_FIELD_KINDS = {
    "protocol_id": "string",
    "protocol_type": "string",
    "name": "string",
    "proposer_id": "string",
    "created_from_proposal_id": "string",
    "proposal_event_id": "string",
    "rule_summary": "string",
    "trigger_condition": "string",
    "required_steps": "strings",
    "required_fields": "strings",
    "enforcement_rule": "string",
    "violation_condition": "string",
    "exception_rule": "string",
    "scope": "string",
    "affected_agents": "strings",
    "affected_actions": "strings",
    "affected_artifacts": "strings",
    "responsible_roles": "string_lists_mapping",
    "success_metric": "string",
    "enforcement_action": "string",
    "sunset_rule": "string",
    "target_process": "string",
    "supporters": "strings",
    "opposers": "strings",
    "adoption_status": "string",
    "usage_events": "strings",
    "violation_events": "strings",
    "enforcement_events": "strings",
    "revisions": "protocol_revisions",
    "revision": "string",
    "version": "scalar",
    "first_tick": "integer",
    "last_active_tick": "integer",
    "retired_tick": "integer",
    "persistence_ticks": "integer",
    "impact_metrics": "number_mapping",
    "emergence_level": "scalar",
    "status": "string",
}
_EPISODE_FIELD_KINDS = {
    "episode_id": "string",
    "episode_type": "string",
    "title": "string",
    "status": "string",
    "start_tick": "integer",
    "end_tick": "integer",
    "participants": "strings",
    "linked_event_ids": "strings",
    "linked_task_ids": "strings",
    "linked_protocol_ids": "strings",
    "linked_proposal_ids": "strings",
    "problem_statement": "string",
    "decision_summary": "string",
    "outcome_summary": "string",
    "produced_protocols": "strings",
    "timeline": "episode_timeline",
}
_EVENT_PAYLOAD_FIELD_KINDS = {
    "summary": "string",
    "title": "string",
    "status": "string",
    "outcome": "string",
    "change_type": "string",
    "progress": "number",
    "related_object_ids": "strings",
}
_GOVERNANCE_DATA_FIELD_KINDS = {
    "summary": "string",
    "status": "string",
    "changes_requested": "strings",
    "related_object_ids": "strings",
    "supporter_ids": "strings",
    "opposer_ids": "strings",
    "approved_by": "strings",
    "rejected_by": "strings",
    "amendment_id": "string",
    "context_id": "string",
    "episode_id": "string",
    "task_id": "string",
    "violation_event_id": "string",
    "state_impact_ref": "string",
    "blocked": "boolean",
    "state_before_hash": "string",
    "state_after_hash": "string",
    "revision_kind": "string",
    "source_proposal_id": "string",
    "previous_rule_summary": "string",
    "rule_summary": "string",
}
_TASK_HISTORY_FIELD_KINDS = {
    "tick": "integer",
    "event": "string",
    "agent_id": "string",
    "protocol_id": "string",
}
_EPISODE_TIMELINE_FIELD_KINDS = {
    "tick": "integer",
    "event_id": "string",
    "event_type": "string",
    "actor_id": "string",
}
_PROTOCOL_REVISION_FIELD_KINDS = {
    "event_id": "string",
    "tick": "integer",
    "revision_kind": "string",
    "source_proposal_id": "string",
    "previous_rule_summary": "string",
    "rule_summary": "string",
}
_ARTIFACT_FIELD_KINDS = {
    "artifact_id": "string",
    "artifact_type": "string",
    "title": "string",
    "summary": "string",
    "status": "string",
    "version": "scalar",
    "owner_id": "string",
    "created_tick": "integer",
    "updated_tick": "integer",
    "related_task_ids": "strings",
    "related_protocol_ids": "strings",
}
_EVIDENCE_FIELD_KINDS = {
    "evidence_id": "string",
    "evidence_type": "string",
    "title": "string",
    "summary": "string",
    "status": "string",
    "source": "string",
    "metric_name": "string",
    "metric_value": "scalar",
    "unit": "string",
    "tick": "integer",
    "related_object_ids": "strings",
    "passed": "boolean",
    "available": "boolean",
}
_RUN_METADATA_FIELD_KINDS = {
    "title": "string",
    "summary": "string",
    "case_id": "string",
    "cell_id": "string",
    "condition": "string",
    "benchmark": "string",
    "model": "string",
    "seed": "integer",
    "exporter_version": "string",
    "exporter_profile": "string",
    "source_commit": "string",
    "redaction_reviewed": "boolean",
}
_REPO_STATE_FIELD_KINDS = {
    "repository_id": "string",
    "name": "string",
    "branch": "string",
    "commit_sha": "string",
    "merge_state": "string",
}
_REPO_COLLECTION_KEYS = {
    "branches": "branch_id",
    "commits": "commit_id",
    "pull_requests": "pr_id",
    "reviews": "review_id",
    "ci_runs": "ci_id",
    "merges": "result_id",
}
_REPO_IDENTIFIER_KEYS = frozenset(_REPO_COLLECTION_KEYS.values())
_REPO_RECORD_FIELD_KINDS = {
    "branch_id": "string",
    "commit_id": "string",
    "pr_id": "string",
    "review_id": "string",
    "ci_id": "string",
    "result_id": "string",
    "title": "string",
    "name": "string",
    "status": "string",
    "sha": "string",
    "ref": "string",
    "actor_id": "string",
    "tick": "integer",
    "related_object_ids": "strings",
    "related_protocol_ids": "strings",
}


class TraceError(ValueError):
    """Raised when a public trace does not satisfy the release contract."""


def _bad_number(value: str) -> None:
    raise TraceError(f"trace contains non-finite number: {value}")


def _load_json_without_duplicate_keys(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TraceError(f"trace contains duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=_bad_number)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TraceError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TraceError(f"{label} must be a list")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TraceError(f"{label} must be a non-empty string")
    return value


def _require_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TraceError(f"{label} must be an integer >= {minimum}")
    return value


def _reject_unknown_keys(value: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise TraceError(f"{label} contains a non-string field name")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise TraceError(f"{label} contains unsupported fields: {', '.join(unknown)}")


def _validate_finite_numbers(value: Any, path: str = "trace") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TraceError(f"{path} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_numbers(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_numbers(item, f"{path}[{index}]")


def _validate_public_values(value: Any, path: str = "trace") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TraceError(f"public trace contains a non-string field name at {path}")
            normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
            normalized = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
            if normalized in _BLOCKED_KEYS:
                raise TraceError(f"public trace contains blocked field at {path}.{key}")
            _validate_public_values(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_public_values(item, f"{path}[{index}]")
        return
    if isinstance(value, str):
        if _CREDENTIAL_VALUE_RE.search(value):
            raise TraceError(f"public trace contains credential-like text at {path}")
        if _LOCAL_PATH_RE.search(value):
            raise TraceError(f"public trace contains a local filesystem path at {path}")


def _validate_string_list(value: Any, label: str) -> None:
    items = _require_list(value, label)
    seen: set[str] = set()
    for index, item in enumerate(items):
        text = _require_string(item, f"{label}[{index}]")
        if text in seen:
            raise TraceError(f"{label} contains a duplicate value: {text}")
        seen.add(text)


def _validate_field_kind(value: Any, label: str, kind: str) -> None:
    if value is None:
        return
    if kind == "string":
        if not isinstance(value, str):
            raise TraceError(f"{label} must be a string")
        return
    if kind == "strings":
        _validate_string_list(value, label)
        return
    if kind == "integer":
        _require_integer(value, label)
        return
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TraceError(f"{label} contains a non-finite or non-numeric value")
        if isinstance(value, float) and not math.isfinite(value):
            raise TraceError(f"{label} contains a non-finite or non-numeric value")
        return
    if kind == "boolean":
        if not isinstance(value, bool):
            raise TraceError(f"{label} must be a boolean")
        return
    if kind == "number_mapping":
        mapping = _require_mapping(value, label)
        for key, item in mapping.items():
            _require_string(key, f"{label} field name")
            _validate_field_kind(item, f"{label}.{key}", "number")
        return
    if kind == "string_lists_mapping":
        mapping = _require_mapping(value, label)
        for key, item in mapping.items():
            _require_string(key, f"{label} field name")
            _validate_string_list(item, f"{label}.{key}")
        return
    nested_record_kinds = {
        "task_history": _TASK_HISTORY_FIELD_KINDS,
        "episode_timeline": _EPISODE_TIMELINE_FIELD_KINDS,
        "protocol_revisions": _PROTOCOL_REVISION_FIELD_KINDS,
    }
    if kind in nested_record_kinds:
        for index, item in enumerate(_require_list(value, label)):
            _validate_typed_record(
                item,
                label=f"{label}[{index}]",
                field_kinds=nested_record_kinds[kind],
            )
        return
    if kind == "scalar":
        if isinstance(value, str):
            _require_string(value, label)
            return
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            return
        if isinstance(value, float) and math.isfinite(value):
            return
        raise TraceError(f"{label} must be a public JSON scalar")
    raise AssertionError(f"unknown trace field kind: {kind}")


def _validate_typed_record(
    value: Any,
    *,
    label: str,
    field_kinds: Mapping[str, str],
) -> dict[str, Any]:
    record = _require_mapping(value, label)
    _reject_unknown_keys(record, frozenset(field_kinds), label)
    for key, item in record.items():
        _validate_field_kind(item, f"{label}.{key}", field_kinds[key])
    return record


def _validate_object_collection(
    value: Any,
    *,
    label: str,
    identifier: str,
    field_kinds: Mapping[str, str],
) -> set[str]:
    seen: set[str] = set()
    for index, item in enumerate(_require_list(value, label)):
        record = _validate_typed_record(
            item,
            label=f"{label}[{index}]",
            field_kinds=field_kinds,
        )
        object_id = _require_string(record.get(identifier), f"{label}[{index}].{identifier}")
        if object_id in seen:
            raise TraceError(f"{label} contains duplicate {identifier}: {object_id}")
        seen.add(object_id)
    return seen


def _validate_artifacts(value: Any, label: str) -> set[str]:
    return _validate_object_collection(
        value,
        label=label,
        identifier="artifact_id",
        field_kinds=_ARTIFACT_FIELD_KINDS,
    )


def _validate_evidence(value: Any, label: str) -> set[str]:
    return _validate_object_collection(
        value,
        label=label,
        identifier="evidence_id",
        field_kinds=_EVIDENCE_FIELD_KINDS,
    )


def _validate_repo_state(value: Any, label: str) -> set[str]:
    repo = _require_mapping(value, label)
    allowed = frozenset(_REPO_STATE_FIELD_KINDS) | frozenset(_REPO_COLLECTION_KEYS)
    _reject_unknown_keys(repo, allowed, label)
    for key, kind in _REPO_STATE_FIELD_KINDS.items():
        if key in repo:
            _validate_field_kind(repo[key], f"{label}.{key}", kind)
    repository_id = _require_string(repo.get("repository_id"), f"{label}.repository_id")
    object_ids = {repository_id}
    for collection, identifier in _REPO_COLLECTION_KEYS.items():
        if collection not in repo:
            continue
        record_ids = _validate_object_collection(
            repo[collection],
            label=f"{label}.{collection}",
            identifier=identifier,
            field_kinds={
                key: kind
                for key, kind in _REPO_RECORD_FIELD_KINDS.items()
                if key not in _REPO_IDENTIFIER_KEYS or key == identifier
            },
        )
        collision = object_ids & record_ids
        if collision:
            raise TraceError(f"{label} reuses public object id: {sorted(collision)[0]}")
        object_ids.update(record_ids)
    return object_ids


def _merge_object_ids(target: set[str], incoming: set[str], label: str) -> None:
    collision = target & incoming
    if collision:
        raise TraceError(f"{label} reuses public object id: {sorted(collision)[0]}")
    target.update(incoming)


def _require_known_reference(value: Any, label: str, known: set[str]) -> str:
    reference = _require_string(value, label)
    if reference not in known:
        raise TraceError(f"{label} references an unpublished object: {reference}")
    return reference


def _require_known_references(value: Any, label: str, known: set[str]) -> list[Any]:
    references = _require_list(value, label)
    _validate_string_list(references, label)
    for index, reference in enumerate(references):
        if reference not in known:
            raise TraceError(f"{label}[{index}] references an unpublished object: {reference}")
    return references


def _validate_optional_reference(
    record: Mapping[str, Any],
    key: str,
    label: str,
    known: set[str],
) -> None:
    if record.get(key) is not None:
        _require_known_reference(record[key], f"{label}.{key}", known)


def _validate_optional_references(
    record: Mapping[str, Any],
    key: str,
    label: str,
    known: set[str],
) -> None:
    if record.get(key) is not None:
        _require_known_references(record[key], f"{label}.{key}", known)


def _validate_structure(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise TraceError(f"trace must use {TRACE_SCHEMA_VERSION}")
    _reject_unknown_keys(payload, _TOP_LEVEL_KEYS, "trace")
    _require_string(payload.get("run_id"), "trace.run_id")
    organization_id = _require_string(payload.get("organization_id"), "trace.organization_id")
    config_sha256 = _require_string(payload.get("config_sha256"), "trace.config_sha256")
    if not _SHA256_RE.fullmatch(config_sha256):
        raise TraceError("trace.config_sha256 must be a lowercase SHA-256 digest")

    privacy = _require_mapping(payload.get("privacy"), "trace.privacy")
    _reject_unknown_keys(privacy, _PRIVACY_KEYS, "trace.privacy")
    if set(privacy) != _PRIVACY_KEYS:
        raise TraceError("trace.privacy must declare every public privacy boundary")
    for key in sorted(_PRIVACY_KEYS):
        if privacy[key] is not False:
            raise TraceError(f"trace.privacy.{key} must be false")

    if "run_metadata" in payload:
        _validate_typed_record(
            payload["run_metadata"],
            label="trace.run_metadata",
            field_kinds=_RUN_METADATA_FIELD_KINDS,
        )
    if "evaluation_annotations" in payload:
        _validate_evidence(payload["evaluation_annotations"], "trace.evaluation_annotations")

    frames = _require_list(payload.get("frames"), "trace.frames")
    if not frames:
        raise TraceError("trace.frames must be a non-empty list")
    last_tick = -1
    seen_events: set[str] = set()
    seen_governance_events: set[str] = set()
    seen_decisions: set[str] = set()
    for index, item in enumerate(frames):
        label = f"trace.frames[{index}]"
        frame = _require_mapping(item, label)
        _reject_unknown_keys(frame, _FRAME_KEYS, label)
        required = {
            "frame_id",
            "sequence",
            "tick",
            "organization",
            "events",
            "episodes",
            "decisions",
            "governance_events",
        }
        if not required <= set(frame):
            raise TraceError(f"{label} is missing required public snapshot fields")
        if frame["frame_id"] != f"frame_{index:06d}":
            raise TraceError(f"{label}.frame_id must match its sequence")
        if _require_integer(frame["sequence"], f"{label}.sequence") != index:
            raise TraceError(f"{label}.sequence must be contiguous from zero")
        tick = _require_integer(frame["tick"], f"{label}.tick")
        if tick < last_tick:
            raise TraceError("trace frame ticks must be non-decreasing")
        last_tick = tick

        organization = _require_mapping(frame["organization"], f"{label}.organization")
        _reject_unknown_keys(organization, _ORGANIZATION_KEYS, f"{label}.organization")
        if organization.get("organization_id") != organization_id:
            raise TraceError(f"{label}.organization_id does not match the trace")
        if _require_integer(organization.get("tick"), f"{label}.organization.tick") != tick:
            raise TraceError(f"{label}.organization.tick does not match the frame")
        if "name" in organization:
            _require_string(organization["name"], f"{label}.organization.name")
        for key in ("agents", "tasks", "proposals", "protocols"):
            if key not in organization:
                raise TraceError(f"{label}.organization.{key} is required")
        agent_ids = _validate_object_collection(
            organization["agents"],
            label=f"{label}.organization.agents",
            identifier="agent_id",
            field_kinds=_AGENT_FIELD_KINDS,
        )
        task_ids = _validate_object_collection(
            organization["tasks"],
            label=f"{label}.organization.tasks",
            identifier="task_id",
            field_kinds=_TASK_FIELD_KINDS,
        )
        proposal_ids = _validate_object_collection(
            organization["proposals"],
            label=f"{label}.organization.proposals",
            identifier="proposal_id",
            field_kinds=_PROPOSAL_FIELD_KINDS,
        )
        protocol_ids = _validate_object_collection(
            organization["protocols"],
            label=f"{label}.organization.protocols",
            identifier="protocol_id",
            field_kinds=_PROTOCOL_FIELD_KINDS,
        )

        public_object_ids = {organization_id}
        for object_ids in (agent_ids, task_ids, proposal_ids, protocol_ids):
            _merge_object_ids(public_object_ids, object_ids, label)

        episode_ids = _validate_object_collection(
            frame["episodes"],
            label=f"{label}.episodes",
            identifier="episode_id",
            field_kinds=_EPISODE_FIELD_KINDS,
        )
        _merge_object_ids(public_object_ids, episode_ids, label)

        if "artifacts" in frame and "artifacts" in organization:
            raise TraceError(f"{label} publishes artifacts in two locations")
        artifact_value = frame.get("artifacts", organization.get("artifacts"))
        if artifact_value is not None:
            artifact_ids = _validate_artifacts(artifact_value, f"{label}.artifacts")
            _merge_object_ids(public_object_ids, artifact_ids, label)

        if "repo_state" in frame and "repo_state" in organization:
            raise TraceError(f"{label} publishes repo_state in two locations")
        repo_value = frame.get("repo_state", organization.get("repo_state"))
        if repo_value is not None:
            repo_ids = _validate_repo_state(repo_value, f"{label}.repo_state")
            _merge_object_ids(public_object_ids, repo_ids, label)

        if "evaluation_annotations" in frame and "evidence" in organization:
            raise TraceError(f"{label} publishes evaluation evidence in two locations")
        evidence_value = frame.get("evaluation_annotations", organization.get("evidence"))
        if evidence_value is not None:
            evidence_ids = _validate_evidence(evidence_value, f"{label}.evaluation_annotations")
            _merge_object_ids(public_object_ids, evidence_ids, label)

        for agent_index, agent in enumerate(organization["agents"]):
            agent_label = f"{label}.organization.agents[{agent_index}]"
            _validate_optional_references(agent, "active_task_ids", agent_label, task_ids)
            _validate_optional_reference(agent, "current_task_id", agent_label, task_ids)
            _validate_optional_references(
                agent,
                "relevant_protocol_ids",
                agent_label,
                protocol_ids,
            )
        for task_index, task in enumerate(organization["tasks"]):
            task_label = f"{label}.organization.tasks[{task_index}]"
            _validate_optional_reference(task, "owner_id", task_label, agent_ids)
            _validate_optional_references(task, "dependencies", task_label, task_ids)
            _validate_optional_references(
                task,
                "related_artifact_ids",
                task_label,
                artifact_ids if artifact_value is not None else set(),
            )
            _validate_optional_references(
                task,
                "related_protocol_ids",
                task_label,
                protocol_ids,
            )
        for proposal_index, proposal in enumerate(organization["proposals"]):
            proposal_label = f"{label}.organization.proposals[{proposal_index}]"
            _validate_optional_reference(
                proposal,
                "proposer_agent_id",
                proposal_label,
                agent_ids,
            )
            for key in (
                "affected_agents",
                "approval_required_from",
                "approved_by",
                "supporters",
                "rejected_by",
                "opposers",
            ):
                _validate_optional_references(proposal, key, proposal_label, agent_ids)
            _validate_optional_references(
                proposal,
                "affected_protocols",
                proposal_label,
                protocol_ids,
            )
        for protocol_index, protocol in enumerate(organization["protocols"]):
            protocol_label = f"{label}.organization.protocols[{protocol_index}]"
            _validate_optional_reference(protocol, "proposer_id", protocol_label, agent_ids)
            _validate_optional_reference(
                protocol,
                "created_from_proposal_id",
                protocol_label,
                proposal_ids,
            )
            for key in ("affected_agents", "supporters", "opposers"):
                _validate_optional_references(protocol, key, protocol_label, agent_ids)
        if artifact_value is not None:
            for artifact_index, artifact in enumerate(artifact_value):
                artifact_label = f"{label}.artifacts[{artifact_index}]"
                _validate_optional_reference(artifact, "owner_id", artifact_label, agent_ids)
                _validate_optional_references(
                    artifact,
                    "related_task_ids",
                    artifact_label,
                    task_ids,
                )
                _validate_optional_references(
                    artifact,
                    "related_protocol_ids",
                    artifact_label,
                    protocol_ids,
                )
        if evidence_value is not None:
            for evidence_index, evidence in enumerate(evidence_value):
                _validate_optional_references(
                    evidence,
                    "related_object_ids",
                    f"{label}.evaluation_annotations[{evidence_index}]",
                    public_object_ids,
                )

        events = _require_list(frame["events"], f"{label}.events")
        if len(events) > 1:
            raise TraceError(f"{label}.events may contain at most one post-event snapshot event")
        for event_index, event_item in enumerate(events):
            event_label = f"{label}.events[{event_index}]"
            event = _require_mapping(event_item, event_label)
            _reject_unknown_keys(event, _EVENT_KEYS, event_label)
            event_id = _require_string(event.get("event_id"), f"{event_label}.event_id")
            if event_id in seen_events:
                raise TraceError(f"trace contains duplicate public event: {event_id}")
            if event_id in public_object_ids:
                raise TraceError(f"{event_label}.event_id reuses public object id: {event_id}")
            seen_events.add(event_id)
            if _require_integer(event.get("tick"), f"{event_label}.tick") != tick:
                raise TraceError(f"{event_label}.tick must match its snapshot frame")
            event_type = _require_string(event.get("event_type"), f"{event_label}.event_type")
            if event_type in _PRIVATE_EVENT_TYPES:
                raise TraceError(f"{event_label} contains a private event type")
            if event.get("visibility") not in {"organization", "public"}:
                raise TraceError(f"{event_label} is not a public event")
            _require_known_reference(event.get("actor_id"), f"{event_label}.actor_id", agent_ids)
            _require_known_references(
                event.get("object_ids"),
                f"{event_label}.object_ids",
                public_object_ids,
            )
            event_payload = _validate_typed_record(
                event.get("payload"),
                label=f"{event_label}.payload",
                field_kinds=_EVENT_PAYLOAD_FIELD_KINDS,
            )
            if "related_object_ids" in event_payload:
                _require_known_references(
                    event_payload["related_object_ids"],
                    f"{event_label}.payload.related_object_ids",
                    public_object_ids,
                )
            public_object_ids.add(event_id)

        for decision_index, decision_item in enumerate(
            _require_list(frame["decisions"], f"{label}.decisions")
        ):
            decision_label = f"{label}.decisions[{decision_index}]"
            decision = _require_mapping(decision_item, decision_label)
            _reject_unknown_keys(decision, _DECISION_KEYS, decision_label)
            required_decision_fields = {
                "decision_id",
                "tick",
                "agent_id",
                "chosen_action_id",
                "chosen_object_id",
            }
            if not required_decision_fields <= set(decision):
                raise TraceError(f"{decision_label} is missing required public decision fields")
            decision_id = _require_string(
                decision.get("decision_id"), f"{decision_label}.decision_id"
            )
            if decision_id in seen_decisions:
                raise TraceError(f"trace contains duplicate public decision: {decision_id}")
            if decision_id in public_object_ids:
                raise TraceError(
                    f"{decision_label}.decision_id reuses public object id: {decision_id}"
                )
            seen_decisions.add(decision_id)
            if _require_integer(decision.get("tick"), f"{decision_label}.tick") != tick:
                raise TraceError(f"{decision_label}.tick must match its snapshot frame")
            _require_known_reference(
                decision.get("agent_id"),
                f"{decision_label}.agent_id",
                agent_ids,
            )
            if decision["chosen_action_id"] is not None:
                _require_string(
                    decision["chosen_action_id"],
                    f"{decision_label}.chosen_action_id",
                )
            if decision.get("chosen_object_id") is not None:
                _require_known_reference(
                    decision["chosen_object_id"],
                    f"{decision_label}.chosen_object_id",
                    public_object_ids,
                )
            public_object_ids.add(decision_id)

        for event_index, event_item in enumerate(
            _require_list(frame["governance_events"], f"{label}.governance_events")
        ):
            event_label = f"{label}.governance_events[{event_index}]"
            event = _require_mapping(event_item, event_label)
            _reject_unknown_keys(event, _GOVERNANCE_EVENT_KEYS, event_label)
            event_id = _require_string(event.get("event_id"), f"{event_label}.event_id")
            if event_id in seen_governance_events:
                raise TraceError(f"trace contains duplicate governance event: {event_id}")
            if event_id in public_object_ids:
                raise TraceError(f"{event_label}.event_id reuses public object id: {event_id}")
            seen_governance_events.add(event_id)
            if _require_integer(event.get("tick"), f"{event_label}.tick") > tick:
                raise TraceError(f"{event_label}.tick cannot be later than its snapshot frame")
            _require_string(event.get("event_type"), f"{event_label}.event_type")
            _require_known_reference(
                event.get("protocol_id"),
                f"{event_label}.protocol_id",
                protocol_ids,
            )
            if event.get("actor_id") is not None:
                _require_known_reference(
                    event["actor_id"],
                    f"{event_label}.actor_id",
                    agent_ids,
                )
            data = _validate_typed_record(
                event.get("data"),
                label=f"{event_label}.data",
                field_kinds=_GOVERNANCE_DATA_FIELD_KINDS,
            )
            if "related_object_ids" in data:
                _require_known_references(
                    data["related_object_ids"],
                    f"{event_label}.data.related_object_ids",
                    public_object_ids,
                )
            public_object_ids.add(event_id)


def validate_trace(payload: Any, *, verify_digest: bool = True) -> dict[str, Any]:
    """Validate and return one public trace payload without mutating it."""

    trace = _require_mapping(payload, "trace")
    _validate_finite_numbers(trace)
    _validate_public_values(trace)
    _validate_structure(trace)
    if verify_digest:
        expected = trace.get("trace_sha256")
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            raise TraceError("trace.trace_sha256 must be a lowercase SHA-256 digest")
        unhashed = dict(trace)
        unhashed.pop("trace_sha256", None)
        if expected != canonical_sha256(unhashed):
            raise TraceError("trace digest mismatch")
    return trace


def build_trace(
    *,
    run_id: str,
    organization_id: str,
    config_digest: str,
    frames: list[dict[str, Any]],
    run_metadata: Mapping[str, Any] | None = None,
    evaluation_annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build, validate, and digest a public trace from curated public frames."""

    trace = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_id": run_id,
        "organization_id": organization_id,
        "config_sha256": config_digest,
        "privacy": {
            "private_reflections_included": False,
            "private_memories_included": False,
            "provider_messages_included": False,
        },
        "frames": frames,
    }
    if run_metadata is not None:
        trace["run_metadata"] = dict(run_metadata)
    if evaluation_annotations is not None:
        trace["evaluation_annotations"] = list(evaluation_annotations)
    validate_trace(trace, verify_digest=False)
    trace["trace_sha256"] = canonical_sha256(trace)
    return validate_trace(trace)


def load_trace(path: str | Path) -> dict[str, Any]:
    """Load a digest-bound public trace and reject malformed or private content."""

    trace_path = Path(path).expanduser().resolve()
    try:
        payload = _load_json_without_duplicate_keys(trace_path.read_text(encoding="utf-8"))
    except TraceError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceError(f"cannot load trace {trace_path}: {exc}") from exc
    return validate_trace(payload)


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "TraceError",
    "build_trace",
    "load_trace",
    "validate_trace",
]
