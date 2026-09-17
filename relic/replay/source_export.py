"""Fail-closed projection of source-runner state into ``relic-trace-v1``.

The source runner deliberately keeps rich replay snapshots for private
development.  This module must never consume those snapshots.  Instead it
reads a small, named set of public runtime fields while a run is live and
writes the public contract incrementally.  The result is suitable for the
release Inspector, while private messages, memories, model traffic, workspace
contents, reflection prose, and evaluator internals remain outside the trace.
"""

from __future__ import annotations

import enum
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from relic.replay.hashing import canonical_sha256
from relic.replay.trace import TraceError, build_trace, load_trace


PUBLIC_TRACE_FILENAME = "relic-trace-v1.json"
SOURCE_EXPORTER_VERSION = "relic-source-public-exporter-v1"
SOURCE_EXPORTER_PROFILE = "source-runner-explicit-public-state-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_EVENT_TYPES = frozenset(
    {
        "commitment_event",
        "external_signal_event",
        "file_share_event",
        "governance_event",
        "meeting_event",
        "product_event",
        "proposal_event",
        "protocol_event",
        "recovery_event",
        "repo_event",
        "speech_act_event",
        "task_progress_event",
        "tool_event",
    }
)
_SOURCE_CONTEXT_KEYS = frozenset(
    {
        "benchmark",
        "case_id",
        "condition",
        "config_digest",
        "model",
        "organization_id",
        "run_id",
        "seed",
        "source_commit",
    }
)
_PUBLIC_METRICS = frozenset(
    {
        "causal_fix_rate",
        "llm_tokens_per_merged_pr",
        "merged_pr_count",
        "oss_hidden_pass_rate",
        "release_count",
        "task_completion_rate",
    }
)


class SourceTraceExportError(ValueError):
    """A stable, non-secret public trace export failure."""

    def __init__(self, code: str):
        self.code = str(code)
        super().__init__(self.code)


@dataclass(frozen=True)
class SourceTraceExportResult:
    """Public receipt for one append-only projection update."""

    path: Path
    trace_sha256: str
    frames: int
    last_tick: int

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": "relic-public-trace-export-result-v1",
            "path": str(self.path),
            "trace_sha256": self.trace_sha256,
            "frames": self.frames,
            "last_tick": self.last_tick,
        }


def public_trace_path(run_directory: str | Path) -> Path:
    """Return the canonical public trace location for one run directory."""

    return Path(run_directory).expanduser() / "public" / PUBLIC_TRACE_FILENAME


def resolve_public_trace_path(
    *,
    trace_path: str | Path | None = None,
    run_directory: str | Path | None = None,
    cell_directory: str | Path | None = None,
) -> Path:
    """Resolve exactly one valid exported trace without inspecting private state."""

    supplied = [value is not None for value in (trace_path, run_directory, cell_directory)]
    if sum(supplied) != 1:
        raise SourceTraceExportError("public_trace_source_must_be_exactly_one")
    if trace_path is not None:
        candidate = Path(trace_path).expanduser()
    else:
        root = Path(run_directory or cell_directory or "").expanduser()
        canonical = public_trace_path(root)
        legacy = root / "public" / "trace.json"
        candidate = canonical if canonical.is_file() else legacy
    if candidate.is_symlink() or not candidate.is_file():
        raise SourceTraceExportError("public_trace_missing")
    resolved = candidate.resolve()
    try:
        load_trace(resolved)
    except (OSError, TraceError) as exc:
        raise SourceTraceExportError("public_trace_invalid") from exc
    return resolved


def copy_public_trace_prefix(source: str | Path, destination: str | Path) -> Path:
    """Copy one verified trace as the append-only prefix of a resumed run."""

    source_path = Path(source).expanduser()
    destination_path = Path(destination).expanduser()
    if source_path.is_symlink() or not source_path.is_file():
        raise SourceTraceExportError("public_trace_resume_source_missing")
    try:
        trace = load_trace(source_path)
    except (OSError, TraceError) as exc:
        raise SourceTraceExportError("public_trace_resume_source_invalid") from exc
    if destination_path.exists():
        if destination_path.is_symlink():
            raise SourceTraceExportError("public_trace_destination_symlink_forbidden")
        try:
            existing = load_trace(destination_path)
        except (OSError, TraceError) as exc:
            raise SourceTraceExportError("public_trace_destination_invalid") from exc
        if existing != trace:
            raise SourceTraceExportError("public_trace_destination_prefix_mismatch")
        return destination_path.resolve()
    _atomic_write_json(destination_path, trace)
    return destination_path.resolve()


def export_source_world_trace(
    world: Any,
    *,
    destination: str | Path,
    context: Mapping[str, Any] | None = None,
    run_record: Mapping[str, Any] | None = None,
) -> SourceTraceExportResult:
    """Append one live source-world update using only explicit public fields.

    ``world`` is intentionally duck-typed.  This keeps the public adapter
    independent of the legacy inspector and makes its field boundary auditable:
    no runtime snapshot, replay buffer, or generic object serialization is
    accepted as input.
    """

    output = Path(destination).expanduser()
    if output.is_symlink():
        raise SourceTraceExportError("public_trace_destination_symlink_forbidden")
    resolved_context = _resolve_context(world, context)
    state = _project_state(world, resolved_context["organization_id"])
    previous = _load_previous_trace(output, resolved_context)
    previous_frames = list(previous.get("frames", ())) if previous else []
    last_tick = int(previous_frames[-1]["tick"]) if previous_frames else -1
    if state["tick"] < last_tick:
        raise SourceTraceExportError("public_trace_tick_regression")
    if not previous_frames and state["tick"] != 0:
        raise SourceTraceExportError("public_trace_history_missing")
    if previous_frames and state["tick"] > last_tick + 1:
        raise SourceTraceExportError("public_trace_tick_gap")

    seen_event_ids, seen_decision_ids, seen_governance_ids = _seen_ids(previous_frames)
    raw_events = _project_events(world, state["object_ids"])
    decisions = _project_decisions(world, state["object_ids"], state["agent_ids"])
    governance = _project_governance_events(world, state["protocol_ids"], state["agent_ids"])
    new_events = _new_items(raw_events, seen_event_ids, state["tick"], "event")
    new_decisions = _new_items(decisions, seen_decision_ids, state["tick"], "decision")
    new_governance = _new_items(
        governance,
        seen_governance_ids,
        state["tick"],
        "governance_event",
    )
    evidence = _project_run_record_evidence(run_record, state["object_ids"])
    prior_evidence_ids = _prior_evidence_ids(previous_frames)
    new_evidence = [item for item in evidence if item["evidence_id"] not in prior_evidence_ids]

    additions: list[dict[str, Any]] = []
    if not previous_frames or state["tick"] > last_tick or new_events or new_decisions or new_governance:
        event_frames = [[item] for item in new_events]
        if not event_frames:
            event_frames = [[]]
        for index, events in enumerate(event_frames):
            additions.append(
                _frame(
                    state,
                    events=events,
                    decisions=new_decisions if index == 0 else [],
                    governance_events=new_governance if index == 0 else [],
                )
            )
    if new_evidence:
        additions.append(
            _frame(
                state,
                events=[],
                decisions=[],
                governance_events=[],
                evaluation_annotations=new_evidence,
            )
        )

    frames = previous_frames + additions
    _number_frames(frames)
    trace = build_trace(
        run_id=resolved_context["run_id"],
        organization_id=resolved_context["organization_id"],
        config_digest=resolved_context["config_digest"],
        frames=frames,
        run_metadata=resolved_context["run_metadata"],
    )
    _atomic_write_json(output, trace)
    return SourceTraceExportResult(
        path=output.resolve(),
        trace_sha256=trace["trace_sha256"],
        frames=len(frames),
        last_tick=state["tick"],
    )


def append_run_record_evidence(
    trace_path: str | Path,
    run_record: Mapping[str, Any],
) -> SourceTraceExportResult:
    """Append a batch-level source-record verdict without rewriting history."""

    path = Path(trace_path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise SourceTraceExportError("public_trace_missing")
    try:
        trace = load_trace(path)
    except (OSError, TraceError) as exc:
        raise SourceTraceExportError("public_trace_invalid") from exc
    frames = list(trace["frames"])
    final = frames[-1]
    organization = final["organization"]
    object_ids = _frame_object_ids(final)
    evidence = _project_run_record_evidence(run_record, object_ids)
    existing = _prior_evidence_ids(frames)
    additions = [item for item in evidence if item["evidence_id"] not in existing]
    if additions:
        state = {
            "tick": int(final["tick"]),
            "organization": organization,
            "episodes": final["episodes"],
            "artifacts": final.get("artifacts"),
            "repo_state": final.get("repo_state"),
        }
        frames.append(
            _frame(
                state,
                events=[],
                decisions=[],
                governance_events=[],
                evaluation_annotations=additions,
            )
        )
        _number_frames(frames)
        trace = build_trace(
            run_id=trace["run_id"],
            organization_id=trace["organization_id"],
            config_digest=trace["config_sha256"],
            frames=frames,
            run_metadata=trace.get("run_metadata"),
        )
        _atomic_write_json(path, trace)
    return SourceTraceExportResult(
        path=path.resolve(),
        trace_sha256=trace["trace_sha256"],
        frames=len(trace["frames"]),
        last_tick=int(trace["frames"][-1]["tick"]),
    )


def _resolve_context(world: Any, supplied: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(supplied or {})
    if set(raw) - _SOURCE_CONTEXT_KEYS:
        raise SourceTraceExportError("public_trace_context_unknown_field")
    world_run_id = _required_text(getattr(world, "run_id", None), "public_trace_run_id_missing")
    run_id = _optional_text(raw.get("run_id")) or world_run_id
    organization_id = _optional_text(raw.get("organization_id")) or run_id
    condition = _optional_text(raw.get("condition")) or _optional_text(
        getattr(world, "experiment_condition", None)
    )
    benchmark = _optional_text(raw.get("benchmark"))
    model = _optional_text(raw.get("model"))
    case_id = _optional_text(raw.get("case_id"))
    seed = raw.get("seed")
    if seed is None:
        scenario = getattr(world, "scenario", None)
        seed = getattr(scenario, "seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SourceTraceExportError("public_trace_seed_missing")
    source_commit = _optional_text(raw.get("source_commit"))
    identity = {
        "schema_version": SOURCE_EXPORTER_PROFILE,
        "run_id": run_id,
        "organization_id": organization_id,
        "condition": condition,
        "benchmark": benchmark,
        "model": model,
        "seed": seed,
        "case_id": case_id,
        "source_commit": source_commit,
    }
    config_digest = _optional_text(raw.get("config_digest")) or canonical_sha256(identity)
    if not _SHA256.fullmatch(config_digest):
        raise SourceTraceExportError("public_trace_config_digest_invalid")
    metadata: dict[str, Any] = {
        "exporter_version": SOURCE_EXPORTER_VERSION,
        "exporter_profile": SOURCE_EXPORTER_PROFILE,
        "redaction_reviewed": True,
    }
    for key, value in (
        ("case_id", case_id),
        ("condition", condition),
        ("benchmark", benchmark),
        ("model", model),
        ("seed", seed),
        ("source_commit", source_commit),
    ):
        if value is not None:
            metadata[key] = value
    return {
        "run_id": run_id,
        "organization_id": organization_id,
        "config_digest": config_digest,
        "run_metadata": metadata,
    }


def _load_previous_trace(path: Path, context: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink():
        raise SourceTraceExportError("public_trace_destination_symlink_forbidden")
    try:
        trace = load_trace(path)
    except (OSError, TraceError) as exc:
        raise SourceTraceExportError("public_trace_existing_invalid") from exc
    immutable = {
        "run_id": context["run_id"],
        "organization_id": context["organization_id"],
        "config_sha256": context["config_digest"],
        "run_metadata": context["run_metadata"],
    }
    if any(trace.get(key) != value for key, value in immutable.items()):
        raise SourceTraceExportError("public_trace_identity_mismatch")
    return trace


def _project_state(world: Any, organization_id: str) -> dict[str, Any]:
    tick = _required_tick(getattr(world, "world_tick", None), "public_trace_tick_missing")
    agents = _project_agents(world)
    tasks = _project_tasks(world, set(agents))
    _attach_agent_task_state(world, agents, set(tasks))
    proposals = _project_proposals(world, set(agents))
    protocols = _project_protocols(world, set(agents), set(proposals))
    episodes = _project_episodes(world, set(agents), set(tasks), set(proposals), set(protocols))
    artifacts = _project_artifacts(world, set(agents), set(tasks), set(protocols))
    repo_state = _project_repo_state(world, set(agents))
    organization = {
        "organization_id": organization_id,
        "tick": tick,
        "agents": [agents[key] for key in sorted(agents)],
        "tasks": [tasks[key] for key in sorted(tasks)],
        "proposals": [proposals[key] for key in sorted(proposals)],
        "protocols": [protocols[key] for key in sorted(protocols)],
    }
    if artifacts:
        organization["artifacts"] = [artifacts[key] for key in sorted(artifacts)]
    if repo_state is not None:
        organization["repo_state"] = repo_state
    object_ids = {organization_id, *agents, *tasks, *proposals, *protocols, *episodes, *artifacts}
    if repo_state is not None:
        object_ids.update(_repo_object_ids(repo_state))
    return {
        "tick": tick,
        "organization": organization,
        "episodes": [episodes[key] for key in sorted(episodes)],
        "artifacts": None,
        "repo_state": None,
        "agent_ids": set(agents),
        "protocol_ids": set(protocols),
        "object_ids": object_ids,
    }


def _project_agents(world: Any) -> dict[str, dict[str, Any]]:
    raw = _required_mapping(getattr(world, "agents", None), "public_trace_agents_missing")
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(raw, key=str):
        agent_id = _required_text(key, "public_trace_agent_id_invalid")
        agent = raw[key]
        observed_id = _optional_text(getattr(agent, "id", None))
        if observed_id is not None and observed_id != agent_id:
            raise SourceTraceExportError("public_trace_agent_id_mismatch")
        item: dict[str, Any] = {"agent_id": agent_id}
        for field, source_field in (("role", "role"), ("status", "current_status")):
            value = _optional_text(getattr(agent, source_field, None))
            if value is not None:
                item[field] = value
        result[agent_id] = item
    return result


def _project_tasks(world: Any, agent_ids: set[str]) -> dict[str, dict[str, Any]]:
    raw = _required_mapping(getattr(world, "tasks", None), "public_trace_tasks_missing")
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(raw, key=str):
        task_id = _required_text(key, "public_trace_task_id_invalid")
        task = raw[key]
        observed_id = _optional_text(getattr(task, "task_id", None))
        if observed_id is not None and observed_id != task_id:
            raise SourceTraceExportError("public_trace_task_id_mismatch")
        item: dict[str, Any] = {"task_id": task_id}
        title = _optional_text(getattr(task, "title", None))
        if title is not None:
            item["title"] = title
        status = _optional_enum_text(getattr(task, "status", None))
        if status is not None:
            item["status"] = status
        owner_id = _optional_text(getattr(task, "owner_id", None))
        if owner_id is not None:
            if owner_id not in agent_ids:
                raise SourceTraceExportError("public_trace_task_owner_unpublished")
            item["owner_id"] = owner_id
        priority = _public_scalar(getattr(task, "priority", None))
        if priority is not None:
            item["priority"] = priority
        visibility = _optional_text(getattr(task, "visibility", None))
        if visibility is not None:
            item["visibility"] = visibility
        progress = _public_number(getattr(task, "progress_score", None))
        if progress is not None:
            item["progress_score"] = progress
        result[task_id] = item
    task_ids = set(result)
    for key in sorted(raw, key=str):
        task_id = _required_text(key, "public_trace_task_id_invalid")
        dependencies = _optional_text_sequence(getattr(raw[key], "dependencies", None))
        if dependencies:
            if not set(dependencies) <= task_ids:
                raise SourceTraceExportError("public_trace_task_dependency_unpublished")
            result[task_id]["dependencies"] = dependencies
    return result


def _attach_agent_task_state(world: Any, agents: dict[str, dict[str, Any]], task_ids: set[str]) -> None:
    raw = _required_mapping(getattr(world, "agents", None), "public_trace_agents_missing")
    for agent_id, item in agents.items():
        agent = raw[agent_id]
        active = _optional_text_sequence(getattr(agent, "active_tasks", None))
        if active:
            if not set(active) <= task_ids:
                raise SourceTraceExportError("public_trace_agent_task_unpublished")
            item["active_task_ids"] = active
        current = _optional_text(getattr(agent, "current_task_id", None))
        if current is not None:
            if current not in task_ids:
                raise SourceTraceExportError("public_trace_agent_current_task_unpublished")
            item["current_task_id"] = current


def _project_proposals(world: Any, agent_ids: set[str]) -> dict[str, dict[str, Any]]:
    manager = getattr(world, "proposal_manager", None)
    raw = getattr(manager, "proposals", {}) if manager is not None else {}
    if not isinstance(raw, Mapping):
        raise SourceTraceExportError("public_trace_proposals_invalid")
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(raw, key=str):
        proposal_id = _required_text(key, "public_trace_proposal_id_invalid")
        proposal = raw[key]
        if _optional_text(getattr(proposal, "proposal_id", None)) != proposal_id:
            raise SourceTraceExportError("public_trace_proposal_id_mismatch")
        item: dict[str, Any] = {"proposal_id": proposal_id}
        for field in ("proposal_type", "status"):
            value = _optional_enum_text(getattr(proposal, field, None))
            if value is not None:
                item[field] = value
        proposer = _optional_text(getattr(proposal, "proposer_agent_id", None))
        if proposer is not None:
            if proposer not in agent_ids:
                raise SourceTraceExportError("public_trace_proposal_proposer_unpublished")
            item["proposer_agent_id"] = proposer
        created = _optional_tick(getattr(proposal, "created_at_tick", None))
        if created is not None:
            item["created_at_tick"] = created
        updated = _optional_tick(getattr(proposal, "updated_at_tick", None))
        if updated is not None:
            item["updated_at_tick"] = updated
        result[proposal_id] = item
    return result


def _project_protocols(
    world: Any,
    agent_ids: set[str],
    proposal_ids: set[str],
) -> dict[str, dict[str, Any]]:
    registry = getattr(world, "protocol_registry", None)
    raw = getattr(registry, "protocols", {}) if registry is not None else {}
    if not isinstance(raw, Mapping):
        raise SourceTraceExportError("public_trace_protocols_invalid")
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(raw, key=str):
        protocol_id = _required_text(key, "public_trace_protocol_id_invalid")
        protocol = raw[key]
        if _optional_text(getattr(protocol, "protocol_id", None)) != protocol_id:
            raise SourceTraceExportError("public_trace_protocol_id_mismatch")
        item: dict[str, Any] = {"protocol_id": protocol_id}
        for field in ("protocol_type", "scope", "adoption_status", "status"):
            value = _optional_enum_text(getattr(protocol, field, None))
            if value is not None:
                item[field] = value
        proposer = _optional_text(getattr(protocol, "proposer_id", None))
        if proposer is not None:
            if proposer not in agent_ids:
                raise SourceTraceExportError("public_trace_protocol_proposer_unpublished")
            item["proposer_id"] = proposer
        source_proposal = _optional_text(getattr(protocol, "created_from_proposal_id", None))
        if source_proposal is not None:
            if source_proposal not in proposal_ids:
                raise SourceTraceExportError("public_trace_protocol_proposal_unpublished")
            item["created_from_proposal_id"] = source_proposal
        for field in ("supporters", "opposers"):
            values = _optional_text_sequence(getattr(protocol, field, None))
            if values:
                if not set(values) <= agent_ids:
                    raise SourceTraceExportError("public_trace_protocol_agent_unpublished")
                item[field] = values
        for field in ("first_tick", "last_active_tick", "retired_tick", "persistence_ticks"):
            value = _optional_tick(getattr(protocol, field, None))
            if value is not None:
                item[field] = value
        result[protocol_id] = item
    return result


def _project_episodes(
    world: Any,
    agent_ids: set[str],
    task_ids: set[str],
    proposal_ids: set[str],
    protocol_ids: set[str],
) -> dict[str, dict[str, Any]]:
    manager = getattr(world, "episode_manager", None)
    raw = getattr(manager, "episodes", {}) if manager is not None else {}
    if not isinstance(raw, Mapping):
        raise SourceTraceExportError("public_trace_episodes_invalid")
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(raw, key=str):
        episode_id = _required_text(key, "public_trace_episode_id_invalid")
        episode = raw[key]
        if _optional_text(getattr(episode, "episode_id", None)) != episode_id:
            raise SourceTraceExportError("public_trace_episode_id_mismatch")
        item: dict[str, Any] = {"episode_id": episode_id}
        for field in ("episode_type", "status"):
            value = _optional_enum_text(getattr(episode, field, None))
            if value is not None:
                item[field] = value
        for target, source, known in (
            ("participants", "participants", agent_ids),
            ("linked_task_ids", "linked_task_ids", task_ids),
            ("linked_proposal_ids", "linked_proposal_ids", proposal_ids),
            ("linked_protocol_ids", "linked_protocol_ids", protocol_ids),
        ):
            values = _published_text_sequence(getattr(episode, source, None), known)
            if values:
                item[target] = values
        for target, source in (("start_tick", "start_tick"), ("end_tick", "end_tick")):
            value = _optional_tick(getattr(episode, source, None))
            if value is not None:
                item[target] = value
        result[episode_id] = item
    return result


def _project_artifacts(
    world: Any,
    agent_ids: set[str],
    task_ids: set[str],
    protocol_ids: set[str],
) -> dict[str, dict[str, Any]]:
    raw = getattr(world, "product_artifacts", {})
    if not isinstance(raw, Mapping):
        raise SourceTraceExportError("public_trace_artifacts_invalid")
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(raw, key=str):
        artifact_id = _required_text(key, "public_trace_artifact_id_invalid")
        artifact = raw[key]
        if _optional_text(getattr(artifact, "artifact_id", None)) != artifact_id:
            raise SourceTraceExportError("public_trace_artifact_id_mismatch")
        item: dict[str, Any] = {"artifact_id": artifact_id}
        for field in ("artifact_type", "status"):
            value = _optional_enum_text(getattr(artifact, field, None))
            if value is not None:
                item[field] = value
        version = _public_scalar(getattr(artifact, "revision", None))
        if version is not None:
            item["version"] = version
        owner = _optional_text(getattr(artifact, "owner_agent_id", None))
        if owner is not None:
            if owner not in agent_ids:
                raise SourceTraceExportError("public_trace_artifact_owner_unpublished")
            item["owner_id"] = owner
        for target, source, known in (
            ("related_task_ids", "linked_task_ids", task_ids),
            ("related_protocol_ids", "linked_protocol_ids", protocol_ids),
        ):
            values = _published_text_sequence(getattr(artifact, source, None), known)
            if values:
                item[target] = values
        for target, source in (("created_tick", "created_at_tick"), ("updated_tick", "updated_at_tick")):
            value = _optional_tick(getattr(artifact, source, None))
            if value is not None:
                item[target] = value
        result[artifact_id] = item
    return result


def _project_repo_state(world: Any, agent_ids: set[str]) -> dict[str, Any] | None:
    system = getattr(world, "repo_system", None)
    repo = getattr(system, "repo", None) if system is not None else None
    repository_id = _optional_text(getattr(repo, "repo_id", None))
    if repo is None or repository_id is None:
        return None
    result: dict[str, Any] = {"repository_id": repository_id}
    for target, source in (("name", "name"), ("branch", "main_branch")):
        value = _optional_text(getattr(repo, source, None))
        if value is not None:
            result[target] = value
    collections = (
        ("branches", "branches", "branch_id", "owner_id", "last_sync_tick"),
        ("commits", "commits", "commit_id", "author_id", "timestamp"),
        ("pull_requests", "pull_requests", "pr_id", "author_id", "opened_tick"),
        ("ci_runs", "ci_runs", "ci_id", None, "created_at_tick"),
    )
    for target, source, identifier, actor_field, tick_field in collections:
        values = getattr(repo, source, {})
        if not isinstance(values, Mapping):
            raise SourceTraceExportError("public_trace_repo_collection_invalid")
        records: list[dict[str, Any]] = []
        for key in sorted(values, key=str):
            object_id = _required_text(key, "public_trace_repo_id_invalid")
            item_source = values[key]
            if _optional_text(getattr(item_source, identifier, None)) != object_id:
                raise SourceTraceExportError("public_trace_repo_id_mismatch")
            item: dict[str, Any] = {identifier: object_id}
            status = _optional_enum_text(getattr(item_source, "status", None))
            if status is not None:
                item["status"] = status
            if actor_field is not None:
                actor = _optional_text(getattr(item_source, actor_field, None))
                if actor is not None:
                    if actor not in agent_ids:
                        raise SourceTraceExportError("public_trace_repo_actor_unpublished")
                    item["actor_id"] = actor
            tick = _optional_tick(getattr(item_source, tick_field, None))
            if tick is not None:
                item["tick"] = tick
            records.append(item)
        if records:
            result[target] = records
    reviews = getattr(repo, "review_comments", ())
    if isinstance(reviews, Sequence) and not isinstance(reviews, (str, bytes)):
        public_reviews: list[dict[str, Any]] = []
        for review in reviews:
            if not isinstance(review, Mapping):
                raise SourceTraceExportError("public_trace_review_invalid")
            review_id = _optional_text(review.get("review_id"))
            reviewer = _optional_text(review.get("reviewer"))
            if review_id is None or reviewer is None or reviewer not in agent_ids:
                raise SourceTraceExportError("public_trace_review_identity_invalid")
            item: dict[str, Any] = {"review_id": review_id, "actor_id": reviewer}
            approved = review.get("approve")
            if isinstance(approved, bool):
                item["status"] = "approved" if approved else "changes_requested"
            tick = _optional_tick(review.get("tick"))
            if tick is not None:
                item["tick"] = tick
            public_reviews.append(item)
        if public_reviews:
            result["reviews"] = public_reviews
    return result


def _project_events(world: Any, object_ids: set[str]) -> list[dict[str, Any]]:
    raw = getattr(world, "events", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SourceTraceExportError("public_trace_events_invalid")
    agent_ids = _agent_ids_from_object_ids(world, object_ids)
    result: list[dict[str, Any]] = []
    for index, event in enumerate(raw):
        if not isinstance(event, Mapping):
            raise SourceTraceExportError("public_trace_event_invalid")
        event_type = _optional_text(event.get("type"))
        actor = _optional_text(event.get("agent_id"))
        if event_type not in _SOURCE_EVENT_TYPES or actor not in agent_ids:
            continue
        tick = _optional_tick(event.get("tick"))
        if tick is None:
            raise SourceTraceExportError("public_trace_event_tick_missing")
        references = _source_event_references(event, object_ids)
        result.append(
            {
                "event_id": f"source_event_{index:08d}",
                "tick": tick,
                "event_type": event_type,
                "actor_id": actor,
                "object_ids": references,
                "payload": {},
                "visibility": "organization",
            }
        )
    return result


def _project_decisions(
    world: Any,
    object_ids: set[str],
    agent_ids: set[str],
) -> list[dict[str, Any]]:
    raw = getattr(world, "action_log", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SourceTraceExportError("public_trace_action_log_invalid")
    result: list[dict[str, Any]] = []
    for index, action in enumerate(raw):
        if not isinstance(action, Mapping):
            raise SourceTraceExportError("public_trace_action_invalid")
        actor = _optional_text(action.get("agent_id"))
        action_type = _optional_text(action.get("action_type"))
        tick = _optional_tick(action.get("tick"))
        if actor not in agent_ids or action_type is None or tick is None:
            continue
        target = _optional_text(action.get("target"))
        result.append(
            {
                "decision_id": f"source_action_{index:08d}",
                "tick": tick,
                "agent_id": actor,
                "chosen_action_id": action_type,
                "chosen_object_id": target if target in object_ids else None,
            }
        )
    return result


def _project_governance_events(
    world: Any,
    protocol_ids: set[str],
    agent_ids: set[str],
) -> list[dict[str, Any]]:
    registry = getattr(world, "protocol_registry", None)
    raw = getattr(registry, "events", ()) if registry is not None else ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SourceTraceExportError("public_trace_governance_events_invalid")
    result: list[dict[str, Any]] = []
    for event in raw:
        event_id = _optional_text(getattr(event, "event_id", None))
        event_type = _optional_text(getattr(event, "event_type", None))
        protocol_id = _optional_text(getattr(event, "protocol_id", None))
        tick = _optional_tick(getattr(event, "tick", None))
        actor = _optional_text(getattr(event, "actor_id", None))
        if (
            event_id is None
            or event_type is None
            or protocol_id not in protocol_ids
            or tick is None
        ):
            raise SourceTraceExportError("public_trace_governance_event_identity_invalid")
        if actor is not None and actor not in agent_ids:
            raise SourceTraceExportError("public_trace_governance_actor_unpublished")
        item: dict[str, Any] = {
            "event_id": f"source_governance_{event_id}",
            "event_type": event_type,
            "protocol_id": protocol_id,
            "tick": tick,
            "data": {},
        }
        if actor is not None:
            item["actor_id"] = actor
        result.append(item)
    return result


def _project_run_record_evidence(
    record: Mapping[str, Any] | None,
    object_ids: set[str],
) -> list[dict[str, Any]]:
    if record is None:
        return []
    if not isinstance(record, Mapping):
        raise SourceTraceExportError("public_trace_run_record_invalid")
    status = _optional_text(record.get("status"))
    if status is None:
        raise SourceTraceExportError("public_trace_run_record_status_missing")
    evidence: list[dict[str, Any]] = [
        {
            "evidence_id": f"source_run_status_{status}",
            "evidence_type": "run_record",
            "status": status,
            "source": "orgenv_experiment_run_record",
            "available": True,
        }
    ]
    final = record.get("final_evaluation")
    if isinstance(final, Mapping):
        final_status = _optional_text(final.get("status"))
        if final_status is not None:
            item: dict[str, Any] = {
                "evidence_id": f"source_final_evaluation_{final_status}",
                "evidence_type": "final_evaluation",
                "status": final_status,
                "source": "orgenv_experiment_run_record",
                "available": True,
            }
            ready = final.get("formal_claim_ready")
            if isinstance(ready, bool):
                item["passed"] = ready
            evidence.append(item)
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        for metric in sorted(_PUBLIC_METRICS):
            value = _public_number(metrics.get(metric))
            if value is None:
                continue
            evidence.append(
                {
                    "evidence_id": f"source_metric_{metric}",
                    "evidence_type": "metric",
                    "status": "recorded",
                    "source": "orgenv_experiment_run_record",
                    "metric_name": metric,
                    "metric_value": value,
                    "available": True,
                }
            )
    return evidence


def _source_event_references(event: Mapping[str, Any], object_ids: set[str]) -> list[str]:
    references: list[str] = []
    for field in (
        "artifact_id",
        "branch_id",
        "ci_id",
        "commit_id",
        "object_id",
        "pr_id",
        "proposal_id",
        "protocol_id",
        "task_id",
    ):
        value = _optional_text(event.get(field))
        if value is not None and value in object_ids and value not in references:
            references.append(value)
    return references


def _agent_ids_from_object_ids(world: Any, object_ids: set[str]) -> set[str]:
    raw = _required_mapping(getattr(world, "agents", None), "public_trace_agents_missing")
    return {str(key) for key in raw if str(key) in object_ids}


def _new_items(
    items: Sequence[Mapping[str, Any]],
    seen: set[str],
    current_tick: int,
    kind: str,
) -> list[dict[str, Any]]:
    identifier = {
        "event": "event_id",
        "decision": "decision_id",
        "governance_event": "event_id",
    }[kind]
    result: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item[identifier])
        if item_id in seen:
            continue
        tick = int(item["tick"])
        if tick != current_tick:
            raise SourceTraceExportError(f"public_trace_{kind}_not_streamed")
        result.append(dict(item))
    return result


def _seen_ids(frames: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    events: set[str] = set()
    decisions: set[str] = set()
    governance: set[str] = set()
    for frame in frames:
        events.update(str(item["event_id"]) for item in frame.get("events", ()))
        decisions.update(str(item["decision_id"]) for item in frame.get("decisions", ()))
        governance.update(str(item["event_id"]) for item in frame.get("governance_events", ()))
    return events, decisions, governance


def _prior_evidence_ids(frames: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(item["evidence_id"])
        for frame in frames
        for item in frame.get("evaluation_annotations", ())
    }


def _frame_object_ids(frame: Mapping[str, Any]) -> set[str]:
    organization = frame["organization"]
    result = {str(organization["organization_id"])}
    for collection, identifier in (
        ("agents", "agent_id"),
        ("tasks", "task_id"),
        ("proposals", "proposal_id"),
        ("protocols", "protocol_id"),
    ):
        result.update(str(item[identifier]) for item in organization.get(collection, ()))
    for item in frame.get("episodes", ()):
        result.add(str(item["episode_id"]))
    artifacts = frame.get("artifacts", organization.get("artifacts", ()))
    result.update(str(item["artifact_id"]) for item in artifacts or ())
    repo = frame.get("repo_state", organization.get("repo_state"))
    if isinstance(repo, Mapping):
        result.update(_repo_object_ids(repo))
    return result


def _repo_object_ids(repo: Mapping[str, Any]) -> set[str]:
    result = {str(repo["repository_id"])}
    for collection, identifier in (
        ("branches", "branch_id"),
        ("commits", "commit_id"),
        ("pull_requests", "pr_id"),
        ("reviews", "review_id"),
        ("ci_runs", "ci_id"),
        ("merges", "result_id"),
    ):
        result.update(str(item[identifier]) for item in repo.get(collection, ()))
    return result


def _frame(
    state: Mapping[str, Any],
    *,
    events: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    governance_events: list[dict[str, Any]],
    evaluation_annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "frame_id": "",
        "sequence": 0,
        "tick": int(state["tick"]),
        "organization": state["organization"],
        "events": events,
        "episodes": state["episodes"],
        "decisions": decisions,
        "governance_events": governance_events,
    }
    if state.get("artifacts") is not None:
        frame["artifacts"] = state["artifacts"]
    if state.get("repo_state") is not None:
        frame["repo_state"] = state["repo_state"]
    if evaluation_annotations:
        frame["evaluation_annotations"] = evaluation_annotations
    return frame


def _number_frames(frames: list[dict[str, Any]]) -> None:
    for sequence, frame in enumerate(frames):
        frame["sequence"] = sequence
        frame["frame_id"] = f"frame_{sequence:06d}"


def _required_mapping(value: Any, code: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise SourceTraceExportError(code)
    return value


def _required_text(value: Any, code: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise SourceTraceExportError(code)
    return text


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_enum_text(value: Any) -> str | None:
    if isinstance(value, enum.Enum):
        value = value.value
    return _optional_text(value)


def _required_tick(value: Any, code: str) -> int:
    tick = _optional_tick(value)
    if tick is None:
        raise SourceTraceExportError(code)
    return tick


def _optional_tick(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_text_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SourceTraceExportError("public_trace_public_sequence_invalid")
    result: list[str] = []
    for item in value:
        text = _optional_text(item)
        if text is None or text in result:
            raise SourceTraceExportError("public_trace_public_sequence_invalid")
        result.append(text)
    return result


def _published_text_sequence(value: Any, known: set[str]) -> list[str]:
    """Keep only references already present in the public projection.

    Source episodes and artifacts can link private messages, wishes, workspace
    documents, or evaluator-only objects alongside public records.  Omitting a
    link is safer than exposing the private target or claiming it is public.
    """

    return [item for item in _optional_text_sequence(value) if item in known]


def _public_scalar(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, str) and value.strip():
        return value
    return None


def _public_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    if parent.is_symlink():
        raise SourceTraceExportError("public_trace_parent_symlink_forbidden")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        details = path.stat()
        if not stat.S_ISREG(details.st_mode):
            raise SourceTraceExportError("public_trace_destination_not_regular")
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "PUBLIC_TRACE_FILENAME",
    "SOURCE_EXPORTER_PROFILE",
    "SOURCE_EXPORTER_VERSION",
    "SourceTraceExportError",
    "SourceTraceExportResult",
    "append_run_record_evidence",
    "copy_public_trace_prefix",
    "export_source_world_trace",
    "public_trace_path",
    "resolve_public_trace_path",
]
