"""Interactive OpenAI coding agent backed by bounded repository tools."""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any, Mapping, Sequence

from .agent_progress import (
    AgentProgressSignal,
    detect_agent_progress_stall,
    detect_post_failure_diagnostic_saturation,
    post_failure_diagnostic_count,
    suppressed_progress_tool_names,
)
from .coding_tools import (
    DEFAULT_MAX_EXPLORATION_CALLS_PER_REVISION,
    CodingToolSession,
    _normalized_check_command,
    coding_tool_definitions,
    tool_result_json,
)
from .code_landing.environment import build_workspace_execution_profile
from .code_landing.event_store import RunRecord, SQLiteEventStore
from .execution import CommandExecutor, probe_command_executor_executables
from .hashing import canonicalize, stable_hash
from .openai_runtime import (
    build_openai_client,
    openai_call_watchdog,
    openai_runtime_identity,
    prepare_openai_response_request,
)
from .workspace_agent import (
    CodingAgentPatchProposal,
    WorkspaceFileContext,
    _agent_payload,
)
from .workspace_update import (
    WorkspaceFilePatch,
    WorkspacePackageInfo,
    WorkspaceVerificationResult,
)

TOOL_LOOP_RUNTIME_VERSION = "91"
DEFAULT_TOOL_LOOP_CONVERSATION_MAX_CHARS = 80_000
DEFAULT_PORTABLE_REPLAY_MAX_CHARS = 48_000
MIN_PORTABLE_REPLAY_MAX_CHARS = 16_000
RESPONSE_TRANSPORT_ENV = "SOCIETY_CORE_WORKSPACE_CODING_RESPONSE_TRANSPORT"
REQUEST_TIMEOUT_ENV = "SOCIETY_CORE_WORKSPACE_CODING_REQUEST_TIMEOUT_SECONDS"
DEFAULT_TOOL_LOOP_REQUEST_TIMEOUT_SECONDS = 600.0
MAX_TOOL_LOOP_REQUEST_TIMEOUT_SECONDS = 900.0
DEFAULT_TOOL_LOOP_MAX_TURNS = 64
DEFAULT_TOOL_LOOP_MAX_TOOL_CALLS = 256
DEFAULT_TOOL_LOOP_MAX_PATCH_BYTES = 500_000
MAX_REVISION_OUTCOME_LEDGER_ENTRIES = 12
REPEATED_BLOCKED_MUTATION_LIMIT = 3
REPEATED_PHASE_TOOL_VIOLATION_LIMIT = 2
REPEATED_ACTION_OBSERVATION_HARD_LIMIT = 6
_MUTATION_TOOL_NAMES = frozenset(
    {"apply_patch", "replace_lines", "replace_line_ranges"}
)
_REVISION_MUTATION_TOOL_NAMES = _MUTATION_TOOL_NAMES | {"revert_file"}
_EXPLORATION_TOOL_NAMES = frozenset({"list_files", "read_file", "search"})
_VERIFICATION_TOOL_NAMES = frozenset({"run_check", "run_python_check"})
_DIFF_TOOL_NAMES = frozenset({"git_diff"})
_MUTATION_RECOVERY_TOOL_NAMES = frozenset({"read_file"})
_MUTATION_RECOVERY_EDIT_TOOL_NAMES = frozenset({"replace_lines", "replace_line_ranges"})
_NO_EFFECT_MUTATION_STATUSES = frozenset({"blocked", "no_change", "no_op"})


def resolve_response_transport(
    *,
    endpoint_kind: str,
    requested: str | None = None,
) -> str:
    selected = (
        requested
        if requested is not None
        else os.environ.get(RESPONSE_TRANSPORT_ENV, "auto")
    )
    normalized = str(selected).strip().lower()
    if normalized not in {"auto", "sync", "stream"}:
        raise ValueError("invalid_workspace_coding_response_transport")
    if normalized == "auto":
        return "stream" if endpoint_kind == "custom_compatible" else "sync"
    return normalized


def resolve_tool_request_timeout_seconds(
    *,
    timeout_seconds: float,
    requested: float | str | None = None,
) -> float:
    """Bound one model request independently from the end-to-end agent budget."""

    raw = requested
    if raw is None:
        raw = os.environ.get(REQUEST_TIMEOUT_ENV)
    if raw is None or str(raw).strip() == "":
        return min(
            max(float(timeout_seconds), 1.0),
            DEFAULT_TOOL_LOOP_REQUEST_TIMEOUT_SECONDS,
        )
    try:
        resolved = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_workspace_coding_request_timeout") from exc
    if resolved <= 0:
        raise ValueError("workspace_coding_request_timeout_must_be_positive")
    return min(resolved, MAX_TOOL_LOOP_REQUEST_TIMEOUT_SECONDS)


@dataclass(frozen=True)
class ToolLoopTrace:
    status: str
    response_ids: tuple[str, ...]
    request_count: int
    tool_calls: tuple[dict[str, Any], ...]
    total_tokens: int
    blocked_reason: str | None
    trace_hash: str


def _workflow_phase(
    tools: CodingToolSession,
    *,
    implementation_required: bool,
) -> str:
    issues = tools.finalization_issues()
    if tools.changed_paths and not issues:
        return "ready_to_finalize"
    if tools.mutation_recovery_observation_available:
        return "mutation_recovery"
    if getattr(tools, "mutation_recovery_relocation_requirement", None) is not None:
        return "mutation_recovery_relocate"
    if getattr(tools, "mutation_recovery_edit_lease", None) is not None:
        return "mutation_recovery_edit"
    unresolved_failure = any(
        issue.startswith("current_revision_check_failure_unresolved:")
        for issue in issues
    )
    if (
        unresolved_failure
        and int(getattr(tools, "unverified_repair_revision_count", 0)) > 0
    ):
        return "verification"
    if unresolved_failure:
        if implementation_required:
            return "implementation"
        return "repair_diagnosis"
    if any(issue.startswith("current_revision_") for issue in issues):
        return "verification"
    if issues == ("final_diff_not_inspected_after_latest_patch",):
        return "diff_inspection"
    if implementation_required:
        return "implementation"
    return "focused_exploration"


def _workflow_phase_message(
    phase: str,
    *,
    revision: int,
    diagnostic_count: int | None,
    mutation_recovery_requirement: Mapping[str, Any] | None = None,
    mutation_recovery_edit_lease: Mapping[str, Any] | None = None,
    failure_diagnostic: Mapping[str, Any] | None = None,
    unresolved_check_replays: Sequence[Mapping[str, Any]] = (),
) -> dict[str, str] | None:
    instructions = {
        "verification": (
            "The candidate revision changed. Run a focused current-revision behavior "
            "or required project check now. Do not spend this phase rereading source."
        ),
        "repair_diagnosis": (
            "A current-revision check failed. Use the bounded diagnostic window only "
            "to identify the immediate cause, then repair or revert the revision."
        ),
        "mutation_recovery": (
            "The last mutation did not change the revision. Re-read the exact target "
            "file once and use its fresh hash and content for the next mutation. Other "
            "observations cannot satisfy this recovery step."
        ),
        "mutation_recovery_edit": (
            "The recovery observation is now an exact lease. Apply replace_lines or "
            "replace_line_ranges only on the leased path, file hash, and observed line "
            "range. Copy expected_old exactly from the leased source without its numeric "
            "line prefix. The controller may resolve a mistaken requested line number "
            "only when expected_old has one unique exact location inside the lease. The "
            "leased source is already present in the conversation; further reads and "
            "apply_patch are unavailable."
        ),
        "mutation_recovery_relocate": (
            "The exact expected_old text was absent from the current lease, so that "
            "range cannot support the intended edit. Use read_file once on an explicit, "
            "non-overlapping range of the same file, guided by source locations already "
            "seen in the conversation. Search and mutation tools are unavailable until "
            "that replacement observation is bound."
        ),
        "implementation": (
            "The observation or post-failure diagnostic budget is exhausted. Apply a "
            "supported repair or revert the defective revision; no more probes are "
            "available in this phase."
        ),
        "diff_inspection": (
            "Required current-revision checks have passed. Inspect the final diff now, "
            "then return the strict final JSON."
        ),
        "ready_to_finalize": (
            "All typed finalization evidence is current. Return the strict final JSON "
            "without another repository tool call."
        ),
    }
    instruction = instructions.get(phase)
    if instruction is None:
        return None
    if (
        phase == "repair_diagnosis"
        and failure_diagnostic is not None
        and failure_diagnostic.get("raw_observation_required") is True
    ):
        instruction = (
            "The failed acceptance check exposed only aggregate assertion values. "
            "Use one run_python_check diagnostic to reproduce one failing boundary and "
            "emit its raw user-visible stdout or stderr and source attribution. Then "
            "repair or revert the revision; a print-only probe is diagnostic evidence, "
            "not final verification."
        )
    if phase == "verification" and unresolved_check_replays:
        instruction = (
            "The candidate changed after one or more failed checks. Re-run each "
            "exact unresolved command listed below before introducing a new probe. "
            "A passing replacement command does not clear the original failure."
        )
    payload: dict[str, Any] = {
        "phase": phase,
        "revision": revision,
        "required_action": instruction,
    }
    if diagnostic_count is not None:
        payload["post_failure_diagnostic_calls"] = diagnostic_count
    if phase == "mutation_recovery" and mutation_recovery_requirement is not None:
        payload["mutation_recovery"] = dict(mutation_recovery_requirement)
    if phase == "mutation_recovery_edit" and mutation_recovery_edit_lease is not None:
        payload["mutation_recovery_edit_lease"] = dict(mutation_recovery_edit_lease)
    if phase == "repair_diagnosis" and failure_diagnostic is not None:
        payload["latest_failure"] = dict(failure_diagnostic)
    if phase == "verification" and unresolved_check_replays:
        payload["unresolved_check_replays"] = tuple(
            dict(item) for item in unresolved_check_replays
        )
    return {
        "role": "user",
        "content": "WORKFLOW_PHASE\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }


def _latest_failed_check_diagnostic(
    transcript: tuple[Any, ...],
    *,
    current_revision: int,
) -> dict[str, Any] | None:
    for item in reversed(transcript):
        if int(_item_value(item, "revision", 0) or 0) != current_revision:
            continue
        name = str(_item_value(item, "name", ""))
        result = _item_value(item, "result", {})
        if (
            name not in _VERIFICATION_TOOL_NAMES
            or not isinstance(result, Mapping)
            or result.get("status") != "failed"
            or result.get("finalization_blocking") is False
        ):
            continue
        evidence = str(
            result.get("stderr_tail")
            or result.get("stdout_tail")
            or result.get("reason")
            or ""
        )[-800:]
        aggregate_assertion = re.search(
            r"AssertionError:\s*\[[^\n]{0,700}\]\s*$",
            evidence,
        )
        return {
            "tool": name,
            "check_role": result.get("check_role"),
            "exit_code": result.get("exit_code"),
            "evidence": evidence,
            "raw_observation_required": aggregate_assertion is not None,
        }
    return None


def _tool_definitions_for_workflow_phase(
    definitions: list[dict[str, Any]],
    *,
    phase: str,
    mutation_recovery_requirement: Mapping[str, Any] | None = None,
    mutation_recovery_edit_lease: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Enforce workflow transitions at the tool surface, not only in the prompt."""

    allowed_by_phase = {
        "verification": _VERIFICATION_TOOL_NAMES,
        "mutation_recovery": _MUTATION_RECOVERY_TOOL_NAMES,
        "mutation_recovery_relocate": _MUTATION_RECOVERY_TOOL_NAMES,
        "mutation_recovery_edit": _MUTATION_RECOVERY_EDIT_TOOL_NAMES,
        "implementation": _REVISION_MUTATION_TOOL_NAMES,
        "diff_inspection": _DIFF_TOOL_NAMES | _REVISION_MUTATION_TOOL_NAMES,
        "ready_to_finalize": _REVISION_MUTATION_TOOL_NAMES,
    }
    allowed = allowed_by_phase.get(phase)
    if allowed is None:
        return definitions
    scoped = [definition for definition in definitions if definition["name"] in allowed]
    if not scoped:
        # Liveness guard: a phase whose allowed tools were all withheld by
        # availability gates would leave the model with no way to change
        # session state, so the loop would spin finalization rejections
        # until the turn budget discards the candidate. Changing or
        # reverting the candidate is the universally sanctioned escape, so
        # fall back to the mutation surface instead of an empty one.
        return [
            definition
            for definition in definitions
            if definition["name"] in _REVISION_MUTATION_TOOL_NAMES
        ]
    if phase not in {
        "mutation_recovery",
        "mutation_recovery_relocate",
        "mutation_recovery_edit",
    }:
        return scoped

    recovery_state = (
        mutation_recovery_requirement
        if phase == "mutation_recovery"
        else mutation_recovery_edit_lease
    )
    if recovery_state is None:
        return scoped
    target_path = str(recovery_state.get("path") or "")
    if not target_path:
        return scoped
    constrained: list[dict[str, Any]] = []
    for definition in scoped:
        parameters = copy.deepcopy(definition["parameters"])
        properties = parameters["properties"]
        properties["path"] = {
            "type": "string",
            "enum": [target_path],
            "description": "The only file eligible for this mutation recovery.",
        }
        if phase == "mutation_recovery_relocate" and definition["name"] == "read_file":
            total_lines = int(recovery_state.get("total_lines") or 0)
            properties["start_line"] = {
                "type": "integer",
                "minimum": 1,
                **({"maximum": total_lines} if total_lines > 0 else {}),
                "description": (
                    "Explicit start of a range that does not overlap the current lease."
                ),
            }
            properties["end_line"] = {
                "type": "integer",
                "minimum": 1,
                **({"maximum": total_lines} if total_lines > 0 else {}),
                "description": (
                    "Explicit end of a range that does not overlap the current lease."
                ),
            }
        if phase == "mutation_recovery_edit" and definition["name"] in {
            "replace_lines",
            "replace_line_ranges",
        }:
            file_hash = str(recovery_state.get("file_hash") or "")
            observed_start = int(recovery_state.get("observed_start_line") or 0)
            observed_end = int(recovery_state.get("observed_end_line") or 0)
            properties["expected_file_hash"] = {
                "type": "string",
                "enum": [file_hash],
                "description": "The exact hash bound to the recovery observation.",
            }
            if definition["name"] == "replace_lines":
                properties["start_line"] = {
                    "type": "integer",
                    "minimum": observed_start,
                    "maximum": observed_end,
                }
                properties["end_line"] = {
                    "type": "integer",
                    "minimum": observed_start,
                    "maximum": observed_end,
                }
                properties["expected_old"] = {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Exact current text of the inclusive source lines, copied from "
                        "the lease without numeric line prefixes."
                    ),
                }
                parameters["required"].append("expected_old")
            else:
                item_properties = properties["replacements"]["items"]["properties"]
                item_properties["start_line"] = {
                    "type": "integer",
                    "minimum": observed_start,
                    "maximum": observed_end,
                }
                item_properties["end_line"] = {
                    "type": "integer",
                    "minimum": observed_start,
                    "maximum": observed_end,
                }
                item_properties["expected_old"] = {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Exact current text of these inclusive source lines, copied "
                        "from the lease without numeric line prefixes."
                    ),
                }
                properties["replacements"]["items"]["required"].append("expected_old")
        if phase == "mutation_recovery":
            description = (
                "Refresh the exact pending mutation target. The controller also "
                "redirects a noncompliant path and records that intervention."
            )
        elif phase == "mutation_recovery_relocate":
            description = (
                "Replace a misplaced recovery observation with one explicit, "
                "non-overlapping range of the same hash-bound file."
            )
        elif definition["name"] == "read_file":
            description = (
                "Refresh one explicit range of the leased file when the current "
                "recovery observation is not the intended edit location."
            )
        else:
            description = (
                "Apply a guarded line edit constrained to the exact recovery "
                "observation path, hash, range, and expected_old source text. A line "
                "number drift is resolved only when expected_old is unique in the lease."
            )
        constrained.append(
            {
                **definition,
                "description": description,
                "parameters": parameters,
            }
        )
    return constrained


class ModelStreamTerminalError(RuntimeError):
    def __init__(self, event_type: str, response: Any) -> None:
        details = _stream_terminal_details(response)
        rendered = (
            ":" + json.dumps(details, sort_keys=True, separators=(",", ":"))
            if details
            else ""
        )
        super().__init__(f"model_stream_terminal_event:{event_type}{rendered}")
        self.event_type = event_type
        self.details = details
        self.retryable = not _stream_terminal_is_explicit_client_failure(details)


def _replay_budget_finalization_tools(
    tools: CodingToolSession,
    *,
    fallback_commands: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if not tools.changed_paths:
        return ()
    issues = tools.finalization_issues()
    needs_check = any(issue.startswith("current_revision_") for issue in issues)
    replayed: list[str] = []
    if needs_check:
        # Every replay source is compared on the session's normalized command
        # form: mandatory commands arrive raw from the trusted report,
        # unresolved ledger keys are normalized, and historical run_check
        # arguments are whatever the model typed. Deduplicating on raw
        # strings re-executes the same underlying command once per spelling.
        replayed_normalized: set[str] = set()

        def _replay_check(command: str) -> str:
            result = tools.replay_call("run_check", {"command": command})
            replayed.append(command)
            replayed_normalized.add(_normalized_check_command(command))
            return str(result.get("status") or "")

        historical_commands = tuple(
            dict.fromkeys(
                _normalized_check_command(
                    str(
                        invocation.arguments.get("command")
                        if invocation.name == "run_check"
                        else invocation.result.get("normalized_command") or ""
                    )
                )
                for invocation in tools.transcript
                if invocation.name in {"run_check", "run_python_check"}
                and invocation.revision > 0
                and invocation.result.get("status") == "passed"
                and (
                    invocation.arguments.get("command")
                    if invocation.name == "run_check"
                    else invocation.result.get("normalized_command")
                )
            )
        )
        mandatory_commands = tuple(dict.fromkeys(fallback_commands))
        for command in mandatory_commands:
            if _normalized_check_command(command) in replayed_normalized:
                continue
            status = _replay_check(command)
            if status == "blocked":
                # A blocked replay means the harness may not run this command
                # here (retry limit, runtime), not that the candidate failed
                # it; a later source may still clear the obligation.
                continue
            if status != "passed":
                return tuple(replayed)
        unresolved_commands = tuple(
            str(command)
            for command in getattr(tools, "unresolved_check_commands", ())
            if str(command)
        )
        for command in unresolved_commands:
            if _normalized_check_command(command) in replayed_normalized:
                continue
            status = _replay_check(command)
            if status == "blocked":
                continue
            if status != "passed":
                return tuple(replayed)
        remaining_issues = tools.finalization_issues()
        if any(issue.startswith("current_revision_") for issue in remaining_issues):
            for command in historical_commands:
                if command in replayed_normalized:
                    continue
                status = _replay_check(command)
                if status == "blocked":
                    continue
                if status != "passed":
                    return tuple(replayed)
                if not any(
                    issue.startswith("current_revision_")
                    for issue in tools.finalization_issues()
                ):
                    break
    remaining = tools.finalization_issues()
    if remaining == ("final_diff_not_inspected_after_latest_patch",):
        tools.replay_call("git_diff", {})
    return tuple(replayed)


def _trusted_finalization_commands(
    source_report: Mapping[str, Any],
) -> tuple[str, ...]:
    commands: list[str] = []
    reproducer_plan = source_report.get("reproducer_plan")
    if isinstance(reproducer_plan, Mapping):
        raw_commands = reproducer_plan.get("executable_commands")
        if isinstance(raw_commands, (list, tuple)):
            commands.extend(str(command).strip() for command in raw_commands)
    project_commands = source_report.get("required_project_verification_commands")
    if isinstance(project_commands, (list, tuple)):
        commands.extend(str(command).strip() for command in project_commands)
    return tuple(dict.fromkeys(command for command in commands if command))


def _recorded_model_usage(
    event_store: SQLiteEventStore,
    run_id: str,
) -> tuple[int, tuple[str, ...], int]:
    """Recover API attempt and token accounting across process interruptions."""

    events = event_store.load_verified_events(run_id)
    request_count = sum(event.event_type == "model_request_started" for event in events)
    responses = tuple(
        event for event in events if event.event_type == "model_response_received"
    )
    response_ids = tuple(
        str(event.payload.get("response_id") or "")
        for event in responses
        if str(event.payload.get("response_id") or "")
    )
    total_tokens = sum(
        int(event.payload.get("total_tokens") or 0) for event in responses
    )
    return request_count, response_ids, total_tokens


def _repeated_blocked_mutation(
    transcript: tuple[Any, ...],
    *,
    limit: int = REPEATED_BLOCKED_MUTATION_LIMIT,
) -> dict[str, Any] | None:
    if limit <= 1:
        raise ValueError("repeated_blocked_mutation_limit_too_small")
    if len(transcript) < limit:
        return None
    latest = transcript[-1]
    name = str(getattr(latest, "name", ""))
    arguments = dict(getattr(latest, "arguments", {}) or {})
    result = dict(getattr(latest, "result", {}) or {})
    status = str(result.get("status") or "")
    if (
        name not in _REVISION_MUTATION_TOOL_NAMES
        or status not in _NO_EFFECT_MUTATION_STATUSES
    ):
        return None
    signature = stable_hash(
        {
            "name": name,
            "arguments": arguments,
            "status": status,
            "reason": result.get("reason"),
        }
    )
    count = 0
    for invocation in reversed(transcript):
        invocation_result = dict(getattr(invocation, "result", {}) or {})
        invocation_signature = stable_hash(
            {
                "name": str(getattr(invocation, "name", "")),
                "arguments": dict(getattr(invocation, "arguments", {}) or {}),
                "status": invocation_result.get("status"),
                "reason": invocation_result.get("reason"),
            }
        )
        if (
            invocation_result.get("status") not in _NO_EFFECT_MUTATION_STATUSES
            or invocation_signature != signature
        ):
            break
        count += 1
    if count < limit:
        family_signature, family_reason, path = _blocked_mutation_family(latest)
        if family_signature is None:
            return None
        family_count = 0
        latest_revision = int(getattr(latest, "revision", 0) or 0)
        for invocation in reversed(transcript[-12:]):
            if int(getattr(invocation, "revision", 0) or 0) != latest_revision:
                break
            invocation_result = dict(getattr(invocation, "result", {}) or {})
            invocation_name = str(getattr(invocation, "name", ""))
            if invocation_name in _MUTATION_TOOL_NAMES and invocation_result.get(
                "status"
            ) in {"applied", "no_op"}:
                break
            invocation_family, _, _ = _blocked_mutation_family(invocation)
            if invocation_family is None:
                continue
            if invocation_family != family_signature:
                break
            family_count += 1
        if family_count < limit:
            return None
        return {
            "count": family_count,
            "mode": "failure_family",
            "name": name,
            "path": path,
            "reason": str(result.get("reason") or status),
            "reason_family": family_reason,
            "signature": family_signature,
        }
    return {
        "count": count,
        "mode": "exact_action",
        "name": name,
        "path": str(arguments.get("path") or ""),
        "reason": str(result.get("reason") or status),
        "signature": signature,
    }


def _blocked_mutation_family(
    invocation: Any,
) -> tuple[str | None, str | None, str]:
    name = str(getattr(invocation, "name", ""))
    arguments = dict(getattr(invocation, "arguments", {}) or {})
    result = dict(getattr(invocation, "result", {}) or {})
    if name not in _MUTATION_TOOL_NAMES or result.get("status") != "blocked":
        return None, None, ""
    path = str(arguments.get("path") or "")
    reason = str(result.get("reason") or "blocked")
    if reason.startswith("syntax_guard_failed:"):
        parts = reason.split(":", 4)
        reason_family = ":".join(parts[:3])
    else:
        reason_family = reason.split(":", 1)[0]
    signature = stable_hash(
        {
            "name": name,
            "path": path,
            "reason_family": reason_family,
        }
    )
    return signature, reason_family, path


class OpenAIToolLoopWorkspaceCodingAgent:
    source = "openai_interactive_workspace_agent"
    runtime_version = TOOL_LOOP_RUNTIME_VERSION

    def __init__(
        self,
        *,
        executor: CommandExecutor,
        model: str = "gpt-5.5-2026-04-23",
        client: Any | None = None,
        timeout_seconds: float = 600.0,
        request_timeout_seconds: float | None = None,
        max_turns: int = DEFAULT_TOOL_LOOP_MAX_TURNS,
        max_tool_calls: int = DEFAULT_TOOL_LOOP_MAX_TOOL_CALLS,
        max_patch_bytes: int = DEFAULT_TOOL_LOOP_MAX_PATCH_BYTES,
        max_exploration_calls_per_revision: int = (
            DEFAULT_MAX_EXPLORATION_CALLS_PER_REVISION
        ),
        max_request_attempts: int = 3,
        reasoning_effort: str = "xhigh",
        requested_reasoning_effort: str | None = None,
        tool_protocol: str | None = None,
        response_transport: str | None = None,
        event_store: SQLiteEventStore | None = None,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns_must_be_positive")
        if max_tool_calls <= 0:
            raise ValueError("max_tool_calls_must_be_positive")
        if max_patch_bytes <= 0:
            raise ValueError("max_patch_bytes_must_be_positive")
        if max_exploration_calls_per_revision <= 0:
            raise ValueError("max_exploration_calls_per_revision_must_be_positive")
        if max_request_attempts <= 0:
            raise ValueError("max_request_attempts_must_be_positive")
        selected_tool_protocol = (
            (
                tool_protocol
                or os.environ.get(
                    "SOCIETY_CORE_WORKSPACE_CODING_TOOL_PROTOCOL",
                    "auto",
                )
            )
            .strip()
            .lower()
        )
        if selected_tool_protocol not in {"auto", "native", "portable"}:
            raise ValueError("invalid_tool_protocol")
        resolved_request_timeout_seconds = resolve_tool_request_timeout_seconds(
            timeout_seconds=timeout_seconds,
            requested=request_timeout_seconds,
        )
        if client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is required for OpenAIToolLoopWorkspaceCodingAgent"
                )
            client = build_openai_client(
                api_key=api_key,
                timeout_seconds=resolved_request_timeout_seconds,
                max_retries=0,
            )
            runtime_identity = openai_runtime_identity()
            self.endpoint_kind = str(runtime_identity["endpoint_kind"])
            self.endpoint_hash = str(runtime_identity["endpoint_hash"])
            self.default_header_names = tuple(runtime_identity["default_header_names"])
            self.default_headers_hash = str(runtime_identity["default_headers_hash"])
            self.response_storage_disabled = bool(
                runtime_identity["response_storage_disabled"]
            )
        else:
            self.endpoint_kind = "injected_client_unattested"
            self.endpoint_hash = stable_hash("injected_client_unattested")
            self.default_header_names = ()
            self.default_headers_hash = stable_hash({})
            self.response_storage_disabled = True
        self._client = client
        self.executor = executor
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.request_timeout_seconds = resolved_request_timeout_seconds
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.max_patch_bytes = max_patch_bytes
        self.max_exploration_calls_per_revision = max_exploration_calls_per_revision
        self.max_request_attempts = max_request_attempts
        self.reasoning_effort = reasoning_effort
        self.requested_reasoning_effort = requested_reasoning_effort or reasoning_effort
        self.tool_protocol = selected_tool_protocol
        self.response_transport = resolve_response_transport(
            endpoint_kind=self.endpoint_kind,
            requested=response_transport,
        )
        if self.response_transport == "stream" and not callable(
            getattr(getattr(self._client, "responses", None), "stream", None)
        ):
            raise ValueError("streaming_responses_transport_unavailable")
        self.development_stage = "promote"
        self.event_store = event_store
        self.last_run_id: str | None = None
        self.last_trace = _trace(status="not_started")
        self.run_namespace = "standalone"
        self.run_attempt = 1
        self._learned_tool_protocol: str | None = None

    def set_run_context(self, *, namespace: str, attempt: int) -> None:
        if not namespace:
            raise ValueError("run_namespace_required")
        if attempt < 1:
            raise ValueError("run_attempt_must_be_positive")
        self.run_namespace = namespace
        self.run_attempt = attempt

    def propose(
        self,
        *,
        source_report: Mapping[str, Any],
        workspace_root: Path,
        package: WorkspacePackageInfo,
        file_contexts: tuple[WorkspaceFileContext, ...],
        previous_verification: tuple[WorkspaceVerificationResult, ...],
        iteration: int,
    ) -> CodingAgentPatchProposal:
        selected_files = tuple(context.path for context in file_contexts)
        artifact_id = _artifact_id(source_report)
        themes = _themes(source_report)
        try:
            available_executables = probe_command_executor_executables(
                self.executor,
                root=workspace_root,
                timeout_seconds=min(max(self.timeout_seconds, 1.0), 30.0),
            )
        except RuntimeError:
            available_executables = None
        initial_prompt = _initial_prompt(
            source_report=source_report,
            workspace_root=workspace_root,
            package=package,
            file_contexts=file_contexts,
            selected_files=selected_files,
            previous_verification=previous_verification,
            development_stage=self.development_stage,
            iteration=iteration,
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            max_exploration_calls_per_revision=(
                self.max_exploration_calls_per_revision
            ),
            available_executables=available_executables,
        )
        event_run = self._start_event_run(
            source_report=source_report,
            workspace_root=workspace_root,
            iteration=iteration,
            initial_prompt=initial_prompt,
            package=package,
            file_contexts=file_contexts,
            previous_verification=previous_verification,
            available_executables=available_executables,
        )
        if event_run is not None and not event_run.created and event_run.result_hash:
            try:
                replayed = self._replay_persisted_proposal(event_run.run_id)
            except ValueError as exc:
                reason = f"persisted_event_integrity_failure:{str(exc)[:160]}"
                self.last_trace = _trace(status="blocked", blocked_reason=reason)
                return _blocked_proposal(
                    source=self.source,
                    model=self.model,
                    iteration=iteration,
                    artifact_id=artifact_id,
                    themes=themes,
                    selected_files=selected_files,
                    reason=reason,
                )
            if replayed is not None:
                self.last_trace = _trace(status="replayed")
                return replayed
        try:
            checkpoint = (
                self._load_latest_checkpoint(event_run.run_id)
                if event_run is not None and not event_run.created
                else None
            )
        except ValueError as exc:
            reason = f"persisted_event_integrity_failure:{str(exc)[:160]}"
            self.last_trace = _trace(status="blocked", blocked_reason=reason)
            return _blocked_proposal(
                source=self.source,
                model=self.model,
                iteration=iteration,
                artifact_id=artifact_id,
                themes=themes,
                selected_files=selected_files,
                reason=reason,
            )
        response_ids: list[str] = []
        request_count = 0
        total_tokens = 0
        final_raw = ""
        last_finalization_issues: tuple[str, ...] = ()
        with CodingToolSession.create(
            workspace_root,
            executor=self.executor,
            max_tool_calls=self.max_tool_calls,
            max_patch_bytes=self.max_patch_bytes,
            max_check_timeout_seconds=min(max(self.timeout_seconds, 1.0), 600.0),
            max_exploration_calls_per_revision=(
                self.max_exploration_calls_per_revision
            ),
            required_check_commands=_trusted_finalization_commands(source_report),
            available_executables=available_executables,
        ) as tools:
            conversation: list[Any] = [
                {
                    "role": "user",
                    "content": initial_prompt,
                }
            ]
            start_turn = 0
            active_tool_protocol = (
                "portable"
                if self.tool_protocol == "portable"
                or (
                    self.tool_protocol == "auto"
                    and self._learned_tool_protocol == "portable"
                )
                else "native"
            )
            if (
                self.tool_protocol == "auto"
                and self._learned_tool_protocol == "portable"
            ):
                self._record_event(
                    event_run,
                    event_type="tool_protocol_capability_reused",
                    payload={
                        "protocol": "portable",
                        "reason": "prior_observed_continuation_error",
                    },
                    idempotency_key="tool_protocol_capability_reused",
                )
            if checkpoint is not None:
                try:
                    stored_conversation = checkpoint.get("conversation")
                    if isinstance(stored_conversation, list) and stored_conversation:
                        conversation = stored_conversation
                    stored_patches = tuple(
                        WorkspaceFilePatch(**dict(item))
                        for item in checkpoint.get("patches", ())
                        if isinstance(item, Mapping)
                    )
                    tools.restore_patches(stored_patches)
                    tools.restore_usage(
                        transcript=tuple(
                            item
                            for item in checkpoint.get("tool_transcript", ())
                            if isinstance(item, Mapping)
                        ),
                        patch_bytes=int(checkpoint.get("patch_bytes") or 0),
                    )
                    tools.restore_passed_checks(
                        tuple(
                            str(item)
                            for item in checkpoint.get("passed_check_commands", ())
                        )
                    )
                    tools.restore_diff_inspection(
                        bool(
                            checkpoint.get(
                                "diff_inspected_for_current_revision",
                                False,
                            )
                        )
                    )
                    response_ids.extend(
                        str(item)
                        for item in checkpoint.get("response_ids", ())
                        if str(item)
                    )
                    total_tokens = int(checkpoint.get("total_tokens") or 0)
                    request_count = int(checkpoint.get("request_count") or 0)
                    start_turn = max(0, int(checkpoint.get("next_turn") or 0))
                    stored_tool_protocol = str(
                        checkpoint.get("active_tool_protocol") or "native"
                    )
                    if (
                        self.tool_protocol == "auto"
                        and stored_tool_protocol == "portable"
                    ):
                        active_tool_protocol = "portable"
                        self._learned_tool_protocol = "portable"
                except (OSError, TypeError, ValueError) as exc:
                    reason = f"checkpoint_restore_failed:{type(exc).__name__}:{str(exc)[:160]}"
                    self.last_trace = _trace(status="blocked", blocked_reason=reason)
                    return _blocked_proposal(
                        source=self.source,
                        model=self.model,
                        iteration=iteration,
                        artifact_id=artifact_id,
                        themes=themes,
                        selected_files=selected_files,
                        reason=reason,
                    )
            if (
                self.event_store is not None
                and event_run is not None
                and not event_run.created
            ):
                (
                    request_count,
                    recorded_response_ids,
                    total_tokens,
                ) = _recorded_model_usage(self.event_store, event_run.run_id)
                response_ids = list(recorded_response_ids)
            typed_finalization_origin = "turn_budget"
            seen_progress_signals: set[tuple[int, str, str]] = set()
            seen_suppressed_tool_states: set[tuple[int, tuple[str, ...]]] = set()
            seen_ready_to_finalize_revisions: set[int] = set()
            seen_implementation_phase_revisions = {
                int(item)
                for item in (
                    checkpoint.get("implementation_phase_revisions", ())
                    if checkpoint is not None
                    else ()
                )
            }
            seen_workflow_phase_states = {
                (int(item[0]), str(item[1]))
                for item in (
                    checkpoint.get("workflow_phase_states", ())
                    if checkpoint is not None
                    else ()
                )
                if isinstance(item, (list, tuple)) and len(item) == 2
            }
            phase_tool_violation_counts = {
                str(key): int(value)
                for key, value in (
                    checkpoint.get("phase_tool_violation_counts", {}).items()
                    if checkpoint is not None
                    and isinstance(
                        checkpoint.get("phase_tool_violation_counts"),
                        Mapping,
                    )
                    else ()
                )
            }
            for _turn in range(start_turn, self.max_turns):
                conversation = _compact_conversation(
                    conversation,
                    tools,
                    require_reasoning_items=(
                        active_tool_protocol == "native"
                        and str(self.model).startswith("gpt-5")
                    ),
                )
                suppressed_tool_names = suppressed_progress_tool_names(
                    tools.transcript,
                    current_revision=tools.revision,
                )
                suppressed_state = (
                    tools.revision,
                    tuple(sorted(suppressed_tool_names)),
                )
                if (
                    suppressed_tool_names
                    and suppressed_state not in seen_suppressed_tool_states
                ):
                    seen_suppressed_tool_states.add(suppressed_state)
                    self._record_event(
                        event_run,
                        event_type="progress_tools_suppressed",
                        payload={
                            "turn": _turn,
                            "revision": tools.revision,
                            "tool_names": sorted(suppressed_tool_names),
                            "reason": "repeated_semantically_equivalent_observation",
                        },
                        idempotency_key=(
                            f"turn:{_turn}:progress_tools_suppressed:"
                            f"{stable_hash(suppressed_state)[:16]}"
                        ),
                    )
                ready_to_finalize = bool(
                    tools.changed_paths and not tools.finalization_issues()
                )
                implementation_phase_signal = detect_post_failure_diagnostic_saturation(
                    tools.transcript,
                    current_revision=tools.revision,
                ) or detect_agent_progress_stall(
                    tools.transcript,
                    current_revision=tools.revision,
                    exploration_calls_current_revision=(
                        tools.exploration_calls_current_revision
                    ),
                )
                if (
                    implementation_phase_signal is not None
                    and tools.revision not in seen_implementation_phase_revisions
                ):
                    seen_implementation_phase_revisions.add(tools.revision)
                    self._record_event(
                        event_run,
                        event_type="implementation_phase_entered",
                        payload={
                            "turn": _turn,
                            "revision": tools.revision,
                            "reason": implementation_phase_signal.reason,
                            "count": implementation_phase_signal.count,
                            "exploration_tools_retained": False,
                            "verification_tools_retained": False,
                            "mutation_tools_retained": True,
                        },
                        idempotency_key=(
                            f"revision:{tools.revision}:implementation_phase_entered"
                        ),
                    )
                implementation_phase_required = (
                    tools.revision in seen_implementation_phase_revisions
                )
                workflow_phase = _workflow_phase(
                    tools,
                    implementation_required=implementation_phase_required,
                )
                workflow_phase_state = (tools.revision, workflow_phase)
                if workflow_phase_state not in seen_workflow_phase_states:
                    seen_workflow_phase_states.add(workflow_phase_state)
                    diagnostic_count = post_failure_diagnostic_count(
                        tools.transcript,
                        current_revision=tools.revision,
                    )
                    phase_message = _workflow_phase_message(
                        workflow_phase,
                        revision=tools.revision,
                        diagnostic_count=diagnostic_count,
                        mutation_recovery_requirement=(
                            tools.mutation_recovery_requirement
                        ),
                        mutation_recovery_edit_lease=(
                            tools.mutation_recovery_edit_lease
                        ),
                        failure_diagnostic=_latest_failed_check_diagnostic(
                            tools.transcript,
                            current_revision=tools.revision,
                        ),
                        unresolved_check_replays=tools.unresolved_check_replays,
                    )
                    if phase_message is not None:
                        conversation.append(phase_message)
                    self._record_event(
                        event_run,
                        event_type="workflow_phase_entered",
                        payload={
                            "turn": _turn,
                            "revision": tools.revision,
                            "phase": workflow_phase,
                            "post_failure_diagnostic_calls": diagnostic_count,
                            "mutation_recovery": (
                                tools.mutation_recovery_requirement
                                if workflow_phase == "mutation_recovery"
                                else None
                            ),
                            "mutation_recovery_relocation": (
                                tools.mutation_recovery_relocation_requirement
                                if workflow_phase == "mutation_recovery_relocate"
                                else None
                            ),
                            "mutation_recovery_edit_lease": (
                                tools.mutation_recovery_edit_lease
                                if workflow_phase == "mutation_recovery_edit"
                                else None
                            ),
                            "latest_failure": (
                                _latest_failed_check_diagnostic(
                                    tools.transcript,
                                    current_revision=tools.revision,
                                )
                                if workflow_phase == "repair_diagnosis"
                                else None
                            ),
                        },
                        idempotency_key=(
                            f"revision:{tools.revision}:workflow_phase:{workflow_phase}"
                        ),
                    )
                if (
                    ready_to_finalize
                    and tools.revision not in seen_ready_to_finalize_revisions
                ):
                    seen_ready_to_finalize_revisions.add(tools.revision)
                    self._record_event(
                        event_run,
                        event_type="evidence_satisfied_tools_suppressed",
                        payload={
                            "turn": _turn,
                            "revision": tools.revision,
                            "changed_paths": tools.changed_paths,
                            "suppressed_tool_classes": (
                                "exploration",
                                "verification",
                                "diff",
                            ),
                            "mutation_tools_retained": True,
                        },
                        idempotency_key=(
                            f"revision:{tools.revision}:"
                            "evidence_satisfied_tools_suppressed"
                        ),
                    )
                active_tool_definitions = _tool_definitions_for_workflow_phase(
                    coding_tool_definitions(
                        allow_exploration=tools.exploration_tools_available,
                        allow_verification=tools.verification_tools_available,
                        allow_diff=tools.diff_tool_available,
                        blocked_tool_names=suppressed_tool_names,
                        available_executables=available_executables,
                    ),
                    phase=workflow_phase,
                    mutation_recovery_requirement=(tools.mutation_recovery_requirement),
                    mutation_recovery_edit_lease=(tools.mutation_recovery_edit_lease),
                )
                active_tool_names = frozenset(
                    definition["name"] for definition in active_tool_definitions
                )
                request = {
                    "model": self.model,
                    "tools": active_tool_definitions,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "max_output_tokens": 20_000,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "interactive_workspace_result",
                            "strict": True,
                            "schema": _final_response_schema(),
                        }
                    },
                    "timeout": self.request_timeout_seconds,
                }
                request_input = _tool_protocol_request_input(
                    conversation,
                    tools,
                    protocol=active_tool_protocol,
                    max_replay_chars=DEFAULT_PORTABLE_REPLAY_MAX_CHARS,
                )
                portable_replay_max_chars = DEFAULT_PORTABLE_REPLAY_MAX_CHARS
                _configure_request_input(
                    request,
                    request_input=request_input,
                    event_run=event_run,
                    turn=_turn,
                    model=self.model,
                    active_tool_definitions=active_tool_definitions,
                )
                if str(self.model).startswith("gpt-5"):
                    request["reasoning"] = {"effort": self.reasoning_effort}
                response = None
                for request_attempt in range(self.max_request_attempts):
                    request_instance_id = os.urandom(12).hex()
                    request_count += 1
                    request_event_payload = {
                        "turn": _turn,
                        "request_attempt": request_attempt + 1,
                        "request_instance_id": request_instance_id,
                        "model": self.model,
                        "reasoning_effort": self.reasoning_effort,
                        "request_timeout_seconds": self.request_timeout_seconds,
                        "response_transport": self.response_transport,
                        "tool_protocol": active_tool_protocol,
                        "input_hash": stable_hash(_serialize_item(request["input"])),
                        "input_chars": _serialized_chars(request["input"]),
                        "portable_replay_max_chars": (
                            portable_replay_max_chars
                            if active_tool_protocol == "portable"
                            else None
                        ),
                        "tool_names": tuple(tool["name"] for tool in request["tools"]),
                    }
                    self._record_event(
                        event_run,
                        event_type="model_request_started",
                        payload=request_event_payload,
                        idempotency_key=f"model_request:{request_instance_id}",
                    )
                    try:
                        with openai_call_watchdog(
                            self.request_timeout_seconds,
                            label="workspace coding model request",
                        ):
                            response, stream_event_count = _create_model_response(
                                self._client.responses,
                                prepare_openai_response_request(request),
                                transport=self.response_transport,
                            )
                        if self.response_transport == "stream":
                            stream_payload = {
                                "turn": _turn,
                                "request_attempt": request_attempt + 1,
                                "request_instance_id": request_instance_id,
                                "stream_event_count": stream_event_count,
                                "response_id": str(getattr(response, "id", None) or ""),
                            }
                            self._record_event(
                                event_run,
                                event_type="model_stream_completed",
                                payload=stream_payload,
                                idempotency_key=(
                                    "model_stream_completed:"
                                    f"{request_instance_id}:"
                                    f"{stable_hash(stream_payload)[:16]}"
                                ),
                            )
                        break
                    except Exception as exc:
                        retryable = _is_retryable_model_request_error(exc)
                        failure_payload = {
                            "turn": _turn,
                            "request_attempt": request_attempt + 1,
                            "request_instance_id": request_instance_id,
                            "error_type": type(exc).__name__,
                            "retryable": retryable,
                            "error": str(exc)[:240],
                        }
                        self._record_event(
                            event_run,
                            event_type="model_request_failed",
                            payload=failure_payload,
                            idempotency_key=(
                                f"model_request_failed:{request_instance_id}"
                            ),
                        )
                        if (
                            retryable
                            and request_attempt + 1 < self.max_request_attempts
                        ):
                            previous_protocol = active_tool_protocol
                            if (
                                self.tool_protocol == "auto"
                                and active_tool_protocol == "native"
                                and _turn > 0
                            ):
                                active_tool_protocol = "portable"
                                self._learned_tool_protocol = "portable"
                                self._record_event(
                                    event_run,
                                    event_type=("tool_protocol_fallback_activated"),
                                    payload={
                                        "turn": _turn,
                                        "failed_request_attempt": (request_attempt + 1),
                                        "from_protocol": "native",
                                        "to_protocol": "portable",
                                        "reason": "retryable_continuation_error",
                                    },
                                    idempotency_key=(
                                        f"turn:{_turn}:tool_protocol_fallback"
                                    ),
                                )
                            if active_tool_protocol == "portable":
                                next_attempt = request_attempt + 2
                                previous_replay_max_chars = portable_replay_max_chars
                                portable_replay_max_chars = (
                                    _portable_replay_budget_for_attempt(next_attempt)
                                )
                                request_input = _tool_protocol_request_input(
                                    conversation,
                                    tools,
                                    protocol=active_tool_protocol,
                                    max_replay_chars=portable_replay_max_chars,
                                )
                                _configure_request_input(
                                    request,
                                    request_input=request_input,
                                    event_run=event_run,
                                    turn=_turn,
                                    model=self.model,
                                    active_tool_definitions=(active_tool_definitions),
                                )
                                if (
                                    previous_protocol == "portable"
                                    and portable_replay_max_chars
                                    < previous_replay_max_chars
                                ):
                                    self._record_event(
                                        event_run,
                                        event_type=(
                                            "request_context_compaction_escalated"
                                        ),
                                        payload={
                                            "turn": _turn,
                                            "failed_request_attempt": (
                                                request_attempt + 1
                                            ),
                                            "next_request_attempt": next_attempt,
                                            "request_instance_id": (
                                                request_instance_id
                                            ),
                                            "from_max_chars": (
                                                previous_replay_max_chars
                                            ),
                                            "to_max_chars": (portable_replay_max_chars),
                                            "input_chars": _serialized_chars(
                                                request_input
                                            ),
                                        },
                                        idempotency_key=(
                                            "request_context_compaction:"
                                            f"{request_instance_id}"
                                        ),
                                    )
                            retry_delay = _model_request_retry_delay_seconds(
                                exc,
                                request_attempt,
                            )
                            self._record_event(
                                event_run,
                                event_type="model_request_retry_scheduled",
                                payload={
                                    "turn": _turn,
                                    "request_attempt": request_attempt + 1,
                                    "request_instance_id": request_instance_id,
                                    "next_request_attempt": request_attempt + 2,
                                    "delay_seconds": retry_delay,
                                },
                                idempotency_key=(
                                    f"model_request_retry:{request_instance_id}"
                                ),
                            )
                            sleep(retry_delay)
                            continue
                        reason = (
                            f"openai_tool_loop_failed:{type(exc).__name__}:"
                            f"attempts={request_attempt + 1}:{str(exc)[:220]}"
                        )
                        self.last_trace = _trace(
                            status="blocked",
                            response_ids=tuple(response_ids),
                            request_count=request_count,
                            tool_calls=tuple(
                                canonicalize(item) for item in tools.transcript
                            ),
                            total_tokens=total_tokens,
                            blocked_reason=reason,
                        )
                        return _blocked_proposal(
                            source=self.source,
                            model=self.model,
                            iteration=iteration,
                            artifact_id=artifact_id,
                            themes=themes,
                            selected_files=selected_files,
                            reason=reason,
                        )
                assert response is not None
                response_id = str(getattr(response, "id", "") or "")
                if response_id:
                    response_ids.append(response_id)
                total_tokens += _response_total_tokens(response)
                output = list(getattr(response, "output", ()) or ())
                response_payload = {
                    "turn": _turn,
                    "request_instance_id": request_instance_id,
                    "response_id": response_id,
                    "tool_protocol": active_tool_protocol,
                    "output": _serialize_item(output),
                    "output_text": str(getattr(response, "output_text", "") or ""),
                    "total_tokens": _response_total_tokens(response),
                }
                self._record_event(
                    event_run,
                    event_type="model_response_received",
                    payload=response_payload,
                    idempotency_key=(
                        f"model_response:{request_instance_id}:"
                        f"{stable_hash(response_payload)[:16]}"
                    ),
                )
                conversation.extend(output)
                function_calls = tuple(
                    item
                    for item in output
                    if _item_value(item, "type") == "function_call"
                )
                if function_calls:
                    repeated_phase_tool_violation: dict[str, Any] | None = None
                    for call_index, call in enumerate(function_calls):
                        arguments: dict[str, Any] = {}
                        tool_name = str(_item_value(call, "name", ""))
                        try:
                            arguments = json.loads(
                                str(_item_value(call, "arguments", "{}"))
                            )
                            if not isinstance(arguments, dict):
                                raise ValueError("tool_arguments_must_be_object")
                        except (json.JSONDecodeError, ValueError) as exc:
                            result = {
                                "status": "blocked",
                                "reason": f"invalid_tool_arguments:{type(exc).__name__}",
                            }
                        else:
                            if tool_name not in active_tool_names:
                                violation_signature = stable_hash(
                                    {
                                        "phase": workflow_phase,
                                        "name": tool_name,
                                        "arguments": arguments,
                                    }
                                )
                                violation_count = (
                                    phase_tool_violation_counts.get(
                                        violation_signature,
                                        0,
                                    )
                                    + 1
                                )
                                phase_tool_violation_counts[violation_signature] = (
                                    violation_count
                                )
                                result = {
                                    "status": "blocked",
                                    "reason": (
                                        "tool_unavailable_in_workflow_phase:"
                                        f"{workflow_phase}:{tool_name}"
                                    ),
                                    "violation_count": violation_count,
                                }
                                if (
                                    violation_count
                                    >= REPEATED_PHASE_TOOL_VIOLATION_LIMIT
                                ):
                                    repeated_phase_tool_violation = {
                                        "phase": workflow_phase,
                                        "name": tool_name,
                                        "count": violation_count,
                                        "signature": violation_signature,
                                    }
                            else:
                                result = tools.call(tool_name, arguments)
                        call_id = str(_item_value(call, "call_id", ""))
                        tool_event_payload = {
                            "turn": _turn,
                            "request_instance_id": request_instance_id,
                            "call_index": call_index,
                            "call_id": call_id,
                            "name": tool_name,
                            "arguments": arguments,
                            "result": result,
                        }
                        self._record_event(
                            event_run,
                            event_type="tool_call_completed",
                            payload=tool_event_payload,
                            idempotency_key=(
                                f"tool:{request_instance_id}:"
                                f"{call_id or call_index}:"
                                f"{stable_hash(tool_event_payload)[:16]}"
                            ),
                        )
                        conversation.append(
                            {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": tool_result_json(result),
                            }
                        )
                    if repeated_phase_tool_violation is not None:
                        typed_finalization_origin = "phase_tool_violation"
                        replayed_checks = _replay_budget_finalization_tools(
                            tools,
                            fallback_commands=_trusted_finalization_commands(
                                source_report
                            ),
                        )
                        finalization_issues = tools.finalization_issues()
                        self._record_event(
                            event_run,
                            event_type="phase_tool_violation_circuit_opened",
                            payload={
                                **repeated_phase_tool_violation,
                                "turn": _turn,
                                "replayed_checks": replayed_checks,
                                "finalization_issues": finalization_issues,
                            },
                            idempotency_key=(
                                "phase_tool_violation_circuit:"
                                + repeated_phase_tool_violation["signature"]
                            ),
                        )
                        if tools.changed_paths and not finalization_issues:
                            break
                        reason = (
                            "repeated_phase_tool_violation_circuit_open:"
                            + str(repeated_phase_tool_violation["phase"])
                            + ":"
                            + str(repeated_phase_tool_violation["name"])
                        )
                        self.last_trace = _trace(
                            status="blocked",
                            response_ids=tuple(response_ids),
                            request_count=request_count,
                            tool_calls=tuple(
                                canonicalize(item) for item in tools.transcript
                            ),
                            total_tokens=total_tokens,
                            blocked_reason=reason,
                        )
                        proposal = _blocked_proposal(
                            source=self.source,
                            model=self.model,
                            iteration=iteration,
                            artifact_id=artifact_id,
                            themes=themes,
                            selected_files=selected_files,
                            reason=reason,
                        )
                        self._persist_blocked_proposal(event_run, proposal)
                        return proposal
                    if tools.infrastructure_failure_reason:
                        reason = (
                            "execution_infrastructure_unavailable:"
                            + tools.infrastructure_failure_reason
                        )
                        self._record_event(
                            event_run,
                            event_type="execution_infrastructure_failed",
                            payload={"turn": _turn, "reason": reason},
                            idempotency_key=f"turn:{_turn}:execution_infrastructure_failed",
                        )
                        self.last_trace = _trace(
                            status="blocked",
                            response_ids=tuple(response_ids),
                            request_count=request_count,
                            tool_calls=tuple(
                                canonicalize(item) for item in tools.transcript
                            ),
                            total_tokens=total_tokens,
                            blocked_reason=reason,
                        )
                        return _blocked_proposal(
                            source=self.source,
                            model=self.model,
                            iteration=iteration,
                            artifact_id=artifact_id,
                            themes=themes,
                            selected_files=selected_files,
                            reason=reason,
                        )
                    repeated_mutation = _repeated_blocked_mutation(tools.transcript)
                    if repeated_mutation is not None:
                        typed_finalization_origin = "repeated_blocked_mutation"
                        replayed_checks = _replay_budget_finalization_tools(
                            tools,
                            fallback_commands=_trusted_finalization_commands(
                                source_report
                            ),
                        )
                        finalization_issues = tools.finalization_issues()
                        self._record_event(
                            event_run,
                            event_type="repeated_blocked_mutation_detected",
                            payload={
                                **repeated_mutation,
                                "turn": _turn,
                                "replayed_checks": replayed_checks,
                                "finalization_issues": finalization_issues,
                            },
                            idempotency_key=(f"turn:{_turn}:repeated_blocked_mutation"),
                        )
                        if tools.changed_paths and not finalization_issues:
                            break
                        reason = (
                            "repeated_blocked_mutation_circuit_open:"
                            + str(repeated_mutation["name"])
                            + ":"
                            + ",".join(finalization_issues)
                        )
                        self.last_trace = _trace(
                            status="blocked",
                            response_ids=tuple(response_ids),
                            request_count=request_count,
                            tool_calls=tuple(
                                canonicalize(item) for item in tools.transcript
                            ),
                            total_tokens=total_tokens,
                            blocked_reason=reason,
                        )
                        proposal = _blocked_proposal(
                            source=self.source,
                            model=self.model,
                            iteration=iteration,
                            artifact_id=artifact_id,
                            themes=themes,
                            selected_files=selected_files,
                            reason=reason,
                        )
                        self._persist_blocked_proposal(event_run, proposal)
                        return proposal
                    progress_signal = detect_agent_progress_stall(
                        tools.transcript,
                        current_revision=tools.revision,
                        exploration_calls_current_revision=(
                            tools.exploration_calls_current_revision
                        ),
                    )
                    if progress_signal is not None:
                        if (
                            progress_signal.reason == "repeated_action_observation"
                            and progress_signal.count
                            >= REPEATED_ACTION_OBSERVATION_HARD_LIMIT
                        ):
                            reason = (
                                "repeated_action_observation_circuit_open:"
                                f"revision_{progress_signal.revision}:"
                                f"count_{progress_signal.count}"
                            )
                            self._record_event(
                                event_run,
                                event_type="progress_circuit_breaker_opened",
                                payload={
                                    "turn": _turn,
                                    "reason": progress_signal.reason,
                                    "count": progress_signal.count,
                                    "revision": progress_signal.revision,
                                    "signature": progress_signal.signature,
                                    "required_action": (
                                        "discard_candidate_and_try_independent_strategy"
                                    ),
                                },
                                idempotency_key=(
                                    f"turn:{_turn}:progress_circuit_breaker:"
                                    f"{progress_signal.signature}"
                                ),
                            )
                            self.last_trace = _trace(
                                status="blocked",
                                response_ids=tuple(response_ids),
                                request_count=request_count,
                                tool_calls=tuple(
                                    canonicalize(item) for item in tools.transcript
                                ),
                                total_tokens=total_tokens,
                                blocked_reason=reason,
                            )
                            proposal = _blocked_proposal(
                                source=self.source,
                                model=self.model,
                                iteration=iteration,
                                artifact_id=artifact_id,
                                themes=themes,
                                selected_files=selected_files,
                                reason=reason,
                            )
                            self._persist_blocked_proposal(event_run, proposal)
                            return proposal
                        progress_key = (
                            progress_signal.revision,
                            progress_signal.reason,
                            progress_signal.signature,
                        )
                        if progress_key not in seen_progress_signals:
                            seen_progress_signals.add(progress_key)
                            conversation.append(
                                _progress_intervention_message(progress_signal)
                            )
                            self._record_event(
                                event_run,
                                event_type="progress_intervention",
                                payload={
                                    "turn": _turn,
                                    "reason": progress_signal.reason,
                                    "count": progress_signal.count,
                                    "revision": progress_signal.revision,
                                    "signature": progress_signal.signature,
                                    "required_action": (
                                        progress_signal.required_action
                                    ),
                                },
                                idempotency_key=(
                                    f"turn:{_turn}:progress_intervention:"
                                    f"{progress_signal.signature}"
                                ),
                            )
                    conversation.append(
                        _execution_budget_message(
                            turn=_turn,
                            max_turns=self.max_turns,
                            tool_calls_used=len(tools.transcript),
                            max_tool_calls=self.max_tool_calls,
                            changed_path_count=len(tools.changed_paths),
                            finalization_issues=tools.finalization_issues(),
                            exploration_calls_current_revision=(
                                tools.exploration_calls_current_revision
                            ),
                            suppressed_tool_names=suppressed_tool_names,
                        )
                    )
                    try:
                        checkpoint_patches = tools.export_patches()
                    except (OSError, ValueError) as exc:
                        reason = f"unsafe_patch_export:{type(exc).__name__}"
                        self.last_trace = _trace(
                            status="blocked",
                            response_ids=tuple(response_ids),
                            request_count=request_count,
                            tool_calls=tuple(
                                canonicalize(item) for item in tools.transcript
                            ),
                            total_tokens=total_tokens,
                            blocked_reason=reason,
                        )
                        return _blocked_proposal(
                            source=self.source,
                            model=self.model,
                            iteration=iteration,
                            artifact_id=artifact_id,
                            themes=themes,
                            selected_files=selected_files,
                            reason=reason,
                        )
                    self._record_event(
                        event_run,
                        event_type="checkpoint",
                        payload={
                            "turn": _turn,
                            "next_turn": _turn + 1,
                            "conversation": _serialize_item(conversation),
                            "patches": canonicalize(checkpoint_patches),
                            "tool_transcript": canonicalize(tools.transcript),
                            "patch_bytes": tools.patch_bytes,
                            "passed_check_commands": tools.passed_check_commands,
                            "diff_inspected_for_current_revision": (
                                tools.diff_inspected_for_current_revision
                            ),
                            "response_ids": tuple(response_ids),
                            "request_count": request_count,
                            "total_tokens": total_tokens,
                            "active_tool_protocol": active_tool_protocol,
                            "implementation_phase_revisions": tuple(
                                sorted(seen_implementation_phase_revisions)
                            ),
                            "workflow_phase_states": tuple(
                                sorted(seen_workflow_phase_states)
                            ),
                            "phase_tool_violation_counts": dict(
                                sorted(phase_tool_violation_counts.items())
                            ),
                        },
                        idempotency_key=f"turn:{_turn}:checkpoint",
                    )
                    continue

                final_raw = str(getattr(response, "output_text", "") or "")
                try:
                    final = json.loads(final_raw)
                    if not isinstance(final, dict):
                        raise ValueError("final_response_must_be_object")
                except (json.JSONDecodeError, ValueError) as exc:
                    issues_before_replay = tools.finalization_issues()
                    replayed_checks = _replay_budget_finalization_tools(
                        tools,
                        fallback_commands=_trusted_finalization_commands(source_report),
                    )
                    finalization_issues = tools.finalization_issues()
                    if replayed_checks or finalization_issues != issues_before_replay:
                        self._record_event(
                            event_run,
                            event_type=("invalid_final_response_finalization_replayed"),
                            payload={
                                "turn": _turn,
                                "error_type": type(exc).__name__,
                                "issues_before": issues_before_replay,
                                "issues_after": finalization_issues,
                                "replayed_checks": replayed_checks,
                                "diff_inspected_for_current_revision": (
                                    tools.diff_inspected_for_current_revision
                                ),
                            },
                            idempotency_key=(
                                f"turn:{_turn}:invalid_final_response_"
                                "finalization_replayed"
                            ),
                        )
                    if tools.changed_paths and not finalization_issues:
                        try:
                            patches = tools.export_patches()
                        except (OSError, ValueError) as export_exc:
                            reason = f"unsafe_patch_export:{type(export_exc).__name__}"
                            self.last_trace = _trace(
                                status="blocked",
                                response_ids=tuple(response_ids),
                                request_count=request_count,
                                tool_calls=tuple(
                                    canonicalize(item) for item in tools.transcript
                                ),
                                total_tokens=total_tokens,
                                blocked_reason=reason,
                            )
                            return _blocked_proposal(
                                source=self.source,
                                model=self.model,
                                iteration=iteration,
                                artifact_id=artifact_id,
                                themes=themes,
                                selected_files=selected_files,
                                reason=reason,
                                raw_response_hash=stable_hash(final_raw),
                            )
                        final_commands = tools.passed_check_commands
                        recovery_payload = {
                            "artifact_id": artifact_id,
                            "error_type": type(exc).__name__,
                            "origin": "typed_invalid_final_response_recovery",
                            "patches": canonicalize(patches),
                            "themes": themes,
                            "verification_commands": final_commands,
                        }
                        proposal = CodingAgentPatchProposal(
                            proposal_id=(
                                "interactive_proposal_"
                                + stable_hash((iteration, recovery_payload))[:24]
                            ),
                            source=self.source,
                            model=self.model,
                            iteration=iteration,
                            artifact_id=artifact_id,
                            themes=themes,
                            patches=patches,
                            verification_commands=final_commands,
                            selected_files=selected_files,
                            rationale=(
                                "Typed finalization preserved a gate-ready candidate "
                                "after the model's final structured response was "
                                "unparseable."
                            ),
                            support_refs=tools.changed_paths,
                            raw_response_hash=stable_hash(final_raw),
                            blocked_reason=None,
                            stage=self.development_stage,
                        )
                        self.last_trace = _trace(
                            status="completed",
                            response_ids=tuple(response_ids),
                            request_count=request_count,
                            tool_calls=tuple(
                                canonicalize(item) for item in tools.transcript
                            ),
                            total_tokens=total_tokens,
                            blocked_reason=None,
                        )
                        self._record_event(
                            event_run,
                            event_type="invalid_final_response_recovered",
                            payload={
                                "turn": _turn,
                                "error_type": type(exc).__name__,
                                "raw_response_hash": stable_hash(final_raw),
                                "changed_paths": tools.changed_paths,
                                "patch_count": len(patches),
                                "verification_commands": final_commands,
                            },
                            idempotency_key=(
                                f"turn:{_turn}:invalid_final_response_recovered"
                            ),
                        )
                        self._persist_completed_proposal(event_run, proposal)
                        return proposal
                    reason = f"tool_loop_final_response_invalid:{type(exc).__name__}"
                    self.last_trace = _trace(
                        status="blocked",
                        response_ids=tuple(response_ids),
                        request_count=request_count,
                        tool_calls=tuple(
                            canonicalize(item) for item in tools.transcript
                        ),
                        total_tokens=total_tokens,
                        blocked_reason=reason,
                    )
                    return _blocked_proposal(
                        source=self.source,
                        model=self.model,
                        iteration=iteration,
                        artifact_id=artifact_id,
                        themes=themes,
                        selected_files=selected_files,
                        reason=reason,
                        raw_response_hash=stable_hash(final_raw),
                    )
                try:
                    patches = tools.export_patches()
                except (OSError, ValueError) as exc:
                    reason = f"unsafe_patch_export:{type(exc).__name__}"
                    self.last_trace = _trace(
                        status="blocked",
                        response_ids=tuple(response_ids),
                        request_count=request_count,
                        tool_calls=tuple(
                            canonicalize(item) for item in tools.transcript
                        ),
                        total_tokens=total_tokens,
                        blocked_reason=reason,
                    )
                    return _blocked_proposal(
                        source=self.source,
                        model=self.model,
                        iteration=iteration,
                        artifact_id=artifact_id,
                        themes=themes,
                        selected_files=selected_files,
                        reason=reason,
                        raw_response_hash=stable_hash(final_raw),
                    )
                final_block = final.get("blocked_reason")
                if final_block:
                    reason = str(final_block)[:500]
                else:
                    reason = None
                if reason is None:
                    last_finalization_issues = tools.finalization_issues()
                    if last_finalization_issues:
                        feedback = {
                            "status": "FINALIZATION_REJECTED",
                            "issues": last_finalization_issues,
                            "repair_guidance": _finalization_repair_guidance(
                                last_finalization_issues
                            ),
                            "required_action": (
                                "Continue using repository tools. Apply a concrete patch if "
                                "needed, run checks on the latest revision, and inspect the "
                                "latest diff before returning the final JSON again."
                            ),
                        }
                        conversation.append(
                            {
                                "role": "user",
                                "content": "FINALIZATION_REJECTED\n"
                                + json.dumps(
                                    feedback,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            }
                        )
                        self._record_event(
                            event_run,
                            event_type="finalization_rejected",
                            payload={
                                "turn": _turn,
                                "issues": last_finalization_issues,
                            },
                            idempotency_key=f"turn:{_turn}:finalization_rejected",
                        )
                        self._record_event(
                            event_run,
                            event_type="checkpoint",
                            payload={
                                "turn": _turn,
                                "next_turn": _turn + 1,
                                "conversation": _serialize_item(conversation),
                                "patches": canonicalize(patches),
                                "tool_transcript": canonicalize(tools.transcript),
                                "patch_bytes": tools.patch_bytes,
                                "passed_check_commands": tools.passed_check_commands,
                                "diff_inspected_for_current_revision": (
                                    tools.diff_inspected_for_current_revision
                                ),
                                "response_ids": tuple(response_ids),
                                "request_count": request_count,
                                "total_tokens": total_tokens,
                                "active_tool_protocol": active_tool_protocol,
                                "implementation_phase_revisions": tuple(
                                    sorted(seen_implementation_phase_revisions)
                                ),
                                "workflow_phase_states": tuple(
                                    sorted(seen_workflow_phase_states)
                                ),
                                "phase_tool_violation_counts": dict(
                                    sorted(phase_tool_violation_counts.items())
                                ),
                            },
                            idempotency_key=f"turn:{_turn}:finalization_checkpoint",
                        )
                        continue
                status = "completed" if reason is None else "blocked"
                self.last_trace = _trace(
                    status=status,
                    response_ids=tuple(response_ids),
                    request_count=request_count,
                    tool_calls=tuple(canonicalize(item) for item in tools.transcript),
                    total_tokens=total_tokens,
                    blocked_reason=reason,
                )
                if reason is not None:
                    return _blocked_proposal(
                        source=self.source,
                        model=self.model,
                        iteration=iteration,
                        artifact_id=artifact_id,
                        themes=themes,
                        selected_files=selected_files,
                        reason=reason,
                        raw_response_hash=stable_hash(final_raw),
                    )
                final_commands = tools.passed_check_commands
                proposal = CodingAgentPatchProposal(
                    proposal_id=f"interactive_proposal_{stable_hash((iteration, patches, final))[:24]}",
                    source=self.source,
                    model=self.model,
                    iteration=iteration,
                    artifact_id=str(final.get("artifact_id") or artifact_id or "")
                    or None,
                    themes=tuple(
                        str(theme)[:80]
                        for theme in final.get("themes", themes)
                        if str(theme).strip()
                    )[:12]
                    or themes,
                    patches=patches,
                    verification_commands=final_commands,
                    selected_files=selected_files,
                    rationale=str(final.get("rationale") or "")[:4000],
                    support_refs=tuple(
                        str(ref)[:200]
                        for ref in final.get("support_refs", ())
                        if str(ref).strip()
                    )[:24],
                    raw_response_hash=stable_hash(final_raw),
                    blocked_reason=None,
                    stage=self.development_stage,
                )
                self._persist_completed_proposal(event_run, proposal)
                return proposal

            replayed_checks = _replay_budget_finalization_tools(
                tools,
                fallback_commands=_trusted_finalization_commands(source_report),
            )
            if replayed_checks:
                replay_event_type = {
                    "turn_budget": "budget_finalization_replayed",
                    "repeated_blocked_mutation": (
                        "repeated_mutation_finalization_replayed"
                    ),
                    "phase_tool_violation": (
                        "phase_tool_violation_finalization_replayed"
                    ),
                }[typed_finalization_origin]
                self._record_event(
                    event_run,
                    event_type=replay_event_type,
                    payload={
                        "commands": replayed_checks,
                        "finalization_issues": tools.finalization_issues(),
                    },
                    idempotency_key=replay_event_type,
                )
            if tools.changed_paths:
                last_finalization_issues = tools.finalization_issues()
            if tools.changed_paths and not last_finalization_issues:
                try:
                    patches = tools.export_patches()
                except (OSError, ValueError) as exc:
                    reason = f"unsafe_patch_export:{type(exc).__name__}"
                    self.last_trace = _trace(
                        status="blocked",
                        response_ids=tuple(response_ids),
                        request_count=request_count,
                        tool_calls=tuple(
                            canonicalize(item) for item in tools.transcript
                        ),
                        total_tokens=total_tokens,
                        blocked_reason=reason,
                    )
                    return _blocked_proposal(
                        source=self.source,
                        model=self.model,
                        iteration=iteration,
                        artifact_id=artifact_id,
                        themes=themes,
                        selected_files=selected_files,
                        reason=reason,
                    )
                final_commands = tools.passed_check_commands
                finalization_payload = {
                    "artifact_id": artifact_id,
                    "origin": f"typed_{typed_finalization_origin}_finalization",
                    "patches": canonicalize(patches),
                    "themes": themes,
                    "verification_commands": final_commands,
                }
                proposal = CodingAgentPatchProposal(
                    proposal_id=(
                        "interactive_proposal_"
                        + stable_hash((iteration, finalization_payload))[:24]
                    ),
                    source=self.source,
                    model=self.model,
                    iteration=iteration,
                    artifact_id=artifact_id,
                    themes=themes,
                    patches=patches,
                    verification_commands=final_commands,
                    selected_files=selected_files,
                    rationale=(
                        "Typed finalization preserved a gate-ready candidate after "
                        + {
                            "turn_budget": (
                                "the interactive turn budget was exhausted."
                            ),
                            "repeated_blocked_mutation": (
                                "a repeated blocked mutation opened the circuit."
                            ),
                            "phase_tool_violation": (
                                "a repeated workflow-phase violation opened the circuit."
                            ),
                        }[typed_finalization_origin]
                    ),
                    support_refs=tools.changed_paths,
                    raw_response_hash=stable_hash(finalization_payload),
                    blocked_reason=None,
                    stage=self.development_stage,
                )
                self.last_trace = _trace(
                    status="completed",
                    response_ids=tuple(response_ids),
                    request_count=request_count,
                    tool_calls=tuple(canonicalize(item) for item in tools.transcript),
                    total_tokens=total_tokens,
                    blocked_reason=None,
                )
                completed_event_type = {
                    "turn_budget": "budget_finalization_completed",
                    "repeated_blocked_mutation": (
                        "repeated_mutation_finalization_completed"
                    ),
                    "phase_tool_violation": (
                        "phase_tool_violation_finalization_completed"
                    ),
                }[typed_finalization_origin]
                self._record_event(
                    event_run,
                    event_type=completed_event_type,
                    payload={
                        "changed_paths": tools.changed_paths,
                        "patch_count": len(patches),
                        "tool_call_count": len(tools.transcript),
                        "turn_budget": self.max_turns,
                        "verification_commands": final_commands,
                    },
                    idempotency_key=completed_event_type,
                )
                self._persist_completed_proposal(event_run, proposal)
                return proposal

        reason = (
            "tool_loop_finalization_requirements_unmet:"
            + ",".join(last_finalization_issues)
            if last_finalization_issues
            else "tool_loop_turn_budget_exhausted"
        )
        self.last_trace = _trace(
            status="blocked",
            response_ids=tuple(response_ids),
            request_count=request_count,
            tool_calls=tuple(canonicalize(item) for item in tools.transcript),
            total_tokens=total_tokens,
            blocked_reason=reason,
        )
        proposal = _blocked_proposal(
            source=self.source,
            model=self.model,
            iteration=iteration,
            artifact_id=artifact_id,
            themes=themes,
            selected_files=selected_files,
            reason=reason,
            raw_response_hash=stable_hash(final_raw) if final_raw else None,
        )
        self._persist_blocked_proposal(event_run, proposal)
        return proposal

    def seal_last_proposal_run(
        self,
        proposal: CodingAgentPatchProposal,
    ) -> None:
        """Finalize an explicit proposal once its caller has consumed the attempt."""

        if self.event_store is None or self.last_run_id is None:
            return
        run = self.event_store.get_run(self.last_run_id)
        if run.status != "running":
            return
        if proposal.blocked_reason is not None:
            self._persist_blocked_proposal(run, proposal)
        else:
            self._persist_completed_proposal(run, proposal)

    def _start_event_run(
        self,
        *,
        source_report: Mapping[str, Any],
        workspace_root: Path,
        iteration: int,
        initial_prompt: str,
        package: WorkspacePackageInfo,
        file_contexts: tuple[WorkspaceFileContext, ...],
        previous_verification: tuple[WorkspaceVerificationResult, ...],
        available_executables: tuple[str, ...] | None,
    ) -> RunRecord | None:
        if self.event_store is None:
            return None
        repo_digest = build_workspace_execution_profile(workspace_root).repo_hash
        report_id = str(source_report.get("report_id") or "workspace_report")
        run = self.event_store.start_run(
            task_id=(
                f"{report_id}:namespace:{self.run_namespace}:iteration:{iteration}"
            ),
            repo_digest=repo_digest,
            spec_hash=stable_hash(
                {
                    "initial_prompt": initial_prompt,
                    "source_report": source_report,
                    "package": package,
                    "file_contexts": file_contexts,
                    "previous_verification": previous_verification,
                }
            ),
            model_snapshot=self.model,
            config_hash=stable_hash(
                {
                    "max_turns": self.max_turns,
                    "max_tool_calls": self.max_tool_calls,
                    "max_patch_bytes": self.max_patch_bytes,
                    "max_exploration_calls_per_revision": (
                        self.max_exploration_calls_per_revision
                    ),
                    "max_request_attempts": self.max_request_attempts,
                    "request_timeout_seconds": self.request_timeout_seconds,
                    "reasoning_effort": self.reasoning_effort,
                    "tool_protocol": self.tool_protocol,
                    "response_transport": self.response_transport,
                    "development_stage": self.development_stage,
                    "run_namespace": self.run_namespace,
                    "run_attempt": self.run_attempt,
                    "executor_policy": canonicalize(self.executor.policy),
                    "available_executables": available_executables,
                    "tool_schema_hash": stable_hash(
                        {
                            "exploration": coding_tool_definitions(
                                available_executables=available_executables
                            ),
                            "implementation_only": coding_tool_definitions(
                                allow_exploration=False,
                                available_executables=available_executables,
                            ),
                            "mutation_only": coding_tool_definitions(
                                allow_exploration=False,
                                allow_verification=False,
                                available_executables=available_executables,
                            ),
                        }
                    ),
                    "runtime_version": TOOL_LOOP_RUNTIME_VERSION,
                }
            ),
            attempt=self.run_attempt,
        )
        self.last_run_id = run.run_id
        return run

    def _record_event(
        self,
        run: RunRecord | None,
        *,
        event_type: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> None:
        if self.event_store is None or run is None:
            return
        self.event_store.append_event(
            run_id=run.run_id,
            event_type=event_type,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    def _persist_completed_proposal(
        self,
        run: RunRecord | None,
        proposal: CodingAgentPatchProposal,
    ) -> None:
        if self.event_store is None or run is None:
            return
        payload = {
            "proposal": canonicalize(proposal),
            "trace": canonicalize(self.last_trace),
        }
        event = self.event_store.append_event(
            run_id=run.run_id,
            event_type="proposal_completed",
            payload=payload,
            idempotency_key="proposal_completed",
        )
        self.event_store.complete_run(
            run.run_id,
            status="proposal_ready",
            result_hash=event.payload_hash,
        )

    def _persist_blocked_proposal(
        self,
        run: RunRecord | None,
        proposal: CodingAgentPatchProposal,
    ) -> None:
        if self.event_store is None or run is None:
            return
        self.event_store.append_terminal_event(
            run_id=run.run_id,
            event_type="proposal_blocked",
            payload={
                "proposal": canonicalize(proposal),
                "trace": canonicalize(self.last_trace),
            },
            idempotency_key="proposal_blocked",
            status="blocked",
        )

    def _replay_persisted_proposal(
        self,
        run_id: str,
    ) -> CodingAgentPatchProposal | None:
        assert self.event_store is not None
        run = self.event_store.get_run(run_id)
        terminal_event_type = {
            "blocked": "proposal_blocked",
            "proposal_ready": "proposal_completed",
        }.get(run.status)
        if terminal_event_type is None:
            raise ValueError(f"unsupported_terminal_run_status:{run.status}")
        events = self.event_store.load_verified_events(
            run_id,
            terminal_event_type=terminal_event_type,
        )
        completed = next(
            (
                event
                for event in reversed(events)
                if event.event_type == terminal_event_type
            ),
            None,
        )
        if completed is None:
            return None
        raw = completed.payload.get("proposal")
        if not isinstance(raw, Mapping):
            return None
        patches = tuple(
            WorkspaceFilePatch(**dict(item))
            for item in raw.get("patches", ())
            if isinstance(item, Mapping)
        )
        return CodingAgentPatchProposal(
            proposal_id=str(raw.get("proposal_id") or "replayed_proposal"),
            source=str(raw.get("source") or self.source),
            model=str(raw["model"]) if raw.get("model") is not None else None,
            iteration=int(raw.get("iteration") or 0),
            artifact_id=(
                str(raw["artifact_id"]) if raw.get("artifact_id") is not None else None
            ),
            themes=tuple(str(item) for item in raw.get("themes", ())),
            patches=patches,
            verification_commands=tuple(
                str(item) for item in raw.get("verification_commands", ())
            ),
            selected_files=tuple(str(item) for item in raw.get("selected_files", ())),
            rationale=str(raw.get("rationale") or ""),
            support_refs=tuple(str(item) for item in raw.get("support_refs", ())),
            raw_response_hash=(
                str(raw["raw_response_hash"])
                if raw.get("raw_response_hash") is not None
                else None
            ),
            blocked_reason=(
                str(raw["blocked_reason"])
                if raw.get("blocked_reason") is not None
                else None
            ),
            stage=str(raw.get("stage") or "promote"),
        )

    def _load_latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        assert self.event_store is not None
        events = self.event_store.load_verified_events(run_id)
        checkpoint = next(
            (event for event in reversed(events) if event.event_type == "checkpoint"),
            None,
        )
        if checkpoint is None:
            return None
        return dict(checkpoint.payload)


def _implementation_worker_payload(
    *,
    source_report: Mapping[str, Any],
    workspace_root: Path,
    package: WorkspacePackageInfo,
    file_contexts: tuple[WorkspaceFileContext, ...],
    previous_verification: tuple[WorkspaceVerificationResult, ...],
    iteration: int,
    development_stage: str,
) -> dict[str, Any]:
    """Project the controller state into a compact implementation-only context."""

    full = _agent_payload(
        source_report=source_report,
        workspace_root=workspace_root,
        package=package,
        file_contexts=file_contexts,
        previous_verification=previous_verification,
        iteration=iteration,
        development_stage=development_stage,
    )
    raw_tasks = full.get("development_task_specs_v2", ())
    tasks = tuple(
        _worker_task_spec(item) for item in raw_tasks if isinstance(item, Mapping)
    )[:8]
    repo = full.get("repo_intelligence_pack", {})
    code_runtime = full.get("code_max_runtime", {})
    environment = (
        code_runtime.get("environment", {}) if isinstance(code_runtime, Mapping) else {}
    )
    runtime_repo = (
        code_runtime.get("repo", {}) if isinstance(code_runtime, Mapping) else {}
    )
    payload: dict[str, Any] = {
        "iteration": full.get("iteration"),
        "development_stage": full.get("development_stage"),
        "workspace_root_name": full.get("workspace_root_name"),
        "package": full.get("package"),
        "source_report": full.get("source_report", {}),
        "required_project_verification_commands": tuple(
            full.get("required_project_verification_commands", ())
        )[:16],
        "selected_files": _worker_selected_files(
            file_contexts,
            task_specs=tasks,
        ),
        "selected_file_excerpt_policy": {
            "max_files": 4,
            "max_excerpt_chars": 3_000,
            "selection": "task_term_windows_with_line_numbers",
            "full_file_available_via_read_file": True,
        },
        "previous_verification_failures": full.get(
            "previous_verification_failures", ()
        ),
        "repair_mode": full.get("repair_mode", False),
        "repair_policy": full.get("repair_policy", {}),
        "privacy_boundary": full.get("privacy_boundary"),
        "patch_policy": full.get("patch_policy", {}),
        "development_task_specs_v2": tasks,
        "spec_maturity_gate": full.get("spec_maturity_gate", {}),
        "repo_intelligence_pack": _worker_repo_summary(repo),
        "repo_impact_context": tuple(runtime_repo.get("impact_context", ()))[:12]
        if isinstance(runtime_repo, Mapping)
        else (),
        "reproducer_plan": full.get("reproducer_plan", {}),
        "verification_environment": _worker_environment_summary(environment),
        "code_landing_runtime_policy": full.get("code_landing_runtime_policy", {}),
        "maintainer_landing_gate": full.get("maintainer_landing_gate", {}),
        "workspace_progress": full.get("workspace_progress", {}),
    }
    if not any(task.get("task_id") for task in tasks):
        payload["public_intent_fallback"] = tuple(
            item
            for item in full.get("development_intent_specs", ())
            if isinstance(item, Mapping)
        )[:4]
    return payload


def _worker_task_spec(item: Mapping[str, Any]) -> dict[str, Any]:
    tuple_fields = {
        "acceptance_oracles",
        "behavior_surface",
        "boundary_conditions",
        "candidate_path_hints",
        "contract_dimensions",
        "non_goals",
        "public_evidence_refs",
        "relevant_symbols_hint",
        "reproduction_recipe",
    }
    allowed = (
        "task_id",
        "source_theme",
        "task_type",
        "public_evidence_refs",
        "user_pain",
        "observed_behavior",
        "expected_behavior",
        "non_goals",
        "reproduction_recipe",
        "acceptance_oracles",
        "candidate_path_hints",
        "relevant_symbols_hint",
        "spec_maturity",
        "risk_level",
        "behavior_surface",
        "contract_dimensions",
        "boundary_conditions",
    )
    return {
        key: (
            tuple(item.get(key, ()))
            if key in tuple_fields and isinstance(item.get(key, ()), (list, tuple))
            else item.get(key)
        )
        for key in allowed
    }


def _worker_selected_files(
    raw_items: Any,
    *,
    task_specs: tuple[Mapping[str, Any], ...] = (),
) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_items, (list, tuple)):
        return ()
    terms = _worker_context_terms(task_specs)
    selected: list[dict[str, Any]] = []
    for raw in raw_items[:4]:
        if isinstance(raw, Mapping):
            item = dict(raw)
        elif isinstance(raw, WorkspaceFileContext):
            item = canonicalize(raw)
        else:
            continue
        excerpt = str(item.get("excerpt") or "")
        compacted = _task_relevant_excerpt(
            excerpt,
            terms=terms,
            max_chars=3_000,
        )
        if compacted != excerpt:
            item["excerpt"] = compacted
            item["truncated"] = True
        selected.append(item)
    return tuple(selected)


def _worker_context_terms(
    task_specs: tuple[Mapping[str, Any], ...],
) -> tuple[str, ...]:
    raw_terms: list[str] = []
    for task in task_specs:
        for field in (
            "relevant_symbols_hint",
            "behavior_surface",
            "candidate_path_hints",
        ):
            value = task.get(field, ())
            if isinstance(value, (list, tuple)):
                raw_terms.extend(str(item) for item in value)
        for field in ("user_pain", "observed_behavior", "expected_behavior"):
            raw_terms.append(str(task.get(field) or ""))
    tokens: list[str] = []
    for raw in raw_terms:
        tokens.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", raw))
        path = Path(raw)
        if path.suffix:
            tokens.append(path.stem)
    common = {
        "behavior",
        "candidate",
        "command",
        "configuration",
        "expected",
        "files",
        "input",
        "observed",
        "public",
        "source",
        "value",
    }
    prioritized = sorted(
        dict.fromkeys(token for token in tokens if token.casefold() not in common),
        key=lambda token: (
            not ("_" in token or token.isupper()),
            -len(token),
            token.casefold(),
        ),
    )
    return tuple(prioritized[:32])


def _task_relevant_excerpt(
    excerpt: str,
    *,
    terms: tuple[str, ...],
    max_chars: int,
) -> str:
    if len(excerpt) <= max_chars:
        return excerpt
    lines = excerpt.splitlines(keepends=True)
    lowered_terms = tuple(term.casefold() for term in terms if term)
    scored_lines: list[tuple[int, int]] = []
    for line_index, line in enumerate(lines):
        lowered_line = line.casefold()
        score = sum(
            len(lowered_terms) - term_index
            for term_index, term in enumerate(lowered_terms)
            if term in lowered_line
        )
        if score:
            scored_lines.append((score, line_index))
    if not scored_lines:
        head_chars = int(max_chars * 0.7)
        return (
            excerpt[:head_chars]
            + "\n\n/* ... middle omitted; use read_file for exact context ... */\n\n"
            + excerpt[-(max_chars - head_chars) :]
        )[:max_chars]
    windows: list[tuple[int, int]] = []
    used_chars = 0
    for _, index in sorted(scored_lines, key=lambda item: (-item[0], item[1])):
        start = max(0, index - 12)
        end = min(len(lines), index + 13)
        header = f"/* lines {start + 1}-{end}; read_file before editing */\n"
        chunk = header + "".join(lines[start:end])
        if any(
            start <= selected_end and end >= selected_start
            for selected_start, selected_end in windows
        ):
            continue
        if windows and used_chars + len(chunk) > max_chars:
            continue
        windows.append((start, end))
        used_chars += len(chunk)
    chunks = [
        f"/* lines {start + 1}-{end}; read_file before editing */\n"
        + "".join(lines[start:end])
        for start, end in sorted(windows)
    ]
    return "\n".join(chunks)[:max_chars]


def _worker_repo_summary(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    candidates = raw.get("candidate_files_by_theme", {})
    candidate_summary = (
        {
            str(theme): tuple(paths)[:8]
            for theme, paths in candidates.items()
            if isinstance(paths, (list, tuple))
        }
        if isinstance(candidates, Mapping)
        else {}
    )
    symbol_index = raw.get("symbol_index", {})
    symbol_summary = (
        {
            str(symbol): tuple(paths)[:6]
            for symbol, paths in list(symbol_index.items())[:12]
            if isinstance(paths, (list, tuple))
        }
        if isinstance(symbol_index, Mapping)
        else {}
    )
    test_map = raw.get("test_map", {})
    test_summary = (
        {
            str(path): tuple(tests)[:8]
            for path, tests in list(test_map.items())[:10]
            if isinstance(tests, (list, tuple))
        }
        if isinstance(test_map, Mapping)
        else {}
    )
    return {
        "repo_hash": raw.get("repo_hash"),
        "file_count": raw.get("file_count"),
        "package_manager": raw.get("package_manager"),
        "test_commands": tuple(raw.get("test_commands", ()))[:12],
        "lint_commands": tuple(raw.get("lint_commands", ()))[:12],
        "entrypoints": tuple(raw.get("entrypoints", ()))[:16],
        "source_roots": tuple(raw.get("source_roots", ()))[:12],
        "test_files": tuple(raw.get("test_files", ()))[:16],
        "config_files": tuple(raw.get("config_files", ()))[:12],
        "candidate_files_by_theme": candidate_summary,
        "symbol_index": symbol_summary,
        "test_map": test_summary,
        "style_notes": tuple(raw.get("style_notes", ()))[:12],
    }


def _worker_environment_summary(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    baseline = raw.get("project_command_baseline", {})
    baseline_results = (
        tuple(baseline.get("results", ()))[:8] if isinstance(baseline, Mapping) else ()
    )
    return {
        "setup_status": raw.get("setup_status"),
        "project_verification_status": raw.get("project_verification_status"),
        "available_verification_executables": tuple(
            raw.get("available_verification_executables", ())
        )[:16],
        "eligible_project_verification_commands": tuple(
            raw.get("eligible_project_verification_commands", ())
        )[:16],
        "discovered_test_commands": tuple(raw.get("discovered_test_commands", ()))[:12],
        "discovered_lint_commands": tuple(raw.get("discovered_lint_commands", ()))[:12],
        "discovered_build_commands": tuple(raw.get("discovered_build_commands", ()))[
            :12
        ],
        "missing_declared_dependencies": tuple(
            raw.get("missing_declared_dependencies", ())
        )[:16],
        "project_command_baseline": {"results": baseline_results},
    }


def _initial_prompt(
    *,
    source_report: Mapping[str, Any],
    workspace_root: Path,
    package: WorkspacePackageInfo,
    file_contexts: tuple[WorkspaceFileContext, ...],
    selected_files: tuple[str, ...],
    previous_verification: tuple[WorkspaceVerificationResult, ...],
    development_stage: str,
    iteration: int,
    max_turns: int,
    max_tool_calls: int,
    max_exploration_calls_per_revision: int,
    available_executables: tuple[str, ...] | None,
) -> str:
    candidate_search = source_report.get("candidate_search", {})
    strategy = (
        str(candidate_search.get("strategy") or "reproducer_first_root_cause")
        if isinstance(candidate_search, Mapping)
        else "reproducer_first_root_cause"
    )
    integration_mode = bool(
        isinstance(candidate_search, Mapping)
        and candidate_search.get("integration_mode")
    )
    repair_mode = bool(
        isinstance(candidate_search, Mapping) and candidate_search.get("repair_capsule")
    )
    hypothesis_instruction = (
        "Use the controller-owned repair capsule as the parent revision contract. "
        "Preserve its already validated production behavior, address every exact "
        "finding, and avoid rediscovering or rewriting unaffected logic."
        if repair_mode
        else "Use the bounded public sibling evidence as competing partial hypotheses. "
        "Re-derive each claimed contract from the repository, synthesize only "
        "complementary behavior, and independently test the integrated result; do not "
        "blindly copy sibling patches."
        if integration_mode
        else (
            "Form an independent implementation hypothesis that follows this strategy "
            "instead of imitating sibling candidates."
        )
    )
    payload = {
        "agent_context": _implementation_worker_payload(
            source_report=source_report,
            workspace_root=workspace_root,
            package=package,
            file_contexts=file_contexts,
            previous_verification=previous_verification,
            iteration=iteration,
            development_stage=development_stage,
        ),
        "candidate_search": candidate_search,
        "selected_file_hints": selected_files,
        "execution_budget": {
            "max_turns": max_turns,
            "max_tool_calls": max_tool_calls,
            "max_exploration_calls_per_revision": (max_exploration_calls_per_revision),
            "policy": (
                "Use focused inspection, then reserve enough calls to edit, run a "
                "latest-revision behavior check, inspect the final diff, and finalize."
            ),
        },
        "available_verification_executables": available_executables,
    }
    python_guidance = (
        "Use run_python_check with raw multiline Python for complex behavioral "
        "probes instead of constructing a quoted python -c command. "
        if available_executables is not None
        and any(
            executable in available_executables
            for executable in ("python", "python3", "python3.12", "python3.13")
        )
        else (
            "Runtime capability discovery was unavailable. Use only declared required "
            "checks and treat missing executables as infrastructure evidence, never as "
            "a reason to revert product code. "
        )
        if available_executables is None
        else (
            "Python is unavailable in this frozen runtime. Do not call "
            "run_python_check or run Python through run_check; express behavioral "
            "probes with an available project runtime. An unavailable runtime is "
            "infrastructure evidence and never a reason to revert product code. "
        )
    )
    return (
        "You are the implementation worker in a production code-landing system. "
        f"CANDIDATE_STRATEGY: {strategy}. {hypothesis_instruction} Inspect the "
        "repository dynamically with list_files, search, and read_file. Apply concrete "
        "source and test changes in the disposable candidate workspace, run focused "
        "tests through the highest public entrypoint. Build a boundary-case matrix from "
        "the typed boundary_conditions entries. For scalar configuration input, keep "
        "omitted, empty, malformed, boundary, and representative valid states distinct, "
        "and trace through adapters that may normalize or drop one of those states. A "
        "single malformed example never proves the whole invalid-input class. Every "
        "confirmed command in reproducer_plan.executable_commands and "
        "required_project_verification_commands is a mandatory current-revision "
        "acceptance check; never substitute an easier proxy. Distinct checks add only a capped "
        "diagnostic-reading grant and never reset the revision's implementation budget, "
        "so convert evidence into an edit instead of cycling through probes. Treat each "
        "boundary entry as a "
        "required contrastive state for the implementation and focused tests; do not "
        "stop after "
        "the first happy-path reproducer passes. Trace the "
        "repo_impact_context before architecture-scale edits: inspect affected imports, "
        "dependents, and mapped tests, then preserve or deliberately update those "
        "contracts. A new helper, class, callback, or module is not an implementation "
        "unless the same candidate also wires it into a reachable changed production "
        "callsite or it is itself the requested public entrypoint. Before verification, "
        "confirm that every newly introduced abstraction is referenced on the affected "
        "runtime path; remove abandoned helpers instead of testing an unwired file. Trace the "
        "affected behavior through public entrypoints and wrappers implicated by the task "
        "and repository; do not invent unrelated surfaces. Run focused "
        "tool batches: decide and execute the next concrete probe, edit, or check from "
        "the evidence already available instead of spending a turn restating the whole "
        "problem. Run focused "
        "probes with assertions or explicit nonzero exits; print-only probes are diagnostic, not verification. "
        "Only change a project test file when you can execute a command that names that "
        "exact test file in the frozen environment. If project test collection is "
        "unavailable because a declared test dependency is missing, keep the production "
        "fix source-only and use the evaluator-owned behavior probes; never add an "
        "unexecuted test merely to satisfy a test-presence heuristic. "
        "Run all required checks on the latest revision, inspect the final diff after "
        "the latest edit, and "
        "repair failures before finishing. Use revert_file when an attempted path is worse. Never "
        "return a roadmap instead of code. Do not access the network or controller "
        "secrets. Do not weaken tests or delete behavior to obtain a pass. Use only the "
        "provided public evidence and repository state. Commands execute as direct argv, "
        "not through a shell; never use shell redirection or heredoc syntax. Never pass "
        "git commands to run_check: use the dedicated git_diff tool once, and after that "
        "diagnose the failed behavior check or edit the candidate instead of repeating "
        "the diff. " + python_guidance + "A missing "
        "declared project dependency is environment/setup evidence, not a product defect; "
        "do not add a fallback unless the public contract says that dependency is optional. "
        "read_file returns a revision hash and, for Python class ranges, an AST-derived "
        "structural_context. Treat its slots, instance attributes, and dominant forwarding "
        "receiver as current-source contracts; do not invent a receiver that conflicts "
        "with them. After an exact fragment mismatch, prefer "
        "replace_lines with that hash instead of retrying stale text. For multiple "
        "non-overlapping edits to the same file revision, use replace_line_ranges once "
        "instead of issuing parallel edits that make each other's hashes stale. When git_diff reports "
        "ready_to_finalize, return the strict JSON immediately and do not call git_diff "
        "again for the same revision. "
        "At completion return the strict "
        "JSON summary; patches are exported from tool state, so do not embed source code "
        "in the final response. Final verification_commands may only repeat commands "
        "that passed run_check or run_python_check on the current revision; unexecuted "
        "entries are ignored.\n\n"
        "INPUT_JSON:\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    )


def _execution_budget_message(
    *,
    turn: int,
    max_turns: int,
    tool_calls_used: int,
    max_tool_calls: int,
    changed_path_count: int,
    finalization_issues: tuple[str, ...] = (),
    exploration_calls_current_revision: int = 0,
    suppressed_tool_names: frozenset[str] = frozenset(),
) -> dict[str, str]:
    remaining_turns = max(0, max_turns - turn - 1)
    remaining_tool_calls = max(0, max_tool_calls - tool_calls_used)
    implementation_threshold = max(8, max_turns // 3)
    if changed_path_count and not finalization_issues:
        phase = "ready_to_finalize"
        required_action = (
            "Return the final JSON now. Do not call git_diff again or invoke another "
            "repository tool unless you first identify a concrete unresolved defect in "
            "the already inspected diff."
        )
    elif changed_path_count:
        phase = "verify_and_finalize"
        required_action = (
            "Resolve only the listed finalization issues, then return the final JSON. "
            "Do not repeat a check or diff that is already current."
        )
    elif (
        exploration_calls_current_revision >= 12
        or remaining_turns <= implementation_threshold
        or remaining_tool_calls <= implementation_threshold
    ):
        phase = "implement_now"
        required_action = (
            "Stop broad exploration. Choose the best-supported implementation, then "
            "reserve tools for a focused check and final diff inspection."
        )
    else:
        phase = "focused_exploration"
        required_action = (
            "Continue only high-signal inspection and move to implementation as soon "
            "as one causal path is supported."
        )
    payload = {
        "changed_path_count": changed_path_count,
        "finalization_issues": finalization_issues,
        "phase": phase,
        "ready_to_finalize": bool(changed_path_count and not finalization_issues),
        "remaining_tool_calls": remaining_tool_calls,
        "remaining_turns": remaining_turns,
        "required_action": required_action,
        "suppressed_redundant_tools": tuple(sorted(suppressed_tool_names)),
    }
    if exploration_calls_current_revision:
        payload["exploration_calls_current_revision"] = (
            exploration_calls_current_revision
        )
    return {
        "role": "user",
        "content": "EXECUTION_BUDGET\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }


def _progress_intervention_message(signal: AgentProgressSignal) -> dict[str, str]:
    return {
        "role": "user",
        "content": "PROGRESS_INTERVENTION\n"
        + json.dumps(
            {
                "reason": signal.reason,
                "count": signal.count,
                "revision": signal.revision,
                "required_action": signal.required_action,
                "instruction": (
                    "The observation budget for this revision is exhausted. State one "
                    "causal hypothesis, then apply a syntax-valid edit with the "
                    "available mutation tools. If the evidence does not support an "
                    "edit, return a blocked result instead of requesting more probes."
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _finalization_repair_guidance(issues: tuple[str, ...]) -> tuple[str, ...]:
    guidance: list[str] = []
    for issue in issues:
        if issue == "effective_patch_required":
            guidance.append("Apply a concrete source or test patch before finalizing.")
        elif issue == "current_revision_check_required":
            guidance.append("Run at least one focused run_check after the latest edit.")
        elif issue == "current_revision_behavior_check_required":
            guidance.append(
                "Run an assertive behavior or reproduction check; syntax-only checks "
                "do not satisfy this requirement."
            )
        elif issue.startswith("current_revision_required_acceptance_check_required:"):
            guidance.append(
                "Run the exact confirmed acceptance command from the reproducer plan "
                "on the current revision and repair the implementation if it fails."
            )
        elif issue.startswith("current_revision_check_failure_unresolved:"):
            guidance.append(
                "A prior check is still failing. Repair the candidate and rerun that "
                "exact normalized command successfully before finalizing."
            )
        elif issue.startswith("current_revision_declaration_check_required:"):
            paths = issue.split(":", 1)[1]
            guidance.append(
                "Run a project type checker or an assertive run_check that explicitly "
                f"names and validates these declaration paths: {paths}."
            )
        elif issue == "final_diff_not_inspected_after_latest_patch":
            guidance.append("Call git_diff after the latest patch before finalizing.")
        else:
            guidance.append(f"Resolve finalization issue: {issue}.")
    return tuple(guidance)


def _compact_conversation(
    conversation: list[Any],
    tools: CodingToolSession,
    *,
    max_chars: int = DEFAULT_TOOL_LOOP_CONVERSATION_MAX_CHARS,
    require_reasoning_items: bool = False,
) -> list[Any]:
    if max_chars < 10_000:
        raise ValueError("conversation_max_chars_too_small")
    normalized = _sanitize_response_item_groups(
        conversation,
        require_reasoning_items=require_reasoning_items,
    )
    if _serialized_chars(normalized) <= max_chars:
        return normalized
    memory = {
        "changed_paths": tuple(getattr(tools, "changed_paths", ()) or ()),
        "passed_check_commands": tuple(
            getattr(tools, "passed_check_commands", ()) or ()
        ),
        "diff_inspected_for_current_revision": bool(
            getattr(tools, "diff_inspected_for_current_revision", False)
        ),
        "recent_tool_calls": tuple(
            _tool_memory_entry(item)
            for item in tuple(getattr(tools, "transcript", ()) or ())[-16:]
        ),
        "revision_outcome_ledger": _revision_outcome_ledger(
            tuple(getattr(tools, "transcript", ()) or ())
        ),
    }
    memory_message = {
        "role": "user",
        "content": "COMPACTED_WORKING_MEMORY\n"
        + json.dumps(memory, sort_keys=True, separators=(",", ":"), default=str),
    }
    first = normalized[0] if normalized else {"role": "user", "content": ""}
    for start in _safe_conversation_tail_starts(normalized):
        candidate = [first, memory_message, *normalized[start:]]
        if _serialized_chars(candidate) < max_chars and not _has_orphaned_tool_outputs(
            candidate
        ):
            return candidate
    return [first, memory_message]


def _sanitize_response_item_groups(
    conversation: list[Any],
    *,
    require_reasoning_items: bool,
) -> list[Any]:
    if not require_reasoning_items:
        return list(conversation)
    retained: list[Any] = []
    invalid_call_ids: set[str] = set()
    function_batch_valid = True
    response_has_reasoning = False
    for index, item in enumerate(conversation):
        item_type = _item_value(item, "type")
        item_role = _item_value(item, "role")
        if item_type == "reasoning":
            response_has_reasoning = True
        elif item_role in {"developer", "system", "user"}:
            response_has_reasoning = False
        if item_type == "function_call":
            previous_type = (
                _item_value(conversation[index - 1], "type") if index else None
            )
            if previous_type != "function_call":
                server_item_id = str(_item_value(item, "id", ""))
                function_batch_valid = not server_item_id or response_has_reasoning
            if function_batch_valid:
                retained.append(item)
            else:
                call_id = str(_item_value(item, "call_id", ""))
                if call_id:
                    invalid_call_ids.add(call_id)
            continue
        if (
            item_type == "function_call_output"
            and str(_item_value(item, "call_id", "")) in invalid_call_ids
        ):
            response_has_reasoning = False
            continue
        if item_type == "function_call_output":
            response_has_reasoning = False
        retained.append(item)
    retained_call_ids = {
        str(_item_value(item, "call_id", ""))
        for item in retained
        if _item_value(item, "type") == "function_call"
    }
    return [
        item
        for item in retained
        if _item_value(item, "type") != "function_call_output"
        or str(_item_value(item, "call_id", "")) in retained_call_ids
    ]


def _safe_conversation_tail_starts(conversation: list[Any]) -> tuple[int, ...]:
    user_boundaries = {
        index
        for index, item in enumerate(conversation[1:], start=1)
        if _item_value(item, "role") == "user"
    }
    function_batch_boundaries: set[int] = set()
    for index, item in enumerate(conversation[1:], start=1):
        if (
            _item_value(item, "type") != "function_call"
            or _item_value(conversation[index - 1], "type") == "function_call"
        ):
            continue
        boundary = index
        while (
            boundary > 1
            and _item_value(conversation[boundary - 1], "type") == "reasoning"
        ):
            boundary -= 1
        function_batch_boundaries.add(boundary)
    return tuple(sorted(user_boundaries | function_batch_boundaries))


def _has_orphaned_tool_outputs(conversation: list[Any]) -> bool:
    call_ids = {
        str(_item_value(item, "call_id", ""))
        for item in conversation
        if _item_value(item, "type") == "function_call"
    }
    output_ids = {
        str(_item_value(item, "call_id", ""))
        for item in conversation
        if _item_value(item, "type") == "function_call_output"
    }
    return not output_ids.issubset(call_ids)


def _tool_memory_entry(item: Any) -> dict[str, Any]:
    arguments = _item_value(item, "arguments", {})
    result = _item_value(item, "result", {})
    if not isinstance(arguments, Mapping):
        arguments = {}
    if not isinstance(result, Mapping):
        result = {}
    command = str(arguments.get("command") or result.get("normalized_command") or "")
    return {
        "index": _item_value(item, "index", 0),
        "name": _item_value(item, "name", "unknown"),
        "path": arguments.get("path") or result.get("path"),
        "query": arguments.get("query"),
        "command_hash": stable_hash(command)[:16] if command else None,
        "status": result.get("status"),
        "reason": result.get("reason"),
        "changed_paths": result.get("changed_paths"),
    }


def _revision_outcome_ledger(
    transcript: tuple[Any, ...],
    *,
    max_revisions: int = MAX_REVISION_OUTCOME_LEDGER_ENTRIES,
) -> tuple[dict[str, Any], ...]:
    if max_revisions <= 0:
        return ()
    revisions: dict[int, dict[str, Any]] = {}
    for item in transcript:
        revision = int(_item_value(item, "revision", 0) or 0)
        if revision <= 0:
            continue
        name = str(_item_value(item, "name", "unknown"))
        arguments = _item_value(item, "arguments", {})
        result = _item_value(item, "result", {})
        if not isinstance(arguments, Mapping):
            arguments = {}
        if not isinstance(result, Mapping):
            result = {}
        entry = revisions.setdefault(
            revision,
            {"revision": revision, "mutations": [], "checks": []},
        )
        if name in _REVISION_MUTATION_TOOL_NAMES:
            entry["mutations"].append(
                {
                    "tool": name,
                    "path": arguments.get("path") or result.get("path"),
                    "rationale": str(arguments.get("rationale") or "")[:240],
                    "status": result.get("status"),
                    "reason": str(result.get("reason") or "")[:240] or None,
                }
            )
            entry["mutations"] = entry["mutations"][-2:]
            continue
        if name not in {"run_check", "run_python_check"}:
            continue
        command = str(
            arguments.get("command") or result.get("normalized_command") or ""
        )
        evidence = str(
            result.get("stderr_tail")
            or result.get("stdout_tail")
            or result.get("reason")
            or ""
        )
        entry["checks"].append(
            {
                "tool": name,
                "command_hash": stable_hash(command)[:16] if command else None,
                "status": result.get("status"),
                "exit_code": result.get("exit_code"),
                "evidence": evidence[-400:],
            }
        )
        entry["checks"] = entry["checks"][-2:]
    ordered = tuple(revisions[index] for index in sorted(revisions))
    return ordered[-max_revisions:]


def _serialized_chars(value: Any) -> int:
    return len(json.dumps(_serialize_item(value), separators=(",", ":"), default=str))


def _final_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "artifact_id": {"type": ["string", "null"]},
            "themes": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "verification_commands": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 24,
            },
            "rationale": {"type": "string"},
            "support_refs": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 24,
            },
            "blocked_reason": {"type": ["string", "null"]},
        },
        "required": [
            "artifact_id",
            "themes",
            "verification_commands",
            "rationale",
            "support_refs",
            "blocked_reason",
        ],
        "additionalProperties": False,
    }


def _configure_request_input(
    request: dict[str, Any],
    *,
    request_input: list[Any],
    event_run: RunRecord | None,
    turn: int,
    model: str,
    active_tool_definitions: list[dict[str, Any]],
) -> None:
    request["input"] = request_input
    request["extra_headers"] = {
        "Idempotency-Key": stable_hash(
            {
                "run_id": event_run.run_id if event_run else None,
                "turn": turn,
                "model": model,
                "input": _serialize_item(request_input),
                "tool_names": tuple(tool["name"] for tool in active_tool_definitions),
            }
        )
    }


def _create_model_response(
    responses: Any,
    request: Mapping[str, Any],
    *,
    transport: str,
) -> tuple[Any, int]:
    if transport == "sync":
        return responses.create(**dict(request)), 0
    if transport != "stream":
        raise ValueError("invalid_model_response_transport")

    stream_event_count = 0
    completed_response = None
    stream = responses.create(**dict(request), stream=True)
    with stream:
        for event in stream:
            stream_event_count += 1
            event_type = _stream_event_value(event, "type")
            if event_type == "response.completed":
                completed_response = _stream_event_value(event, "response")
            elif event_type in {"response.failed", "response.incomplete"}:
                raise ModelStreamTerminalError(
                    str(event_type),
                    _stream_event_value(event, "response"),
                )
    if completed_response is None:
        raise RuntimeError("model_stream_missing_response_completed")
    return completed_response, stream_event_count


def _stream_event_value(event: Any, field_name: str) -> Any:
    if isinstance(event, Mapping):
        return event.get(field_name)
    return getattr(event, field_name, None)


def _stream_terminal_details(response: Any) -> dict[str, str]:
    error = _stream_event_value(response, "error")
    incomplete = _stream_event_value(response, "incomplete_details")
    details: dict[str, str] = {}
    for prefix, value in (("error", error), ("incomplete", incomplete)):
        for field_name in ("code", "type", "reason", "message"):
            field_value = _stream_event_value(value, field_name)
            if field_value is not None:
                details[f"{prefix}_{field_name}"] = str(field_value)[:500]
    status = _stream_event_value(response, "status")
    if status is not None:
        details["status"] = str(status)[:80]
    return details


def _stream_terminal_is_explicit_client_failure(
    details: Mapping[str, str],
) -> bool:
    text = " ".join(str(value).casefold() for value in details.values())
    return any(
        marker in text
        for marker in (
            "content_filter",
            "context_length",
            "invalid_request",
            "policy_violation",
            "safety",
            "unsupported",
        )
    )


def _tool_protocol_request_input(
    conversation: list[Any],
    tools: CodingToolSession,
    *,
    protocol: str,
    max_replay_chars: int = DEFAULT_PORTABLE_REPLAY_MAX_CHARS,
) -> list[Any]:
    if max_replay_chars < MIN_PORTABLE_REPLAY_MAX_CHARS:
        raise ValueError("portable_replay_max_chars_too_small")
    if protocol == "native" or len(conversation) <= 1:
        return list(conversation)
    if protocol != "portable":
        raise ValueError("invalid_active_tool_protocol")

    first = _portable_task_capsule(_serialize_item(conversation[0]))
    records = tuple(
        record
        for item in conversation[1:]
        if (record := _portable_replay_record(item)) is not None
    )
    controller_state = {
        "changed_paths": tuple(getattr(tools, "changed_paths", ()) or ()),
        "passed_check_commands": tuple(
            getattr(tools, "passed_check_commands", ()) or ()
        ),
        "diff_inspected_for_current_revision": bool(
            getattr(tools, "diff_inspected_for_current_revision", False)
        ),
        "recent_tool_calls": tuple(
            _tool_memory_entry(item)
            for item in tuple(getattr(tools, "transcript", ()) or ())[-20:]
        ),
        "revision_outcome_ledger": _revision_outcome_ledger(
            tuple(getattr(tools, "transcript", ()) or ())
        ),
    }
    retained: list[dict[str, Any]] = []
    for record in reversed(records):
        candidate = [record, *retained]
        payload = _portable_replay_payload(controller_state, candidate)
        if len(payload) <= max_replay_chars:
            retained = candidate
    replay = _portable_replay_payload(controller_state, retained)
    return [
        first,
        {
            "role": "user",
            "content": "PORTABLE_TOOL_REPLAY_V1\n" + replay,
        },
    ]


def _portable_task_capsule(first: Any) -> Any:
    if not isinstance(first, Mapping):
        return first
    content = str(first.get("content") or "")
    marker = content.find('{"agent_context":')
    if marker < 0:
        return {
            "role": str(first.get("role") or "user"),
            "content": _bounded_portable_text(content, max_chars=20_000),
        }
    try:
        payload = json.loads(content[marker:])
    except json.JSONDecodeError:
        return {
            "role": str(first.get("role") or "user"),
            "content": _bounded_portable_text(content, max_chars=20_000),
        }
    agent_context = payload.get("agent_context", {})
    if not isinstance(agent_context, Mapping):
        agent_context = {}
    repo = agent_context.get("repo_intelligence_pack", {})
    if not isinstance(repo, Mapping):
        repo = {}
    selected_files = agent_context.get("selected_files", ())
    selected_file_paths = tuple(
        str(item.get("path") or "")
        for item in selected_files
        if isinstance(item, Mapping) and str(item.get("path") or "")
    )
    compact_agent_context = {
        "iteration": agent_context.get("iteration"),
        "development_stage": agent_context.get("development_stage"),
        "package": agent_context.get("package"),
        "source_report": agent_context.get("source_report", {}),
        "required_project_verification_commands": agent_context.get(
            "required_project_verification_commands",
            (),
        ),
        "development_task_specs_v2": agent_context.get("development_task_specs_v2", ()),
        "previous_verification_failures": agent_context.get(
            "previous_verification_failures", ()
        ),
        "repair_mode": agent_context.get("repair_mode", False),
        "repair_policy": agent_context.get("repair_policy", {}),
        "patch_policy": agent_context.get("patch_policy", {}),
        "reproducer_plan": agent_context.get("reproducer_plan", {}),
        "verification_environment": agent_context.get("verification_environment", {}),
        "repo_impact_context": tuple(
            agent_context.get("repo_impact_context", ()) or ()
        )[:8],
        "repo_intelligence_pack": {
            "package_manager": repo.get("package_manager"),
            "test_commands": tuple(repo.get("test_commands", ()) or ())[:8],
            "lint_commands": tuple(repo.get("lint_commands", ()) or ())[:8],
            "entrypoints": tuple(repo.get("entrypoints", ()) or ())[:10],
            "source_roots": tuple(repo.get("source_roots", ()) or ())[:8],
            "candidate_files_by_theme": repo.get("candidate_files_by_theme", {}),
            "test_map": repo.get("test_map", {}),
            "style_notes": tuple(repo.get("style_notes", ()) or ())[:8],
        },
        "selected_file_paths": selected_file_paths,
        "workspace_progress": agent_context.get("workspace_progress", {}),
        "maintainer_landing_gate": agent_context.get("maintainer_landing_gate", {}),
        "dynamic_repository_access": True,
    }
    capsule = {
        "protocol": "portable_task_capsule_v1",
        "instruction": (
            "Continue the same production coding task at full reasoning strength. "
            "The typed task, reproducer, boundary conditions, failure ledger, and "
            "controller state are authoritative. Use repository tools to reread exact "
            "source instead of relying on omitted excerpts. Execute the next concrete "
            "probe, edit, or current-revision check; preserve public contracts and do "
            "not weaken tests."
        ),
        "agent_context": compact_agent_context,
        "candidate_search": payload.get("candidate_search", {}),
        "execution_budget": payload.get("execution_budget", {}),
        "selected_file_hints": tuple(payload.get("selected_file_hints", ()) or ())[:16],
    }
    return {
        "role": str(first.get("role") or "user"),
        "content": "PORTABLE_TASK_CAPSULE_V1\n"
        + json.dumps(capsule, sort_keys=True, separators=(",", ":"), default=str),
    }


def _portable_replay_budget_for_attempt(attempt: int) -> int:
    if attempt <= 0:
        raise ValueError("request_attempt_must_be_positive")
    budgets = (
        DEFAULT_PORTABLE_REPLAY_MAX_CHARS,
        32_000,
        20_000,
        MIN_PORTABLE_REPLAY_MAX_CHARS,
    )
    return budgets[min(attempt - 1, len(budgets) - 1)]


def _portable_replay_payload(
    controller_state: Mapping[str, Any],
    history: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "protocol": "portable_tool_replay_v1",
            "instruction": (
                "Continue the same coding task from this controller-owned tool "
                "history. Treat tool results and controller state as authoritative. "
                "Use the currently supplied tools for the next concrete action; do "
                "not merely restate prior work."
            ),
            "controller_state": controller_state,
            "history": history,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _portable_replay_record(item: Any) -> dict[str, Any] | None:
    item_type = str(_item_value(item, "type", "") or "")
    role = str(_item_value(item, "role", "") or "")
    if item_type == "reasoning":
        return None
    if item_type == "function_call":
        return {
            "kind": "tool_call",
            "name": str(_item_value(item, "name", "")),
            "arguments": _bounded_portable_text(
                str(_item_value(item, "arguments", "{}")),
                max_chars=8_000,
            ),
            "call_id": str(_item_value(item, "call_id", "")),
        }
    if item_type == "function_call_output":
        return {
            "kind": "tool_result",
            "call_id": str(_item_value(item, "call_id", "")),
            "output": _bounded_portable_text(
                str(_item_value(item, "output", "")),
                max_chars=12_000,
            ),
        }
    if role in {"developer", "system", "user", "assistant"}:
        content = _portable_message_text(_item_value(item, "content", ""))
        if not content:
            return None
        return {
            "kind": "message",
            "role": role,
            "content": _bounded_portable_text(content, max_chars=16_000),
        }
    return None


def _portable_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            text = _item_value(item, "text")
            if text is not None:
                parts.append(str(text))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(_serialize_item(content), sort_keys=True, default=str)


def _bounded_portable_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    head = int(max_chars * 0.7)
    return (
        value[:head]
        + "\n... portable replay truncated ...\n"
        + value[-(max_chars - head) :]
    )


def _response_total_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    if isinstance(usage, Mapping):
        return int(usage.get("total_tokens") or 0)
    return int(getattr(usage, "total_tokens", 0) or 0)


def _is_retryable_model_request_error(exc: Exception) -> bool:
    if isinstance(exc, ModelStreamTerminalError):
        return exc.retryable
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 409, 429} or status_code >= 500
    name = type(exc).__name__.casefold()
    detail = f"{name}:{str(exc)}".casefold()
    return any(
        marker in detail
        for marker in (
            "apiconnection",
            "apitimeout",
            "internalserver",
            "ratelimit",
            "serviceunavailable",
            "stream_read_error",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "timed out",
            "upstream timeout",
        )
    )


def _model_request_retry_delay_seconds(exc: Exception, attempt: int) -> float:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_RETRY_SLEEP_SECONDS")
    if raw is not None:
        try:
            default_delay = max(0.0, min(60.0, float(raw)))
        except ValueError:
            default_delay = min(30.0, 1.5 * (2**attempt))
    else:
        default_delay = min(30.0, 1.5 * (2**attempt))

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    retry_after = None
    if headers is not None:
        try:
            retry_after = headers.get("retry-after")
        except (AttributeError, TypeError):
            retry_after = None
    try:
        provider_delay = max(0.0, min(60.0, float(retry_after)))
    except (TypeError, ValueError):
        provider_delay = 0.0
    return max(default_delay, provider_delay)


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _serialize_item(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _serialize_item(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_item(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _serialize_item(model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return {
            str(key): _serialize_item(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _trace(
    *,
    status: str,
    response_ids: tuple[str, ...] = (),
    request_count: int = 0,
    tool_calls: tuple[dict[str, Any], ...] = (),
    total_tokens: int = 0,
    blocked_reason: str | None = None,
) -> ToolLoopTrace:
    payload = {
        "status": status,
        "response_ids": response_ids,
        "request_count": request_count,
        "tool_calls": tool_calls,
        "total_tokens": total_tokens,
        "blocked_reason": blocked_reason,
    }
    return ToolLoopTrace(
        status=status,
        response_ids=response_ids,
        request_count=request_count,
        tool_calls=tool_calls,
        total_tokens=total_tokens,
        blocked_reason=blocked_reason,
        trace_hash=stable_hash(payload),
    )


def _blocked_proposal(
    *,
    source: str,
    model: str,
    iteration: int,
    artifact_id: str | None,
    themes: tuple[str, ...],
    selected_files: tuple[str, ...],
    reason: str,
    raw_response_hash: str | None = None,
) -> CodingAgentPatchProposal:
    return CodingAgentPatchProposal(
        proposal_id=f"interactive_blocked_{stable_hash((iteration, reason))[:24]}",
        source=source,
        model=model,
        iteration=iteration,
        artifact_id=artifact_id,
        themes=themes,
        patches=(),
        verification_commands=(),
        selected_files=selected_files,
        rationale="Interactive coding loop could not produce a promotable candidate.",
        support_refs=(),
        raw_response_hash=raw_response_hash,
        blocked_reason=reason,
    )


def _artifact_id(source_report: Mapping[str, Any]) -> str | None:
    proposal = source_report.get("proposal")
    if isinstance(proposal, Mapping) and proposal.get("artifact_id"):
        return str(proposal["artifact_id"])
    if source_report.get("artifact_id"):
        return str(source_report["artifact_id"])
    return None


def _themes(source_report: Mapping[str, Any]) -> tuple[str, ...]:
    value = source_report.get("proposed_themes")
    if isinstance(value, (list, tuple)):
        return tuple(str(item)[:80] for item in value if str(item).strip())[:12]
    return ()
