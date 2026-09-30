"""Semantic progress detection for bounded repository-agent trajectories."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .coding_tools import is_behavior_check_command
from .hashing import stable_hash


DEFAULT_PROGRESS_SOFT_EXPLORATION_LIMIT = 12
DEFAULT_REPEATED_ACTION_OBSERVATION_LIMIT = 4
DEFAULT_DIAGNOSTIC_CHURN_LIMIT = 8
DEFAULT_BEHAVIOR_PROBE_LIMIT = 6
DEFAULT_POST_FAILURE_DIAGNOSTIC_LIMIT = 3
_REPETITION_SENSITIVE_TOOLS = frozenset(
    {"git_diff", "list_files", "read_file", "run_check", "search"}
)
_SUPPRESSIBLE_REDUNDANT_TOOLS = frozenset(
    {"git_diff", "list_files", "run_check", "run_python_check", "search"}
)
_MUTATION_TOOLS = frozenset(
    {"apply_patch", "replace_lines", "replace_line_ranges", "revert_file"}
)
_POST_FAILURE_DIAGNOSTIC_TOOLS = frozenset(
    {
        "git_diff",
        "list_files",
        "read_file",
        "run_check",
        "run_python_check",
        "search",
    }
)
_VOLATILE_RESULT_FIELDS = frozenset(
    {
        "after_hash",
        "attempt",
        "before_hash",
        "current_file_hash",
        "elapsed_sec",
        "file_hash",
        "next_action",
        "repeated_same_revision",
    }
)


@dataclass(frozen=True)
class AgentProgressSignal:
    reason: str
    count: int
    revision: int
    signature: str
    required_action: str


def post_failure_diagnostic_count(
    transcript: tuple[Any, ...],
    *,
    current_revision: int,
) -> int | None:
    """Count diagnostic actions after the latest failed check in this revision."""

    current = tuple(
        item
        for item in transcript
        if int(_item_value(item, "revision", 0) or 0) == current_revision
    )
    latest_failure_index: int | None = None
    for index, item in enumerate(current):
        result = _mapping_value(item, "result")
        if (
            str(_item_value(item, "name", "")) in {"run_check", "run_python_check"}
            and result.get("status") == "failed"
            and result.get("finalization_blocking") is not False
            and result.get("check_role")
            in {None, "", "behavior_probe", "required_acceptance"}
        ):
            latest_failure_index = index
    if latest_failure_index is None:
        return None
    signatures = {
        signature
        for item in current[latest_failure_index + 1 :]
        if (signature := _diagnostic_evidence_signature(item)) is not None
    }
    return len(signatures)


def detect_post_failure_diagnostic_saturation(
    transcript: tuple[Any, ...],
    *,
    current_revision: int,
    diagnostic_limit: int = DEFAULT_POST_FAILURE_DIAGNOSTIC_LIMIT,
) -> AgentProgressSignal | None:
    """Require a repair after a bounded diagnosis window following a failed check."""

    if diagnostic_limit <= 0:
        raise ValueError("post_failure_diagnostic_limit_must_be_positive")
    count = post_failure_diagnostic_count(
        transcript,
        current_revision=current_revision,
    )
    if count is None or count < diagnostic_limit:
        return None
    return AgentProgressSignal(
        reason="post_failure_diagnostic_budget_exhausted",
        count=count,
        revision=current_revision,
        signature=stable_hash(
            {
                "reason": "post_failure_diagnostic_budget_exhausted",
                "revision": current_revision,
            }
        ),
        required_action="repair_or_revert_the_current_revision",
    )


def detect_agent_progress_stall(
    transcript: tuple[Any, ...],
    *,
    current_revision: int,
    exploration_calls_current_revision: int,
    soft_exploration_limit: int = DEFAULT_PROGRESS_SOFT_EXPLORATION_LIMIT,
    repeated_action_limit: int = DEFAULT_REPEATED_ACTION_OBSERVATION_LIMIT,
    diagnostic_churn_limit: int = DEFAULT_DIAGNOSTIC_CHURN_LIMIT,
    behavior_probe_limit: int = DEFAULT_BEHAVIOR_PROBE_LIMIT,
) -> AgentProgressSignal | None:
    """Return one deterministic intervention signal for the current revision."""

    if soft_exploration_limit <= 0:
        raise ValueError("soft_exploration_limit_must_be_positive")
    if repeated_action_limit <= 1:
        raise ValueError("repeated_action_limit_must_exceed_one")
    if diagnostic_churn_limit <= 1:
        raise ValueError("diagnostic_churn_limit_must_exceed_one")
    if behavior_probe_limit <= 1:
        raise ValueError("behavior_probe_limit_must_exceed_one")

    current = tuple(
        item
        for item in transcript
        if int(_item_value(item, "revision", 0) or 0) == current_revision
    )
    if current:
        latest = current[-1]
        latest_name = str(_item_value(latest, "name", ""))
        latest_result = _mapping_value(latest, "result")
        if (
            latest_name in {"run_check", "run_python_check"}
            and latest_result.get("status") == "blocked"
            and latest_result.get("reason")
            == "failed_check_retry_limit_current_revision"
        ):
            failed_attempts = max(
                2,
                int(latest_result.get("failed_attempts") or 0),
            )
            return AgentProgressSignal(
                reason="failed_check_retry_exhausted",
                count=failed_attempts,
                revision=current_revision,
                signature=stable_hash(
                    {
                        "reason": "failed_check_retry_exhausted",
                        "revision": current_revision,
                        "command": (
                            latest_result.get("normalized_command")
                            or _mapping_value(latest, "arguments").get("command")
                            or _mapping_value(latest, "arguments").get("script")
                            or ""
                        ),
                    }
                ),
                required_action=(
                    "change_or_revert_the_candidate_before_rerunning_this_check"
                ),
            )
        if latest_name in _REPETITION_SENSITIVE_TOOLS:
            signature = _action_observation_signature(latest)
            count = sum(
                1
                for item in current
                if _action_observation_signature(item) == signature
            )
            if count >= repeated_action_limit:
                return AgentProgressSignal(
                    reason="repeated_action_observation",
                    count=count,
                    revision=current_revision,
                    signature=signature,
                    required_action=(
                        "state_one_causal_hypothesis_then_edit_or_run_a_distinct_probe"
                    ),
                )

    behavior_probe_count = sum(
        1
        for item in current
        if str(_item_value(item, "name", "")) in {"run_check", "run_python_check"}
        and _mapping_value(item, "result").get("status") in {"passed", "failed"}
        and _mapping_value(item, "result").get("finalization_blocking") is not False
        and _is_behavior_check(item)
    )
    if behavior_probe_count >= behavior_probe_limit:
        signature = stable_hash(
            {
                "reason": "behavior_probe_saturation",
                "revision": current_revision,
                "window": behavior_probe_count // behavior_probe_limit,
            }
        )
        return AgentProgressSignal(
            reason="behavior_probe_saturation",
            count=behavior_probe_count,
            revision=current_revision,
            signature=signature,
            required_action="apply_the_supported_causal_edit_or_return_blocked",
        )

    diagnostic_count = 0
    for item in current:
        name = str(_item_value(item, "name", ""))
        result = _mapping_value(item, "result")
        if name in _MUTATION_TOOLS and result.get("status") in {
            "applied",
            "reverted",
        }:
            diagnostic_count = 0
            continue
        if name not in {"run_check", "run_python_check"} or result.get(
            "status"
        ) not in {"passed", "failed"}:
            continue
        if _is_behavior_check(item):
            diagnostic_count = 0
        else:
            diagnostic_count += 1
    if diagnostic_count >= diagnostic_churn_limit:
        signature = stable_hash(
            {
                "reason": "diagnostic_churn_without_implementation",
                "revision": current_revision,
                "window": diagnostic_count // diagnostic_churn_limit,
            }
        )
        return AgentProgressSignal(
            reason="diagnostic_churn_without_implementation",
            count=diagnostic_count,
            revision=current_revision,
            signature=signature,
            required_action=("apply_a_causal_edit_or_run_the_named_acceptance_oracle"),
        )

    if exploration_calls_current_revision >= soft_exploration_limit:
        signature = stable_hash(
            {
                "reason": "exploration_without_implementation",
                "revision": current_revision,
            }
        )
        return AgentProgressSignal(
            reason="exploration_without_implementation",
            count=exploration_calls_current_revision,
            revision=current_revision,
            signature=signature,
            required_action=(
                "state_one_causal_hypothesis_then_edit_or_run_a_distinct_probe"
            ),
        )
    return None


def suppressed_progress_tool_names(
    transcript: tuple[Any, ...],
    *,
    current_revision: int,
    repeated_action_limit: int = DEFAULT_REPEATED_ACTION_OBSERVATION_LIMIT,
) -> frozenset[str]:
    """Return tools whose repeated observations add no information this revision."""

    if repeated_action_limit <= 1:
        raise ValueError("repeated_action_limit_must_exceed_one")
    counts: dict[tuple[str, str], int] = {}
    exhausted_check_tools: set[str] = set()
    for item in transcript:
        if int(_item_value(item, "revision", 0) or 0) != current_revision:
            continue
        name = str(_item_value(item, "name", ""))
        result = _mapping_value(item, "result")
        if (
            name in {"run_check", "run_python_check"}
            and result.get("status") == "blocked"
            and result.get("reason") == "failed_check_retry_limit_current_revision"
        ):
            exhausted_check_tools.add(name)
        if name not in _SUPPRESSIBLE_REDUNDANT_TOOLS:
            continue
        key = (name, _action_observation_signature(item))
        counts[key] = counts.get(key, 0) + 1
    return frozenset(
        {
            *exhausted_check_tools,
            *(
                name
                for (name, _), count in counts.items()
                if count >= repeated_action_limit
            ),
        }
    )


def _action_observation_signature(item: Any) -> str:
    name = str(_item_value(item, "name", ""))
    result = _mapping_value(item, "result")
    stable_result = {
        str(key): _compact_value(value)
        for key, value in result.items()
        if str(key) not in _VOLATILE_RESULT_FIELDS
        and not (
            name == "read_file"
            and bool(result.get("cached"))
            and str(key)
            in {
                "covered_ranges",
                "end_line",
                "file_hash",
                "start_line",
            }
        )
    }
    return stable_hash(
        {
            "action": _semantic_action_signature(name, item),
            "result": stable_result,
        }
    )


def _semantic_action_signature(name: str, item: Any) -> str:
    arguments = _mapping_value(item, "arguments")
    if name == "search":
        return stable_hash(
            {
                "name": name,
                "query": str(arguments.get("query") or ""),
                "glob": str(arguments.get("glob") or ""),
            }
        )
    if name == "list_files":
        return stable_hash(
            {
                "name": name,
                "glob": str(arguments.get("glob") or ""),
            }
        )
    if name == "read_file" and bool(_mapping_value(item, "result").get("cached")):
        return stable_hash(
            {
                "name": name,
                "path": Path(str(arguments.get("path") or "")).as_posix(),
            }
        )
    return _action_signature(item)


def _is_behavior_check(item: Any) -> bool:
    name = str(_item_value(item, "name", ""))
    arguments = _mapping_value(item, "arguments")
    if name == "run_python_check":
        candidate = str(arguments.get("script") or "")
    elif name == "run_check":
        candidate = str(arguments.get("command") or "")
    else:
        return False
    return is_behavior_check_command(candidate)


def _diagnostic_evidence_signature(item: Any) -> str | None:
    name = str(_item_value(item, "name", ""))
    if name not in _POST_FAILURE_DIAGNOSTIC_TOOLS:
        return None
    result = _mapping_value(item, "result")
    status = str(result.get("status") or "")
    if name in {"run_check", "run_python_check"}:
        if status not in {"passed", "failed"}:
            return None
    elif name == "read_file":
        if status != "ok" or bool(result.get("cached")):
            return None
    elif name == "git_diff":
        if status != "ok" or bool(result.get("repeated_same_revision")):
            return None
        if not result.get("diff") and not result.get("changed_paths"):
            return None
    elif status != "ok":
        return None
    return _action_observation_signature(item)


def _action_signature(item: Any) -> str:
    name = str(_item_value(item, "name", ""))
    arguments = _mapping_value(item, "arguments")
    normalized_arguments = {
        str(key): _compact_value(value) for key, value in arguments.items()
    }
    command = normalized_arguments.get("command")
    if isinstance(command, str):
        try:
            normalized_arguments["command"] = shlex.join(shlex.split(command))
        except ValueError:
            pass
    path = normalized_arguments.get("path")
    if isinstance(path, str):
        normalized_arguments["path"] = Path(path).as_posix()
    return stable_hash({"name": name, "arguments": normalized_arguments})


def _compact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _compact_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        compacted = tuple(_compact_value(item) for item in value[:20])
        return {
            "items": compacted,
            "truncated": len(value) > 20,
            "total_items": len(value),
        }
    if isinstance(value, str) and len(value) > 2_000:
        return {
            "prefix": value[:1_000],
            "suffix": value[-1_000:],
            "total_chars": len(value),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _mapping_value(item: Any, name: str) -> dict[str, Any]:
    value = _item_value(item, name, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _item_value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)
