"""Bounded repository tools for an interactive coding-agent loop."""

from __future__ import annotations

import ast
import difflib
import fnmatch
import json
import os
import re
import shlex
import sys
import tomllib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .execution import CommandExecutor
from .hashing import stable_hash
from .repository_paths import ignored_repository_path
from .safe_files import UnsafeRegularFileError, read_regular_file_text
from .workspace_transaction import CandidateWorkspace
from .verification_policy import VERIFICATION_EXECUTABLES
from .workspace_update import WorkspaceFilePatch, run_workspace_verification


_ALLOWED_CHECK_EXECUTABLES = VERIFICATION_EXECUTABLES
_PYTHON_EXECUTABLES = ("python", "python3", "python3.12", "python3.13")
_EXPLORATION_TOOL_NAMES = frozenset({"list_files", "search", "read_file"})
_VERIFICATION_TOOL_NAMES = frozenset({"run_check", "run_python_check"})
_REVISION_MUTATION_TOOL_NAMES = frozenset(
    {"apply_patch", "replace_lines", "replace_line_ranges", "revert_file"}
)
_FINALIZATION_REPLAY_TOOL_NAMES = frozenset(
    {"run_check", "run_python_check", "git_diff"}
)
_NO_EFFECT_MUTATION_STATUSES = frozenset({"blocked", "no_change", "no_op"})
_NONBLOCKING_PROBE_SETUP_EXCEPTIONS = frozenset(
    {
        "AttributeError",
        "ImportError",
        "IndentationError",
        "ModuleNotFoundError",
        "NameError",
        "SyntaxError",
        "UnboundLocalError",
    }
)
_RECOVERABLE_MUTATION_REASON_PREFIXES = (
    "expected_file_hash_required",
    "invalid_line_range:",
    "replace_fragment_not_found:",
    "stale_file_revision:",
)
_SOURCE_FILE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".swift",
        ".ts",
        ".tsx",
    }
)
_TEST_PATH_PARTS = frozenset({"fixture", "fixtures", "spec", "specs", "test", "tests"})
MAX_UNWIRED_HELPER_BEHAVIOR_FAILURES = 2
DEFAULT_MAX_EXPLORATION_CALLS_PER_REVISION = 16
MAX_UNVERIFIED_REPAIR_REVISIONS = 2
MAX_MUTATION_RECOVERY_RELOCATIONS = 2
MIN_MUTATION_RECOVERY_RELOCATION_MISMATCHES = 2
DIAGNOSTIC_EXPLORATION_CALLS_PER_DISTINCT_CHECK = 4
MAX_DIAGNOSTIC_EXPLORATION_GRANT_PER_REVISION = 8
MAX_PYTHON_CHECK_SCRIPT_CHARS = 100_000
_JAVASCRIPT_ASSERT_BINDING_PATTERNS = (
    re.compile(
        r"\b(?:const|let|var)\s+([a-z_$][a-z0-9_$]*)\s*=\s*"
        r"require\(\s*['\"](?:node:)?assert(?:/strict)?['\"]\s*\)"
    ),
    re.compile(
        r"\bimport\s+(?:\*\s+as\s+)?([a-z_$][a-z0-9_$]*)\s+from\s+"
        r"['\"](?:node:)?assert(?:/strict)?['\"]"
    ),
)
_DIRECT_JAVASCRIPT_ASSERT_CALL_RE = re.compile(
    r"require\(\s*['\"](?:node:)?assert(?:/strict)?['\"]\s*\)\s*"
    r"(?:\(|\.\s*[a-z_$][a-z0-9_$]*\s*\()"
)


def _normalize_apply_patch_arguments(
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = {str(key): value for key, value in arguments.items()}
    operation = str(normalized.get("operation") or "")
    if (
        operation == "replace_fragment"
        and normalized.get("new") is None
        and normalized.get("content") is not None
    ):
        normalized["new"] = normalized["content"]
        normalized["content"] = None
    elif (
        operation in {"create_or_replace", "append_if_missing"}
        and normalized.get("content") is None
        and normalized.get("new") is not None
    ):
        normalized["content"] = normalized["new"]
        normalized["new"] = None
    return normalized


def _closest_fragment_context(
    current: str,
    expected_fragment: str,
    *,
    max_lines: int = 48,
    max_chars: int = 12_000,
) -> dict[str, Any]:
    lines = current.splitlines(keepends=True)
    if not lines or not expected_fragment.strip():
        return {}
    expected_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", expected_fragment))
    anchor = next(
        (line.strip() for line in expected_fragment.splitlines() if line.strip()),
        expected_fragment.strip(),
    )
    best_index = 0
    best_score: tuple[int, int, float] = (-1, -1, -1.0)
    for index, line in enumerate(lines):
        line_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", line))
        shared = expected_tokens & line_tokens
        score = (
            len(shared),
            sum(len(token) for token in shared),
            difflib.SequenceMatcher(
                None,
                anchor.casefold(),
                line.strip().casefold(),
                autojunk=False,
            ).ratio(),
        )
        if score > best_score:
            best_score = score
            best_index = index
    expected_line_count = max(1, len(expected_fragment.splitlines()))
    span = min(max_lines, max(24, expected_line_count + 20))
    start = max(0, best_index - span // 3)
    end = min(len(lines), start + span)
    start = max(0, end - span)
    context = "".join(lines[start:end])
    if len(context) > max_chars:
        context = context[:max_chars]
    return {
        "closest_context": context,
        "closest_context_start_line": start + 1,
        "closest_context_end_line": end,
        "current_file_hash": stable_hash(current),
        "shared_identifier_count": max(0, best_score[0]),
    }


@dataclass(frozen=True)
class ToolInvocation:
    index: int
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    revision: int = 0


@dataclass(frozen=True)
class _FileRevisionSnapshot:
    applied_revision: int
    existed: bool
    content: str
    rationale: str | None


def _mutation_recovery_requirement(
    invocation: ToolInvocation,
) -> dict[str, Any] | None:
    result = dict(invocation.result or {})
    status = str(result.get("status") or "")
    if status not in _NO_EFFECT_MUTATION_STATUSES:
        return None
    path = str(result.get("path") or invocation.arguments.get("path") or "")
    if not path:
        return None
    reason = str(result.get("reason") or status)
    if reason.startswith(_RECOVERABLE_MUTATION_REASON_PREFIXES):
        requirement: dict[str, Any] = {
            "path": path,
            "reason": reason,
            "source_tool": invocation.name,
        }
        context_start = result.get("closest_context_start_line")
        context_end = result.get("closest_context_end_line")
        if isinstance(context_start, int) and isinstance(context_end, int):
            if context_start > 0 and context_end >= context_start:
                requirement["suggested_start_line"] = context_start
                requirement["suggested_end_line"] = context_end
        return requirement
    return None


def _mutation_recovery_observed_ranges(
    lease: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    raw_ranges = lease.get("observed_ranges")
    if isinstance(raw_ranges, (list, tuple)):
        for raw_range in raw_ranges:
            if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                continue
            try:
                start, end = (int(raw_range[0]), int(raw_range[1]))
            except (TypeError, ValueError):
                continue
            if start > 0 and end >= start and (start, end) not in ranges:
                ranges.append((start, end))
    if not ranges:
        try:
            start = int(lease["observed_start_line"])
            end = int(lease["observed_end_line"])
        except (KeyError, TypeError, ValueError):
            return ()
        if start > 0 and end >= start:
            ranges.append((start, end))
    return tuple(ranges)


def _is_test_like_path(path: str) -> bool:
    parts = tuple(part.casefold() for part in Path(path).parts)
    stem = Path(path).stem.casefold()
    return bool(
        _TEST_PATH_PARTS.intersection(parts)
        or stem.startswith("test_")
        or stem.endswith("_test")
        or stem.endswith(".spec")
        or stem.endswith(".test")
    )


def _is_production_source_path(path: str) -> bool:
    return Path(
        path
    ).suffix.casefold() in _SOURCE_FILE_SUFFIXES and not _is_test_like_path(path)


def _call_path_reference_tokens(path: str) -> tuple[str, ...]:
    source = Path(path)
    stem = source.stem
    without_suffix = source.with_suffix("").as_posix()
    variants = {
        source.name,
        stem,
        without_suffix,
        without_suffix.replace("/", "."),
    }
    parts = list(source.with_suffix("").parts)
    while parts and parts[0].casefold() in {"app", "lib", "package", "src"}:
        parts.pop(0)
    if parts:
        relative = "/".join(parts)
        variants.add(relative)
        variants.add(relative.replace("/", "."))
        if len(parts) > 1:
            variants.add(f".{parts[-1]}")
    return tuple(
        sorted(
            (item for item in variants if len(item) >= 4),
            key=lambda item: (-len(item), item),
        )
    )


def _content_references_any_token(
    content: str,
    tokens: tuple[str, ...],
) -> bool:
    return any(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
            content,
        )
        is not None
        for token in tokens
    )


class CodingToolSession:
    """Tool surface backed by a disposable candidate workspace."""

    def __init__(
        self,
        *,
        candidate: CandidateWorkspace,
        executor: CommandExecutor,
        max_tool_calls: int,
        max_patch_bytes: int,
        max_read_lines: int,
        max_search_results: int,
        max_check_timeout_seconds: float,
        max_exploration_calls_per_revision: int,
        required_check_commands: tuple[str, ...],
        available_executables: tuple[str, ...] | None,
    ) -> None:
        self._candidate = candidate
        self.root = candidate.root.resolve()
        self.executor = executor
        self.max_tool_calls = max_tool_calls
        self.max_patch_bytes = max_patch_bytes
        self.max_read_lines = max_read_lines
        self.max_search_results = max_search_results
        self.max_check_timeout_seconds = max_check_timeout_seconds
        self.max_exploration_calls_per_revision = max_exploration_calls_per_revision
        self._required_check_commands = tuple(
            dict.fromkeys(
                _normalized_check_command(command)
                for command in required_check_commands
                if command.strip()
            )
        )
        self.capabilities_observed = available_executables is not None
        self.available_executables = (
            tuple(dict.fromkeys(available_executables))
            if available_executables is not None
            else _ALLOWED_CHECK_EXECUTABLES
        )
        self._tool_call_count = 0
        self._finalization_replay_depth = 0
        self._patch_bytes = 0
        self._patch_sequence = 0
        self._revision = 0
        self._last_diff_revision: int | None = None
        self._rationales: dict[str, str] = {}
        self._passed_check_commands: list[str] = []
        self._last_check_status: str | None = None
        self._transcript: list[ToolInvocation] = []
        self._infrastructure_failure_reason: str | None = None
        self._exploration_calls_current_revision = 0
        self._exploration_limit_current_revision = (
            self.max_exploration_calls_per_revision
        )
        self._executed_check_commands_current_revision: set[str] = set()
        self._failed_checks_current_revision: dict[str, dict[str, Any]] = {}
        self._withdrawn_created_paths: set[str] = set()
        self._read_coverage_current_revision: dict[str, list[tuple[int, int]]] = {}
        self._file_revision_history: dict[str, list[_FileRevisionSnapshot]] = {}
        self._mutation_recovery_edit_lease: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        workspace_root: Path,
        *,
        executor: CommandExecutor,
        max_tool_calls: int = 64,
        max_patch_bytes: int = 500_000,
        max_read_lines: int = 800,
        max_search_results: int = 80,
        max_check_timeout_seconds: float = 600.0,
        max_exploration_calls_per_revision: int = (
            DEFAULT_MAX_EXPLORATION_CALLS_PER_REVISION
        ),
        required_check_commands: tuple[str, ...] = (),
        available_executables: tuple[str, ...] | None = None,
    ) -> CodingToolSession:
        if max_tool_calls <= 0:
            raise ValueError("max_tool_calls_must_be_positive")
        if max_patch_bytes <= 0:
            raise ValueError("max_patch_bytes_must_be_positive")
        if max_check_timeout_seconds <= 0:
            raise ValueError("max_check_timeout_seconds_must_be_positive")
        if max_exploration_calls_per_revision <= 0:
            raise ValueError("max_exploration_calls_per_revision_must_be_positive")
        return cls(
            candidate=CandidateWorkspace.create(workspace_root),
            executor=executor,
            max_tool_calls=max_tool_calls,
            max_patch_bytes=max_patch_bytes,
            max_read_lines=max_read_lines,
            max_search_results=max_search_results,
            max_check_timeout_seconds=max_check_timeout_seconds,
            max_exploration_calls_per_revision=(max_exploration_calls_per_revision),
            required_check_commands=required_check_commands,
            available_executables=available_executables,
        )

    @property
    def transcript(self) -> tuple[ToolInvocation, ...]:
        return tuple(self._transcript)

    @property
    def tool_call_count(self) -> int:
        return self._tool_call_count

    @property
    def patch_bytes(self) -> int:
        return self._patch_bytes

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return self._candidate.changed_paths()

    @property
    def passed_check_commands(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._passed_check_commands))

    def _referenced_created_paths(self, command: str) -> tuple[str, ...]:
        """Candidate-created paths a check command textually depends on."""

        created: dict[str, None] = {}
        for path in self._candidate.changed_paths():
            canonical = self._candidate.canonical_root / path
            if canonical.exists() or canonical.is_symlink():
                continue
            created[path] = None
        for path in sorted(self._withdrawn_created_paths):
            created.setdefault(path, None)
        return tuple(
            path
            for path in created
            if _content_references_any_token(
                command,
                _call_path_reference_tokens(path),
            )
        )

    def _failed_check_is_active(self, details: Mapping[str, Any]) -> bool:
        """A failure is dormant while a created file it depends on is withdrawn.

        The unresolved ledger survives revisions so a failure can only be
        resolved by the exact command passing again. But when the candidate
        withdraws a file it created (revert_file deletes it), a command that
        depended on that file can never pass again, which would strand every
        future candidate behind an unsatisfiable gate. Such an entry goes
        dormant while the file is absent and re-arms if the file is
        recreated, so the exact-replay obligation cannot be laundered.
        """

        for path in details.get("referenced_created_paths") or ():
            if not (self.root / str(path)).exists():
                return False
        return True

    def _active_failed_checks(self) -> dict[str, dict[str, Any]]:
        return {
            command: details
            for command, details in self._failed_checks_current_revision.items()
            if self._failed_check_is_active(details)
        }

    @property
    def unresolved_check_commands(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_failed_checks()))

    @property
    def unresolved_check_replays(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "command": command,
                "failed_revision": int(details.get("revision") or 0),
                "check_role": str(details.get("check_role") or ""),
            }
            for command, details in sorted(self._active_failed_checks().items())
        )

    @property
    def diff_inspected_for_current_revision(self) -> bool:
        return self._last_diff_revision == self._revision

    @property
    def infrastructure_failure_reason(self) -> str | None:
        return self._infrastructure_failure_reason

    @property
    def exploration_calls_current_revision(self) -> int:
        return self._exploration_calls_current_revision

    @property
    def unverified_repair_revision_count(self) -> int:
        active_failures = self._active_failed_checks()
        if not active_failures:
            return 0
        latest_failure_revision = max(
            int(item.get("revision") or 0) for item in active_failures.values()
        )
        return max(0, self._revision - latest_failure_revision)

    @property
    def mutation_recovery_observation_available(self) -> bool:
        """Allow a targeted fresh read after a recoverable no-effect mutation."""

        return self.mutation_recovery_requirement is not None

    @property
    def mutation_recovery_edit_lease(self) -> dict[str, Any] | None:
        """Return the exact file observation that constrains a recovery edit."""

        if self._mutation_recovery_edit_lease is None:
            return None
        return dict(self._mutation_recovery_edit_lease)

    @property
    def mutation_recovery_relocation_requirement(self) -> dict[str, Any] | None:
        """Allow bounded non-overlapping observations after a misplaced lease."""

        lease = self._mutation_recovery_edit_lease
        if lease is None or int(lease.get("relocation_count") or 0) >= (
            MAX_MUTATION_RECOVERY_RELOCATIONS
        ):
            return None
        mismatch_count = 0
        for invocation in reversed(self._transcript):
            if invocation.revision != self._revision:
                continue
            result = dict(invocation.result or {})
            if (
                invocation.name == "read_file"
                and result.get("status") == "ok"
                and result.get("recovery_refresh") is True
            ):
                return None
            if invocation.name not in _REVISION_MUTATION_TOOL_NAMES:
                continue
            reason = str(result.get("reason") or "")
            if result.get("status") == "blocked" and reason.startswith(
                "expected_old_not_found_in_recovery_lease:"
            ):
                mismatch_count += 1
                if mismatch_count < MIN_MUTATION_RECOVERY_RELOCATION_MISMATCHES:
                    continue
                return {
                    **dict(lease),
                    "reason": reason,
                    "source_tool": invocation.name,
                    "mismatch_count": mismatch_count,
                }
            return None
        return None

    @property
    def mutation_recovery_requirement(self) -> dict[str, Any] | None:
        if self._mutation_recovery_edit_lease is not None:
            return None
        pending: dict[str, Any] | None = None
        for invocation in self._transcript:
            if invocation.revision != self._revision:
                continue
            result = dict(invocation.result or {})
            if invocation.name in _REVISION_MUTATION_TOOL_NAMES:
                pending = _mutation_recovery_requirement(invocation)
                continue
            if (
                pending is not None
                and invocation.name == "read_file"
                and result.get("status") == "ok"
                and result.get("recovery_refresh") is True
                and str(result.get("path") or invocation.arguments.get("path") or "")
                == pending["path"]
            ):
                pending = None
        return pending

    @property
    def unwired_created_source_paths(self) -> tuple[str, ...]:
        changed_paths = self._candidate.changed_paths()
        changed_contents: dict[str, str] = {}
        for path in changed_paths:
            if _is_test_like_path(path):
                continue
            try:
                changed_contents[path] = read_regular_file_text(
                    self.root,
                    self.root / path,
                    max_bytes=4_000_000,
                )
            except (UnicodeDecodeError, UnsafeRegularFileError):
                continue
        check_text = "\n".join(
            (
                *self._passed_check_commands,
                *self._failed_checks_current_revision,
            )
        )
        unwired: list[str] = []
        for path in changed_paths:
            history = self._file_revision_history.get(path, ())
            if (
                not history
                or history[0].existed
                or not _is_production_source_path(path)
            ):
                continue
            tokens = _call_path_reference_tokens(path)
            referenced_by_production = any(
                other_path != path and _content_references_any_token(content, tokens)
                for other_path, content in changed_contents.items()
            )
            explicitly_checked_as_entrypoint = _content_references_any_token(
                check_text,
                tokens,
            )
            if not referenced_by_production and not explicitly_checked_as_entrypoint:
                unwired.append(path)
        return tuple(unwired)

    @property
    def unwired_behavior_failure_count(self) -> int:
        return sum(
            1
            for invocation in self._transcript
            if invocation.name in _VERIFICATION_TOOL_NAMES
            and invocation.result.get("status") == "failed"
            and invocation.result.get("behavior_check") is True
            and invocation.result.get("finalization_blocking") is not False
        )

    @property
    def exploration_tools_available(self) -> bool:
        return (
            self._exploration_calls_current_revision
            < self._exploration_limit_current_revision
            or self.mutation_recovery_observation_available
            or self.mutation_recovery_edit_lease is not None
        )

    @property
    def verification_tools_available(self) -> bool:
        if self.exploration_tools_available:
            return True
        if self._revision <= 0:
            return False
        if self._last_check_status is None:
            return True
        passed = {
            _normalized_check_command(command)
            for command in self._passed_check_commands
        }
        if any(command not in passed for command in self._required_check_commands):
            return True
        if self._locked_check_obligations_outstanding():
            return True
        return bool(self._failed_checks_current_revision)

    def _outstanding_behavior_check_obligation(self) -> bool:
        return _requires_behavior_check(
            self._candidate.changed_paths()
        ) and not any(
            is_behavior_check_command(command)
            for command in self.passed_check_commands
        )

    def _uncovered_declaration_paths(self) -> tuple[str, ...]:
        return tuple(
            path
            for path in self._candidate.changed_paths()
            if is_declaration_path(path)
            and not any(
                command_covers_declaration(command, path)
                for command in self.passed_check_commands
            )
        )

    def _locked_check_obligations_outstanding(self) -> bool:
        """Finalization check evidence is missing that a locked check could supply.

        The exploration lock only admits required commands (or a behavior
        check when the finalization gate still needs one), so the tool
        surface must stay open exactly while an admissible check could
        still clear an outstanding finalization obligation. Otherwise the
        workflow phase demands verification while every verification tool
        is withheld and the loop can only spin finalization rejections.
        """

        changed_paths = self._candidate.changed_paths()
        if not changed_paths:
            return False
        if self._outstanding_behavior_check_obligation():
            return True
        if self._uncovered_declaration_paths():
            return True
        return not self._required_check_commands and not self._passed_check_commands

    def _check_clears_locked_finalization_obligation(
        self, normalized_command: str
    ) -> bool:
        """The lock must never make a finalization obligation unsatisfiable."""

        if not self._candidate.changed_paths():
            return False
        if self._outstanding_behavior_check_obligation() and is_behavior_check_command(
            normalized_command
        ):
            return True
        return any(
            command_covers_declaration(normalized_command, path)
            for path in self._uncovered_declaration_paths()
        )

    @property
    def diff_tool_available(self) -> bool:
        return bool(self.changed_paths) and not self.diff_inspected_for_current_revision

    @property
    def revision(self) -> int:
        return self._revision

    def finalization_issues(self) -> tuple[str, ...]:
        changed_paths = self._candidate.changed_paths()
        if not changed_paths:
            return ("effective_patch_required",)
        issues: list[str] = []
        if not self.passed_check_commands:
            issues.append("current_revision_check_required")
        if _requires_behavior_check(changed_paths) and not any(
            is_behavior_check_command(command) for command in self.passed_check_commands
        ):
            issues.append("current_revision_behavior_check_required")
        normalized_passed_checks = {
            _normalized_check_command(command) for command in self.passed_check_commands
        }
        issues.extend(
            "current_revision_required_acceptance_check_required:"
            + stable_hash(command)[:16]
            for command in self._required_check_commands
            if command not in normalized_passed_checks
        )
        issues.extend(
            "current_revision_check_failure_unresolved:" + stable_hash(command)[:16]
            for command in sorted(self._active_failed_checks())
        )
        uncovered_declarations = tuple(
            path
            for path in changed_paths
            if is_declaration_path(path)
            and not any(
                command_covers_declaration(command, path)
                for command in self.passed_check_commands
            )
        )
        if uncovered_declarations:
            issues.append(
                "current_revision_declaration_check_required:"
                + ",".join(uncovered_declarations)
            )
        if not self.diff_inspected_for_current_revision:
            issues.append("final_diff_not_inspected_after_latest_patch")
        return tuple(issues)

    def replay_call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Run one harness-driven finalization replay tool outside the model budget.

        The tool-call budget bounds the model's interactive loop. Finalization
        replay is deterministic harness code with a call count bounded by the
        recorded transcript, so a candidate that only needs its checks
        re-executed (or the final diff recorded) must not be discarded because
        the model already spent the budget. The same reasoning exempts replays
        from the per-revision exploration lock in ``_run_check``. Only
        verification tools are reachable through this path; the model-facing
        ``call`` stays gated.
        """
        if name not in _FINALIZATION_REPLAY_TOOL_NAMES:
            raise ValueError(f"finalization_replay_tool_not_permitted:{name}")
        self._finalization_replay_depth += 1
        try:
            return self.call(name, arguments)
        finally:
            self._finalization_replay_depth -= 1

    def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if (
            self._finalization_replay_depth == 0
            and self._tool_call_count >= self.max_tool_calls
        ):
            return {"status": "blocked", "reason": "tool_call_budget_exhausted"}
        self._tool_call_count += 1
        normalized = {str(key): value for key, value in arguments.items()}
        if name == "apply_patch":
            normalized = _normalize_apply_patch_arguments(normalized)
        target_path = str(normalized.get("path") or "")
        unwired_paths = self.unwired_created_source_paths
        if (
            name in _REVISION_MUTATION_TOOL_NAMES - {"revert_file"}
            and target_path in unwired_paths
            and self.unwired_behavior_failure_count
            >= MAX_UNWIRED_HELPER_BEHAVIOR_FAILURES
        ):
            result = {
                "status": "blocked",
                "path": target_path,
                "reason": f"unwired_created_source_path:{target_path}",
                "behavior_failure_count": self.unwired_behavior_failure_count,
                "unwired_created_source_paths": unwired_paths,
                "required_action": (
                    "wire_the_new_source_into_an_existing_production_callsite_or_"
                    "revert_the_unwired_file"
                ),
            }
            self._transcript.append(
                ToolInvocation(
                    index=self._tool_call_count,
                    name=name,
                    arguments=normalized,
                    result=result,
                    revision=self._revision,
                )
            )
            return result
        recovery_requirement = self.mutation_recovery_requirement
        recovery_edit_lease = self.mutation_recovery_edit_lease
        recovery_relocation = self.mutation_recovery_relocation_requirement
        if recovery_edit_lease is not None and name in _REVISION_MUTATION_TOOL_NAMES:
            violation = self._mutation_recovery_edit_contract_violation(
                name,
                normalized,
                recovery_edit_lease,
            )
            if violation is not None:
                self._transcript.append(
                    ToolInvocation(
                        index=self._tool_call_count,
                        name=name,
                        arguments=normalized,
                        result=violation,
                        revision=self._revision,
                    )
                )
                return violation
        mutation_recovery_observation = bool(
            (recovery_requirement is not None or recovery_edit_lease is not None)
            and name == "read_file"
        )
        recovery_read_arguments = normalized
        if recovery_relocation is not None and name == "read_file":
            requested_start = normalized.get("start_line")
            requested_end = normalized.get("end_line")
            if (
                isinstance(requested_start, bool)
                or not isinstance(requested_start, int)
                or isinstance(requested_end, bool)
                or not isinstance(requested_end, int)
                or requested_start < 1
                or requested_end < requested_start
                or requested_end - requested_start + 1 > self.max_read_lines
                or requested_end
                > int(
                    recovery_edit_lease.get("total_lines")
                    or recovery_edit_lease["observed_end_line"]
                )
            ):
                result = {
                    "status": "blocked",
                    "path": str(recovery_edit_lease["path"]),
                    "reason": (
                        "mutation_recovery_relocation_requires_explicit_line_range"
                    ),
                    "recovery_lease": dict(recovery_edit_lease),
                }
                self._transcript.append(
                    ToolInvocation(
                        index=self._tool_call_count,
                        name=name,
                        arguments=normalized,
                        result=result,
                        revision=self._revision,
                    )
                )
                return result
            observed_ranges = _mutation_recovery_observed_ranges(recovery_edit_lease)
            if any(
                requested_start <= observed_end and requested_end >= observed_start
                for observed_start, observed_end in observed_ranges
            ):
                result = {
                    "status": "blocked",
                    "path": str(recovery_edit_lease["path"]),
                    "reason": (
                        "mutation_recovery_relocation_requires_nonoverlapping_range"
                    ),
                    "recovery_lease": dict(recovery_edit_lease),
                    "requested_line_range": (requested_start, requested_end),
                }
                self._transcript.append(
                    ToolInvocation(
                        index=self._tool_call_count,
                        name=name,
                        arguments=normalized,
                        result=result,
                        revision=self._revision,
                    )
                )
                return result
        if recovery_edit_lease is not None and name in _EXPLORATION_TOOL_NAMES:
            if name != "read_file":
                result = {
                    "status": "blocked",
                    "reason": "mutation_recovery_edit_target_read_only",
                    "target_path": recovery_edit_lease["path"],
                    "required_action": (
                        "read_an_explicit_range_of_the_leased_file_or_apply_a_"
                        "guarded_line_edit"
                    ),
                }
                self._transcript.append(
                    ToolInvocation(
                        index=self._tool_call_count,
                        name=name,
                        arguments=normalized,
                        result=result,
                        revision=self._revision,
                    )
                )
                return result
            recovery_read_arguments = dict(normalized)
            recovery_read_arguments["path"] = recovery_edit_lease["path"]
        if recovery_requirement is not None and name in _EXPLORATION_TOOL_NAMES:
            if name != "read_file":
                result = {
                    "status": "blocked",
                    "reason": "mutation_recovery_target_read_required",
                    "target_path": recovery_requirement["path"],
                    "origin_reason": recovery_requirement["reason"],
                    "required_action": "read_the_exact_target_path_then_retry_mutation",
                }
                self._transcript.append(
                    ToolInvocation(
                        index=self._tool_call_count,
                        name=name,
                        arguments=normalized,
                        result=result,
                        revision=self._revision,
                    )
                )
                return result
            recovery_read_arguments = dict(normalized)
            recovery_read_arguments["path"] = recovery_requirement["path"]
            suggested_start = recovery_requirement.get("suggested_start_line")
            suggested_end = recovery_requirement.get("suggested_end_line")
            explicit_range = str(normalized.get("path") or "") == recovery_requirement[
                "path"
            ] and (
                isinstance(normalized.get("start_line"), int)
                or isinstance(normalized.get("end_line"), int)
            )
            if (
                not explicit_range
                and isinstance(suggested_start, int)
                and isinstance(suggested_end, int)
            ):
                recovery_read_arguments["start_line"] = suggested_start
                recovery_read_arguments["end_line"] = suggested_end
        if name in _EXPLORATION_TOOL_NAMES and not self.exploration_tools_available:
            result = {
                "status": "blocked",
                "reason": ("implementation_action_required_after_exploration_budget"),
                "exploration_calls": self._exploration_calls_current_revision,
                "revision": self._revision,
            }
            self._transcript.append(
                ToolInvocation(
                    index=self._tool_call_count,
                    name=name,
                    arguments=normalized,
                    result=result,
                    revision=self._revision,
                )
            )
            return result
        if name in _EXPLORATION_TOOL_NAMES:
            self._exploration_calls_current_revision += 1
        try:
            if name == "search":
                result = self._search(normalized)
            elif name == "list_files":
                result = self._list_files(normalized)
            elif name == "read_file":
                result = self._read_file(
                    recovery_read_arguments,
                    force_refresh=mutation_recovery_observation,
                )
            elif name == "apply_patch":
                result = self._apply_patch(normalized)
            elif name == "replace_lines":
                result = self._replace_lines(normalized)
            elif name == "replace_line_ranges":
                result = self._replace_line_ranges(normalized)
            elif name == "revert_file":
                result = self._revert_file(normalized)
            elif name == "run_check":
                result = self._run_check(normalized)
            elif name == "run_python_check":
                result = self._run_python_check(normalized)
            elif name == "git_diff":
                result = self._git_diff()
            else:
                result = {"status": "blocked", "reason": f"unknown_tool:{name}"}
        except (OSError, UnicodeError, ValueError) as exc:
            result = {
                "status": "blocked",
                "reason": str(exc) or type(exc).__name__,
            }
        if mutation_recovery_observation and result.get("status") == "ok":
            requested_path = str(normalized.get("path") or "")
            requested_start = normalized.get("start_line")
            requested_end = normalized.get("end_line")
            origin_reason = str(
                (
                    recovery_requirement["reason"]
                    if recovery_requirement is not None
                    else recovery_edit_lease["recovery_origin_reason"]
                )
            )
            observed_start = int(result.get("start_line") or 0)
            observed_end = int(result.get("end_line") or 0)
            file_hash = str(result.get("file_hash") or "")
            if observed_start < 1 or observed_end < observed_start or not file_hash:
                result = {
                    "status": "blocked",
                    "path": str(result.get("path") or ""),
                    "reason": "mutation_recovery_observation_empty",
                    "required_action": (
                        "read_a_nonempty_explicit_range_of_the_exact_target_file"
                    ),
                }
            else:
                lease = {
                    "path": str(result["path"]),
                    "file_hash": file_hash,
                    "observed_start_line": observed_start,
                    "observed_end_line": observed_end,
                    "observed_ranges": [
                        *(
                            [
                                list(item)
                                for item in _mutation_recovery_observed_ranges(
                                    recovery_edit_lease
                                )
                            ]
                            if recovery_edit_lease is not None
                            else []
                        ),
                        [observed_start, observed_end],
                    ],
                    "total_lines": int(result.get("total_lines") or observed_end),
                    "relocation_count": (
                        int(recovery_edit_lease.get("relocation_count") or 0) + 1
                        if recovery_edit_lease is not None
                        else 0
                    ),
                    "recovery_origin_reason": origin_reason,
                }
                if recovery_edit_lease is not None:
                    lease["relocated_from"] = {
                        "observed_start_line": int(
                            recovery_edit_lease["observed_start_line"]
                        ),
                        "observed_end_line": int(
                            recovery_edit_lease["observed_end_line"]
                        ),
                    }
                self._mutation_recovery_edit_lease = lease
                result["recovery_origin_reason"] = origin_reason
                result["recovery_lease"] = lease
                result["recovery_lease_created"] = recovery_requirement is not None
                result["recovery_lease_refreshed"] = recovery_edit_lease is not None
                result["recovery_lease_relocated"] = recovery_relocation is not None
            result["requested_path"] = requested_path
            result["recovery_redirected"] = (
                requested_path
                != str(
                    (
                        recovery_requirement["path"]
                        if recovery_requirement is not None
                        else recovery_edit_lease["path"]
                    )
                )
                or requested_start != recovery_read_arguments.get("start_line")
                or requested_end != recovery_read_arguments.get("end_line")
            )
        self._transcript.append(
            ToolInvocation(
                index=self._tool_call_count,
                name=name,
                arguments=normalized,
                result=result,
                revision=self._revision,
            )
        )
        return result

    def _mutation_recovery_edit_contract_violation(
        self,
        name: str,
        arguments: Mapping[str, Any],
        lease: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        target_path = str(lease["path"])
        expected_hash = str(lease["file_hash"])
        observed_start = int(lease["observed_start_line"])
        observed_end = int(lease["observed_end_line"])

        def blocked(reason: str) -> dict[str, Any]:
            return {
                "status": "blocked",
                "path": str(arguments.get("path") or ""),
                "reason": f"mutation_recovery_edit_contract_violation:{reason}",
                "recovery_contract_violation": True,
                "recovery_lease": dict(lease),
                "required_action": (
                    "use_replace_lines_or_replace_line_ranges_on_the_exact_leased_"
                    "path_hash_and_observed_line_range"
                ),
            }

        if name not in {"replace_lines", "replace_line_ranges"}:
            return blocked("guarded_line_edit_required")
        if str(arguments.get("path") or "") != target_path:
            return blocked("path_mismatch")
        if str(arguments.get("expected_file_hash") or "") != expected_hash:
            return blocked("file_hash_mismatch")
        if name == "replace_lines":
            try:
                start = int(arguments.get("start_line") or 0)
                end = int(arguments.get("end_line") or 0)
            except (TypeError, ValueError):
                return blocked("invalid_line_range")
            if start < observed_start or end < start or end > observed_end:
                return blocked("line_range_outside_observation")
            if not str(arguments.get("expected_old") or ""):
                return blocked("expected_old_required")
            return None

        replacements = arguments.get("replacements")
        if not isinstance(replacements, (list, tuple)) or not replacements:
            return blocked("line_replacements_required")
        for replacement in replacements:
            if not isinstance(replacement, Mapping):
                return blocked("invalid_line_replacement")
            try:
                start = int(replacement.get("start_line") or 0)
                end = int(replacement.get("end_line") or 0)
            except (TypeError, ValueError):
                return blocked("invalid_line_range")
            if start < observed_start or end < start or end > observed_end:
                return blocked("line_range_outside_observation")
            if not str(replacement.get("expected_old") or ""):
                return blocked("expected_old_required")
        return None

    def _mutation_recovery_edit_feedback(self, payload: str) -> dict[str, Any]:
        lease = self._mutation_recovery_edit_lease
        if lease is None:
            return {}
        lines = payload.splitlines()
        start = max(1, int(lease["observed_start_line"]))
        end = min(len(lines), int(lease["observed_end_line"]))
        numbered_content = "\n".join(
            f"{line_number}: {lines[line_number - 1]}"
            for line_number in range(start, end + 1)
        )
        truncated = len(numbered_content) > 16_000
        if truncated:
            numbered_content = numbered_content[:16_000]
        return {
            "recovery_lease": dict(lease),
            "recovery_context": {
                "start_line": start,
                "end_line": end,
                "numbered_content": numbered_content,
                "truncated": truncated,
            },
            "required_action": (
                "copy_expected_old_exactly_from_recovery_context_without_numeric_"
                "prefix_and_submit_one_syntax_valid_atomic_edit"
            ),
        }

    def export_patches(self) -> tuple[WorkspaceFilePatch, ...]:
        patches: list[WorkspaceFilePatch] = []
        for index, path in enumerate(self._candidate.changed_paths()):
            target = self.root / path
            try:
                content = read_regular_file_text(
                    self.root,
                    target,
                    max_bytes=self.max_patch_bytes,
                )
            except (UnicodeDecodeError, UnsafeRegularFileError) as exc:
                raise ValueError(
                    f"unsafe_regular_file:{path}:{type(exc).__name__}"
                ) from None
            patches.append(
                WorkspaceFilePatch(
                    patch_id=f"tool_loop_patch_{index:03d}",
                    path=path,
                    operation="create_or_replace",
                    content=content,
                    rationale=self._rationales.get(
                        path, "Interactive coding tool change."
                    ),
                )
            )
        return tuple(patches)

    def restore_patches(self, patches: tuple[WorkspaceFilePatch, ...]) -> None:
        for patch in patches:
            snapshot = self._capture_file_revision_snapshot(patch.path)
            results = self._candidate.apply((patch,))
            if not results or results[0].status not in {"applied", "no_op"}:
                reason = (
                    results[0].blocked_reason if results else "missing_patch_result"
                )
                raise ValueError(f"checkpoint_patch_restore_failed:{reason}")
            self._rationales[patch.path] = patch.rationale
            if results[0].status == "applied":
                self._mark_revision_changed()
                self._file_revision_history.setdefault(patch.path, []).append(
                    _FileRevisionSnapshot(
                        applied_revision=self._revision,
                        existed=snapshot.existed,
                        content=snapshot.content,
                        rationale=snapshot.rationale,
                    )
                )

    def restore_passed_checks(self, commands: tuple[str, ...]) -> None:
        restored = tuple(
            command
            for command in commands
            if _normalized_check_command(command)
            not in self._failed_checks_current_revision
        )
        self._passed_check_commands.extend(restored)
        if restored and self._last_check_status is None:
            self._last_check_status = "passed"

    def restore_diff_inspection(self, inspected: bool) -> None:
        self._last_diff_revision = self._revision if inspected else None

    def restore_usage(
        self,
        *,
        transcript: tuple[ToolInvocation | Mapping[str, Any], ...],
        patch_bytes: int,
    ) -> None:
        restored: list[ToolInvocation] = []
        for fallback_index, item in enumerate(transcript, start=1):
            if isinstance(item, ToolInvocation):
                invocation = item
            else:
                invocation = ToolInvocation(
                    index=int(item.get("index") or fallback_index),
                    name=str(item.get("name") or "unknown"),
                    arguments=dict(item.get("arguments") or {}),
                    result=dict(item.get("result") or {}),
                    revision=int(item.get("revision") or 0),
                )
            restored.append(invocation)
        restored_count = max(
            (invocation.index for invocation in restored),
            default=0,
        )
        if restored_count > self.max_tool_calls:
            raise ValueError("checkpoint_tool_call_budget_exceeded")
        normalized_patch_bytes = max(self._patch_bytes, int(patch_bytes))
        if normalized_patch_bytes > self.max_patch_bytes:
            raise ValueError("checkpoint_patch_budget_exceeded")
        self._transcript = restored
        self._tool_call_count = restored_count
        self._patch_bytes = normalized_patch_bytes
        self._mutation_recovery_edit_lease = None
        if restored:
            self._revision = max(
                self._revision,
                max(invocation.revision for invocation in restored),
            )
            self._restore_file_revision_history(restored)
            for invocation in restored:
                if (
                    invocation.name == "revert_file"
                    and invocation.result.get("withdrawn_created_path") is True
                    and str(invocation.result.get("path") or "")
                ):
                    self._withdrawn_created_paths.add(
                        str(invocation.result["path"])
                    )
            restored_checks = tuple(
                invocation
                for invocation in sorted(restored, key=lambda item: item.index)
                if invocation.name in {"run_check", "run_python_check"}
                and invocation.result.get("status") in {"passed", "failed"}
            )
            current_checks = tuple(
                invocation
                for invocation in restored_checks
                if invocation.revision == self._revision
            )
            if current_checks:
                self._last_check_status = str(current_checks[-1].result["status"])
            for invocation in restored_checks:
                command = _normalized_check_command(
                    _invocation_check_command(invocation)
                )
                if not command:
                    continue
                if invocation.revision == self._revision:
                    self._executed_check_commands_current_revision.add(command)
                if invocation.result.get("status") == "passed":
                    self._failed_checks_current_revision.pop(command, None)
                    continue
                previous = self._failed_checks_current_revision.get(command, {})
                restored_failure = dict(invocation.result)
                restored_failure["attempt"] = int(
                    restored_failure.get("attempt")
                    or int(previous.get("attempt") or 0) + 1
                )
                restored_failure.setdefault("revision", invocation.revision)
                restored_failure.setdefault(
                    "revision_attempt",
                    restored_failure["attempt"],
                )
                restored_failure.setdefault("check_role", self._check_role(command))
                restored_failure.setdefault(
                    "behavior_check", is_behavior_check_command(command)
                )
                classification = _probe_setup_failure_classification(
                    check_role=str(restored_failure["check_role"]),
                    stderr_tail=str(restored_failure.get("stderr_tail") or ""),
                )
                if classification is not None:
                    restored_failure.setdefault(
                        "failure_classification",
                        classification,
                    )
                    restored_failure.setdefault("finalization_blocking", False)
                if restored_failure.get("finalization_blocking") is False:
                    self._failed_checks_current_revision.pop(command, None)
                    continue
                self._failed_checks_current_revision[command] = restored_failure
        self._exploration_calls_current_revision = 0
        self._exploration_limit_current_revision = (
            self.max_exploration_calls_per_revision
        )
        self._read_coverage_current_revision = {}
        restored_check_commands: set[str] = set()
        for invocation in sorted(restored, key=lambda item: item.index):
            if invocation.revision != self._revision:
                continue
            recovery_lease = invocation.result.get("recovery_lease")
            if (
                invocation.name == "read_file"
                and invocation.result.get("status") == "ok"
                and isinstance(recovery_lease, Mapping)
            ):
                self._mutation_recovery_edit_lease = {
                    "path": str(recovery_lease["path"]),
                    "file_hash": str(recovery_lease["file_hash"]),
                    "observed_start_line": int(recovery_lease["observed_start_line"]),
                    "observed_end_line": int(recovery_lease["observed_end_line"]),
                    "observed_ranges": [
                        list(item)
                        for item in _mutation_recovery_observed_ranges(recovery_lease)
                    ],
                    "total_lines": int(
                        recovery_lease.get("total_lines")
                        or recovery_lease["observed_end_line"]
                    ),
                    "relocation_count": int(
                        recovery_lease.get("relocation_count") or 0
                    ),
                    "recovery_origin_reason": str(
                        recovery_lease["recovery_origin_reason"]
                    ),
                }
                if isinstance(recovery_lease.get("relocated_from"), Mapping):
                    self._mutation_recovery_edit_lease["relocated_from"] = {
                        "observed_start_line": int(
                            recovery_lease["relocated_from"]["observed_start_line"]
                        ),
                        "observed_end_line": int(
                            recovery_lease["relocated_from"]["observed_end_line"]
                        ),
                    }
            if (
                invocation.name == "read_file"
                and invocation.result.get("status") == "ok"
            ):
                path = str(invocation.result.get("path") or "")
                start = int(invocation.result.get("start_line") or 0)
                end = int(invocation.result.get("end_line") or 0)
                if path and start > 0 and end >= start:
                    self._record_read_coverage(path, start, end)
            if (
                invocation.name in _EXPLORATION_TOOL_NAMES
                and invocation.result.get("reason")
                != "implementation_action_required_after_exploration_budget"
            ):
                self._exploration_calls_current_revision += 1
                continue
            if invocation.name not in {
                "run_check",
                "run_python_check",
            } or invocation.result.get("status") not in {"passed", "failed"}:
                continue
            command = _normalized_check_command(_invocation_check_command(invocation))
            if not command or command in restored_check_commands:
                continue
            restored_check_commands.add(command)
            if self._revision > 0:
                self._exploration_limit_current_revision = min(
                    self.max_exploration_calls_per_revision
                    + MAX_DIAGNOSTIC_EXPLORATION_GRANT_PER_REVISION,
                    max(
                        self._exploration_limit_current_revision,
                        self._exploration_calls_current_revision
                        + DIAGNOSTIC_EXPLORATION_CALLS_PER_DISTINCT_CHECK,
                    ),
                )

    def close(self) -> None:
        self._candidate.close()

    def __enter__(self) -> CodingToolSession:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "")
        if not query:
            return {"status": "blocked", "reason": "search_query_required"}
        base_path = _safe_path(self.root, str(arguments.get("path") or "."))
        if not base_path.exists():
            return {"status": "blocked", "reason": "search_path_not_found"}
        pattern = str(arguments.get("glob") or "*")
        requested = int(arguments.get("max_results") or self.max_search_results)
        limit = max(1, min(requested, self.max_search_results))
        needle = query.casefold()
        matches: list[dict[str, Any]] = []
        for path in _iter_repository_files(base_path, self.root):
            relative = path.relative_to(self.root).as_posix()
            if (
                pattern
                and not fnmatch.fnmatch(path.name, pattern)
                and not fnmatch.fnmatch(relative, pattern)
            ):
                continue
            try:
                lines = read_regular_file_text(
                    self.root,
                    path,
                    max_bytes=2_000_000,
                ).splitlines()
            except (UnicodeDecodeError, UnsafeRegularFileError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if needle in line.casefold():
                    matches.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "text": line[:500],
                        }
                    )
                    if len(matches) >= limit:
                        return {"status": "ok", "matches": matches, "truncated": True}
        return {"status": "ok", "matches": matches, "truncated": False}

    def _list_files(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        base_path = _safe_path(self.root, str(arguments.get("path") or "."))
        if not base_path.exists():
            return {"status": "blocked", "reason": "list_path_not_found"}
        pattern = str(arguments.get("glob") or "*")
        requested = int(arguments.get("max_results") or self.max_search_results)
        limit = max(1, min(requested, self.max_search_results))
        files: list[dict[str, Any]] = []
        for path in _iter_repository_files(base_path, self.root):
            relative = path.relative_to(self.root).as_posix()
            if (
                pattern
                and not fnmatch.fnmatch(path.name, pattern)
                and not fnmatch.fnmatch(relative, pattern)
            ):
                continue
            files.append({"path": relative, "byte_size": path.stat().st_size})
            if len(files) >= limit:
                return {"status": "ok", "files": files, "truncated": True}
        return {"status": "ok", "files": files, "truncated": False}

    def _read_file(
        self,
        arguments: Mapping[str, Any],
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        relative = str(arguments.get("path") or "")
        path = _safe_path(self.root, relative)
        try:
            payload = read_regular_file_text(
                self.root,
                path,
                max_bytes=4_000_000,
            )
        except (UnicodeDecodeError, UnsafeRegularFileError):
            return {"status": "blocked", "reason": f"unsafe_regular_file:{relative}"}
        start = max(1, int(arguments.get("start_line") or 1))
        end = int(
            arguments.get("end_line") or (start + min(399, self.max_read_lines - 1))
        )
        if end < start:
            return {"status": "blocked", "reason": "invalid_read_range"}
        if end - start + 1 > self.max_read_lines:
            return {
                "status": "blocked",
                "reason": f"read_range_exceeds_limit:{self.max_read_lines}",
            }
        lines = payload.splitlines()
        actual_end = min(end, len(lines))
        normalized_path = path.relative_to(self.root).as_posix()
        structural_context = (
            _python_structural_context(payload, start_line=start, end_line=actual_end)
            if path.suffix == ".py" and actual_end >= start
            else None
        )
        if (
            not force_refresh
            and actual_end >= start
            and _range_is_covered(
                self._read_coverage_current_revision.get(normalized_path, ()),
                start,
                actual_end,
            )
        ):
            result = {
                "status": "ok",
                "cached": True,
                "observation": "read_range_already_observed",
                "path": normalized_path,
                "file_hash": stable_hash(payload),
                "start_line": start,
                "end_line": actual_end,
                "covered_ranges": tuple(
                    self._read_coverage_current_revision.get(normalized_path, ())
                ),
                "required_action": (
                    "request_an_unseen_range_or_apply_the_supported_edit"
                ),
            }
            if structural_context is not None:
                result["structural_context"] = structural_context
            return result
        selected = lines[start - 1 : end]
        content = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=start)
        )
        if actual_end >= start:
            self._record_read_coverage(normalized_path, start, actual_end)
        result = {
            "status": "ok",
            "path": normalized_path,
            "file_hash": stable_hash(payload),
            "start_line": start,
            "end_line": actual_end,
            "total_lines": len(lines),
            "content": content,
            "recovery_refresh": force_refresh,
        }
        if structural_context is not None:
            result["structural_context"] = structural_context
        return result

    def _record_read_coverage(self, path: str, start: int, end: int) -> None:
        ranges = [*self._read_coverage_current_revision.get(path, ()), (start, end)]
        merged: list[tuple[int, int]] = []
        for candidate_start, candidate_end in sorted(ranges):
            if merged and candidate_start <= merged[-1][1] + 1:
                merged[-1] = (
                    merged[-1][0],
                    max(merged[-1][1], candidate_end),
                )
            else:
                merged.append((candidate_start, candidate_end))
        self._read_coverage_current_revision[path] = merged

    def _replace_lines(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        relative = str(arguments.get("path") or "")
        path = _safe_path(self.root, relative)
        try:
            payload = read_regular_file_text(
                self.root,
                path,
                max_bytes=4_000_000,
            )
        except (UnicodeDecodeError, UnsafeRegularFileError):
            return {"status": "blocked", "reason": f"unsafe_regular_file:{relative}"}
        current_hash = stable_hash(payload)
        expected_hash = str(arguments.get("expected_file_hash") or "")
        if not expected_hash:
            return {"status": "blocked", "reason": "expected_file_hash_required"}
        if expected_hash != current_hash:
            return {
                "status": "blocked",
                "reason": f"stale_file_revision:{relative}",
                "current_file_hash": current_hash,
                "required_action": "read_file_then_retry_replace_lines",
            }
        start = int(arguments.get("start_line") or 0)
        end = int(arguments.get("end_line") or 0)
        lines = payload.splitlines(keepends=True)
        if start < 1 or end < start or end > len(lines):
            return {
                "status": "blocked",
                "reason": f"invalid_line_range:{start}:{end}:{len(lines)}",
            }
        requested_range = (start, end)
        expected_old = str(arguments.get("expected_old") or "")
        if expected_old:
            lease = self._mutation_recovery_edit_lease
            allowed_start = (
                int(lease["observed_start_line"]) if lease is not None else start
            )
            allowed_end = int(lease["observed_end_line"]) if lease is not None else end
            resolved, matches = _resolve_expected_line_range(
                lines,
                requested_start=start,
                requested_end=end,
                expected_old=expected_old,
                allowed_start=allowed_start,
                allowed_end=allowed_end,
            )
            if resolved is None:
                reason = (
                    "expected_old_ambiguous_in_recovery_lease"
                    if len(matches) > 1
                    else "expected_old_not_found_in_recovery_lease"
                )
                return {
                    "status": "blocked",
                    "path": relative,
                    "reason": f"{reason}:{relative}",
                    "requested_line_range": requested_range,
                    "matching_line_ranges": [list(item) for item in matches],
                    **self._mutation_recovery_edit_feedback(payload),
                }
            start, end = resolved
        original = "".join(lines[start - 1 : end])
        replacement = _preserve_replaced_line_boundary(
            original,
            str(arguments.get("new") or ""),
        )
        old_fragment, new_fragment = _unique_line_range_replacement(
            payload,
            lines,
            start=start,
            end=end,
            replacement=replacement,
        )
        result = self._apply_patch(
            {
                "path": relative,
                "operation": "replace_fragment",
                "old": old_fragment,
                "new": new_fragment,
                "rationale": arguments.get("rationale"),
            }
        )
        if result.get("status") == "applied":
            updated = read_regular_file_text(self.root, path, max_bytes=4_000_000)
            result["file_hash"] = stable_hash(updated)
            result["replaced_line_range"] = (start, end)
            result["requested_line_range"] = requested_range
            result["recovery_line_relocated"] = requested_range != (start, end)
            result["resolved_replacement"] = {
                "start_line": start,
                "end_line": end,
                "new": str(arguments.get("new") or ""),
                "expected_old": expected_old or None,
            }
        elif self._mutation_recovery_edit_lease is not None:
            result.update(self._mutation_recovery_edit_feedback(payload))
            result["requested_line_range"] = requested_range
            result["resolved_line_range"] = (start, end)
        return result

    def _replace_line_ranges(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        relative = str(arguments.get("path") or "")
        path = _safe_path(self.root, relative)
        try:
            payload = read_regular_file_text(
                self.root,
                path,
                max_bytes=4_000_000,
            )
        except (UnicodeDecodeError, UnsafeRegularFileError):
            return {"status": "blocked", "reason": f"unsafe_regular_file:{relative}"}
        current_hash = stable_hash(payload)
        expected_hash = str(arguments.get("expected_file_hash") or "")
        if not expected_hash:
            return {"status": "blocked", "reason": "expected_file_hash_required"}
        if expected_hash != current_hash:
            return {
                "status": "blocked",
                "reason": f"stale_file_revision:{relative}",
                "current_file_hash": current_hash,
                "required_action": "read_file_then_retry_replace_line_ranges",
            }
        raw_replacements = arguments.get("replacements")
        if not isinstance(raw_replacements, (list, tuple)) or not raw_replacements:
            return {"status": "blocked", "reason": "line_replacements_required"}
        if len(raw_replacements) > 64:
            return {"status": "blocked", "reason": "line_replacement_limit_exceeded"}

        lines = payload.splitlines(keepends=True)
        replacements: list[tuple[int, int, str, str, tuple[int, int], str]] = []
        for raw in raw_replacements:
            if not isinstance(raw, Mapping):
                return {"status": "blocked", "reason": "invalid_line_replacement"}
            start = int(raw.get("start_line") or 0)
            end = int(raw.get("end_line") or 0)
            if start < 1 or end < start or end > len(lines):
                return {
                    "status": "blocked",
                    "reason": f"invalid_line_range:{start}:{end}:{len(lines)}",
                }
            requested_range = (start, end)
            expected_old = str(raw.get("expected_old") or "")
            if expected_old:
                lease = self._mutation_recovery_edit_lease
                allowed_start = (
                    int(lease["observed_start_line"]) if lease is not None else start
                )
                allowed_end = (
                    int(lease["observed_end_line"]) if lease is not None else end
                )
                resolved, matches = _resolve_expected_line_range(
                    lines,
                    requested_start=start,
                    requested_end=end,
                    expected_old=expected_old,
                    allowed_start=allowed_start,
                    allowed_end=allowed_end,
                )
                if resolved is None:
                    reason = (
                        "expected_old_ambiguous_in_recovery_lease"
                        if len(matches) > 1
                        else "expected_old_not_found_in_recovery_lease"
                    )
                    return {
                        "status": "blocked",
                        "path": relative,
                        "reason": f"{reason}:{relative}",
                        "requested_line_range": requested_range,
                        "matching_line_ranges": [list(item) for item in matches],
                        **self._mutation_recovery_edit_feedback(payload),
                    }
                start, end = resolved
            original = "".join(lines[start - 1 : end])
            raw_new = str(raw.get("new") or "")
            replacements.append(
                (
                    start,
                    end,
                    _preserve_replaced_line_boundary(
                        original,
                        raw_new,
                    ),
                    expected_old,
                    requested_range,
                    raw_new,
                )
            )
        replacements.sort(key=lambda item: (item[0], item[1]))
        previous_end = 0
        for start, end, _, _, _, _ in replacements:
            if start <= previous_end:
                return {
                    "status": "blocked",
                    "reason": f"overlapping_line_ranges:{start}:{end}",
                }
            previous_end = end

        outer_start = replacements[0][0]
        outer_end = replacements[-1][1]
        rebuilt: list[str] = []
        cursor = outer_start
        for start, end, new, _, _, _ in replacements:
            rebuilt.extend(lines[cursor - 1 : start - 1])
            rebuilt.append(new)
            cursor = end + 1
        rebuilt.extend(lines[cursor - 1 : outer_end])
        old_fragment, new_fragment = _unique_line_range_replacement(
            payload,
            lines,
            start=outer_start,
            end=outer_end,
            replacement="".join(rebuilt),
        )
        result = self._apply_patch(
            {
                "path": relative,
                "operation": "replace_fragment",
                "old": old_fragment,
                "new": new_fragment,
                "rationale": arguments.get("rationale"),
            }
        )
        if result.get("status") == "applied":
            updated = read_regular_file_text(self.root, path, max_bytes=4_000_000)
            result["file_hash"] = stable_hash(updated)
            result["replaced_line_ranges"] = [
                (start, end) for start, end, _, _, _, _ in replacements
            ]
            result["requested_line_ranges"] = [
                requested for _, _, _, _, requested, _ in replacements
            ]
            result["recovery_line_relocated"] = any(
                requested != (start, end)
                for start, end, _, _, requested, _ in replacements
            )
            result["resolved_replacements"] = [
                {
                    "start_line": start,
                    "end_line": end,
                    "new": raw_new,
                    "expected_old": expected_old or None,
                }
                for start, end, _, expected_old, _, raw_new in replacements
            ]
        elif self._mutation_recovery_edit_lease is not None:
            result.update(self._mutation_recovery_edit_feedback(payload))
            result["requested_line_ranges"] = [
                requested for _, _, _, _, requested, _ in replacements
            ]
            result["resolved_line_ranges"] = [
                (start, end) for start, end, _, _, _, _ in replacements
            ]
        return result

    def _apply_patch(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = str(arguments.get("path") or "")
        target = _safe_path(self.root, path)
        operation = str(arguments.get("operation") or "")
        content = str(arguments.get("content") or "")
        old = str(arguments.get("old") or "")
        new = str(arguments.get("new") or "")
        patch_bytes = sum(len(value.encode("utf-8")) for value in (content, old, new))
        if self._patch_bytes + patch_bytes > self.max_patch_bytes:
            return {"status": "blocked", "reason": "patch_budget_exceeded"}
        self._patch_sequence += 1
        patch = WorkspaceFilePatch(
            patch_id=f"interactive_patch_{self._patch_sequence:03d}",
            path=path,
            operation=operation,
            content=content,
            old=old,
            new=new,
            rationale=str(arguments.get("rationale") or "")[:1000],
        )
        syntax_issue = _prospective_syntax_issue(
            root=self.root,
            target=target,
            patch=patch,
        )
        if syntax_issue is not None:
            return {
                "status": "blocked",
                "path": path,
                "reason": syntax_issue,
                "required_action": (
                    "submit_one_syntax_valid_atomic_edit_for_this_file"
                ),
            }
        snapshot = self._capture_file_revision_snapshot(path)
        (result,) = self._candidate.apply((patch,))
        if result.status == "applied":
            self._patch_bytes += patch_bytes
            self._rationales[path] = patch.rationale
            self._mark_revision_changed()
            self._file_revision_history.setdefault(path, []).append(
                _FileRevisionSnapshot(
                    applied_revision=self._revision,
                    existed=snapshot.existed,
                    content=snapshot.content,
                    rationale=snapshot.rationale,
                )
            )
        payload = {
            "status": result.status,
            "path": result.path,
            "before_hash": result.before_hash,
            "after_hash": result.after_hash,
            "reason": result.blocked_reason,
        }
        if (
            result.status == "blocked"
            and str(result.blocked_reason or "").startswith(
                "replace_fragment_not_found:"
            )
            and old
        ):
            try:
                current = read_regular_file_text(
                    self.root,
                    target,
                    max_bytes=4_000_000,
                )
            except (UnicodeDecodeError, UnsafeRegularFileError):
                current = ""
            closest = _closest_fragment_context(current, old)
            if closest:
                payload.update(closest)
                payload["required_action"] = (
                    "replace_an_exact_fragment_from_closest_context_or_use_"
                    "replace_lines_with_the_current_file_hash"
                )
        return payload

    def _revert_file(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        relative = str(arguments.get("path") or "")
        target = _safe_path(self.root, relative)
        changed_before = relative in self._candidate.changed_paths()
        if not changed_before:
            return {"status": "no_change", "path": relative}
        history = self._file_revision_history.get(relative)
        if history:
            snapshot = history[-1]
        else:
            snapshot = self._canonical_file_revision_snapshot(relative)
        self._patch_sequence += 1
        if snapshot.existed:
            patch = WorkspaceFilePatch(
                patch_id=f"interactive_revert_{self._patch_sequence:03d}",
                path=relative,
                operation="create_or_replace",
                content=snapshot.content,
                rationale="Restore the file state before its latest candidate edit.",
            )
            (result,) = self._candidate.apply((patch,))
            if result.status not in {"applied", "no_op"}:
                return {
                    "status": "blocked",
                    "path": relative,
                    "reason": result.blocked_reason,
                }
        else:
            if target.is_symlink() or (target.exists() and not target.is_file()):
                return {
                    "status": "blocked",
                    "reason": f"unsafe_regular_file:{relative}",
                }
            if target.exists():
                target.unlink()
        if history:
            history.pop()
            if not history:
                self._file_revision_history.pop(relative, None)
        if snapshot.rationale is None:
            self._rationales.pop(relative, None)
        else:
            self._rationales[relative] = snapshot.rationale
        self._mark_revision_changed()
        payload = {
            "status": "reverted",
            "path": relative,
            "rolled_back_revision": snapshot.applied_revision,
            "rollback_scope": "previous_file_revision",
        }
        if not snapshot.existed:
            self._withdrawn_created_paths.add(relative)
            payload["withdrawn_created_path"] = True
        return payload

    def _run_check(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command") or "")
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return {
                "status": "blocked",
                "reason": f"invalid_command:{type(exc).__name__}",
            }
        if not argv:
            return {"status": "blocked", "reason": "empty_command"}
        normalized_command = shlex.join(argv)
        check_allowed_while_locked = (
            normalized_command in self._required_check_commands
            or (
                not self._required_check_commands
                and is_behavior_check_command(normalized_command)
            )
            or self._check_clears_locked_finalization_obligation(normalized_command)
        )
        if (
            self._finalization_replay_depth == 0
            and not self.exploration_tools_available
            and not check_allowed_while_locked
        ):
            # The exploration lock disciplines the model's interactive
            # spending; harness finalization replays re-execute recorded
            # commands the finalization gate itself demands, so the lock
            # must not make that obligation unsatisfiable.
            return {
                "status": "blocked",
                "reason": (
                    "required_oracle_or_implementation_action_only_after_"
                    "exploration_budget"
                ),
                "normalized_command": normalized_command,
                "required_action": (
                    "run_a_confirmed_required_check_or_change_the_candidate"
                ),
            }
        previous_failure = self._failed_checks_current_revision.get(normalized_command)
        same_failure_revision = (
            int((previous_failure or {}).get("revision") or -1) == self._revision
        )
        previous_revision_attempt = (
            int(
                (previous_failure or {}).get("revision_attempt")
                or (previous_failure or {}).get("attempt")
                or 0
            )
            if same_failure_revision
            else 0
        )
        if same_failure_revision and previous_revision_attempt >= 2:
            return {
                "status": "blocked",
                "reason": "failed_check_retry_limit_current_revision",
                "normalized_command": normalized_command,
                "revision": self._revision,
                "failed_attempts": previous_revision_attempt,
                "required_action": (
                    "change_or_revert_the_candidate_before_rerunning_this_check"
                ),
            }
        attempt = int((previous_failure or {}).get("attempt") or 0) + 1
        check_role = self._check_role(normalized_command)
        behavior_check = is_behavior_check_command(normalized_command)
        executable = Path(argv[0]).name
        if executable not in _ALLOWED_CHECK_EXECUTABLES and argv[0] != sys.executable:
            if executable == "git":
                if self.diff_inspected_for_current_revision:
                    required_action = (
                        "do_not_repeat_diff;diagnose_the_failed_check_then_edit_or_"
                        "revert_the_candidate"
                        if self._failed_checks_current_revision
                        else "do_not_repeat_diff;return_the_strict_final_json"
                    )
                else:
                    required_action = "use_the_dedicated_git_diff_tool_once"
                return {
                    "status": "blocked",
                    "reason": "git_command_requires_dedicated_diff_tool",
                    "normalized_command": normalized_command,
                    "diff_inspected_for_current_revision": (
                        self.diff_inspected_for_current_revision
                    ),
                    "failed_check_count": len(self._failed_checks_current_revision),
                    "required_action": required_action,
                }
            return {
                "status": "blocked",
                "reason": f"unsupported_executable:{executable}",
            }
        if self.capabilities_observed and executable not in self.available_executables:
            return {
                "status": "blocked",
                "reason": f"execution_runtime_unavailable:{executable}",
                "normalized_command": normalized_command,
                "available_executables": self.available_executables,
                "failure_classification": "runtime_unavailable",
                "finalization_blocking": False,
                "required_action": (
                    "use_an_available_runtime_or_a_required_project_command;"
                    "do_not_revert_product_code_for_this_environment_failure"
                ),
            }
        requested_timeout = float(arguments.get("timeout_seconds") or 120.0)
        timeout_seconds = max(
            1.0, min(requested_timeout, self.max_check_timeout_seconds)
        )
        with CandidateWorkspace.create(self.root) as verification_workspace:
            (result,) = run_workspace_verification(
                verification_workspace.root,
                (command,),
                timeout_seconds=timeout_seconds,
                allowed_executables=_ALLOWED_CHECK_EXECUTABLES,
                executor=self.executor,
            )
        if result.status == "passed":
            self._passed_check_commands.append(command)
            self._last_check_status = "passed"
        elif result.status == "failed":
            self._passed_check_commands = [
                passed
                for passed in self._passed_check_commands
                if _normalized_check_command(passed) != normalized_command
            ]
            self._last_check_status = "failed"
        elif result.status == "infra_error":
            self._infrastructure_failure_reason = (
                result.blocked_reason or "execution_backend_infra_error"
            )
        rendered = {
            "status": result.status,
            "exit_code": result.exit_code,
            "elapsed_sec": result.elapsed_sec,
            "stdout_tail": result.stdout_tail,
            "stderr_tail": result.stderr_tail,
            "reason": result.blocked_reason,
            "attempt": attempt,
            "revision": self._revision,
            "revision_attempt": previous_revision_attempt + 1,
            "check_role": check_role,
            "behavior_check": behavior_check,
        }
        failure_classification = (
            _probe_setup_failure_classification(
                check_role=check_role,
                stderr_tail=result.stderr_tail,
            )
            if result.status == "failed"
            else None
        )
        if (
            result.status == "failed"
            and check_role == "diagnostic_probe"
            and failure_classification is None
        ):
            failure_classification = _diagnostic_tooling_failure_classification(
                normalized_command
            )
        if failure_classification is not None:
            rendered["failure_classification"] = failure_classification
            rendered["finalization_blocking"] = False
        unwired_paths = self.unwired_created_source_paths
        if unwired_paths:
            rendered["call_path_warnings"] = {
                "unwired_created_source_paths": unwired_paths,
                "behavior_failure_count": (
                    self.unwired_behavior_failure_count
                    + int(result.status == "failed" and behavior_check)
                ),
                "required_action": (
                    "wire_each_new_source_into_a_reachable_production_callsite_"
                    "or_revert_it"
                ),
            }
        if result.status in {"passed", "failed"}:
            is_new_check = (
                normalized_command not in self._executed_check_commands_current_revision
            )
            self._executed_check_commands_current_revision.add(normalized_command)
            if self._revision > 0 and is_new_check:
                self._exploration_limit_current_revision = min(
                    self.max_exploration_calls_per_revision
                    + MAX_DIAGNOSTIC_EXPLORATION_GRANT_PER_REVISION,
                    max(
                        self._exploration_limit_current_revision,
                        self._exploration_calls_current_revision
                        + DIAGNOSTIC_EXPLORATION_CALLS_PER_DISTINCT_CHECK,
                    ),
                )
        if result.status == "passed":
            self._failed_checks_current_revision.pop(normalized_command, None)
        elif (
            result.status == "failed"
            and rendered.get("finalization_blocking") is not False
        ):
            rendered["referenced_created_paths"] = self._referenced_created_paths(
                normalized_command
            )
            self._failed_checks_current_revision[normalized_command] = dict(rendered)
        return rendered

    def _run_python_check(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        script = arguments.get("script")
        if not isinstance(script, str) or not script.strip():
            return {"status": "blocked", "reason": "python_check_script_required"}
        if len(script) > MAX_PYTHON_CHECK_SCRIPT_CHARS:
            return {
                "status": "blocked",
                "reason": (
                    "python_check_script_too_large:"
                    f"{len(script)}>{MAX_PYTHON_CHECK_SCRIPT_CHARS}"
                ),
            }
        executable = _python_check_executable(
            self.executor,
            available_executables=(
                self.available_executables if self.capabilities_observed else None
            ),
        )
        if executable is None:
            return {
                "status": "blocked",
                "reason": "python_runtime_unavailable",
                "available_executables": self.available_executables,
                "failure_classification": "runtime_unavailable",
                "finalization_blocking": False,
                "required_action": (
                    "use_run_check_with_an_available_runtime;"
                    "do_not_revert_product_code_for_this_environment_failure"
                ),
            }
        command = shlex.join((executable, "-c", script))
        result = self._run_check(
            {
                "command": command,
                "timeout_seconds": arguments.get("timeout_seconds"),
            }
        )
        return {**result, "normalized_command": command}

    def _git_diff(self) -> dict[str, Any]:
        repeated_same_revision = self._last_diff_revision == self._revision
        chunks: list[str] = []
        for path in self._candidate.changed_paths():
            before_path = self._candidate.canonical_root / path
            after_path = self.root / path
            before = _safe_diff_lines(
                self._candidate.canonical_root,
                before_path,
            )
            after = _safe_diff_lines(self.root, after_path)
            chunks.extend(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
        rendered = "".join(chunks)
        self._last_diff_revision = self._revision
        finalization_issues = self.finalization_issues()
        ready_to_finalize = not finalization_issues
        if ready_to_finalize and repeated_same_revision:
            next_action = "return_final_json_without_repeating_diff"
        elif ready_to_finalize:
            next_action = "return_final_json"
        else:
            next_action = "resolve_finalization_issues"
        payload = {
            "status": "ok",
            "changed_paths": self._candidate.changed_paths(),
            "diff": rendered[-100_000:],
            "truncated": len(rendered) > 100_000,
            "repeated_same_revision": repeated_same_revision,
            "ready_to_finalize": ready_to_finalize,
            "finalization_issues": finalization_issues,
            "next_action": next_action,
        }
        unwired_paths = self.unwired_created_source_paths
        if unwired_paths:
            payload["call_path_warnings"] = {
                "unwired_created_source_paths": unwired_paths,
                "behavior_failure_count": self.unwired_behavior_failure_count,
                "required_action": (
                    "wire_each_new_source_into_a_reachable_production_callsite_"
                    "or_revert_it"
                ),
            }
        return payload

    def _mark_revision_changed(self) -> None:
        self._revision += 1
        self._mutation_recovery_edit_lease = None
        self._exploration_calls_current_revision = 0
        self._exploration_limit_current_revision = (
            self.max_exploration_calls_per_revision
        )
        self._executed_check_commands_current_revision.clear()
        self._passed_check_commands.clear()
        self._read_coverage_current_revision.clear()
        self._last_check_status = None
        self._last_diff_revision = None

    def _capture_file_revision_snapshot(self, relative: str) -> _FileRevisionSnapshot:
        target = _safe_path(self.root, relative)
        if not target.exists() and not target.is_symlink():
            return _FileRevisionSnapshot(
                applied_revision=self._revision + 1,
                existed=False,
                content="",
                rationale=self._rationales.get(relative),
            )
        if target.is_symlink():
            raise ValueError(f"unsafe_regular_file:{relative}")
        try:
            content = read_regular_file_text(
                self.root,
                target,
                max_bytes=4_000_000,
            )
        except (UnicodeDecodeError, UnsafeRegularFileError):
            raise ValueError(f"unsafe_regular_file:{relative}") from None
        return _FileRevisionSnapshot(
            applied_revision=self._revision + 1,
            existed=True,
            content=content,
            rationale=self._rationales.get(relative),
        )

    def _canonical_file_revision_snapshot(self, relative: str) -> _FileRevisionSnapshot:
        canonical = _safe_path(self._candidate.canonical_root, relative)
        if not canonical.exists() and not canonical.is_symlink():
            return _FileRevisionSnapshot(
                applied_revision=0,
                existed=False,
                content="",
                rationale=None,
            )
        if canonical.is_symlink():
            raise ValueError(f"unsafe_regular_file:{relative}")
        try:
            content = read_regular_file_text(
                self._candidate.canonical_root,
                canonical,
                max_bytes=4_000_000,
            )
        except (UnicodeDecodeError, UnsafeRegularFileError):
            raise ValueError(f"unsafe_regular_file:{relative}") from None
        return _FileRevisionSnapshot(
            applied_revision=0,
            existed=True,
            content=content,
            rationale=None,
        )

    def _check_role(self, normalized_command: str) -> str:
        if normalized_command in self._required_check_commands:
            return "required_acceptance"
        if is_behavior_check_command(normalized_command):
            return "behavior_probe"
        return "diagnostic_probe"

    def _restore_file_revision_history(self, transcript: list[ToolInvocation]) -> None:
        states: dict[str, _FileRevisionSnapshot] = {}
        histories: dict[str, list[_FileRevisionSnapshot]] = {}
        invalid_paths: set[str] = set()
        for invocation in sorted(transcript, key=lambda item: item.index):
            if invocation.name not in {
                "apply_patch",
                "replace_lines",
                "replace_line_ranges",
                "revert_file",
            }:
                continue
            status = str(invocation.result.get("status") or "")
            if status not in {"applied", "reverted"}:
                continue
            relative = str(invocation.arguments.get("path") or "")
            if not relative or relative in invalid_paths:
                continue
            try:
                state = states.setdefault(
                    relative,
                    self._canonical_file_revision_snapshot(relative),
                )
            except ValueError:
                invalid_paths.add(relative)
                continue
            history = histories.setdefault(relative, [])
            if invocation.name == "revert_file":
                states[relative] = (
                    history.pop()
                    if history
                    else self._canonical_file_revision_snapshot(relative)
                )
                continue
            updated = _replay_mutation_content(invocation, state)
            if updated is None:
                invalid_paths.add(relative)
                continue
            history.append(
                _FileRevisionSnapshot(
                    applied_revision=invocation.revision,
                    existed=state.existed,
                    content=state.content,
                    rationale=state.rationale,
                )
            )
            states[relative] = _FileRevisionSnapshot(
                applied_revision=invocation.revision,
                existed=True,
                content=updated,
                rationale=str(invocation.arguments.get("rationale") or "") or None,
            )

        for relative, state in states.items():
            if relative in invalid_paths:
                continue
            try:
                actual = self._capture_file_revision_snapshot(relative)
            except ValueError:
                continue
            if actual.existed != state.existed or actual.content != state.content:
                continue
            history = histories.get(relative, [])
            if history:
                self._file_revision_history[relative] = history
            else:
                self._file_revision_history.pop(relative, None)


def _replay_mutation_content(
    invocation: ToolInvocation,
    state: _FileRevisionSnapshot,
) -> str | None:
    arguments = invocation.arguments
    if invocation.name == "apply_patch":
        operation = str(arguments.get("operation") or "")
        content = str(arguments.get("content") or "")
        old = str(arguments.get("old") or "")
        new = str(arguments.get("new") or "")
        current = state.content if state.existed else ""
        if operation == "create_or_replace":
            return content
        if operation == "replace_fragment":
            if not old or old not in current:
                return None
            return current.replace(old, new, 1)
        if operation == "append_if_missing":
            if content in current:
                return current
            separator = "" if not current or current.endswith("\n") else "\n"
            return current + separator + content
        return None
    if not state.existed:
        return None
    lines = state.content.splitlines(keepends=True)
    if invocation.name == "replace_lines":
        resolved = invocation.result.get("resolved_replacement")
        replacement_arguments = resolved if isinstance(resolved, Mapping) else arguments
        start = int(replacement_arguments.get("start_line") or 0)
        end = int(replacement_arguments.get("end_line") or 0)
        if start < 1 or end < start or end > len(lines):
            return None
        original = "".join(lines[start - 1 : end])
        replacement = _preserve_replaced_line_boundary(
            original,
            str(replacement_arguments.get("new") or ""),
        )
        return "".join(lines[: start - 1]) + replacement + "".join(lines[end:])
    if invocation.name != "replace_line_ranges":
        return None
    resolved_replacements = invocation.result.get("resolved_replacements")
    raw_replacements = (
        resolved_replacements
        if isinstance(resolved_replacements, (list, tuple))
        else arguments.get("replacements")
    )
    if not isinstance(raw_replacements, (list, tuple)) or not raw_replacements:
        return None
    replacements: list[tuple[int, int, str]] = []
    for raw in raw_replacements:
        if not isinstance(raw, Mapping):
            return None
        start = int(raw.get("start_line") or 0)
        end = int(raw.get("end_line") or 0)
        if start < 1 or end < start or end > len(lines):
            return None
        original = "".join(lines[start - 1 : end])
        replacements.append(
            (
                start,
                end,
                _preserve_replaced_line_boundary(
                    original,
                    str(raw.get("new") or ""),
                ),
            )
        )
    replacements.sort(key=lambda item: (item[0], item[1]))
    previous_end = 0
    for start, end, _ in replacements:
        if start <= previous_end:
            return None
        previous_end = end
    rebuilt = list(lines)
    for start, end, replacement in reversed(replacements):
        rebuilt[start - 1 : end] = [replacement]
    return "".join(rebuilt)


def _safe_diff_lines(root: Path, path: Path) -> list[str]:
    if not path.exists() and not path.is_symlink():
        return []
    try:
        return read_regular_file_text(
            root.resolve(),
            path,
            max_bytes=4_000_000,
        ).splitlines(keepends=True)
    except (UnicodeDecodeError, UnsafeRegularFileError) as exc:
        raise ValueError(
            f"unsafe_regular_file:{path.name}:{type(exc).__name__}"
        ) from None


def _normalized_check_command(command: str) -> str:
    try:
        return shlex.join(shlex.split(command))
    except ValueError:
        return command.strip()


def _probe_setup_failure_classification(
    *,
    check_role: str,
    stderr_tail: str,
) -> str | None:
    """Identify failures caused by the agent-authored probe itself."""

    if check_role == "required_acceptance" or not stderr_tail:
        return None
    frames = re.findall(r'File "([^"]+)"(?:, line \d+)?', stderr_tail)
    if not frames or frames[-1] != "<string>":
        return None
    exception_matches = re.findall(
        r"(?m)^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))(?::|$)",
        stderr_tail,
    )
    if not exception_matches:
        return None
    exception_name = exception_matches[-1]
    if exception_name not in _NONBLOCKING_PROBE_SETUP_EXCEPTIONS:
        return None
    return f"probe_setup_error:{exception_name}"


def _diagnostic_tooling_failure_classification(command: str) -> str | None:
    """Keep failed VCS inspection attempts out of product acceptance debt."""

    try:
        semantic = " ".join(shlex.split(command)).lower()
    except ValueError:
        semantic = command.lower()
    direct_vcs_inspection = re.search(r"\bgit\s+(?:diff|show|status)\b", semantic)
    nested_vcs_inspection = re.search(
        r"execfilesync\(\s*['\"]git['\"]\s*,\s*\[\s*['\"]"
        r"(?:diff|show|status)['\"]",
        semantic,
    )
    if direct_vcs_inspection or nested_vcs_inspection:
        return "diagnostic_tooling_observation"
    return None


def _range_is_covered(
    ranges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    start: int,
    end: int,
) -> bool:
    return any(
        covered_start <= start and covered_end >= end
        for covered_start, covered_end in ranges
    )


def _invocation_check_command(invocation: ToolInvocation) -> str:
    if invocation.name == "run_check":
        return str(invocation.arguments.get("command") or "")
    if invocation.name == "run_python_check":
        return str(invocation.result.get("normalized_command") or "")
    return ""


def _python_check_executable(
    executor: CommandExecutor,
    *,
    available_executables: tuple[str, ...] | None = None,
) -> str | None:
    if available_executables is not None:
        return next(
            (
                executable
                for executable in _PYTHON_EXECUTABLES
                if executable in available_executables
            ),
            None,
        )
    policy = getattr(executor, "policy", None)
    return "python" if getattr(policy, "backend", None) == "docker" else sys.executable


def _preserve_replaced_line_boundary(original: str, replacement: str) -> str:
    if replacement and original.endswith("\n") and not replacement.endswith("\n"):
        return replacement + "\n"
    return replacement


def _strip_one_terminal_line_ending(value: str) -> str:
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith(("\n", "\r")):
        return value[:-1]
    return value


def _display_line_range(lines: list[str], start: int, end: int) -> str:
    return "\n".join(
        _strip_one_terminal_line_ending(line) for line in lines[start - 1 : end]
    )


def _resolve_expected_line_range(
    lines: list[str],
    *,
    requested_start: int,
    requested_end: int,
    expected_old: str,
    allowed_start: int,
    allowed_end: int,
) -> tuple[tuple[int, int] | None, tuple[tuple[int, int], ...]]:
    """Resolve a copied source fragment inside one exact observation lease."""

    normalized_expected = "\n".join(
        _strip_one_terminal_line_ending(line)
        for line in expected_old.splitlines(keepends=True)
    )
    if not normalized_expected and expected_old:
        normalized_expected = _strip_one_terminal_line_ending(expected_old)
    expected_line_count = max(1, len(normalized_expected.split("\n")))
    if (
        allowed_start <= requested_start <= requested_end <= allowed_end
        and _display_line_range(lines, requested_start, requested_end)
        == normalized_expected
    ):
        requested = (requested_start, requested_end)
        return requested, (requested,)

    last_start = allowed_end - expected_line_count + 1
    matches = tuple(
        (candidate_start, candidate_start + expected_line_count - 1)
        for candidate_start in range(allowed_start, last_start + 1)
        if _display_line_range(
            lines,
            candidate_start,
            candidate_start + expected_line_count - 1,
        )
        == normalized_expected
    )
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


def _unique_line_range_replacement(
    payload: str,
    lines: list[str],
    *,
    start: int,
    end: int,
    replacement: str,
) -> tuple[str, str]:
    """Bind a line edit to the smallest exact context unique in this file revision."""

    left = start - 1
    right = end
    while True:
        old_fragment = "".join(lines[left:right])
        if payload.find(old_fragment) == payload.rfind(old_fragment):
            new_fragment = (
                "".join(lines[left : start - 1])
                + replacement
                + "".join(lines[end:right])
            )
            return old_fragment, new_fragment
        if left > 0:
            left -= 1
        if right < len(lines):
            right += 1


def _prospective_syntax_issue(
    *,
    root: Path,
    target: Path,
    patch: WorkspaceFilePatch,
) -> str | None:
    prospective = _prospective_patch_content(root=root, target=target, patch=patch)
    if prospective is None:
        return None
    before, after = prospective
    suffix = target.suffix.casefold()
    if suffix in {".py", ".pyi"}:
        if before and _python_syntax_error(before, target.as_posix()) is not None:
            return None
        issue = _python_syntax_error(after, target.as_posix())
        if issue is not None:
            return f"syntax_guard_failed:{patch.path}:python:{issue}"
        decorator_issue = _python_decorator_transfer_error(before, after)
        if decorator_issue is not None:
            return f"semantic_guard_failed:{patch.path}:python:{decorator_issue}"
    elif suffix == ".json":
        if before and _json_syntax_error(before) is not None:
            return None
        issue = _json_syntax_error(after)
        if issue is not None:
            return f"syntax_guard_failed:{patch.path}:json:{issue}"
    elif suffix == ".toml":
        if before and _toml_syntax_error(before) is not None:
            return None
        issue = _toml_syntax_error(after)
        if issue is not None:
            return f"syntax_guard_failed:{patch.path}:toml:{issue}"
    return None


def _prospective_patch_content(
    *,
    root: Path,
    target: Path,
    patch: WorkspaceFilePatch,
) -> tuple[str, str] | None:
    if target.is_symlink() or (target.exists() and not target.is_file()):
        return None
    if target.exists():
        try:
            before = read_regular_file_text(root, target, max_bytes=4_000_000)
        except (UnicodeDecodeError, UnsafeRegularFileError):
            return None
    else:
        before = ""

    if patch.operation == "create_or_replace":
        return before, patch.content
    if patch.operation == "replace_fragment":
        if not patch.old:
            return None
        if patch.old in before:
            return before, before.replace(patch.old, patch.new, 1)
        if patch.new and patch.new in before:
            return before, before
        return None
    if patch.operation == "append_if_missing":
        if patch.content in before:
            return before, before
        separator = "" if not before or before.endswith("\n") else "\n"
        return before, before + separator + patch.content
    return None


def _python_structural_context(
    source: str,
    *,
    start_line: int,
    end_line: int,
) -> dict[str, Any] | None:
    """Summarize the local class contract visible in a Python source range."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    midpoint = (start_line + end_line) // 2
    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and int(node.lineno) <= end_line
        and int(node.end_lineno or node.lineno) >= start_line
    ]
    if not classes:
        return None

    def class_rank(node: ast.ClassDef) -> tuple[int, int, int]:
        node_end = int(node.end_lineno or node.lineno)
        overlap = max(
            0,
            min(end_line, node_end) - max(start_line, int(node.lineno)) + 1,
        )
        contains_midpoint = int(node.lineno) <= midpoint <= node_end
        return (0 if contains_midpoint else 1, -overlap, node_end - int(node.lineno))

    target = min(classes, key=class_rank)
    slots: list[str] = []
    for statement in target.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if not any(
            isinstance(candidate, ast.Name) and candidate.id == "__slots__"
            for candidate in targets
        ):
            continue
        try:
            raw_slots = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            continue
        if isinstance(raw_slots, str):
            slots.append(raw_slots)
        elif isinstance(raw_slots, (list, tuple)):
            slots.extend(str(item) for item in raw_slots if isinstance(item, str))

    instance_attributes: set[str] = set(slots)
    receiver_calls: Counter[str] = Counter()
    method_call_patterns: dict[str, tuple[str, ...]] = {}
    method_types = (ast.AsyncFunctionDef, ast.FunctionDef)
    for method in (node for node in target.body if isinstance(node, method_types)):
        patterns: list[str] = []
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Store)
            ):
                instance_attributes.add(node.attr)
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            receiver = node.func.value
            if (
                isinstance(receiver, ast.Attribute)
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id == "self"
            ):
                pattern = f"self.{receiver.attr}.{node.func.attr}"
                receiver_calls[receiver.attr] += 1
                patterns.append(pattern)
        method_end = int(method.end_lineno or method.lineno)
        if (
            patterns
            and int(method.lineno) <= end_line
            and method_end >= start_line
            and len(method_call_patterns) < 20
        ):
            method_call_patterns[method.name] = tuple(dict.fromkeys(patterns))

    dominant_receiver = receiver_calls.most_common(1)[0][0] if receiver_calls else None
    payload: dict[str, Any] = {
        "language": "python",
        "enclosing_class": target.name,
        "class_line_range": [
            int(target.lineno),
            int(target.end_lineno or target.lineno),
        ],
        "bases": tuple(ast.unparse(base) for base in target.bases),
        "slots": tuple(dict.fromkeys(slots)),
        "instance_attributes": tuple(sorted(instance_attributes)),
        "receiver_call_counts": dict(sorted(receiver_calls.items())),
        "method_call_patterns_in_read_range": method_call_patterns,
    }
    if dominant_receiver is not None:
        payload["dominant_forward_receiver"] = f"self.{dominant_receiver}"
        payload["repair_advisory"] = (
            "Preserve the class's observed receiver/forwarding contract unless a "
            "verified architectural change requires replacing it."
        )
    return payload


def _python_syntax_error(content: str, filename: str) -> str | None:
    try:
        compile(content, filename, "exec", dont_inherit=True)
    except SyntaxError as exc:
        return (
            f"line={int(exc.lineno or 0)}:column={int(exc.offset or 0)}:"
            f"{str(exc.msg)[:160]}"
        )
    return None


def _python_decorator_transfer_error(before: str, after: str) -> str | None:
    if not before:
        return None
    try:
        before_tree = ast.parse(before)
        after_tree = ast.parse(after)
    except SyntaxError:
        return None
    definition_types = (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)
    before_definitions = {
        node.name: node
        for node in before_tree.body
        if isinstance(node, definition_types)
    }
    after_definitions = {
        node.name: node
        for node in after_tree.body
        if isinstance(node, definition_types)
    }
    new_definitions = {
        name: node
        for name, node in after_definitions.items()
        if name not in before_definitions
    }
    if not new_definitions:
        return None
    for existing_name, before_node in before_definitions.items():
        after_node = after_definitions.get(existing_name)
        if after_node is None:
            continue
        before_decorators = {
            ast.dump(item, include_attributes=False)
            for item in before_node.decorator_list
        }
        after_decorators = {
            ast.dump(item, include_attributes=False)
            for item in after_node.decorator_list
        }
        transferred = before_decorators - after_decorators
        if not transferred:
            continue
        for new_name, new_node in new_definitions.items():
            new_decorators = {
                ast.dump(item, include_attributes=False)
                for item in new_node.decorator_list
            }
            if transferred.intersection(new_decorators):
                return (
                    "decorator_stack_split:"
                    f"existing={existing_name}:introduced={new_name}"
                )
    return None


def _json_syntax_error(content: str) -> str | None:
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return f"line={exc.lineno}:column={exc.colno}:{str(exc.msg)[:160]}"
    return None


def _toml_syntax_error(content: str) -> str | None:
    try:
        tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        return str(exc)[:240]
    return None


def _iter_repository_files(base_path: Path, root: Path) -> Iterator[Path]:
    if base_path.is_symlink():
        return
    if base_path.is_file():
        if not _ignored(base_path, root):
            yield base_path
        return
    for current, directories, filenames in os.walk(base_path, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if not (current_path / name).is_symlink()
            and not _ignored(current_path / name, root)
        )
        for name in sorted(filenames):
            path = current_path / name
            if path.is_symlink() or _ignored(path, root):
                continue
            yield path


def coding_tool_definitions(
    *,
    allow_exploration: bool = True,
    allow_verification: bool = True,
    allow_diff: bool = True,
    blocked_tool_names: frozenset[str] = frozenset(),
    available_executables: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    definitions = [
        _function_tool(
            "search",
            "Search UTF-8 repository text using a bounded literal query.",
            {
                "query": {"type": "string"},
                "path": {"type": ["string", "null"]},
                "glob": {"type": ["string", "null"]},
                "max_results": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 80,
                },
            },
            ("query",),
        ),
        _function_tool(
            "list_files",
            "List bounded repository files by relative path and glob.",
            {
                "path": {"type": ["string", "null"]},
                "glob": {"type": ["string", "null"]},
                "max_results": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 80,
                },
            },
            (),
        ),
        _function_tool(
            "read_file",
            (
                "Read a bounded line range from a repository file. Request unseen "
                "ranges; fully covered ranges are rejected until the file revision "
                "changes."
            ),
            {
                "path": {"type": "string"},
                "start_line": {"type": ["integer", "null"], "minimum": 1},
                "end_line": {"type": ["integer", "null"], "minimum": 1},
            },
            ("path",),
        ),
        _function_tool(
            "apply_patch",
            (
                "Apply one guarded structured patch in the disposable candidate "
                "workspace. Use content for create_or_replace or append_if_missing; "
                "use old and new for replace_fragment."
            ),
            {
                "path": {"type": "string"},
                "operation": {
                    "type": "string",
                    "enum": [
                        "create_or_replace",
                        "replace_fragment",
                        "append_if_missing",
                    ],
                },
                "content": {"type": ["string", "null"]},
                "old": {"type": ["string", "null"]},
                "new": {"type": ["string", "null"]},
                "rationale": {"type": ["string", "null"]},
            },
            ("path", "operation"),
        ),
        _function_tool(
            "replace_lines",
            (
                "Replace an exact inclusive line range on a known file revision. Use the "
                "file_hash returned by read_file to avoid stale fragment retries."
            ),
            {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "new": {"type": "string"},
                "expected_file_hash": {"type": "string"},
                "rationale": {"type": ["string", "null"]},
            },
            (
                "path",
                "start_line",
                "end_line",
                "new",
                "expected_file_hash",
            ),
        ),
        _function_tool(
            "replace_line_ranges",
            (
                "Atomically replace multiple non-overlapping inclusive line ranges on "
                "one known file revision. Use one file_hash returned by read_file."
            ),
            {
                "path": {"type": "string"},
                "expected_file_hash": {"type": "string"},
                "replacements": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 64,
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                            "new": {"type": "string"},
                        },
                        "required": ["start_line", "end_line", "new"],
                        "additionalProperties": False,
                    },
                },
                "rationale": {"type": ["string", "null"]},
            },
            ("path", "expected_file_hash", "replacements"),
        ),
        _function_tool(
            "revert_file",
            (
                "Roll back one changed candidate file to the state before its latest "
                "successful edit."
            ),
            {"path": {"type": "string"}},
            ("path",),
        ),
        _function_tool(
            "run_check",
            "Run one bounded verification command through the configured execution backend.",
            {
                "command": {"type": "string"},
                "timeout_seconds": {
                    "type": ["number", "null"],
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            ("command",),
        ),
        _function_tool(
            "run_python_check",
            (
                "Run a multiline Python verification script directly through the "
                "configured execution backend. Prefer this tool to python -c in "
                "run_check when a probe needs quoting, helper functions, or several "
                "assertions; pass raw Python without shell quoting."
            ),
            {
                "script": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PYTHON_CHECK_SCRIPT_CHARS,
                },
                "timeout_seconds": {
                    "type": ["number", "null"],
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            ("script",),
        ),
        _function_tool(
            "git_diff",
            "Inspect the current candidate diff without invoking git.",
            {},
            (),
        ),
    ]
    return [
        definition
        for definition in definitions
        if (allow_exploration or definition["name"] not in _EXPLORATION_TOOL_NAMES)
        and (allow_verification or definition["name"] not in _VERIFICATION_TOOL_NAMES)
        and (allow_diff or definition["name"] != "git_diff")
        and (
            definition["name"] != "run_python_check"
            or available_executables is None
            or any(
                executable in available_executables
                for executable in _PYTHON_EXECUTABLES
            )
        )
        and definition["name"] not in blocked_tool_names
    ]


def _function_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


def tool_result_json(result: Mapping[str, Any]) -> str:
    return json.dumps(result, sort_keys=True, separators=(",", ":"), default=str)


def is_behavior_check_command(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    lower = " ".join(tokens).lower()
    normalized_tokens = tuple(token.lower().replace("\\", "/") for token in tokens)
    if _is_syntax_only_invocation(normalized_tokens, lower):
        return False
    if _is_nonexecuting_test_invocation(normalized_tokens):
        return False
    if any(
        marker in f" {lower} "
        for marker in (
            " assert ",
            "assert ",
            "assert(",
            ".assert(",
            "throw new error",
            "process.exit(1",
            "process.exitcode = 1",
            "process.exitcode=1",
            "sys.exit(1",
            "raise assertionerror",
        )
    ):
        return True
    if _has_javascript_assertion_invocation(lower):
        return True
    if _has_test_runner_invocation(normalized_tokens):
        return True
    for token in normalized_tokens:
        normalized = token.strip("'\"")
        parts = [part for part in normalized.split("/") if part]
        basename = parts[-1] if parts else normalized
        if any(part in {"test", "tests", "__tests__"} for part in parts):
            return True
        if (
            "repro" in basename
            or "smoke" in basename
            or basename.startswith("test_")
            or basename.endswith("_test.py")
            or ".test." in basename
            or ".spec." in basename
        ):
            return True
    return False


def _has_javascript_assertion_invocation(lower_command: str) -> bool:
    if _DIRECT_JAVASCRIPT_ASSERT_CALL_RE.search(lower_command):
        return True
    for binding_pattern in _JAVASCRIPT_ASSERT_BINDING_PATTERNS:
        for match in binding_pattern.finditer(lower_command):
            binding = re.escape(match.group(1))
            invocation = re.compile(
                rf"(?<![a-z0-9_$]){binding}\s*"
                rf"(?:\(|\.\s*[a-z_$][a-z0-9_$]*\s*\()"
            )
            if invocation.search(lower_command, match.end()):
                return True
    return False


def _is_nonexecuting_test_invocation(tokens: tuple[str, ...]) -> bool:
    test_runner_index = _test_runner_index(tokens)
    package_test_index = _package_test_index(tokens)
    if test_runner_index is None and package_test_index is None:
        return False
    no_execution_flags = {
        "-h",
        "--co",
        "--collect-only",
        "--fixtures",
        "--fixtures-per-test",
        "--help",
        "--list-tests",
        "--listtests",
        "--markers",
        "--setup-plan",
        "--show-config",
        "--showconfig",
        "--version",
    }
    relevant = (
        tokens[test_runner_index + 1 :]
        if test_runner_index is not None
        else tokens[package_test_index + 1 :]
    )
    if any(
        token in no_execution_flags
        or any(token.startswith(f"{flag}=") for flag in no_execution_flags)
        for token in relevant
    ):
        return True
    return bool(
        relevant
        and Path(relevant[0]).name in {"list"}
        and test_runner_index is not None
        and Path(tokens[test_runner_index]).name == "vitest"
    )


def _has_test_runner_invocation(tokens: tuple[str, ...]) -> bool:
    return (
        _test_runner_index(tokens) is not None
        or _package_test_index(tokens) is not None
    )


def _test_runner_index(tokens: tuple[str, ...]) -> int | None:
    runner_names = {
        "ava",
        "jest",
        "mocha",
        "nose",
        "nose2",
        "py.test",
        "pytest",
        "tap",
        "tox",
        "unittest",
        "vitest",
    }
    for index, token in enumerate(tokens):
        if Path(token).name in runner_names:
            return index
        if Path(token).name == "node" and "--test" in tokens[index + 1 :]:
            return index
    return None


def _package_test_index(tokens: tuple[str, ...]) -> int | None:
    package_runners = {"npm", "pnpm", "yarn"}
    for index, token in enumerate(tokens):
        if Path(token).name not in package_runners:
            continue
        following = tokens[index + 1 :]
        if following and following[0] == "test":
            return index
        if (
            len(following) >= 2
            and following[0] == "run"
            and following[1].startswith("test")
        ):
            return index
    return None


def _is_syntax_only_invocation(tokens: tuple[str, ...], command: str) -> bool:
    return (
        (bool(tokens) and Path(tokens[0]).name == "node" and "--check" in tokens)
        or "py_compile" in command
        or "compileall" in command
        or "json.tool" in command
        or "tsc --noemit" in command
    )


def is_declaration_path(path: str) -> bool:
    lower = Path(path).as_posix().lower()
    return lower.endswith((".d.ts", ".d.mts", ".d.cts", ".pyi"))


def command_covers_declaration(command: str, path: str) -> bool:
    lower = command.lower().replace("\\", "/")
    normalized_path = Path(path).as_posix().lower()
    name = Path(normalized_path).name
    if normalized_path in lower or name in lower:
        return True
    if normalized_path.endswith((".d.ts", ".d.mts", ".d.cts")):
        return any(marker in lower for marker in ("tsc ", 'tsc"', "tsc'"))
    return any(marker in lower for marker in ("mypy", "pyright", "pyre"))


def _requires_behavior_check(paths: tuple[str, ...]) -> bool:
    source_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".java",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".swift",
        ".ts",
        ".tsx",
        ".vue",
    }
    config_names = {"package.json", "pyproject.toml", "setup.cfg", "tox.ini"}
    for raw_path in paths:
        path = Path(raw_path)
        normalized = path.as_posix().lower()
        if any(part in {"test", "tests", "__tests__", "docs"} for part in path.parts):
            continue
        if path.suffix.lower() in source_suffixes or path.name.lower() in config_names:
            return True
        if normalized.startswith(("src/", "lib/", "app/")):
            return True
    return False


def _safe_path(root: Path, relative_path: str) -> Path:
    if not relative_path:
        raise ValueError("unsafe_path:")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe_path:{relative_path}")
    target = (root / path).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"unsafe_path:{relative_path}") from None
    return target


def _ignored(path: Path, root: Path) -> bool:
    return ignored_repository_path(path.relative_to(root))
