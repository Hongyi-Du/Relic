"""Company-side coding agent and workspace execution boundary."""

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
from difflib import SequenceMatcher
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from time import sleep
from types import FrameType

from .execution import CommandExecutor
from .hashing import canonicalize, stable_hash
from .openai_runtime import (
    build_openai_client,
    prepare_openai_response_request,
)
from .org_adapter import (
    CREATIVE_FEATURE_THEMES,
    CompanyOptimizationProposal,
    CompanyPublicFeedbackReport,
    CompanyVisibleTrace,
)
from .workspace_transaction import CandidateWorkspace
from .workspace_update import WorkspaceFilePatch, WorkspaceVerificationResult
from .verification_policy import VERIFICATION_EXECUTABLES

DEFAULT_COMPANY_CODING_MODEL = "gpt-5.5"
DEFAULT_COMPANY_CODING_REASONING_EFFORT = "xhigh"


@dataclass(frozen=True)
class ProjectWorkspaceManifest:
    root_name: str
    file_count: int
    files: tuple[str, ...]
    manifest_hash: str


@dataclass(frozen=True)
class DevelopmentChangeUnit:
    unit_id: str
    theme: str
    objective: str
    target_paths: tuple[str, ...]
    support_refs: tuple[str, ...]
    validation_commands: tuple[str, ...]
    scale_score: float


@dataclass(frozen=True)
class ProjectEdit:
    path: str
    operation: str
    content: str
    rationale: str
    support_refs: tuple[str, ...]


@dataclass(frozen=True)
class ProjectVerificationResult:
    command: str
    status: str
    exit_code: int | None
    elapsed_sec: float
    stdout_tail: str = ""
    stderr_tail: str = ""
    blocked_reason: str | None = None
    backend: str = ""
    stdout_hash: str | None = None
    stderr_hash: str | None = None
    runtime_ref: str | None = None


@dataclass(frozen=True)
class CompanyDevelopmentRun:
    run_id: str
    artifact_id: str
    proposal_id: str
    source: str
    model: str | None
    mode: str
    change_units: tuple[DevelopmentChangeUnit, ...]
    edits: tuple[ProjectEdit, ...]
    verification_commands: tuple[str, ...]
    evidence_hash: str
    workspace_manifest_hash: str | None = None
    patch_set_hash: str | None = None
    private_state_boundary: str = "public_traces_and_project_files_only"
    applied_paths: tuple[str, ...] = ()
    blocked_reason: str | None = None
    validation_issues: tuple[str, ...] = ()
    verification_results: tuple[ProjectVerificationResult, ...] = ()


class HeuristicCompanyCodingAgent:
    source = "heuristic_company_coding_agent_v17"

    def develop(
        self,
        *,
        proposal: CompanyOptimizationProposal,
        report: CompanyPublicFeedbackReport,
        public_traces: tuple[CompanyVisibleTrace, ...] = (),
        workspace_manifest: ProjectWorkspaceManifest | None = None,
    ) -> CompanyDevelopmentRun:
        allowed_themes = _allowed_development_themes(proposal)
        if not allowed_themes:
            return _deferred_development_run(
                proposal=proposal,
                report=report,
                source=self.source,
                model=None,
                public_traces=public_traces,
                workspace_manifest=workspace_manifest,
            )
        units = tuple(
            _change_unit_for_theme(
                theme=theme,
                proposal=proposal,
                index=index,
            )
            for index, theme in enumerate(allowed_themes[:8])
        )
        edits = tuple(edit for unit in units for edit in _edits_for_unit(unit))
        validation_commands = _dedupe(
            command for unit in units for command in unit.validation_commands
        )
        validation_commands = _dedupe(
            (*validation_commands, *_generated_test_commands(edits))
        )
        run_id = _development_run_id(
            proposal=proposal,
            report=report,
            source=self.source,
            model=None,
            change_units=units,
            edits=edits,
            workspace_manifest=workspace_manifest,
        )
        return CompanyDevelopmentRun(
            run_id=run_id,
            artifact_id=proposal.artifact_id,
            proposal_id=proposal.proposal_id,
            source=self.source,
            model=None,
            mode="plan",
            change_units=units,
            edits=edits,
            verification_commands=validation_commands,
            evidence_hash=stable_hash((proposal, report, public_traces[:80])),
            workspace_manifest_hash=workspace_manifest.manifest_hash
            if workspace_manifest
            else None,
            patch_set_hash=stable_hash(edits),
        )


class OpenAICompanyCodingAgent:
    source = "openai_company_coding_agent_v17"

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_seconds: float = 300.0,
        strict_live: bool = False,
        request_attempts: int = 2,
        retry_backoff_seconds: float = 0.0,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAICompanyCodingAgent"
            )
        if request_attempts < 1:
            raise ValueError("company_coding_request_attempts_must_be_positive")
        if retry_backoff_seconds < 0:
            raise ValueError("company_coding_retry_backoff_must_be_nonnegative")
        self.model = (
            model
            or os.environ.get("SOCIETY_CORE_COMPANY_CODING_MODEL")
            or os.environ.get("SOCIETY_CORE_COMPANY_OPTIMIZER_MODEL")
            or DEFAULT_COMPANY_CODING_MODEL
        )
        requested_reasoning_effort = os.environ.get(
            "SOCIETY_CORE_COMPANY_CODING_REASONING_EFFORT",
            DEFAULT_COMPANY_CODING_REASONING_EFFORT,
        )
        self.requested_reasoning_effort = requested_reasoning_effort
        self.reasoning_effort = _normalized_company_coding_reasoning_effort(
            requested_reasoning_effort,
            model=self.model,
        )
        self.timeout_seconds = timeout_seconds
        self.strict_live = strict_live
        self.request_attempts = request_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self._client = build_openai_client(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=0,
        )
        self._fallback = HeuristicCompanyCodingAgent()

    def develop(
        self,
        *,
        proposal: CompanyOptimizationProposal,
        report: CompanyPublicFeedbackReport,
        public_traces: tuple[CompanyVisibleTrace, ...] = (),
        workspace_manifest: ProjectWorkspaceManifest | None = None,
    ) -> CompanyDevelopmentRun:
        fallback = self._fallback.develop(
            proposal=proposal,
            report=report,
            public_traces=public_traces,
            workspace_manifest=workspace_manifest,
        )
        allowed_themes = _allowed_development_themes(proposal)
        if not allowed_themes:
            return replace(
                fallback,
                source=f"{self.source}:kernel_deferred",
                model=self.model,
            )
        payload = _development_payload(
            proposal=proposal,
            report=report,
            public_traces=public_traces,
            workspace_manifest=workspace_manifest,
        )
        prompt = (
            "You are the company's strongest internal coding agent.\n"
            "Convert public product feedback into concrete project edits.\n"
            "Use only the provided public traces, proposal, and workspace manifest.\n"
            "Do not invent private user data or hidden company facts.\n"
            "Create change units only for allowed_development_themes in the input.\n"
            "Prefer substantial, multi-file changes when the evidence supports them.\n"
            "Target source, test, or documentation paths only when public evidence and the workspace manifest justify them.\n"
            "All edits will be applied later by a guarded sandbox policy; do not assume direct write access.\n"
            "Return only JSON matching the schema.\n\n"
            f"INPUT_JSON:\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
        )
        request = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": 16000,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "company_development_run",
                    "strict": True,
                    "schema": _development_json_schema(allowed_themes=allowed_themes),
                }
            },
            "timeout": self.timeout_seconds,
        }
        if not self.model.startswith("gpt-5"):
            request["temperature"] = 0
        if self.model.startswith("gpt-5"):
            request["reasoning"] = {"effort": self.reasoning_effort}
        last_error: Exception | None = None
        last_failure_kind = "openai_live_error"
        for attempt in range(self.request_attempts):
            attempt_request = dict(request)
            if attempt:
                attempt_request["input"] = (
                    prompt
                    + "\n\nYour previous response could not be parsed or validated. "
                    "Regenerate the complete answer from the original input. Return one "
                    "complete JSON object matching the strict schema and no other text."
                )
            try:
                with _wall_clock_timeout(self.timeout_seconds):
                    response = self._client.responses.create(
                        **prepare_openai_response_request(attempt_request)
                    )
            except Exception as exc:  # pragma: no cover - live network fallback.
                last_error = exc
                last_failure_kind = "openai_live_error"
                if attempt + 1 < self.request_attempts:
                    sleep(self.retry_backoff_seconds * (2**attempt))
                continue
            try:
                parsed = _parse_json_object(_response_text(response))
                units = tuple(
                    _unit_from_json(
                        item,
                        proposal,
                        index,
                        allowed_themes=allowed_themes,
                    )
                    for index, item in enumerate(parsed["change_units"])
                )
                if not units:
                    raise ValueError("Company coding agent returned no change units")
                edits = tuple(
                    _edit_from_json(item, proposal) for item in parsed["edits"]
                )
                verification_commands = _dedupe(
                    str(command)[:240] for command in parsed["verification_commands"]
                )
                break
            except Exception as exc:  # pragma: no cover - malformed live output.
                last_error = exc
                last_failure_kind = "openai_parse_error"
                if attempt + 1 < self.request_attempts:
                    sleep(self.retry_backoff_seconds * (2**attempt))
        else:
            assert last_error is not None
            failure = (
                f"{last_failure_kind}:{type(last_error).__name__}:"
                f"attempts={self.request_attempts}:"
                f"{str(last_error)[:120]}"
            )
            if self.strict_live:
                raise RuntimeError(
                    f"live_company_coding_request_exhausted:{failure}"
                ) from last_error
            return replace(
                fallback,
                source=f"{self.source}:fallback_heuristic",
                model=self.model,
                blocked_reason=failure,
            )
        run_id = _development_run_id(
            proposal=proposal,
            report=report,
            source=self.source,
            model=self.model,
            change_units=units,
            edits=edits,
            workspace_manifest=workspace_manifest,
        )
        return CompanyDevelopmentRun(
            run_id=run_id,
            artifact_id=proposal.artifact_id,
            proposal_id=proposal.proposal_id,
            source=self.source,
            model=self.model,
            mode="plan",
            change_units=units,
            edits=edits,
            verification_commands=verification_commands,
            evidence_hash=stable_hash((proposal, report, public_traces[:80])),
            workspace_manifest_hash=workspace_manifest.manifest_hash
            if workspace_manifest
            else None,
            patch_set_hash=stable_hash(edits),
        )


def build_company_coding_agent(
    *,
    provider: str = "heuristic",
    model: str | None = None,
    timeout_seconds: float = 300.0,
    strict_live: bool = False,
    request_attempts: int = 2,
    retry_backoff_seconds: float = 0.0,
):
    if provider == "heuristic":
        return HeuristicCompanyCodingAgent()
    if provider == "openai":
        return OpenAICompanyCodingAgent(
            model=model,
            timeout_seconds=timeout_seconds,
            strict_live=strict_live,
            request_attempts=request_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    raise ValueError(f"Unsupported company coding agent provider: {provider}")


def materialize_development_code_edits(
    run: CompanyDevelopmentRun,
) -> CompanyDevelopmentRun:
    """Ensure a development plan has source and executable test edits."""

    existing_paths = {edit.path for edit in run.edits}
    additional_edits: list[ProjectEdit] = []
    for unit in run.change_units:
        for edit in _edits_for_unit(unit):
            if edit.path not in existing_paths:
                additional_edits.append(edit)
                existing_paths.add(edit.path)
    if not additional_edits:
        edits = run.edits
    else:
        edits = (*run.edits, *additional_edits)
    verification_commands = _dedupe(
        (*run.verification_commands, *_generated_test_commands(edits))
    )
    if edits == run.edits and verification_commands == run.verification_commands:
        return run
    return replace(
        run,
        edits=edits,
        verification_commands=verification_commands,
        patch_set_hash=stable_hash(edits),
    )


def build_workspace_manifest(
    workspace_root: Path,
    *,
    max_files: int = 300,
) -> ProjectWorkspaceManifest:
    root = workspace_root.resolve()
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_ignored_path(path, root):
            continue
        files.append(path.relative_to(root).as_posix())
        if len(files) >= max_files:
            break
    manifest_hash = stable_hash({"root_name": root.name, "files": files})
    return ProjectWorkspaceManifest(
        root_name=root.name,
        file_count=len(files),
        files=tuple(files),
        manifest_hash=manifest_hash,
    )


def apply_project_edits(
    run: CompanyDevelopmentRun,
    workspace_root: Path,
    *,
    dry_run: bool = False,
    allow_replace_existing: bool = False,
    guarded_replacement: bool = True,
    max_edits: int = 24,
    max_total_bytes: int = 250_000,
    max_replacement_deletion_ratio: float = 1.01,
    max_replacement_deleted_lines: int = 1_000_000,
    reject_suspicious_code_unicode: bool = True,
) -> CompanyDevelopmentRun:
    root = workspace_root.resolve()
    if len(run.edits) > max_edits:
        return replace(run, mode="blocked", blocked_reason="too_many_project_edits")
    total_bytes = sum(len(edit.content.encode("utf-8")) for edit in run.edits)
    if total_bytes > max_total_bytes:
        return replace(run, mode="blocked", blocked_reason="project_edits_too_large")
    validation_issues = _validate_project_edits(
        root=root,
        edits=run.edits,
        allow_replace_existing=allow_replace_existing,
        max_replacement_deletion_ratio=max_replacement_deletion_ratio,
        max_replacement_deleted_lines=max_replacement_deleted_lines,
        reject_suspicious_code_unicode=reject_suspicious_code_unicode,
    )
    if guarded_replacement and validation_issues:
        return replace(
            run,
            mode="blocked",
            blocked_reason=f"project_edit_validation_failed:{validation_issues[0]}",
            validation_issues=validation_issues,
        )
    applied: list[str] = []
    for edit in run.edits:
        target = _safe_target_path(root, edit.path)
        operation = _normalize_operation(edit.operation)
        if operation not in {"create_or_replace", "append"}:
            return replace(
                run,
                mode="blocked",
                blocked_reason=f"unsupported_edit_operation:{edit.operation}",
            )
        if (
            operation == "create_or_replace"
            and target.exists()
            and not allow_replace_existing
        ):
            return replace(
                run,
                mode="blocked",
                blocked_reason=f"would_replace_existing_file:{target.relative_to(root).as_posix()}",
            )
        applied.append(target.relative_to(root).as_posix())
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if operation == "append" and target.exists():
            existing = target.read_text(encoding="utf-8")
            separator = "" if existing.endswith("\n") else "\n"
            target.write_text(existing + separator + edit.content, encoding="utf-8")
        else:
            target.write_text(edit.content, encoding="utf-8")
    return replace(
        run,
        mode="dry_run" if dry_run else "applied",
        applied_paths=tuple(applied),
        validation_issues=validation_issues,
    )


def apply_and_verify_project_edits(
    run: CompanyDevelopmentRun,
    workspace_root: Path,
    *,
    allow_replace_existing: bool = False,
    guarded_replacement: bool = True,
    max_edits: int = 24,
    max_total_bytes: int = 250_000,
    max_replacement_deletion_ratio: float = 1.01,
    max_replacement_deleted_lines: int = 1_000_000,
    reject_suspicious_code_unicode: bool = True,
    timeout_seconds: float = 120.0,
    executor: CommandExecutor | None = None,
) -> CompanyDevelopmentRun:
    """Apply, verify, and promote legacy edits as one rollback-safe transaction."""

    root = workspace_root.resolve()
    if executor is None:
        return replace(
            run,
            mode="blocked",
            blocked_reason="verification_executor_required",
        )
    command_executor = executor
    validated = apply_project_edits(
        run,
        root,
        dry_run=True,
        allow_replace_existing=allow_replace_existing,
        guarded_replacement=guarded_replacement,
        max_edits=max_edits,
        max_total_bytes=max_total_bytes,
        max_replacement_deletion_ratio=max_replacement_deletion_ratio,
        max_replacement_deleted_lines=max_replacement_deleted_lines,
        reject_suspicious_code_unicode=reject_suspicious_code_unicode,
    )
    if validated.mode == "blocked":
        return validated
    if not run.verification_commands:
        return replace(
            run,
            mode="blocked",
            blocked_reason="transactional_promotion_requires_verification",
        )

    with CandidateWorkspace.create(root) as candidate:
        for index, edit in enumerate(run.edits):
            operation = _normalize_operation(edit.operation)
            target = _safe_target_path(candidate.root.resolve(), edit.path)
            content = edit.content
            if operation == "append":
                existing = target.read_text(encoding="utf-8") if target.exists() else ""
                separator = "" if not existing or existing.endswith("\n") else "\n"
                content = existing + separator + edit.content
            (patch_result,) = candidate.apply(
                (
                    WorkspaceFilePatch(
                        patch_id=f"legacy_edit_{index}_{stable_hash(edit)[:12]}",
                        path=edit.path,
                        operation="create_or_replace",
                        content=content,
                        rationale=edit.rationale,
                    ),
                )
            )
            if patch_result.status == "blocked":
                return replace(
                    run,
                    mode="blocked",
                    blocked_reason=patch_result.blocked_reason,
                    validation_issues=validated.validation_issues,
                )
        promotion = candidate.promote(
            verification_commands=run.verification_commands,
            timeout_seconds=timeout_seconds,
            executor=command_executor,
        )

    verification_results = tuple(
        _project_verification_result(item) for item in promotion.verification_results
    )
    if promotion.status == "release_candidate":
        return replace(
            run,
            mode="verified",
            applied_paths=promotion.changed_paths,
            blocked_reason=None,
            validation_issues=validated.validation_issues,
            verification_results=verification_results,
        )
    mode = (
        "verification_failed"
        if promotion.status == "verification_failed"
        else "blocked"
    )
    return replace(
        run,
        mode=mode,
        applied_paths=(),
        blocked_reason=promotion.blocked_reason,
        validation_issues=validated.validation_issues,
        verification_results=verification_results,
    )


def _project_verification_result(
    result: WorkspaceVerificationResult,
) -> ProjectVerificationResult:
    return ProjectVerificationResult(
        command=result.command,
        status=result.status,
        exit_code=result.exit_code,
        elapsed_sec=result.elapsed_sec,
        stdout_tail=result.stdout_tail,
        stderr_tail=result.stderr_tail,
        blocked_reason=result.blocked_reason,
        backend=result.backend,
        stdout_hash=result.stdout_hash,
        stderr_hash=result.stderr_hash,
        runtime_ref=result.runtime_ref,
    )


def run_project_verification(
    run: CompanyDevelopmentRun,
    workspace_root: Path,
    *,
    timeout_seconds: float = 120.0,
    max_commands: int = 24,
    allowed_executables: tuple[str, ...] = VERIFICATION_EXECUTABLES,
    executor: CommandExecutor | None = None,
) -> CompanyDevelopmentRun:
    if run.mode != "applied":
        return replace(
            run,
            verification_results=(
                ProjectVerificationResult(
                    command="",
                    status="skipped",
                    exit_code=None,
                    elapsed_sec=0.0,
                    blocked_reason=f"run_mode_not_applied:{run.mode}",
                ),
            ),
        )
    root = workspace_root.resolve()
    unique_commands = _dedupe(run.verification_commands)
    if not unique_commands:
        return replace(
            run,
            mode="verification_failed",
            blocked_reason="verification_commands_required",
            verification_results=(
                ProjectVerificationResult(
                    command="verification_command_set",
                    status="blocked",
                    exit_code=None,
                    elapsed_sec=0.0,
                    blocked_reason="verification_commands_required",
                ),
            ),
        )
    if len(unique_commands) > max_commands:
        reason = (
            f"verification_command_limit_exceeded:{len(unique_commands)}>{max_commands}"
        )
        return replace(
            run,
            mode="verification_failed",
            blocked_reason=reason,
            verification_results=(
                ProjectVerificationResult(
                    command="verification_command_set",
                    status="blocked",
                    exit_code=None,
                    elapsed_sec=0.0,
                    blocked_reason=reason,
                ),
            ),
        )
    if executor is None:
        return replace(
            run,
            mode="verification_failed",
            blocked_reason="verification_executor_required",
            verification_results=(
                ProjectVerificationResult(
                    command="verification_command_set",
                    status="blocked",
                    exit_code=None,
                    elapsed_sec=0.0,
                    blocked_reason="verification_executor_required",
                ),
            ),
        )
    command_executor = executor
    results: list[ProjectVerificationResult] = []
    for command in unique_commands:
        results.append(
            _run_single_verification_command(
                command=command,
                workspace_root=root,
                timeout_seconds=timeout_seconds,
                allowed_executables=allowed_executables,
                executor=command_executor,
            )
        )
    if all(result.status == "passed" for result in results):
        mode = "verified"
    else:
        mode = "verification_failed"
    return replace(run, mode=mode, verification_results=tuple(results))


def _run_single_verification_command(
    *,
    command: str,
    workspace_root: Path,
    timeout_seconds: float,
    allowed_executables: tuple[str, ...],
    executor: CommandExecutor,
) -> ProjectVerificationResult:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return ProjectVerificationResult(
            command=command,
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            blocked_reason=f"invalid_command:{type(exc).__name__}",
        )
    if not argv:
        return ProjectVerificationResult(
            command=command,
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            blocked_reason="empty_command",
        )
    executable = Path(argv[0]).name
    if executable not in allowed_executables and argv[0] != sys.executable:
        return ProjectVerificationResult(
            command=command,
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            blocked_reason=f"unsupported_executable:{executable}",
        )
    outcome = executor.run(
        root=workspace_root,
        argv=tuple(argv),
        timeout_seconds=timeout_seconds,
    )
    return ProjectVerificationResult(
        command=command,
        status=outcome.status,
        exit_code=outcome.exit_code,
        elapsed_sec=outcome.elapsed_sec,
        stdout_tail=outcome.stdout_tail,
        stderr_tail=outcome.stderr_tail,
        blocked_reason=outcome.blocked_reason,
        backend=outcome.backend,
        stdout_hash=outcome.stdout_hash,
        stderr_hash=outcome.stderr_hash,
        runtime_ref=outcome.runtime_ref,
    )


def _tail_text(value: bytes | str, *, max_chars: int = 2000) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-max_chars:]


def _validate_project_edits(
    *,
    root: Path,
    edits: tuple[ProjectEdit, ...],
    allow_replace_existing: bool,
    max_replacement_deletion_ratio: float,
    max_replacement_deleted_lines: int,
    reject_suspicious_code_unicode: bool,
) -> tuple[str, ...]:
    issues: list[str] = []
    for edit in edits:
        target = _safe_target_path(root, edit.path)
        relative_path = target.relative_to(root).as_posix()
        operation = _normalize_operation(edit.operation)
        if operation not in {"create_or_replace", "append"}:
            issues.append(
                f"unsupported_edit_operation:{relative_path}:{edit.operation}"
            )
            continue

        existing_content = target.read_text(encoding="utf-8") if target.exists() else ""
        if operation == "create_or_replace" and target.exists():
            if not allow_replace_existing:
                issues.append(f"would_replace_existing_file:{relative_path}")
            else:
                deleted_lines, added_lines, deletion_ratio = _replacement_diff_stats(
                    existing_content,
                    edit.content,
                )
                if (
                    deleted_lines > max_replacement_deleted_lines
                    or deletion_ratio > max_replacement_deletion_ratio
                ):
                    issues.append(
                        "large_replacement_deletion:"
                        f"{relative_path}:deleted={deleted_lines}:added={added_lines}:"
                        f"ratio={deletion_ratio:.3f}"
                    )

        final_content = (
            existing_content
            + ("\n" if existing_content and not existing_content.endswith("\n") else "")
            + edit.content
            if operation == "append" and target.exists()
            else edit.content
        )
        if reject_suspicious_code_unicode:
            issues.extend(_suspicious_unicode_issues(relative_path, final_content))
    return tuple(issues)


def _normalize_operation(operation: str) -> str:
    cleaned = _safe_slug(operation)
    if cleaned in {"create", "replace", "write", "overwrite", "create_or_replace"}:
        return "create_or_replace"
    if cleaned in {"append", "add"}:
        return "append"
    return cleaned


def _replacement_diff_stats(old: str, new: str) -> tuple[int, int, float]:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    deleted = 0
    added = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=old_lines, b=new_lines).get_opcodes():
        if tag in {"delete", "replace"}:
            deleted += i2 - i1
        if tag in {"insert", "replace"}:
            added += j2 - j1
    denominator = max(1, len(old_lines))
    return deleted, added, deleted / denominator


def _suspicious_unicode_issues(relative_path: str, content: str) -> tuple[str, ...]:
    if not _is_code_like_path(relative_path):
        return ()
    for line_number, line in enumerate(content.splitlines(), start=1):
        for char in line:
            if _is_suspicious_code_unicode(char):
                codepoint = f"U+{ord(char):04X}"
                return (
                    f"suspicious_code_unicode:{relative_path}:{line_number}:{codepoint}",
                )
    return ()


def _is_code_like_path(relative_path: str) -> bool:
    suffix = Path(relative_path).suffix.lower()
    return suffix in {
        ".cjs",
        ".cts",
        ".js",
        ".jsx",
        ".json",
        ".mjs",
        ".mts",
        ".py",
        ".sh",
        ".ts",
        ".tsx",
        ".yaml",
        ".yml",
    }


def _is_suspicious_code_unicode(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x200B <= codepoint <= 0x200F
        or 0x202A <= codepoint <= 0x202E
        or 0x2066 <= codepoint <= 0x2069
        or 0xFF00 <= codepoint <= 0xFFEF
    )


def _allowed_development_themes(
    proposal: CompanyOptimizationProposal,
) -> tuple[str, ...]:
    if proposal.theme_decisions:
        explicitly_adopted = {
            decision.theme
            for decision in proposal.theme_decisions
            if decision.status == "adopt"
        }
        product_themes = tuple(
            theme for theme in proposal.priority_themes if theme in explicitly_adopted
        )
    else:
        product_themes = proposal.priority_themes
    return tuple(dict.fromkeys((*product_themes, *proposal.safety_obligation_themes)))


def _theme_support_refs_for_development(
    proposal: CompanyOptimizationProposal,
    theme: str,
) -> tuple[str, ...]:
    for decision in proposal.theme_decisions:
        if decision.theme == theme:
            return decision.support_refs[:8]
    if theme in proposal.safety_obligation_themes:
        return proposal.support_refs[:8]
    return ()


def _deferred_development_run(
    *,
    proposal: CompanyOptimizationProposal,
    report: CompanyPublicFeedbackReport,
    source: str,
    model: str | None,
    public_traces: tuple[CompanyVisibleTrace, ...],
    workspace_manifest: ProjectWorkspaceManifest | None,
) -> CompanyDevelopmentRun:
    run_id = _development_run_id(
        proposal=proposal,
        report=report,
        source=source,
        model=model,
        change_units=(),
        edits=(),
        workspace_manifest=workspace_manifest,
    )
    return CompanyDevelopmentRun(
        run_id=run_id,
        artifact_id=proposal.artifact_id,
        proposal_id=proposal.proposal_id,
        source=source,
        model=model,
        mode="deferred",
        change_units=(),
        edits=(),
        verification_commands=(),
        evidence_hash=stable_hash((proposal, report, public_traces[:80])),
        workspace_manifest_hash=(
            workspace_manifest.manifest_hash if workspace_manifest else None
        ),
        patch_set_hash=stable_hash(()),
        blocked_reason="no_adopted_theme_or_safety_obligation",
    )


def _change_unit_for_theme(
    *,
    theme: str,
    proposal: CompanyOptimizationProposal,
    index: int,
) -> DevelopmentChangeUnit:
    support_refs = _theme_support_refs_for_development(proposal, theme)
    target_paths = _target_paths_for_theme(theme)
    unit_hash = stable_hash((proposal.proposal_id, theme, index, target_paths))[:12]
    return DevelopmentChangeUnit(
        unit_id=f"dev_unit_{index:02d}_{unit_hash}",
        theme=theme,
        objective=_objective_for_theme(theme),
        target_paths=target_paths,
        support_refs=support_refs,
        validation_commands=_validation_for_theme(theme),
        scale_score=_scale_for_theme(theme),
    )


def _edits_for_unit(unit: DevelopmentChangeUnit) -> tuple[ProjectEdit, ...]:
    slug = _safe_slug(unit.theme) or "product_improvement"
    doc_path = next(
        (path for path in unit.target_paths if Path(path).suffix.lower() == ".md"),
        f"docs/company-development/{slug}.md",
    )
    source_path = f"src/company-development/{slug}.mjs"
    test_path = f"tests/company-development/{slug}.test.mjs"
    return (
        _documentation_edit_for_unit(unit, doc_path),
        _source_edit_for_unit(unit, source_path),
        _test_edit_for_unit(unit, source_path, test_path),
    )


def _documentation_edit_for_unit(unit: DevelopmentChangeUnit, path: str) -> ProjectEdit:
    content = (
        f"# {unit.theme}\n\n"
        f"Objective: {unit.objective}\n\n"
        "Evidence refs:\n" + "\n".join(f"- {ref}" for ref in unit.support_refs) + "\n\n"
        "Implementation notes:\n"
        "- Convert public feedback into code, tests, and documentation changes.\n"
        "- Keep product behavior measurable and regression-tested.\n"
    )
    return ProjectEdit(
        path=path,
        operation="create_or_replace",
        content=content,
        rationale=f"Implement feedback-driven work for {unit.theme}.",
        support_refs=unit.support_refs,
    )


def _source_edit_for_unit(unit: DevelopmentChangeUnit, path: str) -> ProjectEdit:
    payload = {
        "unitId": unit.unit_id,
        "theme": unit.theme,
        "objective": unit.objective,
        "scaleScore": unit.scale_score,
        "supportRefs": unit.support_refs,
    }
    content = (
        f"export const developmentUnit = Object.freeze({json.dumps(canonicalize(payload), indent=2)});\n\n"
        "function clamp01(value) {\n"
        "  if (!Number.isFinite(value)) return 0;\n"
        "  return Math.max(0, Math.min(1, value));\n"
        "}\n\n"
        "export function evaluateUpgradeSignal(signal = {}) {\n"
        "  const publicPriority = clamp01(Number(signal.publicPriority ?? 0));\n"
        "  const failureRate = clamp01(Number(signal.failureRate ?? 0));\n"
        "  const adoptionFriction = clamp01(Number(signal.adoptionFriction ?? 0));\n"
        "  const evidenceWeight = clamp01((publicPriority + failureRate + adoptionFriction) / 3);\n"
        "  return clamp01(developmentUnit.scaleScore * 0.6 + evidenceWeight * 0.4);\n"
        "}\n\n"
        "export function shouldPrioritizeUpgrade(signal = {}) {\n"
        "  return evaluateUpgradeSignal(signal) >= 0.55;\n"
        "}\n"
    )
    return ProjectEdit(
        path=path,
        operation="create_or_replace",
        content=content,
        rationale=f"Add executable product-improvement logic for {unit.theme}.",
        support_refs=unit.support_refs,
    )


def _test_edit_for_unit(
    unit: DevelopmentChangeUnit,
    source_path: str,
    test_path: str,
) -> ProjectEdit:
    relative_import = _relative_import_path(test_path, source_path)
    content = (
        "import assert from 'node:assert/strict';\n"
        f"import {{ developmentUnit, evaluateUpgradeSignal, shouldPrioritizeUpgrade }} from '{relative_import}';\n\n"
        f"assert.equal(developmentUnit.unitId, {json.dumps(unit.unit_id)});\n"
        f"assert.equal(developmentUnit.theme, {json.dumps(unit.theme)});\n"
        "assert.ok(developmentUnit.supportRefs.length >= 1);\n\n"
        "const weakSignal = evaluateUpgradeSignal({ publicPriority: 0, failureRate: 0, adoptionFriction: 0 });\n"
        "const strongSignal = evaluateUpgradeSignal({ publicPriority: 1, failureRate: 1, adoptionFriction: 1 });\n"
        "assert.ok(strongSignal > weakSignal);\n"
        "assert.equal(shouldPrioritizeUpgrade({ publicPriority: 1, failureRate: 1, adoptionFriction: 1 }), true);\n"
    )
    return ProjectEdit(
        path=test_path,
        operation="create_or_replace",
        content=content,
        rationale=f"Add executable regression coverage for {unit.theme}.",
        support_refs=unit.support_refs,
    )


def _relative_import_path(from_path: str, to_path: str) -> str:
    relative = os.path.relpath(
        Path(to_path),
        start=Path(from_path).parent,
    )
    if not relative.startswith("."):
        relative = f"./{relative}"
    return Path(relative).as_posix()


def _generated_test_commands(edits: tuple[ProjectEdit, ...]) -> tuple[str, ...]:
    return tuple(
        f"node {edit.path}" for edit in edits if edit.path.endswith(".test.mjs")
    )


def _target_paths_for_theme(theme: str) -> tuple[str, ...]:
    mapping = {
        "production_build_reliability": (
            "docs/company-development/production-build-reliability.md",
            "src/company-development/production_build_reliability.mjs",
            "tests/company-development/production_build_reliability.test.mjs",
        ),
        "dependency_resolution": (
            "docs/company-development/dependency-resolution.md",
            "src/company-development/dependency_resolution.mjs",
            "tests/company-development/dependency_resolution.test.mjs",
        ),
        "documentation_and_migration_path": (
            "docs/company-development/migration-path.md",
            "src/company-development/documentation_and_migration_path.mjs",
            "tests/company-development/documentation_and_migration_path.test.mjs",
        ),
        "framework_generalization": (
            "docs/company-development/framework-generalization.md",
            "src/company-development/framework_generalization.mjs",
            "tests/company-development/framework_generalization.test.mjs",
        ),
        "plugin_ecosystem": (
            "docs/company-development/plugin-api-roadmap.md",
            "src/company-development/plugin_ecosystem.mjs",
            "tests/company-development/plugin_ecosystem.test.mjs",
        ),
        "legacy_browser_compatibility": (
            "docs/company-development/legacy-browser-compatibility.md",
            "src/company-development/legacy_browser_compatibility.mjs",
            "tests/company-development/legacy_browser_compatibility.test.mjs",
        ),
    }
    slug = _safe_slug(theme)
    return mapping.get(
        theme,
        (
            f"docs/company-development/{slug}.md",
            f"src/company-development/{slug}.mjs",
            f"tests/company-development/{slug}.test.mjs",
        ),
    )


def _objective_for_theme(theme: str) -> str:
    mapping = {
        "production_build_reliability": "Reduce build failures and add regression coverage for public failure reports.",
        "dependency_resolution": "Make dependency resolution behavior predictable and diagnosable.",
        "documentation_and_migration_path": "Lower adoption friction with clearer migration and troubleshooting paths.",
        "framework_generalization": "Generalize the product beyond the initial framework niche.",
        "plugin_ecosystem": "Expose stable extension points and compatibility expectations.",
        "legacy_browser_compatibility": "Document and test compatibility boundaries for older browsers.",
        "programmable_workflows": "Expose composable workflow primitives with stable public contracts.",
        "guided_workflow_templates": "Provide adaptable templates that shorten first value without hiding configuration.",
        "collaborative_reuse": "Support provenance-preserving sharing and reuse of successful workflows.",
        "workflow_automation": "Automate repeatable workflow steps while preserving explicit review points.",
    }
    return mapping.get(
        theme, "Convert public feedback into a measurable product improvement."
    )


def _validation_for_theme(theme: str) -> tuple[str, ...]:
    if theme in {
        "production_build_reliability",
        "dependency_resolution",
        "plugin_ecosystem",
    }:
        return ("npm test", "npm run build")
    if theme in {"documentation_and_migration_path", "framework_generalization"}:
        return ("npm test", "npm run docs:check")
    if theme in CREATIVE_FEATURE_THEMES:
        return ("npm test", "npm run build")
    return ("npm test",)


def _scale_for_theme(theme: str) -> float:
    if theme in {
        "production_build_reliability",
        "dependency_resolution",
        "framework_generalization",
    }:
        return 0.82
    if theme in {"plugin_ecosystem", "documentation_and_migration_path"}:
        return 0.74
    if theme in CREATIVE_FEATURE_THEMES:
        return 0.70
    return 0.55


def _development_payload(
    *,
    proposal: CompanyOptimizationProposal,
    report: CompanyPublicFeedbackReport,
    public_traces: tuple[CompanyVisibleTrace, ...],
    workspace_manifest: ProjectWorkspaceManifest | None,
) -> dict:
    return canonicalize(
        {
            "proposal": {
                "proposal_id": proposal.proposal_id,
                "artifact_id": proposal.artifact_id,
                "requested_direction": proposal.requested_direction,
                "priority_themes": proposal.priority_themes,
                "theme_decisions": proposal.theme_decisions,
                "safety_obligation_themes": proposal.safety_obligation_themes,
                "allowed_development_themes": _allowed_development_themes(proposal),
                "support_refs": proposal.support_refs,
                "public_trace_hash": proposal.public_trace_hash,
                "optimizer_source": proposal.optimizer_source,
                "optimizer_model": proposal.optimizer_model,
                "optimizer_reasoning": proposal.optimizer_reasoning,
            },
            "public_feedback_report": report,
            "public_trace_summaries": public_traces[:80],
            "workspace_manifest": workspace_manifest,
            "privacy_boundary": "public_traces_and_project_files_only",
            "path_policy": "relative_paths_only_no_parent_directory_new_files_under_docs_company_development_by_default",
        }
    )


def _development_json_schema(*, allowed_themes: tuple[str, ...] | None = None) -> dict:
    string_array = {"type": "array", "items": {"type": "string"}}
    theme_schema: dict = {"type": "string", "maxLength": 80}
    if allowed_themes:
        theme_schema["enum"] = list(allowed_themes)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "change_units": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "theme": theme_schema,
                        "objective": {"type": "string", "maxLength": 300},
                        "target_paths": {
                            "type": "array",
                            "maxItems": 4,
                            "items": {"type": "string", "maxLength": 160},
                        },
                        "support_refs": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string", "maxLength": 120},
                        },
                        "validation_commands": {
                            "type": "array",
                            "maxItems": 6,
                            "items": {"type": "string", "maxLength": 160},
                        },
                        "scale_score": {"type": "number"},
                    },
                    "required": [
                        "theme",
                        "objective",
                        "target_paths",
                        "support_refs",
                        "validation_commands",
                        "scale_score",
                    ],
                },
            },
            "edits": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string", "maxLength": 160},
                        "operation": {"type": "string", "maxLength": 32},
                        "content": {"type": "string", "maxLength": 5000},
                        "rationale": {"type": "string", "maxLength": 300},
                        "support_refs": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string", "maxLength": 120},
                        },
                    },
                    "required": [
                        "path",
                        "operation",
                        "content",
                        "rationale",
                        "support_refs",
                    ],
                },
            },
            "verification_commands": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string", "maxLength": 300},
            },
        },
        "required": ["change_units", "edits", "verification_commands"],
    }


def _parse_json_object(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(
                f"Company coding agent returned non-JSON text: {raw[:200]!r}"
            ) from None
        value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Company coding agent returned JSON that is not an object")
    return value


def _response_text(response) -> str:
    raw = getattr(response, "output_text", "") or ""
    if raw.strip():
        return raw.strip()
    fragments: list[str] = []
    for item in getattr(response, "output", []) or []:
        item_content = (
            item.get("content", [])
            if isinstance(item, dict)
            else getattr(item, "content", [])
        )
        for content in item_content or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("value")
            else:
                text = getattr(content, "text", None) or getattr(content, "value", None)
            if text:
                fragments.append(text)
    return "\n".join(fragments).strip()


def _unit_from_json(
    item: dict,
    proposal: CompanyOptimizationProposal,
    index: int,
    *,
    allowed_themes: tuple[str, ...] | None = None,
) -> DevelopmentChangeUnit:
    theme = _safe_slug(str(item["theme"])) or "product_improvement"
    if allowed_themes is not None and theme not in allowed_themes:
        raise ValueError(f"Unsupported development theme: {theme}")
    target_paths = tuple(
        _safe_relative_path(str(path)) for path in item["target_paths"][:8]
    )
    support_refs = tuple(str(ref)[:120] for ref in item["support_refs"][:12])
    commands = tuple(str(command)[:240] for command in item["validation_commands"][:8])
    scale_score = max(0.0, min(1.0, float(item["scale_score"])))
    unit_hash = stable_hash(
        (proposal.proposal_id, theme, index, target_paths, support_refs)
    )[:12]
    return DevelopmentChangeUnit(
        unit_id=f"dev_unit_{index:02d}_{unit_hash}",
        theme=theme,
        objective=str(item["objective"])[:500],
        target_paths=target_paths,
        support_refs=support_refs,
        validation_commands=commands,
        scale_score=scale_score,
    )


def _edit_from_json(item: dict, proposal: CompanyOptimizationProposal) -> ProjectEdit:
    operation = str(item["operation"])
    if operation not in {"create_or_replace", "append"}:
        operation = "create_or_replace"
    return ProjectEdit(
        path=_safe_relative_path(str(item["path"])),
        operation=operation,
        content=str(item["content"])[:80_000],
        rationale=str(item["rationale"])[:500],
        support_refs=tuple(str(ref)[:120] for ref in item["support_refs"][:12])
        or proposal.support_refs[:8],
    )


def _development_run_id(
    *,
    proposal: CompanyOptimizationProposal,
    report: CompanyPublicFeedbackReport,
    source: str,
    model: str | None,
    change_units: tuple[DevelopmentChangeUnit, ...],
    edits: tuple[ProjectEdit, ...],
    workspace_manifest: ProjectWorkspaceManifest | None,
) -> str:
    digest = stable_hash(
        {
            "proposal_id": proposal.proposal_id,
            "report_id": report.report_id,
            "source": source,
            "model": model,
            "change_units": change_units,
            "edits": edits,
            "workspace_manifest": workspace_manifest,
        }
    )[:24]
    return f"company_development_{digest}"


def _safe_target_path(root: Path, relative_path: str) -> Path:
    target = (root / _safe_relative_path(relative_path)).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Project edit escapes workspace: {relative_path}")
    return target


def _safe_relative_path(path: str) -> str:
    normalized = Path(path).as_posix().lstrip("/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe project edit path: {path}")
    return "/".join(parts)


def _safe_slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    return "_".join(part for part in cleaned.split("_") if part)[:80]


def _is_ignored_path(path: Path, root: Path) -> bool:
    ignored = {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        "outputs",
        "runs",
    }
    rel_parts = path.relative_to(root).parts
    return any(part in ignored for part in rel_parts) or path.suffix == ".pyc"


def _dedupe(values) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return tuple(result)


def _normalized_company_coding_reasoning_effort(
    requested: str | None, *, model: str | None
) -> str:
    requested = (requested or DEFAULT_COMPANY_CODING_REASONING_EFFORT).lower()
    order = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}
    if requested not in order:
        requested = DEFAULT_COMPANY_CODING_REASONING_EFFORT
    if not str(model or "").startswith("gpt-5"):
        return requested
    if os.environ.get("SOCIETY_CORE_COMPANY_CODING_ALLOW_LOW_REASONING") == "1":
        return requested
    if order[requested] < order["high"]:
        return DEFAULT_COMPANY_CODING_REASONING_EFFORT
    if requested == "high" and str(model or "").startswith("gpt-5.5"):
        return DEFAULT_COMPANY_CODING_REASONING_EFFORT
    return requested


@contextmanager
def _wall_clock_timeout(timeout_seconds: float):
    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError(f"Company coding agent call exceeded {timeout_seconds:.3f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0 or previous_timer[1] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
