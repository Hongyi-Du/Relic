"""Autonomous workspace development loop for public-feedback experiments."""

from __future__ import annotations

import difflib
from itertools import combinations
import json
import math
import multiprocessing as mp
import os
import queue as queue_module
import re
import shlex
import sys
import tomllib
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from time import sleep
from types import SimpleNamespace
from typing import Any, Mapping, Protocol, Sequence

from .code_landing import (
    build_code_max_context_summary,
    candidate_from_proposal,
    run_code_max_runtime,
)
from .code_landing.evidence import (
    EvidenceLedger,
    VerificationEvidence,
    assess_landing_evidence,
)
from .code_landing.environment import build_workspace_execution_profile
from .code_landing.event_store import (
    RunRecord,
    SQLiteEventStore,
    redact_sensitive_payload,
)
from .code_landing.surfaces import has_security_surface
from .coding_tools import command_covers_declaration, is_behavior_check_command
from .execution import (
    CommandExecutor,
    ensure_command_executor_ready,
    probe_command_executor_executables,
)
from .hashing import canonicalize, stable_hash
from .openai_runtime import (
    build_openai_client,
    openai_call_watchdog,
    openai_runtime_identity,
    prepare_openai_response_request,
)
from .product_experience import CREATIVE_FEATURE_THEMES
from .safe_files import (
    UnsafeRegularFileError,
    read_regular_file_bytes,
    read_regular_file_text,
)
from .workspace_transaction import (
    CandidateWorkspace,
    WorkspaceCommitReceipt,
    WorkspacePromotionResult,
    acknowledge_committed_workspace_promotion,
    recover_interrupted_workspace_promotion,
)
from .verification_policy import VERIFICATION_EXECUTABLES
from .workspace_update import (
    WorkspaceFilePatch,
    WorkspacePackageInfo,
    WorkspacePatchResult,
    WorkspaceVerificationResult,
    load_workspace_package_info,
    run_workspace_verification,
    workspace_update_mode,
)

DEFAULT_WORKSPACE_CODING_MODEL = "gpt-5.5-2026-04-23"
DEFAULT_WORKSPACE_CODING_REASONING_EFFORT = "xhigh"
MIN_WORKSPACE_CODING_REASONING_EFFORT = "high"
DEFAULT_WORKSPACE_CODING_MAX_ITERATIONS = 8
DEFAULT_WORKSPACE_CODING_MAX_SELECTED_FILES = 32
DEFAULT_WORKSPACE_CODING_MAX_ACTIVE_THEMES = 1
DEFAULT_WORKSPACE_CODING_MAX_EXCERPT_CHARS = 2500
DEFAULT_WORKSPACE_CODING_RETRY_EXCERPT_CHARS = 12_000
DEFAULT_WORKSPACE_CODING_PROMPT_FILE_LIMIT = 12
DEFAULT_WORKSPACE_CODING_MAX_OUTPUT_TOKENS = 20000
MAX_AGENT_VERIFICATION_COMMAND_CHARS = 2_000
MAX_AGENT_TRACE_FAILURE_FEEDBACK = 8
MAX_AGENT_TRACE_FAILURE_TAIL_CHARS = 1_600
DEFAULT_WORKSPACE_CODING_EXPLORATION_ITERATIONS = 0
DEFAULT_WORKSPACE_CODING_REQUEST_ATTEMPTS = 4
DEFAULT_WORKSPACE_CODING_PARSE_REPAIR_ATTEMPTS = 1
DEFAULT_WORKSPACE_CODING_CANDIDATE_INFRA_RETRIES = 2
DEFAULT_WORKSPACE_CODING_CANDIDATE_INFRA_ROUND_RETRIES = 8
DEFAULT_WORKSPACE_CODING_PROVIDER_CIRCUIT_EXHAUSTED_SLOTS = 2
DEFAULT_WORKSPACE_CODING_CANDIDATE_STRATEGY_RETRIES = 1
DEFAULT_WORKSPACE_CODING_EVALUATOR_INFRA_RETRIES = 1
WORKSPACE_CONTROLLER_RUNTIME_VERSION = "48"
CANDIDATE_EARLY_STOP_POLICY_VERSION = "validated_diverse_quorum_v1"
CANDIDATE_EARLY_STOP_MODE = "shadow"
CANDIDATE_EARLY_STOP_PREFIX_COUNT = 3

_VOLATILE_CONTROLLER_EVENT_NUMERIC_FIELDS = frozenset(
    {
        "duration_sec",
        "duration_seconds",
        "elapsed_sec",
        "elapsed_seconds",
        "latency_ms",
        "wall_clock_sec",
    }
)

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "vendor",
}
IGNORED_SUFFIXES = {
    ".bmp",
    ".class",
    ".dll",
    ".dylib",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".lock",
    ".map",
    ".min.js",
    ".mp4",
    ".o",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".wasm",
    ".webp",
    ".zip",
}

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "architecture_generalization": (
        "architecture",
        "command",
        "config",
        "pipeline",
        "source",
    ),
    "dependency_resolution": (
        "dependency",
        "import",
        "install",
        "module",
        "resolve",
        "web_modules",
    ),
    "diagnostics_and_recovery": (
        "diagnostic",
        "error",
        "fallback",
        "message",
        "overlay",
        "recovery",
    ),
    "documentation_and_migration_path": (
        "docs",
        "help",
        "init",
        "migration",
        "readme",
        "troubleshoot",
    ),
    "framework_generalization": (
        "adapter",
        "framework",
        "local",
        "plugin",
        "source",
    ),
    "plugin_ecosystem": (
        "hook",
        "plugin",
        "transform",
        "extension",
        "middleware",
    ),
    "programmable_workflows": (
        "api",
        "command",
        "compose",
        "config",
        "pipeline",
        "workflow",
    ),
    "guided_workflow_templates": (
        "example",
        "preset",
        "scaffold",
        "starter",
        "template",
        "workflow",
    ),
    "collaborative_reuse": (
        "export",
        "import",
        "provenance",
        "share",
        "template",
        "workflow",
    ),
    "workflow_automation": (
        "automation",
        "command",
        "hook",
        "pipeline",
        "task",
        "workflow",
    ),
    "production_build_reliability": (
        "build",
        "bundle",
        "cache",
        "optimize",
        "production",
    ),
    "token_count_resilience": (
        "encoding",
        "formatter",
        "output",
        "tiktoken",
        "token",
        "tokens",
    ),
    "max_file_size_enforcement": (
        "bytes",
        "exclude",
        "ingestion",
        "max_file_size",
        "size",
    ),
    "include_submodules": (
        "clone",
        "entrypoint",
        "git",
        "submodule",
        "submodules",
    ),
    "ignore_pattern_reliability": (
        "gitignore",
        "gitingestignore",
        "ignore",
        "pathspec",
        "patterns",
    ),
    "pattern_filtering_consistency": (
        "exclude_patterns",
        "filter",
        "include_patterns",
        "patterns",
        "query",
    ),
    "http_client_portability": (
        "check_repo_exists",
        "curl",
        "http",
        "httpx",
        "reachability",
        "urllib",
    ),
    "path_security": (
        "path",
        "relative_to",
        "resolve",
        "safe",
        "traversal",
    ),
}

_THEME_FAILURE_EVENT_MARKERS: dict[str, tuple[str, ...]] = {
    "diagnostics_and_recovery": ("diagnostic", "error", "failure", "recovery"),
    "token_count_resilience": ("token", "count", "network"),
    "max_file_size_enforcement": ("file_size", "max_file", "size_leak"),
    "include_submodules": ("submodule",),
    "ignore_pattern_reliability": (
        "ignore",
        "gitignore",
        "tool_ignore",
        "gitingestignore",
    ),
    "pattern_filtering_consistency": ("pattern", "include", "exclude", "filter"),
    "http_client_portability": ("curl", "http", "client"),
    "path_security": ("path", "traversal"),
    "production_build_reliability": ("build", "production"),
}


@dataclass(frozen=True)
class WorkspaceFileContext:
    path: str
    content_hash: str
    excerpt: str
    truncated: bool
    byte_size: int
    score: float
    context_role: str = "normal"


@dataclass(frozen=True)
class CodingAgentPatchProposal:
    proposal_id: str
    source: str
    model: str | None
    iteration: int
    artifact_id: str | None
    themes: tuple[str, ...]
    patches: tuple[WorkspaceFilePatch, ...]
    verification_commands: tuple[str, ...]
    selected_files: tuple[str, ...]
    rationale: str
    support_refs: tuple[str, ...]
    raw_response_hash: str | None = None
    blocked_reason: str | None = None
    stage: str = "promote"


@dataclass(frozen=True)
class CodingAgentLoopResult:
    run_id: str
    developer_mode: str
    source_report_id: str | None
    package: WorkspacePackageInfo
    selected_files: tuple[str, ...]
    proposals: tuple[CodingAgentPatchProposal, ...]
    patch_results: tuple[WorkspacePatchResult, ...]
    verification_results: tuple[WorkspaceVerificationResult, ...]
    iterations: int
    final_mode: str
    failed_commands: tuple[str, ...]
    provenance_hash: str
    privacy_boundary: str = "public_report_and_workspace_files_only"
    development_stage: str = "guarded_promotion_agent"
    exploration_attempts: int = 0
    risky_patch_bundles: int = 0
    promoted_candidates: int = 0
    verified_effective_patches: int = 0
    rejected_by_gate: int = 0
    repair_iterations: int = 0
    reasoning_effort: str | None = None
    requested_reasoning_effort: str | None = None
    run_config: dict[str, Any] = field(default_factory=dict)
    patch_coverage: dict[str, Any] = field(default_factory=dict)
    code_landing_metrics: dict[str, Any] = field(default_factory=dict)
    evidence_level: str = "proposal_ready"
    evidence_hash: str | None = None
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    evidence_records: tuple[VerificationEvidence, ...] = ()


class CodeLandingFailureReason(str, Enum):
    MISSING_REPRO = "missing_repro"
    SPEC_TOO_AMBIGUOUS = "spec_too_ambiguous"
    BAD_CONTEXT = "bad_context"
    WRONG_FILE = "wrong_file"
    WRONG_SYMBOL = "wrong_symbol"
    PATCH_DID_NOT_APPLY = "patch_did_not_apply"
    PATCH_NO_EFFECT = "patch_no_effect"
    TEST_ONLY_PATCH = "test_only_patch"
    REGRESSION_BREAK = "regression_break"
    BROAD_REWRITE = "broad_rewrite"
    BEHAVIOR_NOT_VERIFIED = "behavior_not_verified"
    HISTORICAL_ALIGNMENT_MISS = "historical_alignment_miss"


@dataclass(frozen=True)
class AcceptanceOracle:
    oracle_id: str
    kind: str
    command: str | None
    description: str
    source: str
    dimension_ids: tuple[str, ...] = ()
    witness_role: str = "repair"
    expected_on_base: str = "not_recorded"
    expected_on_patch: str = "pass"
    required: bool = True
    evidence_hash: str | None = None


@dataclass(frozen=True)
class DevelopmentTaskSpecV2:
    task_id: str
    source_theme: str
    task_type: str
    public_evidence_refs: tuple[str, ...]
    user_pain: str
    observed_behavior: str
    expected_behavior: str
    non_goals: tuple[str, ...]
    reproduction_recipe: tuple[str, ...]
    reproduction_expected_failure: str | None
    acceptance_oracles: tuple[AcceptanceOracle, ...]
    repo_entrypoints: tuple[str, ...]
    candidate_path_hints: tuple[str, ...]
    relevant_symbols_hint: tuple[str, ...]
    ambiguity_level: str
    missing_information: tuple[str, ...]
    spec_maturity: str
    risk_level: str
    behavior_surface: tuple[str, ...]
    contract_dimensions: tuple[str, ...] = ()
    boundary_conditions: tuple[str, ...] = ()
    source_explicit_intent_spec: bool = False


@dataclass(frozen=True)
class RepoIntelligencePack:
    repo_hash: str
    file_tree: tuple[str, ...]
    package_manager: str | None
    test_commands: tuple[str, ...]
    lint_commands: tuple[str, ...]
    entrypoints: tuple[str, ...]
    source_roots: tuple[str, ...]
    test_files: tuple[str, ...]
    config_files: tuple[str, ...]
    docs_files: tuple[str, ...]
    symbol_index: dict[str, tuple[str, ...]]
    test_map: dict[str, tuple[str, ...]]
    candidate_files_by_theme: dict[str, tuple[str, ...]]
    style_notes: tuple[str, ...]
    recent_failure_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReproArtifact:
    task_id: str
    kind: str
    files_created: tuple[str, ...]
    command: str
    expected_on_base: str
    observed_on_base: str
    reproducibility_status: str
    evidence_hash: str
    source: str


@dataclass(frozen=True)
class ReproducerPlan:
    plan_id: str
    artifacts: tuple[ReproArtifact, ...]
    executable_commands: tuple[str, ...]
    confirmed_count: int
    blocked_count: int
    missing_repro_tasks: tuple[str, ...]


@dataclass(frozen=True)
class CodeLandingFunnelRecord:
    task_id: str
    theme: str
    public_feedback_contains_pain: bool
    optimizer_selected_theme: bool
    spec_maturity: str
    repro_confirmed: bool
    localization_file_hit: bool | None
    localization_symbol_hit: bool | None
    patch_applies: bool
    repro_passed: bool
    targeted_tests_passed: bool
    regression_tests_passed: bool | None
    landing_gate_passed: bool
    historical_theme_overlap: bool | None
    evaluator_passed: bool | None
    failure_reason: str | None


@dataclass(frozen=True)
class EngineeringWorkPackage:
    package_id: str
    owner_role: str
    mission: str
    source_themes: tuple[str, ...]
    task_ids: tuple[str, ...]
    candidate_paths: tuple[str, ...]
    required_outputs: tuple[str, ...]
    gate: str
    risk_level: str


@dataclass(frozen=True)
class EngineeringOrganizationPlan:
    plan_id: str
    release_strategy: str
    architecture_delta_required: bool
    public_api_delta_required: bool
    developer_experience_delta_required: bool
    security_review_required: bool
    work_packages: tuple[EngineeringWorkPackage, ...]
    execution_order: tuple[str, ...]
    verifier_contract: tuple[str, ...]
    evidence_boundary: str


class WorkspaceCodingAgent(Protocol):
    source: str
    model: str | None

    def propose(
        self,
        *,
        source_report: Mapping[str, Any],
        workspace_root: Path,
        package: WorkspacePackageInfo,
        file_contexts: tuple[WorkspaceFileContext, ...],
        previous_verification: tuple[WorkspaceVerificationResult, ...],
        iteration: int,
    ) -> CodingAgentPatchProposal: ...


class HeuristicWorkspaceCodingAgent:
    """Deterministic fallback used for ablations and offline smoke tests."""

    source = "heuristic_workspace_coding_agent_ablation"
    model = None
    development_stage = "promote"

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
        themes = _extract_themes(source_report)
        artifact_id = _extract_artifact_id(source_report)
        support_refs = _extract_support_refs(source_report)
        payload = {
            "artifact_id": artifact_id,
            "package": {
                "name": package.name,
                "version": package.version,
                "manifest_hash": package.manifest_hash,
            },
            "themes": themes,
            "selected_files": tuple(context.path for context in file_contexts),
            "previous_failures": _verification_summary(previous_verification),
            "source_report_id": source_report.get("report_id"),
        }
        content = json.dumps(canonicalize(payload), indent=2, sort_keys=True) + "\n"
        patches = (
            WorkspaceFilePatch(
                patch_id=f"heuristic_workspace_report_{iteration:02d}",
                path="company-development/workspace-agent-report.json",
                operation="create_or_replace",
                content=content,
                rationale="Record public-feedback development context for offline ablation.",
            ),
        )
        return _proposal_from_parts(
            source=self.source,
            model=self.model,
            iteration=iteration,
            artifact_id=artifact_id,
            themes=themes,
            patches=patches,
            verification_commands=(),
            selected_files=tuple(context.path for context in file_contexts),
            rationale="Deterministic offline fallback; not the main autonomous condition.",
            support_refs=support_refs,
            raw_response_hash=None,
        )


class OpenAIWorkspaceCodingAgent:
    source = "openai_workspace_coding_agent"
    runtime_version = WORKSPACE_CONTROLLER_RUNTIME_VERSION

    def __init__(
        self, *, model: str | None = None, timeout_seconds: float = 300.0
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAIWorkspaceCodingAgent"
            )
        self.model = (
            model
            or os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_MODEL")
            or DEFAULT_WORKSPACE_CODING_MODEL
        )
        requested_reasoning_effort = os.environ.get(
            "SOCIETY_CORE_WORKSPACE_CODING_REASONING_EFFORT",
            DEFAULT_WORKSPACE_CODING_REASONING_EFFORT,
        )
        self.requested_reasoning_effort = requested_reasoning_effort
        self.reasoning_effort = _normalized_reasoning_effort(
            requested_reasoning_effort,
            model=self.model,
        )
        self.timeout_seconds = timeout_seconds
        self.development_stage = "promote"
        runtime_identity = openai_runtime_identity()
        self.endpoint_kind = str(runtime_identity["endpoint_kind"])
        self.endpoint_hash = str(runtime_identity["endpoint_hash"])
        self.default_header_names = tuple(runtime_identity["default_header_names"])
        self.default_headers_hash = str(runtime_identity["default_headers_hash"])
        self.response_storage_disabled = bool(
            runtime_identity["response_storage_disabled"]
        )
        self._api_key = api_key
        self._client = build_openai_client(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=0,
        )

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
        payload = _agent_payload(
            source_report=source_report,
            workspace_root=workspace_root,
            package=package,
            file_contexts=file_contexts,
            previous_verification=previous_verification,
            iteration=iteration,
            development_stage=self.development_stage,
        )
        stage_instruction = _stage_prompt(self.development_stage)
        prompt = (
            "You are the company's internal autonomous coding agent.\n"
            "Your job is to convert public product feedback into concrete, reviewable "
            "workspace patches.\n"
            f"{stage_instruction}\n"
            "Use only the provided public experiment report, selected file excerpts, "
            "package metadata, and verification failures.\n"
            "Do not claim access to private user state, hidden company plans, the "
            "internet, or files not included in the context.\n"
            "This is a closed-loop execution step: decide the implementation plan and "
            "immediately return the code/test/config patches that realize it. Do not "
            "return a roadmap, planning-only document, or future-work note instead of "
            "workspace changes.\n"
            "If workspace_progress.remaining_requested_themes is non-empty, prioritize "
            "those themes over already covered themes. Cover as many remaining themes "
            "as can be safely verified in this patch bundle.\n"
            "Use development_task_specs_v2 as the engineering task cards. Prefer "
            "implementation_ready or testable tasks first. If a task is below "
            "testable, improve the task only through public evidence and do not "
            "invent private facts.\n"
            "Use repo_intelligence_pack to localize likely source files, test files, "
            "entrypoints, and symbols before editing. Use reproducer_plan and any "
            "previous base_reproducer_confirmed_failure entries as the first behavior "
            "target; make the same repro pass after the patch when possible.\n"
            "Use engineering_organization_plan as the release-candidate operating "
            "structure. Follow its execution_order and work_packages: architecture "
            "packages define module boundaries, api_surface packages define public "
            "contracts, developer_experience packages cover user-visible migration "
            "or diagnostics, security_review packages add guardrails, implementation "
            "packages land source/tests, verifier packages choose evidence commands, "
            "and evidence_boundary packages constrain final claims.\n"
            "Use code_max_runtime as the high-signal engineering context: it contains "
            "the executable environment, repo intelligence, acceptance oracles, "
            "localization hypotheses, patch strategies, and runtime policy. Treat it "
            "as the primary way to decide where to edit and how the patch will be "
            "judged. Before a multi-file or architecture-scale edit, inspect its "
            "repo.impact_context imports, dependents, and mapped tests so public callers "
            "and integration boundaries are changed deliberately.\n"
            "Prefer source-level changes with tests when the context supports them. "
            "Use documentation-only changes only when no relevant product code or test "
            "file is present in selected_files.\n"
            "Use dependency-light verification: commands must run in this unpacked "
            "workspace with only the visible files and already available dependencies. "
            "Do not include broad package commands such as npm test unless the runtime "
            "context shows they are viable; prefer focused node/python commands, "
            "self-contained regression files, or explicit stubs for unavailable "
            "third-party modules.\n"
            "Return JSON only. Paths must be relative to the workspace root. Supported "
            "patch operations are create_or_replace, replace_fragment, and "
            "append_if_missing.\n\n"
            "Repair rule: if a previous failure reports replace_fragment_not_found, "
            "do not retry the same unavailable fragment. Either copy an exact old "
            "substring from the current selected file excerpts, or use a new file / "
            "append_if_missing patch that can be safely applied by the guarded kernel.\n\n"
            "JSON rule: patches to .json files must leave the full file as valid JSON. "
            "Do not append raw key/value fragments to package.json; use an exact "
            "replace_fragment inside an existing JSON object or create_or_replace the "
            "entire valid JSON file.\n\n"
            "Non-destructive rule: never delete a full function, command, or config "
            "section to make checks pass. replace_fragment patches must preserve or "
            "extend behavior; the guarded kernel blocks large deletions.\n\n"
            f"INPUT_JSON:\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
        )
        request: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": _workspace_coding_max_output_tokens(),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "workspace_patch_proposal",
                    "strict": True,
                    "schema": _workspace_patch_json_schema(),
                }
            },
            "timeout": self.timeout_seconds,
        }
        if not str(self.model).startswith("gpt-5"):
            request["temperature"] = 0
        if str(self.model).startswith("gpt-5"):
            request["reasoning"] = {"effort": self.reasoning_effort}
        selected_files = tuple(context.path for context in file_contexts)
        request = prepare_openai_response_request(request)
        response, last_request_error = self._create_response_with_retries(request)
        if response is None:
            assert last_request_error is not None
            return _blocked_proposal(
                source=self.source,
                model=self.model,
                iteration=iteration,
                artifact_id=_extract_artifact_id(source_report),
                themes=_extract_themes(source_report),
                selected_files=selected_files,
                reason=(
                    f"openai_request_failed:{_request_error_type(last_request_error)}:"
                    f"attempts={_workspace_coding_request_attempts()}:"
                    f"{_request_error_detail(last_request_error)}"
                ),
            )
        raw = _response_text(response)
        parsed: Mapping[str, Any] | None = None
        parse_exception: Exception | None = None
        for repair_attempt in range(_workspace_coding_parse_repair_attempts() + 1):
            try:
                parsed = _parse_json_object(raw)
                break
            except Exception as exc:
                parse_exception = exc
                if repair_attempt >= _workspace_coding_parse_repair_attempts():
                    break
                repair_request = dict(request)
                repair_request["input"] = _json_repair_prompt(
                    prompt=prompt,
                    raw_response=raw,
                    error=exc,
                )
                response, last_request_error = self._create_response_with_retries(
                    repair_request
                )
                if response is None:
                    assert last_request_error is not None
                    return _blocked_proposal(
                        source=self.source,
                        model=self.model,
                        iteration=iteration,
                        artifact_id=_extract_artifact_id(source_report),
                        themes=_extract_themes(source_report),
                        selected_files=selected_files,
                        reason=(
                            f"openai_request_failed:{_request_error_type(last_request_error)}:"
                            f"attempts={_workspace_coding_request_attempts()}:"
                            f"{_request_error_detail(last_request_error)}"
                        ),
                        raw_response_hash=stable_hash(raw),
                    )
                raw = _response_text(response)
        if parsed is None:
            assert parse_exception is not None
            detail = str(parse_exception).replace("\n", " ")[:220]
            parse_error = detail.startswith("Workspace coding agent returned")
            reason = f"openai_response_invalid:{type(parse_exception).__name__}"
            if parse_error:
                reason = f"{reason}:{detail}"
            return _blocked_proposal(
                source=self.source,
                model=self.model,
                iteration=iteration,
                artifact_id=_extract_artifact_id(source_report),
                themes=_extract_themes(source_report),
                selected_files=selected_files,
                reason=reason,
                raw_response_hash=stable_hash(raw),
            )
        try:
            patches = tuple(
                _patch_from_json(item, index)
                for index, item in enumerate(parsed["patches"])
            )
            verification_commands = _agent_verification_commands_from_json(
                parsed["verification_commands"]
            )
        except Exception as exc:
            detail = str(exc).replace("\n", " ")[:220]
            reason = f"openai_response_invalid:{type(exc).__name__}"
            return _blocked_proposal(
                source=self.source,
                model=self.model,
                iteration=iteration,
                artifact_id=_extract_artifact_id(source_report),
                themes=_extract_themes(source_report),
                selected_files=selected_files,
                reason=reason,
                raw_response_hash=stable_hash(raw),
            )
        return _proposal_from_parts(
            source=self.source,
            model=self.model,
            iteration=iteration,
            artifact_id=_optional_string(parsed.get("artifact_id"))
            or _extract_artifact_id(source_report),
            themes=tuple(str(theme)[:80] for theme in parsed["themes"][:12]),
            patches=patches,
            verification_commands=verification_commands,
            selected_files=selected_files,
            rationale=str(parsed["rationale"])[:2000],
            support_refs=tuple(str(ref)[:160] for ref in parsed["support_refs"][:16]),
            raw_response_hash=stable_hash(raw),
            blocked_reason=_optional_string(parsed.get("blocked_reason")),
        )

    def _create_response_with_retries(
        self,
        request: Mapping[str, Any],
    ) -> tuple[Any | None, Exception | None]:
        max_request_attempts = _workspace_coding_request_attempts()
        watchdog_seconds = _workspace_coding_call_watchdog_seconds(self.timeout_seconds)
        last_request_error: Exception | None = None
        for attempt in range(max_request_attempts):
            try:
                if _workspace_coding_process_requests_enabled():
                    response, process_error = _create_response_with_process_timeout(
                        api_key=self._api_key,
                        request=request,
                        timeout_seconds=watchdog_seconds,
                    )
                    if process_error is not None:
                        raise process_error
                    return response, None
                with openai_call_watchdog(
                    watchdog_seconds,
                    label="workspace coding agent call",
                ):
                    return self._client.responses.create(**dict(request)), None
            except Exception as exc:  # pragma: no cover - live network boundary.
                last_request_error = exc
                if attempt + 1 >= max_request_attempts:
                    break
                sleep(_workspace_coding_retry_sleep_seconds(attempt))
        return None, last_request_error


def workspace_coding_agent_request_identity(
    *,
    provider: str,
    model: str | None,
    timeout_seconds: float,
    tool_max_turns: int | None = None,
    tool_max_calls: int | None = None,
    tool_max_patch_bytes: int | None = None,
) -> dict[str, Any]:
    """Resolve the exact non-secret agent configuration requested for a run."""

    normalized_provider = provider.replace("-", "_")
    if normalized_provider == "heuristic":
        return {
            "provider": "heuristic",
            "source": HeuristicWorkspaceCodingAgent.source,
            "model": None,
        }
    selected_model = model or DEFAULT_WORKSPACE_CODING_MODEL
    requested_reasoning_effort = os.environ.get(
        "SOCIETY_CORE_WORKSPACE_CODING_REASONING_EFFORT",
        DEFAULT_WORKSPACE_CODING_REASONING_EFFORT,
    )
    runtime_identity = openai_runtime_identity()
    identity: dict[str, Any] = {
        "provider": normalized_provider,
        "model": selected_model,
        "timeout_seconds": timeout_seconds,
        "requested_reasoning_effort": requested_reasoning_effort,
        "reasoning_effort": _normalized_reasoning_effort(
            requested_reasoning_effort,
            model=selected_model,
        ),
        **runtime_identity,
    }
    if normalized_provider == "openai":
        return canonicalize(
            {
                **identity,
                "source": OpenAIWorkspaceCodingAgent.source,
                "runtime_version": OpenAIWorkspaceCodingAgent.runtime_version,
            }
        )
    if normalized_provider == "openai_tools":
        from .tool_loop_agent import (
            DEFAULT_TOOL_LOOP_MAX_PATCH_BYTES,
            DEFAULT_TOOL_LOOP_MAX_TOOL_CALLS,
            DEFAULT_TOOL_LOOP_MAX_TURNS,
            TOOL_LOOP_RUNTIME_VERSION,
            resolve_response_transport,
            resolve_tool_request_timeout_seconds,
        )

        return canonicalize(
            {
                **identity,
                "source": "openai_interactive_workspace_agent",
                "runtime_version": TOOL_LOOP_RUNTIME_VERSION,
                "max_turns": (
                    DEFAULT_TOOL_LOOP_MAX_TURNS
                    if tool_max_turns is None
                    else tool_max_turns
                ),
                "max_tool_calls": (
                    DEFAULT_TOOL_LOOP_MAX_TOOL_CALLS
                    if tool_max_calls is None
                    else tool_max_calls
                ),
                "max_patch_bytes": (
                    DEFAULT_TOOL_LOOP_MAX_PATCH_BYTES
                    if tool_max_patch_bytes is None
                    else tool_max_patch_bytes
                ),
                "max_request_attempts": _workspace_coding_request_attempts(),
                "request_timeout_seconds": (
                    resolve_tool_request_timeout_seconds(
                        timeout_seconds=timeout_seconds,
                    )
                ),
                "tool_protocol": os.environ.get(
                    "SOCIETY_CORE_WORKSPACE_CODING_TOOL_PROTOCOL",
                    "auto",
                )
                .strip()
                .lower(),
                "response_transport": resolve_response_transport(
                    endpoint_kind=str(identity["endpoint_kind"]),
                ),
            }
        )
    raise ValueError(f"Unsupported workspace coding agent provider: {provider}")


def workspace_coding_controller_request_identity() -> dict[str, Any]:
    """Resolve non-secret retry and circuit settings that affect controller behavior."""

    candidate_infra_retries = _workspace_coding_candidate_infra_retries()
    candidate_round_retries = _workspace_coding_candidate_infra_round_retries()
    return canonicalize(
        {
            "runtime_version": WORKSPACE_CONTROLLER_RUNTIME_VERSION,
            "candidate_infrastructure_retry_limit": candidate_infra_retries,
            "candidate_infrastructure_round_retry_limit": candidate_round_retries,
            "candidate_strategy_retry_limit": (
                DEFAULT_WORKSPACE_CODING_CANDIDATE_STRATEGY_RETRIES
            ),
            "evaluator_infrastructure_retry_limit": (
                DEFAULT_WORKSPACE_CODING_EVALUATOR_INFRA_RETRIES
            ),
            "provider_circuit_exhausted_slot_limit": (
                _workspace_coding_provider_circuit_exhausted_slots()
            ),
            "candidate_early_stop_policy_version": (
                CANDIDATE_EARLY_STOP_POLICY_VERSION
            ),
            "candidate_early_stop_mode": CANDIDATE_EARLY_STOP_MODE,
            "candidate_early_stop_prefix_count": CANDIDATE_EARLY_STOP_PREFIX_COUNT,
            "candidate_infrastructure_retry_delays": tuple(
                _candidate_infra_retry_sleep_seconds(attempt)
                for attempt in range(candidate_infra_retries + candidate_round_retries)
            ),
        }
    )


def build_workspace_coding_agent(
    *,
    provider: str = "heuristic",
    model: str | None = None,
    timeout_seconds: float = 300.0,
    executor: CommandExecutor | None = None,
    event_store: Any | None = None,
    tool_max_turns: int | None = None,
    tool_max_calls: int | None = None,
    tool_max_patch_bytes: int | None = None,
) -> WorkspaceCodingAgent:
    request_identity = workspace_coding_agent_request_identity(
        provider=provider,
        model=model,
        timeout_seconds=timeout_seconds,
        tool_max_turns=tool_max_turns,
        tool_max_calls=tool_max_calls,
        tool_max_patch_bytes=tool_max_patch_bytes,
    )
    if provider == "heuristic":
        return HeuristicWorkspaceCodingAgent()
    if provider == "openai":
        return OpenAIWorkspaceCodingAgent(model=model, timeout_seconds=timeout_seconds)
    if provider in {"openai_tools", "openai-tools"}:
        from .tool_loop_agent import OpenAIToolLoopWorkspaceCodingAgent

        if executor is None:
            raise ValueError("workspace_executor_required")
        selected_model = str(request_identity["model"])
        requested_reasoning_effort = str(request_identity["requested_reasoning_effort"])
        return OpenAIToolLoopWorkspaceCodingAgent(
            executor=executor,
            model=selected_model,
            timeout_seconds=timeout_seconds,
            request_timeout_seconds=float(request_identity["request_timeout_seconds"]),
            event_store=event_store,
            max_request_attempts=int(request_identity["max_request_attempts"]),
            reasoning_effort=str(request_identity["reasoning_effort"]),
            requested_reasoning_effort=requested_reasoning_effort,
            max_turns=int(request_identity["max_turns"]),
            max_tool_calls=int(request_identity["max_tool_calls"]),
            max_patch_bytes=int(request_identity["max_patch_bytes"]),
            tool_protocol=str(request_identity["tool_protocol"]),
            response_transport=str(request_identity["response_transport"]),
        )
    raise ValueError(f"Unsupported workspace coding agent provider: {provider}")


def workspace_coding_agent_runtime_identity(
    agent: WorkspaceCodingAgent,
) -> dict[str, Any]:
    """Return the non-secret agent configuration bound into replay identities."""

    identity: dict[str, Any] = {
        "source": str(getattr(agent, "source", agent.__class__.__name__)),
        "model": getattr(agent, "model", None),
    }
    optional_fields = (
        "runtime_version",
        "timeout_seconds",
        "request_timeout_seconds",
        "reasoning_effort",
        "requested_reasoning_effort",
        "max_turns",
        "max_tool_calls",
        "max_patch_bytes",
        "max_exploration_calls_per_revision",
        "max_request_attempts",
        "tool_protocol",
        "response_transport",
        "endpoint_kind",
        "endpoint_hash",
        "default_header_names",
        "default_headers_hash",
        "response_storage_disabled",
    )
    for field_name in optional_fields:
        if hasattr(agent, field_name):
            identity[field_name] = getattr(agent, field_name)
    return canonicalize(identity)


def build_development_intent_specs(
    source_report: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Convert public feedback themes into concrete implementation work items."""

    existing = source_report.get("development_intent_specs")
    if isinstance(existing, (list, tuple)):
        normalized = tuple(
            _normalize_development_intent_spec(item, index)
            for index, item in enumerate(existing)
            if isinstance(item, Mapping)
        )
        if normalized:
            return normalized

    themes = _extract_themes(source_report)
    support_refs = _extract_support_refs(source_report)
    real_observations = _real_experience_observation_summaries(source_report)
    specs: list[dict[str, Any]] = []
    for index, theme in enumerate(themes):
        path_hints = _candidate_path_hints_for_theme(theme)
        public_failure_events = _public_failure_events_for_theme(
            source_report,
            theme,
        )
        spec_payload = {
            "report_id": source_report.get("report_id"),
            "theme": theme,
            "index": index,
            "support_refs": support_refs,
            "real_observations": real_observations,
            "public_failure_events": public_failure_events,
        }
        spec_hash = stable_hash(spec_payload)[:10]
        specs.append(
            {
                "intent_id": f"intent_{_slug_label(theme)}_{spec_hash}",
                "theme": theme,
                "_generated_intent_spec": True,
                "user_pain": _user_pain_for_theme(
                    theme,
                    source_report,
                    real_observations,
                    public_failure_events=public_failure_events,
                ),
                "reproduction_steps": _reproduction_steps_for_theme(
                    theme,
                    real_observations,
                    public_failure_events=public_failure_events,
                ),
                "expected_behavior": _expected_behavior_for_theme(theme),
                "contract_dimensions": _contract_dimensions_for_theme(
                    theme,
                    public_failure_events,
                ),
                "candidate_path_hints": path_hints,
                "acceptance_tests": _acceptance_tests_for_theme(
                    theme,
                    path_hints,
                    behavior_surface=_behavior_surface_for_theme(theme),
                ),
                "support_refs": support_refs,
                "evidence_refs": public_failure_events
                or tuple(
                    observation["failure_event"]
                    for observation in real_observations
                    if observation.get("failure_event")
                ),
            }
        )
    return tuple(specs)


def build_development_task_specs_v2(
    source_report: Mapping[str, Any],
    *,
    repo_intelligence: RepoIntelligencePack | None = None,
) -> tuple[DevelopmentTaskSpecV2, ...]:
    """Upgrade public feedback themes into typed implementation task cards."""

    specs = build_development_intent_specs(source_report)
    upgraded: list[DevelopmentTaskSpecV2] = []
    for index, spec in enumerate(specs):
        theme = str(spec.get("theme") or f"unspecified_theme_{index}")[:80]
        support_refs = _string_tuple(
            spec.get("support_refs"),
            fallback=_extract_support_refs(source_report),
            max_items=16,
        )
        path_hints = _string_tuple(
            spec.get("candidate_path_hints"),
            fallback=_candidate_path_hints_for_theme(theme),
            max_items=12,
        )
        explicit_intent_spec = bool(spec.get("_source_explicit_intent_spec"))
        explicit_path_hints = bool(spec.get("_explicit_candidate_path_hints"))
        relevant_symbols_hint = _string_tuple(
            spec.get("relevant_symbols_hint"),
            fallback=_symbols_for_theme(theme),
            max_items=12,
        )
        behavior_surface = _string_tuple(
            spec.get("behavior_surface"),
            fallback=_behavior_surface_for_theme(theme),
            max_items=8,
        )
        if not (explicit_intent_spec and explicit_path_hints):
            path_hints = _repo_augmented_path_hints(
                theme=theme,
                path_hints=path_hints,
                repo_intelligence=repo_intelligence,
                relevant_symbols=relevant_symbols_hint,
                behavior_surface=behavior_surface,
            )
        repo_entrypoints = _repo_entrypoints_for_spec(
            theme=theme,
            path_hints=path_hints,
            repo_intelligence=repo_intelligence,
        )
        acceptance_oracles = _acceptance_oracles_for_spec(spec, index)
        explicit_acceptance_tests = bool(spec.get("_explicit_acceptance_tests"))
        if not any(oracle.command for oracle in acceptance_oracles) and not (
            explicit_intent_spec and not explicit_acceptance_tests
        ):
            acceptance_oracles = (
                *acceptance_oracles,
                *_repo_backed_acceptance_oracles_for_theme(
                    theme=theme,
                    index=index,
                    path_hints=path_hints,
                    repo_intelligence=repo_intelligence,
                    behavior_surface=behavior_surface,
                ),
            )
        non_goals = _non_goals_for_spec(spec)
        observed_behavior = _observed_behavior_for_spec(spec)
        expected_behavior = str(
            spec.get("expected_behavior") or _expected_behavior_for_theme(theme)
        )[:500]
        reproduction_recipe = _string_tuple(
            spec.get("reproduction_steps"),
            fallback=_reproduction_steps_for_theme(theme, ()),
            max_items=8,
        )
        contract_dimensions = _string_tuple(
            spec.get("contract_dimensions"),
            fallback=_contract_dimensions_for_theme(
                theme,
                _string_tuple(spec.get("evidence_refs"), fallback=(), max_items=16),
            ),
            max_items=16,
        )
        boundary_conditions = tuple(
            dict.fromkeys(
                (
                    *_string_tuple(
                        spec.get("boundary_conditions"),
                        fallback=(),
                        max_items=16,
                    ),
                    *_boundary_conditions_for_spec(
                        theme=theme,
                        user_pain=str(spec.get("user_pain") or ""),
                        observed_behavior=observed_behavior,
                        expected_behavior=expected_behavior,
                        reproduction_recipe=reproduction_recipe,
                        contract_dimensions=contract_dimensions,
                    ),
                )
            )
        )[:16]
        missing_information = _missing_information_for_task_spec(
            user_pain=str(spec.get("user_pain") or ""),
            observed_behavior=observed_behavior,
            expected_behavior=expected_behavior,
            reproduction_recipe=reproduction_recipe,
            acceptance_oracles=acceptance_oracles,
            candidate_path_hints=path_hints,
            public_evidence_refs=support_refs,
        )
        spec_maturity = _classify_task_spec_maturity(
            missing_information=missing_information,
            reproduction_recipe=reproduction_recipe,
            acceptance_oracles=acceptance_oracles,
            candidate_path_hints=path_hints,
        )
        task_payload = {
            "theme": theme,
            "index": index,
            "support_refs": support_refs,
            "candidate_path_hints": path_hints,
            "acceptance_oracles": acceptance_oracles,
            "contract_dimensions": contract_dimensions,
            "boundary_conditions": boundary_conditions,
        }
        upgraded.append(
            DevelopmentTaskSpecV2(
                task_id=str(
                    spec.get("intent_id")
                    or f"task_{_slug_label(theme)}_{stable_hash(task_payload)[:10]}"
                )[:120],
                source_theme=theme,
                task_type=str(spec.get("task_type") or _task_type_for_theme(theme)),
                public_evidence_refs=support_refs,
                user_pain=str(spec.get("user_pain") or _default_user_pain(theme))[:500],
                observed_behavior=observed_behavior,
                expected_behavior=expected_behavior,
                non_goals=non_goals,
                reproduction_recipe=reproduction_recipe,
                reproduction_expected_failure=_optional_string(
                    spec.get("reproduction_expected_failure")
                )
                or (
                    "The base workspace should lack the proposed capability."
                    if theme in CREATIVE_FEATURE_THEMES
                    else "The base workspace should expose the reported friction."
                ),
                acceptance_oracles=acceptance_oracles,
                repo_entrypoints=repo_entrypoints,
                candidate_path_hints=path_hints,
                relevant_symbols_hint=relevant_symbols_hint,
                ambiguity_level=_ambiguity_level(missing_information),
                missing_information=missing_information,
                spec_maturity=spec_maturity,
                risk_level=str(spec.get("risk_level") or _risk_level_for_theme(theme)),
                behavior_surface=behavior_surface,
                contract_dimensions=contract_dimensions,
                boundary_conditions=boundary_conditions,
                source_explicit_intent_spec=explicit_intent_spec,
            )
        )
    return tuple(upgraded)


def build_spec_maturity_gate(
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    *,
    minimum_maturity: str = "testable",
) -> dict[str, Any]:
    ready_specs = tuple(
        spec
        for spec in task_specs
        if _maturity_rank(spec.spec_maturity) >= _maturity_rank(minimum_maturity)
    )
    blocked_specs = tuple(spec for spec in task_specs if spec not in ready_specs)
    return {
        "minimum_maturity": minimum_maturity,
        "task_count": len(task_specs),
        "ready_task_count": len(ready_specs),
        "blocked_task_count": len(blocked_specs),
        "ready_task_ids": tuple(spec.task_id for spec in ready_specs),
        "blocked_task_ids": tuple(spec.task_id for spec in blocked_specs),
        "blocked_reasons": {
            spec.task_id: spec.missing_information for spec in blocked_specs
        },
        "patch_loop_entry": "ready" if ready_specs else "maturation_required",
    }


def build_repo_intelligence_pack(
    workspace_root: Path,
    *,
    themes: tuple[str, ...] = (),
    task_specs: Sequence[DevelopmentTaskSpecV2 | Mapping[str, Any]] = (),
    max_files: int = 400,
) -> RepoIntelligencePack:
    """Build a compact, deterministic repository context for patch landing."""

    root = workspace_root.resolve()
    text_files = _workspace_text_files(root, max_files=max_files)
    file_tree = tuple(path for path, _ in text_files)
    package_manager = _detect_package_manager(root)
    test_commands, lint_commands = _detect_project_commands(root, file_tree)
    entrypoints = _detect_entrypoints(root)
    source_roots = tuple(
        dirname
        for dirname in ("src", "lib", "packages", "dist-src", "dist")
        if (root / dirname).exists()
    )
    test_files = tuple(path for path in file_tree if _is_test_or_repro_path(path))
    config_files = tuple(path for path in file_tree if _is_config_path(path))
    docs_files = tuple(path for path in file_tree if _is_docs_path(path))
    symbol_index = _build_symbol_index(text_files)
    test_map = _build_test_map(file_tree)
    active_themes = themes or ("product_improvement",)
    candidate_files_by_theme = {
        theme: tuple(
            context.path
            for context in select_workspace_files(
                root,
                (theme,),
                task_specs=task_specs,
                max_files=12,
            )
        )
        for theme in active_themes
    }
    pack_payload = {
        "file_tree": file_tree,
        "package_manager": package_manager,
        "entrypoints": entrypoints,
        "symbols": symbol_index,
        "test_files": test_files,
    }
    return RepoIntelligencePack(
        repo_hash=stable_hash(pack_payload),
        file_tree=file_tree,
        package_manager=package_manager,
        test_commands=test_commands,
        lint_commands=lint_commands,
        entrypoints=entrypoints,
        source_roots=source_roots,
        test_files=test_files,
        config_files=config_files,
        docs_files=docs_files,
        symbol_index=symbol_index,
        test_map=test_map,
        candidate_files_by_theme=candidate_files_by_theme,
        style_notes=_style_notes_for_repo(
            package_manager, test_commands, lint_commands
        ),
    )


def build_reproducer_plan(
    *,
    source_report: Mapping[str, Any],
    workspace_root: Path,
    task_specs: tuple[DevelopmentTaskSpecV2, ...] | None = None,
    repo_intelligence: RepoIntelligencePack | None = None,
    confirm_base: bool = False,
    timeout_seconds: float = 20.0,
    executor: CommandExecutor | None = None,
) -> ReproducerPlan:
    """Select or confirm base-workspace reproduction checks before patching."""

    specs = task_specs or build_development_task_specs_v2(
        source_report,
        repo_intelligence=repo_intelligence,
    )
    artifacts: list[ReproArtifact] = []
    executable_commands: list[str] = []
    seen_commands: set[str] = set()
    for spec in specs:
        spec_oracles = tuple(
            oracle
            for oracle in spec.acceptance_oracles
            if oracle.command
            and oracle.command not in seen_commands
            and _command_referenced_paths_exist(oracle.command, workspace_root)
        )
        if not spec_oracles:
            continue
        for oracle in spec_oracles:
            command = oracle.command
            assert command is not None
            seen_commands.add(command)
            status = "unconfirmed"
            observed = ""
            expected_on_base = _oracle_expected_on_base(oracle)
            if confirm_base:
                with CandidateWorkspace.create(
                    workspace_root
                ) as verification_workspace:
                    verification = run_workspace_verification(
                        verification_workspace.root,
                        (command,),
                        timeout_seconds=timeout_seconds,
                        executor=executor,
                    )[0]
                observed = "\n".join(
                    part
                    for part in (verification.stdout_tail, verification.stderr_tail)
                    if part
                )[-2400:]
                status = _base_reproducer_status(
                    verification,
                    expected_on_base=expected_on_base,
                )
            artifact_payload = {
                "task_id": spec.task_id,
                "command": command,
                "status": status,
                "observed": observed,
                "source": oracle.source,
                "expected_on_base": expected_on_base,
            }
            artifacts.append(
                ReproArtifact(
                    task_id=spec.task_id,
                    kind=_oracle_kind_for_command(command),
                    files_created=(),
                    command=command,
                    expected_on_base=expected_on_base,
                    observed_on_base=observed,
                    reproducibility_status=status,
                    evidence_hash=stable_hash(artifact_payload),
                    source=oracle.source,
                )
            )
            if oracle.required and status not in {"invalid", "preservation_failed"}:
                executable_commands.append(command)
    missing_repro_tasks = tuple(
        spec.task_id
        for spec in specs
        if not any(
            artifact.task_id == spec.task_id
            and artifact.expected_on_base == "fail"
            and artifact.reproducibility_status != "invalid"
            for artifact in artifacts
        )
    )
    plan_payload = {
        "report_id": source_report.get("report_id"),
        "artifacts": artifacts,
        "missing_repro_tasks": missing_repro_tasks,
    }
    return ReproducerPlan(
        plan_id=f"reproducer_plan_{stable_hash(plan_payload)[:24]}",
        artifacts=tuple(artifacts),
        executable_commands=tuple(dict.fromkeys(executable_commands)),
        confirmed_count=sum(
            1
            for artifact in artifacts
            if artifact.reproducibility_status == "confirmed"
        ),
        blocked_count=sum(
            1 for artifact in artifacts if artifact.reproducibility_status == "invalid"
        ),
        missing_repro_tasks=missing_repro_tasks,
    )


def _oracle_expected_on_base(oracle: AcceptanceOracle) -> str:
    if oracle.expected_on_base in {"fail", "pass"}:
        return oracle.expected_on_base
    return "pass" if oracle.witness_role == "preservation" else "fail"


def _base_reproducer_status(
    result: WorkspaceVerificationResult,
    *,
    expected_on_base: str,
) -> str:
    if _verification_environment_unavailable(result):
        return "invalid"
    if expected_on_base == "pass":
        return (
            "preservation_confirmed"
            if result.status == "passed"
            else ("preservation_failed")
        )
    return "not_reproduced" if result.status == "passed" else "confirmed"


def _verification_environment_unavailable(result: WorkspaceVerificationResult) -> bool:
    if result.status in {"blocked", "timeout", "infra_error"}:
        return True
    combined = "\n".join(
        part
        for part in (
            result.stdout_tail,
            result.stderr_tail,
            result.blocked_reason or "",
        )
        if part
    ).casefold()
    posix_command_not_found = any(
        line.strip().endswith(": not found") for line in combined.splitlines()
    )
    missing_command = result.exit_code in {126, 127} and (
        "command not found" in combined
        or "not recognized as an internal or external command" in combined
        or posix_command_not_found
    )
    infrastructure_markers = (
        "could not determine executable to run",
        "missing script:",
        "err_pnpm_no_script",
        "no module named pytest",
        "no module named 'pytest'",
        "error: no test specified",
        "cannot connect to the docker daemon",
        "is the docker daemon running",
        "docker_daemon_unavailable",
    )
    return missing_command or any(
        marker in combined for marker in infrastructure_markers
    )


def _project_command_baseline(
    *,
    workspace_root: Path,
    commands: tuple[str, ...],
    timeout_seconds: float,
    executor: CommandExecutor,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    eligible_commands: list[str] = []
    for command in commands:
        with CandidateWorkspace.create(workspace_root) as baseline_workspace:
            (verification,) = run_workspace_verification(
                baseline_workspace.root,
                (command,),
                timeout_seconds=timeout_seconds,
                executor=executor,
            )
        if verification.status == "passed":
            classification = "eligible"
            eligible_commands.append(command)
        elif _verification_environment_unavailable(verification):
            classification = "unavailable"
        else:
            classification = "baseline_failed"
        results.append(
            {
                "command": command,
                "classification": classification,
                "status": verification.status,
                "exit_code": verification.exit_code,
                "elapsed_sec": verification.elapsed_sec,
                "stdout_tail": verification.stdout_tail,
                "stderr_tail": verification.stderr_tail,
                "blocked_reason": verification.blocked_reason,
                "backend": verification.backend,
                "runtime_ref": verification.runtime_ref,
            }
        )
    return {
        "eligible_commands": tuple(eligible_commands),
        "excluded_commands": tuple(
            item["command"] for item in results if item["classification"] != "eligible"
        ),
        "results": tuple(results),
    }


def _with_project_command_baseline(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    enriched = dict(summary)
    environment = dict(_mapping_get(summary, "environment", {}))
    environment["project_command_baseline"] = canonicalize(baseline)
    results = tuple(baseline.get("results", ()))
    eligible_count = len(tuple(baseline.get("eligible_commands", ())))
    if not results:
        verification_status = "not_configured"
    elif eligible_count == len(results):
        verification_status = "ready"
    elif eligible_count:
        verification_status = "partial"
    elif any(item.get("classification") == "baseline_failed" for item in results):
        verification_status = "baseline_failing"
    else:
        verification_status = "unavailable"
    environment["project_verification_status"] = verification_status
    environment["eligible_project_verification_commands"] = tuple(
        baseline.get("eligible_commands", ())
    )
    enriched["environment"] = environment
    return enriched


def _with_execution_capabilities(
    summary: Mapping[str, Any],
    available_executables: tuple[str, ...],
) -> dict[str, Any]:
    enriched = dict(summary)
    environment = dict(_mapping_get(summary, "environment", {}))
    environment["available_verification_executables"] = available_executables
    enriched["environment"] = environment
    return enriched


def _project_command_baseline_from_summary(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    environment = _mapping_get(summary, "environment", {})
    return dict(_mapping_get(environment, "project_command_baseline", {}))


def build_engineering_organization_plan(
    *,
    source_report: Mapping[str, Any],
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    repo_intelligence: RepoIntelligencePack,
    reproducer_plan: ReproducerPlan,
) -> EngineeringOrganizationPlan:
    """Create role-owned engineering work packages before patch generation."""

    del source_report
    themes = tuple(dict.fromkeys(spec.source_theme for spec in task_specs))
    architecture_delta_required = _requires_architecture_delta(task_specs)
    public_api_delta_required = _requires_public_api_delta(task_specs)
    developer_experience_delta_required = _requires_developer_experience_delta(
        task_specs
    )
    security_review_required = any(
        spec.risk_level == "high"
        or spec.source_theme in {"path_security", "dependency_resolution"}
        or has_security_surface(
            spec.behavior_surface, (spec.source_theme, spec.task_type)
        )
        for spec in task_specs
    )
    role_specs = _engineering_role_specs(
        architecture_delta_required=architecture_delta_required,
        public_api_delta_required=public_api_delta_required,
        developer_experience_delta_required=developer_experience_delta_required,
        security_review_required=security_review_required,
    )
    work_packages = tuple(
        _build_engineering_work_package(
            role=role,
            mission=mission,
            gate=gate,
            task_specs=task_specs,
            repo_intelligence=repo_intelligence,
            reproducer_plan=reproducer_plan,
            index=index,
        )
        for index, (role, mission, gate) in enumerate(role_specs)
    )
    release_strategy = (
        "architecture_first"
        if architecture_delta_required or public_api_delta_required
        else "behavior_first"
    )
    plan_payload = {
        "themes": themes,
        "release_strategy": release_strategy,
        "work_packages": work_packages,
        "repo_hash": repo_intelligence.repo_hash,
        "reproducer_plan": reproducer_plan.plan_id,
    }
    return EngineeringOrganizationPlan(
        plan_id=f"engineering_org_plan_{stable_hash(plan_payload)[:24]}",
        release_strategy=release_strategy,
        architecture_delta_required=architecture_delta_required,
        public_api_delta_required=public_api_delta_required,
        developer_experience_delta_required=developer_experience_delta_required,
        security_review_required=security_review_required,
        work_packages=work_packages,
        execution_order=tuple(package.package_id for package in work_packages),
        verifier_contract=(
            "Run reproducer or behavior-level acceptance commands before syntax-only checks.",
            "Check public API compatibility when API-surface packages are present.",
            "Check architecture-delta evidence when architecture packages are present.",
            "Keep evidence claims bounded to passed gates and changed workspace files.",
        ),
        evidence_boundary=(
            "The plan coordinates engineering roles for patch generation; it is not "
            "evidence of behavioral equivalence until verifier packages pass."
        ),
    )


def select_workspace_files(
    root: Path,
    themes: tuple[str, ...],
    *,
    task_specs: Sequence[DevelopmentTaskSpecV2 | Mapping[str, Any]] = (),
    max_files: int = 12,
    max_chars_per_file: int = 12_000,
    max_file_bytes: int = 180_000,
) -> tuple[WorkspaceFileContext, ...]:
    workspace_root = root.resolve()
    keywords = _keywords_for_themes(themes, task_specs)
    contexts: list[WorkspaceFileContext] = []
    for path in sorted(workspace_root.rglob("*")):
        if _is_ignored_workspace_path(path, workspace_root):
            continue
        relative_path = path.relative_to(workspace_root).as_posix()
        try:
            raw = read_regular_file_bytes(
                workspace_root,
                path,
                max_bytes=max_file_bytes,
            )
        except (OSError, UnsafeRegularFileError):
            continue
        if b"\x00" in raw[:4096]:
            continue
        text = raw.decode("utf-8", errors="replace")
        score = _file_relevance_score(relative_path, text, keywords)
        if score <= 0:
            continue
        excerpt = text[:max_chars_per_file]
        contexts.append(
            WorkspaceFileContext(
                path=relative_path,
                content_hash=stable_hash(text),
                excerpt=excerpt,
                truncated=len(text) > len(excerpt),
                byte_size=len(raw),
                score=score,
            )
        )
    contexts.sort(key=lambda item: (-item.score, item.path))
    return tuple(contexts[:max_files])


def select_agent_context_files(
    root: Path,
    themes: tuple[str, ...],
    *,
    repo_intelligence: RepoIntelligencePack,
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    engineering_org_plan: EngineeringOrganizationPlan,
    max_files: int,
) -> tuple[WorkspaceFileContext, ...]:
    """Build a role- and task-aware file context for the workspace coding agent."""

    if max_files <= 0:
        return ()
    workspace_root = root.resolve()
    keywords = _keywords_for_themes(themes, task_specs)
    broad_contexts = select_workspace_files(
        workspace_root,
        themes,
        task_specs=task_specs,
        max_files=max(max_files * 3, max_files),
    )
    contexts_by_path = {context.path: context for context in broad_contexts}
    ordered_paths: list[str] = []
    for spec in task_specs:
        if spec.source_theme not in themes:
            continue
        ordered_paths.extend(spec.candidate_path_hints[:8])
        for oracle in spec.acceptance_oracles[:4]:
            if oracle.command:
                ordered_paths.extend(_paths_from_command(oracle.command))
    ordered_paths.extend(repo_intelligence.entrypoints[:8])
    for theme in themes:
        ordered_paths.extend(
            repo_intelligence.candidate_files_by_theme.get(theme, ())[:6]
        )
    for package in engineering_org_plan.work_packages:
        if not set(package.source_themes).intersection(themes):
            continue
        ordered_paths.extend(package.candidate_paths[:8])
    ordered_paths.extend(repo_intelligence.config_files[:6])
    ordered_paths.extend(repo_intelligence.test_files[:8])
    ordered_paths.extend(repo_intelligence.docs_files[:4])
    ordered_paths.extend(context.path for context in broad_contexts)

    selected: list[WorkspaceFileContext] = []
    seen: set[str] = set()
    for raw_path in ordered_paths:
        path = Path(str(raw_path)).as_posix()
        if not path or path in seen:
            continue
        context = contexts_by_path.get(path)
        if context is None or context.truncated:
            focused_context = _workspace_file_context_for_path(
                workspace_root,
                path,
                keywords=keywords,
            )
            if focused_context is not None:
                context = focused_context
        if context is None:
            continue
        seen.add(context.path)
        selected.append(context)
        if len(selected) >= max_files:
            break
    return tuple(selected)


def _workspace_file_context_for_path(
    workspace_root: Path,
    relative_path: str,
    *,
    keywords: tuple[str, ...],
    max_chars_per_file: int = 12_000,
    max_file_bytes: int = 180_000,
) -> WorkspaceFileContext | None:
    root = workspace_root.resolve()
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return None
    path = root / relative
    if _is_ignored_workspace_path(path, root):
        return None
    try:
        raw = read_regular_file_bytes(root, path, max_bytes=max_file_bytes)
    except (OSError, UnsafeRegularFileError):
        return None
    if b"\x00" in raw[:4096]:
        return None
    text = raw.decode("utf-8", errors="replace")
    normalized_path = relative.as_posix()
    score = _file_relevance_score(normalized_path, text, keywords)
    if score <= 0:
        score = 0.1
    excerpt = _keyword_relevant_excerpt(
        text,
        keywords=keywords,
        max_chars=max_chars_per_file,
    )
    return WorkspaceFileContext(
        path=normalized_path,
        content_hash=stable_hash(text),
        excerpt=excerpt,
        truncated=len(text) > len(excerpt),
        byte_size=len(raw),
        score=score,
    )


def _keyword_relevant_excerpt(
    text: str,
    *,
    keywords: tuple[str, ...],
    max_chars: int,
) -> str:
    if len(text) <= max_chars:
        return text
    lines = text.splitlines(keepends=True)
    normalized_keywords = tuple(
        keyword.casefold() for keyword in keywords if len(keyword.strip()) >= 3
    )
    scored_matches: list[tuple[int, int]] = []
    for line_index, line in enumerate(lines):
        lowered = line.casefold()
        score = sum(
            len(normalized_keywords) - keyword_index
            for keyword_index, keyword in enumerate(normalized_keywords)
            if keyword in lowered
        )
        if score:
            scored_matches.append((score, line_index))
    if not scored_matches:
        return text[:max_chars]
    selected_lines = {
        line_index
        for _, line_index in sorted(
            scored_matches,
            key=lambda item: (-item[0], item[1]),
        )[:24]
    }
    windows: list[tuple[int, int]] = [(0, min(32, len(lines)))]
    for line_index in sorted(selected_lines):
        start = max(0, line_index - 10)
        end = min(len(lines), line_index + 11)
        if windows and start <= windows[-1][1] + 2:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    chunks: list[str] = []
    for start, end in windows:
        header = f"/* lines {start + 1}-{end}; read_file before editing */\n"
        chunk = header + "".join(lines[start:end])
        if chunks and sum(len(item) for item in chunks) + len(chunk) > max_chars:
            continue
        chunks.append(chunk)
    return "\n".join(chunks)[:max_chars]


def _file_contexts_with_retry_failure_priority(
    root: Path,
    file_contexts: tuple[WorkspaceFileContext, ...],
    previous_verification: tuple[WorkspaceVerificationResult, ...],
    *,
    themes: tuple[str, ...],
    task_specs: Sequence[DevelopmentTaskSpecV2 | Mapping[str, Any]] = (),
) -> tuple[WorkspaceFileContext, ...]:
    retry_paths = _retry_failure_paths(root, previous_verification)
    if not retry_paths:
        return file_contexts
    workspace_root = root.resolve()
    keywords = _keywords_for_themes(themes, task_specs)
    contexts_by_path = {context.path: context for context in file_contexts}
    prioritized: list[WorkspaceFileContext] = []
    for blocked_path in retry_paths:
        context = _workspace_file_context_for_path(
            workspace_root,
            blocked_path,
            keywords=keywords,
            max_chars_per_file=_workspace_coding_retry_excerpt_chars(),
        )
        if context is None:
            context = contexts_by_path.get(blocked_path)
        if context is None:
            continue
        prioritized.append(
            replace(
                context,
                score=max(context.score, 1_000_000.0),
                context_role="retry_failure",
            )
        )
    if not prioritized:
        return file_contexts
    seen = {context.path for context in prioritized}
    return tuple(
        [
            *prioritized,
            *(context for context in file_contexts if context.path not in seen),
        ]
    )


def _blocked_patch_paths(
    previous_verification: tuple[WorkspaceVerificationResult, ...],
) -> tuple[str, ...]:
    paths: list[str] = []
    for result in previous_verification:
        if result.status != "blocked" or not result.blocked_reason:
            continue
        if "replace_fragment_not_found" not in result.blocked_reason:
            continue
        prefix = "apply_workspace_patch:"
        if not result.command.startswith(prefix):
            continue
        parts = result.command.split(":", 2)
        if len(parts) != 3:
            continue
        path = Path(parts[2]).as_posix()
        if path and path not in paths:
            paths.append(path)
    return tuple(paths)


def _retry_failure_paths(
    root: Path,
    previous_verification: tuple[WorkspaceVerificationResult, ...],
) -> tuple[str, ...]:
    workspace_root = root.resolve()
    candidates: list[str] = [*_blocked_patch_paths(previous_verification)]
    path_pattern = re.compile(r"(?:[A-Za-z0-9_.@+\-]+/)+(?:[A-Za-z0-9_.@+\-]+)")
    for result in previous_verification:
        if result.status == "passed":
            continue
        candidates.extend(_paths_from_command(result.command))
        for payload in (
            result.stdout_tail,
            result.stderr_tail,
            result.blocked_reason or "",
        ):
            candidates.extend(
                match.group(0) for match in path_pattern.finditer(payload)
            )
    normalized: list[str] = []
    for candidate in candidates:
        value = candidate.strip("'\"()[]{}:, ").removeprefix("file://")
        value = re.sub(r":\d+(?::\d+)?$", "", value)
        path = Path(value)
        if path.is_absolute():
            try:
                relative = path.resolve().relative_to(workspace_root)
            except (OSError, ValueError):
                continue
        else:
            relative = Path(value.removeprefix("./"))
        if not relative.parts or ".." in relative.parts:
            continue
        target = workspace_root / relative
        if not target.exists() or target.is_symlink() or not target.is_file():
            continue
        rendered = relative.as_posix()
        if rendered not in normalized:
            normalized.append(rendered)
    return tuple(normalized)


def _paths_from_command(command: str) -> tuple[str, ...]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    paths: list[str] = []
    for token in tokens:
        normalized = token.strip("'\"").replace("\\", "/")
        if normalized.startswith("-") or any(
            char in normalized for char in (";", "{", "}")
        ):
            continue
        if "/" not in normalized and not normalized.endswith(
            (".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".json")
        ):
            continue
        paths.append(normalized)
    return tuple(dict.fromkeys(paths))


def _start_controller_event_run(
    *,
    event_store: SQLiteEventStore | None,
    source_report: Mapping[str, Any],
    repo_digest: str,
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    agent: WorkspaceCodingAgent,
    config: Mapping[str, Any],
    attempt: int,
    committed_receipt: WorkspaceCommitReceipt | None,
) -> tuple[RunRecord | None, CodingAgentLoopResult | None]:
    if event_store is None:
        return None, None
    if attempt < 1:
        raise ValueError("event_attempt_must_be_positive")
    report_id = _optional_string(source_report.get("report_id")) or "workspace_report"
    task_id = f"{report_id}:controller"
    spec_hash = stable_hash(
        {
            "source_report": source_report,
            "controller_event_schema": 2,
        }
    )
    model_snapshot = str(agent.model or agent.source)
    config_hash = stable_hash(config)
    existing = event_store.find_intent_run(
        task_id=task_id,
        spec_hash=spec_hash,
        model_snapshot=model_snapshot,
        config_hash=config_hash,
        attempt=attempt,
    )
    if existing is not None:
        if existing.status == "running":
            finalized = _recover_unfinalized_terminal_controller_result(
                event_store=event_store,
                run=existing,
                current_repo_digest=repo_digest,
            )
            if finalized is not None:
                return existing, finalized
        if existing.status != "running":
            replayed = _replay_completed_controller_result(
                event_store=event_store,
                run=existing,
                current_repo_digest=repo_digest,
            )
            return existing, replayed
        event_store.load_verified_events(existing.run_id)
        if existing.repo_digest != repo_digest:
            recovered = _recover_promoted_controller_result(
                event_store=event_store,
                run=existing,
                current_repo_digest=repo_digest,
                committed_receipt=committed_receipt,
            )
            if recovered is not None:
                return existing, recovered
            raise RuntimeError(
                f"controller_resume_workspace_digest_mismatch:{existing.run_id}"
            )
        run = existing
    else:
        if committed_receipt is not None:
            raise RuntimeError("orphaned_committed_promotion_receipt")
        run = event_store.start_run(
            task_id=task_id,
            repo_digest=repo_digest,
            spec_hash=spec_hash,
            model_snapshot=model_snapshot,
            config_hash=config_hash,
            attempt=attempt,
        )
    event_store.append_event(
        run_id=run.run_id,
        event_type="loop_started",
        payload=_stable_controller_event_payload(
            {
                "report_id": report_id,
                "repo_digest": repo_digest,
                "agent_source": agent.source,
                "agent_model": agent.model,
                "task_specs_hash": stable_hash(task_specs),
                "config": config,
            }
        ),
        idempotency_key="loop_started",
    )
    return run, None


def _stable_controller_event_payload(value: Any) -> Any:
    normalized = canonicalize(value)
    if isinstance(normalized, Mapping):
        return {
            str(key): (
                0.0
                if str(key) in _VOLATILE_CONTROLLER_EVENT_NUMERIC_FIELDS
                else _stable_controller_event_payload(item)
            )
            for key, item in normalized.items()
        }
    if isinstance(normalized, list):
        return [_stable_controller_event_payload(item) for item in normalized]
    return normalized


def _replay_safe_controller_event_payload(
    *,
    event_store: SQLiteEventStore,
    run: RunRecord,
    event_type: str,
    idempotency_key: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    candidate_payload = canonicalize(payload)
    existing = next(
        (
            event
            for event in event_store.load_verified_events(run.run_id)
            if event.idempotency_key == idempotency_key
        ),
        None,
    )
    if existing is None:
        return candidate_payload
    if existing.event_type != event_type or stable_hash(
        _stable_controller_event_payload(existing.payload)
    ) != stable_hash(_stable_controller_event_payload(candidate_payload)):
        raise ValueError("event_idempotency_conflict")
    return existing.payload


def _record_controller_event(
    *,
    event_store: SQLiteEventStore | None,
    run: RunRecord | None,
    event_type: str,
    payload: Mapping[str, Any],
    idempotency_key: str,
) -> None:
    if event_store is None or run is None:
        return
    replay_safe_payload = _replay_safe_controller_event_payload(
        event_store=event_store,
        run=run,
        event_type=event_type,
        idempotency_key=idempotency_key,
        payload=payload,
    )
    event_store.append_event(
        run_id=run.run_id,
        event_type=event_type,
        payload=replay_safe_payload,
        idempotency_key=idempotency_key,
    )


def _controller_event_chain_head(
    *,
    event_store: SQLiteEventStore | None,
    run: RunRecord | None,
) -> str | None:
    if event_store is None or run is None:
        return None
    events = event_store.load_events(run.run_id)
    return events[-1].event_hash if events else None


def _complete_controller_event_run(
    *,
    event_store: SQLiteEventStore | None,
    run: RunRecord | None,
    result: CodingAgentLoopResult,
    workspace_root: Path,
) -> None:
    if event_store is None or run is None:
        return
    payload = {
        "result": canonicalize(result),
        "post_repo_digest": build_workspace_execution_profile(workspace_root).repo_hash,
    }
    replay_safe_payload = _replay_safe_controller_event_payload(
        event_store=event_store,
        run=run,
        event_type="loop_completed",
        idempotency_key="loop_completed",
        payload=payload,
    )
    event_store.append_terminal_event(
        run_id=run.run_id,
        event_type="loop_completed",
        payload=replay_safe_payload,
        idempotency_key="loop_completed",
        status=result.evidence_level,
    )


def _replay_completed_controller_result(
    *,
    event_store: SQLiteEventStore,
    run: RunRecord,
    current_repo_digest: str,
) -> CodingAgentLoopResult:
    events = event_store.load_verified_events(
        run.run_id,
        terminal_event_type="loop_completed",
    )
    terminal = next(
        (event for event in reversed(events) if event.event_type == "loop_completed"),
        None,
    )
    if terminal is None:
        raise ValueError("completed_controller_missing_terminal_event")
    expected_repo_digest = str(terminal.payload.get("post_repo_digest") or "")
    if not expected_repo_digest:
        raise ValueError("completed_controller_missing_post_repo_digest")
    if expected_repo_digest != current_repo_digest:
        raise RuntimeError(f"controller_replay_workspace_digest_mismatch:{run.run_id}")
    raw_result = terminal.payload.get("result")
    if not isinstance(raw_result, Mapping):
        raise ValueError("completed_controller_missing_result")
    result = _coding_agent_loop_result_from_payload(raw_result)
    if result.evidence_level != run.status:
        raise ValueError("completed_controller_status_mismatch")
    return result


def _recover_unfinalized_terminal_controller_result(
    *,
    event_store: SQLiteEventStore,
    run: RunRecord,
    current_repo_digest: str,
) -> CodingAgentLoopResult | None:
    events = event_store.load_verified_events(run.run_id)
    if not events or events[-1].event_type != "loop_completed":
        return None
    terminal = events[-1]
    expected_repo_digest = str(terminal.payload.get("post_repo_digest") or "")
    if not expected_repo_digest:
        raise ValueError("completed_controller_missing_post_repo_digest")
    if expected_repo_digest != current_repo_digest:
        raise RuntimeError(f"controller_replay_workspace_digest_mismatch:{run.run_id}")
    raw_result = terminal.payload.get("result")
    if not isinstance(raw_result, Mapping):
        raise ValueError("completed_controller_missing_result")
    result = _coding_agent_loop_result_from_payload(raw_result)
    event_store.complete_run(
        run.run_id,
        status=result.evidence_level,
        result_hash=terminal.payload_hash,
    )
    return result


def _recover_promoted_controller_result(
    *,
    event_store: SQLiteEventStore,
    run: RunRecord,
    current_repo_digest: str,
    committed_receipt: WorkspaceCommitReceipt | None,
) -> CodingAgentLoopResult | None:
    events = event_store.load_verified_events(run.run_id)
    promotion_event = next(
        (
            event
            for event in reversed(events)
            if event.event_type == "promotion_completed"
            and str(event.payload.get("post_repo_digest") or "") == current_repo_digest
        ),
        None,
    )
    raw_result: object | None = None
    if promotion_event is not None:
        raw_result = promotion_event.payload.get("recovery_result")
    elif committed_receipt is not None:
        promotion_event = next(
            (
                event
                for event in reversed(events)
                if event.event_type == "promotion_started"
                and str(event.payload.get("baseline_digest") or "")
                == committed_receipt.baseline_digest
                and str(event.payload.get("candidate_digest") or "")
                == committed_receipt.candidate_digest
            ),
            None,
        )
        if promotion_event is not None:
            raw_result = promotion_event.payload.get("recovery_result_on_commit")
    if promotion_event is None:
        return None
    if not isinstance(raw_result, Mapping):
        return None
    result = _coding_agent_loop_result_from_payload(raw_result)
    payload = {
        "result": canonicalize(result),
        "post_repo_digest": current_repo_digest,
        "recovered_from_event": promotion_event.event_id,
    }
    replay_safe_payload = _replay_safe_controller_event_payload(
        event_store=event_store,
        run=run,
        event_type="loop_completed",
        idempotency_key="loop_completed",
        payload=payload,
    )
    event_store.append_terminal_event(
        run_id=run.run_id,
        event_type="loop_completed",
        payload=replay_safe_payload,
        idempotency_key="loop_completed",
        status=result.evidence_level,
    )
    return result


def _coding_agent_loop_result_from_payload(
    raw: Mapping[str, Any],
) -> CodingAgentLoopResult:
    package = WorkspacePackageInfo(**dict(raw["package"]))
    proposals: list[CodingAgentPatchProposal] = []
    for item in raw.get("proposals", ()):
        if not isinstance(item, Mapping):
            raise ValueError("invalid_persisted_controller_proposal")
        proposal = dict(item)
        proposal["patches"] = tuple(
            WorkspaceFilePatch(**dict(patch))
            for patch in proposal.get("patches", ())
            if isinstance(patch, Mapping)
        )
        for key in (
            "themes",
            "verification_commands",
            "selected_files",
            "support_refs",
        ):
            proposal[key] = tuple(str(value) for value in proposal.get(key, ()))
        proposals.append(CodingAgentPatchProposal(**proposal))
    patch_results = tuple(
        WorkspacePatchResult(**dict(item))
        for item in raw.get("patch_results", ())
        if isinstance(item, Mapping)
    )
    verification_results = tuple(
        WorkspaceVerificationResult(**dict(item))
        for item in raw.get("verification_results", ())
        if isinstance(item, Mapping)
    )
    evidence_records = tuple(
        VerificationEvidence(**dict(item))
        for item in raw.get("evidence_records", ())
        if isinstance(item, Mapping)
    )
    return CodingAgentLoopResult(
        run_id=str(raw["run_id"]),
        developer_mode=str(raw["developer_mode"]),
        source_report_id=(
            str(raw["source_report_id"])
            if raw.get("source_report_id") is not None
            else None
        ),
        package=package,
        selected_files=tuple(str(item) for item in raw.get("selected_files", ())),
        proposals=tuple(proposals),
        patch_results=patch_results,
        verification_results=verification_results,
        iterations=int(raw.get("iterations") or 0),
        final_mode=str(raw.get("final_mode") or "blocked"),
        failed_commands=tuple(str(item) for item in raw.get("failed_commands", ())),
        provenance_hash=str(raw["provenance_hash"]),
        privacy_boundary=str(
            raw.get("privacy_boundary") or "public_report_and_workspace_files_only"
        ),
        development_stage=str(
            raw.get("development_stage") or "guarded_promotion_agent"
        ),
        exploration_attempts=int(raw.get("exploration_attempts") or 0),
        risky_patch_bundles=int(raw.get("risky_patch_bundles") or 0),
        promoted_candidates=int(raw.get("promoted_candidates") or 0),
        verified_effective_patches=int(raw.get("verified_effective_patches") or 0),
        rejected_by_gate=int(raw.get("rejected_by_gate") or 0),
        repair_iterations=int(raw.get("repair_iterations") or 0),
        reasoning_effort=(
            str(raw["reasoning_effort"])
            if raw.get("reasoning_effort") is not None
            else None
        ),
        requested_reasoning_effort=(
            str(raw["requested_reasoning_effort"])
            if raw.get("requested_reasoning_effort") is not None
            else None
        ),
        run_config=dict(raw.get("run_config") or {}),
        patch_coverage=dict(raw.get("patch_coverage") or {}),
        code_landing_metrics=dict(raw.get("code_landing_metrics") or {}),
        evidence_level=str(raw.get("evidence_level") or "proposal_ready"),
        evidence_hash=(
            str(raw["evidence_hash"]) if raw.get("evidence_hash") is not None else None
        ),
        evidence_summary=dict(raw.get("evidence_summary") or {}),
        evidence_records=evidence_records,
    )


def _last_verified_candidate_record(
    *,
    canonical_workspace_root: Path,
    promotion_records: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    for promotion_index in range(len(promotion_records) - 1, -1, -1):
        record = promotion_records[promotion_index]
        if str(record.get("status") or "") != "release_candidate":
            continue
        promotion_digest = str(
            record.get("promoted_digest") or record.get("candidate_digest") or ""
        )
        if not promotion_digest:
            continue
        return {
            "workspace_root": str(canonical_workspace_root.resolve()),
            "candidate_repo_digest": build_workspace_execution_profile(
                canonical_workspace_root
            ).repo_hash,
            "candidate_repo_digest_kind": "workspace_execution_profile_repo_hash",
            "promotion_digest": promotion_digest,
            "promotion_digest_kind": "transaction_file_map_digest",
            "verification_status": "passed",
            "source": "promoted_release_candidate",
            "promotion_index": promotion_index,
            "changed_paths": tuple(record.get("changed_paths", ()) or ()),
        }
    return None


def _build_promoted_recovery_result(
    *,
    canonical_workspace_root: Path,
    report_id: str | None,
    package: WorkspacePackageInfo,
    selected_files: tuple[str, ...],
    proposals: tuple[CodingAgentPatchProposal, ...],
    patch_results: tuple[WorkspacePatchResult, ...],
    verification_results: tuple[WorkspaceVerificationResult, ...],
    controller_rounds: int,
    development_rounds: int,
    infrastructure_only_rounds: int,
    candidate_infra_retry_limit: int,
    candidate_infra_round_retry_limit: int,
    candidate_strategy_retry_limit: int,
    candidate_slot_infra_retry_count: int,
    candidate_slot_exhaustion_count: int,
    unresolved_candidate_slot_exhaustion_count: int,
    candidate_strategy_retry_count: int,
    candidate_strategy_exhaustion_count: int,
    exploration_attempts: int,
    risky_patch_bundles: int,
    promoted_candidates: int,
    rejected_by_gate: int,
    repair_iterations: int,
    agent: WorkspaceCodingAgent,
    event_attempt: int,
    max_iterations: int,
    timeout_seconds: float,
    max_selected_files: int,
    include_agent_verification_commands: bool,
    exploration_iterations: int,
    max_active_themes_per_iteration: int,
    candidate_count: int,
    evaluator_verification_commands: tuple[str, ...],
    required_project_verification_commands: tuple[str, ...],
    generated_coverage_commands: tuple[str, ...],
    covered_requested_themes: tuple[str, ...],
    remaining_requested_themes: tuple[str, ...],
    accepted_verification_commands: tuple[str, ...],
    promotion_records: tuple[dict[str, Any], ...],
    candidate_runtime_records: tuple[dict[str, Any], ...],
    selected_candidate_proposal_ids: tuple[str, ...],
    rejected_candidate_proposal_ids: tuple[str, ...],
    enforce_spec_maturity_gate: bool,
    engineering_organization_plan: EngineeringOrganizationPlan,
    code_max_runtime_summary: Mapping[str, Any],
    code_landing_metrics: dict[str, Any],
    evidence_assessment: Any,
    evidence_records: tuple[VerificationEvidence, ...],
    controller_event_run_id: str,
) -> CodingAgentLoopResult:
    coverage = _patch_coverage_summary(
        patch_results=patch_results,
        verification_results=verification_results,
        generated_coverage_commands=generated_coverage_commands,
        proposals=proposals,
    )
    verified_effective_patches = _verified_effective_patch_count(
        patch_results,
        verification_results=verification_results,
        final_mode="verified",
    )
    evidence_summary = canonicalize(evidence_assessment)
    last_verified_candidate = _last_verified_candidate_record(
        canonical_workspace_root=canonical_workspace_root,
        promotion_records=promotion_records,
    )
    run_config = {
        "event_attempt": event_attempt,
        "max_iterations": max_iterations,
        "timeout_seconds": timeout_seconds,
        "max_selected_files": max_selected_files,
        "include_agent_verification_commands": include_agent_verification_commands,
        "exploration_iterations": exploration_iterations,
        "max_active_themes_per_iteration": max_active_themes_per_iteration,
        "candidate_count": candidate_count,
        "candidate_infrastructure_retry_limit": candidate_infra_retry_limit,
        "candidate_infrastructure_round_retry_limit": (
            candidate_infra_round_retry_limit
        ),
        "candidate_strategy_retry_limit": candidate_strategy_retry_limit,
        "candidate_slot_infra_retry_count": candidate_slot_infra_retry_count,
        "candidate_slot_exhaustion_count": candidate_slot_exhaustion_count,
        "unresolved_candidate_slot_exhaustion_count": (
            unresolved_candidate_slot_exhaustion_count
        ),
        "candidate_strategy_retry_count": candidate_strategy_retry_count,
        "candidate_strategy_exhaustion_count": candidate_strategy_exhaustion_count,
        "candidate_proposal_count": len(proposals),
        "promotion_attempt_count": len(promotion_records),
        "canonical_promotion_count": promoted_candidates,
        "controller_round_count": controller_rounds,
        "development_round_count": development_rounds,
        "infrastructure_only_round_count": infrastructure_only_rounds,
        "controller_event_run_id": controller_event_run_id,
        "evaluator_verification_commands": evaluator_verification_commands,
        "required_project_verification_commands": (
            required_project_verification_commands
        ),
        "generated_coverage_commands": generated_coverage_commands,
        "covered_requested_themes": covered_requested_themes,
        "remaining_requested_themes": remaining_requested_themes,
        "accepted_verification_commands": accepted_verification_commands,
        "promotion_records": promotion_records,
        "last_verified_candidate": last_verified_candidate,
        "candidate_runtime_records": candidate_runtime_records,
        "selected_candidate_proposal_ids": selected_candidate_proposal_ids,
        "rejected_candidate_proposal_ids": rejected_candidate_proposal_ids,
        "enforce_spec_maturity_gate": enforce_spec_maturity_gate,
        "engineering_organization_plan": _engineering_org_plan_agent_summary(
            engineering_organization_plan
        ),
        "project_command_baseline": _project_command_baseline_from_summary(
            code_max_runtime_summary
        ),
        "code_max_runtime": canonicalize(code_max_runtime_summary),
        "recovered_after_committed_promotion": True,
    }
    payload = {
        "source_report_id": report_id,
        "package": package,
        "selected_files": selected_files,
        "proposals": proposals,
        "patch_results": patch_results,
        "verification_results": verification_results,
        "controller_rounds": controller_rounds,
        "run_config": run_config,
        "patch_coverage": coverage,
        "code_landing_metrics": code_landing_metrics,
        "evidence": evidence_summary,
        "evidence_records": evidence_records,
    }
    provenance_hash = stable_hash(_stable_controller_event_payload(payload))
    development_stage = _development_stage(
        final_mode="verified",
        promoted_candidates=promoted_candidates,
        exploration_attempts=exploration_attempts,
        verified_effective_patches=verified_effective_patches,
    )
    return CodingAgentLoopResult(
        run_id=f"workspace_agent_recovered_{provenance_hash[:24]}",
        developer_mode=development_stage,
        source_report_id=report_id,
        package=package,
        selected_files=selected_files,
        proposals=proposals,
        patch_results=patch_results,
        verification_results=verification_results,
        iterations=controller_rounds,
        final_mode="verified",
        failed_commands=tuple(
            result.command
            for result in verification_results
            if result.status != "passed"
        ),
        provenance_hash=provenance_hash,
        development_stage=development_stage,
        exploration_attempts=exploration_attempts,
        risky_patch_bundles=risky_patch_bundles,
        promoted_candidates=promoted_candidates,
        verified_effective_patches=verified_effective_patches,
        rejected_by_gate=rejected_by_gate,
        repair_iterations=repair_iterations,
        reasoning_effort=getattr(agent, "reasoning_effort", None),
        requested_reasoning_effort=getattr(
            agent,
            "requested_reasoning_effort",
            None,
        ),
        run_config=run_config,
        patch_coverage=coverage,
        code_landing_metrics=code_landing_metrics,
        evidence_level=evidence_assessment.level,
        evidence_hash=evidence_assessment.evidence_hash,
        evidence_summary=evidence_summary,
        evidence_records=evidence_records,
    )


def run_coding_agent_development_loop(
    *,
    source_report: Mapping[str, Any],
    workspace_root: Path,
    agent: WorkspaceCodingAgent,
    verification_commands: tuple[str, ...] = (),
    required_project_verification_commands: tuple[str, ...] = (),
    max_iterations: int = DEFAULT_WORKSPACE_CODING_MAX_ITERATIONS,
    timeout_seconds: float = 60.0,
    max_selected_files: int = DEFAULT_WORKSPACE_CODING_MAX_SELECTED_FILES,
    include_agent_verification_commands: bool = True,
    exploration_iterations: int = DEFAULT_WORKSPACE_CODING_EXPLORATION_ITERATIONS,
    enforce_spec_maturity_gate: bool = False,
    max_active_themes_per_iteration: int = DEFAULT_WORKSPACE_CODING_MAX_ACTIVE_THEMES,
    executor: CommandExecutor | None = None,
    candidate_count: int = 1,
    event_store: SQLiteEventStore | None = None,
    event_attempt: int = 1,
) -> CodingAgentLoopResult:
    root = workspace_root.resolve()
    if executor is None:
        raise ValueError("workspace_executor_required")
    command_executor = executor
    available_verification_executables = probe_command_executor_executables(
        command_executor,
        root=root,
        timeout_seconds=min(max(timeout_seconds, 1.0), 30.0),
    )
    committed_receipt = recover_interrupted_workspace_promotion(
        root,
        retain_committed=event_store is not None,
    )
    execution_policy_hash = stable_hash(canonicalize(command_executor.policy))
    package = load_workspace_package_info(root)
    themes = _extract_themes(source_report)
    repo_intelligence = build_repo_intelligence_pack(
        root,
        themes=themes,
        task_specs=build_development_intent_specs(source_report),
    )
    task_specs_v2 = build_development_task_specs_v2(
        source_report,
        repo_intelligence=repo_intelligence,
    )
    spec_maturity_gate = build_spec_maturity_gate(task_specs_v2)
    normalized_candidate_count = max(1, min(int(candidate_count), 8))
    candidate_infra_retry_limit = _workspace_coding_candidate_infra_retries()
    candidate_infra_round_retry_limit = (
        _workspace_coding_candidate_infra_round_retries()
    )
    provider_circuit_exhausted_slot_limit = (
        _workspace_coding_provider_circuit_exhausted_slots()
    )
    candidate_strategy_retry_limit = DEFAULT_WORKSPACE_CODING_CANDIDATE_STRATEGY_RETRIES
    candidate_infra_retry_delays = tuple(
        _candidate_infra_retry_sleep_seconds(attempt)
        for attempt in range(candidate_infra_retry_limit)
    )
    controller_config = {
        "verification_commands": verification_commands,
        "required_project_verification_commands": (
            required_project_verification_commands
        ),
        "max_iterations": max_iterations,
        "timeout_seconds": timeout_seconds,
        "max_selected_files": max_selected_files,
        "include_agent_verification_commands": include_agent_verification_commands,
        "exploration_iterations": exploration_iterations,
        "enforce_spec_maturity_gate": enforce_spec_maturity_gate,
        "max_active_themes_per_iteration": max_active_themes_per_iteration,
        "candidate_count": normalized_candidate_count,
        "candidate_infrastructure_retry_limit": candidate_infra_retry_limit,
        "candidate_infrastructure_round_retry_limit": (
            candidate_infra_round_retry_limit
        ),
        "provider_circuit_exhausted_slot_limit": (
            provider_circuit_exhausted_slot_limit
        ),
        "candidate_early_stop_policy_version": CANDIDATE_EARLY_STOP_POLICY_VERSION,
        "candidate_early_stop_mode": CANDIDATE_EARLY_STOP_MODE,
        "candidate_early_stop_prefix_count": CANDIDATE_EARLY_STOP_PREFIX_COUNT,
        "candidate_strategy_retry_limit": candidate_strategy_retry_limit,
        "candidate_infrastructure_retry_delays": (candidate_infra_retry_delays),
        "agent_runtime": workspace_coding_agent_runtime_identity(agent),
        "execution_policy": canonicalize(command_executor.policy),
        "available_verification_executables": available_verification_executables,
        "runtime_version": WORKSPACE_CONTROLLER_RUNTIME_VERSION,
    }
    controller_repo_digest = build_workspace_execution_profile(root).repo_hash
    controller_event_run, replayed_controller_result = _start_controller_event_run(
        event_store=event_store,
        source_report=source_report,
        repo_digest=controller_repo_digest,
        task_specs=task_specs_v2,
        agent=agent,
        config=controller_config,
        attempt=event_attempt,
        committed_receipt=committed_receipt,
    )
    if replayed_controller_result is not None:
        if committed_receipt is not None:
            acknowledge_committed_workspace_promotion(
                root,
                candidate_digest=committed_receipt.candidate_digest,
            )
        return replayed_controller_result
    _set_agent_run_context(
        agent,
        namespace=(
            controller_event_run.run_id
            if controller_event_run is not None
            else "controller_"
            + stable_hash(
                {
                    "report": source_report.get("report_id"),
                    "repo": controller_repo_digest,
                    "config": controller_config,
                    "attempt": event_attempt,
                }
            )[:24]
        ),
        attempt=event_attempt,
    )
    reproducer_plan = build_reproducer_plan(
        source_report=source_report,
        workspace_root=root,
        task_specs=task_specs_v2,
        repo_intelligence=repo_intelligence,
        confirm_base=True,
        timeout_seconds=timeout_seconds,
        executor=command_executor,
    )
    engineering_org_plan = build_engineering_organization_plan(
        source_report=source_report,
        task_specs=task_specs_v2,
        repo_intelligence=repo_intelligence,
        reproducer_plan=reproducer_plan,
    )
    code_max_runtime_summary = build_code_max_context_summary(
        source_report=source_report,
        workspace_root=root,
        task_specs=task_specs_v2,
        confirm_base=False,
        timeout_seconds=min(timeout_seconds, 20.0),
        executor=command_executor,
    )
    code_max_runtime_summary = _with_execution_capabilities(
        code_max_runtime_summary,
        available_verification_executables,
    )
    environment_summary = code_max_runtime_summary.get("environment", {})
    discovered_project_test_commands = _string_tuple(
        _mapping_get(environment_summary, "discovered_test_commands", ()),
        fallback=(),
        max_items=16,
    )
    project_lint_commands = _string_tuple(
        _mapping_get(environment_summary, "discovered_lint_commands", ()),
        fallback=(),
        max_items=16,
    )
    project_build_commands = _string_tuple(
        _mapping_get(environment_summary, "discovered_build_commands", ()),
        fallback=(),
        max_items=16,
    )
    required_project_commands = _dedupe(required_project_verification_commands)
    required_project_test_commands = tuple(
        command
        for command in required_project_commands
        if _is_project_test_suite_command(command)
    )
    required_public_probe_commands = tuple(
        command
        for command in required_project_commands
        if command not in required_project_test_commands
    )
    project_test_commands = _dedupe(
        (*discovered_project_test_commands, *required_project_test_commands)
    )
    discovered_project_verification_commands = _dedupe(
        (*project_test_commands, *project_lint_commands, *project_build_commands)
    )
    project_command_baseline = _project_command_baseline(
        workspace_root=root,
        commands=discovered_project_verification_commands,
        timeout_seconds=timeout_seconds,
        executor=command_executor,
    )
    project_verification_commands = _dedupe(
        (
            *required_project_commands,
            *tuple(project_command_baseline["eligible_commands"]),
        )
    )
    opportunistic_project_commands = tuple(
        item["command"]
        for item in project_command_baseline["results"]
        if item["classification"] == "baseline_failed"
        and item["command"] not in required_project_commands
    )
    code_max_runtime_summary = _with_project_command_baseline(
        code_max_runtime_summary,
        project_command_baseline,
    )
    covered_requested_themes: list[str] = []
    proposals: list[CodingAgentPatchProposal] = []
    all_patch_results: list[WorkspacePatchResult] = []
    claim_patch_results: list[WorkspacePatchResult] = []
    candidate_claim_patch_results: list[WorkspacePatchResult] = []
    all_verification_results: list[WorkspaceVerificationResult] = []
    verified_patch_results: list[WorkspacePatchResult] = []
    previous_verification: tuple[WorkspaceVerificationResult, ...] = (
        _reproducer_feedback(reproducer_plan)
    )
    selected_files: tuple[str, ...] = ()
    final_mode = "not_started"
    executed_generated_coverage_commands: list[str] = []
    exploration_attempts = 0
    risky_patch_bundles = 0
    promoted_candidates = 0
    promotion_attempt_count = 0
    controller_rounds = 0
    development_rounds = 0
    infrastructure_only_rounds = 0
    rejected_by_gate = 0
    repair_iterations = 0
    accepted_verification_commands: list[str] = []
    promotion_records: list[dict[str, Any]] = []
    candidate_runtime_records: list[dict[str, Any]] = []
    candidate_failure_feedback_by_proposal_id: dict[
        str,
        tuple[WorkspaceVerificationResult, ...],
    ] = {}
    candidate_slot_infra_retry_count = 0
    candidate_slot_exhaustion_count = 0
    unresolved_candidate_slot_exhaustion_count = 0
    consecutive_provider_slot_exhaustions = 0
    provider_circuit_open = False
    provider_circuit_reason_code: str | None = None
    candidate_strategy_retry_count = 0
    candidate_strategy_exhaustion_count = 0
    selected_candidate_proposal_ids: list[str] = []
    rejected_candidate_proposal_ids: list[str] = []
    candidate_bank: dict[str, CodingAgentPatchProposal] = {}
    repair_seed_proposal: CodingAgentPatchProposal | None = None
    repair_seed_record: dict[str, Any] | None = None
    targeted_repair_attempts = 0
    evidence_ledger = EvidenceLedger()
    candidate_evidence_ledger = evidence_ledger
    pending_generated_coverage_commands: list[str] = []
    baseline_status_by_command = {
        artifact.command: _reproducer_base_status(artifact.reproducibility_status)
        for artifact in reproducer_plan.artifacts
    }
    baseline_status_by_command.update(
        {
            item["command"]: item["status"]
            for item in project_command_baseline["results"]
        }
    )
    if enforce_spec_maturity_gate and spec_maturity_gate["ready_task_count"] == 0:
        final_mode = "blocked"
        gate_result = WorkspaceVerificationResult(
            command="spec_maturity_gate",
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            stdout_tail=json.dumps(
                canonicalize(spec_maturity_gate),
                sort_keys=True,
                separators=(",", ":"),
            )[:2000],
            stderr_tail="spec_maturity_below_testable",
            blocked_reason="spec_maturity_below_testable",
        )
        all_verification_results.append(gate_result)
        code_landing_metrics = _code_landing_metrics(
            source_report=source_report,
            requested_themes=themes,
            task_specs=task_specs_v2,
            spec_maturity_gate=spec_maturity_gate | {"patch_loop_entry": "blocked"},
            repo_intelligence=repo_intelligence,
            reproducer_plan=reproducer_plan,
            engineering_org_plan=engineering_org_plan,
            code_max_runtime_summary=code_max_runtime_summary,
            patch_results=(),
            verification_results=tuple(all_verification_results),
            final_mode=final_mode,
        )
        result_payload = {
            "source_report_id": _optional_string(source_report.get("report_id")),
            "package": package,
            "final_mode": final_mode,
            "agent_source": agent.source,
            "agent_model": agent.model,
            "code_landing_metrics": code_landing_metrics,
            "controller_event_run_id": (
                controller_event_run.run_id if controller_event_run else None
            ),
        }
        provenance_hash = stable_hash(_stable_controller_event_payload(result_payload))
        result = CodingAgentLoopResult(
            run_id=f"workspace_agent_loop_{provenance_hash[:24]}",
            developer_mode="guarded_promotion_agent",
            source_report_id=_optional_string(source_report.get("report_id")),
            package=package,
            selected_files=(),
            proposals=(),
            patch_results=(),
            verification_results=tuple(all_verification_results),
            iterations=0,
            final_mode=final_mode,
            failed_commands=("spec_maturity_gate",),
            provenance_hash=provenance_hash,
            rejected_by_gate=1,
            reasoning_effort=getattr(agent, "reasoning_effort", None),
            requested_reasoning_effort=getattr(
                agent, "requested_reasoning_effort", None
            ),
            run_config={
                "max_iterations": max_iterations,
                "timeout_seconds": timeout_seconds,
                "max_selected_files": max_selected_files,
                "include_agent_verification_commands": include_agent_verification_commands,
                "exploration_iterations": exploration_iterations,
                "max_active_themes_per_iteration": max_active_themes_per_iteration,
                "candidate_count": normalized_candidate_count,
                "candidate_infrastructure_retry_limit": candidate_infra_retry_limit,
                "candidate_infrastructure_round_retry_limit": (
                    candidate_infra_round_retry_limit
                ),
                "candidate_strategy_retry_limit": candidate_strategy_retry_limit,
                "candidate_proposal_count": 0,
                "promotion_attempt_count": 0,
                "canonical_promotion_count": 0,
                "controller_round_count": 0,
                "development_round_count": 0,
                "infrastructure_only_round_count": 0,
                "controller_event_run_id": (
                    controller_event_run.run_id if controller_event_run else None
                ),
                "evaluator_verification_commands": verification_commands,
                "required_project_verification_commands": (required_project_commands),
                "generated_coverage_commands": (),
                "covered_requested_themes": (),
                "remaining_requested_themes": themes,
                "enforce_spec_maturity_gate": enforce_spec_maturity_gate,
                "engineering_organization_plan": _engineering_org_plan_agent_summary(
                    engineering_org_plan
                ),
                "project_command_baseline": project_command_baseline,
                "code_max_runtime": code_max_runtime_summary,
            },
            patch_coverage=_patch_coverage_summary(
                patch_results=(),
                verification_results=tuple(all_verification_results),
                generated_coverage_commands=(),
            ),
            code_landing_metrics=code_landing_metrics,
        )
        _complete_controller_event_run(
            event_store=event_store,
            run=controller_event_run,
            result=result,
            workspace_root=root,
        )
        return result

    if verification_commands:
        with CandidateWorkspace.create(root) as baseline_workspace:
            baseline_results = baseline_workspace.verify(
                verification_commands,
                timeout_seconds=timeout_seconds,
                executor=command_executor,
            )
        baseline_status_by_command.update(
            {result.command: result.status for result in baseline_results}
        )
        previous_verification = (
            *previous_verification,
            *(result for result in baseline_results if result.status != "passed"),
        )

    candidate_workspace = CandidateWorkspace.create(root)
    working_root = candidate_workspace.root

    for _ in range(max(0, exploration_iterations)):
        _set_agent_stage(agent, "explore")
        file_contexts = select_agent_context_files(
            working_root,
            themes,
            repo_intelligence=repo_intelligence,
            task_specs=task_specs_v2,
            engineering_org_plan=engineering_org_plan,
            max_files=max_selected_files,
        )
        selected_files = tuple(context.path for context in file_contexts)
        proposal, preflight_reason = _propose_with_execution_preflight(
            agent=agent,
            executor=command_executor,
            source_report=source_report,
            workspace_root=working_root,
            package=package,
            file_contexts=file_contexts,
            previous_verification=previous_verification,
            iteration=len(proposals),
            timeout_seconds=timeout_seconds,
        )
        proposal = replace(proposal, stage="explore")
        if preflight_reason is None:
            _seal_agent_proposal_run(agent, proposal)
        trace_record = (
            _execution_preflight_runtime_record(
                agent=agent,
                proposal=proposal,
                reason=preflight_reason,
                iteration=_,
                candidate_index=0,
                strategy="exploration",
                stage="explore",
            )
            if preflight_reason is not None
            else _agent_proposal_runtime_record(
                agent=agent,
                proposal=proposal,
                iteration=_,
                candidate_index=0,
                strategy="exploration",
                stage="explore",
            )
        )
        if trace_record is not None:
            candidate_runtime_records.append(trace_record)
        proposals.append(proposal)
        exploration_attempts += 1
        if _is_risky_patch_bundle(proposal):
            risky_patch_bundles += 1
        if proposal.blocked_reason:
            previous_verification = _proposal_block_feedback(proposal)
            all_verification_results.extend(previous_verification)
            final_mode = "explored"
            continue
        patch_results = candidate_workspace.apply(
            proposal.patches,
            gate_policy="explore",
        )
        all_patch_results.extend(patch_results)
        if any(result.status == "blocked" for result in patch_results):
            previous_verification = _patch_block_feedback(patch_results)
        else:
            previous_verification = ()
        final_mode = "explored"

    if exploration_attempts:
        candidate_workspace.close()
        candidate_workspace = CandidateWorkspace.create(root)
        working_root = candidate_workspace.root
        previous_verification = ()

    max_development_rounds = max(0, max_iterations)
    max_controller_rounds = max_development_rounds + candidate_infra_round_retry_limit
    for _ in range(max_controller_rounds):
        if development_rounds >= max_development_rounds:
            break
        controller_rounds += 1
        _set_agent_stage(agent, "promote")
        remaining_themes = _remaining_requested_themes(
            requested_themes=themes,
            covered_themes=tuple(covered_requested_themes),
        )
        active_themes = _active_theme_window(
            remaining_themes or themes,
            max_active_themes_per_iteration=max_active_themes_per_iteration,
        )
        active_task_specs_v2 = _task_specs_for_themes(task_specs_v2, active_themes)
        active_spec_maturity_gate = build_spec_maturity_gate(active_task_specs_v2)
        active_reproducer_plan = _reproducer_plan_for_task_specs(
            reproducer_plan,
            active_task_specs_v2,
        )
        active_engineering_org_plan = build_engineering_organization_plan(
            source_report=source_report,
            task_specs=active_task_specs_v2,
            repo_intelligence=repo_intelligence,
            reproducer_plan=active_reproducer_plan,
        )
        active_code_max_runtime_summary = build_code_max_context_summary(
            source_report=source_report,
            workspace_root=working_root,
            task_specs=active_task_specs_v2,
            confirm_base=False,
            timeout_seconds=min(timeout_seconds, 20.0),
            executor=command_executor,
        )
        active_code_max_runtime_summary = _with_execution_capabilities(
            active_code_max_runtime_summary,
            available_verification_executables,
        )
        active_code_max_runtime_summary = _with_project_command_baseline(
            active_code_max_runtime_summary,
            project_command_baseline,
        )
        file_contexts = select_agent_context_files(
            working_root,
            active_themes,
            repo_intelligence=repo_intelligence,
            task_specs=active_task_specs_v2,
            engineering_org_plan=active_engineering_org_plan,
            max_files=max_selected_files,
        )
        file_contexts = _file_contexts_with_retry_failure_priority(
            working_root,
            file_contexts,
            previous_verification,
            themes=active_themes,
            task_specs=active_task_specs_v2,
        )
        selected_files = tuple(context.path for context in file_contexts)
        agent_source_report = _source_report_with_workspace_progress(
            source_report=source_report,
            requested_themes=themes,
            covered_themes=tuple(covered_requested_themes),
            active_themes=active_themes,
            changed_paths=_changed_patch_paths(
                (*claim_patch_results, *candidate_claim_patch_results)
            ),
        )
        agent_source_report = _source_report_with_code_landing_context(
            source_report=agent_source_report,
            task_specs=active_task_specs_v2,
            spec_maturity_gate=active_spec_maturity_gate,
            repo_intelligence=repo_intelligence,
            reproducer_plan=active_reproducer_plan,
        )
        agent_source_report = _source_report_with_engineering_organization(
            source_report=agent_source_report,
            engineering_org_plan=active_engineering_org_plan,
        )
        agent_source_report = _source_report_with_code_max_runtime(
            source_report=agent_source_report,
            code_max_runtime_summary=active_code_max_runtime_summary,
        )
        agent_source_report = {
            **agent_source_report,
            "required_project_verification_commands": required_project_commands,
        }
        evaluator_commands = _dedupe(
            (*verification_commands, *active_reproducer_plan.executable_commands)
        )
        repair_fast_path = (
            repair_seed_proposal is not None
            and repair_seed_record is not None
            and targeted_repair_attempts < 2
        )
        declared_component_paths = _declared_component_paths(
            active_task_specs_v2,
            workspace_root=working_root,
        )
        round_candidate_count = 1 if repair_fast_path else normalized_candidate_count
        search_strategies = (
            (
                "targeted_parent_repair: preserve the validated production behavior "
                "of the parent candidate and repair only the exact structured "
                "review or validation findings",
            )
            if repair_fast_path
            else _candidate_search_strategies(
                active_code_max_runtime_summary,
                count=round_candidate_count,
                declared_component_paths=declared_component_paths,
            )
        )
        if repair_fast_path:
            targeted_repair_attempts += 1
        round_proposals: list[CodingAgentPatchProposal] = []
        minimum_candidate_quorum = min(2, round_candidate_count)
        for candidate_index in range(round_candidate_count):
            max_slot_attempts = (
                1 + candidate_infra_retry_limit + candidate_strategy_retry_limit
            )
            candidate_proposal: CodingAgentPatchProposal | None = None
            strategy_feedback: tuple[str, ...] = ()
            slot_infra_retries = 0
            slot_strategy_retries = 0
            for slot_attempt in range(max_slot_attempts):
                candidate_source_report = dict(agent_source_report)
                candidate_source_report["candidate_search"] = _candidate_search_payload(
                    candidate_index=candidate_index,
                    candidate_count=round_candidate_count,
                    strategy=search_strategies[candidate_index],
                    prior_proposals=(
                        *tuple(candidate_bank.values()),
                        *tuple(round_proposals),
                    ),
                    workspace_root=working_root,
                    slot_attempt=slot_strategy_retries + 1,
                    max_slot_attempts=1 + candidate_strategy_retry_limit,
                    repair_seed=(repair_seed_proposal if repair_fast_path else None),
                    repair_record=(repair_seed_record if repair_fast_path else None),
                    declared_component_paths=declared_component_paths,
                    strategy_feedback=strategy_feedback,
                )
                candidate_proposal, preflight_reason = (
                    _propose_with_execution_preflight(
                        agent=agent,
                        executor=command_executor,
                        source_report=candidate_source_report,
                        workspace_root=working_root,
                        package=package,
                        file_contexts=file_contexts,
                        previous_verification=previous_verification,
                        iteration=len(proposals),
                        timeout_seconds=timeout_seconds,
                    )
                )
                candidate_proposal = replace(
                    candidate_proposal,
                    stage="promote",
                )
                if repair_fast_path:
                    assert repair_seed_proposal is not None
                    candidate_proposal = _compose_targeted_repair_proposal(
                        parent=repair_seed_proposal,
                        child=candidate_proposal,
                        workspace_root=working_root,
                    )
                strategy_issues = (
                    ()
                    if candidate_proposal.blocked_reason
                    else _candidate_strategy_compliance_issues(
                        candidate_proposal,
                        strategy=search_strategies[candidate_index],
                        declared_component_paths=declared_component_paths,
                    )
                )
                if strategy_issues:
                    candidate_proposal = replace(
                        candidate_proposal,
                        blocked_reason=(
                            "candidate_strategy_constraint_failed:"
                            + ",".join(strategy_issues)
                        )[:300],
                    )
                candidate_failure_feedback_by_proposal_id[
                    candidate_proposal.proposal_id
                ] = (
                    ()
                    if preflight_reason is not None
                    else _agent_tool_trace_failure_feedback(
                        agent,
                        proposal=candidate_proposal,
                    )
                )
                trace_record = (
                    _execution_preflight_runtime_record(
                        agent=agent,
                        proposal=candidate_proposal,
                        reason=preflight_reason,
                        iteration=_,
                        candidate_index=candidate_index,
                        strategy=search_strategies[candidate_index],
                        stage="promote",
                    )
                    if preflight_reason is not None
                    else _agent_proposal_runtime_record(
                        agent=agent,
                        proposal=candidate_proposal,
                        iteration=_,
                        candidate_index=candidate_index,
                        strategy=search_strategies[candidate_index],
                        stage="promote",
                    )
                )
                if trace_record is not None:
                    trace_record["candidate_slot_attempt"] = slot_attempt + 1
                    trace_record["candidate_slot_max_attempts"] = max_slot_attempts
                    trace_record["strategy_compliance_issues"] = strategy_issues
                    semantic_fingerprint = _candidate_semantic_fingerprint(
                        candidate_proposal
                    )
                    trace_record["semantic_fingerprint"] = semantic_fingerprint
                    prior_candidate = candidate_bank.get(semantic_fingerprint)
                    if prior_candidate is not None:
                        trace_record["semantic_duplicate_of"] = (
                            prior_candidate.proposal_id
                        )
                    candidate_runtime_records.append(trace_record)
                infrastructure_block = _is_candidate_infrastructure_block(
                    candidate_proposal.blocked_reason
                )
                if infrastructure_block:
                    if slot_infra_retries < candidate_infra_retry_limit:
                        candidate_slot_infra_retry_count += 1
                        delay = candidate_infra_retry_delays[slot_infra_retries]
                        slot_infra_retries += 1
                        _record_controller_event(
                            event_store=event_store,
                            run=controller_event_run,
                            event_type="candidate_slot_retry_scheduled",
                            payload={
                                "iteration": _,
                                "candidate_index": candidate_index,
                                "failed_slot_attempt": slot_infra_retries,
                                "next_slot_attempt": slot_infra_retries + 1,
                                "delay_seconds": delay,
                                "blocked_reason": candidate_proposal.blocked_reason,
                                "resume_policy": (
                                    "same_candidate_identity_from_latest_checkpoint"
                                ),
                            },
                            idempotency_key=(
                                f"candidate_slot_retry:{_}:{candidate_index}:"
                                f"{slot_infra_retries}"
                            ),
                        )
                        sleep(delay)
                        continue
                    if preflight_reason is None:
                        _seal_agent_proposal_run(agent, candidate_proposal)
                    proposals.append(candidate_proposal)
                    break
                consecutive_provider_slot_exhaustions = 0
                if preflight_reason is None:
                    _seal_agent_proposal_run(agent, candidate_proposal)
                proposals.append(candidate_proposal)
                if strategy_issues:
                    if slot_strategy_retries >= candidate_strategy_retry_limit:
                        candidate_strategy_exhaustion_count += 1
                        break
                    slot_strategy_retries += 1
                    candidate_strategy_retry_count += 1
                    strategy_feedback = strategy_issues
                    _record_controller_event(
                        event_store=event_store,
                        run=controller_event_run,
                        event_type="candidate_strategy_retry_scheduled",
                        payload={
                            "iteration": _,
                            "candidate_index": candidate_index,
                            "failed_slot_attempt": slot_attempt + 1,
                            "next_slot_attempt": slot_attempt + 2,
                            "strategy": search_strategies[candidate_index],
                            "strategy_issues": strategy_issues,
                            "declared_component_paths": declared_component_paths,
                        },
                        idempotency_key=(
                            "candidate_strategy_retry:"
                            f"{_}:{candidate_index}:{slot_attempt + 1}"
                        ),
                    )
                    continue
                semantic_fingerprint = _candidate_semantic_fingerprint(
                    candidate_proposal
                )
                if (
                    not candidate_proposal.blocked_reason
                    and semantic_fingerprint not in candidate_bank
                ):
                    candidate_bank[semantic_fingerprint] = candidate_proposal
                break
            assert candidate_proposal is not None
            round_proposals.append(candidate_proposal)
            if _is_provider_infrastructure_block(candidate_proposal.blocked_reason):
                consecutive_provider_slot_exhaustions += 1
            elif not _is_candidate_infrastructure_block(
                candidate_proposal.blocked_reason
            ):
                consecutive_provider_slot_exhaustions = 0
            viable_candidate_count = sum(
                1 for item in round_proposals if not item.blocked_reason
            )
            if (
                round_candidate_count >= 2
                and viable_candidate_count < minimum_candidate_quorum
                and consecutive_provider_slot_exhaustions
                >= provider_circuit_exhausted_slot_limit
            ):
                provider_circuit_open = True
                provider_circuit_reason_code = _provider_infrastructure_reason_code(
                    candidate_proposal.blocked_reason
                )
                _record_controller_event(
                    event_store=event_store,
                    run=controller_event_run,
                    event_type="provider_circuit_opened",
                    payload={
                        "iteration": _,
                        "candidate_index": candidate_index,
                        "consecutive_exhausted_provider_slots": (
                            consecutive_provider_slot_exhaustions
                        ),
                        "exhausted_slot_limit": (provider_circuit_exhausted_slot_limit),
                        "viable_candidate_count": viable_candidate_count,
                        "minimum_candidate_quorum": minimum_candidate_quorum,
                        "reason_code": provider_circuit_reason_code,
                        "policy": "pause_without_product_failure",
                    },
                    idempotency_key=f"provider_circuit_opened:{_}",
                )
                break
        exhausted_slots = tuple(
            index
            for index, proposal_item in enumerate(round_proposals)
            if _is_candidate_infrastructure_block(proposal_item.blocked_reason)
        )
        if exhausted_slots:
            candidate_slot_exhaustion_count += len(exhausted_slots)
            unresolved_candidate_slot_exhaustion_count = len(exhausted_slots)
            viable_candidate_count = sum(
                1 for item in round_proposals if not item.blocked_reason
            )
            if viable_candidate_count < minimum_candidate_quorum:
                infrastructure_feedback = tuple(
                    result
                    for index in exhausted_slots
                    for result in _proposal_block_feedback(round_proposals[index])
                )
                final_mode = "blocked"
                _record_controller_event(
                    event_store=event_store,
                    run=controller_event_run,
                    event_type="candidate_round_incomplete",
                    payload={
                        "iteration": _,
                        "required_candidate_count": round_candidate_count,
                        "minimum_candidate_quorum": minimum_candidate_quorum,
                        "viable_candidate_count": viable_candidate_count,
                        "exhausted_candidate_slots": exhausted_slots,
                        "policy": "minimum_two_viable_candidates_before_selection",
                    },
                    idempotency_key=f"candidate_round_incomplete:{_}",
                )
                if provider_circuit_open:
                    previous_verification = infrastructure_feedback
                    all_verification_results.extend(previous_verification)
                    break
                if infrastructure_only_rounds < candidate_infra_round_retry_limit:
                    delay = _candidate_infra_retry_sleep_seconds(
                        candidate_infra_retry_limit + infrastructure_only_rounds
                    )
                    infrastructure_only_rounds += 1
                    _record_controller_event(
                        event_store=event_store,
                        run=controller_event_run,
                        event_type="candidate_round_retry_scheduled",
                        payload={
                            "iteration": _,
                            "next_iteration": _ + 1,
                            "delay_seconds": delay,
                            "exhausted_candidate_slots": exhausted_slots,
                            "reason": "candidate_infrastructure_quorum_unavailable",
                            "infrastructure_retry": infrastructure_only_rounds,
                            "infrastructure_retry_limit": (
                                candidate_infra_round_retry_limit
                            ),
                        },
                        idempotency_key=f"candidate_round_retry:{_}",
                    )
                    previous_verification = ()
                    sleep(delay)
                    continue
                previous_verification = infrastructure_feedback
                all_verification_results.extend(previous_verification)
                break
            _record_controller_event(
                event_store=event_store,
                run=controller_event_run,
                event_type="candidate_round_quorum_reached",
                payload={
                    "iteration": _,
                    "required_candidate_count": round_candidate_count,
                    "minimum_candidate_quorum": minimum_candidate_quorum,
                    "viable_candidate_count": viable_candidate_count,
                    "exhausted_candidate_slots": exhausted_slots,
                    "policy": "strict_validation_over_viable_quorum",
                },
                idempotency_key=f"candidate_round_quorum_reached:{_}",
            )
        else:
            unresolved_candidate_slot_exhaustion_count = 0
        development_rounds += 1
        component_strategy_indexes = tuple(
            index
            for index, strategy in enumerate(search_strategies)
            if "declared_component_ownership" in strategy
        )
        component_outcome_ready = any(
            _proposal_changes_declared_component(
                proposal_item,
                declared_component_paths=declared_component_paths,
            )
            for proposal_item in round_proposals
        )
        if (
            declared_component_paths
            and component_strategy_indexes
            and not component_outcome_ready
        ):
            component_feedback = (
                WorkspaceVerificationResult(
                    command="candidate_strategy_diversity",
                    status="failed",
                    exit_code=None,
                    elapsed_sec=0.0,
                    stdout_tail=json.dumps(
                        {
                            "declared_component_paths": declared_component_paths,
                            "required_outcome": "declared_component_changed",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    stderr_tail="declared_component_change_required",
                    blocked_reason="declared_component_change_required",
                ),
            )
            blocked_round_feedback = _merge_verification_feedback(
                *(
                    (
                        *candidate_failure_feedback_by_proposal_id.get(
                            proposal_item.proposal_id,
                            (),
                        ),
                        *_proposal_block_feedback(proposal_item),
                    )
                    for proposal_item in round_proposals
                    if proposal_item.blocked_reason
                    and not _is_candidate_infrastructure_block(
                        proposal_item.blocked_reason
                    )
                ),
            )
            previous_verification = _merge_verification_feedback(
                blocked_round_feedback,
                component_feedback,
            )
            all_verification_results.extend(previous_verification)
            final_mode = "blocked"
            _record_controller_event(
                event_store=event_store,
                run=controller_event_run,
                event_type="candidate_round_strategy_incomplete",
                payload={
                    "iteration": _,
                    "declared_component_paths": declared_component_paths,
                    "required_outcome": "declared_component_changed",
                    "proposal_ids": tuple(item.proposal_id for item in round_proposals),
                },
                idempotency_key=f"candidate_round_strategy_incomplete:{_}",
            )
            if development_rounds < max_development_rounds:
                continue
            break
        composite_proposals = (
            ()
            if repair_fast_path
            else _compose_disjoint_candidate_proposals(
                tuple(round_proposals),
                start_iteration=len(proposals),
            )
        )
        if composite_proposals:
            proposals.extend(composite_proposals)
            round_proposals.extend(composite_proposals)
            candidate_runtime_records.append(
                {
                    "record_type": "candidate_composition",
                    "iteration": _,
                    "status": "created",
                    "proposal_ids": tuple(
                        proposal.proposal_id for proposal in composite_proposals
                    ),
                    "parent_candidate_count": round_candidate_count,
                    "policy": "disjoint_public_contract_union_requires_full_validation",
                }
            )
        _record_controller_event(
            event_store=event_store,
            run=controller_event_run,
            event_type="candidate_round_completed",
            payload={
                "iteration": _,
                "active_themes": active_themes,
                "proposal_ids": tuple(item.proposal_id for item in round_proposals),
                "blocked_reasons": tuple(
                    item.blocked_reason for item in round_proposals
                ),
            },
            idempotency_key=f"candidate_round:{_}",
        )
        proposal = round_proposals[0]
        selected_runtime_record: dict[str, Any] | None = None
        requires_public_contract_validation = any(
            spec.contract_dimensions
            and any(
                oracle.required
                and oracle.dimension_ids
                and oracle.source == "contract_witness"
                for oracle in spec.acceptance_oracles
            )
            for spec in active_task_specs_v2
        )
        selection_executed = (
            normalized_candidate_count > 1
            or repair_fast_path
            or requires_public_contract_validation
        )
        if selection_executed:
            proposal, runtime_record, rejected_ids = _select_multi_candidate_proposal(
                proposals=tuple(round_proposals),
                source_report=agent_source_report,
                workspace_root=working_root,
                task_specs=active_task_specs_v2,
                evaluator_commands=evaluator_commands,
                project_commands=project_verification_commands,
                strategies=search_strategies,
                executor=command_executor,
                timeout_seconds=timeout_seconds,
            )
            candidate_runtime_records.append(runtime_record)
            selected_runtime_record = runtime_record
            rejected_candidate_proposal_ids.extend(rejected_ids)
            _record_controller_event(
                event_store=event_store,
                run=controller_event_run,
                event_type="candidate_evaluated",
                payload={
                    "iteration": _,
                    "selected_proposal_id": proposal.proposal_id,
                    "selected_semantic_fingerprint": (
                        _candidate_semantic_fingerprint(proposal)
                    ),
                    "record_hash": stable_hash(
                        _stable_controller_event_payload(runtime_record)
                    ),
                    "status": runtime_record.get("status"),
                    "reason": runtime_record.get("reason"),
                    "review_codes": tuple(
                        item.get("code")
                        for item in tuple(
                            runtime_record.get("review_findings", ()) or ()
                        )
                        if isinstance(item, Mapping)
                    ),
                    "validation_failure_codes": tuple(
                        runtime_record.get("validation_failure_codes", ()) or ()
                    ),
                },
                idempotency_key=f"candidate_evaluated:{_}",
            )
            if (
                not repair_fast_path
                and round_candidate_count > CANDIDATE_EARLY_STOP_PREFIX_COUNT
            ):
                shadow_record = _candidate_early_stop_shadow_record(
                    proposals=tuple(
                        round_proposals[:CANDIDATE_EARLY_STOP_PREFIX_COUNT]
                    ),
                    observed_candidate_count=round_candidate_count,
                    strategies=search_strategies[:CANDIDATE_EARLY_STOP_PREFIX_COUNT],
                    declared_component_paths=declared_component_paths,
                    selection_record=runtime_record,
                )
                shadow_payload = {
                    key: value
                    for key, value in shadow_record.items()
                    if key != "evidence_hash"
                } | {"iteration": _}
                shadow_record = {
                    **shadow_payload,
                    "evidence_hash": stable_hash(shadow_payload),
                }
                candidate_runtime_records.append(shadow_record)
                _record_controller_event(
                    event_store=event_store,
                    run=controller_event_run,
                    event_type="candidate_early_stop_shadow_evaluated",
                    payload=shadow_record,
                    idempotency_key=f"candidate_early_stop_shadow:{_}",
                )
        selected_candidate_proposal_ids.append(proposal.proposal_id)
        _record_controller_event(
            event_store=event_store,
            run=controller_event_run,
            event_type="candidate_selected",
            payload={
                "iteration": _,
                "selected_proposal_id": proposal.proposal_id,
                "rejected_proposal_ids": rejected_ids if selection_executed else (),
            },
            idempotency_key=f"candidate_selected:{_}",
        )
        if previous_verification:
            repair_iterations += 1
        if proposal.blocked_reason:
            selection_allows_code_repair = bool(
                selected_runtime_record is not None
                and selected_runtime_record.get("status") == "needs_revision"
                and selected_runtime_record.get("repair_seed_eligible", True)
            )
            preserve_existing_repair_seed = _should_preserve_repair_seed(
                repair_fast_path=repair_fast_path,
                repair_seed_proposal=repair_seed_proposal,
                repair_seed_record=repair_seed_record,
                blocked_reason=proposal.blocked_reason,
                selected_runtime_record=selected_runtime_record,
            )
            if preserve_existing_repair_seed:
                pass
            elif selection_allows_code_repair:
                if not repair_fast_path:
                    targeted_repair_attempts = 0
                repair_seed_proposal = proposal
                repair_seed_record = selected_runtime_record
            else:
                repair_seed_proposal = None
                repair_seed_record = None
                targeted_repair_attempts = 0
            selection_feedback = (
                _candidate_selection_feedback(proposal, selected_runtime_record)
                if selected_runtime_record is not None
                else _proposal_block_feedback(proposal)
            )
            previous_verification = _merge_verification_feedback(
                candidate_failure_feedback_by_proposal_id.get(
                    proposal.proposal_id,
                    (),
                ),
                selection_feedback,
            )
            all_verification_results.extend(previous_verification)
            final_mode = "blocked"
            if development_rounds < max_development_rounds and (
                _is_retryable_agent_block(proposal.blocked_reason)
                or preserve_existing_repair_seed
            ):
                continue
            break

        patch_results = candidate_workspace.apply(proposal.patches)
        repair_seed_proposal = None
        repair_seed_record = None
        targeted_repair_attempts = 0
        all_patch_results.extend(patch_results)
        candidate_claim_patch_results.extend(patch_results)
        _record_controller_event(
            event_store=event_store,
            run=controller_event_run,
            event_type="patch_application_completed",
            payload={
                "iteration": _,
                "proposal_id": proposal.proposal_id,
                "results": tuple(canonicalize(item) for item in patch_results),
            },
            idempotency_key=f"patch_application:{_}",
        )
        if any(result.status == "blocked" for result in patch_results):
            previous_verification = _patch_block_feedback(patch_results)
            final_mode = "blocked"
            rejected_by_gate += sum(
                1 for result in patch_results if result.status == "blocked"
            )
            if development_rounds < max_development_rounds:
                continue
            break
        if not _has_effective_patch(patch_results):
            previous_verification = _patch_no_effect_feedback()
            final_mode = "no_effect"
            rejected_by_gate += len(patch_results)
            if development_rounds < max_development_rounds:
                continue
            break
        agent_commands = (
            proposal.verification_commands
            if include_agent_verification_commands
            else ()
        )
        coverage_commands = _dedupe(
            (
                *pending_generated_coverage_commands,
                *_patch_coverage_commands(
                    patch_results,
                    executor=command_executor,
                    available_executables=available_verification_executables,
                ),
            )
        )
        executed_generated_coverage_commands.extend(coverage_commands)
        commands = _dedupe(
            (
                *evaluator_commands,
                *project_verification_commands,
                *agent_commands,
                *coverage_commands,
            )
        )
        if not commands:
            final_mode = workspace_update_mode(patch_results, ())
            break

        verification_results = candidate_workspace.verify(
            commands,
            timeout_seconds=timeout_seconds,
            executor=command_executor,
        )
        if opportunistic_project_commands and all(
            result.status == "passed" for result in verification_results
        ):
            opportunistic_results = candidate_workspace.verify(
                opportunistic_project_commands,
                timeout_seconds=timeout_seconds,
                executor=command_executor,
            )
            candidate_runtime_records.append(
                {
                    "record_type": "opportunistic_project_verification",
                    "iteration": _,
                    "proposal_id": proposal.proposal_id,
                    "results": tuple(
                        canonicalize(item) for item in opportunistic_results
                    ),
                }
            )
            passed_opportunistic_results = tuple(
                result for result in opportunistic_results if result.status == "passed"
            )
            verification_results = (
                *verification_results,
                *passed_opportunistic_results,
            )
        all_verification_results.extend(verification_results)
        verification_status_by_command = {
            result.command: result.status for result in verification_results
        }
        pending_generated_coverage_commands = [
            command
            for command in coverage_commands
            if verification_status_by_command.get(command) != "passed"
        ]
        _record_controller_event(
            event_store=event_store,
            run=controller_event_run,
            event_type="verification_completed",
            payload={
                "iteration": _,
                "proposal_id": proposal.proposal_id,
                "results": tuple(canonicalize(item) for item in verification_results),
            },
            idempotency_key=f"verification:{_}",
        )
        candidate_evidence_ledger = candidate_evidence_ledger.extend(
            _verification_evidence_records(
                task_id=(
                    active_task_specs_v2[0].task_id
                    if active_task_specs_v2
                    else f"workspace_task_{len(proposals)}"
                ),
                results=verification_results,
                evaluator_commands=_dedupe(
                    (*evaluator_commands, *required_public_probe_commands)
                ),
                project_test_commands=project_test_commands,
                project_lint_commands=project_lint_commands,
                project_build_commands=project_build_commands,
                generated_commands=coverage_commands,
                baseline_status_by_command=baseline_status_by_command,
                baseline_repo_digest=candidate_workspace.baseline_digest,
                candidate_repo_digest=candidate_workspace.candidate_digest,
                execution_policy_hash=execution_policy_hash,
            )
        )
        final_mode = workspace_update_mode(patch_results, verification_results)
        coverage_gate = _patch_coverage_gate(
            patch_results=patch_results,
            verification_results=verification_results,
            generated_coverage_commands=coverage_commands,
            proposal=proposal,
        )
        if coverage_gate is not None:
            all_verification_results.append(coverage_gate)
            verification_results = (*verification_results, coverage_gate)
            final_mode = "verification_failed"
        maintainer_gate = _maintainer_landing_gate(
            source_report=agent_source_report,
            proposal=proposal,
            patch_results=patch_results,
            verification_results=verification_results,
        )
        if maintainer_gate is not None:
            all_verification_results.append(maintainer_gate)
            verification_results = (*verification_results, maintainer_gate)
            final_mode = "verification_failed"
        if final_mode == "verified":
            accepted_verification_commands.extend(
                result.command
                for result in verification_results
                if result.status == "passed"
                and result.command
                not in {"patch_coverage_gate", "maintainer_landing_gate"}
            )
            integration_commands = _compact_promotion_verification_commands(
                (*verification_commands, *accepted_verification_commands)
            )
            promotion_attempt_count += 1
            prospective_promotion = WorkspacePromotionResult(
                status="release_candidate",
                changed_paths=candidate_workspace.changed_paths(),
                verification_results=verification_results,
                baseline_digest=candidate_workspace.baseline_digest,
                candidate_digest=candidate_workspace.candidate_digest,
                promoted_digest=candidate_workspace.candidate_digest,
                rollback_performed=False,
            )
            prospective_promotion_records = (
                *promotion_records,
                canonicalize(prospective_promotion),
            )
            prospective_claim_patch_results = (
                *claim_patch_results,
                *candidate_claim_patch_results,
            )
            prospective_verified_patch_results = (
                *verified_patch_results,
                *candidate_claim_patch_results,
            )
            prospective_covered_themes = list(covered_requested_themes)
            for theme in _proposal_requested_theme_coverage(
                proposal=proposal,
                requested_themes=themes,
                task_specs=task_specs_v2,
                patch_results=patch_results,
            ):
                if theme not in prospective_covered_themes:
                    prospective_covered_themes.append(theme)
            prospective_remaining_themes = _remaining_requested_themes(
                requested_themes=themes,
                covered_themes=tuple(prospective_covered_themes),
            )
            prospective_metrics = _code_landing_metrics(
                source_report=source_report,
                requested_themes=themes,
                task_specs=task_specs_v2,
                spec_maturity_gate=spec_maturity_gate,
                repo_intelligence=repo_intelligence,
                reproducer_plan=reproducer_plan,
                engineering_org_plan=engineering_org_plan,
                code_max_runtime_summary=code_max_runtime_summary,
                patch_results=prospective_claim_patch_results,
                verification_results=tuple(all_verification_results),
                final_mode="verified",
            )
            prospective_verified_count = _verified_effective_patch_count(
                tuple(prospective_verified_patch_results),
                verification_results=tuple(all_verification_results),
                final_mode="verified",
            )
            prospective_assessment = assess_landing_evidence(
                ledger=candidate_evidence_ledger,
                effective_patch=bool(prospective_verified_count),
                integration_replayed=True,
                promotion_status="release_candidate",
                promotion_records=tuple(prospective_promotion_records),
                event_chain_head=_controller_event_chain_head(
                    event_store=event_store,
                    run=controller_event_run,
                ),
            )
            recovery_result_on_commit: CodingAgentLoopResult | None = None
            if event_store is not None and controller_event_run is not None:
                recovery_result_on_commit = _build_promoted_recovery_result(
                    canonical_workspace_root=root,
                    report_id=_optional_string(source_report.get("report_id")),
                    package=package,
                    selected_files=selected_files,
                    proposals=tuple(proposals),
                    patch_results=prospective_claim_patch_results,
                    verification_results=tuple(all_verification_results),
                    controller_rounds=controller_rounds,
                    development_rounds=development_rounds,
                    infrastructure_only_rounds=infrastructure_only_rounds,
                    candidate_infra_retry_limit=candidate_infra_retry_limit,
                    candidate_infra_round_retry_limit=(
                        candidate_infra_round_retry_limit
                    ),
                    candidate_strategy_retry_limit=candidate_strategy_retry_limit,
                    candidate_slot_infra_retry_count=(candidate_slot_infra_retry_count),
                    candidate_slot_exhaustion_count=(candidate_slot_exhaustion_count),
                    unresolved_candidate_slot_exhaustion_count=(
                        unresolved_candidate_slot_exhaustion_count
                    ),
                    candidate_strategy_retry_count=candidate_strategy_retry_count,
                    candidate_strategy_exhaustion_count=(
                        candidate_strategy_exhaustion_count
                    ),
                    exploration_attempts=exploration_attempts,
                    risky_patch_bundles=risky_patch_bundles,
                    promoted_candidates=promoted_candidates + 1,
                    rejected_by_gate=rejected_by_gate,
                    repair_iterations=repair_iterations,
                    agent=agent,
                    event_attempt=event_attempt,
                    max_iterations=max_iterations,
                    timeout_seconds=timeout_seconds,
                    max_selected_files=max_selected_files,
                    include_agent_verification_commands=(
                        include_agent_verification_commands
                    ),
                    exploration_iterations=exploration_iterations,
                    max_active_themes_per_iteration=(max_active_themes_per_iteration),
                    candidate_count=normalized_candidate_count,
                    evaluator_verification_commands=verification_commands,
                    required_project_verification_commands=(required_project_commands),
                    generated_coverage_commands=tuple(
                        executed_generated_coverage_commands
                    ),
                    covered_requested_themes=tuple(prospective_covered_themes),
                    remaining_requested_themes=prospective_remaining_themes,
                    accepted_verification_commands=tuple(
                        dict.fromkeys(accepted_verification_commands)
                    ),
                    promotion_records=tuple(prospective_promotion_records),
                    candidate_runtime_records=tuple(candidate_runtime_records),
                    selected_candidate_proposal_ids=tuple(
                        selected_candidate_proposal_ids
                    ),
                    rejected_candidate_proposal_ids=tuple(
                        rejected_candidate_proposal_ids
                    ),
                    enforce_spec_maturity_gate=enforce_spec_maturity_gate,
                    engineering_organization_plan=engineering_org_plan,
                    code_max_runtime_summary=code_max_runtime_summary,
                    code_landing_metrics=prospective_metrics,
                    evidence_assessment=prospective_assessment,
                    evidence_records=candidate_evidence_ledger.records,
                    controller_event_run_id=controller_event_run.run_id,
                )
            _record_controller_event(
                event_store=event_store,
                run=controller_event_run,
                event_type="promotion_started",
                payload={
                    "iteration": _,
                    "proposal_id": proposal.proposal_id,
                    "baseline_digest": candidate_workspace.baseline_digest,
                    "candidate_digest": candidate_workspace.candidate_digest,
                    "changed_paths": candidate_workspace.changed_paths(),
                    "recovery_result_on_commit": (
                        canonicalize(recovery_result_on_commit)
                        if recovery_result_on_commit is not None
                        else None
                    ),
                },
                idempotency_key=f"promotion_started:{_}",
            )
            promotion = candidate_workspace.promote(
                verification_commands=integration_commands,
                timeout_seconds=timeout_seconds,
                executor=command_executor,
                retain_commit_receipt=(
                    event_store is not None and controller_event_run is not None
                ),
            )
            promotion_records.append(canonicalize(promotion))
            all_verification_results.extend(promotion.verification_results)
            if promotion.status != "release_candidate":
                _record_controller_event(
                    event_store=event_store,
                    run=controller_event_run,
                    event_type="promotion_completed",
                    payload={
                        "iteration": _,
                        "proposal_id": proposal.proposal_id,
                        "promotion": canonicalize(promotion),
                    },
                    idempotency_key=f"promotion:{_}",
                )
                promotion_feedback = _workspace_promotion_feedback(promotion)
                all_verification_results.append(promotion_feedback)
                previous_verification = (promotion_feedback,)
                final_mode = "verification_failed"
                break
            promoted_candidates += 1
            claim_patch_results.extend(candidate_claim_patch_results)
            verified_patch_results.extend(candidate_claim_patch_results)
            evidence_ledger = candidate_evidence_ledger
            for theme in _proposal_requested_theme_coverage(
                proposal=proposal,
                requested_themes=themes,
                task_specs=task_specs_v2,
                patch_results=patch_results,
            ):
                if theme not in covered_requested_themes:
                    covered_requested_themes.append(theme)
            remaining_after_verified = _remaining_requested_themes(
                requested_themes=themes,
                covered_themes=tuple(covered_requested_themes),
            )
            recovery_result: CodingAgentLoopResult | None = None
            post_repo_digest: str | None = None
            if event_store is not None and controller_event_run is not None:
                recovery_metrics = _code_landing_metrics(
                    source_report=source_report,
                    requested_themes=themes,
                    task_specs=task_specs_v2,
                    spec_maturity_gate=spec_maturity_gate,
                    repo_intelligence=repo_intelligence,
                    reproducer_plan=reproducer_plan,
                    engineering_org_plan=engineering_org_plan,
                    code_max_runtime_summary=code_max_runtime_summary,
                    patch_results=tuple(claim_patch_results),
                    verification_results=tuple(all_verification_results),
                    final_mode="verified",
                )
                recovery_verified_patches = _verified_effective_patch_count(
                    tuple(verified_patch_results),
                    verification_results=tuple(all_verification_results),
                    final_mode="verified",
                )
                recovery_assessment = assess_landing_evidence(
                    ledger=evidence_ledger,
                    effective_patch=bool(recovery_verified_patches),
                    integration_replayed=True,
                    promotion_status="release_candidate",
                    promotion_records=tuple(promotion_records),
                    event_chain_head=_controller_event_chain_head(
                        event_store=event_store,
                        run=controller_event_run,
                    ),
                )
                recovery_result = _build_promoted_recovery_result(
                    canonical_workspace_root=root,
                    report_id=_optional_string(source_report.get("report_id")),
                    package=package,
                    selected_files=selected_files,
                    proposals=tuple(proposals),
                    patch_results=tuple(claim_patch_results),
                    verification_results=tuple(all_verification_results),
                    controller_rounds=controller_rounds,
                    development_rounds=development_rounds,
                    infrastructure_only_rounds=infrastructure_only_rounds,
                    candidate_infra_retry_limit=candidate_infra_retry_limit,
                    candidate_infra_round_retry_limit=(
                        candidate_infra_round_retry_limit
                    ),
                    candidate_strategy_retry_limit=candidate_strategy_retry_limit,
                    candidate_slot_infra_retry_count=(candidate_slot_infra_retry_count),
                    candidate_slot_exhaustion_count=(candidate_slot_exhaustion_count),
                    unresolved_candidate_slot_exhaustion_count=(
                        unresolved_candidate_slot_exhaustion_count
                    ),
                    candidate_strategy_retry_count=candidate_strategy_retry_count,
                    candidate_strategy_exhaustion_count=(
                        candidate_strategy_exhaustion_count
                    ),
                    exploration_attempts=exploration_attempts,
                    risky_patch_bundles=risky_patch_bundles,
                    promoted_candidates=promoted_candidates,
                    rejected_by_gate=rejected_by_gate,
                    repair_iterations=repair_iterations,
                    agent=agent,
                    event_attempt=event_attempt,
                    max_iterations=max_iterations,
                    timeout_seconds=timeout_seconds,
                    max_selected_files=max_selected_files,
                    include_agent_verification_commands=(
                        include_agent_verification_commands
                    ),
                    exploration_iterations=exploration_iterations,
                    max_active_themes_per_iteration=(max_active_themes_per_iteration),
                    candidate_count=normalized_candidate_count,
                    evaluator_verification_commands=verification_commands,
                    required_project_verification_commands=(required_project_commands),
                    generated_coverage_commands=tuple(
                        executed_generated_coverage_commands
                    ),
                    covered_requested_themes=tuple(covered_requested_themes),
                    remaining_requested_themes=remaining_after_verified,
                    accepted_verification_commands=tuple(
                        dict.fromkeys(accepted_verification_commands)
                    ),
                    promotion_records=tuple(promotion_records),
                    candidate_runtime_records=tuple(candidate_runtime_records),
                    selected_candidate_proposal_ids=tuple(
                        selected_candidate_proposal_ids
                    ),
                    rejected_candidate_proposal_ids=tuple(
                        rejected_candidate_proposal_ids
                    ),
                    enforce_spec_maturity_gate=enforce_spec_maturity_gate,
                    engineering_organization_plan=engineering_org_plan,
                    code_max_runtime_summary=code_max_runtime_summary,
                    code_landing_metrics=recovery_metrics,
                    evidence_assessment=recovery_assessment,
                    evidence_records=evidence_ledger.records,
                    controller_event_run_id=controller_event_run.run_id,
                )
                post_repo_digest = build_workspace_execution_profile(root).repo_hash
            _record_controller_event(
                event_store=event_store,
                run=controller_event_run,
                event_type="promotion_completed",
                payload={
                    "iteration": _,
                    "proposal_id": proposal.proposal_id,
                    "promotion": canonicalize(promotion),
                    "post_repo_digest": post_repo_digest,
                    "recovery_result": (
                        canonicalize(recovery_result)
                        if recovery_result is not None
                        else None
                    ),
                },
                idempotency_key=f"promotion:{_}",
            )
            if event_store is not None and controller_event_run is not None:
                acknowledge_committed_workspace_promotion(
                    root,
                    candidate_digest=promotion.candidate_digest,
                )
            if remaining_after_verified and development_rounds < max_development_rounds:
                candidate_workspace.close()
                candidate_workspace = CandidateWorkspace.create(root)
                working_root = candidate_workspace.root
                candidate_claim_patch_results = []
                candidate_evidence_ledger = evidence_ledger
                pending_generated_coverage_commands = []
                previous_verification = ()
                continue
            break
        previous_verification = tuple(
            result for result in verification_results if result.status != "passed"
        )
        if not previous_verification:
            break

    candidate_workspace.close()

    iterations = controller_rounds
    failed_commands = tuple(
        result.command
        for result in all_verification_results
        if result.status != "passed"
    )
    report_id = _optional_string(source_report.get("report_id"))
    code_landing_metrics = _code_landing_metrics(
        source_report=source_report,
        requested_themes=themes,
        task_specs=task_specs_v2,
        spec_maturity_gate=spec_maturity_gate,
        repo_intelligence=repo_intelligence,
        reproducer_plan=reproducer_plan,
        engineering_org_plan=engineering_org_plan,
        code_max_runtime_summary=code_max_runtime_summary,
        patch_results=tuple(claim_patch_results),
        verification_results=tuple(all_verification_results),
        final_mode=final_mode,
    )
    verified_effective_patches = (
        _verified_effective_patch_count(
            tuple(verified_patch_results),
            verification_results=tuple(all_verification_results),
            final_mode="verified",
        )
        if verified_patch_results
        else _verified_effective_patch_count(
            tuple(claim_patch_results),
            verification_results=tuple(all_verification_results),
            final_mode=final_mode,
        )
    )
    latest_promotion_status = (
        str(promotion_records[-1].get("status")) if promotion_records else None
    )
    evidence_assessment = assess_landing_evidence(
        ledger=evidence_ledger,
        effective_patch=bool(verified_effective_patches),
        integration_replayed=(
            final_mode == "verified" and latest_promotion_status == "release_candidate"
        ),
        promotion_status=(
            latest_promotion_status if final_mode == "verified" else final_mode
        ),
        promotion_records=tuple(promotion_records),
        event_chain_head=_controller_event_chain_head(
            event_store=event_store,
            run=controller_event_run,
        ),
    )
    evidence_summary = canonicalize(evidence_assessment)
    last_verified_candidate = _last_verified_candidate_record(
        canonical_workspace_root=root,
        promotion_records=tuple(promotion_records),
    )
    result_payload = {
        "source_report_id": report_id,
        "package": package,
        "proposals": tuple(proposals),
        "patch_results": tuple(all_patch_results),
        "claim_patch_results": tuple(claim_patch_results),
        "verification_results": tuple(all_verification_results),
        "final_mode": final_mode,
        "agent_source": agent.source,
        "agent_model": agent.model,
        "exploration_attempts": exploration_attempts,
        "risky_patch_bundles": risky_patch_bundles,
        "promoted_candidates": promoted_candidates,
        "candidate_proposal_count": len(proposals),
        "promotion_attempt_count": promotion_attempt_count,
        "controller_round_count": controller_rounds,
        "development_round_count": development_rounds,
        "infrastructure_only_round_count": infrastructure_only_rounds,
        "provider_circuit_open": provider_circuit_open,
        "provider_circuit_reason_code": provider_circuit_reason_code,
        "provider_circuit_consecutive_exhausted_slots": (
            consecutive_provider_slot_exhaustions
        ),
        "rejected_by_gate": rejected_by_gate,
        "repair_iterations": repair_iterations,
        "promotion_records": tuple(promotion_records),
        "candidate_runtime_records": tuple(candidate_runtime_records),
        "selected_candidate_proposal_ids": tuple(selected_candidate_proposal_ids),
        "rejected_candidate_proposal_ids": tuple(rejected_candidate_proposal_ids),
        "evidence": evidence_summary,
        "evidence_records": evidence_ledger.records,
        "controller_event_run_id": (
            controller_event_run.run_id if controller_event_run else None
        ),
        "reasoning_effort": getattr(agent, "reasoning_effort", None),
        "requested_reasoning_effort": getattr(
            agent, "requested_reasoning_effort", None
        ),
        "run_config": {
            "event_attempt": event_attempt,
            "max_iterations": max_iterations,
            "timeout_seconds": timeout_seconds,
            "max_selected_files": max_selected_files,
            "include_agent_verification_commands": include_agent_verification_commands,
            "exploration_iterations": exploration_iterations,
            "max_active_themes_per_iteration": max_active_themes_per_iteration,
            "candidate_count": normalized_candidate_count,
            "candidate_infrastructure_retry_limit": (candidate_infra_retry_limit),
            "candidate_infrastructure_round_retry_limit": (
                candidate_infra_round_retry_limit
            ),
            "provider_circuit_exhausted_slot_limit": (
                provider_circuit_exhausted_slot_limit
            ),
            "candidate_strategy_retry_limit": candidate_strategy_retry_limit,
            "candidate_infrastructure_retry_delays": (candidate_infra_retry_delays),
            "agent_runtime": workspace_coding_agent_runtime_identity(agent),
            "candidate_proposal_count": len(proposals),
            "promotion_attempt_count": promotion_attempt_count,
            "canonical_promotion_count": promoted_candidates,
            "controller_round_count": controller_rounds,
            "development_round_count": development_rounds,
            "infrastructure_only_round_count": infrastructure_only_rounds,
            "provider_circuit_open": provider_circuit_open,
            "provider_circuit_reason_code": provider_circuit_reason_code,
            "provider_circuit_consecutive_exhausted_slots": (
                consecutive_provider_slot_exhaustions
            ),
            "controller_event_run_id": (
                controller_event_run.run_id if controller_event_run else None
            ),
            "evaluator_verification_command_count": len(verification_commands),
            "covered_requested_themes": tuple(covered_requested_themes),
            "remaining_requested_themes": _remaining_requested_themes(
                requested_themes=themes,
                covered_themes=tuple(covered_requested_themes),
            ),
            "accepted_verification_commands": tuple(
                dict.fromkeys(accepted_verification_commands)
            ),
            "last_verified_candidate": last_verified_candidate,
            "selected_candidate_proposal_ids": tuple(selected_candidate_proposal_ids),
            "rejected_candidate_proposal_ids": tuple(rejected_candidate_proposal_ids),
        },
        "patch_coverage": _patch_coverage_summary(
            patch_results=tuple(claim_patch_results),
            verification_results=tuple(all_verification_results),
            generated_coverage_commands=tuple(executed_generated_coverage_commands),
            proposals=tuple(proposals),
        ),
        "code_landing_metrics": code_landing_metrics,
    }
    provenance_hash = stable_hash(_stable_controller_event_payload(result_payload))
    development_stage = _development_stage(
        final_mode=final_mode,
        promoted_candidates=promoted_candidates,
        exploration_attempts=exploration_attempts,
        verified_effective_patches=verified_effective_patches,
    )
    result = CodingAgentLoopResult(
        run_id=f"workspace_agent_loop_{provenance_hash[:24]}",
        developer_mode=development_stage,
        source_report_id=report_id,
        package=package,
        selected_files=selected_files,
        proposals=tuple(proposals),
        patch_results=tuple(claim_patch_results),
        verification_results=tuple(all_verification_results),
        iterations=iterations,
        final_mode=final_mode,
        failed_commands=failed_commands,
        provenance_hash=provenance_hash,
        development_stage=development_stage,
        exploration_attempts=exploration_attempts,
        risky_patch_bundles=risky_patch_bundles,
        promoted_candidates=promoted_candidates,
        verified_effective_patches=verified_effective_patches,
        rejected_by_gate=rejected_by_gate,
        repair_iterations=repair_iterations,
        reasoning_effort=getattr(agent, "reasoning_effort", None),
        requested_reasoning_effort=getattr(agent, "requested_reasoning_effort", None),
        run_config={
            "event_attempt": event_attempt,
            "max_iterations": max_iterations,
            "timeout_seconds": timeout_seconds,
            "max_selected_files": max_selected_files,
            "include_agent_verification_commands": include_agent_verification_commands,
            "exploration_iterations": exploration_iterations,
            "max_active_themes_per_iteration": max_active_themes_per_iteration,
            "candidate_count": normalized_candidate_count,
            "candidate_infrastructure_retry_limit": (candidate_infra_retry_limit),
            "candidate_infrastructure_round_retry_limit": (
                candidate_infra_round_retry_limit
            ),
            "provider_circuit_exhausted_slot_limit": (
                provider_circuit_exhausted_slot_limit
            ),
            "candidate_strategy_retry_limit": candidate_strategy_retry_limit,
            "candidate_infrastructure_retry_delays": (candidate_infra_retry_delays),
            "agent_runtime": workspace_coding_agent_runtime_identity(agent),
            "candidate_proposal_count": len(proposals),
            "promotion_attempt_count": promotion_attempt_count,
            "canonical_promotion_count": promoted_candidates,
            "controller_round_count": controller_rounds,
            "development_round_count": development_rounds,
            "infrastructure_only_round_count": infrastructure_only_rounds,
            "provider_circuit_open": provider_circuit_open,
            "provider_circuit_reason_code": provider_circuit_reason_code,
            "provider_circuit_consecutive_exhausted_slots": (
                consecutive_provider_slot_exhaustions
            ),
            "controller_event_run_id": (
                controller_event_run.run_id if controller_event_run else None
            ),
            "evaluator_verification_commands": verification_commands,
            "required_project_verification_commands": required_project_commands,
            "generated_coverage_commands": tuple(executed_generated_coverage_commands),
            "covered_requested_themes": tuple(covered_requested_themes),
            "remaining_requested_themes": _remaining_requested_themes(
                requested_themes=themes,
                covered_themes=tuple(covered_requested_themes),
            ),
            "accepted_verification_commands": tuple(
                dict.fromkeys(accepted_verification_commands)
            ),
            "promotion_records": tuple(promotion_records),
            "last_verified_candidate": last_verified_candidate,
            "candidate_runtime_records": tuple(candidate_runtime_records),
            "candidate_slot_infra_retry_count": (candidate_slot_infra_retry_count),
            "candidate_slot_exhaustion_count": candidate_slot_exhaustion_count,
            "unresolved_candidate_slot_exhaustion_count": (
                unresolved_candidate_slot_exhaustion_count
            ),
            "candidate_strategy_retry_count": candidate_strategy_retry_count,
            "candidate_strategy_exhaustion_count": (
                candidate_strategy_exhaustion_count
            ),
            "selected_candidate_proposal_ids": tuple(selected_candidate_proposal_ids),
            "rejected_candidate_proposal_ids": tuple(rejected_candidate_proposal_ids),
            "enforce_spec_maturity_gate": enforce_spec_maturity_gate,
            "engineering_organization_plan": _engineering_org_plan_agent_summary(
                engineering_org_plan
            ),
            "project_command_baseline": project_command_baseline,
            "code_max_runtime": code_max_runtime_summary,
        },
        patch_coverage=_patch_coverage_summary(
            patch_results=tuple(claim_patch_results),
            verification_results=tuple(all_verification_results),
            generated_coverage_commands=tuple(executed_generated_coverage_commands),
            proposals=tuple(proposals),
        ),
        code_landing_metrics=code_landing_metrics,
        evidence_level=evidence_assessment.level,
        evidence_hash=evidence_assessment.evidence_hash,
        evidence_summary=evidence_summary,
        evidence_records=evidence_ledger.records,
    )
    _complete_controller_event_run(
        event_store=event_store,
        run=controller_event_run,
        result=result,
        workspace_root=root,
    )
    return result


def run_coding_agent_development_loop_from_report_path(
    *,
    report_path: Path,
    workspace_root: Path,
    agent: WorkspaceCodingAgent,
    verification_commands: tuple[str, ...] = (),
    required_project_verification_commands: tuple[str, ...] = (),
    max_iterations: int = DEFAULT_WORKSPACE_CODING_MAX_ITERATIONS,
    timeout_seconds: float = 60.0,
    max_selected_files: int = DEFAULT_WORKSPACE_CODING_MAX_SELECTED_FILES,
    include_agent_verification_commands: bool = True,
    exploration_iterations: int = DEFAULT_WORKSPACE_CODING_EXPLORATION_ITERATIONS,
    enforce_spec_maturity_gate: bool = False,
    max_active_themes_per_iteration: int = DEFAULT_WORKSPACE_CODING_MAX_ACTIVE_THEMES,
    executor: CommandExecutor | None = None,
    candidate_count: int = 1,
    event_store: SQLiteEventStore | None = None,
    event_attempt: int = 1,
) -> CodingAgentLoopResult:
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    return run_coding_agent_development_loop(
        source_report=source_report,
        workspace_root=workspace_root,
        agent=agent,
        verification_commands=verification_commands,
        required_project_verification_commands=(required_project_verification_commands),
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
        max_selected_files=max_selected_files,
        include_agent_verification_commands=include_agent_verification_commands,
        exploration_iterations=exploration_iterations,
        enforce_spec_maturity_gate=enforce_spec_maturity_gate,
        max_active_themes_per_iteration=max_active_themes_per_iteration,
        executor=executor,
        candidate_count=candidate_count,
        event_store=event_store,
        event_attempt=event_attempt,
    )


def _agent_payload(
    *,
    source_report: Mapping[str, Any],
    workspace_root: Path,
    package: WorkspacePackageInfo,
    file_contexts: tuple[WorkspaceFileContext, ...],
    previous_verification: tuple[WorkspaceVerificationResult, ...],
    iteration: int,
    development_stage: str,
) -> dict[str, Any]:
    themes = _extract_themes(source_report)
    if isinstance(source_report.get("development_task_specs_v2"), (list, tuple)):
        task_specs_v2 = source_report.get("development_task_specs_v2")
    else:
        repo_pack_for_payload = build_repo_intelligence_pack(
            workspace_root,
            themes=themes,
            task_specs=build_development_intent_specs(source_report),
        )
        task_specs = build_development_task_specs_v2(
            source_report,
            repo_intelligence=repo_pack_for_payload,
        )
        task_specs_v2 = tuple(canonicalize(spec) for spec in task_specs)
    spec_maturity_gate = source_report.get("spec_maturity_gate")
    if not isinstance(spec_maturity_gate, Mapping):
        spec_maturity_gate = build_spec_maturity_gate(
            build_development_task_specs_v2(source_report)
        )
    repo_intelligence_pack = source_report.get("repo_intelligence_pack")
    if not isinstance(repo_intelligence_pack, Mapping):
        repo_intelligence_pack = _repo_intelligence_agent_summary(
            build_repo_intelligence_pack(workspace_root, themes=themes)
        )
    reproducer_plan = source_report.get("reproducer_plan")
    if not isinstance(reproducer_plan, Mapping):
        reproducer_plan = _reproducer_plan_agent_summary(
            build_reproducer_plan(
                source_report=source_report,
                workspace_root=workspace_root,
                task_specs=build_development_task_specs_v2(source_report),
                confirm_base=False,
            )
        )
    engineering_organization_plan = source_report.get("engineering_organization_plan")
    if not isinstance(engineering_organization_plan, Mapping):
        repo_pack_for_org = build_repo_intelligence_pack(workspace_root, themes=themes)
        task_specs_for_org = build_development_task_specs_v2(
            source_report,
            repo_intelligence=repo_pack_for_org,
        )
        repro_plan_for_org = build_reproducer_plan(
            source_report=source_report,
            workspace_root=workspace_root,
            task_specs=task_specs_for_org,
            repo_intelligence=repo_pack_for_org,
            confirm_base=False,
        )
        engineering_organization_plan = _engineering_org_plan_agent_summary(
            build_engineering_organization_plan(
                source_report=source_report,
                task_specs=task_specs_for_org,
                repo_intelligence=repo_pack_for_org,
                reproducer_plan=repro_plan_for_org,
            )
        )
    code_max_runtime = source_report.get("code_max_runtime")
    if not isinstance(code_max_runtime, Mapping):
        repo_pack_for_context = build_repo_intelligence_pack(
            workspace_root, themes=themes
        )
        task_specs_for_context = build_development_task_specs_v2(
            source_report,
            repo_intelligence=repo_pack_for_context,
        )
        code_max_runtime = build_code_max_context_summary(
            source_report=source_report,
            workspace_root=workspace_root,
            task_specs=task_specs_for_context,
            confirm_base=False,
            timeout_seconds=20.0,
        )
    return canonicalize(
        {
            "iteration": iteration,
            "development_stage": development_stage,
            "workspace_root_name": workspace_root.name,
            "package": package,
            "source_report": _report_summary(source_report),
            "selected_files": _agent_file_contexts(file_contexts),
            "selected_file_excerpt_policy": {
                "max_excerpt_chars": _workspace_coding_max_excerpt_chars(),
                "prompt_file_limit": _workspace_coding_prompt_file_limit(),
                "full_file_not_in_prompt": True,
                "path_hash_and_size_preserved": True,
            },
            "previous_verification_failures": _verification_summary(
                previous_verification
            ),
            "repair_mode": any(
                result.status != "passed" for result in previous_verification
            ),
            "repair_policy": _repair_policy(previous_verification),
            "privacy_boundary": "public_report_and_selected_workspace_files_only",
            "patch_policy": {
                "operations": (
                    "create_or_replace",
                    "replace_fragment",
                    "append_if_missing",
                ),
                "relative_paths_only": True,
                "max_patch_count": 64,
                "max_total_patch_bytes": 500000,
                "verification_runs_after_patch": True,
                "json_files_must_parse": True,
                "destructive_replace_fragments_blocked": True,
            },
            "development_intent_specs": _agent_intent_specs_summary(
                build_development_intent_specs(source_report)
            ),
            "development_task_specs_v2": _agent_task_specs_summary(task_specs_v2),
            "spec_maturity_gate": spec_maturity_gate,
            "repo_intelligence_pack": _agent_repo_intelligence_summary(
                repo_intelligence_pack
            ),
            "reproducer_plan": reproducer_plan,
            "required_project_verification_commands": tuple(
                source_report.get("required_project_verification_commands", ())
            )[:16],
            "engineering_organization_plan": engineering_organization_plan,
            "code_max_runtime": _agent_code_max_runtime_summary(code_max_runtime),
            "code_landing_runtime_policy": {
                "spec_maturity_minimum_for_strong_claim": "testable",
                "reproducer_first": True,
                "syntax_checks_are_secondary": True,
                "behavior_or_repro_required_for_source_claim": True,
                "final_claim_requires_maintainer_landing_gate": True,
            },
            "maintainer_landing_gate": _maintainer_gate_policy(source_report),
            "workspace_progress": source_report.get("workspace_progress", {}),
        }
    )


def _stage_prompt(stage: str) -> str:
    if stage == "explore":
        return (
            "Stage: explore. You may propose bold, high-upside changes, broad "
            "refactors, and risky prototypes. This runs in a sandbox copy and is "
            "not evidence yet. Keep paths relative and do not touch secrets or "
            "workspace-external files."
        )
    if stage == "claim":
        return (
            "Stage: claim. Only propose precise, non-destructive changes that can "
            "survive fixed evaluator-owned verification and be reported as evidence."
        )
    return (
        "Stage: promote. Turn the best exploratory idea into a reviewable patch "
        "candidate that can pass the guarded kernel."
    )


def _report_summary(source_report: Mapping[str, Any]) -> dict[str, Any]:
    proposal = source_report.get("proposal")
    if isinstance(proposal, Mapping):
        proposal_summary = {
            key: proposal.get(key)
            for key in (
                "proposal_id",
                "artifact_id",
                "requested_direction",
                "priority_themes",
                "support_refs",
            )
            if key in proposal
        }
    else:
        proposal_summary = {}
    return {
        key: source_report.get(key)
        for key in (
            "report_id",
            "case_id",
            "evaluation_episode_id",
            "product_name",
            "proposed_direction",
            "proposed_themes",
            "experience_summary",
            "upgrade_suggestion_summary",
            "public_feedback_summary",
            "demand_coverage",
            "community_coverage",
            "evaluation_stage",
            "capability_transfer_probe",
            "prior_capability_summary",
            "maintainer_landing_gate",
            "workspace_progress",
        )
        if key in source_report
    } | {"proposal": proposal_summary}


def _maintainer_gate_policy(source_report: Mapping[str, Any]) -> dict[str, Any]:
    configured = source_report.get("maintainer_landing_gate")
    if isinstance(configured, Mapping):
        enabled = bool(configured.get("enabled", True))
        require_behavior = bool(configured.get("require_behavior_verification", True))
        require_intent_trace = bool(configured.get("require_intent_trace", True))
        require_test_or_repro = bool(configured.get("require_test_or_repro", True))
    else:
        enabled = bool(source_report.get("development_intent_specs"))
        require_behavior = True
        require_intent_trace = True
        require_test_or_repro = True
    return {
        "enabled": enabled,
        "require_behavior_verification": require_behavior,
        "require_intent_trace": require_intent_trace,
        "require_test_or_repro": require_test_or_repro,
        "claim_boundary": (
            "A formal evidence patch must be tied to a public development "
            "intent, include behavior-level verification, and look like a "
            "reviewable maintainer change rather than a theme-only edit."
        ),
    }


def _repair_policy(
    previous_verification: tuple[WorkspaceVerificationResult, ...],
) -> dict[str, Any]:
    blocked_reasons = tuple(
        result.blocked_reason or result.stderr_tail
        for result in previous_verification
        if result.status == "blocked"
    )
    return {
        "blocked_reasons": blocked_reasons,
        "replace_fragment_not_found": any(
            "replace_fragment_not_found" in reason for reason in blocked_reasons
        ),
        "instruction": (
            "When a replace_fragment patch was blocked, the next proposal must not "
            "reuse the same old fragment. Use an exact substring visible in the "
            "current selected file excerpts, or switch to append_if_missing / "
            "create_or_replace on a new support file."
        ),
    }


def _workspace_patch_json_schema() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "artifact_id": {"type": ["string", "null"], "maxLength": 160},
            "themes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {"type": "string", "maxLength": 80},
            },
            "patches": {
                "type": "array",
                "minItems": 1,
                "maxItems": 24,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "patch_id": {"type": "string", "maxLength": 120},
                        "path": {"type": "string", "maxLength": 220},
                        "operation": {"type": "string", "maxLength": 40},
                        "content": {"type": "string", "maxLength": 80000},
                        "old": {"type": "string", "maxLength": 80000},
                        "new": {"type": "string", "maxLength": 80000},
                        "rationale": {"type": "string", "maxLength": 1000},
                    },
                    "required": [
                        "patch_id",
                        "path",
                        "operation",
                        "content",
                        "old",
                        "new",
                        "rationale",
                    ],
                },
            },
            "verification_commands": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "string",
                    "maxLength": MAX_AGENT_VERIFICATION_COMMAND_CHARS,
                },
            },
            "rationale": {"type": "string", "maxLength": 2000},
            "support_refs": {
                "type": "array",
                "maxItems": 16,
                "items": {"type": "string", "maxLength": 160},
            },
            "blocked_reason": {"type": ["string", "null"], "maxLength": 300},
        },
        "required": [
            "artifact_id",
            "themes",
            "patches",
            "verification_commands",
            "rationale",
            "support_refs",
            "blocked_reason",
        ],
    }


def _proposal_from_parts(
    *,
    source: str,
    model: str | None,
    iteration: int,
    artifact_id: str | None,
    themes: tuple[str, ...],
    patches: tuple[WorkspaceFilePatch, ...],
    verification_commands: tuple[str, ...],
    selected_files: tuple[str, ...],
    rationale: str,
    support_refs: tuple[str, ...],
    raw_response_hash: str | None,
    blocked_reason: str | None = None,
) -> CodingAgentPatchProposal:
    effective_blocked_reason = blocked_reason if not patches else None
    proposal_payload = {
        "source": source,
        "model": model,
        "iteration": iteration,
        "artifact_id": artifact_id,
        "themes": themes,
        "patches": patches,
        "verification_commands": verification_commands,
        "selected_files": selected_files,
        "rationale": rationale,
        "support_refs": support_refs,
        "raw_response_hash": raw_response_hash,
        "blocked_reason": effective_blocked_reason,
    }
    return CodingAgentPatchProposal(
        proposal_id=f"workspace_patch_proposal_{stable_hash(proposal_payload)[:24]}",
        source=source,
        model=model,
        iteration=iteration,
        artifact_id=artifact_id,
        themes=themes,
        patches=patches,
        verification_commands=verification_commands,
        selected_files=selected_files,
        rationale=rationale,
        support_refs=support_refs,
        raw_response_hash=raw_response_hash,
        blocked_reason=effective_blocked_reason,
    )


def _blocked_proposal(
    *,
    source: str,
    model: str | None,
    iteration: int,
    artifact_id: str | None,
    themes: tuple[str, ...],
    selected_files: tuple[str, ...],
    reason: str,
    raw_response_hash: str | None = None,
) -> CodingAgentPatchProposal:
    return _proposal_from_parts(
        source=source,
        model=model,
        iteration=iteration,
        artifact_id=artifact_id,
        themes=themes,
        patches=(),
        verification_commands=(),
        selected_files=selected_files,
        rationale="Workspace coding agent could not produce a valid guarded patch proposal.",
        support_refs=(),
        raw_response_hash=raw_response_hash,
        blocked_reason=reason,
    )


def _patch_from_json(item: Mapping[str, Any], index: int) -> WorkspaceFilePatch:
    operation = str(item["operation"])
    if operation not in {"create_or_replace", "replace_fragment", "append_if_missing"}:
        operation = "create_or_replace"
    patch_id = str(item.get("patch_id") or f"agent_patch_{index:02d}")[:120]
    return WorkspaceFilePatch(
        patch_id=patch_id,
        path=_safe_relative_path(str(item["path"])),
        operation=operation,
        content=str(item.get("content", ""))[:500_000],
        old=str(item.get("old", ""))[:500_000],
        new=str(item.get("new", ""))[:500_000],
        rationale=str(item.get("rationale", ""))[:1000],
    )


def _agent_verification_commands_from_json(commands: Any) -> tuple[str, ...]:
    if not isinstance(commands, (list, tuple)):
        return ()
    accepted: list[str] = []
    for raw_command in commands[:12]:
        command = str(raw_command).strip()
        if not command or len(command) > MAX_AGENT_VERIFICATION_COMMAND_CHARS:
            continue
        try:
            shlex.split(command)
        except ValueError:
            continue
        accepted.append(command)
    return _dedupe(accepted)


def _extract_themes(source_report: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("proposed_themes",):
        raw = source_report.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            return tuple(str(item)[:80] for item in raw)
    proposal = source_report.get("proposal")
    if isinstance(proposal, Mapping):
        raw = proposal.get("priority_themes")
        if isinstance(raw, (list, tuple)) and raw:
            return tuple(str(item)[:80] for item in raw)
    return ("product_improvement",)


def _normalize_development_intent_spec(
    item: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    theme = str(item.get("theme") or f"unspecified_theme_{index}")[:80]
    candidate_path_hints = _string_tuple(
        item.get("candidate_path_hints"),
        fallback=_candidate_path_hints_for_theme(theme),
        max_items=12,
    )
    behavior_surface = _string_tuple(
        item.get("behavior_surface"), fallback=(), max_items=8
    )
    payload = {
        "theme": theme,
        "index": index,
        "candidate_path_hints": candidate_path_hints,
        "support_refs": item.get("support_refs"),
    }
    intent_id = str(
        item.get("intent_id")
        or f"intent_{_slug_label(theme)}_{stable_hash(payload)[:10]}"
    )
    generated_intent_spec = bool(item.get("_generated_intent_spec"))
    return {
        "intent_id": intent_id[:120],
        "theme": theme,
        "_source_explicit_intent_spec": not generated_intent_spec,
        "_generated_intent_spec": generated_intent_spec,
        "_explicit_candidate_path_hints": "candidate_path_hints" in item,
        "_explicit_acceptance_tests": "acceptance_tests" in item,
        "user_pain": str(item.get("user_pain") or _default_user_pain(theme))[:500],
        "reproduction_steps": _string_tuple(
            item.get("reproduction_steps"),
            fallback=_reproduction_steps_for_theme(theme, ()),
            max_items=8,
        ),
        "expected_behavior": str(
            item.get("expected_behavior") or _expected_behavior_for_theme(theme)
        )[:500],
        "observed_behavior": str(item.get("observed_behavior") or "")[:500],
        "task_type": str(item.get("task_type") or "")[:80],
        "non_goals": _string_tuple(item.get("non_goals"), fallback=(), max_items=8),
        "repo_entrypoints": _string_tuple(
            item.get("repo_entrypoints"), fallback=(), max_items=12
        ),
        "relevant_symbols_hint": _string_tuple(
            item.get("relevant_symbols_hint"),
            fallback=(),
            max_items=12,
        ),
        "reproduction_expected_failure": _optional_string(
            item.get("reproduction_expected_failure")
        ),
        "risk_level": str(item.get("risk_level") or "")[:40],
        "behavior_surface": behavior_surface,
        "candidate_path_hints": candidate_path_hints,
        "acceptance_tests": _string_tuple(
            item.get("acceptance_tests"),
            fallback=_acceptance_tests_for_theme(
                theme,
                candidate_path_hints,
                behavior_surface=behavior_surface
                or _behavior_surface_for_theme(theme),
            ),
            max_items=8,
            max_chars=MAX_AGENT_VERIFICATION_COMMAND_CHARS,
        ),
        "support_refs": _string_tuple(
            item.get("support_refs"), fallback=(), max_items=16
        ),
        "evidence_refs": _string_tuple(
            item.get("evidence_refs"), fallback=(), max_items=16
        ),
        "contract_dimensions": _string_tuple(
            item.get("contract_dimensions"),
            fallback=_contract_dimensions_for_theme(
                theme,
                _string_tuple(item.get("evidence_refs"), fallback=(), max_items=16),
            ),
            max_items=16,
        ),
        "contract_witnesses": _normalized_contract_witnesses(
            item.get("contract_witnesses")
        ),
        "boundary_conditions": _string_tuple(
            item.get("boundary_conditions"),
            fallback=(),
            max_items=16,
        ),
    }


def _real_experience_observation_summaries(
    source_report: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    summary = source_report.get("real_product_experience_summary")
    if not isinstance(summary, Mapping):
        return ()
    tasks = summary.get("tasks", ())
    observations: list[dict[str, str]] = []
    if isinstance(tasks, (list, tuple)):
        for task in tasks[:8]:
            if not isinstance(task, Mapping):
                continue
            observations.append(
                {
                    "task_type": str(task.get("task_type") or "")[:120],
                    "stage": str(task.get("stage") or "")[:120],
                    "success": str(bool(task.get("success"))),
                    "failure_event": str(task.get("failure_event") or "")[:160],
                }
            )
    return tuple(observations)


def _user_pain_for_theme(
    theme: str,
    source_report: Mapping[str, Any],
    observations: tuple[dict[str, str], ...],
    *,
    public_failure_events: tuple[str, ...] = (),
) -> str:
    observation_failures = tuple(
        observation["failure_event"]
        for observation in observations
        if observation.get("failure_event")
    )
    failures = tuple(dict.fromkeys((*public_failure_events, *observation_failures)))
    product_name = str(source_report.get("product_name") or "the product")
    if failures:
        return (
            f"Public users hit {theme} friction in {product_name}; "
            f"observed failure signals: {', '.join(failures[:3])}."
        )
    return _default_user_pain(theme)


def _default_user_pain(theme: str) -> str:
    readable = theme.replace("_", " ")
    if theme in CREATIVE_FEATURE_THEMES:
        return (
            f"Experience-grounded public proposals identify {readable} as a "
            "candidate capability, not as a verified defect."
        )
    return f"Public feedback indicates unresolved {readable} friction."


def _reproduction_steps_for_theme(
    theme: str,
    observations: tuple[dict[str, str], ...],
    *,
    public_failure_events: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if theme in CREATIVE_FEATURE_THEMES:
        return (
            "Replay the public experience path that motivated the feature proposal.",
            f"Prototype the smallest {theme.replace('_', ' ')} contract without changing unrelated behavior.",
            "Demonstrate one supported workflow and one compatibility or fallback case.",
            "Retain a baseline assertion showing the capability was absent before the patch.",
        )
    steps = [
        f"Start from the frozen starter workspace and exercise the {theme.replace('_', ' ')} path.",
        "Record the observed failure, confusing output, or missing workflow behavior.",
    ]
    for failure_event in public_failure_events[:4]:
        steps.append(
            f"Reproduce public aggregate failure signal {failure_event} through the "
            "highest supported entrypoint and retain the baseline-failing assertion."
        )
    for observation in observations[:3]:
        task_type = observation.get("task_type")
        stage = observation.get("stage")
        failure = observation.get("failure_event")
        if task_type:
            suffix = f" and inspect {failure}" if failure else ""
            steps.append(
                f"Replay public experience task {task_type} at stage {stage}{suffix}."
            )
    return tuple(dict.fromkeys(steps))


def _public_failure_events_for_theme(
    source_report: Mapping[str, Any],
    theme: str,
) -> tuple[str, ...]:
    markers = _THEME_FAILURE_EVENT_MARKERS.get(
        theme,
        tuple(part for part in theme.lower().split("_") if len(part) >= 4),
    )
    ranked: list[tuple[int, str]] = []
    for summary_key in ("experience_summary", "real_product_experience_summary"):
        summary = source_report.get(summary_key)
        if not isinstance(summary, Mapping):
            continue
        failure_events = summary.get("failure_events")
        if not isinstance(failure_events, Mapping):
            continue
        for raw_event, raw_count in failure_events.items():
            event = str(raw_event)[:160]
            normalized = event.lower()
            if not event or not any(marker in normalized for marker in markers):
                continue
            try:
                count = max(0, int(raw_count))
            except (TypeError, ValueError):
                count = 0
            ranked.append((count, event))
    ordered = sorted(ranked, key=lambda item: (-item[0], item[1]))
    return tuple(dict.fromkeys(event for _, event in ordered))[:16]


def _contract_dimensions_for_theme(
    theme: str,
    public_failure_events: tuple[str, ...],
) -> tuple[str, ...]:
    dimensions: list[str] = list(public_failure_events)
    if theme == "pattern_filtering_consistency":
        dimensions.extend(
            (
                "standard pattern grammar: *, ?, [], and **",
                "sync, async, CLI, and configuration wrapper continuity",
                "path separator normalization and supported negation semantics",
                "scalar and iterable include/exclude inputs",
            )
        )
    elif theme == "ignore_pattern_reliability":
        dimensions.extend(
            (
                "default repository ignore behavior",
                ".gitignore and tool-specific ignore file continuity",
                "explicit include/exclude precedence",
                "sync, async, CLI, and configuration wrapper continuity",
            )
        )
    elif theme == "include_submodules":
        dimensions.extend(
            (
                "default and opt-in submodule behavior",
                "local and remote repository continuity",
                "sync, async, CLI, and configuration wrapper continuity",
            )
        )
    elif theme == "max_file_size_enforcement":
        dimensions.extend(
            (
                "byte-boundary behavior below, at, and above the configured limit",
                "default and explicit maximum-size continuity",
            )
        )
    elif theme == "http_client_portability":
        dimensions.extend(
            (
                "portable HTTP client behavior without external shell tools",
                "timeout, non-success response, and transport failure semantics",
            )
        )
    elif theme in CREATIVE_FEATURE_THEMES:
        dimensions.extend(
            (
                "minimal public capability contract",
                "backward compatibility and fallback behavior",
                "configuration or API discoverability",
                "behavior-level acceptance example",
            )
        )
    return tuple(dict.fromkeys(dimensions))[:16]


def _boundary_conditions_for_spec(
    *,
    theme: str,
    user_pain: str,
    observed_behavior: str,
    expected_behavior: str,
    reproduction_recipe: tuple[str, ...],
    contract_dimensions: tuple[str, ...],
) -> tuple[str, ...]:
    """Derive public, contrastive boundary states before implementation begins."""

    text = " ".join(
        (
            theme,
            user_pain,
            observed_behavior,
            expected_behavior,
            *reproduction_recipe,
            *contract_dimensions,
        )
    ).casefold()
    structural_text = " ".join((theme, *contract_dimensions)).casefold()
    boundaries: list[str] = []

    partial_state_markers = (
        "partial object",
        "partially initialized",
        "partially initialised",
        "partially constructed",
        "uninitialized",
        "uninitialised",
        "not initialized",
        "not initialised",
        "never assigned",
        "not assigned",
        "missing attribute",
        "absent attribute",
        "missing field",
        "absent field",
    )
    assignment_before_ready = re.search(
        r"\bbefore\b.{0,100}\b(?:assign(?:ed|ment)|initiali[sz](?:ed|ation)|construct(?:ed|ion))\b",
        text,
    )
    if (
        any(marker in text for marker in partial_state_markers)
        or assignment_before_ready
    ):
        boundaries.extend(
            (
                "required attribute or field is absent before assignment",
                "required attribute or field is present but explicitly unset or null",
                "fully initialized valid state preserves existing behavior",
            )
        )
    if "mask" in text and "original" in text and "error" in text:
        boundaries.append(
            "diagnostic handling preserves the original construction error context"
        )

    integer_contract = any(
        marker in text
        for marker in (
            "integer",
            "base-10",
            "nonpositive",
            "non-positive",
            "positive number",
            "negative number",
        )
    )
    environment_contract = any(
        marker in text
        for marker in (
            "environment",
            "envvar",
            "env var",
            "configuration value",
            "config value",
        )
    )
    if integer_contract:
        if environment_contract:
            boundaries.extend(
                (
                    "configuration source is absent so the documented default or fallback applies",
                    "configuration source is present as an empty or whitespace-only scalar and is not silently conflated with absence",
                    "explicit option and environment source precedence preserves the documented attribution",
                )
            )
        boundaries.extend(
            (
                "malformed scalar uses alphabetic, fractional, or non-base-10 syntax",
                "integer input immediately below the documented lower bound",
                "integer input exactly at the documented lower bound",
                "smallest valid integer preserves existing behavior",
                "representative valid integer above the minimum preserves existing behavior",
                "public adapters and the owning abstraction agree on every scalar state",
            )
        )

    if any(
        marker in structural_text for marker in ("limit", "maximum", "minimum", "size")
    ):
        boundaries.extend(
            (
                "numeric input immediately below the documented boundary",
                "numeric input exactly at the documented boundary",
                "numeric input immediately above the documented boundary",
            )
        )
    if any(marker in text for marker in ("empty", "collection", "iterable", "list")):
        boundaries.extend(
            (
                "empty input collection",
                "single-item input collection",
                "multi-item input collection",
            )
        )
    if any(marker in text for marker in ("optional", "boolean", "enabled", "disabled")):
        boundaries.extend(
            (
                "option omitted so the documented default applies",
                "option explicitly disabled",
                "option explicitly enabled",
            )
        )
    if any(
        marker in structural_text for marker in ("path", "pattern", "glob", "separator")
    ):
        boundaries.extend(
            (
                "minimal valid path or pattern",
                "nested or composed path or pattern",
                "invalid path or pattern fails with the documented semantics",
            )
        )

    syntax_markers = (
        "format",
        "formatter",
        "parse",
        "parser",
        "syntax",
        "token",
        "grammar",
    )
    if any(marker in text for marker in syntax_markers):
        boundaries.extend(
            (
                "reported syntax form through the highest public entrypoint",
                "nearest sibling syntax forms handled by the same owning abstraction",
                "nested or composed syntax state preserves output validity",
                "successful output is stable under a second processing pass",
            )
        )
    if "comment" in text:
        boundaries.extend(
            (
                "standalone content before and after each implicated structural boundary",
                "content preservation without relocation that changes semantics",
                "comment-free neighboring state preserves existing behavior",
            )
        )
    if "comprehension" in text:
        boundaries.extend(
            (
                "single-target and unpacking-target comprehension forms",
                "synchronous and asynchronous grammar variants when supported",
                "nearest non-comprehension expression forms sharing the same splitter",
            )
        )

    if not boundaries:
        boundaries.extend(
            (
                "reported failing input or state",
                "nearest valid input or state preserves existing behavior",
            )
        )
    return tuple(dict.fromkeys(boundaries))[:16]


def _expected_behavior_for_theme(theme: str) -> str:
    if theme == "dependency_resolution":
        return "Imports and dependency resolution either succeed or produce actionable diagnostics."
    if theme == "production_build_reliability":
        return "The production build path remains deterministic and fails with clear diagnostics."
    if theme == "plugin_ecosystem":
        return "Plugin or extension hooks behave predictably across documented entry points."
    if theme == "documentation_and_migration_path":
        return "Users can discover, migrate, and troubleshoot the workflow from public docs."
    if theme == "path_security":
        return "Path handling rejects traversal and keeps operations within the workspace boundary."
    creative_expectations = {
        "programmable_workflows": "Users can compose and reuse workflow steps through a documented, testable contract.",
        "guided_workflow_templates": "Users can start from an adaptable template and inspect or override its configuration.",
        "collaborative_reuse": "Users can share and adapt workflows while retaining source and compatibility metadata.",
        "workflow_automation": "Users can automate repeated steps while retaining explicit review and failure controls.",
    }
    if theme in creative_expectations:
        return creative_expectations[theme]
    return f"The {theme.replace('_', ' ')} workflow improves without regressing existing behavior."


def _candidate_path_hints_for_theme(theme: str) -> tuple[str, ...]:
    mapping = {
        "dependency_resolution": (
            "src/resolver.js",
            "src/resolve.js",
            "dist/resolver.js",
            "dist/server/serverPluginModuleResolve.js",
            "src/gitingest/cloning.py",
        ),
        "production_build_reliability": (
            "src/build.js",
            "dist/build/index.js",
            "dist-src/commands/build.js",
            "package.json",
        ),
        "plugin_ecosystem": (
            "src/plugin.js",
            "src/plugins.js",
            "dist/build/buildPluginResolve.js",
            "dist-src/rollup-plugin-remote-cdn.js",
        ),
        "framework_generalization": (
            "src/index.js",
            "dist-src/index.js",
            "dist/server/index.js",
            "README.md",
        ),
        "architecture_generalization": (
            "src/index.js",
            "dist-src/index.js",
            "src/config.js",
            "package.json",
        ),
        "diagnostics_and_recovery": (
            "src/errors.js",
            "src/cli.js",
            "dist/cli.js",
            "src/gitingest/cli.py",
        ),
        "documentation_and_migration_path": (
            "README.md",
            "docs/migration.md",
            "CHANGELOG.md",
        ),
        "path_security": (
            "src/gitingest/utils/path_utils.py",
            "src/gitingest/ingestion.py",
            "tests/test_path_security.py",
        ),
        "token_count_resilience": (
            "src/gitingest/ingestion.py",
            "src/gitingest/utils/file_utils.py",
            "tests/test_token_count.py",
        ),
        "max_file_size_enforcement": (
            "src/gitingest/ingestion.py",
            "src/gitingest/config.py",
            "tests/test_max_file_size.py",
        ),
        "include_submodules": (
            "src/gitingest/cloning.py",
            "src/gitingest/entrypoint.py",
            "tests/test_submodules.py",
        ),
        "ignore_pattern_reliability": (
            "src/gitingest/utils/ignore_patterns.py",
            "src/gitingest/query_parsing.py",
            "tests/test_ignore_patterns.py",
        ),
        "pattern_filtering_consistency": (
            "src/gitingest/query_parsing.py",
            "src/gitingest/ingestion.py",
            "tests/test_pattern_filtering.py",
        ),
        "http_client_portability": (
            "src/gitingest/utils/git_utils.py",
            "server/query_processor.py",
            "tests/test_http_portability.py",
        ),
        "programmable_workflows": (
            "src/workflow.js",
            "src/pipeline.js",
            "src/config.js",
            "tests/workflow.test.js",
        ),
        "guided_workflow_templates": (
            "src/templates.js",
            "src/scaffold.js",
            "examples/",
            "tests/templates.test.js",
        ),
        "collaborative_reuse": (
            "src/workflow.js",
            "src/config.js",
            "src/serialization.js",
            "tests/workflow-sharing.test.js",
        ),
        "workflow_automation": (
            "src/workflow.js",
            "src/commands.js",
            "src/pipeline.js",
            "tests/workflow-automation.test.js",
        ),
    }
    return mapping.get(theme, ("src/index.js", "README.md", "tests/test_regression.py"))


def _acceptance_tests_for_theme(
    theme: str,
    path_hints: tuple[str, ...],
    behavior_surface: tuple[str, ...] = (),
) -> tuple[str, ...]:
    test_hint = next((path for path in path_hints if _is_test_or_repro_path(path)), "")
    if test_hint:
        return (f"Run behavior regression covering {test_hint}.",)
    surface_terms = {
        term
        for term in (str(surface).strip().lower() for surface in behavior_surface)
        if term
    }
    if "docs" in surface_terms:
        return ("Run documentation or CLI help smoke checks for the changed workflow.",)
    return (
        f"Add or run a behavior regression for {theme.replace('_', ' ')}.",
        "Keep syntax checks as secondary evidence, not the only acceptance signal.",
    )


def _task_type_for_theme(theme: str) -> str:
    if theme in {
        "dependency_resolution",
        "diagnostics_and_recovery",
        "path_security",
        "http_client_portability",
        "max_file_size_enforcement",
        "token_count_resilience",
        "ignore_pattern_reliability",
        "pattern_filtering_consistency",
    }:
        return "bugfix"
    if theme in {"production_build_reliability"}:
        return "perf"
    if theme in {"documentation_and_migration_path"}:
        return "docs"
    if (
        theme
        in {"plugin_ecosystem", "framework_generalization"} | CREATIVE_FEATURE_THEMES
    ):
        return "feature"
    return "usability"


def _risk_level_for_theme(theme: str) -> str:
    if theme in {
        "path_security",
        "dependency_resolution",
        "production_build_reliability",
    }:
        return "high"
    if (
        theme
        in {"plugin_ecosystem", "framework_generalization"} | CREATIVE_FEATURE_THEMES
    ):
        return "medium"
    return "low"


def _requires_architecture_delta(task_specs: tuple[DevelopmentTaskSpecV2, ...]) -> bool:
    architecture_themes = {
        "architecture_generalization",
        "framework_generalization",
        "production_build_reliability",
    }
    return any(
        spec.source_theme in architecture_themes
        or "pipeline" in " ".join(spec.candidate_path_hints).lower()
        or "source" in " ".join(spec.candidate_path_hints).lower()
        for spec in task_specs
    )


def _requires_public_api_delta(task_specs: tuple[DevelopmentTaskSpecV2, ...]) -> bool:
    api_surfaces = {"api", "plugin_api", "config", "cli_help"}
    api_themes = {
        "plugin_ecosystem",
        "framework_generalization",
        "documentation_and_migration_path",
        "programmable_workflows",
        "collaborative_reuse",
        "workflow_automation",
    }
    return any(
        spec.source_theme in api_themes
        or bool(api_surfaces.intersection(spec.behavior_surface))
        for spec in task_specs
    )


def _requires_developer_experience_delta(
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
) -> bool:
    return any(
        spec.source_theme
        in {
            "diagnostics_and_recovery",
            "documentation_and_migration_path",
            "guided_workflow_templates",
        }
        or "error_message" in spec.behavior_surface
        or "cli_help" in spec.behavior_surface
        for spec in task_specs
    )


def _engineering_role_specs(
    *,
    architecture_delta_required: bool,
    public_api_delta_required: bool,
    developer_experience_delta_required: bool,
    security_review_required: bool,
) -> tuple[tuple[str, str, str], ...]:
    roles: list[tuple[str, str, str]] = []
    if architecture_delta_required:
        roles.append(
            (
                "architecture",
                "Define the smallest module or pipeline boundary that turns feedback themes into maintainable structure.",
                "architecture_delta_review",
            )
        )
    if public_api_delta_required:
        roles.append(
            (
                "api_surface",
                "Specify public API, plugin, CLI, and configuration contract changes before implementation.",
                "public_api_compatibility",
            )
        )
    if developer_experience_delta_required:
        roles.append(
            (
                "developer_experience",
                "Cover onboarding, diagnostics, migration, and failure-recovery behavior visible to users.",
                "developer_experience_probe",
            )
        )
    if security_review_required:
        roles.append(
            (
                "security_review",
                "Review path, network, dependency, and workspace-boundary risks before landing the patch.",
                "security_regression_probe",
            )
        )
    roles.extend(
        (
            (
                "implementation",
                "Land source and test changes for implementation-ready packages rather than planning-only artifacts.",
                "behavior_reproducer_or_targeted_test",
            ),
            (
                "integration_lead",
                "Merge role outputs into one coherent release candidate with minimal redundant patches.",
                "integration_consistency",
            ),
            (
                "verifier",
                "Run behavior, API, architecture, security, and regression checks in evidence order.",
                "multi_layer_verification",
            ),
            (
                "evidence_boundary",
                "Downgrade claims to the strongest level supported by passed gates and changed files.",
                "claim_boundary_review",
            ),
        )
    )
    return tuple(roles)


def _build_engineering_work_package(
    *,
    role: str,
    mission: str,
    gate: str,
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    repo_intelligence: RepoIntelligencePack,
    reproducer_plan: ReproducerPlan,
    index: int,
) -> EngineeringWorkPackage:
    relevant_specs = _role_relevant_task_specs(role, task_specs)
    if not relevant_specs:
        relevant_specs = task_specs
    source_themes = tuple(dict.fromkeys(spec.source_theme for spec in relevant_specs))
    candidate_paths = _role_candidate_paths(
        role=role,
        task_specs=relevant_specs,
        repo_intelligence=repo_intelligence,
    )
    package_payload = {
        "role": role,
        "index": index,
        "themes": source_themes,
        "task_ids": tuple(spec.task_id for spec in relevant_specs),
        "paths": candidate_paths,
        "gate": gate,
    }
    return EngineeringWorkPackage(
        package_id=f"engpkg_{_slug_label(role)}_{stable_hash(package_payload)[:10]}",
        owner_role=role,
        mission=mission[:500],
        source_themes=source_themes,
        task_ids=tuple(spec.task_id for spec in relevant_specs),
        candidate_paths=candidate_paths,
        required_outputs=_role_required_outputs(
            role=role,
            gate=gate,
            reproducer_plan=reproducer_plan,
        ),
        gate=gate,
        risk_level=_role_risk_level(role, relevant_specs),
    )


def _role_relevant_task_specs(
    role: str,
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
) -> tuple[DevelopmentTaskSpecV2, ...]:
    if role == "architecture":
        themes = {
            "architecture_generalization",
            "framework_generalization",
            "production_build_reliability",
            "dependency_resolution",
        }
    elif role == "api_surface":
        themes = {
            "plugin_ecosystem",
            "framework_generalization",
            "documentation_and_migration_path",
            "programmable_workflows",
            "collaborative_reuse",
            "workflow_automation",
        }
    elif role == "developer_experience":
        themes = {
            "diagnostics_and_recovery",
            "documentation_and_migration_path",
            "guided_workflow_templates",
        }
    elif role == "security_review":
        themes = {
            "path_security",
            "dependency_resolution",
            "http_client_portability",
            "include_submodules",
            "max_file_size_enforcement",
        }
    else:
        return task_specs
    return tuple(spec for spec in task_specs if spec.source_theme in themes)


def _role_candidate_paths(
    *,
    role: str,
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    repo_intelligence: RepoIntelligencePack,
) -> tuple[str, ...]:
    paths: list[str] = []
    for spec in task_specs:
        paths.extend(spec.candidate_path_hints[:8])
        paths.extend(
            repo_intelligence.candidate_files_by_theme.get(spec.source_theme, ())[:8]
        )
    if role == "architecture":
        paths.extend(repo_intelligence.entrypoints[:8])
        paths.extend(repo_intelligence.config_files[:8])
        paths.extend(path for path in repo_intelligence.source_roots[:8])
    elif role == "api_surface":
        paths.extend(
            path
            for symbol, symbol_paths in repo_intelligence.symbol_index.items()
            if any(
                token in symbol.lower()
                for token in ("plugin", "api", "config", "command")
            )
            for path in symbol_paths[:2]
        )
        paths.extend(repo_intelligence.docs_files[:6])
    elif role == "developer_experience":
        paths.extend(repo_intelligence.docs_files[:8])
        paths.extend(
            path
            for path in repo_intelligence.file_tree
            if any(
                token in path.lower()
                for token in ("cli", "error", "diagnostic", "overlay")
            )
        )
    elif role == "security_review":
        paths.extend(
            path
            for path in repo_intelligence.file_tree
            if any(
                token in path.lower()
                for token in ("path", "http", "resolve", "security")
            )
        )
    elif role == "verifier":
        paths.extend(repo_intelligence.test_files[:12])
    elif role == "evidence_boundary":
        paths.extend(repo_intelligence.docs_files[:6])
    filtered = [
        path for path in paths if path and path in set(repo_intelligence.file_tree)
    ]
    if not filtered:
        filtered = [path for path in paths if path]
    return tuple(dict.fromkeys(filtered))[:20]


def _role_required_outputs(
    *,
    role: str,
    gate: str,
    reproducer_plan: ReproducerPlan,
) -> tuple[str, ...]:
    defaults = {
        "architecture": (
            "architecture_delta",
            "module_boundary",
            "migration_order",
        ),
        "api_surface": (
            "public_contract_delta",
            "compatibility_notes",
            "typed_api_or_cli_surface",
        ),
        "developer_experience": (
            "diagnostic_or_migration_path",
            "user_visible_failure_recovery",
        ),
        "security_review": (
            "threat_boundary",
            "negative_or_guardrail_check",
        ),
        "implementation": (
            "source_patch",
            "targeted_regression",
        ),
        "integration_lead": (
            "coherent_patch_bundle",
            "conflict_resolution",
        ),
        "verifier": (
            "behavior_verification_result",
            "regression_verification_result",
        ),
        "evidence_boundary": (
            "allowed_claim_level",
            "unsupported_items",
        ),
    }
    outputs = list(defaults.get(role, ("work_package_result",)))
    if role in {"implementation", "verifier"} and reproducer_plan.executable_commands:
        outputs.append("reproducer_command_pass")
    outputs.append(gate)
    return tuple(dict.fromkeys(outputs))


def _role_risk_level(
    role: str,
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
) -> str:
    if role in {"architecture", "api_surface", "security_review"}:
        return "high"
    if any(spec.risk_level == "high" for spec in task_specs):
        return "high"
    if role in {"implementation", "integration_lead", "verifier"}:
        return "medium"
    return "low"


def _behavior_surface_for_theme(theme: str) -> tuple[str, ...]:
    mapping = {
        "dependency_resolution": ("import", "api", "cli"),
        "diagnostics_and_recovery": ("cli", "error_message"),
        "documentation_and_migration_path": ("docs", "cli_help"),
        "architecture_generalization": ("api", "config", "entrypoint"),
        "framework_generalization": ("api", "config", "entrypoint"),
        "plugin_ecosystem": ("plugin_api", "config"),
        "programmable_workflows": ("api", "config", "workflow"),
        "guided_workflow_templates": ("api", "config", "docs"),
        "collaborative_reuse": ("api", "config", "serialization"),
        "workflow_automation": ("api", "config", "command"),
        "production_build_reliability": ("build", "cli"),
        "path_security": ("filesystem", "api"),
        "http_client_portability": ("network", "api"),
    }
    return mapping.get(theme, ("api", "cli"))


def _symbols_for_theme(theme: str) -> tuple[str, ...]:
    mapping = {
        "dependency_resolution": ("resolve", "resolver", "resolvePackage", "resolveId"),
        "diagnostics_and_recovery": ("diagnostic", "error", "formatError"),
        "production_build_reliability": ("build", "optimize", "bundle"),
        "plugin_ecosystem": ("plugin", "hook", "transform"),
        "programmable_workflows": ("workflow", "pipeline", "compose"),
        "guided_workflow_templates": ("template", "preset", "scaffold"),
        "collaborative_reuse": ("serialize", "export", "import"),
        "workflow_automation": ("automate", "task", "command"),
        "path_security": ("safe", "resolve", "relative_to"),
    }
    return mapping.get(theme, ())


def _repo_entrypoints_for_spec(
    *,
    theme: str,
    path_hints: tuple[str, ...],
    repo_intelligence: RepoIntelligencePack | None,
) -> tuple[str, ...]:
    entries: list[str] = []
    if repo_intelligence is not None:
        entries.extend(repo_intelligence.entrypoints[:8])
        entries.extend(repo_intelligence.candidate_files_by_theme.get(theme, ())[:6])
    entries.extend(path_hints[:6])
    return tuple(dict.fromkeys(entries))[:12]


def _repo_augmented_path_hints(
    *,
    theme: str,
    path_hints: tuple[str, ...],
    repo_intelligence: RepoIntelligencePack | None,
    relevant_symbols: tuple[str, ...] = (),
    behavior_surface: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if repo_intelligence is None:
        return path_hints
    file_tree = set(repo_intelligence.file_tree)
    hints: list[str] = []
    hints.extend(path for path in path_hints if path in file_tree)
    hints.extend(repo_intelligence.candidate_files_by_theme.get(theme, ())[:10])
    surface_terms = {
        term
        for term in (str(surface).strip().lower() for surface in behavior_surface)
        if term
    }
    if "docs" in surface_terms:
        hints.extend(repo_intelligence.docs_files[:6])
    if "entrypoint" in surface_terms:
        hints.extend(repo_intelligence.entrypoints[:6])
    if "config" in surface_terms:
        hints.extend(repo_intelligence.config_files[:8])
    symbol_terms = tuple(
        dict.fromkeys(
            term
            for term in (str(symbol).strip().lower() for symbol in relevant_symbols)
            if len(term) >= 4
        )
    )
    if symbol_terms:
        hints.extend(
            path
            for symbol, paths in repo_intelligence.symbol_index.items()
            if any(term in symbol.lower() for term in symbol_terms)
            for path in paths[:2]
        )
    if not hints:
        hints.extend(path_hints)
    return tuple(dict.fromkeys(hints))[:16]


def _repo_backed_acceptance_oracles_for_theme(
    *,
    theme: str,
    index: int,
    path_hints: tuple[str, ...],
    repo_intelligence: RepoIntelligencePack | None,
    behavior_surface: tuple[str, ...] = (),
) -> tuple[AcceptanceOracle, ...]:
    if repo_intelligence is None:
        return ()
    commands: list[tuple[str, str]] = []
    if repo_intelligence.test_commands:
        commands.append(
            (
                repo_intelligence.test_commands[0],
                f"Run the project test suite after implementing {theme.replace('_', ' ')}.",
            )
        )
    test_file = _first_existing_hint(path_hints, repo_intelligence.test_files)
    if test_file:
        commands.append(
            (
                _test_command_for_file(test_file, repo_intelligence),
                f"Run the targeted regression file {test_file}.",
            )
        )
    else:
        surface_terms = {
            term
            for term in (str(surface).strip().lower() for surface in behavior_surface)
            if term
        }
        docs_file = (
            _first_existing_hint(path_hints, repo_intelligence.docs_files)
            if "docs" in surface_terms
            else None
        )
        if docs_file:
            commands.append(
                (
                    _node_file_contains_command(docs_file),
                    f"Smoke-check that {docs_file} remains readable after the {theme.replace('_', ' ')} update.",
                )
            )
        else:
            source_file = _first_existing_hint(path_hints, repo_intelligence.file_tree)
            if source_file:
                syntax_command = _syntax_command_for_file(source_file)
                if syntax_command:
                    commands.append(
                        (
                            syntax_command,
                            f"Run a syntax guard for the localized {theme.replace('_', ' ')} source file.",
                        )
                    )
    oracles: list[AcceptanceOracle] = []
    for oracle_index, (command, description) in enumerate(
        _dedupe_command_pairs(commands)[:3]
    ):
        payload = {
            "theme": theme,
            "index": index,
            "oracle_index": oracle_index,
            "command": command,
            "source": "repo_intelligence_playbook",
        }
        oracles.append(
            AcceptanceOracle(
                oracle_id=f"oracle_{stable_hash(payload)[:16]}",
                kind=_oracle_kind_for_command(command),
                command=command,
                description=description[:500],
                source="repo_intelligence_playbook",
                witness_role="preservation",
                expected_on_base="pass",
            )
        )
    return tuple(oracles)


def _dedupe_command_pairs(
    commands: list[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for command, description in commands:
        if not command or command in seen:
            continue
        seen.add(command)
        deduped.append((command, description))
    return tuple(deduped)


def _first_existing_hint(
    hints: tuple[str, ...],
    candidates: tuple[str, ...],
) -> str | None:
    candidate_set = set(candidates)
    for hint in hints:
        if hint in candidate_set:
            return hint
    for candidate in candidates:
        if any(_path_matches_hint(candidate, hint) for hint in hints):
            return candidate
    return None


def _test_command_for_file(
    test_file: str,
    repo_intelligence: RepoIntelligencePack,
) -> str:
    quoted = shlex.quote(test_file)
    if repo_intelligence.package_manager == "npm":
        return f"npm test -- {quoted}"
    if repo_intelligence.package_manager == "yarn":
        return f"yarn test {quoted}"
    if repo_intelligence.package_manager == "pnpm":
        return f"pnpm test -- {quoted}"
    if test_file.endswith(".py"):
        return f"pytest {quoted}"
    if test_file.endswith((".js", ".mjs", ".cjs")):
        return f"node --test {quoted}"
    return (
        repo_intelligence.test_commands[0]
        if repo_intelligence.test_commands
        else f"node --test {quoted}"
    )


def _syntax_command_for_file(path: str) -> str | None:
    quoted = shlex.quote(path)
    if path.endswith((".js", ".mjs", ".cjs")):
        return f"node --check {quoted}"
    if path.endswith(".py"):
        return f"python -m py_compile {quoted}"
    if path.endswith(".json"):
        return f"python -m json.tool {quoted}"
    return None


def _node_file_contains_command(path: str) -> str:
    escaped = json.dumps(path)
    return "node -e " + shlex.quote(
        "const fs=require('fs');"
        f"const p={escaped};"
        "const s=fs.readFileSync(p,'utf8');"
        "if(!s.trim()) throw new Error('empty documentation file');"
        "console.log('documentation smoke ok');"
    )


def _acceptance_oracles_for_spec(
    spec: Mapping[str, Any],
    index: int,
) -> tuple[AcceptanceOracle, ...]:
    witnesses = _normalized_contract_witnesses(spec.get("contract_witnesses"))
    oracles: list[AcceptanceOracle] = []
    seen_commands: set[str] = set()
    for witness_index, witness in enumerate(witnesses):
        command = _extract_executable_command(str(witness["command"]))
        oracle_payload = {
            "spec": spec.get("intent_id"),
            "index": index,
            "witness_index": witness_index,
            **witness,
            "command": command,
        }
        if command:
            seen_commands.add(command)
        oracles.append(
            AcceptanceOracle(
                oracle_id=f"oracle_{stable_hash(oracle_payload)[:16]}",
                kind=_oracle_kind_for_command(command)
                if command
                else "behavior_description",
                command=command,
                description=str(witness["description"])[:500],
                source="contract_witness",
                dimension_ids=tuple(witness["dimensions"]),
                witness_role=str(witness["role"]),
                expected_on_base=(
                    "fail" if witness["expected_on_base"] == "failed" else "pass"
                ),
                expected_on_patch="pass",
                required=bool(witness["required"]),
                evidence_hash=str(witness["evidence_hash"]),
            )
        )
    raw_tests = _string_tuple(
        spec.get("acceptance_tests"),
        fallback=(),
        max_items=8,
        max_chars=MAX_AGENT_VERIFICATION_COMMAND_CHARS,
    )
    for oracle_index, description in enumerate(raw_tests):
        command = _extract_executable_command(description)
        if command and command in seen_commands:
            continue
        oracle_payload = {
            "spec": spec.get("intent_id"),
            "index": index,
            "oracle_index": oracle_index,
            "description": description,
            "command": command,
        }
        oracles.append(
            AcceptanceOracle(
                oracle_id=f"oracle_{stable_hash(oracle_payload)[:16]}",
                kind=_oracle_kind_for_command(command)
                if command
                else "behavior_description",
                command=command,
                description=description[:500],
                source="acceptance_tests",
            )
        )
    return tuple(oracles)


def _normalized_contract_witnesses(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_commands: set[str] = set()
    for index, raw in enumerate(value[:32]):
        if not isinstance(raw, Mapping):
            continue
        witness_id = str(raw.get("witness_id") or f"witness_{index}")[:80]
        dimensions = _string_tuple(
            raw.get("dimensions"),
            fallback=(),
            max_items=16,
        )
        command = str(raw.get("command") or "").strip()
        description = str(raw.get("description") or "").strip()[:500]
        role = str(raw.get("role") or "repair")
        expected_on_base = str(raw.get("expected_on_base") or "failed")
        expected_on_patch = str(raw.get("expected_on_patch") or "passed")
        if (
            not dimensions
            or not command
            or len(command) > MAX_AGENT_VERIFICATION_COMMAND_CHARS
            or not description
            or role not in {"repair", "preservation"}
            or expected_on_base not in {"passed", "failed"}
            or expected_on_patch != "passed"
            or witness_id in seen_ids
            or command in seen_commands
        ):
            continue
        payload = {
            "witness_id": witness_id,
            "dimensions": dimensions,
            "command": command,
            "role": role,
            "expected_on_base": expected_on_base,
            "expected_on_patch": expected_on_patch,
            "description": description,
            "required": bool(raw.get("required", True)),
        }
        source_evidence_hash = str(raw.get("evidence_hash") or "")
        payload["evidence_hash"] = (
            source_evidence_hash
            if re.fullmatch(r"[0-9a-f]{64}", source_evidence_hash)
            else stable_hash(payload)
        )
        rows.append(payload)
        seen_ids.add(witness_id)
        seen_commands.add(command)
    return tuple(rows)


def _extract_executable_command(description: str) -> str | None:
    text = description.strip().rstrip(".")
    backtick = re.search(r"`([^`]+)`", text)
    if backtick:
        text = backtick.group(1).strip()
    if text.lower().startswith("run "):
        text = text[4:].strip()
    try:
        tokens = shlex.split(text)
    except ValueError:
        return None
    if not tokens:
        return None
    executable = Path(tokens[0]).name.lower()
    allowed = frozenset(VERIFICATION_EXECUTABLES)
    if (executable in allowed or executable.startswith("python")) and len(
        text
    ) <= MAX_AGENT_VERIFICATION_COMMAND_CHARS:
        return text
    return None


def _oracle_kind_for_command(command: str | None) -> str:
    if not command:
        return "manual_smoke"
    lower = command.lower()
    if "pytest" in lower:
        return "pytest"
    if "node" in lower or "npm" in lower or "yarn" in lower or "pnpm" in lower:
        return "script"
    if " -c " in f" {lower} " or lower.startswith("python"):
        return "import_assertion"
    return "cli"


def _non_goals_for_spec(spec: Mapping[str, Any]) -> tuple[str, ...]:
    configured = _string_tuple(spec.get("non_goals"), fallback=(), max_items=8)
    defaults: tuple[str, ...] = (
        "Do not use private user state or future release metadata.",
        "Do not replace product behavior with planning-only or documentation-only changes when source behavior is available.",
        "Do not broaden the change beyond the public evidence and acceptance oracle.",
    )
    if str(spec.get("theme") or "") in CREATIVE_FEATURE_THEMES:
        defaults = (
            *defaults,
            "Do not present a proposed capability as an observed defect or historical fact.",
        )
    return tuple(dict.fromkeys((*configured, *defaults)))[:8]


def _observed_behavior_for_spec(spec: Mapping[str, Any]) -> str:
    observed = str(spec.get("observed_behavior") or "")[:500]
    if observed:
        return observed
    evidence_refs = _string_tuple(spec.get("evidence_refs"), fallback=(), max_items=3)
    if evidence_refs:
        return f"Public traces report failures: {', '.join(evidence_refs)}."
    return str(spec.get("user_pain") or "Public users report unresolved friction.")[
        :500
    ]


def _missing_information_for_task_spec(
    *,
    user_pain: str,
    observed_behavior: str,
    expected_behavior: str,
    reproduction_recipe: tuple[str, ...],
    acceptance_oracles: tuple[AcceptanceOracle, ...],
    candidate_path_hints: tuple[str, ...],
    public_evidence_refs: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    if not user_pain:
        missing.append("user_pain")
    if not observed_behavior:
        missing.append("observed_behavior")
    if not expected_behavior:
        missing.append("expected_behavior")
    if not reproduction_recipe:
        missing.append("reproduction_recipe")
    if not any(oracle.command for oracle in acceptance_oracles):
        missing.append("executable_acceptance_oracle")
    if not candidate_path_hints:
        missing.append("candidate_path_hints")
    if not public_evidence_refs:
        missing.append("public_evidence_refs")
    return tuple(missing)


def _classify_task_spec_maturity(
    *,
    missing_information: tuple[str, ...],
    reproduction_recipe: tuple[str, ...],
    acceptance_oracles: tuple[AcceptanceOracle, ...],
    candidate_path_hints: tuple[str, ...],
) -> str:
    has_command_oracle = any(oracle.command for oracle in acceptance_oracles)
    if (
        has_command_oracle
        and candidate_path_hints
        and not {
            "user_pain",
            "observed_behavior",
            "expected_behavior",
            "reproduction_recipe",
        }.intersection(missing_information)
    ):
        return "implementation_ready"
    if has_command_oracle:
        return "testable"
    if reproduction_recipe:
        return "reproducible"
    if len(missing_information) <= 4:
        return "issue_like"
    return "theme_only"


def _maturity_rank(value: str) -> int:
    order = {
        "theme_only": 0,
        "issue_like": 1,
        "reproducible": 2,
        "testable": 3,
        "implementation_ready": 4,
    }
    return order.get(value, 0)


def _ambiguity_level(missing_information: tuple[str, ...]) -> str:
    if len(missing_information) <= 1:
        return "low"
    if len(missing_information) <= 3:
        return "medium"
    return "high"


def _workspace_text_files(
    root: Path,
    *,
    max_files: int,
    max_file_bytes: int = 180_000,
) -> tuple[tuple[str, str], ...]:
    files: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if len(files) >= max_files:
            break
        if _is_ignored_workspace_path(path, root):
            continue
        try:
            raw = read_regular_file_bytes(
                root,
                path,
                max_bytes=max_file_bytes,
            )
        except (OSError, UnsafeRegularFileError):
            continue
        if b"\x00" in raw[:4096]:
            continue
        relative = path.relative_to(root).as_posix()
        files.append((relative, raw.decode("utf-8", errors="replace")))
    return tuple(files)


def _detect_package_manager(root: Path) -> str | None:
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (root / "yarn.lock").exists():
        return "yarn"
    if (root / "package.json").exists():
        return "npm"
    if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
        return "python"
    if _find_dist_metadata_for_intelligence(root) is not None:
        return "python"
    return None


def _detect_project_commands(
    root: Path,
    file_tree: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    test_commands: list[str] = []
    lint_commands: list[str] = []
    package_path = root / "package.json"
    if package_path.exists():
        try:
            package = json.loads(
                read_regular_file_text(root, package_path, max_bytes=2_000_000)
            )
        except (OSError, UnsafeRegularFileError, json.JSONDecodeError):
            package = {}
        scripts = package.get("scripts", {}) if isinstance(package, Mapping) else {}
        if isinstance(scripts, Mapping):
            if scripts.get("test"):
                test_commands.append("npm test")
            for name, command in scripts.items():
                if name != "test" and "test" in str(name).lower() and command:
                    test_commands.append(f"npm run {name}")
                if "lint" in str(name).lower() and command:
                    lint_commands.append(f"npm run {name}")
    if (root / "pyproject.toml").exists() or any(
        path.startswith("tests/") for path in file_tree
    ):
        if any(path.endswith(".py") for path in file_tree):
            test_commands.append("pytest")
    if not test_commands and any(
        path.endswith((".test.js", ".test.mjs", ".spec.js", ".spec.mjs"))
        for path in file_tree
    ):
        test_commands.append("node --test")
    if (root / "pyproject.toml").exists() and any(
        path.endswith(".py") for path in file_tree
    ):
        lint_commands.append("ruff check .")
    return _dedupe(test_commands), _dedupe(lint_commands)


def _detect_entrypoints(root: Path) -> tuple[str, ...]:
    entries: list[str] = []
    package_path = root / "package.json"
    if package_path.exists():
        try:
            package = json.loads(
                read_regular_file_text(root, package_path, max_bytes=2_000_000)
            )
        except (OSError, UnsafeRegularFileError, json.JSONDecodeError):
            package = {}
        for key in ("main", "module", "browser"):
            value = package.get(key) if isinstance(package, Mapping) else None
            if isinstance(value, str):
                entries.append(value)
        bin_value = package.get("bin") if isinstance(package, Mapping) else None
        if isinstance(bin_value, str):
            entries.append(bin_value)
        elif isinstance(bin_value, Mapping):
            entries.extend(str(value) for value in bin_value.values())
    pyproject_path = root / "pyproject.toml"
    if pyproject_path.exists():
        try:
            pyproject = tomllib.loads(
                read_regular_file_text(
                    root,
                    pyproject_path,
                    max_bytes=2_000_000,
                )
            )
        except (OSError, UnsafeRegularFileError, tomllib.TOMLDecodeError):
            pyproject = {}
        project = pyproject.get("project", {})
        if isinstance(project, Mapping):
            for table_name in ("scripts", "gui-scripts"):
                scripts = project.get(table_name, {})
                if not isinstance(scripts, Mapping):
                    continue
                for target in scripts.values():
                    entries.extend(_python_entrypoint_paths(root, str(target)))
    for path in ("src/index.js", "src/index.ts", "src/main.py", "src/__init__.py"):
        if (root / path).exists():
            entries.append(path)
    return tuple(dict.fromkeys(entries))[:16]


def _python_entrypoint_paths(root: Path, target: str) -> tuple[str, ...]:
    module = target.partition(":")[0].strip()
    if not module or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module):
        return ()
    module_path = Path(*module.split("."))
    candidates = (
        Path("src") / module_path.with_suffix(".py"),
        Path("src") / module_path / "__init__.py",
        module_path.with_suffix(".py"),
        module_path / "__init__.py",
    )
    return tuple(
        candidate.as_posix() for candidate in candidates if (root / candidate).is_file()
    )


def _find_dist_metadata_for_intelligence(root: Path) -> Path | None:
    metadata = sorted(root.glob("*.dist-info/METADATA"))
    return metadata[0] if metadata else None


def _is_config_path(path: str) -> bool:
    name = Path(path).name.lower()
    return (
        name in {"package.json", "pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini"}
        or "config" in name
        or name.endswith((".yaml", ".yml", ".toml"))
    )


def _is_docs_path(path: str) -> bool:
    normalized = Path(path).as_posix().lower()
    return normalized.startswith("docs/") or Path(normalized).name in {
        "readme.md",
        "changelog.md",
        "contributing.md",
    }


def _build_symbol_index(
    text_files: tuple[tuple[str, str], ...],
    *,
    max_symbols: int = 500,
) -> dict[str, tuple[str, ...]]:
    patterns = (
        re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", re.M
        ),
        re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", re.M),
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", re.M
        ),
        re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.M),
        re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[:(]", re.M),
    )
    symbols: dict[str, list[str]] = {}
    for path, text in text_files:
        if len(symbols) >= max_symbols:
            break
        if Path(path).suffix.lower() not in {
            ".py",
            ".js",
            ".mjs",
            ".cjs",
            ".ts",
            ".tsx",
        }:
            continue
        for pattern in patterns:
            for match in pattern.finditer(text[:80_000]):
                symbol = match.group(1)
                symbols.setdefault(symbol, []).append(path)
                if len(symbols) >= max_symbols:
                    break
            if len(symbols) >= max_symbols:
                break
    return {
        symbol: tuple(dict.fromkeys(paths)) for symbol, paths in sorted(symbols.items())
    }


def _build_test_map(file_tree: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    source_files = tuple(
        path
        for path in file_tree
        if _is_source_or_config_path(path) and not _is_test_or_repro_path(path)
    )
    test_files = tuple(path for path in file_tree if _is_test_or_repro_path(path))
    mapping: dict[str, tuple[str, ...]] = {}
    for source in source_files:
        stem = Path(source).stem.replace(".test", "").replace(".spec", "")
        related = tuple(
            test
            for test in test_files
            if stem and stem.lower() in Path(test).stem.lower()
        )
        if related:
            mapping[source] = related[:8]
    return mapping


def _style_notes_for_repo(
    package_manager: str | None,
    test_commands: tuple[str, ...],
    lint_commands: tuple[str, ...],
) -> tuple[str, ...]:
    notes: list[str] = []
    if package_manager:
        notes.append(f"package_manager:{package_manager}")
    if test_commands:
        notes.append("project_declares_tests")
    if lint_commands:
        notes.append("project_declares_lint")
    return tuple(notes)


def _command_referenced_paths_exist(command: str, workspace_root: Path) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    root = workspace_root.resolve()
    for token in tokens[1:]:
        normalized = token.strip("'\"")
        if not normalized or normalized.startswith("-"):
            continue
        if any(char in normalized for char in (";", "(", ")", "{", "}")):
            continue
        if not (
            "/" in normalized
            or normalized.endswith(
                (".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".json")
            )
        ):
            continue
        candidate = (root / normalized).resolve()
        if not str(candidate).startswith(str(root)) or not candidate.exists():
            return False
    return True


def _reproducer_feedback(
    reproducer_plan: ReproducerPlan,
) -> tuple[WorkspaceVerificationResult, ...]:
    return tuple(
        WorkspaceVerificationResult(
            command=artifact.command,
            status="failed",
            exit_code=1,
            elapsed_sec=0.0,
            stdout_tail="base_reproducer_confirmed_failure",
            stderr_tail=artifact.observed_on_base,
            blocked_reason="base_reproducer_confirmed_failure",
        )
        for artifact in reproducer_plan.artifacts
        if artifact.reproducibility_status == "confirmed"
    )


def _string_tuple(
    value: Any,
    *,
    fallback: tuple[str, ...],
    max_items: int,
    max_chars: int = 300,
) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items = (value,)
    elif isinstance(value, (list, tuple)):
        raw_items = tuple(value) or fallback
    else:
        raw_items = fallback
    return tuple(
        dict.fromkeys(str(item)[:max_chars] for item in raw_items if str(item))
    )[:max_items]


def _slug_label(value: str) -> str:
    slug = "_".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in value).split()
    )
    return slug or "intent"


def _remaining_requested_themes(
    *,
    requested_themes: tuple[str, ...],
    covered_themes: tuple[str, ...],
) -> tuple[str, ...]:
    covered = set(covered_themes)
    return tuple(theme for theme in requested_themes if theme not in covered)


def _active_theme_window(
    themes: tuple[str, ...],
    *,
    max_active_themes_per_iteration: int,
) -> tuple[str, ...]:
    if max_active_themes_per_iteration <= 0:
        return themes
    return themes[:max_active_themes_per_iteration]


def _task_specs_for_themes(
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    themes: tuple[str, ...],
) -> tuple[DevelopmentTaskSpecV2, ...]:
    theme_set = set(themes)
    selected = tuple(spec for spec in task_specs if spec.source_theme in theme_set)
    return selected or task_specs


def _reproducer_plan_for_task_specs(
    reproducer_plan: ReproducerPlan,
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
) -> ReproducerPlan:
    task_ids = {spec.task_id for spec in task_specs}
    artifacts = tuple(
        artifact
        for artifact in reproducer_plan.artifacts
        if artifact.task_id in task_ids
    )
    missing_repro_tasks = tuple(
        task_id
        for task_id in reproducer_plan.missing_repro_tasks
        if task_id in task_ids
    )
    required_commands = set(reproducer_plan.executable_commands)
    payload = {
        "source_plan_id": reproducer_plan.plan_id,
        "task_ids": tuple(sorted(task_ids)),
        "artifacts": artifacts,
        "missing_repro_tasks": missing_repro_tasks,
    }
    return ReproducerPlan(
        plan_id=f"reproducer_plan_{stable_hash(payload)[:24]}",
        artifacts=artifacts,
        executable_commands=tuple(
            artifact.command
            for artifact in artifacts
            if artifact.command in required_commands
        ),
        confirmed_count=sum(
            1
            for artifact in artifacts
            if artifact.reproducibility_status == "confirmed"
        ),
        blocked_count=sum(
            1 for artifact in artifacts if artifact.reproducibility_status == "invalid"
        ),
        missing_repro_tasks=missing_repro_tasks,
    )


def _proposal_requested_theme_coverage(
    *,
    proposal: CodingAgentPatchProposal,
    requested_themes: tuple[str, ...],
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    patch_results: tuple[WorkspacePatchResult, ...],
) -> tuple[str, ...]:
    proposed = set(proposal.themes)
    changed_paths = _changed_patch_paths(patch_results)
    applied_patches = _effective_proposal_patches(
        proposal=proposal,
        patch_results=patch_results,
    )
    covered: list[str] = []
    for theme in requested_themes:
        if theme not in proposed:
            continue
        theme_specs = tuple(spec for spec in task_specs if spec.source_theme == theme)
        if (
            len(requested_themes) == 1
            and changed_paths
            and theme_specs
            and not any(spec.source_explicit_intent_spec for spec in theme_specs)
        ):
            covered.append(theme)
            continue
        if any(
            _changed_path_matches_theme(path, theme, theme_specs)
            for path in changed_paths
        ):
            covered.append(theme)
            continue
        if any(
            _patch_content_matches_theme(patch, theme, theme_specs)
            for patch in applied_patches
        ):
            covered.append(theme)
    return tuple(covered)


def _effective_proposal_patches(
    *,
    proposal: CodingAgentPatchProposal,
    patch_results: tuple[WorkspacePatchResult, ...],
) -> tuple[WorkspaceFilePatch, ...]:
    effective_patch_ids = {
        result.patch_id
        for result in patch_results
        if result.status == "applied" and result.before_hash != result.after_hash
    }
    return tuple(
        patch for patch in proposal.patches if patch.patch_id in effective_patch_ids
    )


def _changed_path_matches_theme(
    path: str,
    theme: str,
    specs: tuple[DevelopmentTaskSpecV2, ...],
) -> bool:
    for spec in specs:
        if any(_path_matches_hint(path, hint) for hint in spec.candidate_path_hints):
            return True
        if any(symbol.lower() in path.lower() for symbol in spec.relevant_symbols_hint):
            return True
    normalized = path.lower().replace("-", "_")
    for keyword in THEME_KEYWORDS.get(theme, (theme.replace("_", " "),)):
        token = keyword.lower().replace("-", "_")
        if token and token in normalized:
            return True
    return any(
        part in normalized for part in theme.lower().split("_") if len(part) >= 5
    )


def _patch_content_matches_theme(
    patch: WorkspaceFilePatch,
    theme: str,
    specs: tuple[DevelopmentTaskSpecV2, ...],
) -> bool:
    haystack = _normalize_theme_evidence_text(
        "\n".join((patch.content, patch.old, patch.new, patch.rationale))
    )
    if not haystack:
        return False
    exact_theme = _normalize_theme_evidence_text(theme)
    if exact_theme and exact_theme in haystack:
        return True
    terms = _theme_evidence_terms(theme, specs)
    hits = {term for term in terms if term and term in haystack}
    return len(hits) >= 2


def _theme_evidence_terms(
    theme: str,
    specs: tuple[DevelopmentTaskSpecV2, ...],
) -> tuple[str, ...]:
    raw_terms: list[str] = list(THEME_KEYWORDS.get(theme, ()))
    raw_terms.extend(part for part in theme.split("_") if len(part) >= 5)
    for spec in specs:
        raw_terms.extend(spec.relevant_symbols_hint)
        raw_terms.extend(Path(hint).stem for hint in spec.candidate_path_hints)
    terms: list[str] = []
    for raw_term in raw_terms:
        term = _normalize_theme_evidence_text(raw_term)
        if len(term) >= 4 and term not in terms:
            terms.append(term)
    return tuple(terms)


def _normalize_theme_evidence_text(value: str) -> str:
    return "_".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in value).split()
    )


def _source_report_with_workspace_progress(
    *,
    source_report: Mapping[str, Any],
    requested_themes: tuple[str, ...],
    covered_themes: tuple[str, ...],
    active_themes: tuple[str, ...],
    changed_paths: tuple[str, ...],
) -> Mapping[str, Any]:
    if isinstance(source_report, dict):
        report = dict(source_report)
    else:
        report = {key: source_report[key] for key in source_report}
    report["proposed_themes"] = active_themes
    report["workspace_progress"] = {
        "requested_themes": requested_themes,
        "covered_requested_themes": covered_themes,
        "remaining_requested_themes": _remaining_requested_themes(
            requested_themes=requested_themes,
            covered_themes=covered_themes,
        ),
        "active_themes_for_this_wave": active_themes,
        "changed_paths_so_far": changed_paths,
    }
    return report


def _source_report_with_code_landing_context(
    *,
    source_report: Mapping[str, Any],
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    spec_maturity_gate: Mapping[str, Any],
    repo_intelligence: RepoIntelligencePack,
    reproducer_plan: ReproducerPlan,
) -> Mapping[str, Any]:
    if isinstance(source_report, dict):
        report = dict(source_report)
    else:
        report = {key: source_report[key] for key in source_report}
    report["development_task_specs_v2"] = tuple(
        canonicalize(spec) for spec in task_specs
    )
    report["spec_maturity_gate"] = canonicalize(spec_maturity_gate)
    report["repo_intelligence_pack"] = _repo_intelligence_agent_summary(
        repo_intelligence
    )
    report["reproducer_plan"] = _reproducer_plan_agent_summary(reproducer_plan)
    return report


def _source_report_with_engineering_organization(
    *,
    source_report: Mapping[str, Any],
    engineering_org_plan: EngineeringOrganizationPlan,
) -> Mapping[str, Any]:
    if isinstance(source_report, dict):
        report = dict(source_report)
    else:
        report = {key: source_report[key] for key in source_report}
    report["engineering_organization_plan"] = _engineering_org_plan_agent_summary(
        engineering_org_plan
    )
    return report


def _source_report_with_code_max_runtime(
    *,
    source_report: Mapping[str, Any],
    code_max_runtime_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(source_report, dict):
        report = dict(source_report)
    else:
        report = {key: source_report[key] for key in source_report}
    report["code_max_runtime"] = canonicalize(code_max_runtime_summary)
    return report


def _repo_intelligence_agent_summary(
    repo_intelligence: RepoIntelligencePack,
) -> dict[str, Any]:
    return {
        "repo_hash": repo_intelligence.repo_hash,
        "file_count": len(repo_intelligence.file_tree),
        "file_tree": repo_intelligence.file_tree[:120],
        "package_manager": repo_intelligence.package_manager,
        "test_commands": repo_intelligence.test_commands,
        "lint_commands": repo_intelligence.lint_commands,
        "entrypoints": repo_intelligence.entrypoints,
        "source_roots": repo_intelligence.source_roots,
        "test_files": repo_intelligence.test_files[:80],
        "config_files": repo_intelligence.config_files[:80],
        "docs_files": repo_intelligence.docs_files[:80],
        "symbol_index": {
            symbol: paths
            for symbol, paths in tuple(repo_intelligence.symbol_index.items())[:120]
        },
        "test_map": {
            source: tests
            for source, tests in tuple(repo_intelligence.test_map.items())[:80]
        },
        "candidate_files_by_theme": repo_intelligence.candidate_files_by_theme,
        "style_notes": repo_intelligence.style_notes,
    }


def _reproducer_plan_agent_summary(reproducer_plan: ReproducerPlan) -> dict[str, Any]:
    return {
        "plan_id": reproducer_plan.plan_id,
        "artifact_count": len(reproducer_plan.artifacts),
        "confirmed_count": reproducer_plan.confirmed_count,
        "blocked_count": reproducer_plan.blocked_count,
        "missing_repro_tasks": reproducer_plan.missing_repro_tasks,
        "executable_commands": reproducer_plan.executable_commands,
        "artifacts": tuple(
            canonicalize(artifact) for artifact in reproducer_plan.artifacts
        ),
    }


def _engineering_org_plan_agent_summary(
    plan: EngineeringOrganizationPlan,
) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "release_strategy": plan.release_strategy,
        "architecture_delta_required": plan.architecture_delta_required,
        "public_api_delta_required": plan.public_api_delta_required,
        "developer_experience_delta_required": plan.developer_experience_delta_required,
        "security_review_required": plan.security_review_required,
        "work_package_count": len(plan.work_packages),
        "execution_order": plan.execution_order,
        "verifier_contract": plan.verifier_contract,
        "evidence_boundary": plan.evidence_boundary,
        "work_packages": tuple(
            {
                "package_id": package.package_id,
                "owner_role": package.owner_role,
                "mission": package.mission,
                "source_themes": package.source_themes,
                "task_ids": package.task_ids,
                "candidate_paths": package.candidate_paths,
                "required_outputs": package.required_outputs,
                "gate": package.gate,
                "risk_level": package.risk_level,
            }
            for package in plan.work_packages
        ),
    }


def _engineering_org_plan_metrics(plan: EngineeringOrganizationPlan) -> dict[str, Any]:
    roles = tuple(package.owner_role for package in plan.work_packages)
    gates = tuple(package.gate for package in plan.work_packages)
    return {
        "plan_id": plan.plan_id,
        "release_strategy": plan.release_strategy,
        "architecture_delta_required": plan.architecture_delta_required,
        "public_api_delta_required": plan.public_api_delta_required,
        "developer_experience_delta_required": plan.developer_experience_delta_required,
        "security_review_required": plan.security_review_required,
        "work_package_count": len(plan.work_packages),
        "roles": roles,
        "gates": gates,
        "high_risk_package_count": sum(
            1 for package in plan.work_packages if package.risk_level == "high"
        ),
    }


def _code_landing_metrics(
    *,
    source_report: Mapping[str, Any],
    requested_themes: tuple[str, ...],
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    spec_maturity_gate: Mapping[str, Any],
    repo_intelligence: RepoIntelligencePack,
    reproducer_plan: ReproducerPlan,
    engineering_org_plan: EngineeringOrganizationPlan,
    code_max_runtime_summary: Mapping[str, Any],
    patch_results: tuple[WorkspacePatchResult, ...],
    verification_results: tuple[WorkspaceVerificationResult, ...],
    final_mode: str,
) -> dict[str, Any]:
    del source_report
    funnel_records = _code_landing_funnel_records(
        requested_themes=requested_themes,
        task_specs=task_specs,
        reproducer_plan=reproducer_plan,
        patch_results=patch_results,
        verification_results=verification_results,
        final_mode=final_mode,
    )
    primary_failure_reason = next(
        (
            record.failure_reason
            for record in funnel_records
            if record.failure_reason is not None
        ),
        None,
    )
    return {
        "development_task_specs_v2": tuple(canonicalize(spec) for spec in task_specs),
        "spec_maturity_gate": canonicalize(spec_maturity_gate),
        "repo_intelligence": _repo_intelligence_metrics_summary(repo_intelligence),
        "reproducer_plan": _reproducer_plan_agent_summary(reproducer_plan),
        "engineering_organization_plan": _engineering_org_plan_metrics(
            engineering_org_plan
        ),
        "code_max_runtime": canonicalize(code_max_runtime_summary),
        "funnel_records": tuple(canonicalize(record) for record in funnel_records),
        "primary_failure_reason": primary_failure_reason,
    }


def _repo_intelligence_metrics_summary(
    repo_intelligence: RepoIntelligencePack,
) -> dict[str, Any]:
    return {
        "repo_hash": repo_intelligence.repo_hash,
        "file_count": len(repo_intelligence.file_tree),
        "package_manager": repo_intelligence.package_manager,
        "test_command_count": len(repo_intelligence.test_commands),
        "lint_command_count": len(repo_intelligence.lint_commands),
        "entrypoints": repo_intelligence.entrypoints,
        "source_roots": repo_intelligence.source_roots,
        "test_file_count": len(repo_intelligence.test_files),
        "symbol_count": len(repo_intelligence.symbol_index),
        "candidate_files_by_theme": repo_intelligence.candidate_files_by_theme,
    }


def _code_landing_funnel_records(
    *,
    requested_themes: tuple[str, ...],
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    reproducer_plan: ReproducerPlan,
    patch_results: tuple[WorkspacePatchResult, ...],
    verification_results: tuple[WorkspaceVerificationResult, ...],
    final_mode: str,
) -> tuple[CodeLandingFunnelRecord, ...]:
    records: list[CodeLandingFunnelRecord] = []
    if not task_specs:
        task_specs = tuple(
            DevelopmentTaskSpecV2(
                task_id=f"task_{_slug_label(theme)}",
                source_theme=theme,
                task_type=_task_type_for_theme(theme),
                public_evidence_refs=(),
                user_pain=_default_user_pain(theme),
                observed_behavior="",
                expected_behavior=_expected_behavior_for_theme(theme),
                non_goals=_non_goals_for_spec({}),
                reproduction_recipe=(),
                reproduction_expected_failure=None,
                acceptance_oracles=(),
                repo_entrypoints=(),
                candidate_path_hints=_candidate_path_hints_for_theme(theme),
                relevant_symbols_hint=_symbols_for_theme(theme),
                ambiguity_level="high",
                missing_information=(
                    "reproduction_recipe",
                    "executable_acceptance_oracle",
                ),
                spec_maturity="theme_only",
                risk_level=_risk_level_for_theme(theme),
                behavior_surface=_behavior_surface_for_theme(theme),
                contract_dimensions=_contract_dimensions_for_theme(theme, ()),
                boundary_conditions=_boundary_conditions_for_spec(
                    theme=theme,
                    user_pain=_default_user_pain(theme),
                    observed_behavior="",
                    expected_behavior=_expected_behavior_for_theme(theme),
                    reproduction_recipe=(),
                    contract_dimensions=_contract_dimensions_for_theme(theme, ()),
                ),
            )
            for theme in requested_themes
        )
    passed_commands = tuple(
        result.command for result in verification_results if result.status == "passed"
    )
    behavior_passed = any(
        _is_behavior_verification_command(command) for command in passed_commands
    )
    maintainer_gate_failed = any(
        result.command == "maintainer_landing_gate" and result.status != "passed"
        for result in verification_results
    )
    patch_applies = any(result.status == "applied" for result in patch_results)
    for spec in task_specs:
        spec_artifacts = tuple(
            artifact
            for artifact in reproducer_plan.artifacts
            if artifact.task_id == spec.task_id
        )
        repro_confirmed = any(
            artifact.reproducibility_status == "confirmed"
            for artifact in spec_artifacts
        )
        repro_passed = any(
            artifact.command in passed_commands for artifact in spec_artifacts
        )
        failure_reason = _code_landing_failure_reason(
            spec=spec,
            repro_confirmed=repro_confirmed,
            patch_results=patch_results,
            verification_results=verification_results,
            final_mode=final_mode,
        )
        records.append(
            CodeLandingFunnelRecord(
                task_id=spec.task_id,
                theme=spec.source_theme,
                public_feedback_contains_pain=bool(
                    spec.public_evidence_refs or spec.user_pain
                ),
                optimizer_selected_theme=spec.source_theme in requested_themes,
                spec_maturity=spec.spec_maturity,
                repro_confirmed=repro_confirmed,
                localization_file_hit=_localization_file_hit(spec, patch_results),
                localization_symbol_hit=None,
                patch_applies=patch_applies,
                repro_passed=repro_passed,
                targeted_tests_passed=behavior_passed,
                regression_tests_passed=None,
                landing_gate_passed=final_mode == "verified"
                and not maintainer_gate_failed,
                historical_theme_overlap=None,
                evaluator_passed=final_mode == "verified",
                failure_reason=failure_reason,
            )
        )
    return tuple(records)


def _localization_file_hit(
    spec: DevelopmentTaskSpecV2,
    patch_results: tuple[WorkspacePatchResult, ...],
) -> bool | None:
    changed_paths = _changed_patch_paths(patch_results)
    if not changed_paths:
        return None
    return any(
        _path_matches_hint(path, hint)
        for path in changed_paths
        for hint in spec.candidate_path_hints
    )


def _code_landing_failure_reason(
    *,
    spec: DevelopmentTaskSpecV2,
    repro_confirmed: bool,
    patch_results: tuple[WorkspacePatchResult, ...],
    verification_results: tuple[WorkspaceVerificationResult, ...],
    final_mode: str,
) -> str | None:
    if final_mode == "verified":
        return None
    if _maturity_rank(spec.spec_maturity) < _maturity_rank("testable"):
        return CodeLandingFailureReason.SPEC_TOO_AMBIGUOUS.value
    if spec.acceptance_oracles and not repro_confirmed:
        return CodeLandingFailureReason.MISSING_REPRO.value
    if any(result.status == "blocked" for result in patch_results):
        return CodeLandingFailureReason.PATCH_DID_NOT_APPLY.value
    if patch_results and not _has_effective_patch(patch_results):
        return CodeLandingFailureReason.PATCH_NO_EFFECT.value
    maintainer_issues = tuple(
        issue
        for result in verification_results
        if result.command == "maintainer_landing_gate" and result.status != "passed"
        for issue in (result.stderr_tail or "").split(";")
        if issue
    )
    if "broad_source_rewrite_without_behavior_review" in maintainer_issues:
        return CodeLandingFailureReason.BROAD_REWRITE.value
    if (
        "missing_behavior_verification_command" in maintainer_issues
        or "missing_test_or_repro_artifact" in maintainer_issues
    ):
        return CodeLandingFailureReason.BEHAVIOR_NOT_VERIFIED.value
    if final_mode == "verification_failed":
        return CodeLandingFailureReason.REGRESSION_BREAK.value
    return CodeLandingFailureReason.BAD_CONTEXT.value


def _extract_artifact_id(source_report: Mapping[str, Any]) -> str | None:
    proposal = source_report.get("proposal")
    if isinstance(proposal, Mapping) and proposal.get("artifact_id"):
        return str(proposal["artifact_id"])[:160]
    development_run = source_report.get("development_run")
    if isinstance(development_run, Mapping) and development_run.get("artifact_id"):
        return str(development_run["artifact_id"])[:160]
    return _optional_string(source_report.get("artifact_id"))


def _extract_support_refs(source_report: Mapping[str, Any]) -> tuple[str, ...]:
    refs: list[str] = []
    proposal = source_report.get("proposal")
    if isinstance(proposal, Mapping):
        raw = proposal.get("support_refs")
        if isinstance(raw, (list, tuple)) and raw:
            refs.extend(str(item)[:160] for item in raw[:16])
    feedback = source_report.get("public_feedback_summary")
    if isinstance(feedback, Mapping):
        artifact_feedback = feedback.get("artifact_feedback", {})
        if isinstance(artifact_feedback, Mapping):
            for item in artifact_feedback.values():
                if not isinstance(item, Mapping):
                    continue
                raw = item.get("support_refs")
                if isinstance(raw, (list, tuple)):
                    refs.extend(str(ref)[:160] for ref in raw[:16])
    return tuple(dict.fromkeys(refs))[:16]


def _spec_evidence_keywords(
    themes: tuple[str, ...],
    task_specs: Sequence[DevelopmentTaskSpecV2 | Mapping[str, Any]],
) -> tuple[str, ...]:
    """Evidence-derived keywords for file selection, substrate-neutral.

    Curated THEME_KEYWORDS only cover legacy themes; substrate-derived themes
    would otherwise fall back to their own name lexemes and lose recall on
    files reachable only through the spec's evidence vocabulary.
    """

    theme_set = set(themes)
    terms: list[str] = []
    for spec in task_specs:
        if isinstance(spec, Mapping):
            spec_theme = str(spec.get("source_theme") or spec.get("theme") or "")
            symbols = spec.get("relevant_symbols_hint") or ()
            path_hints = spec.get("candidate_path_hints") or ()
        else:
            spec_theme = spec.source_theme
            symbols = spec.relevant_symbols_hint
            path_hints = spec.candidate_path_hints
        if spec_theme not in theme_set:
            continue
        terms.extend(str(symbol) for symbol in symbols)
        terms.extend(Path(str(hint)).stem for hint in path_hints)
    return tuple(term for term in terms if len(term) >= 4)


def _keywords_for_themes(
    themes: tuple[str, ...],
    task_specs: Sequence[DevelopmentTaskSpecV2 | Mapping[str, Any]] = (),
) -> tuple[str, ...]:
    keywords: list[str] = []
    for theme in themes:
        keywords.extend(THEME_KEYWORDS.get(theme, (theme.replace("_", " "),)))
        keywords.extend(
            part for part in theme.replace("-", "_").split("_") if len(part) >= 4
        )
    keywords.extend(_spec_evidence_keywords(themes, task_specs))
    keywords.extend(("config", "index", "src", "lib", "test", "readme"))
    return tuple(dict.fromkeys(keyword.lower() for keyword in keywords if keyword))


def _file_relevance_score(path: str, text: str, keywords: tuple[str, ...]) -> float:
    lowered_path = path.lower()
    lowered_text = text[:20000].lower()
    score = 0.0
    name = Path(path).name.lower()
    if name in {"package.json", "pyproject.toml"}:
        score += 200
    if name in {
        "readme.md",
        "index.js",
        "index.ts",
        "index.mjs",
        "config.js",
        "config.ts",
    }:
        score += 50
    if lowered_path.startswith(("src/", "lib/", "packages/", "dist-src/", "dist/")):
        score += 25
    if lowered_path.startswith(("test/", "tests/", "__tests__/")):
        score += 10
    for keyword in keywords:
        if keyword in lowered_path:
            score += 18
        count = lowered_text.count(keyword)
        if count:
            score += min(12, count * 2)
    return score


def _is_ignored_workspace_path(path: Path, root: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    ignored_dirs = set(IGNORED_DIRS)
    if _published_dist_is_primary_source(root) and relative_parts[:1] == ("dist",):
        ignored_dirs.difference_update({"build", "dist"})
    if any(part in ignored_dirs for part in relative_parts):
        return True
    name = path.name.lower()
    suffix = path.suffix.lower()
    return suffix in IGNORED_SUFFIXES or any(
        name.endswith(item) for item in IGNORED_SUFFIXES
    )


def _published_dist_is_primary_source(root: Path) -> bool:
    if not (root / "package.json").exists() or not (root / "dist").is_dir():
        return False
    source_dirs = ("src", "lib", "packages", "dist-src")
    return not any((root / dirname).exists() for dirname in source_dirs)


def _verification_summary(
    results: tuple[WorkspaceVerificationResult, ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "command": result.command,
            "status": result.status,
            "exit_code": result.exit_code,
            "stdout_tail": result.stdout_tail[-1200:],
            "stderr_tail": result.stderr_tail[-1200:],
            "blocked_reason": result.blocked_reason,
        }
        for result in results
    )


def _agent_tool_trace_failure_feedback(
    agent: WorkspaceCodingAgent,
    *,
    proposal: CodingAgentPatchProposal,
) -> tuple[WorkspaceVerificationResult, ...]:
    """Carry bounded, typed tool failures into the next development round."""

    if _is_candidate_infrastructure_block(proposal.blocked_reason):
        return ()
    trace = getattr(agent, "last_trace", None)
    if trace is None:
        return ()
    if isinstance(trace, Mapping):
        tool_calls = tuple(trace.get("tool_calls", ()) or ())
    else:
        tool_calls = tuple(getattr(trace, "tool_calls", ()) or ())

    failed_checks: dict[str, WorkspaceVerificationResult] = {}
    exhausted_checks: dict[str, WorkspaceVerificationResult] = {}
    unwired_paths: dict[str, WorkspaceVerificationResult] = {}
    for item in tool_calls:
        if isinstance(item, Mapping):
            name = str(item.get("name") or "")
            arguments = (
                dict(item.get("arguments") or {})
                if isinstance(item.get("arguments"), Mapping)
                else {}
            )
            result = (
                dict(item.get("result") or {})
                if isinstance(item.get("result"), Mapping)
                else {}
            )
        else:
            name = str(getattr(item, "name", "") or "")
            raw_arguments = getattr(item, "arguments", {})
            raw_result = getattr(item, "result", {})
            arguments = (
                dict(raw_arguments) if isinstance(raw_arguments, Mapping) else {}
            )
            result = dict(raw_result) if isinstance(raw_result, Mapping) else {}

        warnings = result.get("call_path_warnings")
        if isinstance(warnings, Mapping):
            for path in tuple(warnings.get("unwired_created_source_paths", ()) or ()):
                normalized_path = str(path)
                if normalized_path:
                    unwired_paths[normalized_path] = _unwired_trace_feedback(
                        normalized_path,
                        warnings,
                    )

        status = str(result.get("status") or "")
        reason = str(result.get("reason") or "")
        if reason.startswith("unwired_created_source_path:"):
            path = str(
                result.get("path") or arguments.get("path") or reason.partition(":")[2]
            )
            if path:
                unwired_paths[path] = _unwired_trace_feedback(path, result)

        if name not in {"run_check", "run_python_check"}:
            continue
        command = str(
            result.get("normalized_command")
            or arguments.get("command")
            or (
                f"run_python_check:{arguments.get('script')}"
                if arguments.get("script")
                else ""
            )
        )
        if not command:
            continue
        feedback = WorkspaceVerificationResult(
            command=_bounded_redacted_text(
                command,
                max_chars=MAX_AGENT_VERIFICATION_COMMAND_CHARS,
            ),
            status=status,
            exit_code=(
                int(result["exit_code"])
                if isinstance(result.get("exit_code"), int)
                else None
            ),
            elapsed_sec=0.0,
            stdout_tail=_bounded_redacted_text(
                result.get("stdout_tail", ""),
                max_chars=MAX_AGENT_TRACE_FAILURE_TAIL_CHARS,
            ),
            stderr_tail=_bounded_redacted_text(
                result.get("stderr_tail", ""),
                max_chars=MAX_AGENT_TRACE_FAILURE_TAIL_CHARS,
            ),
            blocked_reason=(
                _bounded_redacted_text(reason, max_chars=600) if reason else None
            ),
        )
        if status == "failed" and result.get("finalization_blocking") is not False:
            failed_checks[feedback.command] = feedback
        elif (
            status == "blocked"
            and reason == "failed_check_retry_limit_current_revision"
        ):
            exhausted_checks[feedback.command] = feedback

    ordered: list[WorkspaceVerificationResult] = [
        *failed_checks.values(),
        *(
            feedback
            for command, feedback in exhausted_checks.items()
            if command not in failed_checks
        ),
        *unwired_paths.values(),
    ]
    return tuple(ordered[:MAX_AGENT_TRACE_FAILURE_FEEDBACK])


def _unwired_trace_feedback(
    path: str,
    evidence: Mapping[str, Any],
) -> WorkspaceVerificationResult:
    payload = {
        "path": path,
        "behavior_failure_count": evidence.get("behavior_failure_count"),
        "required_action": evidence.get("required_action"),
    }
    return WorkspaceVerificationResult(
        command=f"call_path:{path}",
        status="blocked",
        exit_code=None,
        elapsed_sec=0.0,
        stdout_tail=_bounded_redacted_text(
            json.dumps(
                canonicalize(payload),
                sort_keys=True,
                separators=(",", ":"),
            ),
            max_chars=MAX_AGENT_TRACE_FAILURE_TAIL_CHARS,
        ),
        stderr_tail=f"unwired_created_source_path:{path}",
        blocked_reason=f"unwired_created_source_path:{path}",
    )


def _bounded_redacted_text(value: Any, *, max_chars: int) -> str:
    redacted = str(redact_sensitive_payload(str(value)))
    if len(redacted) <= max_chars:
        return redacted
    suffix_chars = min(max_chars // 3, 500)
    marker = f"...[truncated:{stable_hash(redacted)[:16]}]..."
    prefix_chars = max(0, max_chars - suffix_chars - len(marker))
    return redacted[:prefix_chars] + marker + redacted[-suffix_chars:]


def _merge_verification_feedback(
    *groups: tuple[WorkspaceVerificationResult, ...],
    max_results: int = 12,
) -> tuple[WorkspaceVerificationResult, ...]:
    merged: list[WorkspaceVerificationResult] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        for result in group:
            key = (
                result.command,
                result.status,
                result.exit_code,
                result.blocked_reason,
                result.stdout_tail,
                result.stderr_tail,
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(result)
            if len(merged) >= max_results:
                return tuple(merged)
    return tuple(merged)


def _patch_block_feedback(
    patch_results: tuple[WorkspacePatchResult, ...],
) -> tuple[WorkspaceVerificationResult, ...]:
    blocked = tuple(result for result in patch_results if result.status == "blocked")
    if not blocked:
        return ()
    return tuple(
        WorkspaceVerificationResult(
            command=f"apply_workspace_patch:{result.patch_id}:{result.path}",
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            stdout_tail=(
                f"operation={result.operation};before_hash={result.before_hash};"
                f"after_hash={result.after_hash}"
            ),
            stderr_tail=result.blocked_reason or "",
            blocked_reason=result.blocked_reason,
        )
        for result in blocked
    )


def _proposal_block_feedback(
    proposal: CodingAgentPatchProposal,
) -> tuple[WorkspaceVerificationResult, ...]:
    return (
        WorkspaceVerificationResult(
            command=f"agent_proposal:{proposal.proposal_id}",
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            stdout_tail=f"source={proposal.source};model={proposal.model}",
            stderr_tail=proposal.blocked_reason or "",
            blocked_reason=proposal.blocked_reason,
        ),
    )


def _candidate_selection_feedback(
    proposal: CodingAgentPatchProposal,
    runtime_record: Mapping[str, Any],
) -> tuple[WorkspaceVerificationResult, ...]:
    findings = tuple(runtime_record.get("review_findings", ()) or ())
    finding_codes = tuple(
        dict.fromkeys(
            str(item.get("code") or "unspecified_review_finding")
            for item in findings
            if isinstance(item, Mapping)
        )
    )
    payload = {
        "reason": runtime_record.get("reason"),
        "landing_reasons": tuple(runtime_record.get("landing_reasons", ()) or ()),
        "validation_failure_codes": tuple(
            runtime_record.get("validation_failure_codes", ()) or ()
        ),
        "review_findings": findings[:8],
        "required_followup": tuple(runtime_record.get("required_followup", ()) or ())[
            :8
        ],
        "parent_proposal_id": proposal.proposal_id,
        "parent_patch_fingerprint": stable_hash(
            {
                "patches": proposal.patches,
                "verification_commands": proposal.verification_commands,
            }
        )[:24],
        "changed_paths": tuple(dict.fromkeys(patch.path for patch in proposal.patches)),
    }
    rendered = json.dumps(
        canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    concise_codes = tuple(
        dict.fromkeys(
            (
                *finding_codes,
                *tuple(payload["validation_failure_codes"]),
                *tuple(payload["landing_reasons"]),
            )
        )
    )
    return (
        WorkspaceVerificationResult(
            command=f"candidate_selection:{proposal.proposal_id}",
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            stdout_tail=rendered[:6000],
            stderr_tail="repair_codes=" + ",".join(concise_codes)[:1200],
            blocked_reason=proposal.blocked_reason,
        ),
    )


def _is_retryable_agent_block(reason: str | None) -> bool:
    return bool(
        reason
        and reason.startswith(
            (
                "openai_request_failed:",
                "openai_tool_loop_failed:",
                "openai_response_invalid:",
                "code_max_landing_needs_revision:",
                "code_max_all_candidates_failed_validation:",
                "code_max_selection_inconsistent:",
                "code_max_runtime_infra_error:",
                "maintainer_preflight_gate_failed:",
            )
        )
    )


def _should_preserve_repair_seed(
    *,
    repair_fast_path: bool,
    repair_seed_proposal: CodingAgentPatchProposal | None,
    repair_seed_record: Mapping[str, Any] | None,
    blocked_reason: str | None,
    selected_runtime_record: Mapping[str, Any] | None,
) -> bool:
    if not (
        repair_fast_path
        and repair_seed_proposal is not None
        and repair_seed_record is not None
    ):
        return False
    if _is_retryable_agent_block(blocked_reason):
        return True
    return bool(
        selected_runtime_record is not None
        and selected_runtime_record.get("reason") == "all_candidate_proposals_blocked"
    )


def _is_candidate_infrastructure_block(reason: str | None) -> bool:
    return bool(
        reason
        and reason.startswith(
            (
                "openai_request_failed:",
                "openai_tool_loop_failed:",
                "execution_backend_preflight_failed:",
                "execution_infrastructure_unavailable:",
            )
        )
    )


def _is_provider_infrastructure_block(reason: str | None) -> bool:
    return bool(
        reason
        and reason.startswith(
            (
                "openai_request_failed:",
                "openai_tool_loop_failed:",
            )
        )
    )


def _provider_infrastructure_reason_code(reason: str | None) -> str:
    if not reason:
        return "unknown_provider_failure"
    if "APITimeoutError" in reason or "Timeout" in reason:
        return "provider_timeout"
    if "InternalServerError" in reason:
        return "provider_server_error"
    if "RateLimitError" in reason:
        return "provider_rate_limit"
    if "APIConnectionError" in reason:
        return "provider_connection_error"
    return "provider_request_error"


def _workspace_coding_candidate_infra_retries() -> int:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_CANDIDATE_INFRA_RETRIES")
    if raw is None:
        return DEFAULT_WORKSPACE_CODING_CANDIDATE_INFRA_RETRIES
    try:
        return max(0, min(8, int(raw)))
    except ValueError:
        return DEFAULT_WORKSPACE_CODING_CANDIDATE_INFRA_RETRIES


def _workspace_coding_candidate_infra_round_retries() -> int:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_CANDIDATE_INFRA_ROUND_RETRIES")
    if raw is None:
        return DEFAULT_WORKSPACE_CODING_CANDIDATE_INFRA_ROUND_RETRIES
    try:
        return max(0, min(16, int(raw)))
    except ValueError:
        return DEFAULT_WORKSPACE_CODING_CANDIDATE_INFRA_ROUND_RETRIES


def _workspace_coding_provider_circuit_exhausted_slots() -> int:
    raw = os.environ.get(
        "SOCIETY_CORE_WORKSPACE_CODING_PROVIDER_CIRCUIT_EXHAUSTED_SLOTS"
    )
    if raw is None:
        return DEFAULT_WORKSPACE_CODING_PROVIDER_CIRCUIT_EXHAUSTED_SLOTS
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return DEFAULT_WORKSPACE_CODING_PROVIDER_CIRCUIT_EXHAUSTED_SLOTS


def _candidate_infra_retry_sleep_seconds(attempt: int) -> float:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_CANDIDATE_RETRY_SLEEP_SECONDS")
    if raw is not None:
        try:
            return max(0.0, min(180.0, float(raw)))
        except ValueError:
            pass
    return min(180.0, 30.0 * (2**attempt))


def _patch_no_effect_feedback() -> tuple[WorkspaceVerificationResult, ...]:
    return (
        WorkspaceVerificationResult(
            command="apply_workspace_patches:no_effect",
            status="failed",
            exit_code=None,
            elapsed_sec=0.0,
            stdout_tail="All proposed patches left target file hashes unchanged.",
            stderr_tail="no_effect_patch_set",
            blocked_reason="no_effect_patch_set",
        ),
    )


def _workspace_promotion_feedback(
    promotion: WorkspacePromotionResult,
) -> WorkspaceVerificationResult:
    status = "failed" if promotion.status == "verification_failed" else "blocked"
    return WorkspaceVerificationResult(
        command="workspace_promotion",
        status=status,
        exit_code=None,
        elapsed_sec=round(
            sum(result.elapsed_sec for result in promotion.verification_results),
            6,
        ),
        stdout_tail=json.dumps(
            {
                "status": promotion.status,
                "changed_paths": promotion.changed_paths,
                "baseline_digest": promotion.baseline_digest,
                "candidate_digest": promotion.candidate_digest,
                "promoted_digest": promotion.promoted_digest,
                "rollback_performed": promotion.rollback_performed,
            },
            sort_keys=True,
            separators=(",", ":"),
        )[:2000],
        blocked_reason=promotion.blocked_reason or f"promotion_{promotion.status}",
    )


def _candidate_search_strategies(
    code_max_summary: Mapping[str, Any],
    *,
    count: int,
    declared_component_paths: tuple[str, ...] = (),
) -> tuple[str, ...]:
    reproducer_strategy = (
        "reproducer_first_root_cause: reproduce the public failure, trace the "
        "causal call path, and implement the smallest complete root-cause fix"
    )
    component_strategy = (
        "declared_component_ownership: treat each existing explicit component path "
        "in the public task as an implementation ownership hypothesis, trace its "
        "callers and fallbacks, and change that component as part of a complete "
        "behavioral fix; do not return an adapter-only patch when an explicit "
        "owning component exists"
    )
    leading_strategies = (
        (component_strategy, reproducer_strategy)
        if declared_component_paths
        else (reproducer_strategy, component_strategy)
    )
    strategy_contracts = (
        *leading_strategies,
        (
            "adversarial_boundary_matrix: encode the user-visible contract at the "
            "highest public entrypoint, derive nearest sibling states owned by the "
            "same abstraction, partition omitted, empty, malformed, boundary, and valid "
            "scalar states when applicable, and falsify a narrow fix with contrastive "
            "tests before implementing a compatibility-preserving systemic repair; "
            "when a sibling candidate edits an adapter or call site, independently "
            "inspect the owning abstraction and prove why the mechanism must stay at "
            "or move across that boundary"
        ),
        (
            "integration_synthesis: inspect bounded public sibling candidate diffs, "
            "combine only complementary behavior supported by the public evidence and "
            "repository contract into an independently justified implementation; cover "
            "only entrypoints and boundary dimensions implicated by that evidence, "
            "actively seek a public counterexample to each sibling fix in the nearest "
            "states handled by the same owning abstraction, retain a baseline-failing "
            "focused test for each claimed dimension, and encode expected results with "
            "assertions or explicit nonzero exits"
        ),
        (
            "architecture_preserving_systemic: repair the owning abstraction and all "
            "relevant call sites without broad unrelated churn"
        ),
        (
            "differential_alternative_path: compare neighboring successful paths and "
            "implement an independent fix validated by differential behavior"
        ),
    )
    discovered: list[str] = []
    plans = code_max_summary.get("patch_strategy_summary", ())
    if isinstance(plans, (list, tuple)):
        for plan in plans:
            strategies = _mapping_get(plan, "strategies", ())
            if isinstance(strategies, (list, tuple)):
                discovered.extend(str(item) for item in strategies if str(item))
    unique = tuple(dict.fromkeys((*strategy_contracts, *discovered)))
    return tuple(unique[index % len(unique)] for index in range(count))


def _candidate_search_payload(
    *,
    candidate_index: int,
    candidate_count: int,
    strategy: str,
    prior_proposals: tuple[CodingAgentPatchProposal, ...],
    workspace_root: Path,
    slot_attempt: int = 1,
    max_slot_attempts: int = 1,
    repair_seed: CodingAgentPatchProposal | None = None,
    repair_record: Mapping[str, Any] | None = None,
    declared_component_paths: tuple[str, ...] = (),
    strategy_feedback: tuple[str, ...] = (),
) -> dict[str, Any]:
    integration_mode = "integration_synthesis" in strategy
    payload: dict[str, Any] = {
        "candidate_index": candidate_index,
        "candidate_count": candidate_count,
        "strategy": strategy,
        "independent_workspace_required": True,
        "integration_mode": integration_mode,
        "slot_attempt": slot_attempt,
        "max_slot_attempts": max_slot_attempts,
        "declared_component_paths": declared_component_paths,
        "boundary_matrix_policy": (
            "Derive only from the public failure and repository structure. Test the "
            "reported state, the nearest valid state, and adjacent states processed by "
            "the same owning abstraction. For scalar configuration, distinguish absent, "
            "empty, malformed, lower-bound, and representative valid values, including "
            "normalization by wrappers or framework adapters. Preserve output validity, "
            "idempotence, and content or state invariants when those properties apply."
        ),
    }
    if "declared_component_ownership" in strategy and declared_component_paths:
        payload["required_changed_paths"] = declared_component_paths
        payload["strategy_constraint_policy"] = (
            "At least one required_changed_paths entry must be present in the final "
            "product patch. Inspect adapters and tests as needed, but an adapter-only "
            "patch is not a valid candidate for this ownership strategy."
        )
    if strategy_feedback:
        payload["strategy_feedback"] = strategy_feedback
        payload["strategy_feedback_policy"] = (
            "The controller rejected the previous attempt for structural strategy "
            "noncompliance. Produce a materially corrected implementation, not a "
            "restatement of the rejected mechanism."
        )
    if prior_proposals:
        payload["candidate_bank"] = tuple(
            {
                "proposal_id": proposal.proposal_id,
                "semantic_fingerprint": _candidate_semantic_fingerprint(proposal),
                "changed_paths": tuple(
                    dict.fromkeys(patch.path for patch in proposal.patches)
                ),
                "verification_commands": proposal.verification_commands[:6],
                "rationale": proposal.rationale[:600],
            }
            for proposal in prior_proposals[-6:]
            if not proposal.blocked_reason
        )
        payload["candidate_bank_policy"] = (
            "Do not regenerate a semantically equivalent candidate. First try to "
            "falsify each prior mechanism with a public boundary state. If a prior "
            "candidate edits an adapter or call site, inspect the owning abstraction; "
            "if it edits the owning abstraction, inspect public adapters and fallback "
            "paths. Return the same mechanism only with concrete repository evidence "
            "that the alternative layer is incorrect. Repair a known candidate only "
            "when the repair capsule identifies a concrete remaining finding."
        )
    if repair_seed is not None and repair_record is not None:
        payload["repair_capsule"] = _candidate_repair_capsule(
            repair_seed,
            repair_record,
            workspace_root=workspace_root,
        )
    if integration_mode:
        payload["sibling_candidate_evidence"] = _bounded_sibling_candidate_evidence(
            prior_proposals,
            workspace_root=workspace_root,
        )
        payload["integration_evidence_policy"] = (
            "Sibling diffs are public candidate evidence, not an oracle. Re-derive the "
            "contract from the repository, integrate only complementary behavior, add "
            "baseline-failing focused tests, and independently verify the final union."
        )
    return payload


def _declared_component_paths(
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    *,
    workspace_root: Path,
) -> tuple[str, ...]:
    root = workspace_root.resolve()
    paths: list[str] = []
    for spec in task_specs:
        if not spec.source_explicit_intent_spec:
            continue
        for raw_path in spec.candidate_path_hints:
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts:
                continue
            normalized = path.as_posix().lstrip("./")
            if not normalized or not (root / normalized).is_file():
                continue
            paths.append(normalized)
    return tuple(dict.fromkeys(paths))


def _candidate_strategy_compliance_issues(
    proposal: CodingAgentPatchProposal,
    *,
    strategy: str,
    declared_component_paths: tuple[str, ...],
) -> tuple[str, ...]:
    if proposal.blocked_reason:
        return ()
    issues: list[str] = []
    changed_paths = {
        Path(patch.path).as_posix().lstrip("./")
        for patch in proposal.patches
        if patch.path and _counts_as_product_patch_path(patch.path)
    }
    if (
        "declared_component_ownership" in strategy
        and declared_component_paths
        and changed_paths.isdisjoint(declared_component_paths)
    ):
        issues.append(
            "declared_component_not_changed:" + ",".join(declared_component_paths[:4])
        )
    return tuple(issues)


def _proposal_changes_declared_component(
    proposal: CodingAgentPatchProposal,
    *,
    declared_component_paths: tuple[str, ...],
) -> bool:
    if proposal.blocked_reason or not declared_component_paths:
        return False
    changed_paths = {
        Path(patch.path).as_posix().lstrip("./")
        for patch in proposal.patches
        if patch.path and _counts_as_product_patch_path(patch.path)
    }
    return not changed_paths.isdisjoint(declared_component_paths)


def _candidate_semantic_fingerprint(
    proposal: CodingAgentPatchProposal,
) -> str:
    return stable_hash(
        {
            "patches": tuple(
                {
                    "path": patch.path,
                    "operation": patch.operation,
                    "content": patch.content,
                    "old": patch.old,
                    "new": patch.new,
                }
                for patch in proposal.patches
            ),
            "verification_commands": tuple(
                dict.fromkeys(proposal.verification_commands)
            ),
        }
    )[:24]


def _candidate_early_stop_shadow_record(
    *,
    proposals: tuple[CodingAgentPatchProposal, ...],
    observed_candidate_count: int,
    strategies: tuple[str, ...],
    declared_component_paths: tuple[str, ...],
    selection_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a conservative early-stop policy without changing execution."""

    viable = tuple(proposal for proposal in proposals if not proposal.blocked_reason)
    exact_fingerprints = {
        _candidate_semantic_fingerprint(proposal) for proposal in viable
    }
    strategy_families = {
        _candidate_strategy_family(strategy)
        for proposal, strategy in zip(proposals, strategies, strict=False)
        if not proposal.blocked_reason
    }
    component_ready = not declared_component_paths or any(
        _proposal_changes_declared_component(
            proposal,
            declared_component_paths=declared_component_paths,
        )
        for proposal in viable
    )
    selected_proposal_id = str(selection_record.get("selected_proposal_id") or "")
    selected_in_prefix = selected_proposal_id in {
        proposal.proposal_id for proposal in viable
    }
    validations = tuple(selection_record.get("validations") or ())
    selected_validation = next(
        (
            item
            for item in validations
            if isinstance(item, Mapping)
            and str(item.get("proposal_id") or "") == selected_proposal_id
        ),
        None,
    )
    selected_validation_ready = bool(
        isinstance(selected_validation, Mapping)
        and selected_validation.get("applies_cleanly") is True
        and selected_validation.get("repro_passed") is True
        and selected_validation.get("candidate_verification_passed") is True
        and selected_validation.get("candidate_behavior_verification_passed") is True
        and selected_validation.get("dimension_coverage_complete") is True
        and selected_validation.get("public_contract_continuity_passed") is True
        and selected_validation.get("added_test_verification_status")
        in {"passed", "not_applicable"}
    )
    landing_ready = selection_record.get(
        "status"
    ) == "completed" and selection_record.get("landing_status") in {
        "candidate_validated",
        "candidate_validated_with_claim_downgrade",
    }
    checks = {
        "minimum_prefix_generated": len(proposals) >= CANDIDATE_EARLY_STOP_PREFIX_COUNT,
        "viable_quorum": len(viable) >= 2,
        "distinct_exact_patch_quorum": len(exact_fingerprints) >= 2,
        "distinct_strategy_family_quorum": len(strategy_families) >= 2,
        "declared_component_ready": component_ready,
        "selected_candidate_in_prefix": selected_in_prefix,
        "selected_candidate_strictly_validated": selected_validation_ready,
        "landing_gate_accepted": landing_ready,
    }
    eligible = all(checks.values())
    payload = {
        "record_type": "candidate_early_stop_shadow",
        "policy_version": CANDIDATE_EARLY_STOP_POLICY_VERSION,
        "mode": CANDIDATE_EARLY_STOP_MODE,
        "counterfactual_only": True,
        "eligible": eligible,
        "checks": checks,
        "prefix_candidate_count": len(proposals),
        "observed_candidate_count": observed_candidate_count,
        "viable_prefix_count": len(viable),
        "distinct_exact_patch_count": len(exact_fingerprints),
        "distinct_strategy_family_count": len(strategy_families),
        "selected_proposal_id": selected_proposal_id or None,
        "estimated_avoided_agent_slots": (
            max(0, observed_candidate_count - len(proposals)) if eligible else 0
        ),
        "claim_boundary": (
            "Retrospective shadow telemetry only; no causal cost or quality claim "
            "and no candidate generation was skipped."
        ),
    }
    return {**payload, "evidence_hash": stable_hash(payload)}


def _candidate_strategy_family(strategy: str) -> str:
    marker = strategy.partition(":")[0].strip()
    if marker:
        return marker
    return stable_hash(strategy)[:12]


def _compose_targeted_repair_proposal(
    *,
    parent: CodingAgentPatchProposal,
    child: CodingAgentPatchProposal,
    workspace_root: Path,
) -> CodingAgentPatchProposal:
    if child.blocked_reason:
        return child
    if not parent.patches or not child.patches:
        return replace(
            child,
            blocked_reason="targeted_repair_composition_conflict:empty_patch_set",
        )
    parent_paths = {
        Path(patch.path).as_posix().lstrip("./") for patch in parent.patches
    }
    composition_hash = stable_hash(
        {
            "parent_proposal_id": parent.proposal_id,
            "parent_patches": parent.patches,
            "child_proposal_id": child.proposal_id,
            "child_patches": child.patches,
        }
    )
    parent_patches = tuple(
        replace(
            patch,
            patch_id=f"repair_parent_{composition_hash[:10]}_{index}",
        )
        for index, patch in enumerate(parent.patches)
    )
    child_patches = tuple(
        replace(
            patch,
            patch_id=f"repair_child_{composition_hash[:10]}_{index}",
        )
        for index, patch in enumerate(child.patches)
    )
    with CandidateWorkspace.create(workspace_root) as composition_workspace:
        parent_results = composition_workspace.apply(parent_patches)
        parent_block = next(
            (result for result in parent_results if result.status == "blocked"),
            None,
        )
        if parent_block is not None:
            return replace(
                child,
                blocked_reason=(
                    "targeted_repair_composition_conflict:parent_patch:"
                    f"{parent_block.path}:{parent_block.blocked_reason}"
                )[:300],
            )
        for patch in child_patches:
            normalized_path = Path(patch.path).as_posix().lstrip("./")
            if (
                patch.operation != "create_or_replace"
                or normalized_path not in parent_paths
            ):
                continue
            baseline_path = workspace_root / normalized_path
            parent_path = composition_workspace.root / normalized_path
            try:
                baseline_content = (
                    read_regular_file_text(
                        workspace_root,
                        baseline_path,
                        max_bytes=4_000_000,
                    )
                    if baseline_path.exists()
                    else ""
                )
                parent_content = read_regular_file_text(
                    composition_workspace.root,
                    parent_path,
                    max_bytes=4_000_000,
                )
            except (OSError, UnicodeError, UnsafeRegularFileError, ValueError) as exc:
                return replace(
                    child,
                    blocked_reason=(
                        "targeted_repair_composition_conflict:"
                        f"unsafe_full_file_comparison:{normalized_path}:"
                        f"{type(exc).__name__}"
                    )[:300],
                )
            if _full_file_child_reverts_parent_change(
                baseline_content,
                parent_content,
                patch.content,
            ):
                return replace(
                    child,
                    blocked_reason=(
                        "targeted_repair_composition_conflict:"
                        f"child_reverts_parent_change:{normalized_path}"
                    )[:300],
                )
        child_results = composition_workspace.apply(child_patches)
        child_block = next(
            (result for result in child_results if result.status == "blocked"),
            None,
        )
        if child_block is not None:
            return replace(
                child,
                blocked_reason=(
                    "targeted_repair_composition_conflict:child_patch:"
                    f"{child_block.path}:{child_block.blocked_reason}"
                )[:300],
            )

    return CodingAgentPatchProposal(
        proposal_id=f"targeted_repair_{composition_hash[:20]}",
        source="targeted_repair_composer",
        model=child.model,
        iteration=child.iteration,
        artifact_id=child.artifact_id or parent.artifact_id,
        themes=tuple(dict.fromkeys((*parent.themes, *child.themes))),
        patches=(*parent_patches, *child_patches),
        verification_commands=_dedupe(
            (*parent.verification_commands, *child.verification_commands)
        ),
        selected_files=tuple(
            dict.fromkeys((*parent.selected_files, *child.selected_files))
        ),
        rationale=(
            "Deterministic targeted repair composition.\n"
            f"Parent rationale:\n{parent.rationale}\n"
            f"Child rationale:\n{child.rationale}"
        ),
        support_refs=tuple(dict.fromkeys((*parent.support_refs, *child.support_refs))),
        raw_response_hash=stable_hash(
            {
                "parent_raw_response_hash": parent.raw_response_hash,
                "child_raw_response_hash": child.raw_response_hash,
                "composition_hash": composition_hash,
            }
        ),
        stage="promote",
    )


def _full_file_child_reverts_parent_change(
    baseline: str,
    parent: str,
    child: str,
) -> bool:
    """Detect definite baseline restoration while allowing further repair."""

    baseline_lines = baseline.splitlines(keepends=True)
    parent_lines = parent.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(
        None,
        baseline_lines,
        parent_lines,
        autojunk=False,
    )
    for (
        tag,
        baseline_start,
        baseline_end,
        parent_start,
        parent_end,
    ) in matcher.get_opcodes():
        if tag == "equal":
            continue
        baseline_fragment = "".join(baseline_lines[baseline_start:baseline_end])
        parent_fragment = "".join(parent_lines[parent_start:parent_end])
        if parent_fragment and parent_fragment in child:
            continue
        if baseline_fragment and baseline_fragment in child:
            return True
        if tag == "insert" and parent_fragment:
            return True
    return False


def _candidate_repair_capsule(
    proposal: CodingAgentPatchProposal,
    runtime_record: Mapping[str, Any],
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    del workspace_root
    validation_evidence = tuple(
        dict(item)
        for item in tuple(runtime_record.get("validations", ()) or ())
        if isinstance(item, Mapping)
        and str(item.get("proposal_id") or "") == proposal.proposal_id
    )
    findings = tuple(
        item
        for item in tuple(runtime_record.get("review_findings", ()) or ())
        if isinstance(item, Mapping)
    )
    parent_candidate = {
        "proposal_id": proposal.proposal_id,
        "source": proposal.source,
        "model": proposal.model,
        "iteration": proposal.iteration,
        "artifact_id": proposal.artifact_id,
        "themes": proposal.themes,
        "patches": canonicalize(proposal.patches),
        "verification_commands": proposal.verification_commands,
        "verification_evidence": validation_evidence,
        "selected_files": proposal.selected_files,
        "rationale": proposal.rationale,
        "support_refs": proposal.support_refs,
        "raw_response_hash": proposal.raw_response_hash,
        "blocked_reason": proposal.blocked_reason,
        "stage": proposal.stage,
    }
    return {
        "parent_proposal_id": proposal.proposal_id,
        "parent_semantic_fingerprint": _candidate_semantic_fingerprint(proposal),
        "parent_candidate": parent_candidate,
        "review_findings": findings,
        "landing_reasons": tuple(runtime_record.get("landing_reasons", ()) or ()),
        "validation_failure_codes": tuple(
            runtime_record.get("validation_failure_codes", ()) or ()
        ),
        "required_followup": tuple(runtime_record.get("required_followup", ()) or ()),
        "instruction": (
            "Start from the parent candidate's validated production behavior. Make the "
            "smallest causal change needed to clear every listed finding. Preserve the "
            "parent source fix when it already passes evaluator-owned behavior probes. "
            "For added_tests_not_verified, either run an exact command naming every "
            "changed test or remove those unverified test edits; do not rewrite the "
            "production fix from scratch."
        ),
    }


def _bounded_sibling_candidate_evidence(
    proposals: tuple[CodingAgentPatchProposal, ...],
    *,
    workspace_root: Path,
    max_total_diff_chars: int = 18_000,
) -> tuple[dict[str, Any], ...]:
    remaining = max(0, max_total_diff_chars)
    evidence: list[dict[str, Any]] = []
    for proposal in tuple(item for item in proposals if not item.blocked_reason)[-2:]:
        rendered_diffs: list[str] = []
        for patch in proposal.patches:
            if remaining <= 0:
                break
            rendered = _bounded_workspace_patch_diff(
                workspace_root,
                patch,
                max_chars=min(8_000, remaining),
            )
            if rendered:
                rendered_diffs.append(rendered)
                remaining -= len(rendered)
        evidence.append(
            {
                "proposal_id": proposal.proposal_id,
                "rationale": proposal.rationale[:1_200],
                "changed_paths": tuple(
                    dict.fromkeys(patch.path for patch in proposal.patches)
                ),
                "verification_commands": proposal.verification_commands[:8],
                "support_refs": proposal.support_refs[:12],
                "bounded_diffs": tuple(rendered_diffs),
            }
        )
    return tuple(evidence)


def _bounded_workspace_patch_diff(
    workspace_root: Path,
    patch: WorkspaceFilePatch,
    *,
    max_chars: int,
) -> str:
    if max_chars <= 0:
        return ""
    try:
        before = read_regular_file_text(
            workspace_root,
            patch.path,
            max_bytes=2_000_000,
        )
    except (UnicodeDecodeError, UnsafeRegularFileError):
        before = ""
    if patch.operation == "create_or_replace":
        after = patch.content
    elif patch.operation == "replace_fragment":
        after = (
            before.replace(patch.old, patch.new, 1)
            if patch.old and patch.old in before
            else patch.new
        )
    elif patch.operation == "append_if_missing":
        separator = "" if not before or before.endswith("\n") else "\n"
        after = (
            before if patch.content in before else before + separator + patch.content
        )
    else:
        return ""
    rendered = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{patch.path}",
            tofile=f"b/{patch.path}",
            lineterm="",
        )
    )
    if len(rendered) <= max_chars:
        return rendered
    suffix = "\n... sibling diff truncated ..."
    return rendered[: max(0, max_chars - len(suffix))] + suffix


def _compose_disjoint_candidate_proposals(
    proposals: tuple[CodingAgentPatchProposal, ...],
    *,
    start_iteration: int,
    max_composites: int = 4,
) -> tuple[CodingAgentPatchProposal, ...]:
    viable = tuple(proposal for proposal in proposals if not proposal.blocked_reason)
    composites: list[CodingAgentPatchProposal] = []
    max_group_size = min(3, len(viable))
    for group_size in range(2, max_group_size + 1):
        for group in combinations(viable, group_size):
            path_sets = [
                {patch.path for patch in proposal.patches} for proposal in group
            ]
            if any(
                left & right
                for index, left in enumerate(path_sets)
                for right in path_sets[index + 1 :]
            ):
                continue
            source_patches = tuple(
                patch for proposal in group for patch in proposal.patches
            )
            if not source_patches or not any(
                _counts_as_product_patch_path(patch.path) for patch in source_patches
            ):
                continue
            total_patch_bytes = sum(
                len(value.encode("utf-8"))
                for patch in source_patches
                for value in (patch.content, patch.old, patch.new)
            )
            if len(source_patches) > 64 or total_patch_bytes > 500_000:
                continue
            parent_ids = tuple(proposal.proposal_id for proposal in group)
            composite_hash = stable_hash(
                {
                    "parent_ids": parent_ids,
                    "patches": source_patches,
                    "verification_commands": tuple(
                        command
                        for proposal in group
                        for command in proposal.verification_commands
                    ),
                }
            )
            patches = tuple(
                replace(
                    patch,
                    patch_id=f"composite_{composite_hash[:10]}_{index}",
                )
                for index, patch in enumerate(source_patches)
            )
            models = tuple(
                dict.fromkeys(
                    proposal.model for proposal in group if proposal.model is not None
                )
            )
            composites.append(
                CodingAgentPatchProposal(
                    proposal_id=f"composite_proposal_{composite_hash[:20]}",
                    source="candidate_composer",
                    model=models[0] if len(models) == 1 else None,
                    iteration=start_iteration + len(composites),
                    artifact_id=next(
                        (
                            proposal.artifact_id
                            for proposal in group
                            if proposal.artifact_id is not None
                        ),
                        None,
                    ),
                    themes=tuple(
                        dict.fromkeys(
                            theme for proposal in group for theme in proposal.themes
                        )
                    ),
                    patches=patches,
                    verification_commands=_dedupe(
                        command
                        for proposal in group
                        for command in proposal.verification_commands
                    ),
                    selected_files=tuple(
                        dict.fromkeys(
                            path
                            for proposal in group
                            for path in proposal.selected_files
                        )
                    ),
                    rationale=(
                        "Compose disjoint, publicly motivated candidate patches and "
                        "validate their combined contract independently. Parents: "
                        + ", ".join(parent_ids)
                    ),
                    support_refs=tuple(
                        dict.fromkeys(
                            ref for proposal in group for ref in proposal.support_refs
                        )
                    ),
                    raw_response_hash=composite_hash,
                    stage="promote",
                )
            )
            if len(composites) >= max_composites:
                return tuple(composites)
    return tuple(composites)


def _propose_with_execution_preflight(
    *,
    agent: WorkspaceCodingAgent,
    executor: CommandExecutor,
    source_report: Mapping[str, Any],
    workspace_root: Path,
    package: WorkspacePackageInfo,
    file_contexts: tuple[WorkspaceFileContext, ...],
    previous_verification: tuple[WorkspaceVerificationResult, ...],
    iteration: int,
    timeout_seconds: float,
) -> tuple[CodingAgentPatchProposal, str | None]:
    try:
        ensure_command_executor_ready(
            executor,
            root=workspace_root,
            timeout_seconds=min(max(timeout_seconds, 1.0), 30.0),
        )
    except RuntimeError as exc:
        reason = str(exc)[:300]
        raw_model = getattr(agent, "model", None)
        return (
            _blocked_proposal(
                source=str(getattr(agent, "source", "workspace_coding_agent")),
                model=str(raw_model) if raw_model is not None else None,
                iteration=iteration,
                artifact_id=_extract_artifact_id(source_report),
                themes=_extract_themes(source_report),
                selected_files=tuple(context.path for context in file_contexts),
                reason=reason,
            ),
            reason,
        )
    with CandidateWorkspace.create(workspace_root) as proposal_workspace:
        proposal = agent.propose(
            source_report=source_report,
            workspace_root=proposal_workspace.root,
            package=package,
            file_contexts=file_contexts,
            previous_verification=previous_verification,
            iteration=iteration,
        )
    return proposal, None


def _execution_preflight_runtime_record(
    *,
    agent: WorkspaceCodingAgent,
    proposal: CodingAgentPatchProposal,
    reason: str,
    iteration: int,
    candidate_index: int,
    strategy: str,
    stage: str,
) -> dict[str, Any]:
    return {
        "record_type": "agent_proposal",
        "stage": stage,
        "iteration": iteration,
        "candidate_index": candidate_index,
        "strategy": strategy,
        "proposal_id": proposal.proposal_id,
        "status": "infra_error",
        "request_count": 0,
        "response_count": 0,
        "tool_call_count": 0,
        "total_tokens": 0,
        "blocked_reason": reason,
        "trace_hash": stable_hash(
            {
                "proposal_id": proposal.proposal_id,
                "reason": reason,
                "status": "infra_error",
            }
        ),
        "max_turns": getattr(agent, "max_turns", None),
        "max_tool_calls": getattr(agent, "max_tool_calls", None),
        "max_patch_bytes": getattr(agent, "max_patch_bytes", None),
        "reasoning_effort": getattr(agent, "reasoning_effort", None),
        "requested_reasoning_effort": getattr(
            agent, "requested_reasoning_effort", None
        ),
    }


def _agent_proposal_runtime_record(
    *,
    agent: WorkspaceCodingAgent,
    proposal: CodingAgentPatchProposal,
    iteration: int,
    candidate_index: int,
    strategy: str,
    stage: str,
) -> dict[str, Any] | None:
    trace = getattr(agent, "last_trace", None)
    if trace is None:
        return None
    if isinstance(trace, Mapping):
        value = trace.get
    else:

        def value(key: str, default: Any = None) -> Any:
            return getattr(trace, key, default)

    response_ids = tuple(str(item) for item in (value("response_ids", ()) or ()))
    tool_calls = tuple(value("tool_calls", ()) or ())
    return {
        "record_type": "agent_proposal",
        "usage_scope_id": getattr(agent, "last_run_id", None),
        "stage": stage,
        "iteration": iteration,
        "candidate_index": candidate_index,
        "strategy": strategy,
        "proposal_id": proposal.proposal_id,
        "status": str(value("status", "unknown")),
        "request_count": int(
            value("request_count", len(response_ids)) or len(response_ids)
        ),
        "response_count": len(response_ids),
        "tool_call_count": len(tool_calls),
        "total_tokens": int(value("total_tokens", 0) or 0),
        "blocked_reason": value("blocked_reason"),
        "trace_hash": value("trace_hash"),
        "max_turns": getattr(agent, "max_turns", None),
        "max_tool_calls": getattr(agent, "max_tool_calls", None),
        "max_patch_bytes": getattr(agent, "max_patch_bytes", None),
        "reasoning_effort": getattr(agent, "reasoning_effort", None),
        "requested_reasoning_effort": getattr(
            agent, "requested_reasoning_effort", None
        ),
    }


def _validation_semantic_coverage(validation: Any) -> float:
    outcomes: list[bool] = []
    required_dimension_ids = tuple(
        getattr(validation, "required_dimension_ids", ()) or ()
    )
    dimension_status_by_id = dict(
        getattr(validation, "dimension_status_by_id", {}) or {}
    )
    outcomes.extend(
        str(dimension_status_by_id.get(dimension_id, "")).casefold() == "passed"
        for dimension_id in required_dimension_ids
    )
    cross_total = max(
        0,
        int(getattr(validation, "cross_candidate_contract_total", 0) or 0),
    )
    cross_passed = max(
        0,
        min(
            cross_total,
            int(getattr(validation, "cross_candidate_contract_pass_count", 0) or 0),
        ),
    )
    outcomes.extend(True for _ in range(cross_passed))
    outcomes.extend(False for _ in range(cross_total - cross_passed))
    candidate_verification_results = (
        getattr(validation, "candidate_verification_results", {}) or {}
    )
    if candidate_verification_results:
        outcomes.append(
            bool(getattr(validation, "candidate_verification_passed", False))
        )
    outcomes.append(
        bool(getattr(validation, "candidate_behavior_verification_passed", False))
    )
    outcomes.append(
        bool(getattr(validation, "public_contract_continuity_passed", True))
    )
    outcomes.append(bool(getattr(validation, "repro_passed", False)))
    return round(sum(outcomes) / len(outcomes), 6) if outcomes else 0.0


def _validation_failure_distance(validation: Any) -> int:
    distance = 0
    if not bool(getattr(validation, "applies_cleanly", False)):
        distance += 1
    if not bool(getattr(validation, "repro_passed", False)):
        distance += 1
    candidate_verification_results = (
        getattr(validation, "candidate_verification_results", {}) or {}
    )
    if candidate_verification_results and not bool(
        getattr(validation, "candidate_verification_passed", False)
    ):
        distance += 1
    if not bool(getattr(validation, "public_contract_continuity_passed", True)):
        distance += 1
    required_dimension_ids = tuple(
        getattr(validation, "required_dimension_ids", ()) or ()
    )
    dimension_status_by_id = dict(
        getattr(validation, "dimension_status_by_id", {}) or {}
    )
    distance += sum(
        1
        for dimension_id in required_dimension_ids
        if str(dimension_status_by_id.get(dimension_id, "")).casefold() != "passed"
    )
    added_test_status = str(
        getattr(validation, "added_test_verification_status", "not_applicable")
        or "not_applicable"
    )
    if added_test_status not in {"not_applicable", "passed"}:
        distance += 1
    if distance == 0 and getattr(validation, "failure_summary", None):
        distance = 1
    return distance


def _failed_candidate_repair_selection(
    *,
    validations: tuple[Any, ...],
    proposal_by_candidate: Mapping[str, CodingAgentPatchProposal],
) -> tuple[CodingAgentPatchProposal | None, dict[str, Any] | None]:
    ranked: list[
        tuple[
            tuple[float, float, int, int, int, int],
            CodingAgentPatchProposal,
            dict[str, Any],
        ]
    ] = []
    for order, validation in enumerate(validations):
        patch_id = str(getattr(validation, "patch_id", "") or "")
        proposal = proposal_by_candidate.get(patch_id)
        if proposal is None:
            continue
        try:
            score = float(getattr(validation, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        semantic_coverage = _validation_semantic_coverage(validation)
        failure_distance = _validation_failure_distance(validation)
        diff_size = max(0, int(getattr(validation, "diff_size", 0) or 0))
        rank = (
            score,
            semantic_coverage,
            -failure_distance,
            int(bool(getattr(validation, "applies_cleanly", False))),
            -diff_size,
            -order,
        )
        evidence = {
            "patch_id": patch_id,
            "proposal_id": proposal.proposal_id,
            "score": score,
            "semantic_coverage": semantic_coverage,
            "failure_distance": failure_distance,
            "applies_cleanly": bool(getattr(validation, "applies_cleanly", False)),
            "repro_passed": bool(getattr(validation, "repro_passed", False)),
            "candidate_behavior_verification_passed": bool(
                getattr(
                    validation,
                    "candidate_behavior_verification_passed",
                    False,
                )
            ),
            "dimension_coverage_complete": bool(
                getattr(validation, "dimension_coverage_complete", True)
            ),
            "effective_diff_size": diff_size,
            "ranking_policy": (
                "verification_score_then_semantic_coverage_then_failure_distance"
            ),
        }
        ranked.append((rank, proposal, evidence))
    if not ranked:
        return None, None
    _, selected, evidence = max(ranked, key=lambda item: item[0])
    return selected, evidence


def _select_multi_candidate_proposal(
    *,
    proposals: tuple[CodingAgentPatchProposal, ...],
    source_report: Mapping[str, Any],
    workspace_root: Path,
    task_specs: tuple[DevelopmentTaskSpecV2, ...],
    evaluator_commands: tuple[str, ...],
    project_commands: tuple[str, ...],
    strategies: tuple[str, ...],
    executor: CommandExecutor,
    timeout_seconds: float,
    evaluator_infrastructure_retry_limit: int = (
        DEFAULT_WORKSPACE_CODING_EVALUATOR_INFRA_RETRIES
    ),
) -> tuple[CodingAgentPatchProposal, dict[str, Any], tuple[str, ...]]:
    viable = tuple(proposal for proposal in proposals if not proposal.blocked_reason)
    if not viable:
        selected = proposals[0]
        return (
            selected,
            {
                "record_type": "candidate_selection",
                "status": "blocked",
                "reason": "all_candidate_proposals_blocked",
                "selected_proposal_id": selected.proposal_id,
            },
            tuple(proposal.proposal_id for proposal in proposals[1:]),
        )
    preflight_gate_issues = {
        proposal.proposal_id: _candidate_preflight_gate_issues(
            source_report=source_report,
            proposal=proposal,
        )
        if not proposal.blocked_reason
        else (f"proposal_blocked:{proposal.blocked_reason}",)
        for proposal in proposals
    }
    gate_ready = tuple(
        proposal
        for proposal in viable
        if not preflight_gate_issues[proposal.proposal_id]
    )
    gate_policy = _maintainer_gate_policy(source_report)
    if gate_policy["enabled"] and not gate_ready:
        selected_base = viable[0]
        selected_issues = preflight_gate_issues[selected_base.proposal_id]
        selected = replace(
            selected_base,
            blocked_reason=(
                "maintainer_preflight_gate_failed:" + ",".join(selected_issues)
            )[:300],
        )
        return (
            selected,
            {
                "record_type": "candidate_selection",
                "status": "needs_revision",
                "reason": "all_candidates_failed_maintainer_preflight",
                "selected_proposal_id": selected.proposal_id,
                "preflight_gate_issues": preflight_gate_issues,
                "gate_ready_candidate_count": 0,
                "runtime_candidate_count": 0,
            },
            tuple(
                proposal.proposal_id
                for proposal in proposals
                if proposal.proposal_id != selected.proposal_id
            ),
        )
    runtime_viable = gate_ready or viable
    task_id = task_specs[0].task_id if task_specs else "workspace_task"
    strategy_by_proposal_id = {
        proposal.proposal_id: (
            "disjoint_contract_composition"
            if proposal.source == "candidate_composer"
            else strategies[index % len(strategies)]
        )
        for index, proposal in enumerate(proposals)
    }
    candidates = tuple(
        candidate_from_proposal(
            proposal,
            task_id=task_id,
            strategy=strategy_by_proposal_id[proposal.proposal_id],
            localization_hypothesis_id="main_loop_localization",
            oracle_ids=(),
        )
        for proposal in runtime_viable
    )
    proposal_by_candidate = {
        candidate.patch_id: proposal
        for candidate, proposal in zip(candidates, runtime_viable, strict=True)
    }
    selection_inconsistent = False
    evaluator_errors: list[str] = []
    runtime = None
    evaluator_attempt_limit = 1 + max(0, evaluator_infrastructure_retry_limit)
    for _evaluator_attempt in range(evaluator_attempt_limit):
        try:
            runtime = run_code_max_runtime(
                source_report=source_report,
                workspace_root=workspace_root,
                task_specs=task_specs,
                candidate_patches=candidates,
                confirm_base=False,
                timeout_seconds=timeout_seconds,
                mode="max",
                executor=executor,
                oracle_commands=_dedupe((*evaluator_commands, *project_commands)),
            )
        except Exception as exc:
            evaluator_errors.append(type(exc).__name__)
            continue
        break
    if runtime is None:
        error_type = evaluator_errors[-1] if evaluator_errors else "UnknownError"
        selected = replace(
            runtime_viable[0],
            blocked_reason=f"code_max_runtime_infra_error:{error_type}",
        )
        record = {
            "record_type": "candidate_selection",
            "status": "infra_error",
            "reason": f"multi_candidate_runtime_failed:{error_type}",
            "selected_proposal_id": selected.proposal_id,
            "repair_seed_eligible": False,
            "evaluator_attempt_count": len(evaluator_errors),
            "evaluator_retry_count": max(0, len(evaluator_errors) - 1),
            "evaluator_error_types": tuple(evaluator_errors),
        }
    else:
        selected_patch_id = runtime.selection.selected_patch_id
        repair_seed_selection: dict[str, Any] | None = None
        if selected_patch_id in proposal_by_candidate:
            selected = proposal_by_candidate[selected_patch_id]
        elif selected_patch_id is None:
            selected, repair_seed_selection = _failed_candidate_repair_selection(
                validations=tuple(runtime.validations),
                proposal_by_candidate=proposal_by_candidate,
            )
            if selected is None:
                selected = runtime_viable[0]
                selection_inconsistent = True
        else:
            selected = runtime_viable[0]
            selection_inconsistent = True
        review_decision = getattr(runtime, "review_decision", None)
        review_findings = tuple(
            {
                "code": str(getattr(finding, "code", "unspecified_review_finding")),
                "role": str(getattr(finding, "reviewer_role", "")),
                "severity": str(getattr(finding, "severity", "")),
                "summary": str(getattr(finding, "summary", ""))[:500],
                "suggested_probe": getattr(finding, "suggested_probe", None),
                "suggested_patch_constraint": getattr(
                    finding,
                    "suggested_patch_constraint",
                    None,
                ),
                "remediation_owner": str(
                    getattr(finding, "remediation_owner", "candidate")
                ),
                "retryable": bool(getattr(finding, "retryable", True)),
                "details": tuple(getattr(finding, "details", ()) or ()),
            }
            for finding in tuple(getattr(review_decision, "findings", ()) or ())
        )
        record = {
            "record_type": "candidate_selection",
            "status": "completed",
            "runtime_id": runtime.runtime_id,
            "selected_patch_id": runtime.selection.selected_patch_id,
            "selected_proposal_id": selected.proposal_id,
            "selection_reason": runtime.selection.selection_reason,
            "landing_status": runtime.landing_decision.status,
            "claim_level": runtime.landing_decision.claim_level,
            "review_status": getattr(review_decision, "status", None),
            "review_findings": review_findings,
            "required_followup": tuple(
                getattr(runtime.selection, "required_followup", ()) or ()
            ),
            "repair_seed_selection": repair_seed_selection,
            "repair_seed_eligible": False,
            "evaluator_attempt_count": len(evaluator_errors) + 1,
            "evaluator_retry_count": len(evaluator_errors),
            "evaluator_error_types": tuple(evaluator_errors),
            "validations": tuple(
                {
                    "patch_id": validation.patch_id,
                    "proposal_id": proposal_by_candidate[
                        validation.patch_id
                    ].proposal_id,
                    "applies_cleanly": validation.applies_cleanly,
                    "repro_passed": validation.repro_passed,
                    "score": validation.score,
                    "failure_summary": validation.failure_summary,
                    "patch_results": canonicalize(
                        getattr(validation, "patch_results", ())
                    ),
                    "oracle_results": canonicalize(
                        getattr(validation, "oracle_results", {})
                    ),
                    "candidate_verification_results": canonicalize(
                        getattr(validation, "candidate_verification_results", {})
                    ),
                    "targeted_tests_passed": bool(
                        getattr(validation, "targeted_tests_passed", False)
                    ),
                    "regression_tests_passed": getattr(
                        validation,
                        "regression_tests_passed",
                        None,
                    ),
                    "evidence_hash": getattr(validation, "evidence_hash", None),
                    "candidate_verification_passed": getattr(
                        validation,
                        "candidate_verification_passed",
                        False,
                    ),
                    "candidate_behavior_verification_passed": getattr(
                        validation,
                        "candidate_behavior_verification_passed",
                        False,
                    ),
                    "added_test_verification_passed": getattr(
                        validation,
                        "added_test_verification_passed",
                        False,
                    ),
                    "added_test_verification_status": getattr(
                        validation,
                        "added_test_verification_status",
                        "not_applicable",
                    ),
                    "public_contract_continuity_passed": getattr(
                        validation,
                        "public_contract_continuity_passed",
                        True,
                    ),
                    "public_contract_continuity_issues": getattr(
                        validation,
                        "public_contract_continuity_issues",
                        (),
                    ),
                    "required_dimension_ids": getattr(
                        validation,
                        "required_dimension_ids",
                        (),
                    ),
                    "dimension_status_by_id": getattr(
                        validation,
                        "dimension_status_by_id",
                        {},
                    ),
                    "dimension_coverage_complete": getattr(
                        validation,
                        "dimension_coverage_complete",
                        True,
                    ),
                    "cross_candidate_contract_pass_count": getattr(
                        validation,
                        "cross_candidate_contract_pass_count",
                        0,
                    ),
                    "cross_candidate_contract_total": getattr(
                        validation,
                        "cross_candidate_contract_total",
                        0,
                    ),
                    "effective_diff_size": getattr(validation, "diff_size", 0),
                    "files_touched_count": getattr(
                        validation,
                        "files_touched_count",
                        len(proposal_by_candidate[validation.patch_id].patches),
                    ),
                }
                for validation in runtime.validations
                if validation.patch_id in proposal_by_candidate
            ),
        }
        accepted_landing_statuses = {
            "candidate_validated",
            "candidate_validated_with_claim_downgrade",
        }
        if selection_inconsistent:
            selected = replace(
                selected,
                blocked_reason=(
                    "code_max_selection_inconsistent:"
                    f"unknown_patch_id:{runtime.selection.selected_patch_id}"
                )[:300],
            )
            record["status"] = "infra_error"
            record["reason"] = "candidate_selection_result_inconsistent"
        elif selected_patch_id is None:
            selected_patch_ids = {
                patch_id
                for patch_id, proposal in proposal_by_candidate.items()
                if proposal.proposal_id == selected.proposal_id
            }
            selected_validations = tuple(
                validation
                for validation in runtime.validations
                if validation.patch_id in selected_patch_ids
            )
            failure_codes = _candidate_validation_failure_codes(selected_validations)
            all_failure_codes = _candidate_validation_failure_codes(
                tuple(runtime.validations)
            )
            reason_suffix = ",".join(failure_codes)
            selected = replace(
                selected,
                blocked_reason=(
                    "code_max_all_candidates_failed_validation:"
                    f"{reason_suffix or 'no_candidate_passed_required_validation'}"
                )[:300],
            )
            record["status"] = "needs_revision"
            record["reason"] = "all_candidates_failed_validation"
            record["validation_failure_codes"] = failure_codes
            record["all_candidate_validation_failure_codes"] = all_failure_codes
            record["repair_seed_eligible"] = True
        elif runtime.landing_decision.status not in accepted_landing_statuses:
            landing_reasons = tuple(
                str(reason)
                for reason in getattr(runtime.landing_decision, "reasons", ())
            )
            reason_suffix = ",".join(landing_reasons) or "unspecified_review_failure"
            selected = replace(
                selected,
                blocked_reason=(
                    f"code_max_landing_needs_revision:{reason_suffix}"[:300]
                ),
            )
            record["status"] = "needs_revision"
            record["landing_reasons"] = landing_reasons
            record["repair_seed_eligible"] = True
    record["preflight_gate_issues"] = preflight_gate_issues
    record["gate_ready_candidate_count"] = len(gate_ready)
    record["runtime_candidate_count"] = len(runtime_viable)
    rejected = tuple(
        proposal.proposal_id
        for proposal in proposals
        if proposal.proposal_id != selected.proposal_id
    )
    return selected, record, rejected


def _candidate_validation_failure_codes(
    validations: tuple[Any, ...],
) -> tuple[str, ...]:
    codes: list[str] = []
    for validation in validations:
        failure_summary = str(getattr(validation, "failure_summary", "") or "")
        if failure_summary:
            codes.append(failure_summary)
        if not bool(getattr(validation, "applies_cleanly", False)):
            codes.append("patch_did_not_apply")
        if not bool(getattr(validation, "repro_passed", False)):
            codes.append("behavior_oracle_failed")
        verification_results = getattr(
            validation,
            "candidate_verification_results",
            {},
        )
        if verification_results and not bool(
            getattr(validation, "candidate_verification_passed", False)
        ):
            codes.append("candidate_verification_failed")
        if not bool(getattr(validation, "dimension_coverage_complete", True)):
            codes.append("required_contract_dimension_failed")
    return tuple(dict.fromkeys(codes))


def _candidate_preflight_gate_issues(
    *,
    source_report: Mapping[str, Any],
    proposal: CodingAgentPatchProposal,
) -> tuple[str, ...]:
    policy = _maintainer_gate_policy(source_report)
    product_paths = tuple(
        dict.fromkeys(
            patch.path
            for patch in proposal.patches
            if patch.path and _counts_as_product_patch_path(patch.path)
        )
    )
    if policy["enabled"] and not product_paths:
        return ("no_product_file_change",)

    issues: list[str] = []
    if policy["enabled"]:
        issues.extend(
            _proposal_theme_alignment_issues(
                source_report=source_report,
                proposal=proposal,
            )
        )
    specs = build_development_intent_specs(source_report)
    matched_specs = _matched_intent_specs(
        specs=specs,
        proposal=proposal,
        changed_paths=product_paths,
    )
    if policy["enabled"] and policy["require_intent_trace"] and not matched_specs:
        issues.append("no_changed_path_or_patch_evidence_matches_development_intent")

    behavior_commands = tuple(
        command
        for command in proposal.verification_commands
        if _is_behavior_verification_command(command)
    )
    source_paths = tuple(
        path for path in product_paths if _is_source_or_config_path(path)
    )
    if (
        policy["enabled"]
        and policy["require_behavior_verification"]
        and source_paths
        and not behavior_commands
    ):
        issues.append("missing_behavior_verification_command")

    acceptance_commands = {
        _shell_command_identity(command)
        for command in _executable_acceptance_commands(specs)
    }
    test_or_repro_paths = tuple(
        path for path in product_paths if _is_test_or_repro_path(path)
    )
    test_or_repro_commands = tuple(
        command
        for command in behavior_commands
        if _command_mentions_test_or_repro(command)
        or _shell_command_identity(command) in acceptance_commands
    )
    if (
        policy["enabled"]
        and policy["require_test_or_repro"]
        and source_paths
        and not test_or_repro_paths
        and not test_or_repro_commands
    ):
        issues.append("missing_test_or_repro_artifact")

    broad_source_rewrite = any(
        patch.operation == "create_or_replace"
        and _is_source_or_config_path(patch.path)
        and not _is_test_or_repro_path(patch.path)
        for patch in proposal.patches
    )
    if policy["enabled"] and broad_source_rewrite and not behavior_commands:
        issues.append("broad_source_rewrite_without_behavior_review")

    if policy["enabled"]:
        for path in product_paths:
            if _coverage_requirement(path) != "project_verification":
                continue
            if not _proposal_verification_covers_path(
                path,
                proposals=(proposal,),
                commands=proposal.verification_commands,
            ):
                issues.append(f"uncovered_project_path:{path}")
    return tuple(dict.fromkeys(issues))


def _executable_acceptance_commands(
    specs: tuple[Mapping[str, Any], ...],
) -> tuple[str, ...]:
    commands: list[str] = []
    for index, spec in enumerate(specs):
        commands.extend(
            oracle.command
            for oracle in _acceptance_oracles_for_spec(spec, index)
            if oracle.command
        )
    return tuple(dict.fromkeys(commands))


def _shell_command_identity(command: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(command))
    except ValueError:
        return (command.strip(),)


def _reproducer_base_status(status: str) -> str:
    return {
        "confirmed": "failed",
        "not_reproduced": "passed",
        "invalid": "blocked",
        "unconfirmed": "not_run",
    }.get(status, "not_run")


def _verification_evidence_records(
    *,
    task_id: str,
    results: tuple[WorkspaceVerificationResult, ...],
    evaluator_commands: tuple[str, ...],
    project_test_commands: tuple[str, ...],
    project_lint_commands: tuple[str, ...],
    project_build_commands: tuple[str, ...],
    generated_commands: tuple[str, ...],
    baseline_status_by_command: Mapping[str, str],
    baseline_repo_digest: str,
    candidate_repo_digest: str,
    execution_policy_hash: str,
) -> tuple[VerificationEvidence, ...]:
    evaluator = set(evaluator_commands)
    project_tests = set(project_test_commands)
    project_lint = set(project_lint_commands)
    project_build = set(project_build_commands)
    generated = set(generated_commands)
    records: list[VerificationEvidence] = []
    for result in results:
        if result.command in evaluator:
            owner = "evaluator"
            kind = (
                "syntax" if _known_syntax_only_command(result.command) else "behavior"
            )
        elif result.command in project_tests:
            owner = "project"
            kind = "test_suite"
        elif result.command in project_lint:
            owner = "project"
            kind = "lint"
        elif result.command in project_build:
            owner = "project"
            kind = "build"
        elif result.command in generated:
            owner = "generated"
            kind = "syntax"
        else:
            owner = "agent"
            kind = (
                "behavior"
                if _is_behavior_verification_command(result.command)
                else "support"
            )
        records.append(
            VerificationEvidence.create(
                task_id=task_id,
                command=result.command,
                owner=owner,
                kind=kind,
                base_status=baseline_status_by_command.get(result.command, "not_run"),
                candidate_status=(
                    result.status
                    if result.status
                    in {"passed", "failed", "blocked", "timeout", "infra_error"}
                    else "infra_error"
                ),
                required=owner != "agent",
                baseline_repo_digest=baseline_repo_digest,
                candidate_repo_digest=candidate_repo_digest,
                execution_policy_hash=execution_policy_hash,
                exit_code=result.exit_code,
                stdout_hash=result.stdout_hash,
                stderr_hash=result.stderr_hash,
            )
        )
    return tuple(records)


def _known_syntax_only_command(command: str) -> bool:
    lowered = command.lower()
    return any(
        marker in lowered
        for marker in (
            "py_compile",
            "compileall",
            "node --check",
            "json.tool",
            "json.parse(",
            "tsc --noemit",
        )
    )


def _is_project_test_suite_command(command: str) -> bool:
    try:
        argv = tuple(shlex.split(command))
    except ValueError:
        return False
    if not argv:
        return False
    lowered = tuple(part.casefold().replace("\\", "/") for part in argv)
    if "-c" in lowered:
        return False
    rendered = " ".join(lowered)
    if any(
        marker in rendered
        for marker in (
            "pytest",
            "unittest",
            "tox",
            "nox",
            "nosetests",
            "npm test",
            "npm run test",
            "pnpm test",
            "yarn test",
            "cargo test",
            "go test",
            "dotnet test",
        )
    ):
        return True
    return any(
        part.startswith(("test/", "tests/"))
        or "/test/" in f"/{part}"
        or Path(part).name.startswith("test_")
        for part in lowered[1:]
    )


def _has_effective_patch(patch_results: tuple[WorkspacePatchResult, ...]) -> bool:
    return any(
        result.status in {"applied", "dry_run"}
        and result.before_hash != result.after_hash
        for result in patch_results
    )


def _patch_coverage_commands(
    patch_results: tuple[WorkspacePatchResult, ...],
    *,
    executor: CommandExecutor,
    available_executables: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    commands: list[str] = []
    for path in _changed_patch_paths(patch_results):
        command = _coverage_command_for_path(
            path,
            executor=executor,
            available_executables=available_executables,
        )
        if command:
            commands.append(command)
    return _dedupe(commands)


def _patch_coverage_gate(
    *,
    patch_results: tuple[WorkspacePatchResult, ...],
    verification_results: tuple[WorkspaceVerificationResult, ...],
    generated_coverage_commands: tuple[str, ...],
    proposal: CodingAgentPatchProposal | None = None,
) -> WorkspaceVerificationResult | None:
    summary = _patch_coverage_summary(
        patch_results=patch_results,
        verification_results=verification_results,
        generated_coverage_commands=generated_coverage_commands,
        proposals=(proposal,) if proposal is not None else (),
    )
    uncovered = tuple(summary.get("uncovered_required_paths", ()))
    if not uncovered:
        return None
    return WorkspaceVerificationResult(
        command="patch_coverage_gate",
        status="failed",
        exit_code=None,
        elapsed_sec=0.0,
        stdout_tail="",
        stderr_tail=f"uncovered_required_paths={','.join(uncovered)}",
        blocked_reason="patch_verification_did_not_cover_changed_files",
    )


def _maintainer_landing_gate(
    *,
    source_report: Mapping[str, Any],
    proposal: CodingAgentPatchProposal,
    patch_results: tuple[WorkspacePatchResult, ...],
    verification_results: tuple[WorkspaceVerificationResult, ...],
) -> WorkspaceVerificationResult | None:
    policy = _maintainer_gate_policy(source_report)
    if not policy["enabled"]:
        return None
    issues = _maintainer_gate_issues(
        source_report=source_report,
        proposal=proposal,
        patch_results=patch_results,
        verification_results=verification_results,
        policy=policy,
    )
    if not issues:
        return None
    return WorkspaceVerificationResult(
        command="maintainer_landing_gate",
        status="failed",
        exit_code=None,
        elapsed_sec=0.0,
        stdout_tail=";".join(f"{key}={value}" for key, value in policy.items()),
        stderr_tail=";".join(issues),
        blocked_reason="maintainer_landing_gate_failed",
    )


def _maintainer_gate_issues(
    *,
    source_report: Mapping[str, Any],
    proposal: CodingAgentPatchProposal,
    patch_results: tuple[WorkspacePatchResult, ...],
    verification_results: tuple[WorkspaceVerificationResult, ...],
    policy: Mapping[str, Any],
) -> tuple[str, ...]:
    specs = build_development_intent_specs(source_report)
    changed_paths = _changed_patch_paths(patch_results)
    product_paths = tuple(
        path for path in changed_paths if _counts_as_product_patch_path(path)
    )
    if not product_paths:
        return ("no_product_file_change",)

    issues: list[str] = []
    issues.extend(
        _proposal_theme_alignment_issues(
            source_report=source_report,
            proposal=proposal,
        )
    )
    matched_specs = _matched_intent_specs(
        specs=specs,
        proposal=proposal,
        changed_paths=product_paths,
    )
    if policy.get("require_intent_trace") and not matched_specs:
        issues.append("no_changed_path_or_patch_evidence_matches_development_intent")

    generated_commands = {
        result.command
        for result in verification_results
        if _known_syntax_only_command(result.command)
    }
    passed_commands = tuple(
        result.command
        for result in verification_results
        if result.status == "passed" and result.command not in generated_commands
    )
    behavior_commands = tuple(
        command
        for command in passed_commands
        if _is_behavior_verification_command(command)
    )
    source_paths = tuple(
        path for path in product_paths if _is_source_or_config_path(path)
    )
    if (
        policy.get("require_behavior_verification")
        and source_paths
        and not behavior_commands
    ):
        issues.append("missing_behavior_verification_command")

    acceptance_commands = {
        _shell_command_identity(command)
        for command in _executable_acceptance_commands(specs)
    }
    test_or_repro_paths = tuple(
        path for path in product_paths if _is_test_or_repro_path(path)
    )
    test_or_repro_commands = tuple(
        command
        for command in behavior_commands
        if _command_mentions_test_or_repro(command)
        or _shell_command_identity(command) in acceptance_commands
    )
    if (
        policy.get("require_test_or_repro")
        and source_paths
        and not test_or_repro_paths
        and not test_or_repro_commands
    ):
        issues.append("missing_test_or_repro_artifact")

    broad_source_rewrites = tuple(
        patch.path
        for proposal_patch_result in patch_results
        for patch in proposal.patches
        if patch.patch_id == proposal_patch_result.patch_id
        and proposal_patch_result.status == "applied"
        and patch.operation == "create_or_replace"
        and _is_source_or_config_path(patch.path)
        and not _is_test_or_repro_path(patch.path)
    )
    if broad_source_rewrites and not behavior_commands:
        issues.append("broad_source_rewrite_without_behavior_review")

    return tuple(dict.fromkeys(issues))


def _matched_intent_specs(
    *,
    specs: tuple[dict[str, Any], ...],
    proposal: CodingAgentPatchProposal,
    changed_paths: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    proposal_themes = set(proposal.themes)
    matched: list[dict[str, Any]] = []
    for spec in specs:
        theme = str(spec.get("theme", ""))
        if not theme or theme not in proposal_themes:
            continue
        if spec.get("_generated_intent_spec") and changed_paths:
            matched.append(spec)
            continue
        hints = tuple(str(path) for path in spec.get("candidate_path_hints", ()))
        path_matches = any(
            _path_matches_hint(path, hint) for path in changed_paths for hint in hints
        )
        content_matches = any(
            _patch_content_matches_intent_spec(patch, spec)
            for patch in proposal.patches
            if patch.path in changed_paths
        )
        if path_matches or content_matches:
            matched.append(spec)
    return tuple(matched)


def _proposal_theme_alignment_issues(
    *,
    source_report: Mapping[str, Any],
    proposal: CodingAgentPatchProposal,
) -> tuple[str, ...]:
    allowed_themes = set(_extract_themes(source_report))
    proposed_themes = tuple(dict.fromkeys(proposal.themes))
    if not proposed_themes:
        return ("proposal_theme_missing",)
    unadopted = tuple(theme for theme in proposed_themes if theme not in allowed_themes)
    if not unadopted:
        return ()
    return ("proposal_theme_not_adopted:" + ",".join(unadopted),)


def _patch_content_matches_intent_spec(
    patch: WorkspaceFilePatch,
    spec: Mapping[str, Any],
) -> bool:
    haystack = _normalize_theme_evidence_text(
        "\n".join((patch.content, patch.old, patch.new, patch.rationale))
    )
    if not haystack:
        return False
    theme = str(spec.get("theme") or "")
    normalized_theme = _normalize_theme_evidence_text(theme)
    if normalized_theme and normalized_theme in haystack:
        return True
    raw_terms = list(THEME_KEYWORDS.get(theme, ()))
    raw_terms.extend(part for part in theme.split("_") if len(part) >= 5)
    raw_terms.extend(
        Path(str(hint)).stem for hint in spec.get("candidate_path_hints", ())
    )
    raw_terms.extend(str(item) for item in spec.get("relevant_symbols_hint", ()))
    terms = {
        normalized
        for raw_term in raw_terms
        if len(normalized := _normalize_theme_evidence_text(raw_term)) >= 4
    }
    return sum(term in haystack for term in terms) >= 2


def _path_matches_hint(path: str, hint: str) -> bool:
    path = Path(path).as_posix().lower()
    hint = Path(hint).as_posix().lower()
    if not hint:
        return False
    return path == hint or path.endswith(f"/{hint}") or hint in path


def _patch_coverage_summary(
    *,
    patch_results: tuple[WorkspacePatchResult, ...],
    verification_results: tuple[WorkspaceVerificationResult, ...],
    generated_coverage_commands: tuple[str, ...],
    proposals: tuple[CodingAgentPatchProposal, ...] = (),
) -> dict[str, Any]:
    changed_paths = _changed_patch_paths(patch_results)
    passed_commands = tuple(
        dict.fromkeys(
            result.command
            for result in verification_results
            if result.status == "passed"
        )
    )
    generated_set = set(generated_coverage_commands)
    passed_generated = tuple(
        command for command in passed_commands if command in generated_set
    )
    non_generated_passed = tuple(
        command for command in passed_commands if command not in generated_set
    )
    generated_by_path = {
        path: command
        for path in changed_paths
        for command in generated_coverage_commands
        if path in _paths_from_command(command)
    }
    runtime_requirement_by_path = {
        path: (
            "not_required"
            if _coverage_requirement(path) == "not_required"
            else "syntax"
            if path in generated_by_path
            else "project_verification"
        )
        for path in changed_paths
    }
    syntax_covered_paths = tuple(
        path
        for path, command in generated_by_path.items()
        if command in passed_generated
    )
    coverage_not_required_paths = tuple(
        path
        for path in changed_paths
        if runtime_requirement_by_path[path] == "not_required"
    )
    project_verified_paths = tuple(
        path
        for path in changed_paths
        if runtime_requirement_by_path[path] == "project_verification"
        and _proposal_verification_covers_path(
            path,
            proposals=proposals,
            commands=non_generated_passed,
        )
    )
    behavior_covered_paths = tuple(
        path
        for path in changed_paths
        if _proposal_verification_covers_path(
            path,
            proposals=proposals,
            commands=non_generated_passed,
        )
    )
    covered = (
        set(syntax_covered_paths)
        | set(coverage_not_required_paths)
        | set(project_verified_paths)
    )
    uncovered = tuple(path for path in changed_paths if path not in covered)
    maintainer_gate_failures = tuple(
        result
        for result in verification_results
        if result.command == "maintainer_landing_gate" and result.status != "passed"
    )
    maintainer_gate_issues = tuple(
        issue
        for result in maintainer_gate_failures
        for issue in (result.stderr_tail or "").split(";")
        if issue
    )
    return {
        "changed_paths": changed_paths,
        "generated_coverage_commands": generated_coverage_commands,
        "passed_generated_coverage_commands": passed_generated,
        "non_generated_passed_commands": non_generated_passed,
        "syntax_covered_paths": syntax_covered_paths,
        "behavior_covered_paths": behavior_covered_paths,
        "project_verified_paths": project_verified_paths,
        "coverage_not_required_paths": coverage_not_required_paths,
        "uncovered_required_paths": uncovered,
        "patch_coverage_ready": not uncovered,
        "behavior_verification_ready": bool(behavior_covered_paths),
        "maintainer_gate_ready": not maintainer_gate_failures,
        "maintainer_gate_issues": maintainer_gate_issues,
    }


def _changed_patch_paths(
    patch_results: tuple[WorkspacePatchResult, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            result.path
            for result in patch_results
            if result.status == "applied"
            and bool(result.path)
            and result.before_hash != result.after_hash
        )
    )


def _coverage_command_for_path(
    path: str,
    *,
    executor: CommandExecutor | None = None,
    available_executables: tuple[str, ...] | None = None,
) -> str | None:
    suffix = Path(path).suffix.lower()
    quoted_path = shlex.quote(path)
    if available_executables is None:
        python_executable = (
            "python"
            if executor is not None and executor.policy.backend != "local"
            else shlex.quote(sys.executable)
        )
        node_available = True
    else:
        python_executable = next(
            (
                executable
                for executable in ("python", "python3")
                if executable in available_executables
            ),
            None,
        )
        node_available = "node" in available_executables
    if suffix == ".py":
        return (
            f"{python_executable} -m py_compile {quoted_path}"
            if python_executable
            else None
        )
    if suffix in {".js", ".mjs", ".cjs"}:
        return f"node --check {quoted_path}" if node_available else None
    if suffix == ".json":
        if node_available:
            code = "JSON.parse(require('fs').readFileSync(process.argv[1],'utf8'))"
            return f"node -e {shlex.quote(code)} {quoted_path}"
        return (
            f"{python_executable} -m json.tool {quoted_path}"
            if python_executable
            else None
        )
    if suffix == ".toml":
        if not python_executable:
            return None
        code = (
            f"import pathlib,tomllib; tomllib.loads(pathlib.Path({path!r}).read_text())"
        )
        return f"{python_executable} -c {shlex.quote(code)}"
    return None


def _coverage_requirement(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {
        ".md",
        ".mdx",
        ".txt",
        ".rst",
        ".yml",
        ".yaml",
        ".css",
        ".html",
    }:
        return "not_required"
    if _coverage_command_for_path(path):
        return "syntax"
    return "project_verification"


def _is_source_or_config_path(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    if suffix in {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
    }:
        return True
    return Path(path).name in {"package.json", "pyproject.toml"}


def _is_test_or_repro_path(path: str) -> bool:
    normalized = Path(path).as_posix().lower()
    name = Path(normalized).name
    return (
        "/test" in normalized
        or "/tests" in normalized
        or normalized.startswith("test")
        or normalized.startswith("tests/")
        or "repro" in normalized
        or ".test." in name
        or ".spec." in name
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _is_behavior_verification_command(command: str) -> bool:
    if _known_syntax_only_command(command):
        return False
    return is_behavior_check_command(command)


def _command_mentions_test_or_repro(command: str) -> bool:
    lower = command.lower()
    if any(
        marker in lower
        for marker in (
            "pytest",
            "npm test",
            "npm run test",
            "pnpm test",
            "pnpm run test",
            "yarn test",
            "yarn run test",
            "vitest",
            "jest",
            "mocha",
            "ava",
            "tap",
        )
    ):
        return True
    if " assert " in f" {lower} ":
        return True
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for token in tokens:
        normalized = token.lower().replace("\\", "/").strip("'\"")
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


def _non_generated_command_covers_path(
    path: str,
    commands: tuple[str, ...],
) -> bool:
    if not commands:
        return False
    name = Path(path).name
    return any(
        path in command or name in command or command_covers_declaration(command, path)
        for command in commands
    )


def _proposal_verification_covers_path(
    path: str,
    *,
    proposals: tuple[CodingAgentPatchProposal, ...],
    commands: tuple[str, ...],
) -> bool:
    if _non_generated_command_covers_path(path, commands):
        return True
    normalized_path = Path(path).as_posix().casefold()
    target_name = Path(normalized_path).name
    for proposal in proposals:
        for patch in proposal.patches:
            if not _is_test_or_repro_path(patch.path):
                continue
            if not _non_generated_command_covers_path(patch.path, commands):
                continue
            test_content = (patch.content or patch.new).replace("\\", "/").casefold()
            if normalized_path in test_content or target_name in test_content:
                return True
    return False


def _normalized_reasoning_effort(requested: str | None, *, model: str | None) -> str:
    requested = (requested or DEFAULT_WORKSPACE_CODING_REASONING_EFFORT).lower()
    order = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}
    if requested not in order:
        requested = DEFAULT_WORKSPACE_CODING_REASONING_EFFORT
    if not str(model or "").startswith("gpt-5"):
        return requested
    if os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_ALLOW_LOW_REASONING") == "1":
        return requested
    minimum = DEFAULT_WORKSPACE_CODING_REASONING_EFFORT
    if order[requested] < order[MIN_WORKSPACE_CODING_REASONING_EFFORT]:
        return minimum
    if requested == "high" and str(model or "").startswith("gpt-5.5"):
        return minimum
    return requested


def _agent_file_contexts(
    file_contexts: tuple[WorkspaceFileContext, ...],
) -> tuple[WorkspaceFileContext, ...]:
    default_max_chars = _workspace_coding_max_excerpt_chars()
    retry_max_chars = _workspace_coding_retry_excerpt_chars()
    file_limit = _workspace_coding_prompt_file_limit()
    contexts: list[WorkspaceFileContext] = []
    for context in file_contexts[:file_limit]:
        max_chars = (
            retry_max_chars
            if context.context_role == "retry_failure"
            else default_max_chars
        )
        contexts.append(
            replace(
                context,
                excerpt=_compact_excerpt(context.excerpt, max_chars=max_chars),
                truncated=context.truncated or len(context.excerpt) > max_chars,
            )
        )
    return tuple(contexts)


def _agent_intent_specs_summary(
    specs: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "intent_id": spec.get("intent_id"),
            "theme": spec.get("theme"),
            "user_pain": spec.get("user_pain"),
            "observed_behavior": spec.get("observed_behavior"),
            "expected_behavior": spec.get("expected_behavior"),
            "candidate_path_hints": spec.get("candidate_path_hints", ()),
            "acceptance_tests": spec.get("acceptance_tests", ()),
            "support_refs": spec.get("support_refs", ()),
            "evidence_refs": spec.get("evidence_refs", ()),
            "contract_dimensions": spec.get("contract_dimensions", ()),
            "contract_witnesses": spec.get("contract_witnesses", ()),
            "boundary_conditions": spec.get("boundary_conditions", ()),
        }
        for spec in specs
    )


def _agent_task_specs_summary(
    task_specs: tuple[Any, ...],
) -> tuple[dict[str, Any], ...]:
    summaries: list[dict[str, Any]] = []
    for spec in task_specs:

        def value(name: str, default: Any = None) -> Any:
            if isinstance(spec, Mapping):
                return spec.get(name, default)
            return getattr(spec, name, default)

        acceptance_oracles = tuple(
            {
                "oracle_id": (
                    oracle.get("oracle_id")
                    if isinstance(oracle, Mapping)
                    else getattr(oracle, "oracle_id", None)
                ),
                "kind": (
                    oracle.get("kind")
                    if isinstance(oracle, Mapping)
                    else getattr(oracle, "kind", None)
                ),
                "command": (
                    oracle.get("command")
                    if isinstance(oracle, Mapping)
                    else getattr(oracle, "command", None)
                ),
                "description": (
                    oracle.get("description")
                    if isinstance(oracle, Mapping)
                    else getattr(oracle, "description", None)
                ),
                "source": (
                    oracle.get("source")
                    if isinstance(oracle, Mapping)
                    else getattr(oracle, "source", None)
                ),
                "dimension_ids": (
                    oracle.get("dimension_ids", ())
                    if isinstance(oracle, Mapping)
                    else getattr(oracle, "dimension_ids", ())
                ),
                "witness_role": (
                    oracle.get("witness_role")
                    if isinstance(oracle, Mapping)
                    else getattr(oracle, "witness_role", None)
                ),
                "expected_on_base": (
                    oracle.get("expected_on_base")
                    if isinstance(oracle, Mapping)
                    else getattr(oracle, "expected_on_base", None)
                ),
                "expected_on_patch": (
                    oracle.get("expected_on_patch")
                    if isinstance(oracle, Mapping)
                    else getattr(oracle, "expected_on_patch", None)
                ),
                "required": (
                    oracle.get("required", True)
                    if isinstance(oracle, Mapping)
                    else getattr(oracle, "required", True)
                ),
                "evidence_hash": (
                    oracle.get("evidence_hash")
                    if isinstance(oracle, Mapping)
                    else getattr(oracle, "evidence_hash", None)
                ),
            }
            for oracle in tuple(value("acceptance_oracles", ()))
        )
        summaries.append(
            {
                "task_id": value("task_id"),
                "source_theme": value("source_theme"),
                "task_type": value("task_type"),
                "public_evidence_refs": value("public_evidence_refs", ()),
                "user_pain": value("user_pain"),
                "observed_behavior": value("observed_behavior"),
                "expected_behavior": value("expected_behavior"),
                "non_goals": value("non_goals", ()),
                "reproduction_recipe": value("reproduction_recipe", ()),
                "acceptance_oracles": acceptance_oracles,
                "candidate_path_hints": value("candidate_path_hints", ()),
                "relevant_symbols_hint": value("relevant_symbols_hint", ()),
                "spec_maturity": value("spec_maturity"),
                "risk_level": value("risk_level"),
                "behavior_surface": value("behavior_surface", ()),
                "contract_dimensions": value("contract_dimensions", ()),
                "boundary_conditions": value("boundary_conditions", ()),
            }
        )
    return tuple(summaries)


def _agent_repo_intelligence_summary(repo: Mapping[str, Any]) -> dict[str, Any]:
    symbol_index = repo.get("symbol_index", {})
    test_map = repo.get("test_map", {})
    return {
        "repo_hash": repo.get("repo_hash"),
        "file_count": repo.get("file_count"),
        "package_manager": repo.get("package_manager"),
        "test_commands": repo.get("test_commands", ()),
        "lint_commands": repo.get("lint_commands", ()),
        "entrypoints": repo.get("entrypoints", ()),
        "source_roots": repo.get("source_roots", ()),
        "test_files": tuple(repo.get("test_files", ()))[:40],
        "config_files": tuple(repo.get("config_files", ()))[:40],
        "docs_files": tuple(repo.get("docs_files", ()))[:20],
        "symbol_index": {
            str(symbol): paths for symbol, paths in tuple(symbol_index.items())[:40]
        }
        if isinstance(symbol_index, Mapping)
        else {},
        "test_map": {
            str(source): tests for source, tests in tuple(test_map.items())[:40]
        }
        if isinstance(test_map, Mapping)
        else {},
        "candidate_files_by_theme": {
            str(theme): tuple(files)[:10]
            for theme, files in repo.get("candidate_files_by_theme", {}).items()
        }
        if isinstance(repo.get("candidate_files_by_theme"), Mapping)
        else {},
        "style_notes": repo.get("style_notes", ()),
    }


def _agent_code_max_runtime_summary(runtime: Mapping[str, Any]) -> dict[str, Any]:
    environment = runtime.get("environment", {})
    repo = runtime.get("repo", {})
    return {
        "runtime_id": runtime.get("runtime_id"),
        "mode": runtime.get("mode"),
        "environment": {
            "repo_hash": _mapping_get(environment, "repo_hash"),
            "setup_status": _mapping_get(environment, "setup_status"),
            "blocked_reason": _mapping_get(environment, "blocked_reason"),
            "language_stack": _mapping_get(environment, "language_stack", ()),
            "package_manager": _mapping_get(environment, "package_manager"),
            "install_commands": _mapping_get(environment, "install_commands", ()),
            "missing_declared_dependencies": _mapping_get(
                environment,
                "missing_declared_dependencies",
                (),
            ),
            "project_verification_status": _mapping_get(
                environment,
                "project_verification_status",
                "not_assessed",
            ),
            "available_verification_executables": _mapping_get(
                environment,
                "available_verification_executables",
                (),
            ),
            "eligible_project_verification_commands": _mapping_get(
                environment,
                "eligible_project_verification_commands",
                (),
            ),
            "discovered_test_commands": _mapping_get(
                environment, "discovered_test_commands", ()
            ),
            "discovered_lint_commands": _mapping_get(
                environment, "discovered_lint_commands", ()
            ),
            "discovered_build_commands": _mapping_get(
                environment, "discovered_build_commands", ()
            ),
            "project_command_baseline": _agent_project_command_baseline_summary(
                _mapping_get(environment, "project_command_baseline", {})
            ),
            "smoke_commands": _mapping_get(environment, "smoke_commands", ()),
        },
        "repo": {
            "repo_hash": _mapping_get(repo, "repo_hash"),
            "file_count": _mapping_get(repo, "file_count"),
            "source_roots": _mapping_get(repo, "source_roots", ()),
            "test_roots": _mapping_get(repo, "test_roots", ()),
            "entrypoints": _mapping_get(repo, "entrypoints", ()),
            "candidate_files_by_theme": {
                str(theme): tuple(files)[:10]
                for theme, files in _mapping_get(
                    repo, "candidate_files_by_theme", {}
                ).items()
            }
            if isinstance(_mapping_get(repo, "candidate_files_by_theme", {}), Mapping)
            else {},
            "impact_context": tuple(_mapping_get(repo, "impact_context", ()))[:24],
        },
        "oracle_summary": tuple(
            {
                "oracle_id": oracle.get("oracle_id"),
                "task_id": oracle.get("task_id"),
                "kind": oracle.get("kind"),
                "command": oracle.get("command"),
                "base_status": oracle.get("base_status"),
                "confidence": oracle.get("confidence"),
            }
            for oracle in tuple(runtime.get("oracle_summary", ()))[:12]
            if isinstance(oracle, Mapping)
        ),
        "localization_summary": tuple(
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "task_id": item.get("task_id"),
                "candidate_files": tuple(item.get("candidate_files", ()))[:10],
                "candidate_symbols": tuple(item.get("candidate_symbols", ()))[:12],
                "confidence": item.get("confidence"),
                "scores": item.get("scores", {}),
            }
            for item in tuple(runtime.get("localization_summary", ()))[:12]
            if isinstance(item, Mapping)
        ),
        "patch_strategy_summary": tuple(
            {
                "task_id": item.get("task_id"),
                "strategies": tuple(item.get("strategies", ()))[:6],
                "allowed_files": tuple(item.get("allowed_files", ()))[:10],
                "required_tests": tuple(item.get("required_tests", ()))[:8],
                "reviewer_roles_required": item.get("reviewer_roles_required", ()),
                "max_diff_lines": item.get("max_diff_lines"),
            }
            for item in tuple(runtime.get("patch_strategy_summary", ()))[:12]
            if isinstance(item, Mapping)
        ),
        "runtime_policy": runtime.get("runtime_policy", {}),
    }


def _agent_project_command_baseline_summary(
    baseline: Any,
) -> dict[str, Any]:
    if not isinstance(baseline, Mapping):
        return {"eligible_commands": (), "results": ()}
    return {
        "eligible_commands": tuple(baseline.get("eligible_commands", ()))[:16],
        "results": tuple(
            {
                "command": item.get("command"),
                "classification": item.get("classification"),
                "status": item.get("status"),
                "exit_code": item.get("exit_code"),
                "blocked_reason": item.get("blocked_reason"),
                "stderr_tail": str(item.get("stderr_tail") or "")[-500:],
            }
            for item in tuple(baseline.get("results", ()))[:24]
            if isinstance(item, Mapping)
        ),
    }


def _mapping_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return default


def _workspace_coding_max_excerpt_chars() -> int:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_MAX_EXCERPT_CHARS")
    if raw is None:
        return DEFAULT_WORKSPACE_CODING_MAX_EXCERPT_CHARS
    try:
        return max(800, min(12_000, int(raw)))
    except ValueError:
        return DEFAULT_WORKSPACE_CODING_MAX_EXCERPT_CHARS


def _workspace_coding_retry_excerpt_chars() -> int:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_RETRY_EXCERPT_CHARS")
    if raw is None:
        return DEFAULT_WORKSPACE_CODING_RETRY_EXCERPT_CHARS
    try:
        return max(
            _workspace_coding_max_excerpt_chars(),
            min(50_000, int(raw)),
        )
    except ValueError:
        return DEFAULT_WORKSPACE_CODING_RETRY_EXCERPT_CHARS


def _workspace_coding_prompt_file_limit() -> int:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_PROMPT_FILE_LIMIT")
    if raw is None:
        return DEFAULT_WORKSPACE_CODING_PROMPT_FILE_LIMIT
    try:
        return max(4, min(64, int(raw)))
    except ValueError:
        return DEFAULT_WORKSPACE_CODING_PROMPT_FILE_LIMIT


def _workspace_coding_max_output_tokens() -> int:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_MAX_OUTPUT_TOKENS")
    if raw is None:
        return DEFAULT_WORKSPACE_CODING_MAX_OUTPUT_TOKENS
    try:
        return max(1024, min(20000, int(raw)))
    except ValueError:
        return DEFAULT_WORKSPACE_CODING_MAX_OUTPUT_TOKENS


def _compact_excerpt(excerpt: str, *, max_chars: int) -> str:
    if len(excerpt) <= max_chars:
        return excerpt
    if max_chars <= 1200:
        return excerpt[:max_chars]
    head_chars = int(max_chars * 0.7)
    tail_chars = max_chars - head_chars
    return (
        excerpt[:head_chars]
        + "\n\n/* ... middle omitted from agent prompt; use exact visible fragments for edits ... */\n\n"
        + excerpt[-tail_chars:]
    )


def _workspace_coding_request_attempts() -> int:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_REQUEST_ATTEMPTS")
    if raw is None:
        return DEFAULT_WORKSPACE_CODING_REQUEST_ATTEMPTS
    try:
        return max(1, min(12, int(raw)))
    except ValueError:
        return DEFAULT_WORKSPACE_CODING_REQUEST_ATTEMPTS


def _workspace_coding_call_watchdog_seconds(timeout_seconds: float) -> float:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_CALL_WATCHDOG_SECONDS")
    if raw is None:
        return max(0.0, float(timeout_seconds))
    try:
        return max(0.0, min(900.0, float(raw)))
    except ValueError:
        return max(0.0, float(timeout_seconds))


class OpenAIRequestProcessError(RuntimeError):
    def __init__(self, remote_type: str, detail: str) -> None:
        super().__init__(f"{remote_type}:{detail}")
        self.remote_type = remote_type


def _request_error_type(error: Exception) -> str:
    remote_type = getattr(error, "remote_type", None)
    if remote_type:
        return str(remote_type)
    return type(error).__name__


def _request_error_detail(error: Exception) -> str:
    detail = str(error).replace("\n", " ").strip()
    return detail[:240] if detail else "no_detail"


def _workspace_coding_process_requests_enabled() -> bool:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_PROCESS_REQUESTS")
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


def _create_response_with_process_timeout(
    *,
    api_key: str,
    request: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[Any | None, Exception | None]:
    if timeout_seconds <= 0:
        return None, TimeoutError("workspace coding agent process timeout disabled")
    context = mp.get_context(_workspace_coding_process_start_method())
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_openai_response_process_worker,
        args=(result_queue, api_key, dict(request)),
    )
    try:
        process.start()
    except Exception as exc:
        return None, exc
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        return None, TimeoutError(
            f"workspace coding agent request exceeded {timeout_seconds:.3f}s "
            "process timeout"
        )
    try:
        payload = result_queue.get_nowait()
    except queue_module.Empty:
        return None, RuntimeError(
            f"workspace coding agent request process exited without result "
            f"exitcode={process.exitcode}"
        )
    if payload.get("ok"):
        return SimpleNamespace(output_text=str(payload.get("output_text") or "")), None
    return None, OpenAIRequestProcessError(
        str(payload.get("error_type") or "Exception"),
        str(payload.get("error") or "")[:1000],
    )


def _openai_response_process_worker(
    result_queue: Any,
    api_key: str,
    request: Mapping[str, Any],
) -> None:
    try:
        client = build_openai_client(
            api_key=api_key,
            timeout_seconds=float(request.get("timeout") or 0) or None,
            max_retries=0,
        )
        response = client.responses.create(**prepare_openai_response_request(request))
        result_queue.put({"ok": True, "output_text": _response_text(response)})
    except Exception as exc:  # pragma: no cover - worker-process boundary.
        result_queue.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        )


def _workspace_coding_process_start_method() -> str:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_PROCESS_START_METHOD")
    if raw:
        method = raw.strip().lower()
        if method in mp.get_all_start_methods():
            return method
    if os.environ.get("PYTEST_CURRENT_TEST") and "fork" in mp.get_all_start_methods():
        return "fork"
    if "spawn" in mp.get_all_start_methods():
        return "spawn"
    return mp.get_all_start_methods()[0]


def _workspace_coding_parse_repair_attempts() -> int:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_PARSE_REPAIR_ATTEMPTS")
    if raw is None:
        return DEFAULT_WORKSPACE_CODING_PARSE_REPAIR_ATTEMPTS
    try:
        return max(0, min(4, int(raw)))
    except ValueError:
        return DEFAULT_WORKSPACE_CODING_PARSE_REPAIR_ATTEMPTS


def _json_repair_prompt(
    *,
    prompt: str,
    raw_response: str,
    error: Exception,
) -> str:
    error_text = str(error).replace("\n", " ")[:500]
    raw_excerpt = raw_response[:1000]
    return (
        f"{prompt}\n\n"
        "STRICT_OUTPUT_REPAIR:\n"
        "Your previous response could not be parsed as the required JSON schema. "
        "Return exactly one JSON object matching the schema. Do not include markdown, "
        "commentary, empty text, analysis, or code fences.\n"
        f"parse_error: {error_text}\n"
        f"previous_response_excerpt: {raw_excerpt!r}\n"
    )


def _workspace_coding_retry_sleep_seconds(attempt: int) -> float:
    raw = os.environ.get("SOCIETY_CORE_WORKSPACE_CODING_RETRY_SLEEP_SECONDS")
    if raw is not None:
        try:
            return max(0.0, min(30.0, float(raw)))
        except ValueError:
            pass
    return min(20.0, 1.5 * (attempt + 1))


def _set_agent_stage(agent: WorkspaceCodingAgent, stage: str) -> None:
    try:
        setattr(agent, "development_stage", stage)
    except Exception:
        return


def _set_agent_run_context(
    agent: WorkspaceCodingAgent,
    *,
    namespace: str,
    attempt: int,
) -> None:
    setter = getattr(agent, "set_run_context", None)
    if callable(setter):
        setter(namespace=namespace, attempt=attempt)


def _seal_agent_proposal_run(
    agent: WorkspaceCodingAgent,
    proposal: CodingAgentPatchProposal,
) -> None:
    sealer = getattr(agent, "seal_last_proposal_run", None)
    if callable(sealer):
        sealer(proposal)


def _is_risky_patch_bundle(proposal: CodingAgentPatchProposal) -> bool:
    total_bytes = sum(
        len(patch.content.encode("utf-8"))
        + len(patch.old.encode("utf-8"))
        + len(patch.new.encode("utf-8"))
        for patch in proposal.patches
    )
    large_replace = any(
        patch.operation == "replace_fragment"
        and len([line for line in patch.old.splitlines() if line.strip()]) >= 25
        for patch in proposal.patches
    )
    broad_write = any(
        patch.operation == "create_or_replace"
        and Path(patch.path).suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".py"}
        for patch in proposal.patches
    )
    return (
        len(proposal.patches) >= 5
        or total_bytes >= 25_000
        or large_replace
        or broad_write
    )


def _verified_effective_patch_count(
    patch_results: tuple[WorkspacePatchResult, ...],
    *,
    verification_results: tuple[WorkspaceVerificationResult, ...],
    final_mode: str,
) -> int:
    if final_mode != "verified" and not _partial_patch_evidence_ready(
        patch_results=patch_results,
        verification_results=verification_results,
    ):
        return 0
    return sum(
        1
        for result in patch_results
        if result.status == "applied"
        and result.before_hash != result.after_hash
        and _counts_as_product_patch_path(result.path)
    )


def _partial_patch_evidence_ready(
    *,
    patch_results: tuple[WorkspacePatchResult, ...],
    verification_results: tuple[WorkspaceVerificationResult, ...],
) -> bool:
    if not patch_results or not verification_results:
        return False
    if any(result.status != "passed" for result in verification_results):
        return False
    summary = _patch_coverage_summary(
        patch_results=patch_results,
        verification_results=verification_results,
        generated_coverage_commands=tuple(
            result.command
            for result in verification_results
            if _known_syntax_only_command(result.command)
        ),
    )
    return (
        bool(summary.get("changed_paths"))
        and bool(summary.get("patch_coverage_ready"))
        and bool(summary.get("behavior_verification_ready"))
        and bool(summary.get("maintainer_gate_ready"))
    )


def _development_stage(
    *,
    final_mode: str,
    promoted_candidates: int,
    exploration_attempts: int,
    verified_effective_patches: int,
) -> str:
    if final_mode == "verified" and verified_effective_patches > 0:
        return "verified_evidence_patch"
    if verified_effective_patches > 0:
        return "partial_verified_evidence_patch"
    if promoted_candidates > 0 or final_mode in {
        "blocked",
        "no_effect",
        "verification_failed",
    }:
        return "guarded_promotion_agent"
    if exploration_attempts > 0:
        return "exploratory_coding_agent"
    return "guarded_promotion_agent"


def _counts_as_product_patch_path(path: str) -> bool:
    normalized = Path(path).as_posix()
    return normalized != "company-development/workspace-agent-report.json"


def _safe_relative_path(path: str) -> str:
    normalized = Path(path).as_posix().lstrip("/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe workspace patch path: {path}")
    return "/".join(parts)


def _dedupe(values) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


def _compact_promotion_verification_commands(
    commands: tuple[str, ...],
) -> tuple[str, ...]:
    """Merge equivalent Python syntax checks without removing behavior checks."""

    ordered: list[str | tuple[str, ...]] = []
    compile_paths: dict[tuple[str, ...], list[str]] = {}
    seen_compile_paths: dict[tuple[str, ...], set[str]] = {}
    for command in _dedupe(commands):
        try:
            argv = tuple(shlex.split(command))
        except ValueError:
            ordered.append(command)
            continue
        try:
            module_index = argv.index("py_compile")
        except ValueError:
            ordered.append(command)
            continue
        if module_index < 2 or argv[module_index - 1] != "-m":
            ordered.append(command)
            continue
        paths = argv[module_index + 1 :]
        if not paths or any(path.startswith("-") for path in paths):
            ordered.append(command)
            continue
        prefix = argv[: module_index + 1]
        if prefix not in compile_paths:
            compile_paths[prefix] = []
            seen_compile_paths[prefix] = set()
            ordered.append(prefix)
        for path in paths:
            if path not in seen_compile_paths[prefix]:
                compile_paths[prefix].append(path)
                seen_compile_paths[prefix].add(path)

    compacted: list[str] = []
    for item in ordered:
        if isinstance(item, str):
            compacted.append(item)
            continue
        paths = compile_paths[item]
        chunk: list[str] = []
        for path in paths:
            candidate = shlex.join((*item, *chunk, path))
            if chunk and len(candidate) > MAX_AGENT_VERIFICATION_COMMAND_CHARS:
                compacted.append(shlex.join((*item, *chunk)))
                chunk = [path]
            else:
                chunk.append(path)
        if chunk:
            compacted.append(shlex.join((*item, *chunk)))
    return tuple(compacted)


def _parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(
                f"Workspace coding agent returned non-JSON text: {raw[:200]!r}"
            ) from None
        value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Workspace coding agent returned JSON that is not an object")
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


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
