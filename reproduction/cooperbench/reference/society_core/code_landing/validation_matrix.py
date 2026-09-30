"""Validation matrix for candidate patches."""

from __future__ import annotations

import ast
import difflib
import json
import shlex
from dataclasses import replace
from pathlib import Path

from society_core.execution import CommandExecutor
from society_core.hashing import stable_hash
from society_core.safe_files import UnsafeRegularFileError, read_regular_file_text
from society_core.workspace_transaction import CandidateWorkspace
from society_core.workspace_update import (
    WorkspaceVerificationResult,
    apply_workspace_patches,
    run_workspace_verification,
)

from .schemas import AcceptanceOracleSpec, CandidatePatch, PatchValidationResult


def validate_candidate_patches(
    *,
    workspace_root: Path,
    candidates: tuple[CandidatePatch, ...],
    oracles: tuple[AcceptanceOracleSpec, ...],
    timeout_seconds: float = 60.0,
    executor: CommandExecutor | None = None,
) -> tuple[PatchValidationResult, ...]:
    validations = tuple(
        validate_candidate_patch(
            workspace_root=workspace_root,
            candidate=candidate,
            oracles=oracles,
            timeout_seconds=timeout_seconds,
            executor=executor,
        )
        for candidate in candidates
    )
    return _attach_cross_candidate_contract_results(
        workspace_root=workspace_root,
        candidates=candidates,
        validations=validations,
        timeout_seconds=timeout_seconds,
        executor=executor,
    )


def validate_candidate_patch(
    *,
    workspace_root: Path,
    candidate: CandidatePatch,
    oracles: tuple[AcceptanceOracleSpec, ...],
    timeout_seconds: float = 60.0,
    executor: CommandExecutor | None = None,
) -> PatchValidationResult:
    selected_oracles = _candidate_oracles(candidate, oracles)
    required_dimension_ids = tuple(
        dict.fromkeys(
            dimension_id
            for oracle in selected_oracles
            if oracle.required
            for dimension_id in oracle.dimension_ids
        )
    )
    public_contract_continuity_issues: tuple[str, ...] = ()
    with CandidateWorkspace.create(workspace_root) as candidate_workspace:
        patch_results = apply_workspace_patches(
            candidate_workspace.root,
            candidate.workspace_patches,
        )
        applies_cleanly = bool(patch_results) and all(
            result.status == "applied" for result in patch_results
        )
        oracle_results = {}
        candidate_verification_results = {}
        effective_diff_size = _diff_size(candidate)
        if applies_cleanly:
            effective_diff_size = _effective_diff_size(
                workspace_root,
                candidate_workspace.root,
                candidate.changed_files,
                fallback=effective_diff_size,
            )
            public_contract_continuity_issues = (
                _python_public_wrapper_continuity_issues(
                    workspace_root,
                    candidate_workspace.root,
                    candidate.changed_files,
                )
            )
            oracle_result_by_command: dict[str, WorkspaceVerificationResult] = {}
            for oracle in selected_oracles:
                with CandidateWorkspace.create(
                    candidate_workspace.root
                ) as oracle_workspace:
                    result = run_workspace_verification(
                        oracle_workspace.root,
                        (oracle.command,),
                        timeout_seconds=timeout_seconds,
                        executor=executor,
                    )[0]
                    oracle_results[oracle.oracle_id] = result
                    oracle_result_by_command.setdefault(oracle.command, result)
            candidate_commands = tuple(
                command
                for command in dict.fromkeys(candidate.verification_commands)
                if command
            )[:8]
            for command in candidate_commands:
                if command in oracle_result_by_command:
                    candidate_verification_results[command] = oracle_result_by_command[
                        command
                    ]
                    continue
                with CandidateWorkspace.create(
                    candidate_workspace.root
                ) as verification_workspace:
                    candidate_verification_results[command] = (
                        run_workspace_verification(
                            verification_workspace.root,
                            (command,),
                            timeout_seconds=timeout_seconds,
                            executor=executor,
                        )[0]
                    )
            for command in _candidate_added_test_commands(
                candidate_workspace.root,
                candidate,
            ):
                if command in candidate_verification_results:
                    continue
                with CandidateWorkspace.create(
                    candidate_workspace.root
                ) as verification_workspace:
                    candidate_verification_results[command] = (
                        run_workspace_verification(
                            verification_workspace.root,
                            (command,),
                            timeout_seconds=timeout_seconds,
                            executor=executor,
                        )[0]
                    )
    repro_oracles = tuple(
        oracle for oracle in selected_oracles if oracle.expected_on_patch == "pass"
    )
    repro_passed = bool(repro_oracles) and all(
        oracle_results.get(oracle.oracle_id) is not None
        and oracle_results[oracle.oracle_id].status == "passed"
        for oracle in repro_oracles
    )
    candidate_verification_passed = bool(candidate_verification_results) and all(
        result.status == "passed" for result in candidate_verification_results.values()
    )
    behavior_commands = tuple(
        command
        for command in candidate_verification_results
        if _is_behavior_verification_command(command)
    )
    candidate_behavior_verification_passed = bool(behavior_commands) and all(
        candidate_verification_results[command].status == "passed"
        for command in behavior_commands
    )
    added_test_commands = tuple(
        command
        for command in behavior_commands
        if _command_mentions_added_test(command, candidate.added_tests)
    )
    added_test_verification_status = _classify_added_test_verification(
        added_tests=candidate.added_tests,
        commands=added_test_commands,
        results=candidate_verification_results,
    )
    added_test_verification_passed = added_test_verification_status == "passed"
    dimension_status_by_id = _dimension_statuses(
        required_dimension_ids=required_dimension_ids,
        oracles=selected_oracles,
        oracle_results=oracle_results,
    )
    dimension_coverage_complete = all(
        status == "passed" for status in dimension_status_by_id.values()
    )
    test_only = _test_only_patch(candidate)
    broad_rewrite = _broad_rewrite_detected(candidate, effective_diff_size)
    public_api_changed = any(
        _is_public_api_file(path) for path in candidate.changed_files
    )
    config_changed = any(_is_config_file(path) for path in candidate.changed_files)
    failure_summary = _failure_summary(
        applies_cleanly=applies_cleanly,
        repro_passed=repro_passed,
        oracle_results=oracle_results,
        test_only=test_only,
        candidate_verification_results=candidate_verification_results,
        candidate_verification_passed=candidate_verification_passed,
        public_contract_continuity_issues=(public_contract_continuity_issues),
        dimension_coverage_complete=dimension_coverage_complete,
    )
    score = _score_validation(
        applies_cleanly=applies_cleanly,
        repro_passed=repro_passed,
        oracle_count=len(repro_oracles),
        test_only=test_only,
        broad_rewrite=broad_rewrite,
        public_api_changed=public_api_changed,
        risk_notes=candidate.risk_notes,
        candidate_verification_passed=candidate_verification_passed,
        candidate_behavior_verification_passed=(candidate_behavior_verification_passed),
        added_test_verification_passed=added_test_verification_passed,
    )
    payload = {
        "candidate": candidate.patch_id,
        "patch_results": patch_results,
        "oracle_results": oracle_results,
        "candidate_verification_results": candidate_verification_results,
        "public_contract_continuity_issues": public_contract_continuity_issues,
        "score": score,
        "failure": failure_summary,
    }
    return PatchValidationResult(
        patch_id=candidate.patch_id,
        applies_cleanly=applies_cleanly,
        patch_results=patch_results,
        oracle_results=oracle_results,
        repro_passed=repro_passed,
        targeted_tests_passed=(
            repro_passed
            and (not candidate_verification_results or candidate_verification_passed)
        ),
        regression_tests_passed=(
            True
            if added_test_verification_status == "passed"
            else False
            if added_test_verification_status == "failed"
            else None
        ),
        lint_passed=None,
        typecheck_passed=None,
        static_analysis_passed=None,
        security_scan_passed=None,
        performance_probe_passed=None,
        diff_size=effective_diff_size,
        files_touched_count=len(candidate.changed_files),
        public_api_changed=public_api_changed,
        config_changed=config_changed,
        test_only_patch=test_only,
        broad_rewrite_detected=broad_rewrite,
        failure_summary=failure_summary,
        score=score,
        evidence_hash=stable_hash(payload),
        candidate_verification_results=candidate_verification_results,
        candidate_verification_passed=candidate_verification_passed,
        candidate_behavior_verification_passed=(candidate_behavior_verification_passed),
        added_test_verification_passed=added_test_verification_passed,
        added_test_verification_status=added_test_verification_status,
        public_contract_continuity_passed=(not public_contract_continuity_issues),
        public_contract_continuity_issues=(public_contract_continuity_issues),
        required_dimension_ids=required_dimension_ids,
        dimension_status_by_id=dimension_status_by_id,
        dimension_coverage_complete=dimension_coverage_complete,
    )


def _candidate_oracles(
    candidate: CandidatePatch,
    oracles: tuple[AcceptanceOracleSpec, ...],
) -> tuple[AcceptanceOracleSpec, ...]:
    selected = tuple(
        oracle for oracle in oracles if oracle.oracle_id in candidate.parent_oracle_ids
    )
    task_oracles = tuple(
        oracle for oracle in oracles if oracle.task_id == candidate.task_id
    )
    required = tuple(oracle for oracle in task_oracles if oracle.required)
    if selected:
        return tuple(
            {oracle.oracle_id: oracle for oracle in (*required, *selected)}.values()
        )
    return task_oracles


def _classify_added_test_verification(
    *,
    added_tests: tuple[str, ...],
    commands: tuple[str, ...],
    results: dict[str, WorkspaceVerificationResult],
) -> str:
    if not added_tests:
        return "not_applicable"
    if not commands:
        return "not_run"
    statuses = tuple(
        results[command].status for command in commands if command in results
    )
    if not statuses:
        return "not_run"
    if all(status == "passed" for status in statuses):
        return "passed"
    if any(status == "failed" for status in statuses):
        return "failed"
    return "unavailable"


def _dimension_statuses(
    *,
    required_dimension_ids: tuple[str, ...],
    oracles: tuple[AcceptanceOracleSpec, ...],
    oracle_results: dict[str, WorkspaceVerificationResult],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for dimension_id in required_dimension_ids:
        witnesses = tuple(
            oracle
            for oracle in oracles
            if oracle.required and dimension_id in oracle.dimension_ids
        )
        if not witnesses:
            statuses[dimension_id] = "uncovered"
            continue
        observed = tuple(oracle_results.get(oracle.oracle_id) for oracle in witnesses)
        if any(result is None for result in observed):
            statuses[dimension_id] = "not_run"
        elif all(
            result.status == "passed" for result in observed if result is not None
        ):
            statuses[dimension_id] = "passed"
        elif any(
            result.status == "failed" for result in observed if result is not None
        ):
            statuses[dimension_id] = "failed"
        else:
            statuses[dimension_id] = "unavailable"
    return statuses


def _test_only_patch(candidate: CandidatePatch) -> bool:
    if not candidate.changed_files:
        return False
    return all(_is_test_file(path) for path in candidate.changed_files)


def _is_test_file(path: str) -> bool:
    lowered = path.lower()
    return (
        "/test" in f"/{lowered}"
        or lowered.startswith("test_")
        or lowered.endswith((".test.js", ".spec.js"))
    )


def _broad_rewrite_detected(
    candidate: CandidatePatch,
    effective_diff_size: int,
) -> bool:
    return effective_diff_size > 500 or any(
        "broad_rewrite" in note for note in candidate.risk_notes
    )


def complex_change_evidence_ready(validation: PatchValidationResult) -> bool:
    """Require stronger functional evidence for architecture-scale candidates."""

    if not validation.broad_rewrite_detected:
        return True
    peer_contracts_passed = (
        validation.cross_candidate_contract_total == 0
        or validation.cross_candidate_contract_pass_count
        == validation.cross_candidate_contract_total
    )
    return all(
        (
            validation.applies_cleanly,
            validation.repro_passed,
            bool(validation.candidate_verification_results),
            validation.candidate_verification_passed,
            validation.candidate_behavior_verification_passed,
            validation.added_test_verification_passed,
            validation.qualified_added_contract_count > 0,
            validation.public_contract_continuity_passed,
            peer_contracts_passed,
        )
    )


def _is_public_api_file(path: str) -> bool:
    lowered = path.lower()
    return (
        lowered.endswith((".d.ts", "__init__.py"))
        or lowered in {"index.js", "index.ts", "src/index.js", "src/index.ts"}
        or "api" in lowered
    )


def _is_config_file(path: str) -> bool:
    name = Path(path).name.lower()
    return name in {"package.json", "pyproject.toml", "tsconfig.json"} or name.endswith(
        ".config.js"
    )


def _diff_size(candidate: CandidatePatch) -> int:
    if candidate.unified_diff:
        return len(candidate.unified_diff.splitlines())
    return sum(
        len(patch.content.splitlines())
        + len(patch.old.splitlines())
        + len(patch.new.splitlines())
        for patch in candidate.workspace_patches
    )


def _effective_diff_size(
    baseline_root: Path,
    candidate_root: Path,
    changed_files: tuple[str, ...],
    *,
    fallback: int,
) -> int:
    baseline_root = baseline_root.resolve()
    candidate_root = candidate_root.resolve()
    changed_lines = 0
    try:
        for relative in changed_files:
            before_path = baseline_root / relative
            after_path = candidate_root / relative
            before = (
                read_regular_file_text(
                    baseline_root,
                    before_path,
                    max_bytes=16_000_000,
                ).splitlines(keepends=True)
                if before_path.exists()
                else []
            )
            after = (
                read_regular_file_text(
                    candidate_root,
                    after_path,
                    max_bytes=16_000_000,
                ).splitlines(keepends=True)
                if after_path.exists()
                else []
            )
            for line in difflib.unified_diff(before, after, n=0):
                if line.startswith(("+++", "---")):
                    continue
                if line.startswith(("+", "-")):
                    changed_lines += 1
    except (OSError, UnicodeDecodeError, UnsafeRegularFileError):
        return fallback
    return changed_lines


def _failure_summary(
    *,
    applies_cleanly: bool,
    repro_passed: bool,
    oracle_results: dict[str, object],
    test_only: bool,
    candidate_verification_results: dict[str, object],
    candidate_verification_passed: bool,
    public_contract_continuity_issues: tuple[str, ...],
    dimension_coverage_complete: bool,
) -> str | None:
    if not applies_cleanly:
        return "patch_did_not_apply"
    if test_only:
        return "test_only_patch"
    if public_contract_continuity_issues:
        return "public_contract_continuity_failed"
    if not dimension_coverage_complete:
        return "required_contract_dimension_failed"
    if candidate_verification_results and not candidate_verification_passed:
        return "candidate_verification_failed"
    if not repro_passed:
        if not oracle_results:
            return "missing_oracle_execution"
        return "repro_failed"
    return None


def _score_validation(
    *,
    applies_cleanly: bool,
    repro_passed: bool,
    oracle_count: int,
    test_only: bool,
    broad_rewrite: bool,
    public_api_changed: bool,
    risk_notes: tuple[str, ...],
    candidate_verification_passed: bool,
    candidate_behavior_verification_passed: bool,
    added_test_verification_passed: bool,
) -> float:
    _ = risk_notes
    score = 0.0
    if applies_cleanly:
        score += 0.15
    if repro_passed:
        score += 0.3
    score += min(0.1, 0.05 * oracle_count)
    if candidate_verification_passed:
        score += 0.1
    if candidate_behavior_verification_passed:
        score += 0.1
    if added_test_verification_passed:
        score += 0.1
    if not test_only:
        score += 0.1
    if not broad_rewrite:
        score += 0.05
    if public_api_changed:
        score -= 0.05
    return round(max(0.0, min(1.0, score)), 4)


def _is_behavior_verification_command(command: str) -> bool:
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    lowered = tuple(part.casefold() for part in argv)
    rendered = " ".join(lowered)
    if any(marker in lowered for marker in ("compileall", "py_compile", "--version")):
        return False
    if any(
        marker in rendered
        for marker in (
            "pytest",
            "unittest",
            "npm test",
            "npm run test",
            "pnpm test",
            "yarn test",
            ".test.",
            ".spec.",
            "/test_",
            "test/",
            "tests/",
        )
    ):
        return True
    return "-c" in lowered and any(
        marker in rendered for marker in ("assert ", "raise ", "process.exit")
    )


def _command_mentions_added_test(
    command: str,
    added_tests: tuple[str, ...],
) -> bool:
    normalized = command.replace("\\", "/")
    return any(path.replace("\\", "/") in normalized for path in added_tests)


def _candidate_added_test_commands(
    workspace_root: Path,
    candidate: CandidatePatch,
) -> tuple[str, ...]:
    commands: list[str] = []
    for command in tuple(dict.fromkeys(candidate.verification_commands))[:8]:
        if not command:
            continue
        if _is_behavior_verification_command(command) and _command_mentions_added_test(
            command, candidate.added_tests
        ):
            commands.append(command)
        resolved = _resolve_direct_package_script_command(workspace_root, command)
        if (
            resolved
            and _is_behavior_verification_command(resolved)
            and _command_mentions_added_test(resolved, candidate.added_tests)
        ):
            commands.append(resolved)
    return tuple(dict.fromkeys(commands))


def _resolve_direct_package_script_command(
    workspace_root: Path,
    command: str,
) -> str | None:
    try:
        argv = tuple(shlex.split(command))
    except ValueError:
        return None
    normalized = tuple(part.casefold() for part in argv)
    script_name: str | None = None
    if normalized in {("npm", "test"), ("pnpm", "test"), ("yarn", "test")}:
        script_name = "test"
    elif (
        len(normalized) == 3
        and normalized[0] in {"npm", "pnpm", "yarn"}
        and normalized[1] == "run"
    ):
        script_name = argv[2]
    if script_name is None:
        return None

    package_path = workspace_root / "package.json"
    if not package_path.is_file():
        return None
    try:
        package = json.loads(
            read_regular_file_text(
                workspace_root,
                package_path,
                max_bytes=1_000_000,
            )
        )
    except (
        OSError,
        UnicodeDecodeError,
        UnsafeRegularFileError,
        json.JSONDecodeError,
    ):
        return None
    if not isinstance(package, dict):
        return None
    scripts = package.get("scripts")
    if not isinstance(scripts, dict):
        return None
    script = scripts.get(script_name)
    if not isinstance(script, str) or not script.strip():
        return None
    return _strict_direct_test_command(script)


def _strict_direct_test_command(script: str) -> str | None:
    if any(marker in script for marker in ("\n", "\r", "\x00", "`", "$(", "${")):
        return None
    try:
        lexer = shlex.shlex(script, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        argv = tuple(lexer)
    except ValueError:
        return None
    if not argv:
        return None
    forbidden_tokens = {
        "&",
        "&&",
        "|",
        "||",
        ";",
        ";;",
        "(",
        ")",
        "<",
        ">",
        ">>",
        "<<",
    }
    if any(token in forbidden_tokens for token in argv):
        return None
    executable = Path(argv[0]).name.casefold()
    if executable not in {
        "node",
        "nodejs",
        "python",
        "python3",
        "pytest",
        "bun",
        "deno",
    }:
        return None
    return shlex.join(argv)


def _attach_cross_candidate_contract_results(
    *,
    workspace_root: Path,
    candidates: tuple[CandidatePatch, ...],
    validations: tuple[PatchValidationResult, ...],
    timeout_seconds: float,
    executor: CommandExecutor | None,
) -> tuple[PatchValidationResult, ...]:
    if executor is None:
        return validations
    validation_by_id = {validation.patch_id: validation for validation in validations}
    qualified_contracts: list[tuple[str, str, tuple[object, ...], str]] = []
    for candidate in candidates:
        validation = validation_by_id.get(candidate.patch_id)
        if validation is None or not validation.added_test_verification_passed:
            continue
        test_patches = tuple(
            patch for patch in candidate.workspace_patches if _is_test_file(patch.path)
        )
        if not test_patches:
            continue
        with CandidateWorkspace.create(workspace_root) as owner_workspace:
            owner_results = apply_workspace_patches(
                owner_workspace.root,
                candidate.workspace_patches,
            )
            if not owner_results or any(
                result.status not in {"applied", "no_op"} for result in owner_results
            ):
                continue
            commands = _candidate_added_test_commands(
                owner_workspace.root,
                candidate,
            )
        for command in commands:
            with CandidateWorkspace.create(workspace_root) as baseline_test_workspace:
                patch_results = apply_workspace_patches(
                    baseline_test_workspace.root,
                    test_patches,
                )
                if not patch_results or any(
                    result.status not in {"applied", "no_op"}
                    for result in patch_results
                ):
                    continue
                baseline_result = run_workspace_verification(
                    baseline_test_workspace.root,
                    (command,),
                    timeout_seconds=timeout_seconds,
                    executor=executor,
                )[0]
            if baseline_result.status != "failed":
                continue
            contract_id = stable_hash(
                {
                    "owner_patch_id": candidate.patch_id,
                    "command": command,
                    "test_paths": tuple(patch.path for patch in test_patches),
                }
            )[:16]
            qualified_contracts.append(
                (candidate.patch_id, command, test_patches, contract_id)
            )
    if not qualified_contracts:
        return validations

    enriched: list[PatchValidationResult] = []
    for candidate, validation in zip(candidates, validations, strict=True):
        peer_contracts = tuple(
            contract
            for contract in qualified_contracts
            if contract[0] != candidate.patch_id
        )
        passed = 0
        failures: list[str] = []
        for owner_patch_id, command, test_patches, contract_id in peer_contracts:
            with CandidateWorkspace.create(workspace_root) as target_workspace:
                target_results = apply_workspace_patches(
                    target_workspace.root,
                    candidate.workspace_patches,
                )
                if not target_results or any(
                    result.status not in {"applied", "no_op"}
                    for result in target_results
                ):
                    failures.append(
                        f"peer_contract_apply_failed:{owner_patch_id}:{contract_id}"
                    )
                    continue
                peer_results = apply_workspace_patches(
                    target_workspace.root,
                    test_patches,
                )
                if not peer_results or any(
                    result.status not in {"applied", "no_op"} for result in peer_results
                ):
                    failures.append(
                        f"peer_contract_stage_failed:{owner_patch_id}:{contract_id}"
                    )
                    continue
                result = run_workspace_verification(
                    target_workspace.root,
                    (command,),
                    timeout_seconds=timeout_seconds,
                    executor=executor,
                )[0]
            if result.status == "passed":
                passed += 1
            else:
                failures.append(
                    f"peer_contract_failed:{owner_patch_id}:{contract_id}:{result.status}"
                )
        enriched.append(
            replace(
                validation,
                qualified_added_contract_count=sum(
                    1
                    for owner_patch_id, _, _, _ in qualified_contracts
                    if owner_patch_id == candidate.patch_id
                ),
                cross_candidate_contract_pass_count=passed,
                cross_candidate_contract_total=len(peer_contracts),
                cross_candidate_contract_failures=tuple(failures),
            )
        )
    return tuple(enriched)


def _python_public_wrapper_continuity_issues(
    baseline_root: Path,
    candidate_root: Path,
    changed_files: tuple[str, ...],
) -> tuple[str, ...]:
    issues: list[str] = []
    for relative in changed_files:
        if not relative.endswith(".py"):
            continue
        baseline_path = baseline_root / relative
        candidate_path = candidate_root / relative
        if not baseline_path.is_file() or not candidate_path.is_file():
            continue
        try:
            baseline_tree = ast.parse(
                read_regular_file_text(
                    baseline_root,
                    baseline_path,
                    max_bytes=4_000_000,
                )
            )
            candidate_tree = ast.parse(
                read_regular_file_text(
                    candidate_root,
                    candidate_path,
                    max_bytes=4_000_000,
                )
            )
        except (OSError, SyntaxError, UnicodeDecodeError, UnsafeRegularFileError):
            continue
        baseline_functions = _top_level_python_functions(baseline_tree)
        candidate_functions = _top_level_python_functions(candidate_tree)
        for callee_name, candidate_callee in candidate_functions.items():
            baseline_callee = baseline_functions.get(callee_name)
            if baseline_callee is None or callee_name.startswith("_"):
                continue
            baseline_parameters = _python_parameter_names(baseline_callee)
            candidate_parameters = _python_parameter_names(candidate_callee)
            added_parameters = tuple(
                name
                for name in candidate_parameters
                if name not in baseline_parameters and name not in {"self", "cls"}
            )
            if not added_parameters:
                continue
            for wrapper_name, candidate_wrapper in candidate_functions.items():
                baseline_wrapper = baseline_functions.get(wrapper_name)
                if (
                    wrapper_name == callee_name
                    or wrapper_name.startswith("_")
                    or baseline_wrapper is None
                    or not _is_direct_python_wrapper(baseline_wrapper, callee_name)
                ):
                    continue
                call = _direct_python_wrapper_call(candidate_wrapper, callee_name)
                if call is None:
                    continue
                for parameter in added_parameters:
                    if not _python_wrapper_forwards_parameter(
                        wrapper=candidate_wrapper,
                        callee=candidate_callee,
                        call=call,
                        parameter=parameter,
                    ):
                        issues.append(
                            "public_wrapper_missing_forward:"
                            f"{relative}:{wrapper_name}->{callee_name}:{parameter}"
                        )
    return tuple(dict.fromkeys(issues))


def _top_level_python_functions(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _python_parameter_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    arguments = function.args
    names = [
        argument.arg
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    ]
    if arguments.vararg is not None:
        names.append(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.append(arguments.kwarg.arg)
    return tuple(names)


def _is_direct_python_wrapper(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    callee_name: str,
) -> bool:
    return _direct_python_wrapper_call(function, callee_name) is not None


def _direct_python_wrapper_call(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    callee_name: str,
) -> ast.Call | None:
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    for node in ast.walk(body[0].value):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == callee_name:
            return node
    return None


def _python_wrapper_forwards_parameter(
    *,
    wrapper: ast.FunctionDef | ast.AsyncFunctionDef,
    callee: ast.FunctionDef | ast.AsyncFunctionDef,
    call: ast.Call,
    parameter: str,
) -> bool:
    wrapper_parameters = set(_python_parameter_names(wrapper))
    if parameter not in wrapper_parameters:
        wrapper_kwargs = wrapper.args.kwarg.arg if wrapper.args.kwarg else None
        if wrapper_kwargs is None:
            return False
        return any(
            keyword.arg is None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == wrapper_kwargs
            for keyword in call.keywords
        )
    for keyword in call.keywords:
        if (
            keyword.arg == parameter
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == parameter
        ):
            return True
    positional = (*callee.args.posonlyargs, *callee.args.args)
    positional_names = tuple(argument.arg for argument in positional)
    if parameter not in positional_names:
        return False
    index = positional_names.index(parameter)
    return (
        len(call.args) > index
        and isinstance(call.args[index], ast.Name)
        and call.args[index].id == parameter
    )
