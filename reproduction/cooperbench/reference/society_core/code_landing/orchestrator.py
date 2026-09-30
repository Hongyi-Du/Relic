"""Typed Code-Max runtime orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from society_core.execution import CommandExecutor
from society_core.hashing import canonicalize, stable_hash

from .environment import build_workspace_execution_profile
from .guarded_diff import decide_landing, guard_selected_patch
from .integration import build_development_epics, evaluate_integration_candidate
from .localizer import localize_faults
from .oracle_builder import build_acceptance_oracles
from .patch_strategy import plan_patch_strategies
from .repo_intelligence import build_repo_intelligence_fabric
from .reranker import select_best_patch
from .reviewers import review_selected_patch
from .schemas import AcceptanceOracleSpec, CandidatePatch, CodeMaxRunResult
from .trajectory_store import build_code_landing_trajectory
from .validation_matrix import validate_candidate_patches


def run_code_max_runtime(
    *,
    source_report: Mapping[str, Any],
    workspace_root: Path,
    task_specs: tuple[Any, ...],
    candidate_patches: tuple[CandidatePatch, ...] = (),
    confirm_base: bool = False,
    timeout_seconds: float = 60.0,
    mode: str = "strong",
    executor: CommandExecutor | None = None,
    oracle_commands: tuple[str, ...] = (),
) -> CodeMaxRunResult:
    """Run the full typed landing pipeline for candidate patches."""

    environment = build_workspace_execution_profile(
        workspace_root,
        run_smoke=False,
        timeout_seconds=min(timeout_seconds, 20.0),
        executor=executor,
    )
    repo = build_repo_intelligence_fabric(
        workspace_root,
        execution_profile=environment,
        themes=_themes_from_tasks(task_specs, source_report),
        priority_paths=_task_path_hints(task_specs),
    )
    oracles = build_acceptance_oracles(
        source_report=source_report,
        task_specs=task_specs,
        workspace_root=workspace_root,
        repo=repo,
        confirm_base=confirm_base,
        timeout_seconds=timeout_seconds,
        executor=executor,
    )
    if oracle_commands:
        oracles = _merge_oracles(
            oracles,
            _explicit_oracles(task_specs, oracle_commands),
        )
    localizations = localize_faults(
        task_specs=task_specs,
        repo=repo,
        oracles=oracles,
        top_k_files=_top_k_files(mode),
    )
    strategies = plan_patch_strategies(
        task_specs=task_specs,
        localizations=localizations,
        repo=repo,
        mode=mode,
    )
    validations = validate_candidate_patches(
        workspace_root=workspace_root,
        candidates=candidate_patches,
        oracles=oracles,
        timeout_seconds=timeout_seconds,
        executor=executor,
    )
    (
        selection,
        review_decision,
        guarded_decision,
        landing_decision,
    ) = _select_reviewable_candidate(
        workspace_root=workspace_root,
        candidates=candidate_patches,
        validations=validations,
        task_specs=task_specs,
        reviewer_roles=_reviewer_roles(strategies),
    )
    epics = build_development_epics(
        task_specs=task_specs,
        source_report=source_report,
    )
    integration_decision = evaluate_integration_candidate(
        epics=epics,
        selection=selection,
        review_decision=review_decision,
        landing_decision=landing_decision,
    )
    trajectory = build_code_landing_trajectory(
        repo_hash=repo.repo_hash,
        localizations=localizations,
        strategies=strategies,
        validations=validations,
        selection=selection,
        review_decision=review_decision,
        integration_decision=integration_decision,
        guarded_decision=guarded_decision,
        landing_decision=landing_decision,
    )
    payload = {
        "source_report": source_report.get("report_id"),
        "environment": environment.repo_hash,
        "repo": repo.repo_hash,
        "oracles": tuple(oracle.oracle_id for oracle in oracles),
        "selection": selection.selected_patch_id,
        "landing": landing_decision.status,
        "trajectory": trajectory.trajectory_id,
    }
    return CodeMaxRunResult(
        runtime_id=f"code_max_runtime_{stable_hash(payload)[:24]}",
        environment=environment,
        repo=repo,
        oracles=oracles,
        localizations=localizations,
        strategies=strategies,
        validations=validations,
        selection=selection,
        review_decision=review_decision,
        integration_decision=integration_decision,
        guarded_decision=guarded_decision,
        landing_decision=landing_decision,
        trajectory=trajectory,
    )


def _select_reviewable_candidate(
    *,
    workspace_root: Path,
    candidates: tuple[CandidatePatch, ...],
    validations: tuple[Any, ...],
    task_specs: tuple[Any, ...],
    reviewer_roles: tuple[str, ...],
) -> tuple[Any, Any, Any, Any]:
    """Try already-generated candidates before requesting another model round."""

    remaining = candidates
    attempted_patch_ids: list[str] = []
    first_blocked: tuple[Any, Any, Any, Any] | None = None
    accepted_statuses = {
        "candidate_validated",
        "candidate_validated_with_claim_downgrade",
    }
    while True:
        selection = select_best_patch(
            candidates=remaining,
            validations=validations,
        )
        review_decision = review_selected_patch(
            selection=selection,
            candidates=candidates,
            validations=validations,
            task_specs=task_specs,
            reviewer_roles=reviewer_roles,
        )
        guarded_decision = guard_selected_patch(
            workspace_root=workspace_root,
            selection=selection,
            candidates=candidates,
            validations=validations,
        )
        landing_decision = decide_landing(
            selection=selection,
            review_decision=review_decision,
            guarded_decision=guarded_decision,
            validations=validations,
        )
        outcome = (
            selection,
            review_decision,
            guarded_decision,
            landing_decision,
        )
        if first_blocked is None:
            first_blocked = outcome
        if landing_decision.status in accepted_statuses:
            if attempted_patch_ids:
                selection = replace(
                    selection,
                    rejected_patch_ids=tuple(
                        dict.fromkeys(
                            (*attempted_patch_ids, *selection.rejected_patch_ids)
                        )
                    ),
                    selection_reason=(
                        f"{selection.selection_reason}:"
                        f"review_fallbacks:{len(attempted_patch_ids)}"
                    ),
                )
            return (
                selection,
                review_decision,
                guarded_decision,
                landing_decision,
            )
        selected_patch_id = selection.selected_patch_id
        if selected_patch_id is None:
            break
        attempted_patch_ids.append(selected_patch_id)
        remaining = tuple(
            candidate
            for candidate in remaining
            if candidate.patch_id != selected_patch_id
        )
        if not remaining:
            break
    assert first_blocked is not None
    return first_blocked


def build_code_max_context_summary(
    *,
    source_report: Mapping[str, Any],
    workspace_root: Path,
    task_specs: tuple[Any, ...],
    confirm_base: bool = False,
    timeout_seconds: float = 20.0,
    mode: str = "strong",
    executor: CommandExecutor | None = None,
) -> dict[str, Any]:
    """Build the agent-facing Code-Max context without candidate search."""

    environment = build_workspace_execution_profile(
        workspace_root,
        run_smoke=False,
        timeout_seconds=timeout_seconds,
        executor=executor,
    )
    repo = build_repo_intelligence_fabric(
        workspace_root,
        execution_profile=environment,
        themes=_themes_from_tasks(task_specs, source_report),
        priority_paths=_task_path_hints(task_specs),
    )
    oracles = build_acceptance_oracles(
        source_report=source_report,
        task_specs=task_specs,
        workspace_root=workspace_root,
        repo=repo,
        confirm_base=confirm_base,
        timeout_seconds=timeout_seconds,
        executor=executor,
    )
    localizations = localize_faults(
        task_specs=task_specs,
        repo=repo,
        oracles=oracles,
        top_k_files=_top_k_files(mode),
    )
    strategies = plan_patch_strategies(
        task_specs=task_specs,
        localizations=localizations,
        repo=repo,
        mode=mode,
    )
    payload = {
        "source_report": source_report.get("report_id"),
        "environment": environment.repo_hash,
        "repo": repo.repo_hash,
        "oracles": tuple(oracle.oracle_id for oracle in oracles),
        "localizations": tuple(loc.hypothesis_id for loc in localizations),
        "strategies": tuple(plan.task_id for plan in strategies),
    }
    return {
        "runtime_id": f"code_max_context_{stable_hash(payload)[:24]}",
        "mode": mode,
        "environment": canonicalize(environment),
        "repo": {
            "repo_hash": repo.repo_hash,
            "file_count": len(repo.file_tree),
            "source_roots": repo.source_roots,
            "test_roots": repo.test_roots,
            "package_manager": repo.package_manager,
            "entrypoints": repo.entrypoints,
            "public_api_surfaces": repo.public_api_surfaces[:80],
            "candidate_files_by_theme": repo.candidate_files_by_theme,
            "impact_context": _impact_context(repo, localizations),
            "semantic_search_index_id": repo.semantic_search_index_id,
            "code_knowledge_graph_id": repo.code_knowledge_graph_id,
        },
        "oracle_summary": tuple(canonicalize(oracle) for oracle in oracles),
        "localization_summary": tuple(
            {
                "hypothesis_id": loc.hypothesis_id,
                "task_id": loc.task_id,
                "candidate_files": loc.candidate_files,
                "candidate_symbols": loc.candidate_symbols,
                "confidence": loc.confidence,
                "scores": loc.scores,
                "risk_notes": loc.risk_notes,
            }
            for loc in localizations
        ),
        "patch_strategy_summary": tuple(canonicalize(plan) for plan in strategies),
        "runtime_policy": {
            "reproducer_first": True,
            "multi_candidate_validation": True,
            "function_first_candidate_selection": True,
            "dependency_impact_cone_localization": True,
            "evidence_scaled_change_scope": True,
            "review_before_candidate_validation": True,
            "guarded_kernel_required": True,
            "claim_level_decided_by_evidence": True,
        },
    }


def _top_k_files(mode: str) -> int:
    if mode == "fast":
        return 5
    if mode == "max":
        return 30
    return 15


def _impact_context(
    repo: Any, localizations: tuple[Any, ...]
) -> tuple[dict[str, Any], ...]:
    paths: list[str] = []
    for localization in localizations:
        paths.extend(tuple(getattr(localization, "candidate_files", ()))[:12])
    return tuple(
        {
            "path": path,
            "imports": tuple(repo.import_graph.get(path, ()))[:12],
            "dependents": tuple(repo.reverse_dependency_graph.get(path, ()))[:12],
            "tests": tuple(repo.source_to_tests.get(path, ()))[:12],
        }
        for path in tuple(dict.fromkeys(paths))[:24]
    )


def _reviewer_roles(strategies: tuple[Any, ...]) -> tuple[str, ...]:
    roles: list[str] = ["intent", "regression", "test_quality", "evidence_boundary"]
    for plan in strategies:
        roles.extend(getattr(plan, "reviewer_roles_required", ()))
    return tuple(dict.fromkeys(roles))


def _themes_from_tasks(
    task_specs: tuple[Any, ...],
    source_report: Mapping[str, Any],
) -> tuple[str, ...]:
    themes = tuple(
        dict.fromkeys(
            str(getattr(task, "source_theme", ""))
            for task in task_specs
            if getattr(task, "source_theme", "")
        )
    )
    if themes:
        return themes
    raw = source_report.get("proposed_themes")
    if isinstance(raw, (list, tuple)) and raw:
        return tuple(str(item) for item in raw)
    return ("product_improvement",)


def _task_path_hints(task_specs: tuple[Any, ...]) -> tuple[str, ...]:
    paths: list[str] = []
    for task in task_specs:
        raw_paths = (
            task.get("candidate_path_hints", ())
            if isinstance(task, Mapping)
            else getattr(task, "candidate_path_hints", ())
        )
        if isinstance(raw_paths, (list, tuple)):
            paths.extend(str(path) for path in raw_paths if str(path))
    return tuple(dict.fromkeys(paths))


def _explicit_oracles(
    task_specs: tuple[Any, ...],
    commands: tuple[str, ...],
) -> tuple[AcceptanceOracleSpec, ...]:
    task_id = str(getattr(task_specs[0], "task_id", "task")) if task_specs else "task"
    oracles: list[AcceptanceOracleSpec] = []
    for index, command in enumerate(dict.fromkeys(commands)):
        payload = {"task_id": task_id, "index": index, "command": command}
        evidence_hash = stable_hash(payload)
        oracles.append(
            AcceptanceOracleSpec(
                oracle_id=f"oracle_{evidence_hash[:16]}",
                task_id=task_id,
                kind="behavior",
                command=command,
                files_created=(),
                expected_on_base="not_recorded",
                expected_on_patch="pass",
                base_observed=None,
                base_status="not_run",
                confidence=0.75,
                false_positive_risk="medium",
                evidence_hash=evidence_hash,
            )
        )
    return tuple(oracles)


def _merge_oracles(
    task_oracles: tuple[AcceptanceOracleSpec, ...],
    explicit_oracles: tuple[AcceptanceOracleSpec, ...],
) -> tuple[AcceptanceOracleSpec, ...]:
    merged = [
        oracle for oracle in task_oracles if oracle.required and oracle.dimension_ids
    ]
    seen_commands = {oracle.command for oracle in merged}
    merged.extend(
        oracle for oracle in explicit_oracles if oracle.command not in seen_commands
    )
    return tuple(merged)
