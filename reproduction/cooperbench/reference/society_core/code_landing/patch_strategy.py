"""Patch strategy planning for Code-Max candidate search."""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import LocalizationHypothesis, PatchStrategyPlan, RepoIntelligenceFabric
from .surfaces import has_security_surface


def plan_patch_strategies(
    *,
    task_specs: tuple[Any, ...],
    localizations: tuple[LocalizationHypothesis, ...],
    repo: RepoIntelligenceFabric,
    mode: str = "strong",
) -> tuple[PatchStrategyPlan, ...]:
    plans: list[PatchStrategyPlan] = []
    loc_by_task = {loc.task_id: loc for loc in localizations}
    for spec in task_specs:
        task_id = str(_field(spec, "task_id", "task"))
        theme = str(_field(spec, "source_theme", "product_improvement"))
        behavior_surface = tuple(
            str(item).lower() for item in _field(spec, "behavior_surface", ())
        )
        localization = loc_by_task.get(task_id)
        allowed_files = localization.candidate_files if localization else tuple(_field(spec, "candidate_path_hints", ()))
        strategies = _strategies_for(theme, behavior_surface)
        reviewer_roles = _reviewers_for(theme, behavior_surface)
        plans.append(
            PatchStrategyPlan(
                task_id=task_id,
                strategies=strategies,
                required_candidate_count=_candidate_count(mode, strategies),
                allowed_files=allowed_files,
                protected_files=_protected_files(repo),
                required_tests=_required_tests(spec, repo, allowed_files),
                reviewer_roles_required=reviewer_roles,
                max_diff_lines=_max_diff_lines(theme, mode),
                escalation_policy=_escalation_policy(theme),
            )
        )
    return tuple(plans)


def _strategies_for(theme: str, behavior_surface: tuple[str, ...]) -> tuple[str, ...]:
    strategies = ["minimal_fix"]
    if theme in {"architecture_generalization", "framework_generalization"}:
        strategies = ["architecture_first", "adapter_layer", "minimal_fix"]
    elif theme == "plugin_ecosystem" or _has_api_surface(behavior_surface):
        strategies = ["api_compatible_fix", "compatibility_shim", "minimal_fix"]
    elif theme == "diagnostics_and_recovery":
        strategies = ["error_message_dx", "test_first_fix", "minimal_fix"]
    elif theme == "production_build_reliability":
        strategies = ["regression_repair", "performance_optimization", "minimal_fix"]
    elif theme == "path_security":
        strategies = ["security_hardening", "minimal_fix"]
    if "security_hardening" not in strategies and has_security_surface(
        behavior_surface, (theme,)
    ):
        strategies.insert(0, "security_hardening")
    if "config" in behavior_surface:
        strategies.append("config_alignment")
    return tuple(dict.fromkeys(strategies))


def _reviewers_for(theme: str, behavior_surface: tuple[str, ...]) -> tuple[str, ...]:
    reviewers = ["intent", "regression", "evidence_boundary"]
    if theme in {"plugin_ecosystem", "framework_generalization"} or _has_api_surface(
        behavior_surface
    ):
        reviewers.append("api")
    if theme in {"architecture_generalization", "production_build_reliability"}:
        reviewers.append("architecture")
    if theme in {"path_security", "dependency_resolution"} or has_security_surface(
        behavior_surface, (theme,)
    ):
        reviewers.append("security")
    if theme == "diagnostics_and_recovery":
        reviewers.append("developer_experience")
    return tuple(dict.fromkeys(reviewers))


def _has_api_surface(surface: tuple[str, ...]) -> bool:
    return any(
        item == "api" or item.startswith("api_") or item.endswith("_api")
        for item in surface
    )


def _candidate_count(mode: str, strategies: tuple[str, ...]) -> int:
    if mode == "fast":
        return max(3, len(strategies))
    if mode == "max":
        return max(64, len(strategies) * 8)
    return max(12, len(strategies) * 3)


def _protected_files(repo: RepoIntelligenceFabric) -> tuple[str, ...]:
    return tuple(path for path in repo.file_tree if path.endswith((".lock", ".min.js")) or path in repo.generated_files)


def _required_tests(
    spec: Any,
    repo: RepoIntelligenceFabric,
    allowed_files: tuple[str, ...],
) -> tuple[str, ...]:
    tests: list[str] = []
    for path in allowed_files:
        tests.extend(repo.source_to_tests.get(path, ()))
    tests.extend(str(oracle.command) for oracle in _field(spec, "acceptance_oracles", ()) if getattr(oracle, "command", None))
    return tuple(dict.fromkeys(tests))[:12]


def _max_diff_lines(theme: str, mode: str) -> int:
    base = 80
    if theme in {"architecture_generalization", "framework_generalization"}:
        base = 240
    if mode == "fast":
        return int(base * 0.75)
    if mode == "max":
        return int(base * 3)
    return base


def _escalation_policy(theme: str) -> str:
    if theme in {"architecture_generalization", "plugin_ecosystem"}:
        return "escalate_to_architecture_or_api_review_after_first_failed_candidate"
    if theme == "path_security":
        return "escalate_to_security_review_on_any_guardrail_regression"
    return "generate_additional_candidate_after_repro_or_targeted_test_failure"


def _field(obj: Any, name: str, default: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)
