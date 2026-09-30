"""Integration-epic gates for Code-Max runs."""

from __future__ import annotations

from typing import Any, Mapping

from society_core.hashing import stable_hash

from .schemas import (
    DevelopmentEpic,
    IntegrationDecision,
    LandingDecision,
    PatchSelectionDecision,
    ReviewDecision,
)


def build_development_epics(
    *,
    task_specs: tuple[Any, ...],
    source_report: Mapping[str, Any] | None = None,
) -> tuple[DevelopmentEpic, ...]:
    """Group task specs into release-candidate epics."""

    del source_report
    grouped: dict[str, list[Any]] = {}
    for task in task_specs:
        theme = str(_field(task, "source_theme", "product_improvement"))
        grouped.setdefault(theme, []).append(task)
    epics: list[DevelopmentEpic] = []
    for theme, tasks in grouped.items():
        task_ids = tuple(str(_field(task, "task_id", "")) for task in tasks)
        payload = {"theme": theme, "tasks": task_ids}
        epics.append(
            DevelopmentEpic(
                epic_id=f"epic_{stable_hash(payload)[:16]}",
                source_theme=theme,
                public_evidence_refs=tuple(
                    dict.fromkeys(
                        ref
                        for task in tasks
                        for ref in tuple(_field(task, "public_evidence_refs", ()))
                    )
                ),
                ordered_task_ids=task_ids,
                dependency_edges=_dependency_edges(tasks),
                integration_oracles=tuple(
                    dict.fromkeys(
                        str(oracle.command)
                        for task in tasks
                        for oracle in tuple(_field(task, "acceptance_oracles", ()))
                        if getattr(oracle, "command", None)
                    )
                ),
                release_strategy=_release_strategy(theme, tasks),
                stop_condition=(
                    "all_required_tasks_candidate_validated_or_explicitly_waived"
                ),
            )
        )
    return tuple(epics)


def evaluate_integration_candidate(
    *,
    epics: tuple[DevelopmentEpic, ...],
    selection: PatchSelectionDecision,
    review_decision: ReviewDecision,
    landing_decision: LandingDecision | None = None,
) -> IntegrationDecision:
    """Check whether the selected patch can be part of a coherent epic."""

    if not epics:
        return IntegrationDecision(
            epic_id="none",
            status="blocked",
            candidate_coherent=False,
            reasons=("no_epic",),
            required_repairs=("build_development_epics",),
        )
    epic = _epic_for_task(epics, selection.task_id) or epics[0]
    reasons: list[str] = []
    repairs: list[str] = []
    if selection.selected_patch_id is None:
        reasons.append("no_selected_patch")
        repairs.append("generate_additional_candidates")
    if review_decision.status == "blocked":
        reasons.append("semantic_review_blocked")
        repairs.append("repair_reviewer_findings")
    if landing_decision is not None and landing_decision.status not in {
        "candidate_validated",
        "candidate_validated_with_claim_downgrade",
    }:
        reasons.append("candidate_validation_gate_not_passed")
        repairs.append("repair_candidate_validation_gate")
    status = "candidate_coherent" if not reasons else "needs_revision"
    payload = {
        "epic": epic.epic_id,
        "selection": selection.selected_patch_id,
        "review": review_decision.status,
        "landing": landing_decision.status if landing_decision else None,
        "reasons": reasons,
    }
    return IntegrationDecision(
        epic_id=epic.epic_id,
        status=status,
        candidate_coherent=not reasons,
        reasons=tuple(reasons) or ("candidate_integration_coherent",),
        required_repairs=tuple(repairs),
    )


def _epic_for_task(
    epics: tuple[DevelopmentEpic, ...],
    task_id: str | None,
) -> DevelopmentEpic | None:
    if task_id is None:
        return None
    return next((epic for epic in epics if task_id in epic.ordered_task_ids), None)


def _dependency_edges(tasks: list[Any]) -> tuple[tuple[str, str], ...]:
    edges: list[tuple[str, str]] = []
    task_ids = {str(_field(task, "task_id", "")) for task in tasks}
    for task in tasks:
        task_id = str(_field(task, "task_id", ""))
        for dep in tuple(_field(task, "dependent_tasks", ())) + tuple(
            _field(task, "dependencies", ())
        ):
            dep_id = str(dep)
            if dep_id in task_ids and dep_id != task_id:
                edges.append((dep_id, task_id))
    return tuple(dict.fromkeys(edges))


def _release_strategy(theme: str, tasks: list[Any]) -> str:
    if theme == "path_security" or any(str(_field(task, "risk_level", "")) == "high" for task in tasks):
        return "security_first"
    if theme in {"architecture_generalization", "framework_generalization"}:
        return "architecture_first"
    if theme == "plugin_ecosystem":
        return "api_first"
    if theme == "diagnostics_and_recovery":
        return "dx_first"
    return "behavior_first"


def _field(obj: Any, name: str, default: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)
