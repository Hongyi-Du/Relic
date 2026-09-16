"""Closed registry for paper-measured organizational capabilities."""

from __future__ import annotations

CAPABILITY_KIND_ORGANIZATIONAL = "organizational"
CAPABILITY_KIND_TECHNICAL_FIX = "technical_fix"

PREREGISTERED_CAPABILITIES: tuple[str, ...] = (
    "evidence_workflow",
    "review_gate",
    "experiment_tracker",
    "ownership_map",
    "budget_rule",
    "release_governance",
    "shared_memory",
    "source_credibility_workflow",
    "claim_evidence_binding",
    "customer_triage",
)

ORGANIZATIONAL_CAPABILITY_LABELS: dict[str, str] = {
    "budget_rule": "budget rule",
    "claim_evidence_binding": "claim-evidence binding workflow",
    "customer_triage": "customer triage workflow",
    "evidence_workflow": "evidence workflow",
    "experiment_tracker": "experiment tracker",
    "external_signal_loop": "external signal loop",
    "ownership_map": "ownership map",
    "privacy_boundary": "privacy boundary governance",
    "release_governance": "release governance",
    "review_gate": "review gate",
    "shared_memory": "shared memory",
    "source_credibility_workflow": "source credibility workflow",
    "workflow_integration": "workflow integration capability",
}

EXPLORATORY_CAPABILITIES: tuple[str, ...] = tuple(
    sorted(set(ORGANIZATIONAL_CAPABILITY_LABELS) - set(PREREGISTERED_CAPABILITIES))
)

GENERAL_TECHNICAL_THEME_LABELS: dict[str, str] = {
    "architecture_generalization": "architecture generalization work",
    "dependency_resolution": "dependency resolution work",
    "diagnostics_and_recovery": "diagnostics and recovery work",
    "documentation_and_migration_path": "documentation and migration path work",
    "framework_generalization": "framework generalization work",
    "production_build_reliability": "production build reliability work",
}


def _slug(value: str) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def classify_capability_kind(theme: str) -> str:
    """Classify only registered shared mechanisms as organizational."""

    return (
        CAPABILITY_KIND_ORGANIZATIONAL
        if _slug(theme) in ORGANIZATIONAL_CAPABILITY_LABELS
        else CAPABILITY_KIND_TECHNICAL_FIX
    )


def capability_label(theme: str) -> str:
    """Return a stable public label for an organizational or technical theme."""

    slug = _slug(theme)
    if slug in ORGANIZATIONAL_CAPABILITY_LABELS:
        return ORGANIZATIONAL_CAPABILITY_LABELS[slug]
    if slug in GENERAL_TECHNICAL_THEME_LABELS:
        return GENERAL_TECHNICAL_THEME_LABELS[slug]
    return slug.replace("_", " ")


__all__ = [
    "CAPABILITY_KIND_ORGANIZATIONAL",
    "CAPABILITY_KIND_TECHNICAL_FIX",
    "EXPLORATORY_CAPABILITIES",
    "GENERAL_TECHNICAL_THEME_LABELS",
    "ORGANIZATIONAL_CAPABILITY_LABELS",
    "PREREGISTERED_CAPABILITIES",
    "capability_label",
    "classify_capability_kind",
]
