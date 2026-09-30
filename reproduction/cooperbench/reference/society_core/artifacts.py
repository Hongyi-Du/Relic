"""Artifact fixtures used by Society-Core experiments."""

from __future__ import annotations

from .schemas import Artifact


def lantern_scout_artifact() -> Artifact:
    return Artifact(
        id="lantern_scout",
        provider_id="lanternforge",
        artifact_kind="tool",
        public_claims=("evidence-grounded research assistance",),
        capability_profile={"research_trace": 0.7, "source_checking": 0.6},
        cost_profile={"attention": 0.2, "time": 0.2},
        failure_modes=("overclaim", "source_mismatch", "workflow_friction"),
        reliability_profile={"claim_evidence_linking": 0.55, "source_organization": 0.65},
        task_fit_distribution={"research_task": 0.75, "casual_search": 0.35},
    )


def community_checklist_artifact() -> Artifact:
    return Artifact(
        id="community_checklist",
        provider_id=None,
        artifact_kind="tool",
        public_claims=("evidence checklist for public claims",),
        capability_profile={"source_checking": 0.55, "claim_tracking": 0.5},
        cost_profile={"attention": 0.12, "time": 0.12},
        failure_modes=("incomplete_check", "too_slow"),
        reliability_profile={"claim_evidence_linking": 0.5},
        task_fit_distribution={"research_task": 0.55, "conversation": 0.4},
    )


def placebo_artifact() -> Artifact:
    return Artifact(
        id="placebo_tool",
        provider_id=None,
        artifact_kind="tool",
        public_claims=("generic public tool with unclear usefulness",),
        capability_profile={"source_checking": 0.15, "claim_tracking": 0.1},
        cost_profile={"attention": 0.12, "time": 0.12},
        failure_modes=("low_task_fit", "unclear_value"),
        reliability_profile={"claim_evidence_linking": 0.1},
        task_fit_distribution={"research_task": 0.15, "conversation": 0.15},
    )
