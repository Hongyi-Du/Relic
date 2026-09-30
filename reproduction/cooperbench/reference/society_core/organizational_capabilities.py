"""Event-grounded organizational capability formation ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .company_development import CompanyDevelopmentRun
from .company_safety import CompanySafetyReview
from .code_landing.evidence import evidence_level_at_least
from .hashing import canonicalize, stable_hash
from .org_adapter import CompanyOptimizationProposal
from .workspace_agent import CodingAgentLoopResult


CAPABILITY_KIND_ORGANIZATIONAL = "organizational"
CAPABILITY_KIND_TECHNICAL_FIX = "technical_fix"

# Closed, substrate-agnostic registry of organizational capability themes.
# A theme qualifies only when it names a shared workflow / review gate /
# tracker / role relation / protocol / governance object — the thesis-level
# definition of an organizational capability. Repository-specific technical
# themes (one codebase's bug list) must NEVER be added here: they are product
# fixes, arrive from the loaded substrate's own issue/PR/test semantics, and
# are classified as CAPABILITY_KIND_TECHNICAL_FIX by fallback.
# The ten capabilities the study preregisters. Reporting is per capability, so
# this tuple is the denominator a reader sees: "3 of 10 formed" is only readable
# if the 10 is fixed in advance and matches the write-up. Two of these
# (budget_rule, shared_memory) were named in the plan and absent here, which
# made them permanently unmeasurable — a capability that cannot appear in the
# denominator can never be reported as not having formed either.
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

# Capabilities the registry recognises that the study does NOT preregister.
# They stay classifiable (an organization that builds one has built something
# organizational) but must not silently enlarge the denominator of a
# "N of 10 preregistered capabilities formed" claim.
EXPLORATORY_CAPABILITIES: tuple[str, ...] = tuple(
    sorted(set(ORGANIZATIONAL_CAPABILITY_LABELS) - set(PREREGISTERED_CAPABILITIES))
)

# Generic engineering-theme labels: substrate-neutral prettification only.
# These themes describe technical/product work areas, not organizational
# structures, so they classify as CAPABILITY_KIND_TECHNICAL_FIX.
GENERAL_TECHNICAL_THEME_LABELS: dict[str, str] = {
    "architecture_generalization": "architecture generalization work",
    "dependency_resolution": "dependency resolution work",
    "diagnostics_and_recovery": "diagnostics and recovery work",
    "documentation_and_migration_path": "documentation and migration path work",
    "framework_generalization": "framework generalization work",
    "production_build_reliability": "production build reliability work",
}


def classify_capability_kind(theme: str) -> str:
    """Classify a ledger theme as organizational capability or technical fix.

    Conservative by construction: only themes in the closed organizational
    registry count toward the headline capability-formation metric; every
    other theme — including substrate-derived themes we have never seen —
    falls back to the technical-fix kind, so the headline can understate but
    never overstate organizational capability formation.
    """

    return (
        CAPABILITY_KIND_ORGANIZATIONAL
        if _slug(theme) in ORGANIZATIONAL_CAPABILITY_LABELS
        else CAPABILITY_KIND_TECHNICAL_FIX
    )


def capability_label(theme: str) -> str:
    slug = _slug(theme)
    if slug in ORGANIZATIONAL_CAPABILITY_LABELS:
        return ORGANIZATIONAL_CAPABILITY_LABELS[slug]
    if slug in GENERAL_TECHNICAL_THEME_LABELS:
        return GENERAL_TECHNICAL_THEME_LABELS[slug]
    return slug.replace("_", " ")


@dataclass(frozen=True)
class CapabilityEvent:
    event_id: str
    capability_id: str
    theme: str
    stage: str
    source: str
    support_refs: tuple[str, ...] = ()
    state_impact: str | None = None
    evidence_hash: str = ""


@dataclass(frozen=True)
class CapabilityUseEvidence:
    event_id: str
    actor_id: str
    tick: int
    shared_artifact_ref: str
    episode_id: str | None = None
    task_id: str | None = None
    support_refs: tuple[str, ...] = ()
    occurred_after_adoption: bool = True
    successful: bool = True


@dataclass(frozen=True)
class CapabilityEnforcementEvidence:
    event_id: str
    violation_event_id: str
    state_impact_ref: str
    support_refs: tuple[str, ...] = ()
    blocked_or_repaired: bool = True


@dataclass(frozen=True)
class CapabilityTransferEvidence:
    event_id: str
    held_out_target_id: str
    capability_invocation_ref: str
    baseline_outcome: float
    capability_outcome: float
    support_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityEvidenceBundle:
    shared_artifact_refs: tuple[str, ...] = ()
    origin_episode_id: str | None = None
    origin_task_id: str | None = None
    use_events: tuple[CapabilityUseEvidence, ...] = ()
    enforcement_events: tuple[CapabilityEnforcementEvidence, ...] = ()
    transfer_events: tuple[CapabilityTransferEvidence, ...] = ()


@dataclass(frozen=True)
class OrganizationalCapability:
    capability_id: str
    theme: str
    label: str
    kind: str
    level: int
    level_label: str
    proposal_count: int = 0
    adoption_count: int = 0
    repeated_use_count: int = 0
    persistence_count: int = 0
    enforcement_count: int = 0
    repair_count: int = 0
    transfer_count: int = 0
    support_event_refs: tuple[str, ...] = ()
    evidence_hash: str = ""
    criteria: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityLedger:
    ledger_id: str
    artifact_id: str
    case_id: str
    capabilities: tuple[OrganizationalCapability, ...]
    events: tuple[CapabilityEvent, ...]
    summary: dict[str, Any]
    privacy_boundary: str = "public_feedback_and_workspace_evidence_only"

    def hash(self) -> str:
        return stable_hash(self)


def build_capability_ledger(
    *,
    case_id: str,
    artifact_id: str,
    proposal: CompanyOptimizationProposal,
    safety_review: CompanySafetyReview,
    development_run: CompanyDevelopmentRun,
    workspace_development_run: CodingAgentLoopResult | None = None,
    public_trace_count: int = 0,
    real_product_experience_summary: dict[str, Any] | None = None,
    transfer_evidence_refs: tuple[str, ...] = (),
    transfer_evidence_by_theme: dict[str, tuple[str, ...]] | None = None,
    capability_evidence_by_theme: dict[str, CapabilityEvidenceBundle] | None = None,
) -> CapabilityLedger:
    """Build a paper-facing ledger from already-public company evidence.

    L2+ requires proposal, adoption, repeated use, and persistence evidence.
    L3 adds enforcement/blocking or repair. L4 additionally requires explicit
    held-out transfer evidence, which single historical runs normally do not have.
    """

    real_product_experience_summary = real_product_experience_summary or {}
    transfer_evidence_by_theme = transfer_evidence_by_theme or {}
    capability_evidence_by_theme = capability_evidence_by_theme or {}
    themes = _ordered_themes(
        proposal.priority_themes,
        proposal.safety_obligation_themes,
        tuple(unit.theme for unit in development_run.change_units),
        tuple(obligation.theme for obligation in safety_review.obligations),
        _workspace_themes(workspace_development_run),
        tuple(transfer_evidence_by_theme),
        tuple(capability_evidence_by_theme),
    )
    events: list[CapabilityEvent] = []
    capabilities: list[OrganizationalCapability] = []
    for theme in themes:
        capability_id = f"capability_{_slug(theme)}"
        theme_events = _events_for_theme(
            capability_id=capability_id,
            theme=theme,
            proposal=proposal,
            safety_review=safety_review,
            development_run=development_run,
            workspace_development_run=workspace_development_run,
            public_trace_count=public_trace_count,
            real_product_experience_summary=real_product_experience_summary,
            transfer_evidence_refs=transfer_evidence_refs,
            transfer_evidence_by_theme=transfer_evidence_by_theme,
            evidence_bundle=capability_evidence_by_theme.get(
                _slug(theme), CapabilityEvidenceBundle()
            ),
        )
        events.extend(theme_events)
        capabilities.append(_capability_from_events(capability_id=capability_id, theme=theme, events=theme_events))

    summary = _summary(capabilities, events)
    payload = {
        "case_id": case_id,
        "artifact_id": artifact_id,
        "proposal_id": proposal.proposal_id,
        "safety_review_id": safety_review.review_id,
        "development_run_id": development_run.run_id,
        "workspace_run_id": workspace_development_run.run_id if workspace_development_run else None,
        "capabilities": capabilities,
        "events": events,
        "summary": summary,
    }
    return CapabilityLedger(
        ledger_id=f"capability_ledger_{stable_hash(payload)[:24]}",
        artifact_id=artifact_id,
        case_id=case_id,
        capabilities=tuple(capabilities),
        events=tuple(events),
        summary=summary,
    )


def _events_for_theme(
    *,
    capability_id: str,
    theme: str,
    proposal: CompanyOptimizationProposal,
    safety_review: CompanySafetyReview,
    development_run: CompanyDevelopmentRun,
    workspace_development_run: CodingAgentLoopResult | None,
    public_trace_count: int,
    real_product_experience_summary: dict[str, Any],
    transfer_evidence_refs: tuple[str, ...],
    transfer_evidence_by_theme: dict[str, tuple[str, ...]],
    evidence_bundle: CapabilityEvidenceBundle,
) -> tuple[CapabilityEvent, ...]:
    events: list[CapabilityEvent] = []
    decision = next(
        (
            item
            for item in proposal.theme_decisions
            if item.theme == theme
        ),
        None,
    )
    obligations = tuple(
        obligation
        for obligation in safety_review.obligations
        if obligation.theme == theme
    )
    obligation_refs = tuple(
        dict.fromkeys(
            ref
            for obligation in obligations
            for ref in obligation.support_refs[:4]
        )
    )
    support_refs = decision.support_refs[:12] if decision else obligation_refs
    if public_trace_count > 0:
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="external_signal",
                source="public_society_feedback",
                support_refs=support_refs,
                state_impact=f"public_trace_count={public_trace_count}",
            )
        )
    if theme in proposal.priority_themes:
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="proposal",
                source=proposal.optimizer_source,
                support_refs=support_refs,
                state_impact=proposal.requested_direction,
            )
        )
    if decision is not None and decision.status == "adopt":
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="adoption",
                source=proposal.optimizer_source,
                support_refs=decision.support_refs,
                state_impact="explicit_theme_decision_status=adopt",
            )
        )
    change_units = tuple(unit for unit in development_run.change_units if unit.theme == theme)
    if change_units:
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="implementation",
                source=development_run.source,
                support_refs=tuple(ref for unit in change_units for ref in unit.support_refs[:3]),
                state_impact=f"change_unit_count={len(change_units)}",
            )
        )
    workspace_theme_count = _workspace_theme_count(workspace_development_run, theme)
    if workspace_theme_count and _workspace_persistence_ready(workspace_development_run):
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="materialization",
                source=workspace_development_run.source_report_id or "workspace_agent_loop",
                support_refs=tuple(workspace_development_run.selected_files[:8]),
                state_impact=(
                    f"workspace_final_mode={workspace_development_run.final_mode};"
                    f"verified_effective_patches={workspace_development_run.verified_effective_patches}"
                ),
            )
        )
    if (
        obligations
        or theme in safety_review.added_themes
        or theme in proposal.safety_obligation_themes
    ):
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="safety_obligation",
                source=safety_review.source,
                support_refs=obligation_refs,
                state_impact="release_gate_obligation",
            )
        )
    if workspace_development_run and (
        workspace_development_run.rejected_by_gate > 0 or workspace_development_run.failed_commands
    ):
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="workspace_gate",
                source="guarded_workspace_kernel",
                support_refs=workspace_development_run.failed_commands[:8],
                state_impact=f"rejected_by_gate={workspace_development_run.rejected_by_gate}",
            )
        )
    if workspace_development_run and workspace_development_run.repair_iterations > 0:
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="workspace_repair",
                source="workspace_agent_repair_loop",
                support_refs=workspace_development_run.failed_commands[:8],
                state_impact=f"repair_iterations={workspace_development_run.repair_iterations}",
            )
        )
    theme_transfer_refs = transfer_evidence_by_theme.get(theme, transfer_evidence_refs)
    if theme_transfer_refs:
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="transfer_candidate",
                source="held_out_transfer_evaluator",
                support_refs=theme_transfer_refs,
                state_impact="held_out_episode_or_module_reuse",
            )
        )
    if int(real_product_experience_summary.get("observation_count", 0) or 0) > 0:
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="product_evidence",
                source="workspace_backed_product_experience",
                support_refs=(_optional_hash(real_product_experience_summary.get("observation_hash")),),
                state_impact=f"observations={real_product_experience_summary.get('observation_count')}",
            )
        )
    embedded_refs = set(evidence_bundle.shared_artifact_refs)
    if embedded_refs:
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="embeddedness",
                source="shared_organizational_artifact",
                support_refs=tuple(sorted(embedded_refs)),
                state_impact=f"shared_artifact_count={len(embedded_refs)}",
            )
        )
    qualified_uses: list[CapabilityUseEvidence] = []
    for use in evidence_bundle.use_events:
        qualified = (
            use.occurred_after_adoption
            and use.successful
            and use.shared_artifact_ref in embedded_refs
        )
        stage = "repeated_use" if qualified else "unqualified_use"
        if qualified:
            qualified_uses.append(use)
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage=stage,
                source="event_graph_capability_use",
                support_refs=(
                    use.event_id,
                    use.shared_artifact_ref,
                    *use.support_refs,
                ),
                state_impact=(
                    f"actor={use.actor_id};tick={use.tick};"
                    f"episode={use.episode_id or ''};task={use.task_id or ''}"
                ),
            )
        )
    use_units = {
        (
            use.actor_id,
            use.episode_id
            or use.task_id
            or use.shared_artifact_ref,
            int(use.tick),
        )
        for use in qualified_uses
    }
    repeated_use_ready = len(use_units) >= 2
    persists_beyond_origin = repeated_use_ready and any(
        (
            evidence_bundle.origin_episode_id
            and use.episode_id
            and use.episode_id != evidence_bundle.origin_episode_id
        )
        or (
            evidence_bundle.origin_task_id
            and use.task_id
            and use.task_id != evidence_bundle.origin_task_id
        )
        for use in qualified_uses
    )
    if persists_beyond_origin:
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage="persistence",
                source="event_graph_cross_episode_reuse",
                support_refs=tuple(use.event_id for use in qualified_uses),
                state_impact="use_after_origin_episode_or_task",
            )
        )
    for enforcement in evidence_bundle.enforcement_events:
        stage = (
            "enforcement"
            if enforcement.violation_event_id
            and enforcement.state_impact_ref
            and enforcement.blocked_or_repaired
            else "unqualified_enforcement"
        )
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage=stage,
                source="event_graph_capability_enforcement",
                support_refs=(
                    enforcement.event_id,
                    enforcement.violation_event_id,
                    enforcement.state_impact_ref,
                    *enforcement.support_refs,
                ),
                state_impact=(
                    f"blocked_or_repaired={enforcement.blocked_or_repaired}"
                ),
            )
        )
    for transfer in evidence_bundle.transfer_events:
        stage = (
            "transfer"
            if transfer.held_out_target_id
            and transfer.capability_invocation_ref
            and transfer.capability_outcome > transfer.baseline_outcome
            else "unqualified_transfer"
        )
        events.append(
            _event(
                capability_id=capability_id,
                theme=theme,
                stage=stage,
                source="held_out_capability_transfer_evaluator",
                support_refs=(
                    transfer.event_id,
                    transfer.held_out_target_id,
                    transfer.capability_invocation_ref,
                    *transfer.support_refs,
                ),
                state_impact=(
                    f"baseline={transfer.baseline_outcome};"
                    f"capability={transfer.capability_outcome}"
                ),
            )
        )
    return tuple(events)


def _capability_from_events(
    *,
    capability_id: str,
    theme: str,
    events: tuple[CapabilityEvent, ...],
) -> OrganizationalCapability:
    counts = {stage: sum(1 for event in events if event.stage == stage) for stage in _STAGES}
    criteria = {
        "proposal": counts["proposal"] > 0,
        "adoption": counts["adoption"] > 0,
        "organizational_embeddedness": counts["embeddedness"] > 0,
        "repeated_use": counts["repeated_use"] >= 2,
        "persistence": counts["persistence"] > 0,
        "enforcement_or_repair": counts["enforcement"] > 0 or counts["repair"] > 0,
        "transfer": counts["transfer"] > 0,
    }
    level = _capability_level(criteria)
    payload = {"capability_id": capability_id, "events": events, "criteria": criteria, "level": level}
    return OrganizationalCapability(
        capability_id=capability_id,
        theme=theme,
        label=capability_label(theme),
        kind=classify_capability_kind(theme),
        level=level,
        level_label=_LEVEL_LABELS[level],
        proposal_count=counts["proposal"],
        adoption_count=counts["adoption"],
        repeated_use_count=counts["repeated_use"],
        persistence_count=counts["persistence"],
        enforcement_count=counts["enforcement"],
        repair_count=counts["repair"],
        transfer_count=counts["transfer"],
        support_event_refs=tuple(event.event_id for event in events),
        evidence_hash=stable_hash(payload),
        criteria=criteria,
    )


def _capability_level(criteria: dict[str, bool]) -> int:
    base = (
        "proposal",
        "adoption",
        "organizational_embeddedness",
        "repeated_use",
        "persistence",
    )
    if (
        all(criteria[key] for key in base)
        and criteria["enforcement_or_repair"]
        and criteria["transfer"]
    ):
        return 4
    if all(criteria[key] for key in base) and criteria["enforcement_or_repair"]:
        return 3
    if all(criteria[key] for key in base):
        return 2
    if criteria["proposal"] and criteria["adoption"]:
        return 1
    if criteria["proposal"]:
        return 0
    return 0


def _summary(
    capabilities: list[OrganizationalCapability],
    events: list[CapabilityEvent],
) -> dict[str, Any]:
    organizational = [
        capability
        for capability in capabilities
        if capability.kind == CAPABILITY_KIND_ORGANIZATIONAL
    ]
    technical = [
        capability
        for capability in capabilities
        if capability.kind == CAPABILITY_KIND_TECHNICAL_FIX
    ]
    total = len(organizational)
    l2_plus = sum(1 for capability in organizational if capability.level >= 2)
    l3_plus = sum(1 for capability in organizational if capability.level >= 3)
    l4 = sum(1 for capability in organizational if capability.level >= 4)
    return {
        # Headline capability-formation metrics cover ORGANIZATIONAL
        # capabilities only; repository-level technical fixes are reported
        # separately below and never inflate the headline numbers.
        "headline_metric_scope": CAPABILITY_KIND_ORGANIZATIONAL,
        "capability_count": total,
        "event_count": len(events),
        "strongest_level": max((capability.level for capability in organizational), default=0),
        "l2_persistent_capability_count": l2_plus,
        "l3_institutionalized_capability_count": l3_plus,
        "l4_transfer_capability_count": l4,
        "l2_persistent_capability_rate": _ratio(l2_plus, total),
        "l3_institutionalized_capability_rate": _ratio(l3_plus, total),
        "l4_transfer_capability_rate": _ratio(l4, total),
        "claim_ready": l2_plus > 0,
        "institutionalization_claim_ready": l3_plus > 0,
        "transfer_claim_ready": l4 > 0,
        "level_counts": {
            _LEVEL_LABELS[level]: sum(1 for capability in organizational if capability.level == level)
            for level in range(5)
        },
        "organizational_capability_count": total,
        "technical_fix_count": len(technical),
        "all_ledger_entry_count": len(capabilities),
        "technical_fix_l2_plus_count": sum(
            1 for capability in technical if capability.level >= 2
        ),
        "technical_fix_strongest_level": max(
            (capability.level for capability in technical), default=0
        ),
        "technical_fix_level_counts": {
            _LEVEL_LABELS[level]: sum(1 for capability in technical if capability.level == level)
            for level in range(5)
        },
    }


def _event(
    *,
    capability_id: str,
    theme: str,
    stage: str,
    source: str,
    support_refs: tuple[str, ...] = (),
    state_impact: str | None = None,
) -> CapabilityEvent:
    payload = {
        "capability_id": capability_id,
        "theme": theme,
        "stage": stage,
        "source": source,
        "support_refs": support_refs,
        "state_impact": state_impact,
    }
    event_hash = stable_hash(payload)[:24]
    return CapabilityEvent(
        event_id=f"cap_evt_{_slug(theme)}_{stage}_{event_hash}",
        capability_id=capability_id,
        theme=theme,
        stage=stage,
        source=source,
        support_refs=tuple(ref for ref in support_refs if ref),
        state_impact=state_impact,
        evidence_hash=stable_hash(payload),
    )


def _ordered_themes(*theme_groups: tuple[str, ...]) -> tuple[str, ...]:
    themes: list[str] = []
    for group in theme_groups:
        for theme in group:
            cleaned = _slug(theme)
            if cleaned:
                themes.append(cleaned)
    return tuple(dict.fromkeys(themes))


def _workspace_themes(run: CodingAgentLoopResult | None) -> tuple[str, ...]:
    if run is None:
        return ()
    return tuple(theme for proposal in run.proposals for theme in proposal.themes)


def _workspace_theme_count(run: CodingAgentLoopResult | None, theme: str) -> int:
    if run is None:
        return 0
    if not _workspace_persistence_ready(run):
        return 0
    return sum(1 for proposal in run.proposals if theme in proposal.themes)


def _workspace_persistence_ready(run: CodingAgentLoopResult | None) -> bool:
    if run is None:
        return False
    if not evidence_level_at_least(run.evidence_level, "release_candidate"):
        return False
    return run.verified_effective_patches > 0 or any(
        result.status == "applied"
        and result.before_hash != result.after_hash
        and _counts_as_product_patch_path(result.path)
        for result in run.patch_results
    )


def _counts_as_product_patch_path(path: str) -> bool:
    return Path(path).as_posix() != "company-development/workspace-agent-report.json"


def _slug(value: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(value)).split())


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 12) if denominator else 0.0


def _optional_hash(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return stable_hash(canonicalize(value))


_STAGES = (
    "external_signal",
    "proposal",
    "adoption",
    "implementation",
    "materialization",
    "embeddedness",
    "repeated_use",
    "unqualified_use",
    "product_evidence",
    "persistence",
    "safety_obligation",
    "workspace_gate",
    "workspace_repair",
    "enforcement",
    "unqualified_enforcement",
    "repair",
    "transfer",
    "transfer_candidate",
    "unqualified_transfer",
)

_LEVEL_LABELS = {
    0: "L0_discussion_or_proposal",
    1: "L1_adopted_practice",
    2: "L2_persistent_capability",
    3: "L3_institutionalized_capability",
    4: "L4_transferable_capability",
}
