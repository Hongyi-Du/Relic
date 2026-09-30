"""Boundary adapter between OrgEnv and Society-Core.

The adapter is deliberately narrow: OrgEnv can publish an artifact into the
external society, and the company can later read sanitized public traces. Hidden
company state and private society state do not cross this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .hashing import stable_hash
from .product_experience import CREATIVE_FEATURE_THEMES
from .schemas import Artifact, ContentItem, ExternalEvent, SocietyState


STRATEGIC_UPGRADE_THEMES: frozenset[str] = frozenset(
    {
        "architecture_generalization",
        "plugin_ecosystem",
        "framework_generalization",
        "platform_expansion",
        "ecosystem_strategy",
    }
)


@dataclass(frozen=True)
class CompanyVisibleTrace:
    event_id: str
    tick: int
    action_type: str | None
    artifact_id: str | None
    content_id: str | None
    channel_id: str | None
    public_summary: str
    support_refs: tuple[str, ...]
    upgrade_suggestion_id: str | None = None
    upgrade_theme: str | None = None
    upgrade_priority: float | None = None
    upgrade_confidence: float | None = None
    upgrade_contribution_mode: str | None = None
    upgrade_technical_depth: float | None = None
    upgrade_novelty_score: float | None = None
    author_id: str | None = None
    source_experience_ref: str | None = None
    objective_observation_refs: tuple[str, ...] = ()
    objective_observation_set_hash: str | None = None
    objective_observation_manifest_hash: str | None = None
    generator_source: str | None = None
    requested_change: str | None = None
    semantic_cluster_id: str | None = None


@dataclass(frozen=True)
class CompanyArtifactFeedback:
    artifact_id: str
    public_mentions: int
    trial_events: int
    paid_events: int
    abandonment_events: int
    failure_reports: int
    tutorial_events: int
    requested_direction: str
    support_refs: tuple[str, ...]
    upgrade_suggestion_events: int = 0
    top_upgrade_themes: tuple[str, ...] = ()
    theme_support: dict[str, dict[str, float]] | None = None
    feedback_uncertainty: dict[str, float] | None = None
    technical_review_events: int = 0
    feature_proposal_events: int = 0
    technical_feature_proposal_events: int = 0
    top_technical_themes: tuple[str, ...] = ()
    top_feature_themes: tuple[str, ...] = ()
    theme_support_refs: dict[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class CompanyPublicFeedbackReport:
    report_id: str
    since_tick: int
    trace_count: int
    artifact_feedback: dict[str, CompanyArtifactFeedback]
    caveats: tuple[str, ...] = ()


ThemeDecisionStatus = Literal["adopt", "defer", "reject"]


@dataclass(frozen=True)
class ThemeDecisionEvidence:
    raw_support_count: int = 0
    independent_support_count: int = 0
    mean_confidence: float = 0.0
    mean_technical_depth: float = 0.0
    mean_novelty: float = 0.0
    independent_technical_review_count: int = 0
    independent_feature_proposal_count: int = 0
    independent_technical_feature_proposal_count: int = 0


@dataclass(frozen=True)
class ThemeDecision:
    theme: str
    status: ThemeDecisionStatus
    score: float
    rationale: str
    support_refs: tuple[str, ...] = ()
    evidence: ThemeDecisionEvidence = field(default_factory=ThemeDecisionEvidence)

    def __post_init__(self) -> None:
        if self.status not in {"adopt", "defer", "reject"}:
            raise ValueError(f"Unsupported theme decision status: {self.status}")


@dataclass(frozen=True)
class CompanyOptimizationProposal:
    proposal_id: str
    artifact_id: str
    requested_direction: str
    priority_themes: tuple[str, ...]
    support_refs: tuple[str, ...]
    public_trace_hash: str
    historical_target_overlap: float | None = None
    optimizer_source: str = "deterministic_public_feedback"
    optimizer_model: str | None = None
    optimizer_reasoning: str | None = None
    theme_decisions: tuple[ThemeDecision, ...] = ()
    safety_obligation_themes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.theme_decisions:
            return
        adopted = tuple(
            dict.fromkeys(
                decision.theme
                for decision in self.theme_decisions
                if decision.status == "adopt"
            )
        )
        adopted_set = set(adopted)
        normalized = tuple(
            dict.fromkeys(
                (
                    *(theme for theme in self.priority_themes if theme in adopted_set),
                    *adopted,
                )
            )
        )
        if normalized != self.priority_themes:
            object.__setattr__(self, "priority_themes", normalized)


def _slug(value: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def artifact_from_public_org_release(world: Any, *, release_time: int = 0) -> Artifact:
    cfg = dict(getattr(world, "company_config", {}) or {})
    product = getattr(world, "product", None)
    product_name = cfg.get("product_name") or getattr(product, "name", "LanternScout")
    company_name = cfg.get("company_name", "LanternForge")
    product_purpose = cfg.get("product_purpose") or getattr(product, "summary", "")
    artifact_id = _slug(product_name) or "public_org_artifact"
    provider_id = _slug(company_name) or None
    return Artifact(
        id=artifact_id,
        provider_id=provider_id,
        artifact_kind="tool",
        public_claims=(str(product_purpose),) if product_purpose else (),
        evidence_claims=(),
        capability_profile={},
        cost_profile={},
        failure_modes=("public_claim_overreach", "workflow_friction"),
        release_time=release_time,
    )


def _content_public_summary(content: ContentItem | None) -> str:
    if content is None:
        return ""
    summary = content.text_surface.strip()
    if not summary:
        summary = f"{content.kind.value} by {content.author_id}"
    return summary[:500]


def is_company_visible_public_trace(event: ExternalEvent, content: ContentItem | None = None) -> bool:
    if event.public_visibility != "public":
        return False
    if event.artifact_id:
        return True
    if content and content.visibility_scope == "public" and content.artifact_refs:
        return True
    if event.kind.value in {
        "new_artifact_release",
        "artifact_failure_report",
        "user_tutorial",
        "expert_review_like_post",
        "competitor_claim_like_post",
        "public_apology_or_correction",
    }:
        return True
    return bool(event.company_visible_flag)


def extract_company_visible_public_traces(
    state: SocietyState,
    *,
    since_tick: int = 0,
    max_items: int = 100,
) -> tuple[CompanyVisibleTrace, ...]:
    traces: list[CompanyVisibleTrace] = []
    forbidden_provenance_refs = {
        content.id
        for content in state.content.values()
        if content.visibility_scope != "public"
    }
    forbidden_provenance_refs.update(
        event.event_id
        for event in state.event_log
        if event.public_visibility != "public"
    )
    for event in state.event_log:
        if event.tick < since_tick:
            continue
        content = state.content.get(event.content_id) if event.content_id else None
        if not is_company_visible_public_trace(event, content):
            continue
        artifact_refs = tuple(content.artifact_refs) if content else ()
        evidence_refs = _without_private_refs(
            tuple(content.evidence_refs) if content else (),
            forbidden_provenance_refs,
        )
        summary = _content_public_summary(content)
        if not summary:
            summary = " ".join(
                part
                for part in (event.action_type, event.artifact_id, event.channel_id)
                if part
            )
        event_payload = event.typed_payload if isinstance(event.typed_payload, dict) else {}
        suggestion_payload = _upgrade_suggestion_payload(event_payload)
        suggestion_id = _string_or_none(suggestion_payload.get("suggestion_id"))
        suggestion = (
            state.upgrade_suggestions.get(suggestion_id) if suggestion_id else None
        )
        artifact_id = event.artifact_id or (
            artifact_refs[0] if artifact_refs else None
        )
        author_id = _first_string(
            event.actor_id,
            getattr(suggestion, "author_id", None),
            suggestion_payload.get("author_id"),
            content.author_id if content else None,
        )
        source_experience_ref = _first_string(
            getattr(suggestion, "source_experience_ref", None),
            suggestion_payload.get("source_experience_ref"),
            event_payload.get("source_experience_ref"),
        )
        requested_change = _first_string(
            getattr(suggestion, "requested_change", None),
            suggestion_payload.get("requested_change"),
            event_payload.get("requested_change"),
        )
        upgrade_theme = _first_string(
            getattr(suggestion, "theme", None), suggestion_payload.get("theme")
        )
        objective_observation_refs = _objective_observation_refs(
            event=event,
            content=content,
            suggestion=suggestion,
            suggestion_payload=suggestion_payload,
            event_payload=event_payload,
            source_experience_ref=source_experience_ref,
            forbidden_refs=forbidden_provenance_refs,
        )
        evidence_refs_for_provenance = _evidence_refs_for_provenance(
            content=content,
            suggestion=suggestion,
            suggestion_payload=suggestion_payload,
        )
        observation_set_hash = _first_string(
            getattr(suggestion, "objective_observation_set_hash", None),
            suggestion_payload.get("objective_observation_set_hash"),
            event_payload.get("objective_observation_set_hash"),
            _prefixed_ref_value(
                evidence_refs_for_provenance, "observation_set_hash:"
            ),
        )
        if not observation_set_hash and objective_observation_refs:
            observation_set_hash = stable_hash(objective_observation_refs)
        observation_manifest_hash = _first_string(
            getattr(suggestion, "objective_observation_manifest_hash", None),
            suggestion_payload.get("objective_observation_manifest_hash"),
            event_payload.get("objective_observation_manifest_hash"),
            _prefixed_ref_value(
                evidence_refs_for_provenance, "observation_manifest_hash:"
            ),
        )
        generator_source = _first_string(
            getattr(suggestion, "subjective_feedback_source", None),
            suggestion_payload.get("generator_source"),
            suggestion_payload.get("subjective_feedback_source"),
            event_payload.get("generator_source"),
        )
        traces.append(
            CompanyVisibleTrace(
                event_id=event.event_id,
                tick=event.tick,
                action_type=event.action_type,
                artifact_id=artifact_id,
                content_id=event.content_id,
                channel_id=event.channel_id,
                public_summary=summary[:500],
                support_refs=_without_private_refs(
                    tuple(
                        ref
                        for ref in (
                        event.content_id,
                        event.artifact_id,
                        *artifact_refs,
                        *evidence_refs,
                    )
                        if ref
                    ),
                    forbidden_provenance_refs,
                ),
                upgrade_suggestion_id=suggestion_id,
                upgrade_theme=upgrade_theme,
                upgrade_priority=_float_or_none(
                    getattr(suggestion, "priority", None)
                    if suggestion is not None
                    else suggestion_payload.get("priority")
                ),
                upgrade_confidence=_float_or_none(
                    getattr(suggestion, "confidence", None)
                    if suggestion is not None
                    else suggestion_payload.get("confidence")
                ),
                upgrade_contribution_mode=_string_or_none(
                    getattr(suggestion, "contribution_mode", None)
                    if suggestion is not None
                    else suggestion_payload.get("contribution_mode")
                ),
                upgrade_technical_depth=_float_or_none(
                    getattr(suggestion, "technical_depth", None)
                    if suggestion is not None
                    else suggestion_payload.get("technical_depth")
                ),
                upgrade_novelty_score=_float_or_none(
                    getattr(suggestion, "novelty_score", None)
                    if suggestion is not None
                    else suggestion_payload.get("novelty_score")
                ),
                author_id=author_id,
                source_experience_ref=source_experience_ref,
                objective_observation_refs=objective_observation_refs,
                objective_observation_set_hash=observation_set_hash,
                objective_observation_manifest_hash=observation_manifest_hash,
                generator_source=generator_source,
                requested_change=requested_change,
                semantic_cluster_id=_semantic_cluster_id(
                    artifact_id=artifact_id,
                    theme=upgrade_theme,
                    requested_change=requested_change or summary,
                ),
            )
        )
        if len(traces) >= max_items:
            break
    return tuple(traces)


def company_visible_trace_hash(traces: tuple[CompanyVisibleTrace, ...]) -> str:
    return stable_hash(traces)


def _upgrade_suggestion_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    suggestion = payload.get("upgrade_suggestion")
    return suggestion if isinstance(suggestion, dict) else {}


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_string(*values: Any) -> str | None:
    for value in values:
        normalized = _string_or_none(value)
        if normalized is not None:
            return normalized
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _evidence_refs_for_provenance(
    *,
    content: ContentItem | None,
    suggestion: Any,
    suggestion_payload: dict[str, Any],
) -> tuple[str, ...]:
    refs = (
        *_string_tuple(getattr(suggestion, "evidence_refs", ())),
        *_string_tuple(suggestion_payload.get("evidence_refs")),
        *(tuple(content.evidence_refs) if content else ()),
    )
    return tuple(dict.fromkeys(refs))


def _objective_observation_refs(
    *,
    event: ExternalEvent,
    content: ContentItem | None,
    suggestion: Any,
    suggestion_payload: dict[str, Any],
    event_payload: dict[str, Any],
    source_experience_ref: str | None,
    forbidden_refs: set[str],
) -> tuple[str, ...]:
    if event.observation_refs:
        return _without_private_refs(
            tuple(dict.fromkeys(event.observation_refs)),
            forbidden_refs,
        )
    explicit_refs = _string_tuple(
        suggestion_payload.get("objective_observation_refs")
    ) or _string_tuple(event_payload.get("objective_observation_refs"))
    if explicit_refs:
        return _without_private_refs(
            tuple(dict.fromkeys(explicit_refs)),
            forbidden_refs,
        )
    evidence_refs = _evidence_refs_for_provenance(
        content=content,
        suggestion=suggestion,
        suggestion_payload=suggestion_payload,
    )
    metadata_prefixes = (
        "observation_set_hash:",
        "observation_manifest_hash:",
        "failure:",
        "task:",
        "blocked_stage:",
        "stage:",
        "stage_error:",
    )
    return _without_private_refs(
        tuple(
            ref
            for ref in evidence_refs
            if ref != source_experience_ref
            and not ref.startswith(metadata_prefixes)
        ),
        forbidden_refs,
    )


def _without_private_refs(
    refs: tuple[str, ...],
    forbidden_refs: set[str],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(ref for ref in refs if ref not in forbidden_refs)
    )


def _prefixed_ref_value(refs: tuple[str, ...], prefix: str) -> str | None:
    for ref in refs:
        if ref.startswith(prefix) and len(ref) > len(prefix):
            return ref[len(prefix) :]
    return None


def _normalize_semantic_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(
        "".join(
            character.casefold() if character.isalnum() else " "
            for character in value
        ).split()
    )


def _semantic_cluster_id(
    *,
    artifact_id: str | None,
    theme: str | None,
    requested_change: str | None,
) -> str:
    fingerprint = {
        "artifact": _normalize_semantic_text(artifact_id),
        "theme": _normalize_semantic_text(theme),
        "requested_change": _normalize_semantic_text(requested_change),
    }
    return f"semantic_cluster_{stable_hash(fingerprint)[:24]}"


def _direction_from_counts(
    *,
    failure_reports: int,
    paid_events: int,
    trial_events: int,
    tutorial_events: int,
    abandonment_events: int,
    upgrade_suggestion_events: int,
) -> str:
    if upgrade_suggestion_events >= 2 and failure_reports >= 1:
        return "prioritize_public_reliability_upgrade_requests"
    if upgrade_suggestion_events >= 2:
        return "prioritize_public_upgrade_suggestions"
    if abandonment_events > paid_events and abandonment_events >= 2:
        return "reduce_friction_or_price"
    if failure_reports >= 2 and tutorial_events >= 2:
        return "improve_reliability_and_documentation"
    if failure_reports >= max(2, tutorial_events):
        return "improve_reliability"
    if tutorial_events >= 2 and paid_events >= 1:
        return "amplify_documentation"
    if trial_events >= 3 and paid_events == 0:
        return "improve_value_or_pricing"
    if paid_events >= 2:
        return "continue_current_direction"
    return "insufficient_public_evidence"


def build_company_public_feedback_report(
    traces: tuple[CompanyVisibleTrace, ...],
    *,
    since_tick: int = 0,
) -> CompanyPublicFeedbackReport:
    by_artifact: dict[str, list[CompanyVisibleTrace]] = {}
    for trace in traces:
        if trace.tick < since_tick or not trace.artifact_id:
            continue
        by_artifact.setdefault(trace.artifact_id, []).append(trace)
    feedback: dict[str, CompanyArtifactFeedback] = {}
    for artifact_id, artifact_traces in sorted(by_artifact.items()):
        action_types = [trace.action_type for trace in artifact_traces]
        failure_reports = sum(1 for action in action_types if action in {"share_failure", "warn_peer"})
        tutorial_events = sum(1 for action in action_types if action == "write_tutorial")
        trial_events = sum(1 for action in action_types if action in {"inspect_artifact", "try_artifact_on_task", "reuse_artifact"})
        paid_events = sum(1 for action in action_types if action == "pay_for_artifact")
        abandonment_events = sum(1 for action in action_types if action == "abandon_artifact")
        upgrade_suggestion_events = sum(1 for action in action_types if action == "suggest_upgrade")
        top_upgrade_themes = _top_upgrade_themes(artifact_traces)
        role_theme_summary = _role_theme_summary(artifact_traces)
        feedback[artifact_id] = CompanyArtifactFeedback(
            artifact_id=artifact_id,
            public_mentions=len(artifact_traces),
            trial_events=trial_events,
            paid_events=paid_events,
            abandonment_events=abandonment_events,
            failure_reports=failure_reports,
            tutorial_events=tutorial_events,
            requested_direction=_direction_from_counts(
                failure_reports=failure_reports,
                paid_events=paid_events,
                trial_events=trial_events,
                tutorial_events=tutorial_events,
                abandonment_events=abandonment_events,
                upgrade_suggestion_events=upgrade_suggestion_events,
            ),
            support_refs=tuple(trace.event_id for trace in artifact_traces[:30]),
            upgrade_suggestion_events=upgrade_suggestion_events,
            top_upgrade_themes=top_upgrade_themes,
            theme_support=_theme_support(artifact_traces),
            feedback_uncertainty=_feedback_uncertainty(
                public_mentions=len(artifact_traces),
                upgrade_suggestion_events=upgrade_suggestion_events,
                failure_reports=failure_reports,
                paid_events=paid_events,
            ),
            technical_review_events=role_theme_summary["technical_review_events"],
            feature_proposal_events=role_theme_summary["feature_proposal_events"],
            technical_feature_proposal_events=role_theme_summary[
                "technical_feature_proposal_events"
            ],
            top_technical_themes=role_theme_summary["top_technical_themes"],
            top_feature_themes=role_theme_summary["top_feature_themes"],
            theme_support_refs=_theme_support_refs(artifact_traces),
        )
    caveats = () if feedback else ("no_company_visible_artifact_feedback",)
    return CompanyPublicFeedbackReport(
        report_id="society_core_company_public_feedback_v17",
        since_tick=since_tick,
        trace_count=len(traces),
        artifact_feedback=feedback,
        caveats=caveats,
    )


def propose_company_optimization_from_public_feedback(
    report: CompanyPublicFeedbackReport,
    *,
    artifact_id: str,
    target_improvement_themes: tuple[str, ...] = (),
) -> CompanyOptimizationProposal:
    del target_improvement_themes
    feedback = report.artifact_feedback.get(artifact_id)
    if feedback is None:
        decisions = (
            ThemeDecision(
                theme="collect_more_public_feedback",
                status="defer",
                score=0.0,
                rationale="no_company_visible_artifact_feedback",
            ),
        )
        support_refs: tuple[str, ...] = ()
        requested_direction = "insufficient_public_evidence"
    else:
        requested_direction = feedback.requested_direction
        decisions = _theme_decisions_from_feedback(feedback)
        support_refs = feedback.support_refs
    themes = tuple(
        decision.theme for decision in decisions if decision.status == "adopt"
    )
    proposal_id = stable_hash(
        {
            "report_id": report.report_id,
            "artifact_id": artifact_id,
            "requested_direction": requested_direction,
            "themes": themes,
            "theme_decisions": decisions,
            "support_refs": support_refs,
        }
    )[:24]
    return CompanyOptimizationProposal(
        proposal_id=f"proposal_{artifact_id}_{proposal_id}",
        artifact_id=artifact_id,
        requested_direction=requested_direction,
        priority_themes=themes,
        support_refs=support_refs,
        public_trace_hash=stable_hash(report),
        historical_target_overlap=None,
        theme_decisions=decisions,
    )


def _candidate_themes_from_feedback(
    feedback: CompanyArtifactFeedback,
) -> tuple[str, ...]:
    themes: list[str] = [
        *feedback.top_technical_themes,
        *feedback.top_feature_themes,
        *feedback.top_upgrade_themes,
    ]
    if feedback.failure_reports >= 1:
        themes.append("production_build_reliability")
        themes.append("dependency_resolution")
    if feedback.abandonment_events > feedback.paid_events:
        themes.append("documentation_and_migration_path")
        themes.append("reduce_friction_or_price")
    if feedback.trial_events >= 1 and feedback.paid_events == 0:
        themes.append("clarify_value_proposition")
    if feedback.tutorial_events >= 1:
        themes.append("documentation_and_migration_path")
    if feedback.paid_events >= 1:
        themes.append("amplify_success_cases")
    if not themes:
        themes.append("collect_more_public_feedback")
    return tuple(dict.fromkeys(themes))


def _theme_decisions_from_feedback(
    feedback: CompanyArtifactFeedback,
) -> tuple[ThemeDecision, ...]:
    support = feedback.theme_support or {}
    refs_by_theme = feedback.theme_support_refs or {}
    decisions = [
        _theme_decision(
            theme=theme,
            values=support.get(theme) or {},
            support_refs=refs_by_theme.get(theme, ()),
        )
        for theme in _candidate_themes_from_feedback(feedback)
    ]
    decisions.sort(
        key=lambda decision: (
            decision.status != "adopt",
            -decision.score,
            decision.theme,
        )
    )
    return tuple(decisions)


def _themes_from_feedback(feedback: CompanyArtifactFeedback) -> tuple[str, ...]:
    return tuple(
        decision.theme
        for decision in _theme_decisions_from_feedback(feedback)
        if decision.status == "adopt"
    )


def _decision_evidence(values: dict[str, float]) -> ThemeDecisionEvidence:
    return ThemeDecisionEvidence(
        raw_support_count=int(values.get("raw_count", values.get("count", 0.0))),
        independent_support_count=int(
            values.get("independent_support_count", values.get("count", 0.0))
        ),
        mean_confidence=_bounded_evidence_value(values.get("mean_confidence")),
        mean_technical_depth=_bounded_evidence_value(
            values.get("mean_technical_depth")
        ),
        mean_novelty=_bounded_evidence_value(values.get("mean_novelty")),
        independent_technical_review_count=int(
            _independent_role_count(
                values,
                independent_key="independent_technical_review_count",
                raw_key="technical_review_count",
            )
        ),
        independent_feature_proposal_count=int(
            _independent_role_count(
                values,
                independent_key="independent_feature_proposal_count",
                raw_key="feature_proposal_count",
            )
        ),
        independent_technical_feature_proposal_count=int(
            _independent_role_count(
                values,
                independent_key="independent_technical_feature_proposal_count",
                raw_key="technical_feature_proposal_count",
            )
        ),
    )


def _adoption_arms(evidence: ThemeDecisionEvidence) -> tuple[bool, bool, bool]:
    """The three evidence patterns that make a theme adoptable.

    Shared between the per-theme decision and top_upgrade_themes candidacy so
    a theme whose evidence would be adopted can never be dropped from the
    candidate list before its decision is made.
    """

    expert_support = (
        evidence.independent_technical_review_count >= 1
        and evidence.mean_confidence >= 0.65
        and evidence.mean_technical_depth >= 0.75
    )
    cross_role_technical_support = (
        evidence.independent_technical_feature_proposal_count >= 1
        and evidence.mean_confidence >= 0.65
        and evidence.mean_technical_depth >= 0.70
        and evidence.mean_novelty >= 0.75
    )
    multiple_qualified_supporters = (
        evidence.independent_support_count >= 2
        and evidence.mean_confidence >= 0.65
        and (
            evidence.mean_technical_depth >= 0.65
            or evidence.mean_novelty >= 0.70
        )
    )
    return expert_support, cross_role_technical_support, multiple_qualified_supporters


def _theme_decision(
    *,
    theme: str,
    values: dict[str, float],
    support_refs: tuple[str, ...],
) -> ThemeDecision:
    evidence = _decision_evidence(values)
    expert_support, cross_role_technical_support, multiple_qualified_supporters = (
        _adoption_arms(evidence)
    )
    score = min(
        1.0,
        0.40 * evidence.mean_confidence
        + 0.30 * evidence.mean_technical_depth
        + 0.30 * evidence.mean_novelty
        + 0.05 * min(1.0, evidence.independent_support_count / 3.0)
        + 0.08 * float(expert_support)
        + 0.10 * float(cross_role_technical_support),
    )
    if cross_role_technical_support:
        status: ThemeDecisionStatus = "adopt"
        rationale = "single_high_confidence_cross_role_technical_contribution"
    elif expert_support:
        status = "adopt"
        rationale = "single_high_confidence_expert_technical_contribution"
    elif multiple_qualified_supporters:
        status = "adopt"
        rationale = "multiple_independent_qualified_supporters"
    elif evidence.independent_support_count == 0:
        status = "defer"
        rationale = "no_theme_specific_qualified_support"
    elif evidence.mean_confidence < 0.65:
        status = "defer"
        rationale = "theme_specific_support_below_confidence_threshold"
    elif evidence.independent_support_count == 1:
        status = "defer"
        rationale = "single_nonexpert_source_requires_independent_confirmation"
    else:
        status = "defer"
        rationale = "independent_support_lacks_required_depth_or_novelty"
    return ThemeDecision(
        theme=theme,
        status=status,
        score=round(score, 12),
        rationale=rationale,
        support_refs=tuple(dict.fromkeys(support_refs))[:12],
        evidence=evidence,
    )


def _bounded_evidence_value(value: Any) -> float:
    numeric = _float_or_none(value)
    if numeric is None:
        return 0.0
    return max(0.0, min(1.0, numeric))


def _evidence_grounded_role_themes(
    feedback: CompanyArtifactFeedback,
) -> tuple[str, ...]:
    support = feedback.theme_support or {}
    candidates = tuple(
        dict.fromkeys((*feedback.top_technical_themes, *feedback.top_feature_themes))
    )
    qualified: list[tuple[str, float]] = []
    for theme in candidates:
        values = support.get(theme) or {}
        confidence = float(values.get("mean_confidence", 0.0))
        technical_depth = float(values.get("mean_technical_depth", 0.0))
        novelty = float(values.get("mean_novelty", 0.0))
        independent_support = float(
            values.get(
                "independent_support_count", values.get("count", 0.0)
            )
        )
        technical_reviews = _independent_role_count(
            values,
            independent_key="independent_technical_review_count",
            raw_key="technical_review_count",
        )
        feature_proposals = _independent_role_count(
            values,
            independent_key="independent_feature_proposal_count",
            raw_key="feature_proposal_count",
        )
        technical_features = _independent_role_count(
            values,
            independent_key="independent_technical_feature_proposal_count",
            raw_key="technical_feature_proposal_count",
        )
        cross_role = (
            technical_features >= 1.0
            and technical_depth >= 0.70
            and novelty >= 0.75
            and confidence >= 0.65
        )
        expert_review = (
            technical_reviews >= 1.0
            and technical_depth >= 0.75
            and confidence >= 0.65
        )
        creative_support = (
            feature_proposals >= 2.0
            and novelty >= 0.75
            and confidence >= 0.65
        )
        if not (cross_role or expert_review or creative_support):
            continue
        role_bonus = 0.20 if cross_role else 0.10 if expert_review else 0.0
        score = (
            role_bonus
            + 0.34 * confidence
            + 0.33 * technical_depth
            + 0.33 * novelty
            + 0.05 * min(1.0, independent_support / 3.0)
        )
        qualified.append((theme, score))
    return tuple(
        theme
        for theme, _ in sorted(
            qualified,
            key=lambda item: (-item[1], item[0]),
        )
    )


def _independent_role_count(
    values: dict[str, float], *, independent_key: str, raw_key: str
) -> float:
    if independent_key in values:
        return float(values[independent_key])
    raw_count = float(values.get(raw_key, 0.0))
    if "independent_support_count" in values:
        return min(raw_count, float(values["independent_support_count"]))
    return raw_count


def _theme_support(traces: list[CompanyVisibleTrace]) -> dict[str, dict[str, float]]:
    traces_by_theme: dict[str, list[CompanyVisibleTrace]] = {}
    for trace in traces:
        if trace.action_type != "suggest_upgrade" or not trace.upgrade_theme:
            continue
        traces_by_theme.setdefault(trace.upgrade_theme, []).append(trace)

    normalized: dict[str, dict[str, float]] = {}
    for theme, theme_traces in traces_by_theme.items():
        independent_groups: dict[tuple[str, ...], list[CompanyVisibleTrace]] = {}
        for trace in theme_traces:
            independent_groups.setdefault(
                _independent_support_key(trace), []
            ).append(trace)
        group_count = max(1, len(independent_groups))
        group_values = tuple(independent_groups.values())
        raw_modes = tuple(
            trace.upgrade_contribution_mode or "user_feedback"
            for trace in theme_traces
        )
        authors = {trace.author_id for trace in theme_traces if trace.author_id}
        experiences = {
            trace.source_experience_ref
            for trace in theme_traces
            if trace.source_experience_ref
        }
        observation_sets = {
            observation_set
            for trace in theme_traces
            if (observation_set := _trace_observation_set_hash(trace))
        }
        clusters = {_trace_semantic_cluster_id(trace) for trace in theme_traces}
        normalized[theme] = {
            "count": float(len(theme_traces)),
            "raw_count": float(len(theme_traces)),
            "unique_author_count": float(len(authors)),
            "unique_experience_count": float(len(experiences)),
            "unique_observation_set_count": float(len(observation_sets)),
            "unique_cluster_count": float(len(clusters)),
            "independent_support_count": float(len(independent_groups)),
            "mean_priority": sum(
                _group_mean(group, "upgrade_priority", 0.5)
                for group in group_values
            )
            / group_count,
            "mean_confidence": sum(
                _group_mean(group, "upgrade_confidence", 0.5)
                for group in group_values
            )
            / group_count,
            "mean_technical_depth": sum(
                _group_mean(group, "upgrade_technical_depth", 0.5)
                for group in group_values
            )
            / group_count,
            "mean_novelty": sum(
                _group_mean(group, "upgrade_novelty_score", 0.5)
                for group in group_values
            )
            / group_count,
            "technical_review_count": float(
                sum(mode == "technical_review" for mode in raw_modes)
            ),
            "feature_proposal_count": float(
                sum(
                    mode in {"feature_proposal", "technical_feature_proposal"}
                    for mode in raw_modes
                )
            ),
            "technical_feature_proposal_count": float(
                sum(mode == "technical_feature_proposal" for mode in raw_modes)
            ),
            "independent_technical_review_count": float(
                sum(
                    any(
                        (trace.upgrade_contribution_mode or "user_feedback")
                        == "technical_review"
                        for trace in group
                    )
                    for group in group_values
                )
            ),
            "independent_feature_proposal_count": float(
                sum(
                    any(
                        (trace.upgrade_contribution_mode or "user_feedback")
                        in {"feature_proposal", "technical_feature_proposal"}
                        for trace in group
                    )
                    for group in group_values
                )
            ),
            "independent_technical_feature_proposal_count": float(
                sum(
                    any(
                        (trace.upgrade_contribution_mode or "user_feedback")
                        == "technical_feature_proposal"
                        for trace in group
                    )
                    for group in group_values
                )
            ),
        }
    return normalized


def _theme_support_refs(
    traces: list[CompanyVisibleTrace],
) -> dict[str, tuple[str, ...]]:
    refs: dict[str, list[str]] = {}
    for trace in traces:
        if trace.action_type != "suggest_upgrade" or not trace.upgrade_theme:
            continue
        refs.setdefault(trace.upgrade_theme, []).append(trace.event_id)
    return {
        theme: tuple(dict.fromkeys(theme_refs))
        for theme, theme_refs in refs.items()
    }


def _group_mean(
    traces: list[CompanyVisibleTrace], attribute: str, default: float
) -> float:
    values: list[float] = []
    for trace in traces:
        value = getattr(trace, attribute)
        values.append(float(value) if isinstance(value, (int, float)) else default)
    return sum(values) / max(1, len(values))


def _trace_observation_set_hash(trace: CompanyVisibleTrace) -> str | None:
    if trace.objective_observation_set_hash:
        return trace.objective_observation_set_hash
    if not trace.objective_observation_refs:
        return None
    normalized_refs = tuple(sorted(set(trace.objective_observation_refs)))
    return stable_hash(normalized_refs)


def _trace_semantic_cluster_id(trace: CompanyVisibleTrace) -> str:
    return trace.semantic_cluster_id or _semantic_cluster_id(
        artifact_id=trace.artifact_id,
        theme=trace.upgrade_theme,
        requested_change=trace.requested_change or trace.public_summary,
    )


def _independent_support_key(trace: CompanyVisibleTrace) -> tuple[str, ...]:
    observation_set_hash = _trace_observation_set_hash(trace)
    if observation_set_hash:
        return ("observation_set", observation_set_hash)
    if trace.source_experience_ref:
        return ("experience", trace.source_experience_ref)
    semantic_cluster_id = _trace_semantic_cluster_id(trace)
    if trace.author_id:
        return ("author_cluster", trace.author_id, semantic_cluster_id)
    if semantic_cluster_id:
        return ("semantic_cluster", semantic_cluster_id)
    return ("event", trace.event_id)


def _role_theme_summary(traces: list[CompanyVisibleTrace]) -> dict[str, Any]:
    support = _theme_support(traces)
    technical = {
        theme: values
        for theme, values in support.items()
        if values["technical_review_count"] >= 1.0
    }
    feature = {
        theme: values
        for theme, values in support.items()
        if values["feature_proposal_count"] >= 1.0
    }
    technical_review_events = int(
        sum(values["technical_review_count"] for values in support.values())
    )
    feature_proposal_events = int(
        sum(values["feature_proposal_count"] for values in support.values())
    )
    technical_feature_proposal_events = int(
        sum(
            values["technical_feature_proposal_count"]
            for values in support.values()
        )
    )

    top_technical = tuple(
        theme
        for theme, _ in sorted(
            technical.items(),
            key=lambda item: (
                -item[1]["independent_technical_review_count"],
                -item[1]["mean_technical_depth"],
                -item[1]["mean_confidence"],
                -item[1]["technical_review_count"],
                item[0],
            ),
        )[:5]
    )
    top_feature = tuple(
        theme
        for theme, _ in sorted(
            feature.items(),
            key=lambda item: (
                -item[1]["independent_feature_proposal_count"],
                -item[1]["mean_novelty"],
                -item[1]["mean_confidence"],
                -item[1]["mean_priority"],
                -item[1]["feature_proposal_count"],
                item[0],
            ),
        )[:5]
    )
    return {
        "technical_review_events": technical_review_events,
        "feature_proposal_events": feature_proposal_events,
        "technical_feature_proposal_events": technical_feature_proposal_events,
        "top_technical_themes": top_technical,
        "top_feature_themes": top_feature,
    }


def _feedback_uncertainty(
    *,
    public_mentions: int,
    upgrade_suggestion_events: int,
    failure_reports: int,
    paid_events: int,
) -> dict[str, float]:
    denominator = max(1, public_mentions)
    rates = {
        "suggestion_rate": upgrade_suggestion_events / denominator,
        "failure_rate": failure_reports / denominator,
        "paid_rate": paid_events / denominator,
    }
    uncertainty = {
        "public_mentions": float(public_mentions),
        "effective_n": float(public_mentions),
        **rates,
    }
    for rate_name, rate in rates.items():
        low, high = _wilson_interval_95(
            successes=round(rate * denominator),
            n=public_mentions,
        )
        uncertainty[f"{rate_name}_wilson_95_low"] = low
        uncertainty[f"{rate_name}_wilson_95_high"] = high
        uncertainty[f"{rate_name}_ci95_low"] = low
        uncertainty[f"{rate_name}_ci95_high"] = high
        uncertainty[f"{rate_name}_se"] = (high - low) / (2.0 * 1.959963984540054)
    return uncertainty


def _wilson_interval_95(*, successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    rate = max(0.0, min(1.0, successes / n))
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    center = (rate + z_squared / (2.0 * n)) / denominator
    margin = (
        z
        * ((rate * (1.0 - rate) + z_squared / (4.0 * n)) / n) ** 0.5
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _top_upgrade_themes(traces: list[CompanyVisibleTrace]) -> tuple[str, ...]:
    aggregates = _theme_support(traces)
    ranked = sorted(
        aggregates.items(),
        key=lambda item: (
            -item[1]["independent_support_count"],
            -item[1]["mean_priority"],
            -item[1]["mean_confidence"],
            -item[1]["raw_count"],
            item[0],
        ),
    )
    # Head-slot channel: themes whose evidence already satisfies one of the
    # adoption arms. Qualification is derived from the evidence itself, never
    # from a hardcoded theme list, so a substrate-derived theme with the same
    # support as a curated one gets the same guaranteed candidacy.
    product_ranked = sorted(
        (
            (theme, values)
            for theme, values in aggregates.items()
            if any(_adoption_arms(_decision_evidence(values)))
        ),
        key=lambda item: (
            -_theme_support_score(item[1]),
            -item[1]["independent_support_count"],
            -item[1]["raw_count"],
            item[0],
        ),
    )
    strategic_ranked = sorted(
        (
            (theme, values)
            for theme, values in aggregates.items()
            if theme in STRATEGIC_UPGRADE_THEMES
        ),
        key=lambda item: (
            -(item[1]["mean_priority"] + 0.5 * item[1]["mean_confidence"]),
            -item[1]["independent_support_count"],
            -item[1]["raw_count"],
            item[0],
        ),
    )
    feature_ranked = sorted(
        (
            (theme, values)
            for theme, values in aggregates.items()
            if theme in CREATIVE_FEATURE_THEMES
            and values["independent_feature_proposal_count"] >= 1
            and (
                values["independent_feature_proposal_count"] >= 2
                or (
                    values["mean_novelty"] >= 0.80
                    and values["mean_confidence"] >= 0.65
                )
            )
        ),
        key=lambda item: (
            -item[1]["independent_feature_proposal_count"],
            -item[1]["mean_novelty"],
            -item[1]["mean_confidence"],
            -item[1]["mean_priority"],
            -item[1]["raw_count"],
            item[0],
        ),
    )
    ordered: list[str] = []
    for theme, _ in product_ranked[:6]:
        ordered.append(theme)
    for theme, _ in ranked[:3]:
        ordered.append(theme)
    for theme, _ in feature_ranked[:2]:
        ordered.append(theme)
    for theme, _ in strategic_ranked[:3]:
        ordered.append(theme)
    for theme, _ in ranked:
        ordered.append(theme)
    return tuple(dict.fromkeys(ordered))[:8]


def _theme_support_score(values: dict[str, float]) -> float:
    mean_priority = values["mean_priority"]
    mean_confidence = values["mean_confidence"]
    volume = min(1.0, values["independent_support_count"] / 50.0)
    return 0.48 * mean_priority + 0.34 * mean_confidence + 0.18 * volume
