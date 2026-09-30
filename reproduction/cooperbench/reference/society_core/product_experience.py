"""Typed objective product-experience simulation for v16."""

from __future__ import annotations

import re

from .demand import sample_user_task
from .hashing import stable_hash
from .randomness import SeededRandom
from .schemas import (
    ActionKind,
    AgentState,
    Artifact,
    ProductExperiencePacket,
    ProductUsageStep,
    UpgradeSuggestion,
    clamp01,
)


DEFAULT_ARTIFACT_PRICE = 0.15

CREATIVE_FEATURE_THEMES: frozenset[str] = frozenset(
    {
        "programmable_workflows",
        "guided_workflow_templates",
        "collaborative_reuse",
        "workflow_automation",
    }
)


def artifact_price(artifact: Artifact | None) -> float:
    if artifact is None:
        return DEFAULT_ARTIFACT_PRICE
    return clamp01(float(artifact.cost_profile.get("money", DEFAULT_ARTIFACT_PRICE)))


def simulate_product_experience(
    *,
    agent: AgentState,
    artifact: Artifact,
    rng: SeededRandom,
    tick: int,
    action_kind: ActionKind,
) -> ProductExperiencePacket:
    """Produce an objective experience packet owned by the typed kernel."""

    profile = agent.profile
    body = agent.body
    budgets = agent.action_budgets
    task_type = sample_user_task(agent=agent, artifact=artifact, rng=rng)
    task_fit = _task_fit(artifact, task_type)
    capability = _mean(artifact.capability_profile.values()) or 0.5
    reliability = _mean(artifact.reliability_profile.values()) or capability
    cost = _mean(artifact.cost_profile.values()) or 0.1
    cognitive_capacity = profile.cognitive_capacity

    task_noise = rng.uniform(
        "event", -0.08, 0.08, kind=f"experience_task_difficulty:{artifact.id}"
    )
    task_difficulty = clamp01(
        0.28
        + 0.38 * (1.0 - profile.technical_skill)
        + 0.18 * (1.0 - cognitive_capacity)
        + 0.18 * profile.deadline_pressure
        + 0.12 * body.stress
        + 0.12 * (1.0 - task_fit)
        + task_noise
    )
    baseline_success = clamp01(
        0.18
        + 0.46 * profile.technical_skill
        + 0.20 * cognitive_capacity
        + 0.18 * budgets.attention_budget
        + 0.12 * body.physiological_energy
        - 0.38 * task_difficulty
        - 0.18 * body.fatigue
    )
    usability = clamp01(
        0.25
        + 0.38 * profile.technical_skill
        + 0.18 * cognitive_capacity
        + 0.22 * profile.novelty_seeking
        + 0.15 * budgets.attention_budget
        - 0.18 * body.fatigue
    )
    product_gain = clamp01(
        (0.10 + 0.52 * capability + 0.28 * task_fit + 0.20 * reliability)
        * usability
        * (0.72 + 0.28 * profile.domain_need)
        * (1.0 - 0.42 * task_difficulty)
    )
    friction = clamp01(
        0.08
        + 0.55 * cost * (1.18 - profile.technical_skill)
        + 0.22 * artifact.learning_curve * (1.0 - profile.technical_skill)
        + 0.12 * artifact.learning_curve * (1.0 - cognitive_capacity)
        + 0.16 * body.fatigue
        + rng.uniform("event", 0.0, 0.08, kind=f"experience_friction:{artifact.id}")
    )
    failure_probability = clamp01(
        (1.0 - reliability) * (0.34 + 0.42 * task_difficulty)
        + 0.12 * body.stress
        + 0.08 * profile.deadline_pressure
        - 0.08 * profile.technical_skill
        - 0.07 * cognitive_capacity
    )
    journey_steps = _simulate_usage_journey(
        agent=agent,
        artifact=artifact,
        rng=rng,
        task_type=task_type,
        task_fit=task_fit,
        capability=capability,
        reliability=reliability,
        cost=cost,
        task_difficulty=task_difficulty,
        usability=usability,
        base_friction=friction,
        failure_probability=failure_probability,
    )
    failure_steps = tuple(step for step in journey_steps if step.error_event)
    failure_event = failure_steps[0].error_event if failure_steps else None
    failure_rate = len(failure_steps) / len(journey_steps) if journey_steps else 0.0
    journey_success = _journey_success(journey_steps)
    journey_friction = _mean(step.friction for step in journey_steps)
    diagnostic_clarity = _diagnostic_clarity(journey_steps)
    workaround_success = _workaround_success(journey_steps)
    first_value_time = _first_value_time(journey_steps)
    integration_quality = _stage_success(journey_steps, "integrate_workflow")
    failure_penalty = (
        0.0
        if failure_event is None
        else 0.08 + 0.22 * failure_rate + 0.12 * (1.0 - workaround_success)
    )

    friction = clamp01(0.45 * friction + 0.55 * journey_friction)
    objective_success = clamp01(
        0.45 * (baseline_success + product_gain - 0.36 * friction)
        + 0.55 * journey_success
        - failure_penalty
    )
    success_gain = max(0.0, objective_success - baseline_success)
    time_saved = clamp01(
        0.10
        + 0.43 * product_gain
        + 0.18 * task_fit
        + 0.16 * integration_quality
        - 0.36 * friction
        - 0.18 * first_value_time
        - 0.10 * bool(failure_event)
    )
    observed_reliability = clamp01(
        reliability
        - 0.30 * failure_rate
        - 0.10 * (1.0 - workaround_success if failure_event else 0.0)
        + rng.uniform(
            "event", -0.04, 0.04, kind=f"experience_reliability_noise:{artifact.id}"
        )
    )
    attention_cost = clamp01(
        0.05 + 0.35 * friction + 0.06 * (action_kind == ActionKind.TRY_ARTIFACT_ON_TASK)
    )
    budget_cost = artifact_price(artifact)
    comparison = clamp01(0.5 + objective_success - baseline_success - 0.25 * friction)
    repeat_use_value = clamp01(
        0.28 * success_gain
        + 0.24 * time_saved
        + 0.18 * integration_quality
        + 0.18 * observed_reliability
        + 0.12 * workaround_success
        - 0.14 * friction
    )

    return ProductExperiencePacket(
        experience_id=f"xp_{tick:06d}_{agent.id}_{artifact.id}_{agent.artifact_usage_counts.get(artifact.id, 0):04d}",
        artifact_id=artifact.id,
        tick=tick,
        task_type=task_type,
        task_difficulty=task_difficulty,
        baseline_success_without_product=baseline_success,
        objective_success=objective_success,
        success_gain=success_gain,
        time_saved=time_saved,
        friction=friction,
        failure_event=failure_event,
        observed_reliability=observed_reliability,
        attention_cost=attention_cost,
        budget_cost=budget_cost,
        comparison_to_current_method=comparison,
        journey_steps=journey_steps,
        blocked_stage=_blocked_stage(journey_steps),
        diagnostic_clarity=diagnostic_clarity,
        workaround_success=workaround_success,
        first_value_time=first_value_time,
        integration_quality=integration_quality,
        repeat_use_value=repeat_use_value,
    )


def apply_product_experience(
    agent: AgentState, packet: ProductExperiencePacket
) -> dict[str, float]:
    history = agent.artifact_experience_history.setdefault(packet.artifact_id, [])
    history.append(packet)
    if len(history) > 8:
        del history[:-8]
    agent.latest_artifact_experience[packet.artifact_id] = packet

    old_belief = agent.cognition.beliefs_about_artifacts.get(packet.artifact_id, 0.5)
    old_negative = agent.cognition.artifact_negative_signal_strength.get(
        packet.artifact_id, 0.0
    )
    belief_delta = (
        0.16 * (packet.objective_success - 0.5)
        + 0.12 * (packet.comparison_to_current_method - 0.5)
        + 0.10 * (packet.observed_reliability - 0.5)
        - 0.08 * packet.friction
    )
    agent.cognition.beliefs_about_artifacts[packet.artifact_id] = clamp01(
        old_belief + belief_delta
    )
    if packet.failure_event or packet.objective_success < 0.35:
        agent.cognition.artifact_negative_signal_strength[packet.artifact_id] = clamp01(
            old_negative + 0.12 + 0.18 * (1.0 - packet.objective_success)
        )
    return {
        "experience_objective_success": packet.objective_success,
        "experience_success_gain": packet.success_gain,
        "experience_time_saved": packet.time_saved,
        "experience_friction": packet.friction,
        "experience_belief_delta": agent.cognition.beliefs_about_artifacts[
            packet.artifact_id
        ]
        - old_belief,
        "experience_negative_delta": agent.cognition.artifact_negative_signal_strength.get(
            packet.artifact_id, 0.0
        )
        - old_negative,
    }


def suggest_upgrades_from_experience(
    *,
    agent: AgentState,
    artifact: Artifact,
    packet: ProductExperiencePacket,
    evidence_only: bool = False,
) -> tuple[UpgradeSuggestion, ...]:
    """Convert objective product experience into auditable upgrade demand."""

    profile = agent.profile
    technical_depth = _technical_review_depth(profile=profile, packet=packet)
    novelty_score = _feature_novelty_score(profile)
    experience_pressure = clamp01(
        0.28 * (1.0 - packet.objective_success)
        + 0.18 * packet.friction
        + 0.16 * (1.0 - packet.observed_reliability)
        + 0.12 * (1.0 - packet.time_saved)
        + 0.10 * float(packet.failure_event is not None)
        + 0.08 * (1.0 - packet.diagnostic_clarity)
        + 0.05 * (1.0 - packet.workaround_success)
        + 0.03 * packet.first_value_time
    )
    profile_pressure = clamp01(
        0.20 * profile.domain_need
        + 0.18 * profile.deadline_pressure
        + 0.18 * profile.reliability_preference
        + 0.14 * profile.budget_sensitivity * packet.budget_cost
        + 0.15 * (1.0 - profile.technical_skill)
        + 0.15 * (1.0 - profile.cognitive_capacity)
        + 0.10 * profile.strategic_boldness
    )
    priority = clamp01(0.68 * experience_pressure + 0.32 * profile_pressure)
    confidence = clamp01(
        0.18
        + 0.20 * profile.technical_skill
        + 0.18 * profile.cognitive_capacity
        + 0.18 * packet.observed_reliability
        + 0.12 * (1.0 - packet.friction)
        + 0.08 * float(packet.failure_event is not None)
        + 0.10 * packet.diagnostic_clarity
        + 0.06 * packet.workaround_success
        + 0.06 * profile.domain_need
    )

    primary_theme = _primary_upgrade_theme(
        agent=agent,
        artifact=artifact,
        packet=packet,
        evidence_only=evidence_only,
    )
    journey_failure_themes = (
        ()
        if evidence_only
        else _journey_failure_themes(
            artifact=artifact,
            packet=packet,
        )
    )
    themes = [
        primary_theme,
        *journey_failure_themes,
        *(
            ()
            if evidence_only
            else _artifact_specific_upgrade_themes(
                artifact=artifact,
                packet=packet,
                primary_theme=primary_theme,
            )
        ),
        *_creative_feature_themes(
            agent=agent,
            packet=packet,
            novelty_score=novelty_score,
        ),
        *(
            ()
            if evidence_only
            else _strategic_upgrade_themes(
                agent=agent,
                artifact=artifact,
                packet=packet,
            )
        ),
    ]
    if (
        packet.blocked_stage == "debug_recover"
        and "diagnostics_and_recovery" not in themes
    ):
        themes.append("diagnostics_and_recovery")
    if (
        packet.blocked_stage == "integrate_workflow"
        and "workflow_integration" not in themes
    ):
        themes.append("workflow_integration")
    if (
        packet.blocked_stage == "discover_fit"
        and "use_case_documentation" not in themes
    ):
        themes.append("use_case_documentation")
    if packet.failure_event and packet.friction >= 0.32:
        themes.append("documentation_and_migration_path")
    if (
        packet.failure_event
        and "dependency_resolution" not in themes
        and _looks_like_dependency_failure(packet.failure_event)
    ):
        themes.append("dependency_resolution")
    if (
        packet.observed_reliability < 0.48
        and "production_build_reliability" not in themes
    ):
        themes.append("production_build_reliability")
    if packet.objective_success < 0.38 and "task_fit_improvement" not in themes:
        themes.append("task_fit_improvement")

    evidence_refs = tuple(
        ref
        for ref in (
            packet.experience_id,
            f"failure:{packet.failure_event}" if packet.failure_event else None,
            f"task:{packet.task_type}",
            f"blocked_stage:{packet.blocked_stage}" if packet.blocked_stage else None,
            *_stage_evidence_refs(packet),
        )
        if ref
    )
    suggestions: list[UpgradeSuggestion] = []
    suggestion_limit = min(
        8,
        max(
            5
            if profile.innovation_role in {"creative_originator", "early_builder"}
            else 4
            if profile.strategic_boldness >= 0.72
            else 3,
            len(journey_failure_themes),
        ),
    )
    for index, theme in enumerate(dict.fromkeys(themes)):
        adjusted_priority = clamp01(priority * (1.0 - 0.12 * index))
        contribution_mode = _contribution_mode(
            profile=profile,
            theme=theme,
            technical_depth=technical_depth,
            novelty_score=novelty_score,
        )
        suggestions.append(
            UpgradeSuggestion(
                suggestion_id=_upgrade_suggestion_id(
                    agent_id=agent.id,
                    packet=packet,
                    theme=theme,
                    index=index,
                ),
                artifact_id=artifact.id,
                author_id=agent.id,
                tick=packet.tick,
                source_experience_ref=packet.experience_id,
                theme=theme,
                priority=adjusted_priority,
                confidence=confidence,
                requested_change=_requested_change(theme, packet),
                affected_tasks=_affected_tasks(packet),
                evidence_refs=evidence_refs,
                profile_ref=profile.persona_label,
                contribution_mode=contribution_mode,
                technical_depth=technical_depth,
                novelty_score=novelty_score,
            )
        )
        if len(suggestions) >= suggestion_limit:
            break
    return tuple(suggestions)


def _technical_review_depth(*, profile, packet: ProductExperiencePacket) -> float:
    role_bonus = {
        "professional_engineer": 0.12,
        "technical_practitioner": 0.05,
    }.get(profile.technical_role, 0.0)
    return clamp01(
        0.50 * profile.technical_skill
        + 0.22 * profile.cognitive_capacity
        + 0.12 * packet.diagnostic_clarity
        + 0.08 * float(packet.failure_event is not None)
        + role_bonus
    )


def _feature_novelty_score(profile) -> float:
    return clamp01(
        0.55 * profile.creative_capacity
        + 0.20 * profile.novelty_seeking
        + 0.15 * profile.strategic_boldness
        + 0.10 * profile.domain_need
    )


def _creative_feature_themes(
    *,
    agent: AgentState,
    packet: ProductExperiencePacket,
    novelty_score: float,
) -> tuple[str, ...]:
    profile = agent.profile
    if (
        profile.innovation_role not in {"creative_originator", "early_builder"}
        or novelty_score < 0.65
    ):
        return ()

    themes: list[str] = []
    if packet.integration_quality < 0.50:
        themes.append("programmable_workflows")
    if packet.first_value_time > 0.55:
        themes.append("guided_workflow_templates")
    if packet.repeat_use_value < 0.45:
        themes.append("collaborative_reuse")
    if packet.time_saved < 0.35 or not themes:
        themes.append("workflow_automation")
    return tuple(dict.fromkeys(themes))


def _contribution_mode(
    *,
    profile,
    theme: str,
    technical_depth: float,
    novelty_score: float,
) -> str:
    if theme in CREATIVE_FEATURE_THEMES and novelty_score >= 0.65:
        if (
            profile.technical_role
            in {"professional_engineer", "technical_practitioner"}
            and technical_depth >= 0.65
        ):
            return "technical_feature_proposal"
        return "feature_proposal"
    if (
        profile.technical_role == "professional_engineer"
        and technical_depth >= 0.70
    ) or (
        profile.technical_role == "technical_practitioner"
        and technical_depth >= 0.62
    ):
        return "technical_review"
    return "user_feedback"


def _upgrade_suggestion_id(
    *, agent_id: str, packet: ProductExperiencePacket, theme: str, index: int
) -> str:
    return (
        "upgrade_"
        + stable_hash(
            {
                "agent_id": agent_id,
                "experience_id": packet.experience_id,
                "theme": theme,
                "index": index,
            }
        )[:20]
    )


def _primary_upgrade_theme(
    *,
    agent: AgentState,
    artifact: Artifact,
    packet: ProductExperiencePacket,
    evidence_only: bool = False,
) -> str:
    if not evidence_only:
        artifact_theme = _theme_from_artifact_failure(
            artifact=artifact,
            packet=packet,
        )
        if artifact_theme:
            return artifact_theme
    if packet.blocked_stage == "setup":
        if packet.failure_event and _looks_like_dependency_failure(
            packet.failure_event
        ):
            return "dependency_resolution"
        return "documentation_and_migration_path"
    if packet.blocked_stage == "debug_recover" or packet.diagnostic_clarity < 0.42:
        return "diagnostics_and_recovery"
    if (
        packet.blocked_stage == "integrate_workflow"
        or packet.integration_quality < 0.42
    ):
        return "workflow_integration"
    if packet.blocked_stage == "discover_fit":
        return "use_case_documentation"
    if packet.failure_event:
        failure = packet.failure_event.lower()
        if _looks_like_dependency_failure(failure):
            return "dependency_resolution"
        if any(
            token in failure
            for token in ("build", "bundle", "compile", "server", "hmr", "reload")
        ):
            return "production_build_reliability"
        if any(token in failure for token in ("doc", "guide", "tutorial", "migration")):
            return "documentation_and_migration_path"
        return "production_build_reliability"
    if packet.friction >= 0.52 or (
        not evidence_only
        and artifact.learning_curve
        * (1.0 - agent.profile.cognitive_capacity)
        >= 0.28
    ):
        return "documentation_and_migration_path"
    if packet.observed_reliability < 0.56:
        return "production_build_reliability"
    if packet.objective_success < 0.42:
        return "task_fit_improvement"
    if packet.time_saved < 0.28:
        return "workflow_speedup"
    if packet.repeat_use_value < 0.34:
        return "repeat_use_value"
    if packet.budget_cost * agent.profile.budget_sensitivity >= 0.28:
        return "reduce_friction_or_price"
    return "use_case_documentation"


def _artifact_specific_upgrade_themes(
    *,
    artifact: Artifact,
    packet: ProductExperiencePacket,
    primary_theme: str,
) -> tuple[str, ...]:
    themes: list[str] = []
    task = packet.task_type.lower()
    failure = (packet.failure_event or "").lower()
    if not _looks_like_repo_digest_artifact(artifact):
        # Failure evidence only: task tokens describe what the user attempted,
        # not what broke, and would surface speculative adjacent themes that
        # crowd out stronger channels under the suggestion cap.
        if failure:
            themes.extend(
                _substrate_derived_themes(evidence=failure, artifact=artifact)
            )
    if _looks_like_repo_digest_artifact(artifact):
        if "submodule" in task or "submodule" in failure:
            themes.append("include_submodules")
        if "large" in task or "max_file_size" in failure or "large_file" in failure:
            themes.append("max_file_size_enforcement")
        if "include" in task or "exclude" in task or "pattern" in failure:
            themes.append("pattern_filtering_consistency")
        if "gitignore" in task or "gitingestignore" in failure or "ignore" in failure:
            themes.append("ignore_pattern_reliability")
        if "remote" in task or "curl" in failure or "http" in failure:
            themes.append("http_client_portability")
        if "security" in task or "traversal" in failure:
            themes.append("path_security")
        if "offline" in task or "token" in failure or "tiktoken" in failure:
            themes.append("token_count_resilience")
    return tuple(theme for theme in dict.fromkeys(themes) if theme != primary_theme)


def _journey_failure_themes(
    *,
    artifact: Artifact,
    packet: ProductExperiencePacket,
) -> tuple[str, ...]:
    if not _looks_like_repo_digest_artifact(artifact):
        derived: list[str] = []
        for step in packet.journey_steps:
            if not step.error_event:
                continue
            derived.extend(
                _substrate_derived_themes(
                    evidence=f"{step.stage} {step.objective} {step.error_event}",
                    artifact=artifact,
                )
            )
        return tuple(dict.fromkeys(derived))
    themes: list[str] = []
    for step in packet.journey_steps:
        if not step.error_event:
            continue
        evidence = f"{step.stage} {step.objective} {step.error_event}".lower()
        if "token" in evidence or "offline" in evidence or "tiktoken" in evidence:
            themes.append("token_count_resilience")
        if (
            "max_file_size" in evidence
            or "oversized" in evidence
            or "large" in evidence
        ):
            themes.append("max_file_size_enforcement")
        if "submodule" in evidence:
            themes.append("include_submodules")
        if (
            "gitignore" in evidence
            or "gitingestignore" in evidence
            or "ignore_pattern" in evidence
        ):
            themes.append("ignore_pattern_reliability")
        if (
            "include_exclude" in evidence
            or "pattern_filter" in evidence
            or "wildmatch" in evidence
        ):
            themes.append("pattern_filtering_consistency")
        if "curl" in evidence or "http_client" in evidence or "portability" in evidence:
            themes.append("http_client_portability")
        if "traversal" in evidence or "path_security" in evidence:
            themes.append("path_security")
    return tuple(dict.fromkeys(themes))


def _theme_from_artifact_failure(
    *, artifact: Artifact, packet: ProductExperiencePacket
) -> str | None:
    if not _looks_like_repo_digest_artifact(artifact):
        failure_event = (packet.failure_event or "").strip()
        if not failure_event:
            return None
        # Failure text only: task tokens would let "what I was doing" outvote
        # "what actually failed" (e.g. a dev-server task with a dependency
        # failure must map to the dependency theme, not the server theme).
        derived = _substrate_derived_themes(
            evidence=failure_event, artifact=artifact
        )
        return derived[0] if derived else None
    failure = (packet.failure_event or "").lower()
    task = packet.task_type.lower()
    if "token" in failure or "tiktoken" in failure or "offline" in task:
        return "token_count_resilience"
    if "max_file_size" in failure or "large_file" in failure or "large" in task:
        return "max_file_size_enforcement"
    if "submodule" in failure or "submodule" in task:
        return "include_submodules"
    if "gitignore" in failure or "gitingestignore" in failure or "ignore" in task:
        return "ignore_pattern_reliability"
    if (
        "include_exclude" in failure
        or "pattern" in failure
        or "include_exclude" in task
    ):
        return "pattern_filtering_consistency"
    if "curl" in failure or "http" in failure or "remote" in task:
        return "http_client_portability"
    if "traversal" in failure or "security" in task:
        return "path_security"
    return None


# Substrate-derived technical themes: instead of hardcoding one product's
# failure vocabulary, derive the theme lexicon from what the artifact itself
# declares (reliability_profile keys + failure_modes). The curated repo-digest
# keyword maps above remain the known-substrate path; every other substrate
# gets themes anchored in its own declared semantics rather than nothing.

_SUBSTRATE_TOKEN_STOPWORDS: frozenset[str] = frozenset(
    {
        "and",
        "broken",
        "case",
        "edge",
        "error",
        "errors",
        "event",
        "failed",
        "failure",
        "failures",
        "for",
        "handling",
        "into",
        "issue",
        "issues",
        "leak",
        "misc",
        "missing",
        "mismatch",
        "mode",
        "modes",
        "not",
        "problem",
        "regression",
        "risk",
        "surprise",
        "support",
        "the",
        "with",
    }
)


def _substrate_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.split(r"[^a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in _SUBSTRATE_TOKEN_STOPWORDS
    )


def derive_substrate_theme_lexicon(
    artifact: Artifact,
) -> tuple[tuple[str, frozenset[str]], ...]:
    """Build (theme, keywords) pairs from the artifact's declared semantics.

    Reliability-profile keys become theme names; each failure mode enriches the
    best token-overlapping reliability theme, or becomes a theme of its own when
    nothing overlaps. Deterministic: follows artifact declaration order.
    """

    themes: dict[str, set[str]] = {}
    for reliability_key in artifact.reliability_profile:
        tokens = _substrate_tokens(reliability_key)
        if tokens:
            themes.setdefault(reliability_key, set()).update(tokens)
    for failure_mode in artifact.failure_modes:
        tokens = _substrate_tokens(failure_mode)
        if not tokens:
            continue
        best_theme = None
        best_overlap = 0
        for theme, keywords in themes.items():
            overlap = len(keywords & tokens)
            if overlap > best_overlap:
                best_theme = theme
                best_overlap = overlap
        if best_theme is not None:
            themes[best_theme].update(tokens)
        else:
            themes.setdefault(failure_mode, set()).update(tokens)
    return tuple((theme, frozenset(keywords)) for theme, keywords in themes.items())


def _substrate_derived_themes(
    *, evidence: str, artifact: Artifact
) -> tuple[str, ...]:
    """Themes whose substrate-derived keywords overlap the evidence text,
    strongest overlap first (declaration order breaks ties)."""

    evidence_tokens = _substrate_tokens(evidence)
    if not evidence_tokens:
        return ()
    scored: list[tuple[int, str]] = []
    for theme, keywords in derive_substrate_theme_lexicon(artifact):
        overlap = len(keywords & evidence_tokens)
        if overlap:
            scored.append((overlap, theme))
    scored.sort(key=lambda item: -item[0])
    return tuple(theme for _, theme in scored)


def _looks_like_repo_digest_artifact(artifact: Artifact) -> bool:
    haystack = " ".join(
        (
            artifact.id,
            artifact.artifact_kind,
            *artifact.capability_profile,
            *artifact.reliability_profile,
            *artifact.task_fit_distribution,
            *artifact.failure_modes,
            *artifact.evidence_claims,
            *artifact.public_claims,
        )
    ).lower()
    return any(
        token in haystack
        for token in ("gitingest", "repo_digest", "repository digest", "submodule")
    )


def _strategic_upgrade_themes(
    *, agent: AgentState, artifact: Artifact, packet: ProductExperiencePacket
) -> tuple[str, ...]:
    profile = agent.profile
    strategic_capacity = clamp01(
        0.36 * profile.strategic_boldness
        + 0.24 * profile.technical_skill
        + 0.18 * profile.opinion_leadership
        + 0.14 * profile.cognitive_capacity
        + 0.08 * profile.domain_need
    )
    if strategic_capacity < 0.62:
        return ()
    themes: list[str] = []
    broad_fit_pressure = _broad_fit_pressure(artifact)
    integration_pressure = clamp01(
        0.36 * (1.0 - packet.integration_quality)
        + 0.28 * (1.0 - packet.repeat_use_value)
        + 0.18 * float(packet.blocked_stage in {"discover_fit", "integrate_workflow"})
        + 0.18 * broad_fit_pressure
    )
    ecosystem_pressure = clamp01(
        0.34 * float(packet.blocked_stage in {"setup", "integrate_workflow"})
        + 0.24 * (1.0 - packet.workaround_success)
        + 0.22 * (1.0 - packet.diagnostic_clarity)
        + 0.20 * _plugin_or_extension_gap(artifact)
    )
    if integration_pressure >= 0.52:
        themes.append("architecture_generalization")
    if ecosystem_pressure >= 0.50:
        themes.append("plugin_ecosystem")
    if broad_fit_pressure >= 0.45 and profile.strategic_boldness >= 0.70:
        themes.append("framework_generalization")
    return tuple(dict.fromkeys(themes))


def _looks_like_dependency_failure(failure_event: str) -> bool:
    failure = failure_event.lower()
    return any(
        token in failure
        for token in (
            "dependency",
            "dep",
            "resolve",
            "module",
            "package",
            "install",
            "prebundle",
        )
    )


def _requested_change(theme: str, packet: ProductExperiencePacket) -> str:
    failure = packet.failure_event or "observed friction"
    stage = packet.blocked_stage or "overall_use"
    if theme == "dependency_resolution":
        return (
            f"Reduce dependency-resolution failures in {stage} for {packet.task_type}; "
            f"show the failing package, root cause, and a copyable recovery command for {failure}."
        )
    if theme == "production_build_reliability":
        return (
            f"Harden {stage} for {packet.task_type}; add a reproducible failure path, "
            f"retry behavior, and recovery checks for {failure}."
        )
    if theme == "diagnostics_and_recovery":
        return (
            f"Improve diagnostics in {stage} for {packet.task_type}; map the visible error to cause, "
            f"next step, and workaround when diagnostic clarity is {packet.diagnostic_clarity:.2f}."
        )
    if theme == "workflow_integration":
        return (
            f"Make {packet.task_type} integrate cleanly into existing workflows; preserve state, "
            f"configuration, and repeat-use value after first success."
        )
    if theme == "architecture_generalization":
        return (
            f"Generalize the core architecture behind {packet.task_type}; separate stable core behavior "
            f"from project-specific assumptions so repeated workflow integration failures become one platform fix."
        )
    if theme == "plugin_ecosystem":
        return (
            f"Expose extension points around {stage} for {packet.task_type}; let advanced users package fixes, "
            f"integrations, and framework adapters instead of waiting for one-off core patches."
        )
    if theme == "framework_generalization":
        return (
            f"Turn the successful path for {packet.task_type} into a framework-agnostic capability; "
            f"make templates and adapters first-class rather than treating non-primary use cases as edge cases."
        )
    if theme == "programmable_workflows":
        return (
            f"Expose composable workflow primitives for {packet.task_type}; let users program, combine, "
            f"and reuse the steps around {stage} instead of accepting one fixed product path."
        )
    if theme == "guided_workflow_templates":
        return (
            f"Add adaptable workflow templates for {packet.task_type}; shorten first value while keeping "
            f"the underlying steps visible and editable to experienced users."
        )
    if theme == "collaborative_reuse":
        return (
            f"Let teams share and adapt successful {packet.task_type} configurations; preserve provenance "
            f"and make repeat use cheaper than recreating the workflow."
        )
    if theme == "workflow_automation":
        return (
            f"Automate the repeated low-value steps in {packet.task_type}; keep review points around "
            f"{stage} so users retain control over consequential changes."
        )
    if theme == "documentation_and_migration_path":
        return (
            f"Add task-specific onboarding, migration, and troubleshooting guidance for {packet.task_type}, "
            f"starting from the {stage} step where users lose momentum."
        )
    if theme == "task_fit_improvement":
        return (
            f"Improve objective task fit for {packet.task_type}; close the gap between the product path "
            f"and the baseline at {stage}."
        )
    if theme == "workflow_speedup":
        return f"Remove repeated steps in {packet.task_type} so time saved is visible before {stage} blocks first value."
    if theme == "reduce_friction_or_price":
        return f"Lower setup, attention, or budget cost before asking users to pay for {packet.task_type}."
    if theme == "repeat_use_value":
        return f"Improve repeat-use value for {packet.task_type}; make the second run cheaper than the first run."
    if theme == "token_count_resilience":
        return (
            f"Make token estimation resilient in {stage} for {packet.task_type}; keep repository digest generation "
            f"usable when token-count dependencies fail with {failure}."
        )
    if theme == "max_file_size_enforcement":
        return (
            f"Enforce max-file-size filtering before content emission for {packet.task_type}; prevent large files "
            f"from leaking into the digest after {failure}."
        )
    if theme == "include_submodules":
        return (
            f"Add explicit submodule traversal controls for {packet.task_type}; users should decide whether "
            f"submodule content is included instead of silently missing it after {failure}."
        )
    if theme == "ignore_pattern_reliability":
        return (
            f"Respect .gitignore and product-specific ignore files consistently in {packet.task_type}; "
            f"explain ignored paths when {failure} appears."
        )
    if theme == "pattern_filtering_consistency":
        return (
            f"Make include and exclude pattern filtering deterministic for {packet.task_type}; document precedence "
            f"and add regression checks around {failure}."
        )
    if theme == "http_client_portability":
        return (
            f"Remove brittle shell-level HTTP assumptions in {packet.task_type}; use a portable client path "
            f"with graceful errors for {failure}."
        )
    if theme == "path_security":
        return (
            f"Harden path normalization for {packet.task_type}; reject traversal and unsafe paths before reading "
            f"repository contents after {failure}."
        )
    if packet.failure_event:
        # Substrate-derived themes reach here: keep the request failure-anchored
        # and templated from the theme itself rather than a success framing.
        return (
            f"Improve {theme.replace('_', ' ')} in {stage} for {packet.task_type}; "
            f"add a reproducible check and recovery guidance for {failure}."
        )
    return f"Document and polish the successful {packet.task_type} path so users can reproduce it."


def _affected_tasks(packet: ProductExperiencePacket) -> tuple[str, ...]:
    tasks = [packet.task_type]
    if packet.blocked_stage:
        tasks.append(f"{packet.task_type}:{packet.blocked_stage}")
    return tuple(dict.fromkeys(tasks))


def _stage_evidence_refs(packet: ProductExperiencePacket) -> tuple[str, ...]:
    refs: list[str] = list(packet.objective_observation_refs)
    if packet.objective_observation_set_hash:
        refs.append(
            f"observation_set_hash:{packet.objective_observation_set_hash}"
        )
    if packet.objective_observation_manifest_hash:
        refs.append(
            "observation_manifest_hash:"
            f"{packet.objective_observation_manifest_hash}"
        )
    for step in packet.journey_steps:
        if step.stage == packet.blocked_stage or step.error_event:
            refs.append(
                f"stage:{step.stage}:success={step.success:.3f}:friction={step.friction:.3f}:diagnostic={step.diagnostic_clarity:.3f}"
            )
            if step.error_event:
                refs.append(f"stage_error:{step.stage}:{step.error_event}")
                refs.extend(
                    ref
                    for ref in step.user_visible_evidence
                    if ref.startswith("real_xp_obs_")
                )
    return tuple(dict.fromkeys(refs))[:24]


def _broad_fit_pressure(artifact: Artifact) -> float:
    if not artifact.task_fit_distribution:
        return 0.0
    fits = [clamp01(value) for value in artifact.task_fit_distribution.values()]
    if not fits:
        return 0.0
    low_fit_share = sum(1 for value in fits if value < 0.45) / len(fits)
    spread = max(fits) - min(fits)
    return clamp01(0.58 * low_fit_share + 0.42 * spread)


def _plugin_or_extension_gap(artifact: Artifact) -> float:
    capability = artifact.capability_profile.get("plugin_ecosystem")
    if capability is not None:
        return clamp01(1.0 - capability)
    plugin_fit = artifact.task_fit_distribution.get("plugin_integration")
    if plugin_fit is not None:
        return clamp01(1.0 - plugin_fit)
    return 0.0


def _simulate_usage_journey(
    *,
    agent: AgentState,
    artifact: Artifact,
    rng: SeededRandom,
    task_type: str,
    task_fit: float,
    capability: float,
    reliability: float,
    cost: float,
    task_difficulty: float,
    usability: float,
    base_friction: float,
    failure_probability: float,
) -> tuple[ProductUsageStep, ...]:
    profile = agent.profile
    body = agent.body
    stages = (
        ("discover_fit", "map_public_claim_to_own_task", 0.12, 0.12),
        ("setup", "install_or_configure_first_project", 0.22, 0.22),
        ("first_run", "complete_first_real_task", 0.28, 0.24),
        ("debug_recover", "understand_error_and_recover", 0.20, 0.18),
        ("integrate_workflow", "reuse_inside_existing_workflow", 0.18, 0.20),
    )
    steps: list[ProductUsageStep] = []
    previous_blocked = False
    for stage, objective, stage_difficulty, stage_weight in stages:
        stage_noise = rng.uniform(
            "event", -0.055, 0.055, kind=f"usage_stage_noise:{artifact.id}:{stage}"
        )
        stage_friction = clamp01(
            0.36 * base_friction
            + stage_difficulty
            + 0.18 * artifact.learning_curve * (1.0 - profile.technical_skill)
            + 0.12 * artifact.learning_curve * (1.0 - profile.cognitive_capacity)
            + 0.08 * cost * (1.2 - profile.technical_skill)
            + 0.07 * body.fatigue
            + 0.06 * previous_blocked
            + stage_noise
        )
        support = clamp01(
            0.18
            + 0.20 * capability
            + 0.18 * reliability
            + 0.16 * task_fit
            + 0.14 * usability
            + 0.12 * profile.technical_skill
            + 0.10 * profile.cognitive_capacity
            + 0.06 * agent.action_budgets.attention_budget
        )
        stage_success = clamp01(
            support
            - 0.22 * task_difficulty
            - 0.28 * stage_friction
            - 0.09 * previous_blocked
            + rng.uniform(
                "event",
                -0.045,
                0.045,
                kind=f"usage_stage_success:{artifact.id}:{stage}",
            )
        )
        stage_failure_probability = clamp01(
            failure_probability * (0.70 + stage_weight)
            + 0.20 * (1.0 - stage_success)
            + 0.06 * previous_blocked
        )
        failed = (
            rng.random("event", kind=f"usage_stage_failure_draw:{artifact.id}:{stage}")
            < stage_failure_probability
        )
        error_event = _stage_failure_event(artifact, rng, stage) if failed else None
        diagnostic_clarity = clamp01(
            0.18
            + 0.24 * profile.technical_skill
            + 0.20 * profile.cognitive_capacity
            + 0.18 * reliability
            + 0.14 * (1.0 - artifact.learning_curve)
            + 0.08 * task_fit
            - 0.18 * stage_friction
            - 0.08 * failed
            + rng.uniform(
                "event",
                -0.04,
                0.04,
                kind=f"usage_stage_diagnostic:{artifact.id}:{stage}",
            )
        )
        workaround_success = (
            clamp01(
                0.08
                + 0.24 * profile.technical_skill
                + 0.20 * profile.cognitive_capacity
                + 0.24 * diagnostic_clarity
                + 0.12 * agent.affect.patience
                + 0.08 * agent.action_budgets.attention_budget
                - 0.12 * profile.deadline_pressure
                - 0.12 * body.stress
            )
            if failed
            else 1.0
        )
        time_cost = clamp01(
            0.05 + 0.24 * stage_friction + 0.12 * (1.0 - stage_success) + 0.08 * failed
        )
        evidence = (
            f"stage:{stage}",
            f"objective:{objective}",
            f"task:{task_type}",
            f"error:{error_event}"
            if error_event
            else "completed_without_visible_error",
        )
        steps.append(
            ProductUsageStep(
                stage=stage,
                objective=objective,
                success=stage_success,
                friction=stage_friction,
                time_cost=time_cost,
                diagnostic_clarity=diagnostic_clarity,
                workaround_success=workaround_success,
                error_event=error_event,
                user_visible_evidence=evidence,
            )
        )
        previous_blocked = failed and workaround_success < 0.55
    return tuple(steps)


def _journey_success(steps: tuple[ProductUsageStep, ...]) -> float:
    weights = {
        "discover_fit": 0.12,
        "setup": 0.18,
        "first_run": 0.30,
        "debug_recover": 0.16,
        "integrate_workflow": 0.24,
    }
    total_weight = sum(weights.get(step.stage, 0.1) for step in steps)
    if not steps or total_weight <= 0:
        return 0.5
    return clamp01(
        sum(step.success * weights.get(step.stage, 0.1) for step in steps)
        / total_weight
    )


def _diagnostic_clarity(steps: tuple[ProductUsageStep, ...]) -> float:
    failure_steps = [step for step in steps if step.error_event]
    scoped_steps = failure_steps or list(steps)
    return _mean(step.diagnostic_clarity for step in scoped_steps) or 0.5


def _workaround_success(steps: tuple[ProductUsageStep, ...]) -> float:
    failure_steps = [step for step in steps if step.error_event]
    if not failure_steps:
        return 1.0
    return _mean(step.workaround_success for step in failure_steps) or 0.0


def _first_value_time(steps: tuple[ProductUsageStep, ...]) -> float:
    if not steps:
        return 0.5
    elapsed = 0.0
    total = sum(step.time_cost for step in steps) or 1.0
    for step in steps:
        elapsed += step.time_cost
        if step.stage in {"first_run", "integrate_workflow"} and step.success >= 0.55:
            return clamp01(elapsed / total)
    return 1.0


def _stage_success(steps: tuple[ProductUsageStep, ...], stage: str) -> float:
    for step in steps:
        if step.stage == stage:
            return step.success
    return 0.5


def _blocked_stage(steps: tuple[ProductUsageStep, ...]) -> str | None:
    if not steps:
        return None
    ranked = sorted(
        steps,
        key=lambda step: (
            float(step.error_event is not None),
            1.0 - step.success,
            step.friction,
            1.0 - step.diagnostic_clarity,
        ),
        reverse=True,
    )
    worst = ranked[0]
    if worst.error_event or worst.success < 0.42 or worst.friction > 0.62:
        return worst.stage
    return None


def _stage_failure_event(artifact: Artifact, rng: SeededRandom, stage: str) -> str:
    stage_defaults = {
        "discover_fit": "unclear_fit_or_missing_example",
        "setup": "dependency_resolution_failure",
        "first_run": "runtime_or_build_failure",
        "debug_recover": "diagnostic_dead_end",
        "integrate_workflow": "workflow_integration_gap",
    }
    if not artifact.failure_modes:
        return stage_defaults.get(stage, "generic_failure")
    if stage == "setup":
        preferred = [
            mode
            for mode in artifact.failure_modes
            if _looks_like_dependency_failure(mode)
        ]
    elif stage == "first_run":
        preferred = [
            mode
            for mode in artifact.failure_modes
            if any(
                token in mode.lower()
                for token in (
                    "build",
                    "bundle",
                    "compile",
                    "server",
                    "runtime",
                    "hmr",
                    "reload",
                )
            )
        ]
    elif stage == "debug_recover":
        preferred = [
            mode
            for mode in artifact.failure_modes
            if any(
                token in mode.lower()
                for token in ("error", "debug", "diagnostic", "stack", "message")
            )
        ]
    else:
        preferred = []
    modes = tuple(
        preferred
        or artifact.failure_modes
        or (stage_defaults.get(stage, "generic_failure"),)
    )
    index = rng.choice_index(
        "event",
        [1.0 for _ in modes],
        kind=f"usage_stage_failure_mode:{artifact.id}:{stage}",
    )
    return modes[index]


def _select_task_type(*, profile_need: float, artifact: Artifact) -> str:
    """Compatibility helper retained for older callers; runtime uses demand-first sampling."""

    if not artifact.task_fit_distribution:
        return "general_task"
    ranked = sorted(
        artifact.task_fit_distribution.items(),
        key=lambda item: (item[1] + 0.05 * profile_need, item[0]),
        reverse=True,
    )
    return ranked[0][0]


def _task_fit(artifact: Artifact, task_type: str) -> float:
    if task_type in artifact.task_fit_distribution:
        return clamp01(artifact.task_fit_distribution[task_type])
    return _mean(artifact.task_fit_distribution.values()) or 0.5


def _failure_event(artifact: Artifact, rng: SeededRandom) -> str | None:
    if not artifact.failure_modes:
        return "generic_failure"
    index = rng.choice_index(
        "event",
        [1.0 for _ in artifact.failure_modes],
        kind=f"experience_failure_mode:{artifact.id}",
    )
    return artifact.failure_modes[index]


def _mean(values) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0
