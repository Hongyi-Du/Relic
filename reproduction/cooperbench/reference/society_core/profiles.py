"""Stable v16 agent profile generation."""

from __future__ import annotations

from statistics import NormalDist

from .communities import DOMAIN_COMMUNITIES
from .demand import DOMAIN_TASK_DEMANDS
from .randomness import SeededRandom
from .schemas import AgentProfile, clamp01


_NORMAL = NormalDist()

DEFAULT_TECHNICAL_EXPERT_FRACTION = 0.20
DEFAULT_TECHNICAL_PRACTITIONER_FRACTION = 0.35
DEFAULT_INNOVATION_ORIGINATOR_FRACTION = 0.025
DEFAULT_INNOVATION_EARLY_BUILDER_FRACTION = 0.135
DEFAULT_INNOVATION_PRAGMATIC_ADAPTER_FRACTION = 0.34


def build_population_role_quantiles(
    population_size: int,
    rng: SeededRandom,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return independently randomized, exactly stratified role quantiles."""

    if population_size <= 0:
        return {}, {}
    return (
        _ranked_role_quantiles(population_size, rng, "technical"),
        _ranked_role_quantiles(population_size, rng, "innovation"),
    )


def build_population_stratification_quantiles(
    population_size: int,
    rng: SeededRandom,
) -> dict[str, dict[int, float]]:
    """Return independent randomized ranks for calibrated population margins."""

    technical, innovation = build_population_role_quantiles(population_size, rng)
    return {
        "technical": technical,
        "innovation": innovation,
        "activity": _ranked_role_quantiles(population_size, rng, "activity"),
        "cognitive": _ranked_role_quantiles(population_size, rng, "cognitive"),
    }


def _ranked_role_quantiles(
    population_size: int,
    rng: SeededRandom,
    dimension: str,
) -> dict[int, float]:
    scores = [
        (
            rng.random(
                "profile_roles",
                kind=f"profile_{dimension}_role_rank:{index}",
            ),
            index,
        )
        for index in range(population_size)
    ]
    return {
        index: (rank + 0.5) / population_size
        for rank, (_, index) in enumerate(sorted(scores))
    }


def build_agent_profile(
    agent_id: str,
    index: int,
    rng: SeededRandom,
    *,
    technical_role_quantile: float | None = None,
    innovation_role_quantile: float | None = None,
    activity_quantile: float | None = None,
    cognitive_quantile: float | None = None,
    technical_expert_fraction: float = DEFAULT_TECHNICAL_EXPERT_FRACTION,
    technical_practitioner_fraction: float = (
        DEFAULT_TECHNICAL_PRACTITIONER_FRACTION
    ),
    innovation_originator_fraction: float = (
        DEFAULT_INNOVATION_ORIGINATOR_FRACTION
    ),
    innovation_early_builder_fraction: float = (
        DEFAULT_INNOVATION_EARLY_BUILDER_FRACTION
    ),
    innovation_pragmatic_adapter_fraction: float = (
        DEFAULT_INNOVATION_PRAGMATIC_ADAPTER_FRACTION
    ),
) -> AgentProfile:
    """Create a deterministic heterogeneous user profile.

    The profile is stable identity state. It can be read by the subjective
    intent layer, but it is owned by the typed runtime and is included in replay
    hashes.
    """

    _validate_role_fractions(
        technical_expert_fraction=technical_expert_fraction,
        technical_practitioner_fraction=technical_practitioner_fraction,
        innovation_originator_fraction=innovation_originator_fraction,
        innovation_early_builder_fraction=innovation_early_builder_fraction,
        innovation_pragmatic_adapter_fraction=(
            innovation_pragmatic_adapter_fraction
        ),
    )
    technical_quantile = _bounded_percentile(
        technical_role_quantile
        if technical_role_quantile is not None
        else rng.random("profile", kind="profile_technical_role_quantile")
    )
    innovation_quantile = _bounded_percentile(
        innovation_role_quantile
        if innovation_role_quantile is not None
        else rng.random("profile", kind="profile_innovation_role_quantile")
    )
    technical_role = _technical_role_from_quantile(
        technical_quantile,
        expert_fraction=technical_expert_fraction,
        practitioner_fraction=technical_practitioner_fraction,
    )
    innovation_role = _innovation_role_from_quantile(
        innovation_quantile,
        originator_fraction=innovation_originator_fraction,
        early_builder_fraction=innovation_early_builder_fraction,
        pragmatic_adapter_fraction=innovation_pragmatic_adapter_fraction,
    )
    cognitive_percentile = _bounded_percentile(
        cognitive_quantile
        if cognitive_quantile is not None
        else rng.random("profile", kind="profile_cognitive_percentile")
    )
    cognitive_capacity = _cognitive_capacity_from_percentile(cognitive_percentile)
    intelligence_tier = _intelligence_tier_from_percentile(cognitive_percentile)
    llm_model_tier = _model_tier_from_percentile(cognitive_percentile)
    activity = _activity_profile(rng, quantile=activity_quantile)
    primary_domain = _primary_domain(rng)
    task_demand_weights = _task_demand_weights(primary_domain, rng)
    community_memberships = _community_memberships(primary_domain, rng)

    domain_need = rng.uniform("profile", 0.05, 0.95, kind="profile_domain_need")
    budget_sensitivity = rng.uniform("profile", 0.05, 0.95, kind="profile_budget_sensitivity")
    risk_tolerance = rng.uniform("profile", 0.05, 0.95, kind="profile_risk_tolerance")
    novelty_seeking = rng.uniform("profile", 0.05, 0.95, kind="profile_novelty")
    technical_skill = _technical_skill(
        role=technical_role,
        cognitive_capacity=cognitive_capacity,
        rng=rng,
    )
    creative_capacity = _creative_capacity(
        role=innovation_role,
        novelty_seeking=novelty_seeking,
        risk_tolerance=risk_tolerance,
        cognitive_capacity=cognitive_capacity,
        rng=rng,
    )
    reliability_preference = rng.uniform("profile", 0.05, 0.95, kind="profile_reliability")
    privacy_sensitivity = rng.uniform("profile", 0.05, 0.95, kind="profile_privacy")
    community_trust = rng.uniform("profile", 0.05, 0.95, kind="profile_community_trust")
    peer_susceptibility = rng.uniform("profile", 0.05, 0.95, kind="profile_peer_susceptibility")
    social_activity = clamp01(
        0.62 * activity["posting_propensity"]
        + 0.23 * activity["commenting_propensity"]
        + 0.15 * rng.uniform("profile", 0.05, 0.95, kind="profile_social_activity_noise")
    )
    opinion_leadership = clamp01(
        0.52 * social_activity
        + 0.28 * cognitive_capacity
        + 0.20 * rng.uniform("profile", 0.05, 0.95, kind="profile_opinion_leadership")
    )
    deadline_pressure = rng.uniform("profile", 0.05, 0.95, kind="profile_deadline_pressure")
    strategic_boldness = clamp01(
        0.20 * risk_tolerance
        + 0.18 * novelty_seeking
        + 0.18 * creative_capacity
        + 0.16 * opinion_leadership
        + 0.14 * cognitive_capacity
        + 0.12 * technical_skill
        + 0.10 * domain_need
        - 0.08 * privacy_sensitivity
    )

    persona_label = _persona_label(
        technical_skill=technical_skill,
        technical_role=technical_role,
        innovation_role=innovation_role,
        creative_capacity=creative_capacity,
        domain_need=domain_need,
        budget_sensitivity=budget_sensitivity,
        reliability_preference=reliability_preference,
        social_activity=social_activity,
        opinion_leadership=opinion_leadership,
        strategic_boldness=strategic_boldness,
        index=index,
    )
    return AgentProfile(
        persona_label=f"{persona_label}:{agent_id}",
        primary_domain=primary_domain,
        community_memberships=community_memberships,
        task_demand_weights=task_demand_weights,
        cognitive_percentile=clamp01(cognitive_percentile),
        cognitive_capacity=clamp01(cognitive_capacity),
        intelligence_tier=intelligence_tier,
        llm_model_tier=llm_model_tier,
        activity_tier=activity["activity_tier"],
        technical_role=technical_role,
        innovation_role=innovation_role,
        daily_active_probability=clamp01(activity["daily_active_probability"]),
        reading_propensity=clamp01(activity["reading_propensity"]),
        posting_propensity=clamp01(activity["posting_propensity"]),
        commenting_propensity=clamp01(activity["commenting_propensity"]),
        technical_skill=clamp01(technical_skill),
        creative_capacity=clamp01(creative_capacity),
        domain_need=clamp01(domain_need),
        budget_sensitivity=clamp01(budget_sensitivity),
        risk_tolerance=clamp01(risk_tolerance),
        novelty_seeking=clamp01(novelty_seeking),
        reliability_preference=clamp01(reliability_preference),
        privacy_sensitivity=clamp01(privacy_sensitivity),
        community_trust=clamp01(community_trust),
        peer_susceptibility=clamp01(peer_susceptibility),
        social_activity=clamp01(social_activity),
        opinion_leadership=clamp01(opinion_leadership),
        deadline_pressure=clamp01(deadline_pressure),
        strategic_boldness=clamp01(strategic_boldness),
    )


def _persona_label(
    *,
    technical_skill: float,
    technical_role: str,
    innovation_role: str,
    creative_capacity: float,
    domain_need: float,
    budget_sensitivity: float,
    reliability_preference: float,
    social_activity: float,
    opinion_leadership: float,
    strategic_boldness: float,
    index: int,
) -> str:
    if (
        technical_role == "professional_engineer"
        and innovation_role in {"creative_originator", "early_builder"}
        and creative_capacity > 0.70
    ):
        return "technical_innovator"
    if innovation_role == "creative_originator" and creative_capacity > 0.78:
        return "feature_originator"
    if strategic_boldness > 0.72 and technical_skill > 0.58:
        return "bold_architect"
    if technical_role == "professional_engineer":
        return "professional_engineer"
    if technical_skill > 0.7 and reliability_preference > 0.6:
        return "skeptical_expert"
    if budget_sensitivity > 0.7 and domain_need > 0.55:
        return "budget_constrained_user"
    if social_activity > 0.65 and opinion_leadership > 0.55:
        return "community_amplifier"
    if domain_need > 0.7 and technical_skill < 0.45:
        return "high_need_newcomer"
    if index % 7 == 0:
        return "quiet_lurker"
    return "general_user"


def _technical_role_from_quantile(
    quantile: float,
    *,
    expert_fraction: float,
    practitioner_fraction: float,
) -> str:
    if quantile < expert_fraction:
        return "professional_engineer"
    if quantile < expert_fraction + practitioner_fraction:
        return "technical_practitioner"
    return "general_user"


def _innovation_role_from_quantile(
    quantile: float,
    *,
    originator_fraction: float,
    early_builder_fraction: float,
    pragmatic_adapter_fraction: float,
) -> str:
    if quantile < originator_fraction:
        return "creative_originator"
    if quantile < originator_fraction + early_builder_fraction:
        return "early_builder"
    if quantile < (
        originator_fraction + early_builder_fraction + pragmatic_adapter_fraction
    ):
        return "pragmatic_adapter"
    return "mainstream_evaluator"


def _technical_skill(
    *,
    role: str,
    cognitive_capacity: float,
    rng: SeededRandom,
) -> float:
    bounds = {
        "professional_engineer": (0.78, 0.98),
        "technical_practitioner": (0.42, 0.82),
        "general_user": (0.08, 0.62),
    }
    low, high = bounds[role]
    latent = rng.uniform("profile", low, high, kind="profile_technical_skill")
    return clamp01(0.85 * latent + 0.15 * cognitive_capacity)


def _creative_capacity(
    *,
    role: str,
    novelty_seeking: float,
    risk_tolerance: float,
    cognitive_capacity: float,
    rng: SeededRandom,
) -> float:
    bounds = {
        "creative_originator": (0.82, 0.99),
        "early_builder": (0.64, 0.90),
        "pragmatic_adapter": (0.36, 0.74),
        "mainstream_evaluator": (0.08, 0.58),
    }
    low, high = bounds[role]
    latent = rng.uniform("profile", low, high, kind="profile_creative_capacity")
    return clamp01(
        0.68 * latent
        + 0.17 * novelty_seeking
        + 0.10 * risk_tolerance
        + 0.05 * cognitive_capacity
    )


def _validate_role_fractions(
    *,
    technical_expert_fraction: float,
    technical_practitioner_fraction: float,
    innovation_originator_fraction: float,
    innovation_early_builder_fraction: float,
    innovation_pragmatic_adapter_fraction: float,
) -> None:
    values = (
        technical_expert_fraction,
        technical_practitioner_fraction,
        innovation_originator_fraction,
        innovation_early_builder_fraction,
        innovation_pragmatic_adapter_fraction,
    )
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("profile_role_fractions_must_be_between_zero_and_one")
    if technical_expert_fraction + technical_practitioner_fraction > 1.0:
        raise ValueError("technical_role_fractions_exceed_one")
    if (
        innovation_originator_fraction
        + innovation_early_builder_fraction
        + innovation_pragmatic_adapter_fraction
        > 1.0
    ):
        raise ValueError("innovation_role_fractions_exceed_one")


def _bounded_percentile(value: float) -> float:
    return max(0.001, min(0.999, float(value)))


def _cognitive_capacity_from_percentile(percentile: float) -> float:
    z_score = _NORMAL.inv_cdf(_bounded_percentile(percentile))
    return clamp01((z_score + 3.0) / 6.0)


def _model_tier_from_percentile(percentile: float) -> str:
    if percentile < 0.16:
        return "low"
    if percentile > 0.84:
        return "high"
    return "standard"


def _intelligence_tier_from_percentile(percentile: float) -> str:
    if percentile < 0.16:
        return "general"
    if percentile > 0.84:
        return "high"
    return "medium"


def _activity_profile(
    rng: SeededRandom,
    *,
    quantile: float | None = None,
) -> dict[str, float | str]:
    """Calibrate online activity tiers to a heavy-tailed forum population."""

    draw = _bounded_percentile(
        quantile
        if quantile is not None
        else rng.random("profile", kind="profile_activity_tier")
    )
    if draw < 0.10:
        return {
            "activity_tier": "very_active",
            "daily_active_probability": 0.95,
            "reading_propensity": 0.98,
            "posting_propensity": 0.86,
            "commenting_propensity": 0.92,
        }
    if draw < 0.45:
        return {
            "activity_tier": "normal",
            "daily_active_probability": 0.45,
            "reading_propensity": 0.85,
            "posting_propensity": 0.18,
            "commenting_propensity": 0.35,
        }
    return {
        "activity_tier": "inactive",
        "daily_active_probability": 0.12,
        "reading_propensity": 0.42,
        "posting_propensity": 0.01,
        "commenting_propensity": 0.03,
    }


def _primary_domain(rng: SeededRandom) -> str:
    domains = ("frontend_product", "library_author", "ops_maintainer", "learner_builder")
    weights = [0.42, 0.18, 0.18, 0.22]
    return domains[rng.choice_index("profile", weights, kind="profile_primary_domain")]


def _community_memberships(primary_domain: str, rng: SeededRandom) -> tuple[str, ...]:
    memberships = list(DOMAIN_COMMUNITIES.get(primary_domain, ("frontend_build",)))
    if rng.random("profile", kind="profile_cross_community_membership") < 0.22:
        extras = [community for community in ("framework_authors", "ops_maintenance", "learners") if community not in memberships]
        if extras:
            memberships.append(extras[rng.choice_index("profile", [1.0] * len(extras), kind="profile_extra_community")])
    return tuple(memberships)


def _task_demand_weights(primary_domain: str, rng: SeededRandom) -> dict[str, float]:
    base = DOMAIN_TASK_DEMANDS.get(primary_domain, {})
    weights: dict[str, float] = {}
    for task_id, weight in base.items():
        jitter = rng.uniform("profile", 0.82, 1.18, kind=f"profile_task_demand:{task_id}")
        weights[task_id] = clamp01(weight * jitter)
    return weights
