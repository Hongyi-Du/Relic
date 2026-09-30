"""Typed Society-Core state and event schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class Mode(str, Enum):
    INSTITUTION_GENESIS = "Institution-Genesis"
    REAL_WORLD_LOADOUT = "Real-World-Loadout"


class ActionKind(str, Enum):
    # Physical / survival
    EAT = "eat"
    REST = "rest"
    SLEEP = "sleep"
    MOVE = "move"
    WORK_FOR_RESOURCE = "work_for_resource"
    SEEK_RESOURCE = "seek_resource"
    AVOID_DANGER = "avoid_danger"
    RECOVER = "recover"

    # Information
    NOTICE = "notice"
    IGNORE = "ignore"
    READ = "read"
    SEARCH = "search"
    INSPECT = "inspect"
    VERIFY = "verify"
    COMPARE_SOURCES = "compare_sources"
    ASK_QUESTION = "ask_question"
    BOOKMARK = "bookmark"

    # Communication
    POST = "post"
    REPLY = "reply"
    QUOTE = "quote"
    REPOST = "repost"
    DM = "dm"
    ASK_PEER = "ask_peer"
    ANSWER_PEER = "answer_peer"
    WARN_PEER = "warn_peer"
    HELP_PEER = "help_peer"
    REFUSE_REQUEST = "refuse_request"
    EXPRESS_UNCERTAINTY = "express_uncertainty"
    CHALLENGE_CLAIM = "challenge_claim"
    SHARE_EXPERIENCE = "share_experience"

    # Artifact-neutral
    INSPECT_ARTIFACT = "inspect_artifact"
    TRY_ARTIFACT_ON_TASK = "try_artifact_on_task"
    REUSE_ARTIFACT = "reuse_artifact"
    SUGGEST_UPGRADE = "suggest_upgrade"
    PAY_FOR_ARTIFACT = "pay_for_artifact"
    COMPARE_WITH_ALTERNATIVE = "compare_with_alternative"
    SHARE_FAILURE = "share_failure"
    SHARE_SUCCESS = "share_success"
    SHARE_WORKAROUND = "share_workaround"
    WRITE_TUTORIAL = "write_tutorial"
    ABANDON_ARTIFACT = "abandon_artifact"
    SWITCH_TO_ALTERNATIVE = "switch_to_alternative"

    # Informal social reaction
    CALL_OUT_BEHAVIOR = "call_out_behavior"
    ASK_FOR_EXPLANATION = "ask_for_explanation"
    PRIVATELY_GOSSIP = "privately_gossip"
    REDUCE_TRUST = "reduce_trust"
    AVOID_FUTURE_INTERACTION = "avoid_future_interaction"
    EXCLUDE_FROM_ATTENTION = "exclude_from_attention"
    SUPPORT_CALLOUT = "support_callout"
    OPPOSE_CALLOUT = "oppose_callout"
    MEDIATE_CONFLICT = "mediate_conflict"
    IMITATE_BEHAVIOR = "imitate_behavior"


FORBIDDEN_MACRO_ACTIONS: frozenset[str] = frozenset(
    {
        "adopt_product",
        "create_norm",
        "create_institution",
        "become_moderator",
        "invoke_law",
        "report_violation_to_admin",
        "comply_with_rule",
        "ban_user",
        "declare_reputation_score",
        "declare_product_success",
    }
)


class EventKind(str, Enum):
    ACTION = "action"
    BODY_UPDATE = "body_update"
    RESOURCE_SHOCK = "resource_shock"
    RUMOR = "rumor"
    PUBLIC_CONFLICT = "public_conflict"
    NEW_ARTIFACT_RELEASE = "new_artifact_release"
    ARTIFACT_FAILURE_REPORT = "artifact_failure_report"
    USER_TUTORIAL = "user_tutorial"
    EXPERT_REVIEW_LIKE_POST = "expert_review_like_post"
    COMPETITOR_CLAIM_LIKE_POST = "competitor_claim_like_post"
    PUBLIC_APOLOGY_OR_CORRECTION = "public_apology_or_correction"
    PLATFORM_AFFORDANCE_CHANGE = "platform_affordance_change"
    HEALTH_OR_FATIGUE_SHOCK = "health_or_fatigue_shock"
    COORDINATION_OPPORTUNITY = "coordination_opportunity"


FORBIDDEN_INITIAL_OBJECT_TYPES: frozenset[str] = frozenset(
    {
        "law",
        "legal_code",
        "administrator",
        "moderator",
        "formal_governance",
        "company_governance_rule",
        "forum_rule",
        "terms_of_service",
        "formal_review_board",
        "official_audit_mechanism",
        "fixed_punishment_schedule",
        "public_norm",
        "norm_label",
        "verified_expert_badge",
        "reputation_board",
        "product_success_label",
        "adoption_label",
        "official_community_role",
        "committee",
        "arbitrator",
        "court",
        "procurement_rule",
        "employment_contract",
        "membership_procedure",
        "appeal_procedure",
        "report_queue",
        "authority_ban_mechanism",
        "customer_success_process",
        "formal_customer_support",
        "sla",
        "warranty",
        "contract_system",
    }
)


@dataclass
class BodyState:
    hunger: float = 0.3
    sleep_pressure: float = 0.2
    fatigue: float = 0.2
    stress: float = 0.2
    health_reserve: float = 0.9
    physiological_energy: float = 0.8

    def clamped(self) -> "BodyState":
        return BodyState(
            hunger=clamp01(self.hunger),
            sleep_pressure=clamp01(self.sleep_pressure),
            fatigue=clamp01(self.fatigue),
            stress=clamp01(self.stress),
            health_reserve=clamp01(self.health_reserve),
            physiological_energy=clamp01(self.physiological_energy),
        )


@dataclass
class MaterialResources:
    food_or_budget_token: float = 1.0
    artifact_access: float = 0.0


@dataclass
class ActionBudgets:
    attention_budget: float = 0.8
    time_budget: float = 1.0


@dataclass
class AffectState:
    mood: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    curiosity: float = 0.5
    patience: float = 0.7


@dataclass
class CognitiveState:
    attention_filter: str = "balanced"
    attention_allocation_policy: str = "need_then_social_then_artifact"
    working_memory_items: list[str] = field(default_factory=list)
    beliefs_about_artifacts: dict[str, float] = field(default_factory=dict)
    artifact_peer_signal_strength: dict[str, float] = field(default_factory=dict)
    artifact_negative_signal_strength: dict[str, float] = field(default_factory=dict)
    artifact_peer_signal_count: dict[str, int] = field(default_factory=dict)
    beliefs_about_people: dict[str, float] = field(default_factory=dict)
    beliefs_about_claims: dict[str, float] = field(default_factory=dict)
    private_norm_hypotheses: dict[str, float] = field(default_factory=dict)
    institution_feedback: dict[str, float] = field(default_factory=dict)
    uncertainty: float = 0.4
    goal_stack: list[str] = field(default_factory=list)
    habit_strengths: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentProfile:
    persona_label: str = "general_user"
    primary_domain: str = "general_software"
    community_memberships: tuple[str, ...] = ()
    task_demand_weights: dict[str, float] = field(default_factory=dict)
    cognitive_percentile: float = 0.5
    cognitive_capacity: float = 0.5
    intelligence_tier: str = "medium"
    llm_model_tier: str = "standard"
    activity_tier: str = "normal"
    technical_role: str = "general_user"
    innovation_role: str = "mainstream_evaluator"
    daily_active_probability: float = 0.45
    reading_propensity: float = 0.85
    posting_propensity: float = 0.18
    commenting_propensity: float = 0.35
    technical_skill: float = 0.5
    creative_capacity: float = 0.5
    domain_need: float = 0.5
    budget_sensitivity: float = 0.5
    risk_tolerance: float = 0.5
    novelty_seeking: float = 0.5
    reliability_preference: float = 0.5
    privacy_sensitivity: float = 0.5
    community_trust: float = 0.5
    peer_susceptibility: float = 0.5
    social_activity: float = 0.5
    opinion_leadership: float = 0.5
    deadline_pressure: float = 0.5
    strategic_boldness: float = 0.5


@dataclass
class SocialState:
    tie_strength: dict[str, float] = field(default_factory=dict)
    trust: dict[str, float] = field(default_factory=dict)
    familiarity: dict[str, float] = field(default_factory=dict)
    debt_or_obligation: dict[str, float] = field(default_factory=dict)
    past_interaction_count: dict[str, int] = field(default_factory=dict)
    perceived_status: dict[str, float] = field(default_factory=dict)
    social_trust_as_private_belief: dict[str, float] = field(default_factory=dict)


@dataclass
class MemoryItem:
    event_ref: str
    semantic_summary: str
    source: str
    valence: float = 0.0
    importance: float = 0.5
    uncertainty: float = 0.5
    timestamp: int = 0
    last_retrieved: int = 0
    decay_rate: float = 0.01
    llm_summary_ref_optional: str | None = None


class ContentKind(str, Enum):
    POST = "post"
    REPLY = "reply"
    REPOST = "repost"
    DM = "dm"
    TUTORIAL = "tutorial"
    FAILURE_REPORT = "failure_report"
    SUCCESS_REPORT = "success_report"
    UPGRADE_SUGGESTION = "upgrade_suggestion"
    RUMOR = "rumor"
    QUESTION = "question"
    CALLOUT = "callout"


@dataclass
class ContentItem:
    id: str
    author_id: str
    channel: str
    kind: ContentKind
    created_at: int
    parent_id: str | None = None
    topic_vector: dict[str, float] = field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()
    claim_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    text_surface: str = ""
    visibility_scope: str = "public"
    recipient_ids: tuple[str, ...] = ()
    engagement_counts: dict[str, int] = field(default_factory=dict)
    controversy_proxy: float = 0.0
    indexed_at: int | None = None
    deleted_by_author_flag: bool = False


@dataclass(frozen=True)
class Community:
    id: str
    label: str
    channel_id: str
    topic_vector: dict[str, float] = field(default_factory=dict)
    activity_multiplier: float = 1.0
    technical_depth: float = 0.5
    debate_intensity: float = 0.5


@dataclass(frozen=True)
class Observation:
    agent_id: str
    tick: int
    visible_content_ids: tuple[str, ...]
    dm_ids: tuple[str, ...]
    search_result_ids: tuple[str, ...]
    local_event_ids: tuple[str, ...]
    own_memory_refs: tuple[str, ...]


@dataclass(frozen=True)
class ProductUsageStep:
    stage: str
    objective: str
    success: float
    friction: float
    time_cost: float
    diagnostic_clarity: float
    workaround_success: float
    error_event: str | None = None
    user_visible_evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductExperiencePacket:
    experience_id: str
    artifact_id: str
    tick: int
    task_type: str
    task_difficulty: float
    baseline_success_without_product: float
    objective_success: float
    success_gain: float
    time_saved: float
    friction: float
    failure_event: str | None
    observed_reliability: float
    attention_cost: float
    budget_cost: float
    comparison_to_current_method: float
    journey_steps: tuple[ProductUsageStep, ...] = ()
    blocked_stage: str | None = None
    diagnostic_clarity: float = 0.5
    workaround_success: float = 0.0
    first_value_time: float = 0.5
    integration_quality: float = 0.5
    repeat_use_value: float = 0.5
    objective_observation_refs: tuple[str, ...] = ()
    objective_observation_set_hash: str = ""
    objective_observation_manifest_hash: str = ""
    raw_execution_success: float | None = None

    @property
    def experienced_task_success(self) -> float:
        """Profile-conditioned task outcome retained under the legacy field."""

        return self.objective_success


@dataclass(frozen=True)
class UpgradeSuggestion:
    suggestion_id: str
    artifact_id: str
    author_id: str
    tick: int
    source_experience_ref: str | None
    theme: str
    priority: float
    confidence: float
    requested_change: str
    affected_tasks: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()
    profile_ref: str = ""
    visibility_scope: str = "candidate"
    llm_intent_ref: str | None = None
    subjective_feedback_source: str = ""
    subjective_feedback: str = ""
    llm_suggested_improvement: str = ""
    llm_feature_idea: str = ""
    contribution_mode: str = "user_feedback"
    technical_depth: float = 0.5
    novelty_score: float = 0.5


@dataclass(frozen=True)
class LLMIntentRecord:
    intent_id: str
    artifact_id: str
    tick: int
    source: str
    profile_ref: str
    experience_ref: str | None
    intent_to_try: float
    intent_to_pay: float
    max_willingness_to_pay: float
    intent_to_recommend: float
    subjective_satisfaction: float
    confidence: float
    reason_code: str
    next_action_preference: str
    blocked_reason: str | None = None
    experience_summary: str = ""
    product_feedback: str = ""
    suggested_improvements: tuple[str, ...] = ()
    improvement_themes: tuple[str, ...] = ()
    feature_ideas: tuple[str, ...] = ()
    feature_themes: tuple[str, ...] = ()
    input_evidence_hash: str = ""
    output_evidence_hash: str = ""


@dataclass
class AgentState:
    id: str
    profile: AgentProfile = field(default_factory=AgentProfile)
    body: BodyState = field(default_factory=BodyState)
    affect: AffectState = field(default_factory=AffectState)
    cognition: CognitiveState = field(default_factory=CognitiveState)
    memory: list[MemoryItem] = field(default_factory=list)
    material_resources: MaterialResources = field(default_factory=MaterialResources)
    action_budgets: ActionBudgets = field(default_factory=ActionBudgets)
    location: str = "home"
    social: SocialState = field(default_factory=SocialState)
    channel_access: set[str] = field(default_factory=lambda: {"public_forum", "feed", "dm", "search"})
    artifact_access: set[str] = field(default_factory=set)
    artifact_exposure_sources: dict[str, str] = field(default_factory=dict)
    artifact_exposure_groups: dict[str, str] = field(default_factory=dict)
    artifact_peer_signal_groups: dict[str, str] = field(default_factory=dict)
    artifact_exposure_ticks: dict[str, int] = field(default_factory=dict)
    artifact_exposed_by: dict[str, str] = field(default_factory=dict)
    artifact_exposure_content: dict[str, str] = field(default_factory=dict)
    artifact_exposure_distance: dict[str, int] = field(default_factory=dict)
    artifact_usage_counts: dict[str, int] = field(default_factory=dict)
    artifact_payment_state: dict[str, str] = field(default_factory=dict)
    artifact_payment_ticks: dict[str, int] = field(default_factory=dict)
    latest_artifact_experience: dict[str, ProductExperiencePacket] = field(default_factory=dict)
    artifact_experience_history: dict[str, list[ProductExperiencePacket]] = field(default_factory=dict)
    artifact_upgrade_suggestions: dict[str, list[UpgradeSuggestion]] = field(default_factory=dict)
    latest_llm_intents: dict[str, LLMIntentRecord] = field(default_factory=dict)
    llm_intent_session_history: list[LLMIntentRecord] = field(default_factory=list)
    llm_intent_feedback_history: list[LLMIntentRecord] = field(default_factory=list)
    action_affordances: set[str] = field(default_factory=set)
    subscriptions: set[str] = field(default_factory=lambda: {"public_forum", "feed"})
    search_queries: tuple[str, ...] = ()


@dataclass(frozen=True)
class Artifact:
    id: str
    provider_id: str | None
    artifact_kind: str
    capability_profile: dict[str, float] = field(default_factory=dict)
    cost_profile: dict[str, float] = field(default_factory=dict)
    access_constraints: dict[str, float] = field(default_factory=dict)
    learning_curve: float = 0.5
    failure_modes: tuple[str, ...] = ()
    reliability_profile: dict[str, float] = field(default_factory=dict)
    evidence_claims: tuple[str, ...] = ()
    version: str = "0.1"
    release_time: int = 0
    public_claims: tuple[str, ...] = ()
    task_fit_distribution: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    actor_id: str
    kind: ActionKind
    target_ids: tuple[str, ...] = ()
    channel_id: str | None = None
    artifact_id: str | None = None
    content_id: str | None = None
    typed_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalEvent:
    event_id: str
    tick: int
    kind: EventKind
    actor_id: str | None = None
    action_type: str | None = None
    target_ids: tuple[str, ...] = ()
    channel_id: str | None = None
    artifact_id: str | None = None
    content_id: str | None = None
    location_id: str | None = None
    pre_state_hash: str | None = None
    post_state_hash: str | None = None
    observation_refs: tuple[str, ...] = ()
    memory_refs: tuple[str, ...] = ()
    random_draw_refs: tuple[str, ...] = ()
    typed_payload: dict[str, Any] = field(default_factory=dict)
    public_visibility: str = "private"
    company_visible_flag: bool = False
    detector_visible_only_after_run: bool = True


@dataclass
class EmergentInstitution:
    id: str
    pattern: str
    stage: str
    created_at: int
    support_event_refs: tuple[str, ...]
    role_actor_ids: tuple[str, ...] = ()
    procedure_content_refs: tuple[str, ...] = ()
    stability_score: float = 0.0
    behavioral_feedback: dict[str, float] = field(default_factory=dict)


@dataclass
class SocietyState:
    tick: int = 0
    mode: Mode = Mode.INSTITUTION_GENESIS
    agents: dict[str, AgentState] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    content: dict[str, ContentItem] = field(default_factory=dict)
    communities: dict[str, Community] = field(default_factory=dict)
    community_archetype_map: dict[str, str] = field(default_factory=dict)
    search_index_terms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    upgrade_suggestions: dict[str, UpgradeSuggestion] = field(default_factory=dict)
    event_queue: list[ExternalEvent] = field(default_factory=list)
    event_log: list[ExternalEvent] = field(default_factory=list)
    emergent_institutions: dict[str, EmergentInstitution] = field(default_factory=dict)
    forbidden_initial_objects_seen: tuple[str, ...] = ()


@dataclass
class RunManifest:
    run_id: str
    spec_version: str
    code_commit: str
    python_version: str
    scenario_config_hash: str
    parameter_registry_hash: str
    agent_population_hash: str
    network_seed: int
    action_seed: int
    feed_seed: int
    exogenous_event_seed: int
    llm_enabled: bool
    detectors_enabled_during_run: bool
    detectors_run_after_completion: bool
    detector_construct_validity_hash: str
    bounded_score_registry_hash: str
    claim_audit_hash: str
    robustness_audit_plan_hash: str
    uncertainty_report_plan_hash: str
    type_separation_schema_hash: str
    state_hash_mode: str
    event_log_hash: str
    state_snapshot_hashes: dict[str, str]
