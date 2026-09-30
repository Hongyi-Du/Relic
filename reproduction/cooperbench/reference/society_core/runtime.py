"""Standalone Society-Core runtime."""

from __future__ import annotations

import hashlib
import platform
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from collections.abc import Callable

from .channels import observe_local_slice
from .communities import (
    apply_substrate_communities,
    assign_agent_communities,
    initialize_communities,
    preferred_public_channel,
)
from .construct_validity import construct_validity_suite_hash
from .detectors import DetectorSuite
from .hashing import canonicalize, stable_hash
from .institutions import update_emergent_institutions
from .llm_intent import (
    DEFAULT_LLM_INTENT_HIGH_MODEL,
    DEFAULT_LLM_INTENT_LOW_MODEL,
    DEFAULT_LLM_INTENT_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_INTENT_REQUEST_ATTEMPTS,
    DEFAULT_LLM_INTENT_STANDARD_MODEL,
    HeuristicLLMIntentProvider,
    OpenAILLMIntentProvider,
    intent_input_evidence_hash,
)
from .network import (
    grant_artifact_access,
    initialize_social_graph,
    update_network_exposures_from_observation,
)
from .policy import choose_action
from .product_experience import simulate_product_experience, suggest_upgrades_from_experience
from .profiles import (
    DEFAULT_INNOVATION_EARLY_BUILDER_FRACTION,
    DEFAULT_INNOVATION_ORIGINATOR_FRACTION,
    DEFAULT_INNOVATION_PRAGMATIC_ADAPTER_FRACTION,
    DEFAULT_TECHNICAL_EXPERT_FRACTION,
    DEFAULT_TECHNICAL_PRACTITIONER_FRACTION,
    build_agent_profile,
    build_population_stratification_quantiles,
)
from .randomness import SeededRandom
from .registries import (
    BoundedScoreRegistry,
    ClaimAudit,
    ParameterRegistry,
    default_bounded_score_registry,
    default_claim_audit,
    default_parameter_registry,
)
from .schemas import (
    ActionBudgets,
    Action,
    ActionKind,
    AgentState,
    Artifact,
    BodyState,
    ContentItem,
    ContentKind,
    CognitiveState,
    EventKind,
    ExternalEvent,
    MaterialResources,
    ProductExperiencePacket,
    RunManifest,
    SocietyState,
    UpgradeSuggestion,
    clamp01,
)
from .transition import apply_action_effect, drift_body, make_action_event
from .validators import validate_action, validate_initial_state, validate_type_separation


@dataclass(frozen=True)
class SocietyConfig:
    population_size: int = 200
    seed: int = 42
    spec_version: str = "v17_architecture_hardened"
    scenario_id: str = "society_core_v17_external_society"
    detectors_enabled_during_run: bool = False
    llm_enabled: bool = False
    llm_intent_enabled: bool = True
    llm_intent_provider: str = "heuristic"
    llm_intent_model: str = DEFAULT_LLM_INTENT_STANDARD_MODEL
    llm_intent_low_model: str | None = DEFAULT_LLM_INTENT_LOW_MODEL
    llm_intent_standard_model: str | None = DEFAULT_LLM_INTENT_STANDARD_MODEL
    llm_intent_high_model: str | None = DEFAULT_LLM_INTENT_HIGH_MODEL
    llm_intent_timeout_seconds: float = 20.0
    llm_intent_request_attempts: int = DEFAULT_LLM_INTENT_REQUEST_ATTEMPTS
    llm_intent_max_output_tokens: int = DEFAULT_LLM_INTENT_MAX_OUTPUT_TOKENS
    llm_intent_strict_live: bool = False
    llm_intent_response_cache_dir: str | None = None
    llm_intent_audit_path: str | None = None
    llm_intent_progress_path: str | None = None
    llm_intent_retry_backoff_seconds: float = 0.0
    llm_intent_replay_reference_audit_path: str | None = None
    llm_intent_max_concurrency: int = 1
    llm_intent_cache_ttl_ticks: int = 3
    typed_product_experience_enabled: bool = True
    evidence_only_upgrade_suggestions: bool = False
    state_hash_mode: str = "merkle"
    network_enabled: bool = True
    max_social_out_degree: int = 8
    artifact_initial_access_fraction: float = 0.25
    network_holdout_fraction: float = 0.25
    peer_signal_control_fraction: float = 0.5
    peer_signal_utility_enabled: bool = True
    body_affect_coupling_enabled: bool = True
    technical_expert_fraction: float = DEFAULT_TECHNICAL_EXPERT_FRACTION
    technical_practitioner_fraction: float = (
        DEFAULT_TECHNICAL_PRACTITIONER_FRACTION
    )
    innovation_originator_fraction: float = DEFAULT_INNOVATION_ORIGINATOR_FRACTION
    innovation_early_builder_fraction: float = (
        DEFAULT_INNOVATION_EARLY_BUILDER_FRACTION
    )
    innovation_pragmatic_adapter_fraction: float = (
        DEFAULT_INNOVATION_PRAGMATIC_ADAPTER_FRACTION
    )


class SocietyRuntime:
    def __init__(
        self,
        config: SocietyConfig | None = None,
        parameter_registry: ParameterRegistry | None = None,
        bounded_score_registry: BoundedScoreRegistry | None = None,
        claim_audit: ClaimAudit | None = None,
    ) -> None:
        self.config = config or SocietyConfig()
        if self.config.llm_intent_max_concurrency <= 0:
            raise ValueError("llm_intent_max_concurrency_must_be_positive")
        self.rng = SeededRandom(self.config.seed)
        self.parameter_registry = parameter_registry or default_parameter_registry()
        self.bounded_score_registry = bounded_score_registry or default_bounded_score_registry()
        self.claim_audit = claim_audit or default_claim_audit()
        self.detectors = DetectorSuite()
        self.intent_provider = self._build_intent_provider()
        self.product_experience_packet_provider: Callable[..., ProductExperiencePacket] | None = None
        self.state = self._init_state()
        if self.config.network_enabled:
            initialize_social_graph(
                self.state,
                self.rng,
                max_out_degree=self.config.max_social_out_degree,
            )
        self._agent_hashes: dict[str, str] = {}
        self._profile_hashes: dict[str, str] = {}
        self._agent_memory_item_hashes: dict[str, list[str]] = {}
        self._agent_memory_hashes: dict[str, str] = {}
        self._artifact_hashes: dict[str, str] = {}
        self._content_hashes: dict[str, str] = {}
        self._search_index_hashes: dict[str, str] = {}
        self._suggestion_hashes: dict[str, str] = {}
        self._institution_hashes: dict[str, str] = {}
        self._experience_packet_hashes: dict[tuple[str, str], str] = {}
        self._communities_hash = ""
        self._hash_cache_dirty = True
        validate_initial_state(self.state)

    def _build_intent_provider(self):
        provider = self.config.llm_intent_provider.lower()
        if provider == "heuristic":
            return HeuristicLLMIntentProvider()
        if provider == "openai":
            return OpenAILLMIntentProvider(
                model=self.config.llm_intent_model,
                timeout_seconds=self.config.llm_intent_timeout_seconds,
                request_attempts=self.config.llm_intent_request_attempts,
                max_output_tokens=self.config.llm_intent_max_output_tokens,
                strict_live=self.config.llm_intent_strict_live,
                response_cache_dir=self.config.llm_intent_response_cache_dir,
                audit_path=self.config.llm_intent_audit_path,
                progress_path=self.config.llm_intent_progress_path,
                retry_backoff_seconds=(
                    self.config.llm_intent_retry_backoff_seconds
                ),
                replay_reference_audit_path=(
                    self.config.llm_intent_replay_reference_audit_path
                ),
                model_by_tier={
                    key: value
                    for key, value in {
                        "low": self.config.llm_intent_low_model,
                        "standard": self.config.llm_intent_standard_model,
                        "high": self.config.llm_intent_high_model,
                    }.items()
                    if value
                },
            )
        raise ValueError(f"Unsupported llm_intent_provider: {self.config.llm_intent_provider}")

    def _init_state(self) -> SocietyState:
        agents: dict[str, AgentState] = {}
        profile_quantiles = build_population_stratification_quantiles(
            self.config.population_size,
            self.rng,
        )
        for index in range(self.config.population_size):
            agent_id = f"agent_{index:04d}"
            hunger = self.rng.uniform("body", 0.05, 0.65, kind="init_hunger")
            fatigue = self.rng.uniform("body", 0.05, 0.45, kind="init_fatigue")
            stress = self.rng.uniform("body", 0.05, 0.45, kind="init_stress")
            curiosity = self.rng.uniform("body", 0.2, 0.9, kind="init_curiosity")
            food = self.rng.uniform("body", 0.2, 1.0, kind="init_food")
            attention = self.rng.uniform("body", 0.35, 1.0, kind="init_attention")
            agents[agent_id] = AgentState(
                id=agent_id,
                profile=build_agent_profile(
                    agent_id,
                    index,
                    self.rng,
                    technical_role_quantile=profile_quantiles["technical"][index],
                    innovation_role_quantile=profile_quantiles["innovation"][index],
                    activity_quantile=profile_quantiles["activity"][index],
                    cognitive_quantile=profile_quantiles["cognitive"][index],
                    technical_expert_fraction=self.config.technical_expert_fraction,
                    technical_practitioner_fraction=(
                        self.config.technical_practitioner_fraction
                    ),
                    innovation_originator_fraction=(
                        self.config.innovation_originator_fraction
                    ),
                    innovation_early_builder_fraction=(
                        self.config.innovation_early_builder_fraction
                    ),
                    innovation_pragmatic_adapter_fraction=(
                        self.config.innovation_pragmatic_adapter_fraction
                    ),
                ),
                body=BodyState(
                    hunger=hunger,
                    sleep_pressure=self.rng.uniform("body", 0.05, 0.5, kind="init_sleep"),
                    fatigue=fatigue,
                    stress=stress,
                    health_reserve=self.rng.uniform("body", 0.65, 1.0, kind="init_health"),
                    physiological_energy=self.rng.uniform("body", 0.35, 1.0, kind="init_energy"),
                ),
                material_resources=MaterialResources(food_or_budget_token=food, artifact_access=0.0),
                action_budgets=ActionBudgets(attention_budget=attention, time_budget=1.0),
            )
            agents[agent_id].affect.curiosity = curiosity
            agents[agent_id].cognition = CognitiveState(
                uncertainty=self.rng.uniform("body", 0.2, 0.8, kind="init_uncertainty")
            )
        state = SocietyState(agents=agents)
        initialize_communities(state)
        for agent in state.agents.values():
            assign_agent_communities(agent, state)
            agent.search_queries = _profile_search_queries(agent)
        return state

    def inject_artifact(self, artifact: Artifact, *, log_release_event: bool = True) -> None:
        apply_substrate_communities(self.state, artifact)
        self.state.artifacts[artifact.id] = artifact
        draw_start = len(self.rng.draws)
        agent_ids = sorted(self.state.agents)
        fraction = max(0.0, min(1.0, self.config.artifact_initial_access_fraction))
        initial_access_count = len(agent_ids) if fraction >= 1.0 else max(1, round(len(agent_ids) * fraction))
        scored_agents = [
            (self.rng.random("event", kind=f"artifact_initial_access:{artifact.id}"), agent_id)
            for agent_id in agent_ids
        ]
        selected_agent_ids = tuple(
            agent_id for _, agent_id in sorted(scored_agents, reverse=True)[:initial_access_count]
        )
        selected_set = set(selected_agent_ids)
        remaining_agent_ids = [agent_id for agent_id in agent_ids if agent_id not in selected_set]
        holdout_fraction = max(0.0, min(1.0, self.config.network_holdout_fraction))
        peer_control_fraction = max(0.0, min(1.0, self.config.peer_signal_control_fraction))
        if self.config.network_enabled:
            holdout_agent_ids = self._balanced_holdout_assignment(
                remaining_agent_ids,
                holdout_fraction=holdout_fraction,
                artifact_id=artifact.id,
            )
            holdout_set = set(holdout_agent_ids)
            eligible_agent_ids = tuple(
                agent_id for agent_id in remaining_agent_ids if agent_id not in holdout_set
            )
            access_only_agent_ids = self._balanced_holdout_assignment(
                list(eligible_agent_ids),
                holdout_fraction=peer_control_fraction,
                artifact_id=f"{artifact.id}:peer_signal_control",
            )
            access_only_set = set(access_only_agent_ids)
            peer_signal_agent_ids = tuple(
                agent_id for agent_id in eligible_agent_ids if agent_id not in access_only_set
            )
        else:
            holdout_agent_ids = ()
            eligible_agent_ids = ()
            access_only_agent_ids = ()
            peer_signal_agent_ids = ()
        for agent_id in selected_agent_ids:
            grant_artifact_access(
                self.state.agents[agent_id],
                artifact.id,
                source="release",
                tick=self.state.tick,
                exposed_by=artifact.provider_id,
                distance=0,
            )
            self.state.agents[agent_id].artifact_exposure_groups[artifact.id] = "release_seed"
            self.state.agents[agent_id].artifact_peer_signal_groups[artifact.id] = "release_seed"
        for agent_id in eligible_agent_ids:
            self.state.agents[agent_id].artifact_exposure_groups[artifact.id] = "network_eligible"
        for agent_id in peer_signal_agent_ids:
            self.state.agents[agent_id].artifact_peer_signal_groups[artifact.id] = "peer_signal_eligible"
        for agent_id in access_only_agent_ids:
            self.state.agents[agent_id].artifact_peer_signal_groups[artifact.id] = "access_only_control"
        for agent_id in holdout_agent_ids:
            self.state.agents[agent_id].artifact_exposure_groups[artifact.id] = "network_holdout"
            self.state.agents[agent_id].artifact_peer_signal_groups[artifact.id] = "network_holdout"
        if not self.config.network_enabled:
            for agent_id in remaining_agent_ids:
                self.state.agents[agent_id].artifact_exposure_groups[artifact.id] = "network_disabled"
                self.state.agents[agent_id].artifact_peer_signal_groups[artifact.id] = "network_disabled"
        self._hash_cache_dirty = True
        if log_release_event:
            self.state.event_log.append(
                ExternalEvent(
                    event_id=f"artifact_release_{self.state.tick:06d}_{artifact.id}",
                    tick=self.state.tick,
                    kind=EventKind.NEW_ARTIFACT_RELEASE,
                    actor_id=artifact.provider_id,
                    artifact_id=artifact.id,
                    typed_payload={
                        "artifact_kind": artifact.artifact_kind,
                        "provider_id": artifact.provider_id,
                        "public_claims": artifact.public_claims,
                        "initial_access_count": initial_access_count,
                        "initial_access_fraction": fraction,
                        "initial_access_agent_ids": selected_agent_ids,
                        "network_eligible_count": len(eligible_agent_ids),
                        "network_holdout_count": len(holdout_agent_ids),
                        "network_holdout_fraction": holdout_fraction,
                        "network_holdout_assignment": "blocked_covariate_randomization_v1"
                        if self.config.network_enabled else "network_disabled",
                        "peer_signal_eligible_count": len(peer_signal_agent_ids),
                        "access_only_control_count": len(access_only_agent_ids),
                        "peer_signal_control_fraction": peer_control_fraction,
                        "peer_signal_assignment": "blocked_covariate_randomization_v1"
                        if self.config.network_enabled else "network_disabled",
                    },
                    random_draw_refs=tuple(self.rng.refs_since(draw_start)),
                    public_visibility="public",
                    company_visible_flag=artifact.provider_id is not None,
                    detector_visible_only_after_run=True,
                )
            )

    def _balanced_holdout_assignment(
        self,
        agent_ids: list[str],
        *,
        holdout_fraction: float,
        artifact_id: str,
    ) -> tuple[str, ...]:
        if not agent_ids or holdout_fraction <= 0:
            return ()
        if holdout_fraction >= 1:
            return tuple(agent_ids)
        target_holdout_count = round(len(agent_ids) * holdout_fraction)
        if target_holdout_count <= 0:
            return ()
        target_means = self._balance_means(agent_ids)
        holdout_ids: list[str] = []
        remaining_ids = list(agent_ids)
        for step in range(target_holdout_count):
            scored: list[tuple[float, float, str]] = []
            for agent_id in remaining_ids:
                candidate_ids = holdout_ids + [agent_id]
                loss = self._balance_loss(candidate_ids, target_means)
                tie_breaker = self.rng.random("event", kind=f"artifact_balanced_holdout:{artifact_id}:{step}")
                scored.append((loss, tie_breaker, agent_id))
            _, _, selected_id = min(scored, key=lambda item: (item[0], item[1], item[2]))
            holdout_ids.append(selected_id)
            remaining_ids.remove(selected_id)
        return tuple(sorted(holdout_ids))

    def _agent_balance_vector(self, agent_id: str) -> tuple[float, ...]:
        agent = self.state.agents[agent_id]
        return (
            agent.affect.curiosity,
            agent.action_budgets.attention_budget,
            agent.body.hunger,
            agent.body.fatigue,
            agent.body.stress,
            agent.cognition.uncertainty,
            agent.body.physiological_energy,
            agent.material_resources.food_or_budget_token,
            agent.profile.technical_skill,
            agent.profile.domain_need,
            agent.profile.budget_sensitivity,
            agent.profile.risk_tolerance,
            agent.profile.peer_susceptibility,
        )

    def _balance_means(self, agent_ids: list[str]) -> tuple[float, ...]:
        vectors = [self._agent_balance_vector(agent_id) for agent_id in agent_ids]
        return tuple(
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(len(vectors[0]))
        )

    def _balance_loss(self, candidate_ids: list[str], target_means: tuple[float, ...]) -> float:
        candidate_means = self._balance_means(candidate_ids)
        return max(
            abs(candidate_means[index] - target_means[index])
            for index in range(len(target_means))
        )

    def step(self) -> None:
        validate_type_separation(self.state)
        if self.config.state_hash_mode == "merkle":
            # External scenario injectors mutate state directly before ticks. A once-per-tick
            # rebuild keeps the commitment correct while avoiding per-action full hashing.
            self._rebuild_hash_cache()
        concurrent_refresh = (
            self.config.llm_intent_enabled
            and self.config.llm_intent_max_concurrency > 1
        )
        intent_executor = (
            ThreadPoolExecutor(
                max_workers=self.config.llm_intent_max_concurrency,
                thread_name_prefix="subjective-intent",
            )
            if concurrent_refresh
            else None
        )
        pending_refreshes = (
            self._submit_tick_intent_refreshes(intent_executor)
            if intent_executor is not None
            else {}
        )
        try:
            for agent_id in sorted(self.state.agents):
                agent = self.state.agents[agent_id]
                drift_body(
                    agent,
                    coupling_enabled=self.config.body_affect_coupling_enabled,
                )
                if self.config.llm_intent_enabled:
                    if concurrent_refresh:
                        self._consume_tick_intent_refreshes(
                            agent,
                            pending_refreshes,
                        )
                    else:
                        self._refresh_llm_intents(agent)
                observation = observe_local_slice(agent, self.state)
                if self.config.network_enabled:
                    visible_network_authors = update_network_exposures_from_observation(
                        state=self.state,
                        agent=agent,
                        observation=observation,
                        peer_signal_enabled=self.config.peer_signal_utility_enabled,
                    )
                else:
                    visible_network_authors = ()
                agent.cognition.working_memory_items = list(
                    observation.visible_content_ids[:8]
                )
                self._update_agent_hash(agent_id)
                draw_start = len(self.rng.draws)
                action = choose_action(agent, self.rng, self.state.artifacts)
                action = self._attach_observed_content_target(action, observation)
                action = replace(
                    action,
                    typed_payload={
                        **action.typed_payload,
                        "tick": self.state.tick,
                    },
                )
                experience_generated = False
                if (
                    self.config.typed_product_experience_enabled
                    and action.kind
                    in {
                        ActionKind.TRY_ARTIFACT_ON_TASK,
                        ActionKind.REUSE_ARTIFACT,
                    }
                    and action.artifact_id
                    and action.artifact_id in self.state.artifacts
                ):
                    packet_factory = (
                        self.product_experience_packet_provider
                        or simulate_product_experience
                    )
                    packet = packet_factory(
                        agent=agent,
                        artifact=self.state.artifacts[action.artifact_id],
                        rng=self.rng,
                        tick=self.state.tick,
                        action_kind=action.kind,
                    )
                    self._register_product_experience(
                        agent_id=agent.id,
                        packet=packet,
                    )
                    action = replace(
                        action,
                        typed_payload={
                            **action.typed_payload,
                            "product_experience": packet,
                        },
                    )
                    experience_generated = True
                validate_action(action)
                pre_hash = self.state_hash()
                deltas = apply_action_effect(
                    agent,
                    action,
                    artifacts=self.state.artifacts,
                    coupling_enabled=self.config.body_affect_coupling_enabled,
                )
                if (
                    experience_generated
                    and action.artifact_id
                    and self.config.llm_intent_enabled
                ):
                    self._refresh_llm_intent_for_artifact(
                        agent,
                        action.artifact_id,
                        reason="new_product_experience",
                    )
                content_id = self._maybe_create_content(action)
                self._update_agent_hash(agent_id)
                if action.content_id and action.content_id in self.state.content:
                    self._update_content_hash(action.content_id)
                if content_id:
                    self._update_content_hash(content_id)
                post_hash = self.state_hash()
                event = make_action_event(
                    event_id=f"evt_{self.state.tick:06d}_{agent_id}",
                    tick=self.state.tick,
                    action=action,
                    content_id=content_id,
                    pre_state_hash=pre_hash,
                    post_state_hash=post_hash,
                    observation_refs=(
                        observation.visible_content_ids
                        + observation.dm_ids
                        + observation.search_result_ids
                    ),
                    random_draw_refs=tuple(self.rng.refs_since(draw_start)),
                    typed_payload={
                        "body_deltas": deltas,
                        "action_payload": canonicalize(action.typed_payload),
                        "llm_intent": (
                            canonicalize(
                                agent.latest_llm_intents.get(
                                    action.artifact_id
                                )
                            )
                            if action.artifact_id
                            else None
                        ),
                        "profile_ref": agent.profile.persona_label,
                        "artifact_payment_state": (
                            agent.artifact_payment_state.get(
                                action.artifact_id
                            )
                            if action.artifact_id
                            else None
                        ),
                        "artifact_exposure_source": (
                            agent.artifact_exposure_sources.get(
                                action.artifact_id
                            )
                            if action.artifact_id
                            else None
                        ),
                        "artifact_exposure_distance": (
                            agent.artifact_exposure_distance.get(
                                action.artifact_id
                            )
                            if action.artifact_id
                            else None
                        ),
                        "visible_network_authors": visible_network_authors,
                        "parent_content_id": action.content_id,
                    },
                )
                self.state.event_log.append(event)
                self._append_agent_memory(
                    agent,
                    self._memory_from_event(
                        event.event_id,
                        action.kind.value,
                        self.state.tick,
                    ),
                )
                self._update_agent_hash(agent_id)
                if experience_generated and action.artifact_id:
                    self._record_upgrade_suggestions(
                        agent=agent,
                        artifact=self.state.artifacts[action.artifact_id],
                        packet=action.typed_payload["product_experience"],
                        parent_event_id=event.event_id,
                        observation_refs=(
                            observation.visible_content_ids
                            + observation.dm_ids
                            + observation.search_result_ids
                        ),
                    )
        finally:
            if intent_executor is not None:
                intent_executor.shutdown(wait=True, cancel_futures=True)
        # Detectors may be run for monitoring, but output is intentionally ignored.
        if self.config.detectors_enabled_during_run:
            _ = self.detectors.run_all(self.state)
        emerged_ids = update_emergent_institutions(self.state)
        for institution_id in emerged_ids:
            self._apply_institution_feedback(institution_id)
            self._update_institution_hash(institution_id)
        self.state.tick += 1

    def _memory_from_event(self, event_id: str, summary: str, tick: int):
        from .schemas import MemoryItem

        return MemoryItem(
            event_ref=event_id,
            semantic_summary=summary,
            source="typed_event",
            timestamp=tick,
            last_retrieved=tick,
            uncertainty=0.1,
        )

    def _append_agent_memory(self, agent: AgentState, item) -> None:
        agent.memory.append(item)
        if self.config.state_hash_mode != "merkle":
            return
        item_hashes = self._agent_memory_item_hashes.get(agent.id)
        if item_hashes is None:
            return
        item_hashes.append(self._memory_item_hash(item))
        self._agent_memory_hashes[agent.id] = self._sequence_root(item_hashes)

    def _attach_observed_content_target(self, action, observation):
        from .schemas import ActionKind

        targetable_actions = {
            ActionKind.REPLY,
            ActionKind.REPOST,
            ActionKind.ASK_QUESTION,
            ActionKind.ASK_FOR_EXPLANATION,
            ActionKind.VERIFY,
            ActionKind.COMPARE_SOURCES,
            ActionKind.CHALLENGE_CLAIM,
            ActionKind.HELP_PEER,
            ActionKind.WARN_PEER,
            ActionKind.MEDIATE_CONFLICT,
            ActionKind.REDUCE_TRUST,
        }
        if action.kind not in targetable_actions:
            return action
        if not observation.visible_content_ids:
            return action
        weights = [1.0 / (index + 1) for index, _ in enumerate(observation.visible_content_ids)]
        index = self.rng.choice_index("feed", weights, kind=f"{action.kind.value}_target")
        parent_id = observation.visible_content_ids[index]
        parent = self.state.content.get(parent_id)
        target_ids = (parent.author_id,) if parent else ()
        return replace(action, content_id=parent_id, channel_id="public_forum", target_ids=target_ids)

    def _refresh_llm_intents(self, agent: AgentState) -> None:
        for artifact_id in sorted(agent.artifact_access):
            self._refresh_llm_intent_for_artifact(agent, artifact_id, reason="tick_refresh")

    def _submit_tick_intent_refreshes(
        self,
        executor: ThreadPoolExecutor,
    ) -> dict[tuple[str, str], Future]:
        pending: dict[tuple[str, str], Future] = {}
        for agent_id in sorted(self.state.agents):
            prepared_agent = deepcopy(self.state.agents[agent_id])
            drift_body(
                prepared_agent,
                coupling_enabled=self.config.body_affect_coupling_enabled,
            )
            for artifact_id in sorted(prepared_agent.artifact_access):
                if not self._llm_intent_refresh_needed(
                    prepared_agent,
                    artifact_id,
                    reason="tick_refresh",
                ):
                    continue
                artifact = self.state.artifacts.get(artifact_id)
                if artifact is None:
                    continue
                pending[(agent_id, artifact_id)] = executor.submit(
                    self.intent_provider.generate,
                    agent=prepared_agent,
                    artifact=artifact,
                    tick=self.state.tick,
                )
        return pending

    def _consume_tick_intent_refreshes(
        self,
        agent: AgentState,
        pending: dict[tuple[str, str], Future],
    ) -> None:
        expected_artifact_ids = tuple(
            artifact_id
            for artifact_id in sorted(agent.artifact_access)
            if self._llm_intent_refresh_needed(
                agent,
                artifact_id,
                reason="tick_refresh",
            )
            and artifact_id in self.state.artifacts
        )
        submitted_artifact_ids = tuple(
            artifact_id
            for submitted_agent_id, artifact_id in pending
            if submitted_agent_id == agent.id
        )
        if submitted_artifact_ids != expected_artifact_ids:
            raise RuntimeError(
                "concurrent_intent_refresh_schedule_mismatch:"
                f"agent={agent.id}:submitted={submitted_artifact_ids}:"
                f"expected={expected_artifact_ids}"
            )
        for artifact_id in expected_artifact_ids:
            future = pending[(agent.id, artifact_id)]
            record = future.result()
            self._store_generated_llm_intent(
                agent=agent,
                artifact_id=artifact_id,
                record=record,
                validate_input_commitment=True,
            )

    def _refresh_llm_intent_for_artifact(self, agent: AgentState, artifact_id: str, *, reason: str = "manual") -> None:
        artifact = self.state.artifacts.get(artifact_id)
        if artifact is None:
            return
        previous = agent.latest_llm_intents.get(artifact_id)
        if previous and not self._llm_intent_refresh_needed(agent, artifact_id, reason=reason):
            return
        record = self.intent_provider.generate(
            agent=agent,
            artifact=artifact,
            tick=self.state.tick,
        )
        self._store_generated_llm_intent(
            agent=agent,
            artifact_id=artifact_id,
            record=record,
        )

    def _store_generated_llm_intent(
        self,
        *,
        agent: AgentState,
        artifact_id: str,
        record,
        validate_input_commitment: bool = False,
    ) -> None:
        artifact = self.state.artifacts.get(artifact_id)
        if artifact is None:
            raise RuntimeError(
                f"intent_record_artifact_missing:{artifact_id}"
            )
        expected_experience = agent.latest_artifact_experience.get(artifact_id)
        expected_experience_ref = (
            expected_experience.experience_id if expected_experience else None
        )
        if (
            record.artifact_id != artifact_id
            or record.tick != self.state.tick
            or record.profile_ref != agent.profile.persona_label
            or record.experience_ref != expected_experience_ref
        ):
            raise RuntimeError(
                "intent_record_identity_mismatch:"
                f"agent={agent.id}:artifact={artifact_id}"
            )
        if validate_input_commitment:
            expected_input_hash = intent_input_evidence_hash(
                agent=agent,
                artifact=artifact,
                tick=self.state.tick,
            )
            if record.input_evidence_hash != expected_input_hash:
                raise RuntimeError(
                    "concurrent_intent_input_commitment_mismatch:"
                    f"agent={agent.id}:artifact={artifact_id}"
                )
        signal = _intent_signal_signature(agent, artifact_id)
        stored_record = replace(
            record,
            reason_code=f"{record.reason_code[:48]}:sig_{signal}",
            blocked_reason=record.blocked_reason,
        )
        agent.latest_llm_intents[artifact_id] = stored_record
        agent.llm_intent_session_history.append(stored_record)

    def _llm_intent_refresh_needed(self, agent: AgentState, artifact_id: str, *, reason: str) -> bool:
        if reason in {"new_product_experience", "payment_feedback"}:
            return True
        previous = agent.latest_llm_intents.get(artifact_id)
        if previous is None:
            return True
        if self.state.tick - previous.tick >= self.config.llm_intent_cache_ttl_ticks:
            return True
        current_signal = _intent_signal_signature(agent, artifact_id)
        previous_signal = previous.reason_code.split(":sig_", 1)[-1] if ":sig_" in previous.reason_code else None
        return previous_signal != current_signal

    def _maybe_create_content(self, action) -> str | None:
        from .schemas import ActionKind

        content_kinds = {
            ActionKind.POST: ContentKind.POST,
            ActionKind.REPLY: ContentKind.REPLY,
            ActionKind.REPOST: ContentKind.REPOST,
            ActionKind.DM: ContentKind.DM,
            ActionKind.SHARE_FAILURE: ContentKind.FAILURE_REPORT,
            ActionKind.SHARE_SUCCESS: ContentKind.SUCCESS_REPORT,
            ActionKind.SUGGEST_UPGRADE: ContentKind.UPGRADE_SUGGESTION,
            ActionKind.WRITE_TUTORIAL: ContentKind.TUTORIAL,
            ActionKind.CHALLENGE_CLAIM: ContentKind.CALLOUT,
            ActionKind.ASK_QUESTION: ContentKind.QUESTION,
            ActionKind.ASK_FOR_EXPLANATION: ContentKind.QUESTION,
            ActionKind.VERIFY: ContentKind.POST,
            ActionKind.COMPARE_SOURCES: ContentKind.POST,
            ActionKind.HELP_PEER: ContentKind.REPLY,
            ActionKind.WARN_PEER: ContentKind.CALLOUT,
            ActionKind.MEDIATE_CONFLICT: ContentKind.REPLY,
            ActionKind.REDUCE_TRUST: ContentKind.CALLOUT,
        }
        if action.kind not in content_kinds:
            return None
        parent = self.state.content.get(action.content_id) if action.content_id else None
        if parent and action.kind in {ActionKind.REPLY, ActionKind.REPOST}:
            parent.engagement_counts[action.kind.value] = parent.engagement_counts.get(action.kind.value, 0) + 1
        content_id = f"content_{self.state.tick:06d}_{action.actor_id}_{len(self.state.content):06d}"
        agent = self.state.agents[action.actor_id]
        channel = action.channel_id or ("dm" if action.kind == ActionKind.DM else preferred_public_channel(agent, self.state))
        recipients = tuple(dict.fromkeys(action.target_ids)) if channel == "dm" else ()
        visibility = (
            f"dm:{recipients[0]}"
            if channel == "dm" and len(recipients) == 1
            else ("private" if channel == "dm" else "public")
        )
        text = self._content_text_for_action(action, parent)
        artifact_refs = (action.artifact_id,) if action.artifact_id else (parent.artifact_refs if parent else ())
        topic_vector = self._topic_vector_for_action(action, artifact_refs)
        item = ContentItem(
            id=content_id,
            author_id=action.actor_id,
            channel=channel,
            kind=content_kinds[action.kind],
            created_at=self.state.tick,
            parent_id=parent.id if parent else None,
            artifact_refs=artifact_refs,
            text_surface=text,
            visibility_scope=visibility,
            recipient_ids=recipients,
            indexed_at=self.state.tick + 1,
            topic_vector=topic_vector,
        )
        self.state.content[content_id] = item
        self._index_content(item)
        return content_id

    def _content_text_for_action(self, action: Action, parent: ContentItem | None) -> str:
        packet = action.typed_payload.get("product_experience")
        parts = [action.actor_id, action.kind.value]
        if action.artifact_id:
            parts.append(action.artifact_id)
        if isinstance(packet, ProductExperiencePacket):
            parts.extend([
                f"task:{packet.task_type}",
                f"success:{packet.objective_success:.2f}",
                f"friction:{packet.friction:.2f}",
            ])
            if packet.blocked_stage:
                parts.append(f"blocked:{packet.blocked_stage}")
            if packet.failure_event:
                parts.append(f"failure:{packet.failure_event}")
        if parent:
            parts.append(f"parent:{parent.id}")
        return " ".join(parts)

    def _topic_vector_for_action(self, action: Action, artifact_refs: tuple[str, ...]) -> dict[str, float]:
        topic = {"social": 0.35}
        if artifact_refs:
            topic.update({"artifact": 1.0, artifact_refs[0]: 0.8})
        packet = action.typed_payload.get("product_experience")
        if isinstance(packet, ProductExperiencePacket):
            topic[packet.task_type] = 0.9
            if packet.blocked_stage:
                topic[packet.blocked_stage] = 0.7
            if packet.failure_event:
                topic[packet.failure_event] = 0.7
        if action.kind in {ActionKind.CHALLENGE_CLAIM, ActionKind.WARN_PEER, ActionKind.SHARE_FAILURE}:
            topic["controversy"] = 0.65
        if action.kind in {ActionKind.WRITE_TUTORIAL, ActionKind.HELP_PEER}:
            topic["help"] = 0.65
        return topic

    def _index_content(self, item: ContentItem) -> None:
        terms = {
            token.strip(".,:;()[]{}").lower()
            for token in item.text_surface.split()
            if token.strip(".,:;()[]{}")
        }
        terms.update(item.artifact_refs)
        terms.update(item.claim_refs)
        terms.update(item.topic_vector)
        self.state.search_index_terms[item.id] = tuple(sorted(terms))
        self._update_search_index_hash(item.id)

    def _apply_institution_feedback(self, institution_id: str) -> None:
        institution = self.state.emergent_institutions[institution_id]
        if not institution.behavioral_feedback:
            return
        role_actor_ids = set(institution.role_actor_ids)
        for agent in self.state.agents.values():
            multiplier = 1.25 if agent.id in role_actor_ids else 1.0
            for key, value in institution.behavioral_feedback.items():
                current = agent.cognition.institution_feedback.get(key, 0.0)
                agent.cognition.institution_feedback[key] = max(current, min(1.0, value * multiplier))
            self._update_agent_hash(agent.id)

    def _record_upgrade_suggestions(
        self,
        *,
        agent: AgentState,
        artifact: Artifact,
        packet: ProductExperiencePacket,
        parent_event_id: str,
        observation_refs: tuple[str, ...],
    ) -> tuple[UpgradeSuggestion, ...]:
        suggestions = suggest_upgrades_from_experience(
            agent=agent,
            artifact=artifact,
            packet=packet,
            evidence_only=self.config.evidence_only_upgrade_suggestions,
        )
        recorded: list[UpgradeSuggestion] = []
        intent = agent.latest_llm_intents.get(artifact.id)
        intent_is_grounded = bool(intent and intent.experience_ref == packet.experience_id)
        subjective_feedback = (
            intent.product_feedback if intent and intent_is_grounded else ""
        )
        subjective_feedback_source = intent.source if intent and intent_is_grounded else ""
        live_intent_ref = (
            intent.intent_id
            if intent
            and intent_is_grounded
            and intent.source.startswith("openai_live_v17:")
            and "fallback" not in intent.source
            else None
        )
        improvements = (
            intent.suggested_improvements if intent and intent_is_grounded else ()
        )
        feature_ideas = intent.feature_ideas if intent and intent_is_grounded else ()
        llm_suggestions = self._llm_grounded_upgrade_suggestions(
            agent=agent,
            artifact=artifact,
            packet=packet,
            intent=intent if intent_is_grounded else None,
            existing_themes=tuple(item.theme for item in suggestions),
        )
        suggestions = (*suggestions, *llm_suggestions)
        improvement_by_theme = (
            dict(zip(intent.improvement_themes, improvements, strict=True))
            if intent
            and intent_is_grounded
            and len(intent.improvement_themes) == len(improvements)
            else {}
        )
        feature_by_theme = (
            dict(zip(intent.feature_themes, feature_ideas, strict=True))
            if intent
            and intent_is_grounded
            and len(intent.feature_themes) == len(feature_ideas)
            else {}
        )
        feature_suggestion_index = 0
        for suggestion_index, raw_suggestion in enumerate(suggestions):
            publish = self._should_publish_upgrade_suggestion(agent, raw_suggestion)
            suggested_improvement = (
                raw_suggestion.llm_suggested_improvement
                or improvement_by_theme.get(raw_suggestion.theme, "")
                or improvements[suggestion_index % len(improvements)]
                if improvements
                else ""
            )
            feature_idea = (
                raw_suggestion.llm_feature_idea
                or feature_by_theme.get(raw_suggestion.theme, "")
            )
            if raw_suggestion.contribution_mode in {
                "feature_proposal",
                "technical_feature_proposal",
            } and not feature_idea:
                if feature_ideas:
                    feature_idea = feature_ideas[
                        feature_suggestion_index % len(feature_ideas)
                    ]
                feature_suggestion_index += 1
            suggestion = replace(
                raw_suggestion,
                visibility_scope="public" if publish else "candidate",
                llm_intent_ref=live_intent_ref,
                subjective_feedback_source=subjective_feedback_source,
                subjective_feedback=subjective_feedback,
                llm_suggested_improvement=suggested_improvement,
                llm_feature_idea=feature_idea,
            )
            pre_hash = self.state_hash()
            history = agent.artifact_upgrade_suggestions.setdefault(artifact.id, [])
            if suggestion.suggestion_id not in {item.suggestion_id for item in history}:
                history.append(suggestion)
                if len(history) > 12:
                    del history[:-12]
            self.state.upgrade_suggestions[suggestion.suggestion_id] = suggestion
            content_id = self._create_upgrade_suggestion_content(suggestion) if publish else None
            self._update_agent_hash(agent.id)
            self._update_suggestion_hash(suggestion.suggestion_id)
            if content_id:
                self._update_content_hash(content_id)
            post_hash = self.state_hash()
            event_id = f"upgrade_suggestion_{self.state.tick:06d}_{agent.id}_{suggestion.suggestion_id[-8:]}"
            self.state.event_log.append(
                ExternalEvent(
                    event_id=event_id,
                    tick=self.state.tick,
                    kind=EventKind.ACTION,
                    actor_id=agent.id,
                    action_type=ActionKind.SUGGEST_UPGRADE.value,
                    channel_id="public_forum" if publish else None,
                    artifact_id=artifact.id,
                    content_id=content_id,
                    pre_state_hash=pre_hash,
                    post_state_hash=post_hash,
                    observation_refs=observation_refs,
                    memory_refs=(parent_event_id,),
                    random_draw_refs=(),
                    typed_payload={
                        "upgrade_suggestion": canonicalize(suggestion),
                        "source_experience_ref": packet.experience_id,
                        "parent_event_id": parent_event_id,
                    },
                    public_visibility="public" if publish else "private",
                    company_visible_flag=bool(publish and artifact.provider_id),
                    detector_visible_only_after_run=True,
                )
            )
            self._append_agent_memory(
                agent,
                self._memory_from_event(event_id, f"suggest_upgrade:{suggestion.theme}", self.state.tick),
            )
            self._update_agent_hash(agent.id)
            recorded.append(suggestion)
        return tuple(recorded)

    def _llm_grounded_upgrade_suggestions(
        self,
        *,
        agent: AgentState,
        artifact: Artifact,
        packet: ProductExperiencePacket,
        intent,
        existing_themes: tuple[str, ...],
    ) -> tuple[UpgradeSuggestion, ...]:
        if intent is None:
            return ()
        improvement_pairs = tuple(
            zip(
                intent.improvement_themes,
                intent.suggested_improvements,
            )
        )
        feature_pairs = tuple(
            zip(
                intent.feature_themes,
                intent.feature_ideas,
            )
        )
        seen = set(existing_themes)
        profile = agent.profile
        technical_depth = clamp01(
            0.52 * profile.technical_skill
            + 0.28 * profile.cognitive_capacity
            + 0.20 * packet.diagnostic_clarity
        )
        novelty_score = clamp01(
            0.48 * profile.creative_capacity
            + 0.24 * profile.novelty_seeking
            + 0.18 * profile.strategic_boldness
            + 0.10 * profile.domain_need
        )
        base_priority = clamp01(
            0.34 * intent.confidence
            + 0.20 * profile.domain_need
            + 0.16 * profile.opinion_leadership
            + 0.15 * profile.strategic_boldness
            + 0.15 * max(technical_depth, novelty_score)
        )
        evidence_refs = tuple(
            dict.fromkeys(
                (
                    packet.experience_id,
                    *packet.objective_observation_refs,
                    f"observation_set_hash:{packet.objective_observation_set_hash}",
                    (
                        "observation_manifest_hash:"
                        f"{packet.objective_observation_manifest_hash}"
                    ),
                    f"llm_input_hash:{intent.input_evidence_hash}",
                    f"llm_output_hash:{intent.output_evidence_hash}",
                )
            )
        )
        suggestions: list[UpgradeSuggestion] = []
        for kind, pairs in (
            ("improvement", improvement_pairs),
            ("feature", feature_pairs),
        ):
            for theme, text in pairs:
                if theme in seen:
                    continue
                seen.add(theme)
                contribution_mode = "user_feedback"
                if kind == "feature":
                    contribution_mode = (
                        "technical_feature_proposal"
                        if profile.technical_role
                        in {"professional_engineer", "technical_practitioner"}
                        and technical_depth >= 0.62
                        else "feature_proposal"
                    )
                elif profile.technical_role == "professional_engineer":
                    contribution_mode = "technical_review"
                suggestion_hash = stable_hash(
                    {
                        "intent_id": intent.intent_id,
                        "experience_id": packet.experience_id,
                        "theme": theme,
                        "kind": kind,
                        "text": text,
                    }
                )[:20]
                suggestions.append(
                    UpgradeSuggestion(
                        suggestion_id=f"upgrade_llm_{suggestion_hash}",
                        artifact_id=artifact.id,
                        author_id=agent.id,
                        tick=packet.tick,
                        source_experience_ref=packet.experience_id,
                        theme=theme,
                        priority=clamp01(
                            base_priority
                            * (1.0 - 0.08 * len(suggestions))
                        ),
                        confidence=clamp01(intent.confidence),
                        requested_change=text,
                        affected_tasks=(packet.task_type,),
                        evidence_refs=evidence_refs,
                        profile_ref=profile.persona_label,
                        llm_intent_ref=intent.intent_id,
                        llm_suggested_improvement=(
                            text if kind == "improvement" else ""
                        ),
                        llm_feature_idea=(
                            text if kind == "feature" else ""
                        ),
                        contribution_mode=contribution_mode,
                        technical_depth=technical_depth,
                        novelty_score=novelty_score,
                    )
                )
        return tuple(suggestions)

    def _should_publish_upgrade_suggestion(self, agent: AgentState, suggestion: UpgradeSuggestion) -> bool:
        profile = agent.profile
        publish_score = (
            0.46 * suggestion.priority
            + 0.20 * profile.posting_propensity
            + 0.14 * profile.daily_active_probability
            + 0.14 * profile.opinion_leadership
            + 0.12 * profile.domain_need
            + 0.10 * profile.social_activity
            + 0.08 * profile.community_trust
            + 0.16 * profile.strategic_boldness
            - 0.12 * profile.privacy_sensitivity
            - 0.10 * agent.body.fatigue
        )
        return publish_score >= 0.66 or (
            profile.strategic_boldness >= 0.88
            and suggestion.priority >= 0.92
            and profile.daily_active_probability >= 0.1
            and profile.posting_propensity >= 0.01
        )

    def _create_upgrade_suggestion_content(self, suggestion: UpgradeSuggestion) -> str:
        content_id = f"content_{self.state.tick:06d}_{suggestion.author_id}_{len(self.state.content):06d}"
        text = (
            f"{suggestion.author_id} suggest_upgrade {suggestion.artifact_id} "
            f"theme:{suggestion.theme} priority:{suggestion.priority:.3f} "
            f"confidence:{suggestion.confidence:.3f} request:{suggestion.requested_change}"
        )
        if suggestion.subjective_feedback:
            text += f" feedback:{suggestion.subjective_feedback}"
        if suggestion.llm_suggested_improvement:
            text += f" user_improvement:{suggestion.llm_suggested_improvement}"
        if suggestion.llm_feature_idea:
            text += f" feature_idea:{suggestion.llm_feature_idea}"
        item = ContentItem(
            id=content_id,
            author_id=suggestion.author_id,
            channel="public_forum",
            kind=ContentKind.UPGRADE_SUGGESTION,
            created_at=self.state.tick,
            artifact_refs=(suggestion.artifact_id,),
            evidence_refs=suggestion.evidence_refs,
            text_surface=text,
            visibility_scope="public",
            indexed_at=self.state.tick + 1,
            topic_vector={
                "artifact": 1.0,
                "upgrade_suggestion": 1.0,
                suggestion.theme: suggestion.priority,
            },
            controversy_proxy=1.0 - suggestion.confidence,
        )
        self.state.content[content_id] = item
        self._index_content(item)
        return content_id

    def run(self, ticks: int) -> SocietyState:
        for _ in range(ticks):
            self.step()
        return self.state

    def run_product_experience_panel(
        self,
        artifact_id: str,
        *,
        agent_ids: tuple[str, ...] | None = None,
        grant_missing_access: bool = True,
        packet_provider: Callable[..., ProductExperiencePacket] | None = None,
    ) -> int:
        artifact = self.state.artifacts.get(artifact_id)
        if artifact is None:
            raise ValueError(f"Unknown artifact_id: {artifact_id}")
        selected_agent_ids = agent_ids or tuple(sorted(self.state.agents))
        if (
            self.config.llm_intent_enabled
            and self.config.llm_intent_max_concurrency > 1
        ):
            return self._run_product_experience_panel_concurrently(
                artifact=artifact,
                selected_agent_ids=selected_agent_ids,
                grant_missing_access=grant_missing_access,
                packet_provider=packet_provider,
            )
        experienced_count = 0
        for agent_id in selected_agent_ids:
            agent = self.state.agents[agent_id]
            if grant_missing_access and artifact_id not in agent.artifact_access:
                grant_artifact_access(
                    agent,
                    artifact_id,
                    source="product_experience_panel",
                    tick=self.state.tick,
                    exposed_by=artifact.provider_id,
                    distance=0,
                )
                agent.artifact_exposure_groups.setdefault(artifact_id, "panel_trial")
                agent.artifact_peer_signal_groups.setdefault(artifact_id, "panel_trial")
            if artifact_id not in agent.artifact_access:
                continue
            draw_start = len(self.rng.draws)
            if packet_provider is None:
                packet = simulate_product_experience(
                    agent=agent,
                    artifact=artifact,
                    rng=self.rng,
                    tick=self.state.tick,
                    action_kind=ActionKind.TRY_ARTIFACT_ON_TASK,
                )
            else:
                packet = packet_provider(
                    agent=agent,
                    artifact=artifact,
                    rng=self.rng,
                    tick=self.state.tick,
                    action_kind=ActionKind.TRY_ARTIFACT_ON_TASK,
                )
            self._register_product_experience(
                agent_id=agent.id,
                packet=packet,
            )
            action = Action(
                actor_id=agent.id,
                kind=ActionKind.TRY_ARTIFACT_ON_TASK,
                artifact_id=artifact_id,
                typed_payload={
                    "tick": self.state.tick,
                    "product_experience": packet,
                    "panel_forced": True,
                },
            )
            validate_action(action)
            pre_hash = self.state_hash()
            deltas = apply_action_effect(
                agent,
                action,
                artifacts=self.state.artifacts,
                coupling_enabled=self.config.body_affect_coupling_enabled,
            )
            if self.config.llm_intent_enabled:
                self._refresh_llm_intent_for_artifact(agent, artifact_id, reason="new_product_experience")
            self._update_agent_hash(agent_id)
            post_hash = self.state_hash()
            event = make_action_event(
                event_id=f"panel_xp_{self.state.tick:06d}_{agent_id}_{artifact_id}",
                tick=self.state.tick,
                action=action,
                pre_state_hash=pre_hash,
                post_state_hash=post_hash,
                random_draw_refs=tuple(self.rng.refs_since(draw_start)),
                typed_payload={
                    "body_deltas": deltas,
                    "action_payload": canonicalize(action.typed_payload),
                    "llm_intent": canonicalize(agent.latest_llm_intents.get(artifact_id)),
                    "profile_ref": agent.profile.persona_label,
                    "panel_forced": True,
                },
            )
            self.state.event_log.append(event)
            self._append_agent_memory(
                agent,
                self._memory_from_event(event.event_id, action.kind.value, self.state.tick),
            )
            self._update_agent_hash(agent_id)
            self._record_upgrade_suggestions(
                agent=agent,
                artifact=artifact,
                packet=packet,
                parent_event_id=event.event_id,
                observation_refs=(),
            )
            experienced_count += 1
        return experienced_count

    def _run_product_experience_panel_concurrently(
        self,
        *,
        artifact: Artifact,
        selected_agent_ids: tuple[str, ...],
        grant_missing_access: bool,
        packet_provider: Callable[..., ProductExperiencePacket] | None,
    ) -> int:
        prepared: list[
            tuple[
                str,
                ProductExperiencePacket,
                Action,
                tuple[str, ...],
                Future,
            ]
        ] = []
        with ThreadPoolExecutor(
            max_workers=self.config.llm_intent_max_concurrency,
            thread_name_prefix="product-panel-intent",
        ) as executor:
            for agent_id in selected_agent_ids:
                prepared_agent = deepcopy(self.state.agents[agent_id])
                if (
                    grant_missing_access
                    and artifact.id not in prepared_agent.artifact_access
                ):
                    grant_artifact_access(
                        prepared_agent,
                        artifact.id,
                        source="product_experience_panel",
                        tick=self.state.tick,
                        exposed_by=artifact.provider_id,
                        distance=0,
                    )
                    prepared_agent.artifact_exposure_groups.setdefault(
                        artifact.id,
                        "panel_trial",
                    )
                    prepared_agent.artifact_peer_signal_groups.setdefault(
                        artifact.id,
                        "panel_trial",
                    )
                if artifact.id not in prepared_agent.artifact_access:
                    continue
                draw_start = len(self.rng.draws)
                packet_factory = packet_provider or simulate_product_experience
                packet = packet_factory(
                    agent=prepared_agent,
                    artifact=artifact,
                    rng=self.rng,
                    tick=self.state.tick,
                    action_kind=ActionKind.TRY_ARTIFACT_ON_TASK,
                )
                random_draw_refs = tuple(self.rng.refs_since(draw_start))
                action = Action(
                    actor_id=prepared_agent.id,
                    kind=ActionKind.TRY_ARTIFACT_ON_TASK,
                    artifact_id=artifact.id,
                    typed_payload={
                        "tick": self.state.tick,
                        "product_experience": packet,
                        "panel_forced": True,
                    },
                )
                validate_action(action)
                apply_action_effect(
                    prepared_agent,
                    action,
                    artifacts=self.state.artifacts,
                    coupling_enabled=self.config.body_affect_coupling_enabled,
                )
                future = executor.submit(
                    self.intent_provider.generate,
                    agent=prepared_agent,
                    artifact=artifact,
                    tick=self.state.tick,
                )
                prepared.append(
                    (
                        agent_id,
                        packet,
                        action,
                        random_draw_refs,
                        future,
                    )
                )

            for (
                agent_id,
                packet,
                action,
                random_draw_refs,
                future,
            ) in prepared:
                agent = self.state.agents[agent_id]
                self._register_product_experience(
                    agent_id=agent_id,
                    packet=packet,
                )
                if (
                    grant_missing_access
                    and artifact.id not in agent.artifact_access
                ):
                    grant_artifact_access(
                        agent,
                        artifact.id,
                        source="product_experience_panel",
                        tick=self.state.tick,
                        exposed_by=artifact.provider_id,
                        distance=0,
                    )
                    agent.artifact_exposure_groups.setdefault(
                        artifact.id,
                        "panel_trial",
                    )
                    agent.artifact_peer_signal_groups.setdefault(
                        artifact.id,
                        "panel_trial",
                    )
                if artifact.id not in agent.artifact_access:
                    raise RuntimeError(
                        f"prepared_panel_access_mismatch:{agent_id}"
                    )
                pre_hash = self.state_hash()
                deltas = apply_action_effect(
                    agent,
                    action,
                    artifacts=self.state.artifacts,
                    coupling_enabled=self.config.body_affect_coupling_enabled,
                )
                self._store_generated_llm_intent(
                    agent=agent,
                    artifact_id=artifact.id,
                    record=future.result(),
                    validate_input_commitment=True,
                )
                self._update_agent_hash(agent_id)
                post_hash = self.state_hash()
                event = make_action_event(
                    event_id=(
                        f"panel_xp_{self.state.tick:06d}_"
                        f"{agent_id}_{artifact.id}"
                    ),
                    tick=self.state.tick,
                    action=action,
                    pre_state_hash=pre_hash,
                    post_state_hash=post_hash,
                    random_draw_refs=random_draw_refs,
                    typed_payload={
                        "body_deltas": deltas,
                        "action_payload": canonicalize(
                            action.typed_payload
                        ),
                        "llm_intent": canonicalize(
                            agent.latest_llm_intents.get(artifact.id)
                        ),
                        "profile_ref": agent.profile.persona_label,
                        "panel_forced": True,
                    },
                )
                self.state.event_log.append(event)
                self._append_agent_memory(
                    agent,
                    self._memory_from_event(
                        event.event_id,
                        action.kind.value,
                        self.state.tick,
                    ),
                )
                self._update_agent_hash(agent_id)
                self._record_upgrade_suggestions(
                    agent=agent,
                    artifact=artifact,
                    packet=packet,
                    parent_event_id=event.event_id,
                    observation_refs=(),
                )
        return len(prepared)

    def _register_product_experience(
        self,
        *,
        agent_id: str,
        packet: ProductExperiencePacket,
    ) -> None:
        identity = (agent_id, packet.experience_id)
        packet_hash = stable_hash(packet)
        previous_hash = self._experience_packet_hashes.get(identity)
        if previous_hash is not None:
            conflict_kind = (
                "duplicate"
                if previous_hash == packet_hash
                else "conflicting"
            )
            raise RuntimeError(
                f"{conflict_kind}_product_experience_identity:"
                f"agent={agent_id}:experience={packet.experience_id}"
            )
        self._experience_packet_hashes[identity] = packet_hash

    def state_hash(self) -> str:
        if self.config.state_hash_mode == "canonical":
            return self.canonical_state_hash()
        if self.config.state_hash_mode != "merkle":
            raise ValueError(f"Unsupported state_hash_mode: {self.config.state_hash_mode}")
        if self._hash_cache_dirty:
            self._rebuild_hash_cache()
        hasher = hashlib.sha256()
        self._update_hash_field(hasher, "hash_mode", "merkle_v1")
        self._update_hash_field(hasher, "tick", str(self.state.tick))
        self._update_hash_field(hasher, "agents", self._ordered_leaf_root(self._agent_hashes.items()))
        self._update_hash_field(hasher, "artifacts", self._ordered_leaf_root(self._artifact_hashes.items()))
        self._update_hash_field(hasher, "content", self._ordered_leaf_root(self._content_hashes.items()))
        self._update_hash_field(hasher, "communities", self._communities_hash)
        self._update_hash_field(hasher, "search_index_terms", self._ordered_leaf_root(self._search_index_hashes.items()))
        self._update_hash_field(hasher, "upgrade_suggestions", self._ordered_leaf_root(self._suggestion_hashes.items()))
        self._update_hash_field(
            hasher,
            "emergent_institutions",
            self._ordered_leaf_root(self._institution_hashes.items()),
        )
        return hasher.hexdigest()

    def canonical_state_hash(self) -> str:
        return stable_hash(
            {
                "tick": self.state.tick,
                "agents": self.state.agents,
                "artifacts": self.state.artifacts,
                "content": self.state.content,
                "communities": self.state.communities,
                "search_index_terms": self.state.search_index_terms,
                "upgrade_suggestions": self.state.upgrade_suggestions,
                "emergent_institutions": self.state.emergent_institutions,
            }
        )

    def _rebuild_hash_cache(self) -> None:
        self._profile_hashes = {
            agent_id: stable_hash(agent.profile)
            for agent_id, agent in sorted(self.state.agents.items())
        }
        self._agent_memory_item_hashes = {
            agent_id: [self._memory_item_hash(item) for item in agent.memory]
            for agent_id, agent in sorted(self.state.agents.items())
        }
        self._agent_memory_hashes = {
            agent_id: self._sequence_root(item_hashes)
            for agent_id, item_hashes in sorted(self._agent_memory_item_hashes.items())
        }
        self._agent_hashes = {
            agent_id: self._agent_state_hash(agent)
            for agent_id, agent in sorted(self.state.agents.items())
        }
        self._artifact_hashes = {
            artifact_id: stable_hash(artifact)
            for artifact_id, artifact in sorted(self.state.artifacts.items())
        }
        self._content_hashes = {
            content_id: self._content_item_hash(content)
            for content_id, content in sorted(self.state.content.items())
        }
        self._search_index_hashes = {
            content_id: stable_hash(terms)
            for content_id, terms in sorted(self.state.search_index_terms.items())
        }
        self._suggestion_hashes = {
            suggestion_id: stable_hash(suggestion)
            for suggestion_id, suggestion in sorted(self.state.upgrade_suggestions.items())
        }
        self._institution_hashes = {
            institution_id: stable_hash(institution)
            for institution_id, institution in sorted(self.state.emergent_institutions.items())
        }
        self._communities_hash = stable_hash(self.state.communities)
        self._hash_cache_dirty = False

    def _update_agent_hash(self, agent_id: str) -> None:
        if self.config.state_hash_mode != "merkle":
            return
        self._agent_hashes[agent_id] = self._agent_state_hash(self.state.agents[agent_id])

    def _update_content_hash(self, content_id: str) -> None:
        if self.config.state_hash_mode != "merkle":
            return
        self._content_hashes[content_id] = self._content_item_hash(self.state.content[content_id])

    def _update_search_index_hash(self, content_id: str) -> None:
        if self.config.state_hash_mode != "merkle":
            return
        terms = self.state.search_index_terms.get(content_id)
        if terms is None:
            self._search_index_hashes.pop(content_id, None)
            return
        self._search_index_hashes[content_id] = stable_hash(terms)

    def _update_suggestion_hash(self, suggestion_id: str) -> None:
        if self.config.state_hash_mode != "merkle":
            return
        self._suggestion_hashes[suggestion_id] = stable_hash(self.state.upgrade_suggestions[suggestion_id])

    def _update_institution_hash(self, institution_id: str) -> None:
        if self.config.state_hash_mode != "merkle":
            return
        self._institution_hashes[institution_id] = stable_hash(self.state.emergent_institutions[institution_id])

    @staticmethod
    def _update_hash_field(hasher, key: str, value: str) -> None:
        key_bytes = key.encode("utf-8")
        value_bytes = value.encode("utf-8")
        hasher.update(len(key_bytes).to_bytes(4, "big"))
        hasher.update(key_bytes)
        hasher.update(len(value_bytes).to_bytes(4, "big"))
        hasher.update(value_bytes)

    @classmethod
    def _ordered_leaf_root(cls, items) -> str:
        payload = "\n".join(f"{key}\t{value}" for key, value in items)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _sequence_root(values: list[str]) -> str:
        payload = "\n".join(f"{index}\t{value}" for index, value in enumerate(values))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _memory_item_hash(item) -> str:
        return stable_hash(
            (
                item.event_ref,
                item.semantic_summary,
                item.source,
                item.valence,
                item.importance,
                item.uncertainty,
                item.timestamp,
                item.last_retrieved,
                item.decay_rate,
                item.llm_summary_ref_optional,
            )
        )

    def _agent_memory_hash(self, agent) -> str:
        cached = self._agent_memory_hashes.get(agent.id)
        if cached:
            return cached
        item_hashes = [self._memory_item_hash(item) for item in agent.memory]
        memory_hash = self._sequence_root(item_hashes)
        if self.config.state_hash_mode == "merkle":
            self._agent_memory_item_hashes[agent.id] = item_hashes
            self._agent_memory_hashes[agent.id] = memory_hash
        return memory_hash

    def _agent_state_hash(self, agent) -> str:
        return stable_hash(
            {
                "id": agent.id,
                "profile_hash": self._profile_hashes.get(agent.id) or stable_hash(agent.profile),
                "body": (
                    agent.body.hunger,
                    agent.body.sleep_pressure,
                    agent.body.fatigue,
                    agent.body.stress,
                    agent.body.health_reserve,
                    agent.body.physiological_energy,
                ),
                "affect": (
                    agent.affect.mood,
                    agent.affect.anger,
                    agent.affect.fear,
                    agent.affect.curiosity,
                    agent.affect.patience,
                ),
                "cognition": {
                    "attention_filter": agent.cognition.attention_filter,
                    "attention_allocation_policy": agent.cognition.attention_allocation_policy,
                    "working_memory_items": tuple(agent.cognition.working_memory_items),
                    "beliefs_about_artifacts": agent.cognition.beliefs_about_artifacts,
                    "artifact_peer_signal_strength": agent.cognition.artifact_peer_signal_strength,
                    "artifact_negative_signal_strength": agent.cognition.artifact_negative_signal_strength,
                    "artifact_peer_signal_count": agent.cognition.artifact_peer_signal_count,
                    "beliefs_about_people": agent.cognition.beliefs_about_people,
                    "beliefs_about_claims": agent.cognition.beliefs_about_claims,
                    "private_norm_hypotheses": agent.cognition.private_norm_hypotheses,
                    "institution_feedback": agent.cognition.institution_feedback,
                    "uncertainty": agent.cognition.uncertainty,
                    "goal_stack": tuple(agent.cognition.goal_stack),
                    "habit_strengths": agent.cognition.habit_strengths,
                },
                "memory_hash": self._agent_memory_hash(agent),
                "material_resources": (
                    agent.material_resources.food_or_budget_token,
                    agent.material_resources.artifact_access,
                ),
                "action_budgets": (
                    agent.action_budgets.attention_budget,
                    agent.action_budgets.time_budget,
                ),
                "location": agent.location,
                "social": {
                    "tie_strength": agent.social.tie_strength,
                    "trust": agent.social.trust,
                    "familiarity": agent.social.familiarity,
                    "debt_or_obligation": agent.social.debt_or_obligation,
                    "past_interaction_count": agent.social.past_interaction_count,
                    "perceived_status": agent.social.perceived_status,
                    "social_trust_as_private_belief": agent.social.social_trust_as_private_belief,
                },
                "channel_access": tuple(sorted(agent.channel_access)),
                "artifact_access": tuple(sorted(agent.artifact_access)),
                "artifact_exposure_sources": agent.artifact_exposure_sources,
                "artifact_exposure_groups": agent.artifact_exposure_groups,
                "artifact_peer_signal_groups": agent.artifact_peer_signal_groups,
                "artifact_exposure_ticks": agent.artifact_exposure_ticks,
                "artifact_exposed_by": agent.artifact_exposed_by,
                "artifact_exposure_content": agent.artifact_exposure_content,
                "artifact_exposure_distance": agent.artifact_exposure_distance,
                "artifact_usage_counts": agent.artifact_usage_counts,
                "artifact_payment_state": agent.artifact_payment_state,
                "artifact_payment_ticks": agent.artifact_payment_ticks,
                "latest_artifact_experience": agent.latest_artifact_experience,
                "artifact_experience_history": agent.artifact_experience_history,
                "artifact_upgrade_suggestions": agent.artifact_upgrade_suggestions,
                "latest_llm_intents": agent.latest_llm_intents,
                "llm_intent_session_history": tuple(
                    agent.llm_intent_session_history
                ),
                "llm_intent_feedback_history": tuple(agent.llm_intent_feedback_history),
                "action_affordances": tuple(sorted(agent.action_affordances)),
                "subscriptions": tuple(sorted(agent.subscriptions)),
                "search_queries": agent.search_queries,
            }
        )

    def _content_item_hash(self, content) -> str:
        return stable_hash(
            {
                "id": content.id,
                "author_id": content.author_id,
                "channel": content.channel,
                "kind": content.kind.value,
                "created_at": content.created_at,
                "parent_id": content.parent_id,
                "topic_vector": content.topic_vector,
                "artifact_refs": content.artifact_refs,
                "claim_refs": content.claim_refs,
                "evidence_refs": content.evidence_refs,
                "text_surface": content.text_surface,
                "visibility_scope": content.visibility_scope,
                "engagement_counts": content.engagement_counts,
                "controversy_proxy": content.controversy_proxy,
                "indexed_at": content.indexed_at,
                "deleted_by_author_flag": content.deleted_by_author_flag,
            }
        )

    def event_log_hash(self) -> str:
        return stable_hash(self.state.event_log)

    def manifest(self) -> RunManifest:
        parameter_hash = self.parameter_registry.freeze() if not self.parameter_registry.frozen else self.parameter_registry.hash()
        bounded_hash = self.bounded_score_registry.freeze() if not self.bounded_score_registry.frozen else self.bounded_score_registry.hash()
        return RunManifest(
            run_id=f"{self.config.scenario_id}_seed_{self.config.seed}",
            spec_version=self.config.spec_version,
            code_commit="uncommitted",
            python_version=platform.python_version(),
            scenario_config_hash=stable_hash(self.config),
            parameter_registry_hash=parameter_hash,
            agent_population_hash=stable_hash(self.state.agents),
            network_seed=self.config.seed,
            action_seed=self.config.seed,
            feed_seed=self.config.seed,
            exogenous_event_seed=self.config.seed,
            llm_enabled=self.config.llm_enabled,
            detectors_enabled_during_run=self.config.detectors_enabled_during_run,
            detectors_run_after_completion=True,
            detector_construct_validity_hash=construct_validity_suite_hash(),
            bounded_score_registry_hash=bounded_hash,
            claim_audit_hash=self.claim_audit.hash(),
            robustness_audit_plan_hash=stable_hash("pending_robustness_plan"),
            uncertainty_report_plan_hash=stable_hash("seed_level_interval_required"),
            type_separation_schema_hash=stable_hash("v17_type_separation_profile_experience_intent_community"),
            state_hash_mode=self.config.state_hash_mode,
            event_log_hash=self.event_log_hash(),
            state_snapshot_hashes={f"t{self.state.tick}": self.state_hash()},
        )


def _intent_signal_signature(agent: AgentState, artifact_id: str) -> str:
    return stable_hash(
        {
            "experience": agent.latest_artifact_experience.get(artifact_id),
            "peer_signal": round(agent.cognition.artifact_peer_signal_strength.get(artifact_id, 0.0), 3),
            "negative_signal": round(agent.cognition.artifact_negative_signal_strength.get(artifact_id, 0.0), 3),
            "peer_count": agent.cognition.artifact_peer_signal_count.get(artifact_id, 0),
            "payment_state": agent.artifact_payment_state.get(artifact_id),
        }
    )[:10]


def _profile_search_queries(agent: AgentState) -> tuple[str, ...]:
    terms = set(agent.profile.task_demand_weights)
    terms.add(agent.profile.primary_domain)
    terms.update(agent.profile.community_memberships)
    return tuple(sorted(terms))[:8]
