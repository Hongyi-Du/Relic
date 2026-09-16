"""LivedDecisionController — orchestrates the §13 ReAct decision loop.

This is the single integration point the harness Act phase calls into. It
wires together the scaffold pieces and runs the §13.1 loop:

    candidates  <- candidate sources (env/persona/social/institution/wish)
    [wish]      <- wish parser on the agent's thought -> needs
    [affordance]<- synth + verify -> unlock new candidates
    scored      <- ProfileToPolicy.rank (features + profile weights + mood)
    choice      <- ProfileToPolicy.sample (bounded stochastic)
    -> returns a DecisionResult the caller turns into an AgentAction

It is intentionally adapter-shaped but env-agnostic: everything env-specific
arrives via ports (FeatureExtractor / CandidateSource / WishParser /
AffordanceSynth) or via the shared graphs. With only the NoOp defaults it runs
end-to-end and returns "no decision" — which is what the smoke test asserts.

Harness wiring (follow-up, not done here): construct one controller per
HarnessRunner, hold per-agent ProfileState in a registry, and in
``Agent.act_phase_async`` replace/augment the bare AgenticMiniLoop tool pick
with ``controller.decide(...)``. The LLM still proposes candidates upstream;
the controller only re-weights + samples (resolves the §14.1 tension).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_sdk.lived.perceive.affordance import AffordanceSynthesizer, AffordanceVerifier
from agent_sdk.lived.cognition.appraisal import EventAppraiser, apply_event_pipeline
from agent_sdk.lived.core.contracts import (
    ActionCandidate,
    AffordanceProposal,
    CandidateSource,
    Need,
    ScoredCandidate,
    VerifierVerdict,
)
from agent_sdk.lived.cognition.episode import EpisodeManager, TerminalAppraisal
from agent_sdk.lived.persona.policy import ProfileToPolicy, TriggerContext
from agent_sdk.lived.persona.profile import ProfileState, ProfileUpdater
from agent_sdk.lived.core.ports import (
    CandidateSourcePort,
    FeatureExtractorPort,
    WishParserPort,
)
from agent_sdk.lived.cognition.wish import RuleBasedWishParser


@dataclass
class DecisionResult:
    """Output of one decide() call."""
    chosen: Optional[ScoredCandidate] = None
    ranked: List[ScoredCandidate] = field(default_factory=list)
    needs: List[Need] = field(default_factory=list)
    unlocked: List[AffordanceProposal] = field(default_factory=list)
    pending_proposals: List[AffordanceProposal] = field(default_factory=list)

    @property
    def action_dict(self) -> Optional[Dict[str, Any]]:
        if self.chosen is None:
            return None
        return self.chosen.candidate.to_action_dict()


class LivedDecisionController:
    def __init__(
        self,
        *,
        feature_extractor: Optional[FeatureExtractorPort] = None,
        candidate_sources: Optional[List[CandidateSourcePort]] = None,
        wish_parser: Optional[WishParserPort] = None,
        affordance_synth: Optional[AffordanceSynthesizer] = None,
        affordance_verifier: Optional[AffordanceVerifier] = None,
        profile_updater: Optional[ProfileUpdater] = None,
        appraiser: Optional[EventAppraiser] = None,
        episode_manager: Optional[EpisodeManager] = None,
        top_k: int = 3,
    ):
        self.policy = ProfileToPolicy(feature_extractor=feature_extractor)
        self.candidate_sources: List[CandidateSourcePort] = list(candidate_sources or [])
        self.wish_parser: WishParserPort = wish_parser or RuleBasedWishParser()
        self.affordance_synth = affordance_synth or AffordanceSynthesizer()
        self.affordance_verifier = affordance_verifier or AffordanceVerifier()
        self.profile_updater = profile_updater or ProfileUpdater()
        # post-action services: appraisal (§5) + episodes (§6). Shared across
        # all agents the controller serves.
        self.appraiser = appraiser or EventAppraiser()
        self.episodes = episode_manager or EpisodeManager()
        self.top_k = top_k

    # -- candidate fan-out --------------------------------------------------
    def gather_candidates(self, *, agent_id: str, state: Any) -> List[ActionCandidate]:
        out: List[ActionCandidate] = []
        for src in self.candidate_sources:
            try:
                out.extend(src.generate(agent_id=agent_id, state=state))
            except Exception:
                continue
        return out

    # -- §11/§12 wish -> affordance ----------------------------------------
    def process_wish(
        self, *, agent_id: str, thought: str, state: Any = None,
    ) -> tuple[List[Need], List[AffordanceProposal], List[AffordanceProposal]]:
        """Parse thought -> needs -> synthesize -> verify. Returns
        (needs, unlocked, pending_group_proposals)."""
        needs = self.wish_parser.parse(agent_id=agent_id, thought=thought, state=state)
        unlocked: List[AffordanceProposal] = []
        pending: List[AffordanceProposal] = []
        for need in needs:
            for prop in self.affordance_synth.synthesize(need=need, state=state):
                needs_adoption = prop.name in (
                    "create_contribution_ledger", "propose_rationing_rule",
                    "call_council_meeting",
                )
                verified = self.affordance_verifier.verify(
                    prop, state=state, needs_group_adoption=needs_adoption
                )
                if verified.verdict == VerifierVerdict.UNLOCKED:
                    unlocked.append(verified)
                elif verified.verdict == VerifierVerdict.PENDING_PROPOSAL:
                    pending.append(verified)
        return needs, unlocked, pending

    # -- §13.1 main loop ----------------------------------------------------
    def decide(
        self,
        *,
        profile_state: ProfileState,
        state: Any = None,
        thought: str = "",
        extra_candidates: Optional[List[ActionCandidate]] = None,
        trigger_context: Optional[TriggerContext] = None,
    ) -> DecisionResult:
        agent_id = profile_state.agent_id
        # 1. candidates from env/persona/social/institution sources + caller extras
        candidates = self.gather_candidates(agent_id=agent_id, state=state)
        if extra_candidates:
            candidates.extend(extra_candidates)

        # 2. wish -> affordance -> unlock new attemptable candidates
        needs, unlocked, pending = self.process_wish(
            agent_id=agent_id, thought=thought, state=state,
        )
        for prop in unlocked:
            # The unlocked affordance becomes a WISH-sourced candidate (its
            # first primitive is the entry action; params filled by env later).
            entry = prop.primitive_decomposition[0] if prop.primitive_decomposition else prop.name
            candidates.append(ActionCandidate(
                action_type=entry,
                parameters={"affordance": prop.name},
                source=CandidateSource.WISH,
                rationale=prop.need.motivation,
            ))

        # 3. score + 4. sample (profile + mood drive both)
        ranked = self.policy.rank(
            agent_id=agent_id,
            candidates=candidates,
            profile=profile_state.profile,
            mood=profile_state.mood,
            state=state,
            tctx=trigger_context,
        )
        chosen = self.policy.sample(
            ranked, profile=profile_state.profile, mood=profile_state.mood, top_k=self.top_k,
        )
        return DecisionResult(
            chosen=chosen, ranked=ranked, needs=needs,
            unlocked=unlocked, pending_proposals=pending,
        )

    # -- §5/§6 post-action: appraise -> ordered update -> episode -----------
    def observe_outcome(
        self,
        *,
        profile_state: ProfileState,
        action_type: str,
        turn: int,
        target: Optional[str] = None,
        success: bool = True,
        kind: str = "",
        repeat_count: int = 0,
        event_graph: Any = None,
        social_graph: Any = None,
        knowledge_graph: Any = None,
        match_hints: Optional[Dict[str, Any]] = None,
        **signals: Any,
    ) -> Dict[str, Any]:
        """Close the loop after an action executes (§5.2 + §6): build a
        structured appraisal, run the ordered update (mood/social/skill/stable-
        trait), and route the event into the episode manager. Returns the trace
        plus any episodes that terminated this turn (so the caller can apply
        their terminal appraisals to *all* participants via
        :meth:`apply_terminal`)."""
        appraisal = self.appraiser.appraise(
            actor=profile_state.agent_id, action_type=action_type, turn=turn,
            target=target, success=success, kind=kind, repeat_count=repeat_count,
            **signals,
        )
        # route episode first so the appraisal carries the matched episode id
        route = self.episodes.route_event(appraisal, **(match_hints or {}))
        appraisal.primary_episode_id = route.get("primary_episode_id")
        appraisal.related_episode_id = (route.get("related_episode_ids") or [None])[0]

        trace = apply_event_pipeline(
            appraisal=appraisal, profile_state=profile_state,
            profile_updater=self.profile_updater, event_graph=event_graph,
            social_graph=social_graph, knowledge_graph=knowledge_graph,
            episode_manager=None,  # already routed above; don't double-route
        )
        trace["route"] = route
        # collect terminal appraisals for episodes that just ended
        terminals = [self.episodes.episodes[e].terminal
                     for e in route.get("transitioned", [])
                     if self.episodes.episodes.get(e) and self.episodes.episodes[e].terminal]
        trace["ended_terminals"] = terminals
        return trace

    def apply_terminal(self, profile_state: ProfileState, terminal: TerminalAppraisal) -> Dict[str, Any]:
        """Apply one ended episode's terminal appraisal to a single
        participant's persona — the §6.8 rule that an episode moves stable
        traits exactly once, at the end. No-op if updates aren't allowed or this
        agent didn't participate."""
        for ap in self.episodes.terminal_to_appraisals(terminal):
            if ap.actor == profile_state.agent_id:
                return self.profile_updater.apply_appraisal(profile_state, ap)
        return {}
