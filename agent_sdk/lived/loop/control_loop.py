"""Live agent control loop (Stage B §1, §8, §12, §13).

Wires the minimal live loop end to end, per agent per turn:

    Perception → SelfState → MemoryRetrieval → CandidatePool → FeatureExtraction
    → PCBSP (seeded softmax) → ActionExecution → EventAppraisal
    → Memory / state / graph update → Logging

against a mutable env-agnostic world (a :class:`GroundTruthScene` of
:class:`AgentGT`). Every stage writes its log stream (§12). Speech is an Action
(executed by the speech handler); Perception only *receives* speech (§3). Wish is
NOT in the candidate pool (§2). No stable-trait updates here (§11.7) — only the
env-driven mood (hunger/fatigue) is synced into the PCBSP mood.

This is the test/demonstration harness; the nature_env adapter (world →
GroundTruthScene) is the later live-wiring point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_sdk.lived.perceive.candidates import (
    CandidatePoolGenerator,
    EnvAwareFeatureExtractor,
    PrecomputedFeatureExtractor,
)
from agent_sdk.lived.loop.handlers import execute_action
from agent_sdk.lived.record.logs import Journal, _strip_base
from agent_sdk.lived.cognition.memory import MemoryRetriever, MemoryStore, query_from_perception
from agent_sdk.lived.persona.pcbsp import DecisionContext, PCBSPPolicy
from agent_sdk.lived.perceive.perception import AgentGT, GroundTruthScene, PerceptionBuilder
from agent_sdk.lived.persona.profile import ProfileState


def decide_for_agent(scene: GroundTruthScene, agent: "AgentGT", profile: ProfileState, *,
                     journal: Journal, builder: PerceptionBuilder, candgen: CandidatePoolGenerator,
                     extractor: EnvAwareFeatureExtractor, retriever: MemoryRetriever,
                     store: MemoryStore, run_id: str, run_seed: int, turn: int,
                     top_k: int = 5, social: Any = None, episodes: Any = None,
                     groups: Optional[Dict[str, str]] = None,
                     plan_modifiers: Optional[Dict[str, float]] = None):
    """Run the DECISION half of the loop (perception → self-state → memory →
    candidate pool → feature extraction → PCBSP) and log each stream. Returns
    ``(trace, chosen_action, params, packet, self_state)``. Shared by the sync
    LiveLoop and the async scheduler; it does NOT execute the action.

    ``social`` / ``episodes`` / ``groups`` (all optional) are forwarded into the
    :class:`DecisionContext` so RelationScore (§9) and EpisodeValue (§11) are
    live instead of hard-zero when the caller owns a social graph / episode
    manager (the takeover controller passes the shared LivedSociety state)."""
    uid = agent.id
    packet, self_state, p_appraisals = builder.build_for(scene, uid)
    journal.log_perception(turn_id=turn, agent_id=uid, packet=packet)
    journal.log_self_state(turn_id=turn, agent_id=uid, percept=self_state)
    for ap in p_appraisals:
        journal.log_perception_appraisal(turn_id=turn, agent_id=uid, appraisal=ap)

    profile.mood.hunger_pressure = self_state.hunger_pressure
    profile.mood.fatigue = self_state.fatigue
    profile.mood.stress = self_state.stress

    q = query_from_perception(packet, self_state, run_id=run_id, turn=turn, agent_id=uid,
                              terrain_type=scene.season,
                              recent_event_tags=[s.split()[1] for s in packet.salient_changes if " " in s])
    ctx_mem = retriever.build_context(store, q, turn=turn)
    journal.log_memory_retrieval(turn_id=turn, agent_id=uid, context=ctx_mem)

    candidates = candgen.generate(agent=agent, packet=packet, self_state=self_state, scene=scene)
    journal.record("CandidatePoolLog", source_module="candidates", turn_id=turn, agent_id=uid,
                   candidate_pool=[c.action_type for c in candidates], candidate_count=len(candidates))

    fx_state = {"agent": agent, "self_state": self_state, "scene": scene, "packet": packet}
    # keyed per-candidate (action_type + params), NOT per action_type: two
    # gather_resource candidates with different targets keep distinct features
    # (a food gather's survival_gain must not shadow a material gather's).
    from agent_sdk.lived.perceive.candidates import candidate_key
    feat_table = {}
    for c in candidates:
        feats, sources, reasons = extractor.extract_with_reasons(agent_id=uid, candidate=c, state=fx_state)
        feat_table[candidate_key(c)] = feats
        journal.record("FeatureExtractionLog", source_module="feature_extractor", turn_id=turn,
                       agent_id=uid, action_type=c.action_type,
                       features={k: v for k, v in feats.to_dict().items() if v},
                       feature_sources=sources, extraction_reasons=reasons)

    pol = PCBSPPolicy(feature_extractor=PrecomputedFeatureExtractor(feat_table))
    energy_pct = 100.0 * float(agent.energy) / (float(agent.max_energy) or 100.0)
    hp_pct = 100.0 * float(agent.hp) / (float(agent.max_hp) or 1.0)
    ctx_d = DecisionContext(turn_id=turn, decision_id=uid, run_seed=run_seed,
                            energy=energy_pct, hp=hp_pct, state=fx_state,
                            social=social, episodes=episodes, groups=dict(groups or {}),
                            plan_modifiers=dict(plan_modifiers or {}))
    trace = pol.select(agent_id=uid, candidates=candidates, profile=profile.profile,
                       mood=profile.mood, ctx=ctx_d, top_k=top_k)
    journal.record("PolicyTraceLog", source_module="pcbsp", turn_id=turn, agent_id=uid,
                   **_strip_base(trace.to_dict()))

    chosen = trace.selected_action
    params = {}
    if chosen is not None:
        # the trace carries the CHOSEN candidate's own parameters — with several
        # same-type candidates (gather A vs gather B) the first-of-type lookup
        # would return the wrong target, so it is only a fallback.
        params = dict(trace.selected_parameters or {})
        if not params:
            cand = next((c for c in candidates if c.action_type == chosen), None)
            params = dict(cand.parameters) if cand is not None else {}
    return trace, chosen, params, packet, self_state


@dataclass
class LiveLoop:
    journal: Journal = field(default_factory=Journal)
    run_id: str = "run"
    run_seed: int = 20260608
    builder: PerceptionBuilder = field(default_factory=PerceptionBuilder)
    candgen: CandidatePoolGenerator = field(default_factory=CandidatePoolGenerator)
    extractor: EnvAwareFeatureExtractor = field(default_factory=EnvAwareFeatureExtractor)
    retriever: MemoryRetriever = field(default_factory=MemoryRetriever)
    policy: PCBSPPolicy = field(default_factory=PCBSPPolicy)
    stores: Dict[str, MemoryStore] = field(default_factory=dict)
    top_k: int = 5

    def store_for(self, uid: str) -> MemoryStore:
        s = self.stores.get(uid)
        if s is None:
            s = MemoryStore(uid)
            self.stores[uid] = s
        return s

    def run_turn(self, scene: GroundTruthScene, profiles: Dict[str, ProfileState],
                 turn: int) -> Dict[str, Dict[str, Any]]:
        scene.turn = turn
        scene.run_id = self.run_id
        J = self.journal
        results: Dict[str, Dict[str, Any]] = {}

        for agent in list(scene.agents):
            uid = agent.id
            ps = profiles.get(uid) or ProfileState(agent_id=uid)
            profiles.setdefault(uid, ps)

            # 1-5) decision half (perception → memory → candidates → features → PCBSP)
            trace, chosen, params, packet, self_state = decide_for_agent(
                scene, agent, ps, journal=J, builder=self.builder, candgen=self.candgen,
                extractor=self.extractor, retriever=self.retriever, store=self.store_for(uid),
                run_id=self.run_id, run_seed=self.run_seed, turn=turn, top_k=self.top_k)

            # 6) execute the chosen action (sync Stage B: immediate)
            if chosen is None:
                results[uid] = {"trace": trace.to_dict(), "result": None}
                continue
            exec_res = execute_action(scene, agent, chosen, params, turn=turn, run_id=self.run_id,
                                      clock=scene.clock)
            J.record("ActionExecutionLog", source_module="execution", turn_id=turn, agent_id=uid,
                     **_strip_base({k: v for k, v in exec_res.to_dict().items()
                                    if k not in ("appraisal", "memory_writes")}))

            # 7) event appraisal
            if exec_res.appraisal:
                J.record("EventAppraisalLog", source_module="appraisal", turn_id=turn, agent_id=uid,
                         **_strip_base({k: v for k, v in exec_res.appraisal.items()
                                        if k not in ("trigger_kinds",)}))

            # 8) memory / graph update (incl. listener memories for speech)
            for mw in exec_res.memory_writes:
                tgt = mw.get("agent_id", uid)
                self.store_for(tgt).new(summary=mw.get("summary", ""), tags=list(mw.get("tags", [])),
                                        related_agents=list(mw.get("related_agents", [])),
                                        related_resources=list(mw.get("related_resources", [])),
                                        memory_type=mw.get("memory_type", "episodic_memory"),
                                        salience=float(mw.get("salience", 0.2)), created_turn=turn,
                                        source_event_id=mw.get("source_event_id"))
            J.record("GraphUpdateLog", source_module="graph", turn_id=turn, agent_id=uid,
                     event_id=(exec_res.event or {}).get("event_id"),
                     memory_writes=len(exec_res.memory_writes),
                     state_delta=exec_res.state_delta)
            J.log_replay_timeline(turn_id=turn, agent_id=uid,
                                  text=self._timeline_line(uid, exec_res))

            results[uid] = {"trace": trace.to_dict(), "result": exec_res.to_dict()}
        return results

    def run(self, scene: GroundTruthScene, profiles: Dict[str, ProfileState],
            turns: int) -> List[Dict[str, Dict[str, Any]]]:
        return [self.run_turn(scene, profiles, t) for t in range(turns)]

    def _timeline_line(self, uid: str, res: Any) -> str:
        verb = "did" if res.success else "failed to"
        extra = ""
        if res.speech_event:
            extra = f" (heard by {', '.join(res.listeners) or 'no one'})"
        elif res.produced_resources:
            extra = f" (+{res.produced_resources})"
        return f"{uid} {verb} {res.action_type}{extra}"


# --------------------------------------------------------------------------- #
# §13 minimal live smoke world
# --------------------------------------------------------------------------- #
def build_smoke_world(*, with_bob: bool = True) -> "tuple":
    """One grain meadow + Alice (home, 2 hand slots, at the meadow) [+ Bob].
    Winter countdown active. Returns (scene, profiles)."""
    from agent_sdk.lived.core.contracts import ProfileVector
    alice = AgentGT(id="Alice", x=40, y=40, vision_radius=10, comm_radius=12,
                    energy=60, max_energy=100, hp=10, max_hp=10, satiety=45,  # hungry forager
                    carrying_capacity=2, current_load=0, home_location=[20, 20],
                    home_storage={}, inventory={}, mood={"fatigue": 0.1})
    agents = [alice]
    if with_bob:
        agents.append(AgentGT(id="Bob", x=44, y=40, vision_radius=10, comm_radius=12,
                              energy=70, carrying_capacity=4, home_location=[60, 60],
                              inventory={"reed": 2}))
    scene = GroundTruthScene(
        turn=0, run_id="smoke", agents=agents,
        resources=[{"id": "grain1", "x": 41, "y": 40, "type": "grain", "amount": 20}],
        camp_zone={"x": 40, "y": 40, "radius": 14}, season="winter",   # §13 winter countdown
    )
    profiles = {
        "Alice": ProfileState(agent_id="Alice", role="grain-forager",
                              profile=ProfileVector(curiosity=0.85, long_termism=0.85,
                                                    risk_aversion=0.3)),
    }
    if with_bob:
        profiles["Bob"] = ProfileState(agent_id="Bob", role="reed-weaver",
                                       profile=ProfileVector(altruism=0.8, reciprocity=0.7))
    return scene, profiles
