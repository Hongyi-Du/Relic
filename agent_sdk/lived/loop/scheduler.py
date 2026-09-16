"""Asynchronous tick loop / world scheduler (async infra §8, §16, §17).

Replaces strict round-robin with a discrete-tick async loop:

  per tick → passive decay → process due action COMPLETIONS (+ seeded resource
  tie-break) → advance ongoing-action progress → collect AVAILABLE agents →
  **seeded-shuffle** their decision order (no fixed agent_id bias) → for each
  available agent run the decision half (`decide_for_agent`) and SCHEDULE its
  action with a duration (agent is busy until ``action_end_tick``) → log the
  clock → advance the clock.

The action EFFECT runs at completion (so multi-tick actions are real). All
randomness is event-indexed (run_seed + world_tick + stream) so a fixed seed +
LLM cache reproduces the trajectory (§23). Blocking decision mode (§13): the
decision is rule-based here (LLM queue is §llm_queue for later modules).
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_sdk.lived.core.actions import get_spec
from agent_sdk.lived.perceive.candidates import CandidatePoolGenerator, EnvAwareFeatureExtractor
from agent_sdk.lived.world.clock import WorldClock, passive_energy_decay
from agent_sdk.lived.loop.control_loop import decide_for_agent
from agent_sdk.lived.loop.handlers import action_duration, execute_action
from agent_sdk.lived.record.logs import Journal, _strip_base
from agent_sdk.lived.cognition.memory import MemoryRetriever, MemoryStore
from agent_sdk.lived.perceive.perception import AgentGT, GroundTruthScene, PerceptionBuilder
from agent_sdk.lived.persona.profile import ProfileState


def seeded_rng(*parts: Any) -> random.Random:
    """Event-indexed deterministic RNG (§23)."""
    key = "|".join(str(p) for p in parts)
    return random.Random(int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16))


@dataclass
class AsyncTickLoop:
    scene: GroundTruthScene
    profiles: Dict[str, ProfileState] = field(default_factory=dict)
    journal: Journal = field(default_factory=Journal)
    run_id: str = "run"
    run_seed: int = 20260608
    builder: PerceptionBuilder = field(default_factory=PerceptionBuilder)
    candgen: CandidatePoolGenerator = field(default_factory=CandidatePoolGenerator)
    extractor: EnvAwareFeatureExtractor = field(default_factory=EnvAwareFeatureExtractor)
    retriever: MemoryRetriever = field(default_factory=MemoryRetriever)
    stores: Dict[str, MemoryStore] = field(default_factory=dict)
    enable_passive_decay: bool = True
    top_k: int = 5
    _scheduled: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if self.scene.clock is None:
            self.scene.clock = WorldClock()

    @property
    def clock(self) -> WorldClock:
        return self.scene.clock

    def store_for(self, uid: str) -> MemoryStore:
        s = self.stores.get(uid)
        if s is None:
            s = MemoryStore(uid); self.stores[uid] = s
        return s

    # -- §8 the tick --------------------------------------------------------
    def tick(self) -> Dict[str, Any]:
        clk = self.clock
        wt = clk.world_tick
        self.scene.turn = wt
        self.scene.run_id = self.run_id

        # 2. passive energy decay (awake agents only; sleepers excluded, §7)
        if self.enable_passive_decay:
            dec = passive_energy_decay(clk)
            for a in self.scene.agents:
                if a.current_action_type != "sleep":
                    a.energy = max(0.0, a.energy - dec)

        # 3-4. process due completions (+ seeded resource tie-break, §17)
        self.scene.completed_events = []
        due = [a for a in self.scene.agents
               if a.current_action_status == "in_progress" and a.action_end_tick <= wt]
        for a in self._completion_order(due, wt):
            self._complete(a, wt)

        # 5. advance ongoing-action progress for still-running agents (§18)
        for a in self.scene.agents:
            if a.current_action_status == "in_progress":
                span = max(1, a.action_end_tick - a.action_start_tick)
                a.action_progress = min(1.0, max(0.0, (wt - a.action_start_tick) / span))

        # 6-8. available agents → seeded-shuffle decision order (§16) → decide+schedule
        avail = [a for a in self.scene.agents if a.is_available(wt) and not _dead(a)]
        order = self._decision_order(avail, wt)
        self.journal.record("DecisionOrderLog", source_module="scheduler", turn_id=wt,
                            order=[a.id for a in order])
        for a in order:
            self._decide_and_schedule(a, wt)

        # 21. ClockLog / WorldStateLog
        self.journal.record("WorldStateLog", source_module="clock", turn_id=wt,
                            passive_decay_applied=self.enable_passive_decay,
                            available_agents=[a.id for a in avail],
                            completed_events=[e["id"] for e in self.scene.completed_events],
                            **clk.to_dict())

        clk.tick()
        return {"world_tick": wt, "available": [a.id for a in avail],
                "completed": [e["id"] for e in self.scene.completed_events]}

    def run(self, ticks: int) -> List[Dict[str, Any]]:
        return [self.tick() for _ in range(ticks)]

    # -- ordering -----------------------------------------------------------
    def _decision_order(self, avail: List[AgentGT], wt: int) -> List[AgentGT]:
        rng = seeded_rng(self.run_seed, wt, "decision_order")
        order = list(avail)
        rng.shuffle(order)
        return order

    def _completion_order(self, due: List[AgentGT], wt: int) -> List[AgentGT]:
        """Deterministic order; contested resources (same patch, same tick) get a
        seeded tie-break (§17)."""
        contested: Dict[str, List[AgentGT]] = {}
        for a in due:
            rid = (self._scheduled.get(a.id, {}).get("params", {}) or {}).get("resource_id")
            if rid and self._scheduled.get(a.id, {}).get("action_type") == "gather_resource":
                contested.setdefault(rid, []).append(a)
        rank: Dict[str, tuple] = {}
        for rid, group in contested.items():
            if len(group) <= 1:
                continue
            rng = seeded_rng(self.run_seed, wt, "tie_break", rid)
            sh = list(group); rng.shuffle(sh)
            patch = next((r for r in self.scene.resources if r.get("id") == rid), {})
            self.journal.record("TieBreakLog", source_module="scheduler", turn_id=wt,
                                resource_id=rid, competing_agent_ids=[x.id for x in group],
                                available_amount=patch.get("amount"),
                                winner_order=[x.id for x in sh])
            for i, x in enumerate(sh):
                rank[x.id] = (rid, i)
        return sorted(due, key=lambda a: rank.get(a.id, ("", 0)))

    # -- decision → schedule (§8.12) ---------------------------------------
    def _decide_and_schedule(self, a: AgentGT, wt: int) -> None:
        ps = self.profiles.get(a.id) or ProfileState(agent_id=a.id)
        self.profiles.setdefault(a.id, ps)
        trace, chosen, params, _, _ = decide_for_agent(
            self.scene, a, ps, journal=self.journal, builder=self.builder, candgen=self.candgen,
            extractor=self.extractor, retriever=self.retriever, store=self.store_for(a.id),
            run_id=self.run_id, run_seed=self.run_seed, turn=wt, top_k=self.top_k)
        a.last_decision_tick = wt
        if chosen is None:
            return
        spec = get_spec(chosen)
        dur = action_duration(spec, a, params, clock=self.clock)
        a.current_action_status = "in_progress"
        a.current_action_type = chosen
        a.action_start_tick = wt
        a.action_end_tick = wt + dur
        a.action_progress = 0.0
        a.next_available_tick = wt + dur
        a.busy_reason = chosen
        a.interruptible = bool(spec.interruptible) if spec else True
        self._scheduled[a.id] = {"action_type": chosen, "params": params,
                                 "scheduled_tick": wt, "end_tick": wt + dur}

    # -- completion (apply effect) -----------------------------------------
    def _complete(self, a: AgentGT, wt: int) -> None:
        sched = self._scheduled.pop(a.id, None)
        if sched is None:
            a.current_action_status = "idle"
            a.next_available_tick = wt
            return
        res = execute_action(self.scene, a, sched["action_type"], sched["params"],
                             turn=wt, run_id=self.run_id, clock=self.clock)
        res.action_start_tick = a.action_start_tick
        res.action_end_tick = a.action_end_tick
        res.scheduled_tick = sched["scheduled_tick"]
        res.completion_tick = wt
        res.next_available_tick_before = a.next_available_tick
        a.current_action_status = "idle"
        a.current_action_type = None
        a.action_progress = 0.0
        a.next_available_tick = wt
        a.busy_reason = ""
        res.next_available_tick_after = wt

        self.journal.record("ActionExecutionLog", source_module="execution", turn_id=wt, agent_id=a.id,
                            **_strip_base({k: v for k, v in res.to_dict().items()
                                           if k not in ("appraisal", "memory_writes")}))
        if res.appraisal:
            self.journal.record("EventAppraisalLog", source_module="appraisal", turn_id=wt, agent_id=a.id,
                                **_strip_base({k: v for k, v in res.appraisal.items()
                                               if k not in ("trigger_kinds",)}))
        for mw in res.memory_writes:
            tgt = mw.get("agent_id", a.id)
            self.store_for(tgt).new(summary=mw.get("summary", ""), tags=list(mw.get("tags", [])),
                                    related_agents=list(mw.get("related_agents", [])),
                                    related_resources=list(mw.get("related_resources", [])),
                                    memory_type=mw.get("memory_type", "episodic_memory"),
                                    salience=float(mw.get("salience", 0.2)), created_turn=wt,
                                    source_event_id=mw.get("source_event_id"))
        self.journal.log_replay_timeline(turn_id=wt, agent_id=a.id,
                                         text=f"{a.id} completed {res.action_type} "
                                              f"({'ok' if res.success else 'fail'})")
        self.scene.completed_events.append({
            "id": res.completion_event_id, "actor": a.id, "action_type": res.action_type,
            "location": [a.x, a.y], "success": res.success})


def _dead(a: AgentGT) -> bool:
    return a.hp <= 0 or bool(getattr(a, "extra", {}).get("is_dead"))
