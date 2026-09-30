"""Perception layer (infra task §9-§14) — "what can this agent perceive now?"

Strictly separates **ground truth** (the real world, §20) from **per-agent
perception**: the builder takes a :class:`GroundTruthScene` and applies
DETERMINISTIC visibility rules (§12) to produce, for one agent, a
:class:`PerceptionPacket` + private :class:`SelfStatePercept` (+ derived signals,
§10.2) + structured :class:`PerceptionAppraisal` s (§14). It NEVER selects an
action, sends speech, or updates stable persona.

v1 has NO default global broadcast — every channel is radius/zone/inspection
gated (§12). Self-state is private to its agent (§10).

Env-agnostic: the env builds a ``GroundTruthScene`` from the world; this module
holds only the schemas + the visibility/appraisal logic (the interface later
modules — PlanMonitor, FeatureExtractor, PCBSP — consume, §21).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# §9.2 percept types
PERCEPT_TYPES = (
    "environmental_percept", "terrain_percept", "resource_percept", "object_percept",
    "agent_action_percept", "communication_percept", "public_record_percept",
    "institution_percept", "self_state_percept", "memory_trigger_percept",
)
# §13.1 public-record percept types
PUBLIC_RECORD_TYPES = (
    "public_mark", "contribution_tally", "withdrawal_tally", "rule_mark",
    "storage_record", "warning_mark", "disputed_record",
)
# §13.2 institution percept types
INSTITUTION_TYPES = (
    "active_trial_mechanism", "proposal_state", "support_or_opposition",
    "violation_claim", "enforcement_attempt", "appeal_meeting", "temporary_role_assignment",
)
# §11.2 reserved speech-act types (perception side only; sending is a later layer)
SPEECH_ACT_TYPES = (
    "ask_help", "tell_info", "camp_announce", "public_proposal", "support_proposal",
    "oppose_proposal", "accuse_violation", "explain_action", "promise_action",
    "apologize", "request_compensation", "teach_affordance",
)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except (TypeError, ValueError, IndexError):
        return float("inf")


# --------------------------------------------------------------------------- #
# Ground-truth inputs (§20) — the env fills these
# --------------------------------------------------------------------------- #
@dataclass
class AgentGT:
    """Ground-truth state of one agent (the env supplies it)."""
    id: str
    x: float = 0.0
    y: float = 0.0
    vision_radius: float = 10.0
    comm_radius: float = 15.0
    energy: float = 100.0
    max_energy: float = 100.0
    hp: float = 10.0
    max_hp: float = 10.0
    inventory: Dict[str, int] = field(default_factory=dict)
    equipped_items: List[str] = field(default_factory=list)
    home_location: Optional[Sequence[float]] = None
    home_storage: Dict[str, int] = field(default_factory=dict)
    carrying_capacity: float = 10.0
    current_load: float = 0.0
    hand_slots_used: int = 0
    hand_slots_total: int = 2
    active_plan_ids: List[str] = field(default_factory=list)
    active_episode_ids: List[str] = field(default_factory=list)
    current_foreground_episode_id: Optional[str] = None
    blocked_plan_steps: List[str] = field(default_factory=list)
    recent_failures: List[str] = field(default_factory=list)
    recent_successes: List[str] = field(default_factory=list)
    mood: Dict[str, float] = field(default_factory=dict)
    urgent_needs: List[str] = field(default_factory=list)
    injury_status: str = "none"
    team_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    # --- SocioGenesis survival model (agent_sdk.lived.world.survival, §1) -------- #
    # satiety/fatigue are SEPARATE from energy/mood: energy=stamina, satiety=
    # fullness (decays each tick, only eating restores it), fatigue=0–150 scale
    # (NOT mood["fatigue"][0,1]) with soft/hard caps + overexertion.
    satiety: float = 75.0
    max_satiety: float = 100.0
    satiety_decay_multiplier: float = 1.0
    fatigue: float = 20.0
    fatigue_soft_cap: float = 100.0
    fatigue_hard_cap: float = 150.0
    overexertion_debt: float = 0.0
    overexertion_count: int = 0
    consecutive_overexertion_ticks: int = 0
    forced_rest_until_tick: int = -1
    collapse_count: int = 0
    overwork_strain: float = 0.0
    death_reason: Optional[str] = None
    # --- async runtime fields (async infra §4) -------------------------- #
    current_action_id: Optional[str] = None
    current_action_type: Optional[str] = None
    current_action_status: str = "idle"      # idle/scheduled/in_progress/completed/failed/interrupted/cancelled
    action_start_tick: int = -1
    action_end_tick: int = -1
    next_available_tick: int = 0
    current_episode_id: Optional[str] = None
    foreground_episode_id: Optional[str] = None
    interruptible: bool = True
    action_progress: float = 0.0
    busy_reason: str = ""
    last_decision_tick: int = -1
    pending_llm_call_ids: List[str] = field(default_factory=list)

    def is_available(self, world_tick: int) -> bool:
        return self.current_action_status == "idle" and self.next_available_tick <= world_tick


@dataclass
class GroundTruthScene:
    """The real world this turn (§20). The builder reads it but never mutates it."""
    turn: int = 0
    run_id: str = "run"
    agents: List[AgentGT] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)     # {id,x,y,type,amount}
    objects: List[Dict[str, Any]] = field(default_factory=list)       # {id,x,y,type}
    agent_actions: List[Dict[str, Any]] = field(default_factory=list) # {actor,action,x,y,target,...}
    messages: List[Dict[str, Any]] = field(default_factory=list)      # see §11.1
    public_marks: List[Dict[str, Any]] = field(default_factory=list)  # {id,x,y,type,...}
    public_records: List[Dict[str, Any]] = field(default_factory=list)
    mechanisms: List[Dict[str, Any]] = field(default_factory=list)    # {id,x,y?,type,...}
    episode_events: List[Dict[str, Any]] = field(default_factory=list)
    camp_zone: Optional[Dict[str, float]] = None                      # {x,y,radius}
    season: str = "summer"
    weather: str = "clear"
    # per-agent sets of ids the agent has inspected/accessed (gates §13 records)
    inspected: Dict[str, Set[str]] = field(default_factory=dict)
    # async: world clock + this-tick completed events (ongoing-action perception §18)
    clock: Optional[Any] = None              # agent_sdk.lived.world.clock.WorldClock
    completed_events: List[Dict[str, Any]] = field(default_factory=list)
    campfires: List[Dict[str, Any]] = field(default_factory=list)   # {x,y,radius}


# --------------------------------------------------------------------------- #
# §10.1 SelfStatePercept (private)
# --------------------------------------------------------------------------- #
@dataclass
class SelfStatePercept:
    self_state_percept_id: str
    run_id: str
    turn_id: int
    agent_id: str
    location: Optional[Sequence[float]] = None
    home_location: Optional[Sequence[float]] = None
    distance_to_home: Optional[float] = None
    energy: float = 0.0
    hp: float = 0.0
    hunger_pressure: float = 0.0
    fatigue: float = 0.0
    stress: float = 0.0
    injury_status: str = "none"
    carrying_capacity: float = 0.0
    current_load: float = 0.0
    remaining_capacity: float = 0.0
    hand_slots_used: int = 0
    hand_slots_total: int = 2
    inventory_items: Dict[str, int] = field(default_factory=dict)
    equipped_items: List[str] = field(default_factory=list)
    home_storage_summary: Dict[str, int] = field(default_factory=dict)
    active_plan_ids: List[str] = field(default_factory=list)
    active_episode_ids: List[str] = field(default_factory=list)
    current_foreground_episode_id: Optional[str] = None
    blocked_plan_steps: List[str] = field(default_factory=list)
    recent_failures: List[str] = field(default_factory=list)
    recent_successes: List[str] = field(default_factory=list)
    current_mood_state: Dict[str, float] = field(default_factory=dict)
    urgent_needs: List[str] = field(default_factory=list)
    derived_signals: Dict[str, bool] = field(default_factory=dict)
    self_uncertainty_flags: List[str] = field(default_factory=list)
    # --- SocioGenesis survival model (§19) — satiety/hunger distinct from energy --- #
    satiety: float = 100.0
    max_satiety: float = 100.0
    satiety_stage: str = "normal"
    is_hungry: bool = False
    is_starving: bool = False
    is_critically_starving: bool = False
    starvation_hp_loss: float = 0.0
    fatigue_zone: str = "normal"
    exhaustion_pressure: float = 0.0
    is_exhausted: bool = False
    is_collapsed: bool = False
    overexertion_debt: float = 0.0
    overexertion_count: int = 0
    consecutive_overexertion_ticks: int = 0
    forced_rest_until_tick: int = -1
    collapse_count: int = 0
    overwork_strain: float = 0.0
    body_state: Dict[str, Any] = field(default_factory=dict)   # §25 LLM body-state block

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# §10.2 derived self-state signals
def derive_self_signals(a: AgentGT, *, hunger_pressure: float, winter: bool = False) -> Dict[str, bool]:
    remaining = a.carrying_capacity - a.current_load
    inv = a.inventory or {}
    home = a.home_storage or {}
    food = sum(v for k, v in inv.items() if "food" in k or "grain" in k or "berry" in k or k == "food")
    home_food = sum(v for k, v in home.items() if "food" in k or "grain" in k or k == "food")
    dist_home = _dist((a.x, a.y), a.home_location) if a.home_location else 0.0
    return {
        "is_hungry": hunger_pressure > 0.5,
        "is_exhausted": float(a.mood.get("fatigue", 0)) > 0.6,
        "is_injured": a.injury_status not in ("none", "", None) or a.hp < 0.4 * (a.max_hp or 1),
        "is_overloaded": a.current_load > a.carrying_capacity,
        "is_near_capacity": remaining <= max(1.0, 0.15 * (a.carrying_capacity or 1)),
        "has_food": food > 0,
        "has_materials_for_active_session": bool(a.active_episode_ids) and bool(inv),
        "is_far_from_home": a.home_location is not None and dist_home > 2 * (a.vision_radius or 1),
        "is_at_home": a.home_location is not None and dist_home <= 2.0,
        "home_food_low": home_food < 3,
        "winter_reserve_low": winter and home_food < 10,
        "active_plan_blocked": bool(a.blocked_plan_steps),
        "active_episode_stalled": False,
        "prototype_recently_failed": any("prototype" in f.lower() for f in a.recent_failures),
        "promise_pending": any("promise" in n.lower() for n in a.urgent_needs),
        "debt_pending": any("debt" in n.lower() for n in a.urgent_needs),
    }


# --------------------------------------------------------------------------- #
# §14.1 PerceptionAppraisal
# --------------------------------------------------------------------------- #
@dataclass
class PerceptionAppraisal:
    perception_appraisal_id: str
    run_id: str
    turn_id: int
    agent_id: str
    perceived_event_type: str = ""
    actor: Optional[str] = None
    target: Optional[str] = None
    resource: Optional[str] = None
    object: Optional[str] = None
    location: Optional[Sequence[float]] = None
    confidence: float = 0.6
    visibility: str = "visual"
    observed_by: List[str] = field(default_factory=list)
    potential_fairness_violation: bool = False
    potential_threat: bool = False
    potential_help: bool = False
    potential_teaching_signal: bool = False
    potential_rule_violation: bool = False
    possible_resource_opportunity: bool = False
    possible_bottleneck_signal: bool = False
    memory_salience: float = 0.0
    related_episode_id: Optional[str] = None
    related_mechanism_id: Optional[str] = None
    related_plan_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# §9.1 PerceptionPacket
# --------------------------------------------------------------------------- #
@dataclass
class PerceptionPacket:
    perception_packet_id: str
    run_id: str
    turn_id: int
    agent_id: str
    location: Optional[Sequence[float]] = None
    visible_tiles: List[Any] = field(default_factory=list)
    visible_resources: List[Dict[str, Any]] = field(default_factory=list)
    visible_objects: List[Dict[str, Any]] = field(default_factory=list)
    visible_agents: List[str] = field(default_factory=list)
    visible_agent_actions: List[Dict[str, Any]] = field(default_factory=list)
    heard_messages: List[Dict[str, Any]] = field(default_factory=list)
    overheard_messages: List[Dict[str, Any]] = field(default_factory=list)
    public_marks_seen: List[Dict[str, Any]] = field(default_factory=list)
    public_records_seen: List[Dict[str, Any]] = field(default_factory=list)
    visible_mechanisms: List[Dict[str, Any]] = field(default_factory=list)
    visible_episode_events: List[Dict[str, Any]] = field(default_factory=list)
    home_status: Dict[str, Any] = field(default_factory=dict)
    inventory_status: Dict[str, Any] = field(default_factory=dict)
    body_status: Dict[str, Any] = field(default_factory=dict)
    season_status: str = ""
    weather_status: str = ""
    salient_changes: List[str] = field(default_factory=list)
    uncertainty_flags: List[str] = field(default_factory=list)
    self_state_percept_id: Optional[str] = None
    # --- async ongoing-action perception (§18) -------------------------- #
    ongoing_visible_actions: List[Dict[str, Any]] = field(default_factory=list)
    completed_visible_events: List[Dict[str, Any]] = field(default_factory=list)
    visible_action_progress: Dict[str, float] = field(default_factory=dict)
    agents_currently_busy: List[str] = field(default_factory=list)
    agents_available_nearby: List[str] = field(default_factory=list)
    time_of_day: str = ""
    effective_visual_radius: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def evidence_ids(self) -> Dict[str, List[str]]:
        """Visible evidence ids (for the §6 LLM visibility_context)."""
        pids = [self.perception_packet_id]
        if self.self_state_percept_id:
            pids.append(self.self_state_percept_id)
        return {
            "percepts": pids
            + [m.get("message_id") for m in self.heard_messages if m.get("message_id")]
            + [m.get("message_id") for m in self.overheard_messages if m.get("message_id")],
            "public_records": [r.get("id") for r in (self.public_marks_seen + self.public_records_seen) if r.get("id")],
            "mechanisms": [m.get("id") for m in self.visible_mechanisms if m.get("id")],
            "events": [e.get("id") for e in self.visible_episode_events if e.get("id")],
        }


# --------------------------------------------------------------------------- #
# The builder — deterministic visibility (§12)
# --------------------------------------------------------------------------- #
class PerceptionBuilder:
    def build_self_state(self, scene: GroundTruthScene, a: AgentGT) -> SelfStatePercept:
        # SocioGenesis: hunger comes from SATIETY, not energy (§4/§19).
        from agent_sdk.lived.world import survival as _S
        tick = getattr(scene.clock, "world_tick", scene.turn) if scene.clock is not None else scene.turn
        body = _S.snapshot(a, current_tick=tick)
        hunger = body["hunger_pressure"]
        winter = scene.season == "winter"
        ss = SelfStatePercept(
            self_state_percept_id=f"ss:{scene.run_id}:{scene.turn}:{a.id}",
            run_id=scene.run_id, turn_id=scene.turn, agent_id=a.id,
            location=[a.x, a.y], home_location=a.home_location,
            distance_to_home=_dist((a.x, a.y), a.home_location) if a.home_location else None,
            energy=a.energy, hp=a.hp, hunger_pressure=hunger,
            fatigue=float(a.mood.get("fatigue", 0.0)), stress=float(a.mood.get("stress", 0.0)),
            injury_status=a.injury_status, carrying_capacity=a.carrying_capacity,
            current_load=a.current_load, remaining_capacity=max(0.0, a.carrying_capacity - a.current_load),
            hand_slots_used=a.hand_slots_used, hand_slots_total=a.hand_slots_total,
            inventory_items=dict(a.inventory), equipped_items=list(a.equipped_items),
            home_storage_summary=dict(a.home_storage), active_plan_ids=list(a.active_plan_ids),
            active_episode_ids=list(a.active_episode_ids),
            current_foreground_episode_id=a.current_foreground_episode_id,
            blocked_plan_steps=list(a.blocked_plan_steps), recent_failures=list(a.recent_failures),
            recent_successes=list(a.recent_successes), current_mood_state=dict(a.mood),
            urgent_needs=list(a.urgent_needs),
            satiety=body["satiety"], max_satiety=body["max_satiety"],
            satiety_stage=body["satiety_stage"], is_hungry=body["is_hungry"],
            is_starving=body["is_starving"], is_critically_starving=body["is_critically_starving"],
            fatigue_zone=body["fatigue_zone"], exhaustion_pressure=body["exhaustion_pressure"],
            is_exhausted=body["is_exhausted"], is_collapsed=body["is_collapsed"],
            overexertion_debt=body["overexertion_debt"], overexertion_count=body["overexertion_count"],
            consecutive_overexertion_ticks=body["consecutive_overexertion_ticks"],
            forced_rest_until_tick=body["forced_rest_until_tick"],
            collapse_count=body["collapse_count"], overwork_strain=body["overwork_strain"],
            body_state=body,
        )
        # urgent_needs from survival state (hunger / exhaustion / injury, §19)
        needs = list(a.urgent_needs)
        if body["is_starving"] and "hunger" not in needs:
            needs.append("hunger")
        if body["is_exhausted"] and "exhaustion" not in needs:
            needs.append("exhaustion")
        ss.urgent_needs = needs
        ss.derived_signals = derive_self_signals(a, hunger_pressure=hunger, winter=winter)
        return ss

    def build_packet(self, scene: GroundTruthScene, a: AgentGT,
                     self_state: SelfStatePercept) -> PerceptionPacket:
        pos = (a.x, a.y)
        cr = a.comm_radius
        inspected = scene.inspected.get(a.id, set())
        camp = scene.camp_zone
        # §3.1 day/night vision: clock overrides the agent's base radius; campfire
        # / home proximity grants a night bonus.
        near_fire = self._near_campfire(scene, pos) or (
            a.home_location is not None and _dist(pos, a.home_location) <= 3.0)
        if scene.clock is not None:
            from agent_sdk.lived.world.clock import visual_radius
            # Honor the agent's own vision as the daytime radius (night halves it),
            # so env/individual vision is respected, not overwritten (§3.1).
            vr = visual_radius(scene.clock, near_campfire=near_fire, base=a.vision_radius)
        else:
            vr = a.vision_radius

        pkt = PerceptionPacket(
            perception_packet_id=f"pp:{scene.run_id}:{scene.turn}:{a.id}",
            run_id=scene.run_id, turn_id=scene.turn, agent_id=a.id, location=[a.x, a.y],
            self_state_percept_id=self_state.self_state_percept_id,
            season_status=scene.season, weather_status=scene.weather,
            body_status={"energy": a.energy, "hp": a.hp},
            inventory_status={"items": dict(a.inventory),
                              "remaining_capacity": self_state.remaining_capacity},
            home_status={"home_location": a.home_location, "storage": dict(a.home_storage)},
        )

        # visual (§12.1): resources / objects / agents / actions within vision
        pkt.visible_resources = [r for r in scene.resources if _dist(pos, (r.get("x"), r.get("y"))) <= vr]
        pkt.visible_objects = [o for o in scene.objects if _dist(pos, (o.get("x"), o.get("y"))) <= vr]
        pkt.visible_agents = [o.id for o in scene.agents
                              if o.id != a.id and _dist(pos, (o.x, o.y)) <= vr]
        pkt.visible_agent_actions = [
            ev for ev in scene.agent_actions
            if ev.get("actor") != a.id and _dist(pos, (ev.get("x"), ev.get("y"))) <= vr
        ]

        # speech (§12.2/§12.3/§12.4)
        for m in scene.messages:
            if m.get("speaker_id") == a.id:
                continue
            scope = m.get("visibility_scope") or m.get("scope") or "local"
            mloc = m.get("location") or (m.get("x"), m.get("y"))
            near = _dist(pos, mloc) <= cr
            if scope == "directed":
                if a.id in (m.get("target_agent_ids") or []):
                    pkt.heard_messages.append(m)
                elif near:
                    pkt.overheard_messages.append(m)
            elif scope == "camp":
                if camp and _dist(pos, (camp.get("x"), camp.get("y"))) <= float(camp.get("radius", 0)):
                    pkt.heard_messages.append(m)
            elif scope == "public":
                pkt.heard_messages.append(m)
            else:  # local
                if near:
                    pkt.heard_messages.append(m)

        # public marks/records (§12.5/§12.6/§13): nearby OR inspected
        def _seen(rec: Dict[str, Any]) -> bool:
            rid = rec.get("id")
            if rid in inspected:
                return True
            if "x" in rec and "y" in rec:
                return _dist(pos, (rec.get("x"), rec.get("y"))) <= vr
            return False

        pkt.public_marks_seen = [r for r in scene.public_marks if _seen(r)]
        pkt.public_records_seen = [r for r in scene.public_records if _seen(r)]
        # institution events (§12.7): observed (nearby) or inspected
        pkt.visible_mechanisms = [m for m in scene.mechanisms if _seen(m)]
        pkt.visible_episode_events = [e for e in scene.episode_events if _seen(e)]

        # §18 ongoing-action perception: visible agents who are mid-action
        for g in scene.agents:
            if g.id == a.id or _dist(pos, (g.x, g.y)) > vr:
                continue
            if g.current_action_status == "in_progress" and g.current_action_type:
                from agent_sdk.lived.core.actions import get_spec
                spec = get_spec(g.current_action_type)
                if spec is not None and spec.visibility_during_action in ("private", "instant"):
                    pass  # not observable as an ongoing action
                else:
                    pkt.ongoing_visible_actions.append({
                        "actor": g.id, "action_type": g.current_action_type,
                        "progress": round(g.action_progress, 3),
                        "visibility": spec.visibility_during_action if spec else "visible"})
                    pkt.visible_action_progress[g.id] = round(g.action_progress, 3)
                    pkt.agents_currently_busy.append(g.id)
            elif g.current_action_status == "idle":
                pkt.agents_available_nearby.append(g.id)
        # this-tick completed events within view
        for ev in scene.completed_events:
            if _dist(pos, ev.get("location") or (ev.get("x"), ev.get("y"))) <= vr:
                pkt.completed_visible_events.append(ev)

        if scene.clock is not None:
            pkt.time_of_day = scene.clock.time_of_day
        pkt.effective_visual_radius = float(vr)
        pkt.salient_changes = self._salient(pkt)
        return pkt

    def _near_campfire(self, scene: GroundTruthScene, pos) -> bool:
        for c in (scene.campfires or []):
            if _dist(pos, (c.get("x"), c.get("y"))) <= float(c.get("radius", 3)):
                return True
        camp = scene.camp_zone
        if camp and _dist(pos, (camp.get("x"), camp.get("y"))) <= float(camp.get("radius", 0)):
            return True
        return False

    def _salient(self, pkt: PerceptionPacket) -> List[str]:
        out: List[str] = []
        for ev in pkt.visible_agent_actions:
            act = str(ev.get("action", "")).lower()
            if any(k in act for k in ("withdraw", "accuse", "steal", "over_withdraw", "give", "teach")):
                out.append(f"{ev.get('actor')} {ev.get('action')}")
        if pkt.heard_messages:
            out.append(f"heard {len(pkt.heard_messages)} message(s)")
        return out

    # §14 structured appraisals from perceived actions / records
    def build_appraisals(self, scene: GroundTruthScene, a: AgentGT,
                         pkt: PerceptionPacket) -> List[PerceptionAppraisal]:
        out: List[PerceptionAppraisal] = []
        i = 0
        for ev in pkt.visible_agent_actions:
            act = str(ev.get("action", "")).lower()
            ap = PerceptionAppraisal(
                perception_appraisal_id=f"pa:{scene.run_id}:{scene.turn}:{a.id}:{i}",
                run_id=scene.run_id, turn_id=scene.turn, agent_id=a.id,
                perceived_event_type=act or "agent_action", actor=ev.get("actor"),
                target=ev.get("target"), resource=ev.get("resource"),
                location=[ev.get("x"), ev.get("y")], visibility="visual", observed_by=[a.id],
            )
            if any(k in act for k in ("over_withdraw", "withdraw")):
                ap.possible_bottleneck_signal = True
                ap.potential_fairness_violation = "over" in act
            if "accuse" in act:
                ap.potential_fairness_violation = True; ap.potential_rule_violation = True
            if any(k in act for k in ("attack", "steal", "raid")):
                ap.potential_threat = True
            if any(k in act for k in ("give", "help", "share")):
                ap.potential_help = True
            if "teach" in act:
                ap.potential_teaching_signal = True
            if "gather" in act:
                ap.possible_resource_opportunity = True
            ap.related_episode_id = ev.get("episode_id")
            ap.related_mechanism_id = ev.get("mechanism_id")
            out.append(ap)
            i += 1
        return out

    # -- top-level -----------------------------------------------------------
    def build_for(self, scene: GroundTruthScene, agent_id: str
                  ) -> Tuple[PerceptionPacket, SelfStatePercept, List[PerceptionAppraisal]]:
        a = next((g for g in scene.agents if g.id == agent_id), None)
        if a is None:
            raise KeyError(f"agent {agent_id!r} not in scene")
        ss = self.build_self_state(scene, a)
        pkt = self.build_packet(scene, a, ss)
        appraisals = self.build_appraisals(scene, a, pkt)
        return pkt, ss, appraisals

    def build_all(self, scene: GroundTruthScene
                  ) -> Dict[str, Tuple[PerceptionPacket, SelfStatePercept, List[PerceptionAppraisal]]]:
        return {g.id: self.build_for(scene, g.id) for g in scene.agents}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))
