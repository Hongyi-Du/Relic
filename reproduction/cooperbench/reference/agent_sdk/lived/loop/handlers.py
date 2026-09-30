"""Action execution v1 (Stage B §9 + energy addendum §5) — real handlers.

Each handler mutates the (env-agnostic) mutable scene — :class:`AgentGT` agents +
:class:`GroundTruthScene` resources/objects/messages — and returns an
:class:`ActionExecutionResult` carrying the EventRecord, energy accounting,
state delta, produced/consumed resources, optional :class:`SpeechEvent` +
listeners, memory-write specs, and a deterministic :class:`EventAppraisal` (§10).

Energy model (addendum §5): check ``min_energy_required`` (hard-mask unless the
action ``can_execute_when_exhausted`` / is emergency-survival), compute the
final cost from base + context modifiers, apply energy + fatigue deltas, charge
``failure_energy_cost`` on failure, and on energy < 0 apply an hp penalty +
exhaustion event.

Never updates stable persona (§9.5).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent_sdk.lived.core.actions import ActionCategory, ActionSpec, get_spec
from agent_sdk.lived.cognition.appraisal import EventAppraisal
from agent_sdk.lived.graphs.event_graph import EventRecord
from agent_sdk.lived.perceive.perception import AgentGT, GroundTruthScene
from agent_sdk.lived.core.schema import Intensity

STEP_SIZE = 5.0           # world units per movement "unit" (= one step)
MOVE_SPEED = STEP_SIZE    # world units per tick: v1 uses 1 step = 1 tick so a
#                           movement segment's tick-cost equals the distance it
#                           actually covers (no "wait full route, move one step").
DEFAULT_MAX_STEPS = 2     # steps a single move action advances (one segment)
EXHAUSTION_HP_PENALTY = 5.0


def _clamp(x: float, lo: float = 0.0, hi: float = 1e9) -> float:
    return max(lo, min(hi, float(x)))


def _dist(a, b) -> float:
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except (TypeError, ValueError, IndexError):
        return float("inf")


# --------------------------------------------------------------------------- #
# Result + speech event
# --------------------------------------------------------------------------- #
@dataclass
class SpeechEvent:
    message_id: str
    source_event_id: str
    speaker_id: str
    speech_act_type: str
    content_summary: str = ""
    target_agent_ids: List[str] = field(default_factory=list)
    visibility_scope: str = "local"          # local | directed | camp | public
    location: Optional[List[float]] = None
    is_public: bool = False
    related_plan_id: Optional[str] = None
    related_episode_id: Optional[str] = None
    related_event_ids: List[str] = field(default_factory=list)
    received_turn: int = 0

    def to_message(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionExecutionResult:
    action_type: str
    agent_id: str
    turn_id: int
    success: bool = True
    failure_reason: str = ""
    state_delta: Dict[str, Any] = field(default_factory=dict)
    consumed_resources: Dict[str, float] = field(default_factory=dict)
    produced_resources: Dict[str, float] = field(default_factory=dict)
    # energy accounting (addendum §6)
    energy_before: float = 0.0
    energy_after: float = 0.0
    base_energy_cost: float = 0.0
    final_energy_cost: float = 0.0
    failure_energy_cost: float = 0.0
    fatigue_before: float = 0.0
    fatigue_after: float = 0.0
    modifiers_applied: List[str] = field(default_factory=list)
    overexertion: bool = False
    hp_penalty: float = 0.0
    # outputs
    event: Optional[Dict[str, Any]] = None
    appraisal: Optional[Dict[str, Any]] = None
    speech_event: Optional[Dict[str, Any]] = None
    listeners: List[str] = field(default_factory=list)
    visibility: str = "private"
    memory_writes: List[Dict[str, Any]] = field(default_factory=list)
    # --- async duration / tick fields (§20) ----------------------------- #
    duration_ticks: int = 1
    action_start_tick: int = -1
    action_end_tick: int = -1
    scheduled_tick: int = -1
    completion_tick: int = -1
    action_status: str = "completed"
    interrupted: bool = False
    interrupt_reason: str = ""
    next_available_tick_before: int = -1
    next_available_tick_after: int = -1
    time_of_day_start: str = ""
    time_of_day_end: str = ""
    season: str = ""
    duration_modifiers_applied: List[str] = field(default_factory=list)
    visibility_during_action: str = "visible"
    completion_event_id: Optional[str] = None
    risk_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def action_duration(spec: ActionSpec, agent: AgentGT, params: Dict[str, Any], *,
                    clock: Any = None, move_speed: float = MOVE_SPEED) -> int:
    """Duration in ticks (§22). Sleep depends on the clock; movement advances a
    **single segment** (not the whole route); everything else uses
    ``base_duration_ticks``.

    Movement semantics (v1 §22.1): a move action is ONE segment of up to
    ``STEP_SIZE * max_steps`` units (what :func:`_h_movement` actually moves), so
    its duration is ``ceil(min(remaining_distance, segment) / move_speed)`` — it
    is capped to the segment and never scales with the full target distance.
    Otherwise the agent would block for the full-route time but only step one
    segment on completion. If it has not arrived, the next decision re-issues a
    move toward the same target (return_home / move_to / move_to_known_resource
    are regenerated each turn), so long routes are walked one observable segment
    at a time."""
    if spec is None:
        return 1
    if spec.name == "sleep":
        if clock is not None and clock.is_nighttime:
            return max(1, min(4, clock.ticks_until_daytime))
        return 2
    if spec.distance_based_duration:
        target = (params.get("target_location") or params.get("target_safe_location")
                  or agent.home_location)
        if target is not None:
            d = _dist((agent.x, agent.y), target)
            segment = STEP_SIZE * max(1.0, float(params.get("max_steps", DEFAULT_MAX_STEPS)))
            return max(1, math.ceil(min(d, segment) / move_speed))
        return 1
    return max(1, spec.base_duration_ticks)


# --------------------------------------------------------------------------- #
# Energy accounting
# --------------------------------------------------------------------------- #
def _movement_modifiers(spec: ActionSpec, agent: AgentGT, scene: GroundTruthScene
                        ) -> Tuple[float, List[str]]:
    extra = 0.0
    applied: List[str] = []
    load_ratio = (agent.current_load / agent.carrying_capacity) if agent.carrying_capacity else 0.0
    if "carrying_load" in spec.energy_cost_modifiers and load_ratio > 0.75:
        extra += 1.0; applied.append("carrying_load+1")
    if "weather" in spec.energy_cost_modifiers and scene.season in ("winter",):
        extra += 1.0; applied.append("winter+1")
    if "known_path" in spec.energy_cost_modifiers and agent.home_location is not None:
        extra -= 0.5; applied.append("known_path-0.5")
    if "fatigue" in spec.energy_cost_modifiers and float(agent.mood.get("fatigue", 0)) > 0.6:
        extra += 0.5; applied.append("fatigue+0.5")
    return extra, applied


def _apply_energy(agent: AgentGT, spec: ActionSpec, *, success: bool, units: float,
                  scene: GroundTruthScene, clock: Any = None,
                  override_energy: Optional[float] = None,
                  override_fatigue: Optional[float] = None) -> Dict[str, Any]:
    energy_before = float(agent.energy)
    fatigue_before = float(agent.mood.get("fatigue", 0.0))
    modifiers: List[str] = []
    fatigue_delta = spec.fatigue_delta
    if override_energy is not None:               # sleep: clock/location-based recovery
        final_cost = -float(override_energy)
        if override_fatigue is not None:
            fatigue_delta = float(override_fatigue)
    elif not success:
        final_cost = spec.failure_energy_cost
    elif spec.energy_recovery > 0:
        final_cost = -spec.energy_recovery          # restorative
    else:
        per_unit_extra, modifiers = _movement_modifiers(spec, agent, scene)
        u = max(1.0, units)
        final_cost = spec.base_energy_cost * u + per_unit_extra * (
            u if spec.category == ActionCategory.MOVEMENT else 1.0)
        # §3.2 night / winter-night movement multiplier
        if clock is not None and spec.category == ActionCategory.MOVEMENT:
            from agent_sdk.lived.world.clock import move_cost_multiplier
            mult = move_cost_multiplier(clock)
            if mult != 1.0:
                final_cost *= mult
                modifiers.append(f"night_move_x{mult}")
    agent.energy = energy_before - final_cost
    agent.mood["fatigue"] = _clamp(fatigue_before + fatigue_delta, 0.0, 1.0)
    overexertion = False
    hp_penalty = 0.0
    if agent.energy < 0:
        overexertion = True
        hp_penalty = EXHAUSTION_HP_PENALTY
        agent.hp = max(0.0, agent.hp - hp_penalty)
        agent.energy = 0.0
    agent.energy = min(agent.energy, agent.max_energy)
    return {
        "energy_before": energy_before, "energy_after": float(agent.energy),
        "base_energy_cost": spec.base_energy_cost, "final_energy_cost": final_cost,
        "failure_energy_cost": spec.failure_energy_cost,
        "fatigue_before": fatigue_before, "fatigue_after": float(agent.mood["fatigue"]),
        "modifiers_applied": modifiers, "overexertion": overexertion, "hp_penalty": hp_penalty,
    }


# --------------------------------------------------------------------------- #
# Movement helpers
# --------------------------------------------------------------------------- #
def _move_toward(agent: AgentGT, target, max_steps: float = 1.0) -> float:
    """Move agent toward target by up to max_steps movement units. Returns units."""
    if target is None or len(target) < 2:
        return 0.0
    d = _dist((agent.x, agent.y), target)
    if d == 0:
        return 0.0
    reach = min(d, STEP_SIZE * max(1.0, max_steps))
    ux, uy = (target[0] - agent.x) / d, (target[1] - agent.y) / d
    agent.x += ux * reach
    agent.y += uy * reach
    return reach / STEP_SIZE


def _food_items(inv: Dict[str, int]) -> List[str]:
    return [k for k in inv if any(t in k for t in ("food", "grain", "berry", "meat", "fish")) and inv[k] > 0]


# --------------------------------------------------------------------------- #
# Category handlers — each returns a partial dict (success/.../extras)
# --------------------------------------------------------------------------- #
def _h_movement(scene, agent, action_type, params) -> Dict[str, Any]:
    if action_type == "return_home":
        target = agent.home_location
        if target is None:
            return {"success": False, "failure_reason": "no_home", "units": 0.0}
    elif action_type == "move_to_known_resource":
        zone = params.get("resource_zone_id")
        target = (scene.inspected.get("_zones", {}) or {}).get(zone) if isinstance(scene.inspected.get("_zones"), dict) else None
        target = target or params.get("target_location")
        if target is None:
            # fall back to nearest visible resource of the type
            rt = params.get("target_resource_type")
            cands = [r for r in scene.resources if (rt is None or r.get("type") == rt)]
            target = [cands[0]["x"], cands[0]["y"]] if cands else None
        if target is None:
            return {"success": False, "failure_reason": "zone_unknown", "units": 0.0}
    elif action_type == "seek_safety":
        target = params.get("target_safe_location") or agent.home_location
        if target is None:
            return {"success": False, "failure_reason": "no_safe_location", "units": 0.0}
    else:  # move_to
        target = params.get("target_location")
        if target is None:
            return {"success": False, "failure_reason": "target_unknown", "units": 0.0}
    max_steps = float(params.get("max_steps", DEFAULT_MAX_STEPS))
    dist_before = _dist((agent.x, agent.y), target)
    units = _move_toward(agent, target, max_steps)   # advances ONE segment only
    arrived = _dist((agent.x, agent.y), target) <= 0.5
    return {"success": True, "units": max(1.0, units),
            "state_delta": {"x": agent.x, "y": agent.y, "arrived": arrived,
                            "distance_remaining": round(_dist((agent.x, agent.y), target), 3),
                            "distance_moved": round(dist_before - _dist((agent.x, agent.y), target), 3)}}


def _h_survival(scene, agent, action_type, params) -> Dict[str, Any]:
    if action_type == "eat_food":
        item = params.get("food_item_id")
        foods = _food_items(agent.inventory)
        item = item if item in agent.inventory else (foods[0] if foods else None)
        if not item:
            return {"success": False, "failure_reason": "no_food", "units": 1.0}
        amt = min(int(params.get("amount", 1)), agent.inventory.get(item, 0))
        agent.inventory[item] -= amt
        if agent.inventory[item] <= 0:
            del agent.inventory[item]
        agent.current_load = max(0.0, agent.current_load - amt)
        return {"success": True, "units": 1.0, "consumed": {item: amt},
                "state_delta": {"ate": item, "amount": amt}}
    if action_type == "sleep":
        return {"success": True, "units": 1.0, "state_delta": {"slept": True}, "is_sleep": True}
    # rest
    return {"success": True, "units": 1.0, "state_delta": {"rested": True}}


def _h_resource(scene, agent, action_type, params, *, clock=None) -> Dict[str, Any]:
    rid = params.get("resource_id")
    patch = next((r for r in scene.resources if r.get("id") == rid), None)
    if patch is None:
        return {"success": False, "failure_reason": "resource_not_accessible", "units": 1.0}
    if float(patch.get("amount", 0)) <= 0:
        return {"success": False, "failure_reason": "resource_depleted", "units": 1.0}
    remaining_cap = agent.carrying_capacity - agent.current_load
    if remaining_cap <= 0:
        return {"success": False, "failure_reason": "carrying_capacity_full", "units": 1.0}
    want = float(params.get("amount_requested", 1))
    # §3.3 night gather yield penalty
    if clock is not None:
        from agent_sdk.lived.world.clock import gather_yield_multiplier
        want = want * gather_yield_multiplier(clock)
    got = min(want, float(patch["amount"]), remaining_cap)
    patch["amount"] = float(patch["amount"]) - got
    rtype = patch.get("type", "resource")
    agent.inventory[rtype] = agent.inventory.get(rtype, 0) + int(got)
    agent.current_load += got
    bottleneck = (agent.carrying_capacity - agent.current_load) <= 0
    return {"success": True, "units": 1.0, "produced": {rtype: got},
            "state_delta": {"gathered": rtype, "amount": got, "capacity_full": bottleneck},
            "bottleneck": bottleneck,
            "memory_writes": [{"agent_id": agent.id, "summary": f"gathered {got:g} {rtype}",
                               "tags": ["gather", rtype], "related_resources": [rtype],
                               "salience": 0.3 + (0.3 if bottleneck else 0.0)}]}


def _h_craft(scene, agent, action_type, params) -> Dict[str, Any]:
    """attempt_craft: commit carried materials to a CraftSystem evaluation.
    Skills/knowledge/prototypes live on ``agent.extra`` so the handler stays
    env-agnostic; a usable carry prototype raises carrying_capacity (§22)."""
    from agent_sdk.lived.world.craft import (
        CraftProposal, CraftSystem, effective_capacity, propose_materials, test_prototype)
    fn = str(params.get("intended_function") or "")
    # materials at hand = carried inventory (+ home storage when at home)
    at_home = agent.home_location is not None and _dist((agent.x, agent.y), agent.home_location) <= 2.0
    pool = dict(agent.inventory)
    if at_home:
        for item, qty in agent.home_storage.items():
            pool[item] = pool.get(item, 0) + int(qty)
    mats = {m: int(c) for m, c in (params.get("materials") or {}).items() if int(c) > 0}
    if not mats:
        mats = propose_materials(pool, fn) or {}
    if not mats:
        return {"success": False, "failure_reason": "materials_missing", "units": 1.0}
    for m, c in mats.items():
        if int(pool.get(m, 0)) < c:
            return {"success": False, "failure_reason": "materials_missing", "units": 1.0}
    extra = agent.extra if isinstance(getattr(agent, "extra", None), dict) else {}
    skills = dict(extra.get("skills") or {})
    proposal = CraftProposal(
        proposal_id=f"prop:{agent.id}:{scene.turn}", agent_id=agent.id,
        intended_function=fn, materials=dict(mats),
        arrangement_summary=str(params.get("arrangement_summary", "")),
        related_wish_id=params.get("related_wish_id"))
    r = CraftSystem().evaluate(proposal, skills=skills,
                               knowledge=extra.get("material_knowledge"), tick=scene.turn)
    for m, c in r.materials_consumed.items():
        take = min(int(c), int(agent.inventory.get(m, 0)))
        if take:
            agent.inventory[m] = int(agent.inventory.get(m, 0)) - take
            if agent.inventory[m] <= 0:
                agent.inventory.pop(m, None)
            agent.current_load = max(0.0, agent.current_load - take)
        rest = int(c) - take                     # remainder from home storage
        if rest and at_home:
            agent.home_storage[m] = max(0, int(agent.home_storage.get(m, 0)) - rest)
            if agent.home_storage.get(m) == 0:
                agent.home_storage.pop(m, None)
    for s, g in r.skill_gains.items():
        skills[s] = min(1.0, float(skills.get(s, 0.0)) + g)
    extra["skills"] = skills
    if isinstance(getattr(agent, "extra", None), dict):
        agent.extra = extra
    out: Dict[str, Any] = {
        "success": r.success, "units": 1.0,
        "failure_reason": "" if r.success else "craft_scrap",
        "consumed": {m: float(c) for m, c in r.materials_consumed.items()},
        "prototype_result": "success" if r.success else "fail",
        "state_delta": {"crafted": r.result_type, "quality_score": r.quality_score,
                        "intended_function": fn, "materials": dict(mats)},
        "memory_writes": [{"agent_id": agent.id,
                           "summary": f"crafted {r.result_type} for {fn} (q={r.quality_score:.2f})",
                           "tags": ["craft", r.result_type], "related_resources": sorted(mats),
                           "salience": 0.6 if r.success else 0.5}]}
    if r.prototype is not None:
        proto = r.prototype
        test_prototype(proto, agent_id=agent.id, tick=scene.turn,
                       carried_grain=int(agent.inventory.get("grain", 1)) or 1)
        extra.setdefault("prototypes", []).append(proto)
        out["produced"] = {proto.object_type: 1.0}
        out["state_delta"]["prototype_id"] = proto.object_id
        out["state_delta"]["prototype_status"] = proto.status
        if proto.provides_carry:
            agent.carrying_capacity = effective_capacity(agent.carrying_capacity, [proto])
            out["state_delta"]["carrying_capacity"] = agent.carrying_capacity
        elif not proto.tested or proto.status == "broken":
            out["prototype_result"] = "partial" if r.success else "fail"
    return out


def _h_inventory(scene, agent, action_type, params) -> Dict[str, Any]:
    item = params.get("item_id")
    amt = int(params.get("amount", 1))
    at_home = agent.home_location is not None and _dist((agent.x, agent.y), agent.home_location) <= 2.0
    if action_type == "drop_item":
        if agent.inventory.get(item, 0) < amt:
            return {"success": False, "failure_reason": "item_missing", "units": 1.0}
        agent.inventory[item] -= amt
        if agent.inventory[item] <= 0:
            del agent.inventory[item]
        agent.current_load = max(0.0, agent.current_load - amt)
        scene.objects.append({"id": f"ground:{item}:{scene.turn}", "x": agent.x, "y": agent.y,
                              "type": item, "amount": amt})
        return {"success": True, "units": 1.0, "state_delta": {"dropped": item, "amount": amt}}
    if action_type == "store_item_home":
        if not at_home:
            return {"success": False, "failure_reason": "not_at_home", "units": 1.0}
        if agent.inventory.get(item, 0) < amt:
            return {"success": False, "failure_reason": "item_missing", "units": 1.0}
        agent.inventory[item] -= amt
        if agent.inventory[item] <= 0:
            del agent.inventory[item]
        agent.current_load = max(0.0, agent.current_load - amt)
        agent.home_storage[item] = agent.home_storage.get(item, 0) + amt
        return {"success": True, "units": 1.0, "produced": {item: amt},
                "state_delta": {"stored": item, "amount": amt}}
    if action_type == "retrieve_item_home":
        if not at_home:
            return {"success": False, "failure_reason": "not_at_home", "units": 1.0}
        if agent.home_storage.get(item, 0) < amt:
            return {"success": False, "failure_reason": "item_missing", "units": 1.0}
        if (agent.carrying_capacity - agent.current_load) < amt:
            return {"success": False, "failure_reason": "carrying_capacity_full", "units": 1.0}
        agent.home_storage[item] -= amt
        agent.inventory[item] = agent.inventory.get(item, 0) + amt
        agent.current_load += amt
        return {"success": True, "units": 1.0, "state_delta": {"retrieved": item, "amount": amt}}
    # transfer_item_to_agent
    tgt_id = params.get("target_agent_id")
    tgt = next((g for g in scene.agents if g.id == tgt_id), None)
    if tgt is None or _dist((agent.x, agent.y), (tgt.x, tgt.y)) > agent.comm_radius:
        return {"success": False, "failure_reason": "target_not_nearby", "units": 1.0}
    if agent.inventory.get(item, 0) < amt:
        return {"success": False, "failure_reason": "item_missing", "units": 1.0}
    agent.inventory[item] -= amt
    if agent.inventory[item] <= 0:
        del agent.inventory[item]
    agent.current_load = max(0.0, agent.current_load - amt)
    tgt.inventory[item] = tgt.inventory.get(item, 0) + amt
    tgt.current_load += amt
    return {"success": True, "units": 1.0, "consumed": {item: amt},
            "state_delta": {"gave": item, "amount": amt, "to": tgt_id},
            "social_support_to": tgt_id, "target": tgt_id,
            "memory_writes": [
                {"agent_id": agent.id, "summary": f"gave {amt} {item} to {tgt_id}",
                 "tags": ["give"], "related_agents": [tgt_id], "salience": 0.4},
                {"agent_id": tgt_id, "summary": f"received {amt} {item} from {agent.id}",
                 "tags": ["received", "gratitude"], "related_agents": [agent.id],
                 "memory_type": "social_memory", "salience": 0.4}]}


def _h_inspection(scene, agent, action_type, params) -> Dict[str, Any]:
    key = {"inspect_resource": "resource_id", "inspect_object": "object_id",
           "inspect_agent_visible_state": "target_agent_id",
           "inspect_public_record": "record_id"}.get(action_type)
    target_id = params.get(key) if key else None
    # familiarity count: the extractor diminishes information_gain by it
    counts = scene.inspected.setdefault("_counts", {})
    if action_type == "inspect_area":
        counts["_area"] = int(counts.get("_area", 0)) + 1
        return {"success": True, "units": 1.0, "knowledge_gain": 0.3,
                "state_delta": {"inspected_area": True}}
    if target_id is None:
        return {"success": False, "failure_reason": "not_visible", "units": 1.0}
    counts[str(target_id)] = int(counts.get(str(target_id), 0)) + 1
    summary = f"inspected {target_id}"
    mw = [{"agent_id": agent.id, "summary": summary, "tags": ["inspect"],
           "memory_type": "semantic_memory", "salience": 0.2}]
    return {"success": True, "units": 1.0, "knowledge_gain": 0.4,
            "state_delta": {"inspected": target_id}, "memory_writes": mw}


def _h_speech(scene, agent, action_type, params, *, turn, event_id) -> Dict[str, Any]:
    scope = {"comm_local": "local", "camp_announce": "camp"}.get(action_type, "directed")
    targets = params.get("target_agent_ids") or (
        [params.get("target_agent_id")] if params.get("target_agent_id") else [])
    targets = [t for t in targets if t]
    # determine listeners by visibility (perception delivery is read-side; we
    # compute the audience + post the SpeechEvent into the scene for perception)
    listeners: List[str] = []
    for g in scene.agents:
        if g.id == agent.id:
            continue
        d = _dist((agent.x, agent.y), (g.x, g.y))
        if scope == "directed":
            if g.id in targets or d <= agent.comm_radius:
                listeners.append(g.id)
        elif scope == "camp":
            camp = scene.camp_zone
            if camp and _dist((g.x, g.y), (camp.get("x"), camp.get("y"))) <= float(camp.get("radius", 0)):
                listeners.append(g.id)
        else:  # local
            if d <= agent.comm_radius:
                listeners.append(g.id)
    # directed actions require the target nearby
    if scope == "directed" and targets and not any(t in listeners for t in targets):
        return {"success": False, "failure_reason": "target_not_nearby", "units": 1.0}
    if scope == "local" and not listeners:
        return {"success": False, "failure_reason": "no_listeners", "units": 1.0}
    if scope == "camp" and not scene.camp_zone:
        return {"success": False, "failure_reason": "not_in_camp", "units": 1.0}

    se = SpeechEvent(
        message_id=f"msg:{agent.id}:{turn}:{action_type}", source_event_id=event_id,
        speaker_id=agent.id, speech_act_type=action_type,
        content_summary=str(params.get("content_summary", "")),
        target_agent_ids=targets, visibility_scope=scope, location=[agent.x, agent.y],
        is_public=(scope in ("camp", "public")),
        related_plan_id=params.get("related_plan_id"),
        related_episode_id=params.get("related_episode_id"),
        related_event_ids=list(params.get("related_event_ids") or []),
        received_turn=turn,
    )
    scene.messages.append(se.to_message())     # perception layer delivers it
    out = {"success": True, "units": 1.0, "speech_event": se, "listeners": listeners,
           "communication_sent": True, "visibility": "public" if se.is_public else "witnessed",
           "state_delta": {"spoke": action_type, "listeners": listeners}}
    # listener memories of having heard it (§11.3)
    mw = [{"agent_id": lid, "summary": f"heard {agent.id}: {se.content_summary or action_type}",
           "tags": ["heard", action_type], "related_agents": [agent.id],
           "memory_type": "social_memory", "salience": 0.2,
           "source_event_id": se.message_id} for lid in listeners]
    if action_type == "ask_help":
        out["social_support_to"] = targets[0] if targets else None
    if action_type == "promise_action":
        out["promise_created"] = True
        # prospective memory for speaker + listener
        mw.append({"agent_id": agent.id, "summary": f"promised {params.get('promised_action_type','help')} to {targets[0] if targets else '?'}",
                   "tags": ["promise", "pending"], "memory_type": "prospective_memory",
                   "related_agents": targets, "salience": 0.5})
    out["memory_writes"] = mw
    out["target"] = targets[0] if targets else None
    return out


def _h_wait(scene, agent, action_type, params) -> Dict[str, Any]:
    return {"success": True, "units": 1.0, "state_delta": {"waited": True,
            "reason": params.get("reason", "")}}


_DISPATCH = {
    ActionCategory.MOVEMENT: _h_movement,
    ActionCategory.SURVIVAL: _h_survival,
    ActionCategory.RESOURCE: _h_resource,
    ActionCategory.INVENTORY: _h_inventory,
    ActionCategory.INSPECTION: _h_inspection,
    ActionCategory.WAIT: _h_wait,
}


# --------------------------------------------------------------------------- #
# Public execute
# --------------------------------------------------------------------------- #
def execute_action(scene: GroundTruthScene, agent: AgentGT, action_type: str,
                   params: Optional[Dict[str, Any]] = None, *, turn: int,
                   run_id: str = "run", clock: Any = None) -> ActionExecutionResult:
    params = dict(params or {})
    spec = get_spec(action_type)
    res = ActionExecutionResult(action_type=action_type, agent_id=agent.id, turn_id=turn)
    if spec is None:
        res.success = False; res.failure_reason = "unknown_action"; res.action_status = "failed"
        return res
    res.duration_ticks = action_duration(spec, agent, params, clock=clock)
    res.visibility_during_action = spec.visibility_during_action
    res.completion_event_id = f"{spec.completion_event}:{agent.id}:{turn}"
    if clock is not None:
        res.time_of_day_start = clock.time_of_day; res.season = clock.season

    # param validation (§14.1.2)
    perr = spec.validate_params(params)
    if perr:
        res.success = False; res.failure_reason = perr[0]; res.action_status = "failed"
        res.event = _event(scene, agent, action_type, turn, False).to_dict()
        return res

    # energy hard mask on execution (addendum §5)
    if agent.energy < spec.min_energy_required and not (spec.can_execute_when_exhausted
                                                        or spec.is_emergency_survival):
        res.success = False; res.failure_reason = "too_exhausted"; res.action_status = "failed"
        e = _apply_energy(agent, spec, success=False, units=1.0, scene=scene, clock=clock)
        _fill_energy(res, e); res.event = _event(scene, agent, action_type, turn, False).to_dict()
        res.appraisal = _appraise(res, agent).to_dict()
        return res

    event_id = f"ev:{agent.id}:{turn}:{action_type}"
    if spec.category == ActionCategory.SPEECH or spec.is_speech:
        out = _h_speech(scene, agent, action_type, params, turn=turn, event_id=event_id)
    elif action_type == "attempt_craft":
        out = _h_craft(scene, agent, action_type, params)
    elif spec.category == ActionCategory.RESOURCE:
        out = _h_resource(scene, agent, action_type, params, clock=clock)
    else:
        out = _DISPATCH[spec.category](scene, agent, action_type, params)

    res.success = bool(out.get("success", True))
    res.failure_reason = out.get("failure_reason", "")
    res.state_delta = out.get("state_delta", {})
    res.consumed_resources = out.get("consumed", {})
    res.produced_resources = out.get("produced", {})
    res.memory_writes = out.get("memory_writes", [])
    res.visibility = out.get("visibility", "private")
    if out.get("speech_event") is not None:
        res.speech_event = out["speech_event"].to_message()
        res.listeners = out.get("listeners", [])

    # §6.2 sleep recovery is clock/location dependent (per-tick × duration)
    override_e = override_f = None
    if out.get("is_sleep") and res.success:
        from agent_sdk.lived.world.clock import sleep_recovery, WorldClock
        at_home = agent.home_location is not None and _dist((agent.x, agent.y), agent.home_location) <= 3.0
        near_fire = _near_fire(scene, (agent.x, agent.y))
        ck = clock or WorldClock()
        rec = sleep_recovery(ck, at_home=at_home, near_campfire=near_fire)
        override_e = rec["energy"] * res.duration_ticks
        override_f = rec["fatigue"] * res.duration_ticks
        res.risk_flags = [f"sleep_risk_{rec['risk']}"] if rec["risk"] in ("medium", "high") else []

    e = _apply_energy(agent, spec, success=res.success, units=float(out.get("units", 1.0)),
                      scene=scene, clock=clock, override_energy=override_e, override_fatigue=override_f)
    _fill_energy(res, e)
    res.action_status = "completed" if res.success else "failed"
    if clock is not None:
        res.time_of_day_end = clock.time_of_day
    res.event = _event(scene, agent, action_type, turn, res.success, event_id=event_id).to_dict()
    res.appraisal = _appraise(res, agent, extras=out).to_dict()
    return res


def _near_fire(scene: GroundTruthScene, pos) -> bool:
    for c in (getattr(scene, "campfires", None) or []):
        if _dist(pos, (c.get("x"), c.get("y"))) <= float(c.get("radius", 3)):
            return True
    camp = scene.camp_zone
    return bool(camp and _dist(pos, (camp.get("x"), camp.get("y"))) <= float(camp.get("radius", 0)))


def _fill_energy(res: ActionExecutionResult, e: Dict[str, Any]) -> None:
    for k, v in e.items():
        setattr(res, k, v)


def _event(scene, agent, action_type, turn, success, event_id: str = "") -> EventRecord:
    return EventRecord(event_id=event_id or f"ev:{agent.id}:{turn}:{action_type}", timestamp=turn,
                       actor=agent.id, action_type=action_type, success=success,
                       summary=f"{agent.id} {action_type} ({'ok' if success else 'fail'})")


def _appraise(res: ActionExecutionResult, agent: AgentGT, extras: Optional[Dict[str, Any]] = None
              ) -> EventAppraisal:
    extras = extras or {}
    energy_delta = res.energy_after - res.energy_before
    hunger = float(agent.mood.get("hunger_pressure", 0.0))
    survival_delta = 0.0
    if res.action_type in ("eat_food", "rest") and res.success:
        survival_delta = 0.3 if hunger > 0.5 else 0.1
    resource_delta = sum(res.produced_resources.values()) - sum(res.consumed_resources.values())
    intensity = Intensity.MAJOR if (res.overexertion or extras.get("bottleneck")) else (
        Intensity.MODERATE if res.success and (resource_delta or extras.get("communication_sent")) else Intensity.MINOR)
    ap = EventAppraisal(
        actor=agent.id, action_type=res.action_type, turn=res.turn_id,
        target=extras.get("target"), success=res.success,
        event_id=(res.event or {}).get("event_id"),
        survival_delta=survival_delta, energy_delta=energy_delta / 50.0,
        resource_delta=_clamp(resource_delta / 5.0, 0.0, 1.0),
        fatigue_delta=res.fatigue_after - res.fatigue_before,
        risk_realized=1.0 if res.overexertion else 0.0,
        knowledge_gain=float(extras.get("knowledge_gain", 0.0)),
        prototype_result=str(extras.get("prototype_result", "") or ""),
        possible_bottleneck_signal=bool(extras.get("bottleneck")),
        social_support_received=0.3 if extras.get("social_support_to") else 0.0,
        communication_sent=bool(extras.get("communication_sent")),
        promise_created=bool(extras.get("promise_created")),
        exhaustion_event=res.overexertion,
        emotional_valence=(-0.3 if not res.success else (0.2 if survival_delta or resource_delta else 0.0)),
        intensity=intensity, observed_by=list(res.listeners), visibility=res.visibility,
        related_plan_id=(extras.get("related_plan_id")),
    )
    return ap
