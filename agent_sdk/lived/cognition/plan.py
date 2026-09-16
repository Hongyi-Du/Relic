"""PlanGraph + PlanMonitor (Stage C1, spec Part 1).

Per-agent long-horizon plans (FoodSecurity / ReturnHome / Recover /
MaterialProblemSolving) and a per-tick :class:`PlanMonitor` that — WITHOUT an LLM
and WITHOUT picking actions — advances plan steps, records *why* a step is blocked
(§6 blocked reasons), and emits ``wish_triggers`` / ``reflection_triggers`` +
``candidate_action_hints`` / ``plan_value_modifiers`` for the CandidatePool /
FeatureExtractor / Reflection to consume.

The headline chain it drives (§25): FoodSecurityPlan active → gather grain →
carrying capacity full while grain remains → repeated ``blocked_by_carrying_capacity``
→ carrying_capacity wish trigger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class PlanType:
    FOOD_SECURITY = "food_security"
    RETURN_HOME = "return_home"
    RECOVER = "recover"
    MATERIAL_PROBLEM_SOLVING = "material_problem_solving"


class PlanStatus:
    ACTIVE = "active"; PAUSED = "paused"; COMPLETED = "completed"
    FAILED = "failed"; OBSOLETE = "obsolete"


class StepStatus:
    PENDING = "pending"; READY = "ready"; ACTIVE = "active"; COMPLETED = "completed"
    BLOCKED = "blocked"; FAILED = "failed"; SKIPPED = "skipped"; OBSOLETE = "obsolete"


# §6 blocked reasons
class Blocked:
    CARRYING_CAPACITY = "blocked_by_carrying_capacity"
    NO_FOOD_VISIBLE = "blocked_by_no_food_visible"
    FOOD_TOO_FAR = "blocked_by_food_too_far"
    LOW_ENERGY = "blocked_by_low_energy"
    HIGH_FATIGUE = "blocked_by_high_fatigue"
    LOW_SATIETY = "blocked_by_low_satiety"
    NO_STORAGE = "blocked_by_no_storage"
    STORAGE_FULL = "blocked_by_storage_full"
    UNKNOWN_MATERIAL = "blocked_by_unknown_material"
    MISSING_BINDING = "blocked_by_missing_binding_material"
    LACK_SKILL = "blocked_by_lack_skill"
    NIGHT_RISK = "blocked_by_night_risk"
    PATH_UNKNOWN = "blocked_by_path_unknown"
    RESOURCE_DEPLETED = "blocked_by_resource_depleted"


WISH_TRIGGER_THRESHOLD = 2          # blocked_count to raise a wish (§6)


@dataclass
class PlanStep:
    step_id: str
    plan_id: str
    description: str
    required_action_types: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    completion_conditions: List[str] = field(default_factory=list)
    failure_conditions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: str = StepStatus.PENDING
    target_location: Optional[Any] = None
    target_resource: Optional[str] = None
    linked_episode_id: Optional[str] = None
    linked_wish_id: Optional[str] = None
    blocked_reason: str = ""
    blocked_count: int = 0
    attempt_count: int = 0
    last_attempt_tick: int = -1
    progress: float = 0.0
    policy_alignment_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Plan:
    plan_id: str
    agent_id: str
    plan_type: str
    goal: str
    motivation: str = ""
    priority: float = 0.5
    urgency: float = 0.5
    horizon_ticks: int = 60
    status: str = PlanStatus.ACTIVE
    created_tick: int = 0
    last_review_tick: int = 0
    current_step_ids: List[str] = field(default_factory=list)
    steps: List[PlanStep] = field(default_factory=list)
    blocked_reasons: List[str] = field(default_factory=list)
    related_episode_ids: List[str] = field(default_factory=list)
    related_wish_ids: List[str] = field(default_factory=list)
    related_resource_ids: List[str] = field(default_factory=list)
    progress_score: float = 0.0
    success_condition: str = ""
    failure_condition: str = ""
    fallback_plan_id: Optional[str] = None
    review_triggers: List[str] = field(default_factory=list)

    def step(self, sid: str) -> Optional[PlanStep]:
        return next((s for s in self.steps if s.step_id == sid), None)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "steps"}
        d["steps"] = [s.to_dict() for s in self.steps]
        return d


# --------------------------------------------------------------------------- #
# Plan builders (§4)
# --------------------------------------------------------------------------- #
def _mk_step(plan_id: str, n: str, desc: str, actions: List[str], **kw) -> PlanStep:
    return PlanStep(step_id=f"{plan_id}:{n}", plan_id=plan_id, description=desc,
                    required_action_types=actions, **kw)


def make_food_security_plan(agent_id: str, tick: int) -> Plan:
    pid = f"plan:{agent_id}:food_security:{tick}"
    p = Plan(plan_id=pid, agent_id=agent_id, plan_type=PlanType.FOOD_SECURITY,
             goal="secure food for the near future (winter buffer)",
             motivation="satiety / winter / low home-storage pressure",
             priority=0.8, urgency=0.7, created_tick=tick,
             success_condition="home_storage food >= reserve_target",
             review_triggers=["satiety_low", "winter", "storage_low"])
    p.steps = [
        _mk_step(pid, "locate_food", "find a food source", ["inspect_area", "search_known_resource"]),
        _mk_step(pid, "move_to_food", "move to the food source", ["move_to_known_resource", "move_to"]),
        _mk_step(pid, "gather_food", "gather grain", ["gather_resource"]),
        _mk_step(pid, "return", "carry food home when full", ["return_home"]),
        _mk_step(pid, "store", "store food at home", ["store_item_home"]),
    ]
    p.current_step_ids = [p.steps[2].step_id]   # gather is the live step in the meadow
    return p


def make_return_home_plan(agent_id: str, tick: int) -> Plan:
    pid = f"plan:{agent_id}:return_home:{tick}"
    p = Plan(plan_id=pid, agent_id=agent_id, plan_type=PlanType.RETURN_HOME,
             goal="return to home / safe location", motivation="night / fatigue / carrying valuables",
             priority=0.7, urgency=0.8, created_tick=tick, horizon_ticks=12)
    p.steps = [
        _mk_step(pid, "path", "identify home path", ["return_home"]),
        _mk_step(pid, "move", "move toward home", ["return_home"]),
        _mk_step(pid, "store", "store carried resources", ["store_item_home"]),
        _mk_step(pid, "rest", "rest / sleep if needed", ["rest", "sleep"]),
    ]
    p.current_step_ids = [p.steps[1].step_id]
    return p


def make_recover_plan(agent_id: str, tick: int) -> Plan:
    pid = f"plan:{agent_id}:recover:{tick}"
    p = Plan(plan_id=pid, agent_id=agent_id, plan_type=PlanType.RECOVER,
             goal="recover energy, lower fatigue, avoid collapse",
             motivation="low energy / high fatigue / overexertion", priority=0.85, urgency=0.85,
             created_tick=tick, horizon_ticks=8)
    p.steps = [
        _mk_step(pid, "safe", "move to a safe rest spot", ["return_home", "seek_safety"]),
        _mk_step(pid, "rest", "rest or sleep", ["rest", "sleep"]),
    ]
    p.current_step_ids = [p.steps[1].step_id]
    return p


def make_material_problem_solving_plan(agent_id: str, tick: int, *, wish_id: str,
                                       target_material: str = "reed") -> Plan:
    pid = f"plan:{agent_id}:material:{tick}"
    p = Plan(plan_id=pid, agent_id=agent_id, plan_type=PlanType.MATERIAL_PROBLEM_SOLVING,
             goal="solve a physical bottleneck by crafting an affordance",
             motivation="grounded wish from a repeated bottleneck",
             priority=0.75, urgency=0.6, created_tick=tick, related_wish_ids=[wish_id])
    p.steps = [
        _mk_step(pid, "find_mat", "find needed materials", ["search_known_resource", "inspect_area"],
                 target_resource=target_material),
        _mk_step(pid, "gather_mat", "gather materials", ["gather_resource", "move_to_known_resource"],
                 target_resource=target_material),
        _mk_step(pid, "inspect_mat", "inspect material properties", ["inspect_material_properties"]),
        _mk_step(pid, "craft", "start a craft session", ["start_craft_session", "submit_craft_attempt"]),
        _mk_step(pid, "test", "test the prototype", ["test_prototype"]),
        _mk_step(pid, "resume", "return to the food-security plan", []),
    ]
    p.current_step_ids = [p.steps[0].step_id]
    p.steps[0].linked_wish_id = wish_id
    return p


# --------------------------------------------------------------------------- #
# PlanMonitor (§5)
# --------------------------------------------------------------------------- #
@dataclass
class PlanMonitorResult:
    plan_updates: List[Dict[str, Any]] = field(default_factory=list)
    ready_steps: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    blocked_steps: List[Dict[str, Any]] = field(default_factory=list)
    candidate_action_hints: List[str] = field(default_factory=list)
    plan_value_modifiers: Dict[str, float] = field(default_factory=dict)
    reflection_triggers: List[str] = field(default_factory=list)
    wish_triggers: List[Dict[str, Any]] = field(default_factory=list)


class PlanMonitor:
    """One per run; holds each agent's plans + blocked-counters. Runs every tick."""

    def __init__(self) -> None:
        self.plans: Dict[str, List[Plan]] = {}             # agent_id -> plans
        self._blocked: Dict[str, Dict[str, int]] = {}      # agent_id -> reason -> count

    def plans_for(self, agent_id: str) -> List[Plan]:
        return self.plans.setdefault(agent_id, [])

    def active_plans(self, agent_id: str) -> List[Plan]:
        return [p for p in self.plans_for(agent_id) if p.status == PlanStatus.ACTIVE]

    def add_plan(self, plan: Plan) -> Plan:
        self.plans_for(plan.agent_id).append(plan)
        return plan

    def has_plan(self, agent_id: str, plan_type: str) -> bool:
        return any(p.plan_type == plan_type and p.status == PlanStatus.ACTIVE
                   for p in self.plans_for(agent_id))

    def _bump_block(self, agent_id: str, reason: str) -> int:
        d = self._blocked.setdefault(agent_id, {})
        d[reason] = d.get(reason, 0) + 1
        return d[reason]

    def _clear_block(self, agent_id: str, reason: str) -> None:
        self._blocked.get(agent_id, {}).pop(reason, None)

    def update(self, *, agent_id: str, self_state: Any, packet: Any, inventory: Dict[str, int],
               visible_resources: List[Dict[str, Any]], known_resources: List[str], tick: int,
               winter: bool = False) -> PlanMonitorResult:
        res = PlanMonitorResult()
        ss = self_state
        load = float(getattr(ss, "current_load", 0.0))
        cap = float(getattr(ss, "carrying_capacity", 0.0) or 0.0)
        remaining = float(getattr(ss, "remaining_capacity", cap - load))
        satiety = float(getattr(ss, "satiety", 100.0))
        fatigue_zone = str(getattr(ss, "fatigue_zone", "normal"))
        energy = float(getattr(ss, "energy", 100.0))
        is_hungry = bool(getattr(ss, "is_hungry", satiety < 60))
        home_food = sum(v for k, v in (getattr(ss, "home_storage_summary", {}) or {}).items()
                        if any(t in str(k).lower() for t in ("grain", "food", "berry", "tomato", "beef")))
        food_visible = [r for r in visible_resources if _is_food_type(r.get("type"))]

        # --- (1) FoodSecurityPlan: create when food pressure + a food source -- #
        if (is_hungry or winter or home_food < 10) and (food_visible or "grain" in known_resources):
            if not self.has_plan(agent_id, PlanType.FOOD_SECURITY):
                p = self.add_plan(make_food_security_plan(agent_id, tick))
                res.plan_updates.append({"plan_id": p.plan_id, "change": "created", "type": p.plan_type})
            res.plan_value_modifiers["gather_resource"] = res.plan_value_modifiers.get("gather_resource", 0) + 0.3
            res.candidate_action_hints += ["gather_resource", "store_item_home"]

        # --- (2) carrying bottleneck -> blocked_by_carrying_capacity -> wish --- #
        food_security = self.has_plan(agent_id, PlanType.FOOD_SECURITY)
        carrying_full = cap > 0 and remaining <= 0
        # "food remains" = visible now OR a known food source nearby (§6)
        food_remains = bool(food_visible) or any(_is_food_type(k) for k in known_resources)
        if food_security and carrying_full and food_remains:
            n = self._bump_block(agent_id, Blocked.CARRYING_CAPACITY)
            res.blocked_steps.append({"plan_type": PlanType.FOOD_SECURITY,
                                      "blocked_reason": Blocked.CARRYING_CAPACITY, "blocked_count": n})
            res.candidate_action_hints += ["return_home", "store_item_home", "drop_item"]
            for p in self.active_plans(agent_id):
                if p.plan_type == PlanType.FOOD_SECURITY and Blocked.CARRYING_CAPACITY not in p.blocked_reasons:
                    p.blocked_reasons.append(Blocked.CARRYING_CAPACITY)
            if n >= WISH_TRIGGER_THRESHOLD and not self.has_plan(agent_id, PlanType.MATERIAL_PROBLEM_SOLVING):
                res.wish_triggers.append({
                    "need_type": "carrying_capacity", "blocked_reason": Blocked.CARRYING_CAPACITY,
                    "blocked_count": n, "evidence": _evidence(packet, ss),
                    "target_resource_type": "reed",
                    "related_plan_type": PlanType.FOOD_SECURITY})
                res.reflection_triggers.append("carrying_bottleneck")
        else:
            self._clear_block(agent_id, Blocked.CARRYING_CAPACITY)

        # --- (2b) MaterialProblemSolvingPlan: value the current step's actions #
        # The plan (created by the wish consumer) shapes PCBSP via PlanValue:
        # craft when the carried materials cover a recipe, otherwise gather the
        # missing materials. Judged env-agnostically via MATERIAL_REGISTRY.
        mat_plans = [p for p in self.active_plans(agent_id)
                     if p.plan_type == PlanType.MATERIAL_PROBLEM_SOLVING]
        if mat_plans:
            from agent_sdk.lived.world.craft import craftable_functions
            mp = mat_plans[0]
            owned = dict(inventory or {})
            for k, v in (getattr(ss, "home_storage_summary", {}) or {}).items():
                owned[k] = owned.get(k, 0) + int(v)
            if craftable_functions(owned):
                res.plan_value_modifiers["attempt_craft"] = (
                    res.plan_value_modifiers.get("attempt_craft", 0) + float(mp.priority))
                res.candidate_action_hints += ["attempt_craft"]
            else:
                res.plan_value_modifiers["gather_resource"] = (
                    res.plan_value_modifiers.get("gather_resource", 0) + 0.4)
                res.candidate_action_hints += ["gather_resource"]

        # --- (3) storage bottleneck -> storage_capacity wish (§26) ------------ #
        if food_security and home_food >= 8 and float(getattr(ss, "remaining_capacity", 1)) >= 0:
            # heuristic "brought food home repeatedly" — once storage is sizeable
            pass  # storage wish kept light in v1 (see §26); emitted by controller if needed

        # --- (4) ReturnHomePlan: night / fatigue / carrying valuables --------- #
        night = bool(getattr(packet, "time_of_day", "") == "night")
        if (night or fatigue_zone in ("high", "overexertion") or load > 0.5 * cap if cap else night):
            if not self.has_plan(agent_id, PlanType.RETURN_HOME):
                p = self.add_plan(make_return_home_plan(agent_id, tick))
                res.plan_updates.append({"plan_id": p.plan_id, "change": "created", "type": p.plan_type})
            res.plan_value_modifiers["return_home"] = res.plan_value_modifiers.get("return_home", 0) + 0.3

        # --- (5) RecoverPlan: low energy / high fatigue / forced rest --------- #
        forced = bool(getattr(ss, "is_collapsed", False)) or int(getattr(ss, "forced_rest_until_tick", -1)) > tick
        if energy < 25 or fatigue_zone in ("overexertion", "collapse") or forced:
            if not self.has_plan(agent_id, PlanType.RECOVER):
                p = self.add_plan(make_recover_plan(agent_id, tick))
                res.plan_updates.append({"plan_id": p.plan_id, "change": "created", "type": p.plan_type})
            res.plan_value_modifiers["rest"] = res.plan_value_modifiers.get("rest", 0) + 0.4
            res.plan_value_modifiers["sleep"] = res.plan_value_modifiers.get("sleep", 0) + 0.4

        for p in self.active_plans(agent_id):
            p.last_review_tick = tick
        return res


def _is_food_type(t: Optional[str]) -> bool:
    return bool(t) and any(x in str(t).lower() for x in ("grain", "food", "berry", "tomato", "fruit", "cow"))


def _evidence(packet: Any, ss: Any) -> List[str]:
    ids: List[str] = []
    pid = getattr(packet, "perception_packet_id", None)
    if pid:
        ids.append(pid)
    sid = getattr(ss, "self_state_percept_id", None)
    if sid:
        ids.append(sid)
    for r in (getattr(packet, "visible_resources", []) or [])[:3]:
        if r.get("id"):
            ids.append(str(r["id"]))
    return ids
