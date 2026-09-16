"""Action Registry v1 (Stage B §4 + energy addendum) — all REAL, no placeholders.

Every action here is executable (a handler in ``handlers.py``), produces a log +
event appraisal, and updates state (Stage B §0). Wish is NOT an action — it is a
Reflection-stage cognitive output, kept only as the :class:`WishCandidate` data
structure (§2). Speech IS an Action-Layer action (§3).

Each :class:`ActionSpec` carries an explicit energy model (energy addendum §1):
``base_energy_cost / min_energy_required / failure_energy_cost / fatigue_delta /
energy_cost_modifiers / can_execute_when_exhausted / energy_recovery /
hp_risk_if_overexerted`` — so cost is defined by the template, not only inferred
by the FeatureExtractor (which may *read* these fields).

Env-agnostic: specs are plain data; handlers operate on a mutable scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ActionCategory(str, Enum):
    MOVEMENT = "movement"
    SURVIVAL = "survival"
    RESOURCE = "resource"
    INVENTORY = "inventory"
    INSPECTION = "inspection"
    SPEECH = "speech"
    WAIT = "wait"


@dataclass(frozen=True)
class ActionSpec:
    name: str
    category: ActionCategory
    required_params: Tuple[str, ...] = ()
    optional_params: Tuple[str, ...] = ()
    preconditions: Tuple[str, ...] = ()
    failure_modes: Tuple[str, ...] = ()
    # --- energy model (addendum §1) ------------------------------------- #
    base_energy_cost: float = 0.0
    min_energy_required: float = 0.0
    failure_energy_cost: float = 0.0
    fatigue_delta: float = 0.0
    energy_cost_modifiers: Tuple[str, ...] = ()
    can_execute_when_exhausted: bool = False
    energy_recovery: float = 0.0
    hp_risk_if_overexerted: float = 0.0
    # --- duration model (async infra §5 / §22) -------------------------- #
    base_duration_ticks: int = 1
    # movement = ONE segment: ceil(min(remaining_dist, STEP_SIZE*max_steps)/speed),
    # capped to the segment, NOT the whole route (handlers.action_duration §22.1).
    distance_based_duration: bool = False
    interruptible: bool = True
    visibility_during_action: str = "visible"   # visible | instant | private | low_detail
    completion_event_type: str = ""             # defaults to f"{name}_completed"
    can_overlap_with_passive_perception: bool = True
    requires_agent_available: bool = True
    blocks_agent_until_complete: bool = True
    # --- push-through / overexertion (survival §18) ---------------------- #
    allow_push_through: bool = True          # may run while hungry/tired at extra cost
    push_through_min_energy: float = 0.0     # below this, even push-through is blocked
    overexertion_hp_risk: str = "low"        # very_low | low | medium | high
    overexertion_failure_risk: str = "low"
    critical_only_mask: bool = False         # only offered in a genuine crisis
    # --- classification -------------------------------------------------- #
    is_speech: bool = False
    is_emergency_survival: bool = False     # bypasses min-energy hard mask (§5)
    # feature hints the env-aware extractor leans on (it still reads real state)
    feature_hints: Tuple[str, ...] = ()

    @property
    def completion_event(self) -> str:
        return self.completion_event_type or f"{self.name}_completed"

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        return [f"missing required param '{p}'" for p in self.required_params if p not in (params or {})]


_MOVE_MODS = ("distance", "terrain", "carrying_load", "weather", "fatigue", "known_path", "tool")

ACTION_REGISTRY: Dict[str, ActionSpec] = {
    # ---- 4.1 Movement ---------------------------------------------------- #
    "move_to": ActionSpec(
        "move_to", ActionCategory.MOVEMENT, required_params=("target_location",),
        optional_params=("max_steps", "reason_tag"),
        preconditions=("target reachable", "energy > min"),
        failure_modes=("path_blocked", "too_exhausted", "target_unknown"),
        base_energy_cost=2.0, min_energy_required=5.0, failure_energy_cost=1.0,
        fatigue_delta=0.03, energy_cost_modifiers=_MOVE_MODS, distance_based_duration=True,
        feature_hints=("energy_cost", "time_cost", "information_gain")),
    "return_home": ActionSpec(
        "return_home", ActionCategory.MOVEMENT, optional_params=("target_home_id",),
        preconditions=("agent has home_location",),
        failure_modes=("path_blocked", "no_home"),
        base_energy_cost=2.0, min_energy_required=3.0, failure_energy_cost=1.0,
        fatigue_delta=0.02, energy_cost_modifiers=("distance", "carrying_load", "known_path"),
        distance_based_duration=True,
        feature_hints=("future_security_gain", "storage_value", "survival_gain", "energy_cost")),
    "move_to_known_resource": ActionSpec(
        "move_to_known_resource", ActionCategory.MOVEMENT,
        required_params=("resource_zone_id",), optional_params=("target_resource_type",),
        preconditions=("resource zone known",),
        failure_modes=("zone_unknown", "too_exhausted"),
        base_energy_cost=2.0, min_energy_required=6.0, failure_energy_cost=1.0,
        fatigue_delta=0.03, energy_cost_modifiers=_MOVE_MODS, distance_based_duration=True,
        feature_hints=("resource_gain", "energy_cost", "time_cost")),
    # ---- 4.2 Survival ---------------------------------------------------- #
    "eat_food": ActionSpec(
        "eat_food", ActionCategory.SURVIVAL, optional_params=("food_item_id", "amount"),
        preconditions=("agent has edible food",),
        failure_modes=("no_food", "food_spoiled", "invalid_item"),
        base_energy_cost=0.0, min_energy_required=0.0, energy_recovery=20.0,
        can_execute_when_exhausted=True, is_emergency_survival=True,
        feature_hints=("survival_gain", "energy_gain")),
    "rest": ActionSpec(
        "rest", ActionCategory.SURVIVAL, optional_params=("duration",),
        preconditions=("location safe enough or risk accepted",),
        failure_modes=("unsafe_location", "interrupted"),
        base_energy_cost=0.0, min_energy_required=0.0, energy_recovery=15.0,
        fatigue_delta=-0.03, can_execute_when_exhausted=True, is_emergency_survival=True,
        base_duration_ticks=1, interruptible=True,
        feature_hints=("energy_gain", "time_cost")),
    "sleep": ActionSpec(
        "sleep", ActionCategory.SURVIVAL,
        optional_params=("sleep_duration", "target_until_time", "sleep_location"),
        preconditions=("agent can lie down",),
        failure_modes=("interrupted", "unsafe_location"),
        base_energy_cost=0.0, min_energy_required=0.0, energy_recovery=0.0,  # recovery via clock.sleep_recovery
        fatigue_delta=0.0, can_execute_when_exhausted=True, is_emergency_survival=True,
        base_duration_ticks=4, interruptible=False, blocks_agent_until_complete=True,
        visibility_during_action="low_detail",
        feature_hints=("energy_gain", "survival_gain", "time_cost")),
    "seek_safety": ActionSpec(
        "seek_safety", ActionCategory.SURVIVAL, optional_params=("target_safe_location",),
        preconditions=("danger perceived or hp low or stress high", "safe location known"),
        failure_modes=("no_safe_location", "path_blocked"),
        base_energy_cost=2.0, min_energy_required=1.0, failure_energy_cost=1.0,
        fatigue_delta=0.04, can_execute_when_exhausted=True, is_emergency_survival=True,
        energy_cost_modifiers=("distance", "terrain"), distance_based_duration=True,
        feature_hints=("survival_gain", "risk_cost", "energy_cost")),
    # ---- 4.3 Resource ---------------------------------------------------- #
    "gather_resource": ActionSpec(
        "gather_resource", ActionCategory.RESOURCE,
        required_params=("resource_id",), optional_params=("resource_type", "amount_requested"),
        preconditions=("resource accessible", "amount > 0", "capacity > 0", "energy sufficient"),
        failure_modes=("carrying_capacity_full", "resource_depleted", "too_exhausted",
                       "resource_not_accessible"),
        base_energy_cost=4.0, min_energy_required=8.0, failure_energy_cost=2.0,
        fatigue_delta=0.05, energy_cost_modifiers=("terrain", "tool", "skill", "carrying_load"),
        base_duration_ticks=2,
        feature_hints=("resource_gain", "survival_gain", "future_security_gain", "energy_cost")),
    "attempt_craft": ActionSpec(
        "attempt_craft", ActionCategory.RESOURCE,
        required_params=("intended_function",),
        optional_params=("materials", "arrangement_summary", "related_wish_id"),
        preconditions=("inventory covers the function's required + binding properties",),
        failure_modes=("materials_missing", "unknown_function", "too_exhausted"),
        base_energy_cost=3.0, min_energy_required=6.0, failure_energy_cost=1.5,
        fatigue_delta=0.04, energy_cost_modifiers=("skill", "tool"),
        base_duration_ticks=2,
        feature_hints=("future_security_gain", "experiment_value", "skill_gain",
                       "material_cost", "energy_cost")),
    "inspect_resource": ActionSpec(
        "inspect_resource", ActionCategory.INSPECTION,
        required_params=("resource_id",), optional_params=("resource_type",),
        preconditions=("resource visible or nearby",), failure_modes=("not_visible",),
        base_energy_cost=1.0, min_energy_required=2.0, fatigue_delta=0.01,
        feature_hints=("information_gain", "experiment_value")),
    "inspect_area": ActionSpec(
        "inspect_area", ActionCategory.INSPECTION, optional_params=("radius", "direction"),
        base_energy_cost=1.0, min_energy_required=2.0, fatigue_delta=0.01,
        can_execute_when_exhausted=False, feature_hints=("information_gain",)),
    # ---- 4.4 Inventory / storage ---------------------------------------- #
    "drop_item": ActionSpec(
        "drop_item", ActionCategory.INVENTORY, required_params=("item_id",),
        optional_params=("amount",), preconditions=("item in inventory",),
        failure_modes=("item_missing",), base_energy_cost=0.0, min_energy_required=0.0,
        can_execute_when_exhausted=True, feature_hints=("energy_cost",)),
    "store_item_home": ActionSpec(
        "store_item_home", ActionCategory.INVENTORY,
        required_params=("item_id",), optional_params=("amount",),
        preconditions=("agent at home", "has item", "home storage capacity"),
        failure_modes=("not_at_home", "item_missing", "storage_full"),
        base_energy_cost=1.0, min_energy_required=1.0, fatigue_delta=0.01,
        feature_hints=("future_security_gain", "storage_value")),
    "retrieve_item_home": ActionSpec(
        "retrieve_item_home", ActionCategory.INVENTORY,
        required_params=("item_id",), optional_params=("amount",),
        preconditions=("agent at home", "storage has item", "capacity enough"),
        failure_modes=("not_at_home", "item_missing", "carrying_capacity_full"),
        base_energy_cost=1.0, min_energy_required=1.0, fatigue_delta=0.01,
        feature_hints=("resource_gain", "private_gain")),
    "transfer_item_to_agent": ActionSpec(
        "transfer_item_to_agent", ActionCategory.INVENTORY,
        required_params=("target_agent_id", "item_id"), optional_params=("amount",),
        preconditions=("target visible & in range", "has item"),
        failure_modes=("target_not_nearby", "item_missing", "target_capacity_full", "target_refuses"),
        base_energy_cost=1.0, min_energy_required=1.0, fatigue_delta=0.01,
        feature_hints=("altruistic_gain", "reciprocity_gain", "trust_gain")),
    # ---- 4.5 Inspection / knowledge ------------------------------------- #
    "inspect_object": ActionSpec(
        "inspect_object", ActionCategory.INSPECTION, required_params=("object_id",),
        preconditions=("object visible or accessible",), failure_modes=("not_visible",),
        base_energy_cost=1.0, min_energy_required=1.0, fatigue_delta=0.01,
        feature_hints=("information_gain",)),
    "inspect_agent_visible_state": ActionSpec(
        "inspect_agent_visible_state", ActionCategory.INSPECTION,
        required_params=("target_agent_id",), preconditions=("target visible",),
        failure_modes=("not_visible",), base_energy_cost=1.0, min_energy_required=1.0,
        fatigue_delta=0.01, feature_hints=("information_gain",)),
    "inspect_public_record": ActionSpec(
        "inspect_public_record", ActionCategory.INSPECTION, required_params=("record_id",),
        preconditions=("record visible / nearby / accessible",), failure_modes=("not_accessible",),
        base_energy_cost=1.0, min_energy_required=1.0, fatigue_delta=0.01,
        feature_hints=("information_gain", "verification_value")),
    # ---- 4.6 Speech ----------------------------------------------------- #
    "comm_local": ActionSpec(
        "comm_local", ActionCategory.SPEECH, required_params=("content_summary",),
        optional_params=("speech_intent", "target_agent_ids", "related_event_ids",
                         "related_plan_id", "related_episode_id"),
        preconditions=("listeners within local_comm_radius or target nearby",),
        failure_modes=("no_listeners",), base_energy_cost=1.0, min_energy_required=0.0,
        is_speech=True, can_execute_when_exhausted=True,
        feature_hints=("information_gain", "reputation_gain")),
    "ask_help": ActionSpec(
        "ask_help", ActionCategory.SPEECH,
        required_params=("target_agent_id", "help_type"),
        optional_params=("problem_summary", "related_plan_id", "related_episode_id"),
        preconditions=("target visible and nearby",), failure_modes=("target_not_nearby",),
        base_energy_cost=1.0, min_energy_required=0.0, is_speech=True,
        can_execute_when_exhausted=True,
        feature_hints=("reciprocity_gain", "reputation_risk", "time_cost")),
    "tell_info": ActionSpec(
        "tell_info", ActionCategory.SPEECH,
        required_params=("target_agent_id", "content_summary"),
        optional_params=("info_type", "evidence_memory_ids", "related_resource_id", "related_location"),
        preconditions=("target nearby", "speaker knows info"), failure_modes=("target_not_nearby",),
        base_energy_cost=1.0, min_energy_required=0.0, is_speech=True,
        can_execute_when_exhausted=True,
        feature_hints=("altruistic_gain", "reciprocity_gain", "reputation_gain")),
    "camp_announce": ActionSpec(
        "camp_announce", ActionCategory.SPEECH, required_params=("content_summary",),
        optional_params=("announcement_type", "related_resource_id", "related_mechanism_id",
                         "related_event_ids"),
        preconditions=("agent in camp zone",), failure_modes=("not_in_camp",),
        base_energy_cost=2.0, min_energy_required=1.0, fatigue_delta=0.01, is_speech=True,
        feature_hints=("coordination_gain", "reputation_gain", "public_good_gain", "conflict_risk")),
    "promise_action": ActionSpec(
        "promise_action", ActionCategory.SPEECH,
        required_params=("target_agent_id", "promised_action_type"),
        optional_params=("promise_summary", "deadline_turn", "related_plan_id"),
        preconditions=("target nearby or channel exists",), failure_modes=("target_not_nearby",),
        base_energy_cost=1.0, min_energy_required=0.0, is_speech=True,
        can_execute_when_exhausted=True,
        feature_hints=("reciprocity_gain", "reputation_gain")),
    "explain_action": ActionSpec(
        "explain_action", ActionCategory.SPEECH,
        required_params=("target_agent_ids", "explained_event_id"),
        optional_params=("content_summary",),
        preconditions=("related event exists", "target nearby or public"),
        failure_modes=("event_missing",), base_energy_cost=1.0, min_energy_required=0.0,
        is_speech=True, can_execute_when_exhausted=True,
        feature_hints=("reputation_gain",)),
    "apologize": ActionSpec(
        "apologize", ActionCategory.SPEECH,
        required_params=("target_agent_id", "related_event_id"),
        optional_params=("content_summary",), preconditions=("target nearby",),
        failure_modes=("target_not_nearby",), base_energy_cost=1.0, min_energy_required=0.0,
        is_speech=True, can_execute_when_exhausted=True,
        feature_hints=("reputation_gain", "trust_gain")),
    # ---- 4.7 Wait ------------------------------------------------------- #
    "wait_or_continue": ActionSpec(
        "wait_or_continue", ActionCategory.WAIT, optional_params=("reason",),
        base_energy_cost=0.0, min_energy_required=0.0, energy_recovery=2.0,
        fatigue_delta=-0.03, can_execute_when_exhausted=True,
        feature_hints=("energy_gain",)),
}

SPEECH_ACTIONS = tuple(n for n, s in ACTION_REGISTRY.items() if s.is_speech)
EMERGENCY_ACTIONS = tuple(n for n, s in ACTION_REGISTRY.items() if s.is_emergency_survival)


def get_spec(action_type: str) -> Optional[ActionSpec]:
    return ACTION_REGISTRY.get(action_type)


# --------------------------------------------------------------------------- #
# §2 WishCandidate — Reflection-stage output, NOT an action (kept as data only)
# --------------------------------------------------------------------------- #
@dataclass
class WishCandidate:
    wish_id: str
    agent_id: str
    turn_id: int
    need_type: str
    problem_summary: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    grounding_context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    next_module: str = "affordance"      # affordance / prototype / civic / material_problem_solving
    status: str = "open"
    # --- Stage C1 grounded-wish fields (spec §7) ------------------------- #
    agent_name: str = ""
    created_tick: int = 0
    first_person_summary: str = ""       # §28 always first-person ("我……")
    related_plan_id: Optional[str] = None
    related_step_id: Optional[str] = None
    related_episode_id: Optional[str] = None
    target_resource_type: Optional[str] = None
    target_object_id: Optional[str] = None
    candidate_direction: str = ""
    blocked_reasons: List[str] = field(default_factory=list)
    validator_result: str = ""
    rejected_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
