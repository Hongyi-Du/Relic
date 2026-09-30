"""Craft session + prototype system (Stage C1-C2, spec Part 4/5).

The SocioGenesis craft "brain": a deterministic, LLM-free evaluator that turns a
grounded :class:`CraftProposal` (materials + intended function + arrangement) into
a :class:`PrototypeObject` whose quality / durability / carrying-capacity bonus /
failure chance come from material-fit + structure-fit + function-fit + skill-fit +
grounding + prior-knowledge (§18). Crafting is a multi-tick :class:`CraftSession`
(§15); prototypes must be **tested** before they count as a known affordance (§21).

This is shared by the env-agnostic chain AND the nature_env path (where a basket
recipe also exists in registry.yaml so the existing CraftingEngine handles the
grid craft; this layer adds the quality/durability/capacity that nature_env
lacks). It ABSORBS nature_env's stone tools / skills rather than replacing them:
skills live in :attr:`ProfileState.skills`, and `stone_shaping` etc. map onto the
existing stone-tool recipes.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent_sdk.lived.world.materials import MATERIAL_REGISTRY, MaterialKnowledge, KnowledgeLevel

# §23 the v1 craft skills (proficiency in ProfileState.skills[name] ∈ [0,1]).
CRAFT_SKILLS: Tuple[str, ...] = (
    "reed_handling", "fiber_handling", "basic_tying", "weaving",
    "container_making", "wood_structure", "stone_shaping",
)

# intended_function -> what makes a good build (required props + the skills used).
@dataclass(frozen=True)
class FunctionSpec:
    required_properties: Tuple[str, ...]
    binding_properties: Tuple[str, ...]
    skills: Tuple[str, ...]
    grants: Tuple[str, ...]          # actual_functions on success


FUNCTION_SPECS: Dict[str, FunctionSpec] = {
    "carry_loose_grain": FunctionSpec(
        required_properties=("weave_candidate", "flexible", "long"),
        binding_properties=("bindable", "cord_candidate"),
        skills=("reed_handling", "weaving", "basic_tying", "container_making"),
        grants=("carry_loose_grain", "carry_items")),
    "carry_items": FunctionSpec(
        ("weave_candidate", "flexible"), ("bindable",),
        ("weaving", "container_making", "basic_tying"), ("carry_items",)),
    "store_food": FunctionSpec(
        ("structural", "rigid", "frame_candidate"), ("bindable",),
        ("wood_structure", "basic_tying"), ("store_food", "storage_frame")),
}

def propose_materials(inventory: Dict[str, int], intended_function: str) -> Optional[Dict[str, int]]:
    """Judge whether ``inventory`` covers ``intended_function``'s recipe and, if
    so, propose which materials to commit (env-agnostic: coverage is decided by
    MATERIAL_REGISTRY *properties*, never by material or env names).

    Coverage = every ``required_property`` of the FunctionSpec is contributed by
    some carried material AND at least one carried material has a
    ``binding_property``. Commits up to 3 units per structural material and up
    to 2 of the binder (a workable v1 batch; more never raises quality caps).
    Returns None when the function is unknown or the inventory can't cover it.
    """
    spec = FUNCTION_SPECS.get(intended_function)
    if spec is None:
        return None
    counts = {m: int(c) for m, c in (inventory or {}).items()
              if c and int(c) > 0 and m in MATERIAL_REGISTRY}
    if not counts:
        return None
    req = set(spec.required_properties)
    bind = set(spec.binding_properties)
    picked: Dict[str, int] = {}
    covered: set = set()
    # structural picks: greedily take materials that add uncovered required props
    for m in sorted(counts, key=lambda k: (-len(set(MATERIAL_REGISTRY[k].properties) & req), k)):
        gain = (set(MATERIAL_REGISTRY[m].properties) & req) - covered
        if gain:
            picked[m] = min(3, counts[m])
            covered |= gain
    if covered != req:
        return None
    # binder: some picked material must bind, else add a dedicated binder
    if not any(set(MATERIAL_REGISTRY[m].properties) & bind for m in picked):
        binder = next((m for m in sorted(counts)
                       if set(MATERIAL_REGISTRY[m].properties) & bind), None)
        if binder is None:
            return None
        picked[binder] = min(2, counts[binder])
    return picked


def craftable_functions(inventory: Dict[str, int]) -> List[str]:
    """All intended_functions the inventory can cover, in FUNCTION_SPECS order."""
    return [fn for fn in FUNCTION_SPECS if propose_materials(inventory, fn) is not None]


# §19 quality bands -> prototype stats.
@dataclass(frozen=True)
class ResultBand:
    name: str
    object_type: str
    capacity_bonus: int
    durability: int
    failure_chance: float
    movement_cost_modifier: float


RESULT_BANDS: Tuple[Tuple[float, ResultBand], ...] = (
    (0.85, ResultBand("sturdy_basket", "basket", 3, 6, 0.10, 0.05)),
    (0.65, ResultBand("loose_reed_basket", "basket", 3, 4, 0.20, 0.10)),
    (0.45, ResultBand("fragile_container", "container", 2, 3, 0.35, 0.15)),
    (0.25, ResultBand("weak_bundle", "bundle", 1, 2, 0.50, 0.05)),
    (0.0, ResultBand("scrap", "scrap", 0, 0, 1.0, 0.0)),
)


def band_for(q: float) -> ResultBand:
    for thresh, band in RESULT_BANDS:
        if q >= thresh:
            return band
    return RESULT_BANDS[-1][1]


def _seeded_rng(*parts: Any) -> random.Random:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# --------------------------------------------------------------------------- #
# Proposal / prototype / result / session DTOs
# --------------------------------------------------------------------------- #
@dataclass
class CraftProposal:
    proposal_id: str
    agent_id: str
    intended_function: str
    materials: Dict[str, int]                 # material_type -> count
    problem_target: str = ""
    arrangement_summary: str = ""
    expected_effect: str = ""
    reasoning_source: str = "rule_based"
    related_wish_id: Optional[str] = None
    confidence: float = 0.5


@dataclass
class PrototypeObject:
    object_id: str
    object_type: str
    name: str
    created_by_agent_id: str
    created_tick: int
    materials_used: Dict[str, int]
    intended_function: str
    actual_functions: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    capacity_bonus: int = 0
    durability: int = 0
    max_durability: int = 0
    failure_chance: float = 1.0
    movement_cost_modifier: float = 0.0
    known_to_agents: List[str] = field(default_factory=list)
    tested: bool = False
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "untested"                  # untested|usable|damaged|broken|scrap

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @property
    def provides_carry(self) -> bool:
        return (self.status in ("usable", "damaged") and self.durability > 0
                and any(f in ("carry_loose_grain", "carry_items") for f in self.actual_functions))


@dataclass
class CraftResult:
    quality_score: float
    result_type: str
    fits: Dict[str, float]
    prototype: Optional[PrototypeObject]
    materials_consumed: Dict[str, int]
    materials_returned: Dict[str, int]
    skill_gains: Dict[str, float]
    success: bool
    message: str = ""


@dataclass
class CraftSession:
    craft_session_id: str
    agent_id: str
    intended_function: str
    start_tick: int
    status: str = "active"                     # active|completed|failed|abandoned|interrupted
    agent_name: str = ""
    linked_wish_id: Optional[str] = None
    target_problem: str = ""
    input_materials: Dict[str, int] = field(default_factory=dict)
    material_arrangement: str = ""
    used_skills: List[str] = field(default_factory=list)
    progress: float = 0.0
    attempts: int = 0
    current_prototype_id: Optional[str] = None
    failure_reasons: List[str] = field(default_factory=list)
    related_plan_id: Optional[str] = None
    related_memory_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------- #
# The deterministic evaluator (§18)
# --------------------------------------------------------------------------- #
_WEIGHTS = {"material": 0.25, "structure": 0.20, "function": 0.25,
            "skill": 0.15, "grounding": 0.10, "prior": 0.05}


class CraftSystem:
    """Scores a proposal → quality → prototype (deterministic; no LLM picks the
    output). Materials are never created or destroyed silently — consumed/returned
    are reported so the caller mutates inventory once."""

    def evaluate(self, proposal: CraftProposal, *, skills: Dict[str, float],
                 knowledge: Optional[MaterialKnowledge] = None, inventory: Optional[Dict[str, int]] = None,
                 tick: int = 0) -> CraftResult:
        spec = FUNCTION_SPECS.get(proposal.intended_function)
        mats = {m: c for m, c in proposal.materials.items() if c > 0}
        fits = {
            "material_fit": self._material_fit(mats, spec),
            "structure_fit": self._structure_fit(mats, spec, proposal.arrangement_summary),
            "function_fit": self._function_fit(mats, spec),
            "skill_fit": self._skill_fit(spec, skills),
            "grounding": self._grounding(proposal),
            "prior_knowledge": self._prior(mats, knowledge),
        }
        q = (_WEIGHTS["material"] * fits["material_fit"]
             + _WEIGHTS["structure"] * fits["structure_fit"]
             + _WEIGHTS["function"] * fits["function_fit"]
             + _WEIGHTS["skill"] * fits["skill_fit"]
             + _WEIGHTS["grounding"] * fits["grounding"]
             + _WEIGHTS["prior"] * fits["prior_knowledge"])
        q = max(0.0, min(1.0, q))
        band = band_for(q)

        # materials: scrap returns most (1 lost), otherwise consumed.
        consumed: Dict[str, int] = dict(mats)
        returned: Dict[str, int] = {}
        if band.name == "scrap":
            returned = {m: max(0, c - 1) for m, c in mats.items()}
            consumed = {m: c - returned.get(m, 0) for m, c in mats.items()}

        used_skills = list(spec.skills) if spec else []
        skill_gains = self._skill_gain(used_skills, success=band.name != "scrap",
                                       quality=q)
        proto: Optional[PrototypeObject] = None
        if band.object_type != "scrap":
            proto = PrototypeObject(
                object_id=f"proto:{proposal.agent_id}:{tick}:{band.name}",
                object_type=band.object_type, name=band.name.replace("_", " "),
                created_by_agent_id=proposal.agent_id, created_tick=tick,
                materials_used=dict(consumed), intended_function=proposal.intended_function,
                actual_functions=list(spec.grants) if spec else [proposal.intended_function],
                quality_score=round(q, 3), capacity_bonus=band.capacity_bonus,
                durability=band.durability, max_durability=band.durability,
                failure_chance=band.failure_chance, movement_cost_modifier=band.movement_cost_modifier,
                known_to_agents=[proposal.agent_id], status="untested")
        return CraftResult(quality_score=round(q, 3), result_type=band.name, fits=fits,
                           prototype=proto, materials_consumed=consumed, materials_returned=returned,
                           skill_gains=skill_gains, success=band.name != "scrap",
                           message=f"{band.name} (q={q:.2f})")

    # -- fit components ----------------------------------------------------- #
    def _props(self, mats: Dict[str, int]) -> set:
        out: set = set()
        for m in mats:
            mt = MATERIAL_REGISTRY.get(m)
            if mt:
                out.update(mt.properties)
        return out

    def _material_fit(self, mats: Dict[str, int], spec: Optional[FunctionSpec]) -> float:
        if not mats:
            return 0.0
        if spec is None:
            return 0.3
        props = self._props(mats)
        req = set(spec.required_properties)
        bind = set(spec.binding_properties)
        have_req = len(req & props) / max(1, len(req))
        have_bind = 1.0 if (props & bind) else 0.0
        return max(0.0, min(1.0, 0.6 * have_req + 0.4 * have_bind))

    def _structure_fit(self, mats: Dict[str, int], spec: Optional[FunctionSpec],
                       arrangement: str) -> float:
        # need enough structural units + a binder; arrangement text adds a little.
        props = self._props(mats)
        total = sum(mats.values())
        struct = 1.0 if total >= 3 else total / 3.0
        binder = 1.0 if (props & {"bindable", "cord_candidate"}) else 0.3
        arr = 0.2 if arrangement and any(w in arrangement.lower()
                                         for w in ("wall", "bind", "weave", "wrap", "frame")) else 0.0
        return max(0.0, min(1.0, 0.5 * struct + 0.3 * binder + arr))

    def _function_fit(self, mats: Dict[str, int], spec: Optional[FunctionSpec]) -> float:
        if spec is None or not mats:
            return 0.2                       # unknown function -> weak
        props = self._props(mats)
        # needs the PRIMARY structural property of the function + a binder; a
        # material that merely shares a generic prop (grass is "flexible") is weak.
        primary = bool(props & set(spec.required_properties[:1]))
        has_binder = bool(props & set(spec.binding_properties))
        return min(1.0, (0.6 if primary else 0.2) + (0.4 if has_binder else 0.0))

    def _skill_fit(self, spec: Optional[FunctionSpec], skills: Dict[str, float]) -> float:
        if spec is None or not spec.skills:
            return 0.3
        vals = [float(skills.get(s, 0.0)) for s in spec.skills]
        return max(0.0, min(1.0, sum(vals) / len(vals)))

    def _grounding(self, proposal: CraftProposal) -> float:
        g = 0.4
        if proposal.related_wish_id:
            g += 0.4
        if proposal.reasoning_source and proposal.reasoning_source != "none":
            g += 0.2
        return min(1.0, g)

    def _prior(self, mats: Dict[str, int], knowledge: Optional[MaterialKnowledge]) -> float:
        if knowledge is None:
            return 0.0
        levels = [int(knowledge.level(m)) for m in mats]
        if not levels:
            return 0.0
        return min(1.0, (sum(levels) / len(levels)) / int(KnowledgeLevel.CAN_TEACH))

    def _skill_gain(self, skills: List[str], *, success: bool, quality: float) -> Dict[str, float]:
        if not skills:
            return {}
        if not success:
            per = 0.02                       # §24 failed attempt still teaches a little
        elif quality >= 0.85:
            per = 0.09
        elif quality >= 0.65:
            per = 0.06
        else:
            per = 0.04
        return {s: per for s in skills}


# --------------------------------------------------------------------------- #
# Prototype testing (§21) + repair (§16.6)
# --------------------------------------------------------------------------- #
def test_prototype(proto: PrototypeObject, *, agent_id: str, tick: int,
                   carried_grain: int = 1) -> Dict[str, Any]:
    """Test a carry prototype by carrying grain. Deterministic seeded outcome
    based on failure_chance. Updates durability/status/test_results."""
    rng = _seeded_rng(agent_id, tick, proto.object_id, proto.durability)
    roll = rng.random()
    failed = roll < proto.failure_chance
    spilled = 0
    if failed:
        outcome = "breakage" if rng.random() < 0.4 else "failure"
        proto.durability = max(0, proto.durability - 1)
        if outcome == "breakage" or rng.random() < 0.5:
            spilled = min(carried_grain, 1 + int(2 * proto.failure_chance))
            outcome = "spilled_material" if spilled and outcome != "breakage" else outcome
    else:
        outcome = "success" if proto.quality_score >= 0.55 else "partial_success"
        proto.durability = max(0, proto.durability - (0 if outcome == "success" else 1))
    proto.tested = True
    if proto.durability <= 0:
        proto.status = "broken"
    elif outcome in ("success", "partial_success"):
        proto.status = "usable"
    else:
        proto.status = "damaged"
    rec = {"tick": tick, "test_type": "carry_grain", "outcome": outcome,
           "success": outcome in ("success", "partial_success"),
           "durability_after": proto.durability, "spilled_material": spilled,
           "capacity_effect": proto.capacity_bonus if proto.provides_carry else 0}
    proto.test_results.append(rec)
    return rec


def repair_or_modify_prototype(proto: PrototypeObject, *, repair_materials: Dict[str, int],
                               skills: Dict[str, float]) -> Dict[str, Any]:
    """Repair: restore durability (capped) + small quality bump (§16.6)."""
    if not repair_materials or proto.status == "scrap":
        return {"repaired": False, "reason": "no_material_or_scrap"}
    skill = float(skills.get("container_making", 0.0)) + float(skills.get("basic_tying", 0.0))
    restore = 1 + int(skill)
    proto.durability = min(proto.max_durability, proto.durability + restore)
    proto.quality_score = min(1.0, proto.quality_score + 0.02)
    if proto.durability > 0 and proto.status == "broken":
        proto.status = "usable"
    return {"repaired": True, "durability_after": proto.durability,
            "restored": restore, "materials_used": dict(repair_materials)}


def effective_capacity(base_capacity: float, prototypes: List[PrototypeObject]) -> float:
    """§22 base + Σ usable carry-prototype capacity_bonus."""
    bonus = sum(p.capacity_bonus for p in prototypes if p.provides_carry)
    return float(base_capacity) + float(bonus)
