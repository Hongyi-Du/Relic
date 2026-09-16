"""Material registry + per-agent material knowledge (Stage C1, spec Part 3).

The v1 materials an agent can perceive, acquire, inspect, learn about and craft
with — grain / reed / fiber / wood / stone / grass (+ optional clay). Two pieces:

  * :data:`MATERIAL_REGISTRY` — the static, env-agnostic catalogue (properties,
    source resource types, acquisition actions, slot size, terrain).
  * :class:`MaterialKnowledgeState` — what ONE agent knows about ONE material,
    progressing unknown → heard_of → observed → inspected → tested →
    used_successfully → can_teach as it inspects/gathers/uses it.

Pure + LLM-free. Property learning is deterministic (inspect reveals a known
property). No nutrition/chemistry — just the affordance facts the craft system
and plan monitor reason over.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple


class KnowledgeLevel(IntEnum):
    """Monotonic ladder of how well an agent knows a material (§14)."""
    UNKNOWN = 0
    HEARD_OF = 1
    OBSERVED = 2
    INSPECTED = 3
    TESTED = 4
    USED_SUCCESSFULLY = 5
    CAN_TEACH = 6


@dataclass(frozen=True)
class MaterialType:
    """Static spec for one material (§11)."""
    material_type: str
    display_name: str
    source_resource_types: Tuple[str, ...]
    acquisition_actions: Tuple[str, ...]
    base_slot_size: float
    base_weight: float
    properties: Tuple[str, ...]
    common_locations: Tuple[str, ...] = ()
    terrain_association: Tuple[str, ...] = ()
    seasonal_availability: Tuple[str, ...] = ("spring", "summer", "fall", "winter")
    default_visibility: str = "visible"
    processing_required: bool = False
    spoilage: bool = False
    uses: Tuple[str, ...] = ()


# §12 the v1 materials.
MATERIAL_REGISTRY: Dict[str, MaterialType] = {
    "grain": MaterialType(
        "grain", "grain", ("grain", "grain_meadow", "wild_grain", "meadow"),
        ("gather_resource",), base_slot_size=1.0, base_weight=1.0,
        properties=("edible", "storable", "loose", "small_units",
                    "requires_container_for_efficient_transport"),
        terrain_association=("grain_meadow", "meadow", "grassland"),
        spoilage=True, uses=("eat_food", "store_food", "shared_cache")),
    "reed": MaterialType(
        "reed", "reed", ("reed", "reed_patch", "wetland"),
        ("search_known_resource", "gather_resource"), base_slot_size=0.5, base_weight=0.4,
        properties=("flexible", "light", "long", "weak_alone", "weave_candidate",
                    "poor_binding_alone"),
        terrain_association=("wetland", "river_delta", "marsh"),
        uses=("basket_wall", "mat", "frame", "bundle")),
    "fiber": MaterialType(
        "fiber", "fiber", ("fiber", "bark_fiber", "grass_fiber", "vine", "wetland_plant"),
        ("search_known_resource", "gather_resource", "inspect_material_properties"),
        base_slot_size=0.5, base_weight=0.2,
        properties=("bindable", "flexible", "cord_candidate", "low_structure"),
        terrain_association=("wetland", "woodland", "ancient_forest"),
        uses=("binding", "tying", "basket_reinforcement", "bundle_fastening")),
    "wood": MaterialType(
        "wood", "wood", ("wood", "tree", "fallen_branch", "shrub"),
        ("gather_resource",), base_slot_size=1.0, base_weight=1.5,
        properties=("structural", "rigid", "heavy", "burnable", "frame_candidate"),
        terrain_association=("ancient_forest", "woodland", "forest"),
        uses=("storage_frame", "drying_rack", "tool_handle", "fuel")),
    "stone": MaterialType(
        "stone", "stone", ("stone", "stone_node", "stone_outcrop", "river_stone", "ridge"),
        ("gather_resource",), base_slot_size=1.0, base_weight=2.0,
        properties=("hard", "sharp_if_flaked", "heavy", "cutting_candidate"),
        terrain_association=("rocky_hills", "ridge", "river_delta"),
        uses=("scraper", "cutter", "weight", "tool_head")),
    "grass": MaterialType(
        "grass", "grass", ("grass", "grassland", "meadow"),
        ("gather_resource",), base_slot_size=0.25, base_weight=0.1,
        properties=("flexible", "weak", "drying_candidate", "light"),
        terrain_association=("grassland", "meadow"),
        uses=("filler", "bedding", "weak_binding", "insulation")),
    # --- absorbed from nature_env's existing world vocabulary (not duplicated) --- #
    "wooden_sticks": MaterialType(
        "wooden_sticks", "wooden sticks", ("wooden_sticks", "tree", "wood"),
        ("gather_resource",), base_slot_size=0.5, base_weight=0.5,
        properties=("structural", "rigid", "frame_candidate", "tool_handle_candidate"),
        terrain_association=("ancient_forest", "woodland"),
        uses=("tool_handle", "frame", "campfire", "tent")),
    "char_coal": MaterialType(
        "char_coal", "charcoal", ("char_coal", "tree"),
        ("gather_resource",), base_slot_size=0.25, base_weight=0.2,
        properties=("burnable", "fuel"), uses=("campfire", "fuel")),
    # one canonical "hide" — nature drops it under three names (leather / cow_hide
    # / tiger_hide); all alias here so the material layer matches real drops.
    "hide": MaterialType(
        "hide", "hide", ("leather", "cow_hide", "tiger_hide", "cow", "tiger"),
        ("harvest_cow", "kill_cow", "pickup"), base_slot_size=1.0, base_weight=1.0,
        properties=("flexible", "coverable", "insulating", "structural_soft"),
        terrain_association=("grassland", "savanna"),
        processing_required=True, uses=("tent", "wrap", "cover", "binding_strong")),
}

# resource_type (as seen in the world) -> material_type (canonical). Covers the
# nature_env resource names too (tree -> wood, stone_node -> stone, ...).
_RESOURCE_TO_MATERIAL: Dict[str, str] = {}
for _mt in MATERIAL_REGISTRY.values():
    for _src in _mt.source_resource_types:
        _RESOURCE_TO_MATERIAL.setdefault(_src, _mt.material_type)
# nature_env resource/drop names -> canonical material (absorb the real vocabulary)
_RESOURCE_TO_MATERIAL.update({
    "tree": "wood", "stone_node": "stone",
    "leather": "hide", "cow_hide": "hide", "tiger_hide": "hide", "cow": "hide",
})


def material_for_resource(resource_type: Optional[str]) -> Optional[str]:
    """Map a world resource_type to its canonical material_type (or None)."""
    if not resource_type:
        return None
    rt = str(resource_type).lower()
    if rt in _RESOURCE_TO_MATERIAL:
        return _RESOURCE_TO_MATERIAL[rt]
    for key, mat in _RESOURCE_TO_MATERIAL.items():
        if key in rt:
            return mat
    return rt if rt in MATERIAL_REGISTRY else None


def slot_size(material_type: str) -> float:
    mt = MATERIAL_REGISTRY.get(material_type)
    return mt.base_slot_size if mt else 1.0


# --------------------------------------------------------------------------- #
# Per-agent knowledge (§14)
# --------------------------------------------------------------------------- #
@dataclass
class MaterialKnowledgeState:
    material_type: str
    level: KnowledgeLevel = KnowledgeLevel.UNKNOWN
    known_properties: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    source_memory_ids: List[str] = field(default_factory=list)
    learned_from_agent_id: Optional[str] = None
    last_updated_tick: int = -1
    successful_use_count: int = 0
    failed_use_count: int = 0

    def to_dict(self) -> Dict:
        return {"material_type": self.material_type, "level": self.level.name.lower(),
                "known_properties": sorted(self.known_properties),
                "confidence": round(self.confidence, 3),
                "successful_use_count": self.successful_use_count,
                "failed_use_count": self.failed_use_count,
                "last_updated_tick": self.last_updated_tick}


class MaterialKnowledge:
    """All of one agent's material knowledge states."""

    def __init__(self) -> None:
        self.states: Dict[str, MaterialKnowledgeState] = {}

    def get(self, material_type: str) -> MaterialKnowledgeState:
        s = self.states.get(material_type)
        if s is None:
            s = MaterialKnowledgeState(material_type=material_type)
            self.states[material_type] = s
        return s

    def level(self, material_type: str) -> KnowledgeLevel:
        return self.states[material_type].level if material_type in self.states else KnowledgeLevel.UNKNOWN

    def _bump(self, s: MaterialKnowledgeState, level: KnowledgeLevel, tick: int,
              memory_id: Optional[str], conf: float) -> None:
        if level > s.level:
            s.level = level
        s.last_updated_tick = tick
        s.confidence = max(s.confidence, conf)
        if memory_id:
            s.source_memory_ids.append(memory_id)

    def hear_of(self, material_type: str, *, tick: int, from_agent: Optional[str] = None,
                memory_id: Optional[str] = None) -> MaterialKnowledgeState:
        s = self.get(material_type)
        self._bump(s, KnowledgeLevel.HEARD_OF, tick, memory_id, 0.2)
        if from_agent:
            s.learned_from_agent_id = from_agent
        return s

    def observe(self, material_type: str, *, tick: int, memory_id: Optional[str] = None) -> MaterialKnowledgeState:
        s = self.get(material_type)
        self._bump(s, KnowledgeLevel.OBSERVED, tick, memory_id, 0.4)
        return s

    def inspect(self, material_type: str, *, tick: int, memory_id: Optional[str] = None,
                reveal: Optional[List[str]] = None) -> MaterialKnowledgeState:
        """Inspecting reveals the material's properties (deterministic, §13.4)."""
        s = self.get(material_type)
        props = reveal if reveal is not None else list(
            MATERIAL_REGISTRY.get(material_type, MaterialType(material_type, material_type, (), (), 1.0, 1.0, ())).properties)
        s.known_properties.update(props)
        self._bump(s, KnowledgeLevel.INSPECTED, tick, memory_id, 0.6)
        return s

    def record_use(self, material_type: str, *, success: bool, tick: int,
                   memory_id: Optional[str] = None) -> MaterialKnowledgeState:
        s = self.get(material_type)
        if success:
            s.successful_use_count += 1
            lvl = KnowledgeLevel.CAN_TEACH if s.successful_use_count >= 2 else KnowledgeLevel.USED_SUCCESSFULLY
            self._bump(s, lvl, tick, memory_id, 0.85 if lvl == KnowledgeLevel.CAN_TEACH else 0.7)
        else:
            s.failed_use_count += 1
            self._bump(s, KnowledgeLevel.TESTED, tick, memory_id, max(s.confidence, 0.5))
        return s

    def to_dict(self) -> Dict[str, Dict]:
        return {k: v.to_dict() for k, v in self.states.items()}
