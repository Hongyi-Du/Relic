"""
auto_trigger/types.py — Data types for the Conditioned Reflex system.

All types are env-agnostic. ReflexContext spatial fields are populated by
the environment's build_reflex_context() implementation.

Level 4/5 upgrade (2026-04-18):
  * ReflexKind enum distinguishes SELF / PAIR / MODE reflex objects.
  * UtilityTrace records a 5-axis utility vector for every fire.
  * ReflexPairTemplate binds an Emitter + Responder template pair.
  * ModeTemplate is a no-world-action reflex that activates a mode tag
    with TTL + overlay, realising Gene Regulation (Level 5).
  * ReflexTemplate is extended with vitality + kind + pair/mode metadata.
  * AgentReflexEntry is extended with fine-grained utility & pair/mode
    lifecycle state.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Reflex object kind
# ---------------------------------------------------------------------------

class ReflexKind(str, Enum):
    """Discriminates the three reflex object types introduced in L4/L5."""
    SELF = "self"      # Individual maintenance / routine labour (Level 1-2).
    PAIR = "pair"      # Cooperative emitter/responder closure (Level 3-4).
    MODE = "mode"      # Mode activation with TTL overlay (Level 5).


class PairRole(str, Enum):
    """Role of a template inside a cooperative pair."""
    EMITTER = "emitter"
    RESPONDER = "responder"


# ---------------------------------------------------------------------------
# ReflexSignal types
# ---------------------------------------------------------------------------

class ReflexSignalType(str, Enum):
    NEED_FOOD = "NEED_FOOD"
    NEED_WOOD = "NEED_WOOD"
    NEED_STONE = "NEED_STONE"
    OFFER_FOOD = "OFFER_FOOD"
    OFFER_RESOURCE = "OFFER_RESOURCE"
    DANGER_ALERT = "DANGER_ALERT"
    CLEAR_AREA = "CLEAR_AREA"
    FOLLOW_ME = "FOLLOW_ME"
    AVOID_AREA = "AVOID_AREA"
    MEET_AT = "MEET_AT"
    HELP_BUILD = "HELP_BUILD"
    HELP_CRAFT = "HELP_CRAFT"
    REPRODUCTION_OK = "REPRODUCTION_OK"


@dataclass(frozen=True)
class ReflexSignal:
    """Structured signal emitted by reflex templates, isolated from LLM chat."""
    signal_id: str
    signal_type: str
    sender_id: str
    sender_name: str
    sender_position: Tuple[int, int]
    turn_sent: int
    radius: int
    payload: Dict[str, Any] = field(default_factory=dict)
    # L4 pair tracking — the emitter stamps these; the responder reports them
    # back through the closure tracker on consumption.
    pair_id: str = ""
    emit_id: str = ""

    @staticmethod
    def make_id() -> str:
        return f"rsig_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def make_emit_id() -> str:
        return f"emit_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Spatial snapshots (populated by env, consumed by evaluator)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntitySnapshot:
    entity_id: str
    entity_type: str
    x: int
    y: int
    distance: int
    state: str = ""

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True)
class AgentSnapshot:
    agent_id: str
    name: str
    x: int
    y: int
    distance: int
    energy_pct: float = 1.0
    is_hungry: bool = False

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)


# ---------------------------------------------------------------------------
# ReflexContext — per-turn evaluation snapshot
# ---------------------------------------------------------------------------

@dataclass
class ReflexContext:
    """Pure data snapshot passed to ConditionEvaluator. No world references."""
    agent_id: str = ""
    energy: float = 0.0
    energy_pct: float = 0.0
    hp: float = 0.0
    hp_pct: float = 0.0
    age: int = 0
    x: int = 0
    y: int = 0
    inventory: Dict[str, int] = field(default_factory=dict)

    resource_pressure: str = "none"
    danger_level: str = "none"
    coordination_need: str = "none"
    progress_state: str = "active"

    nearest_entities: Dict[str, EntitySnapshot] = field(default_factory=dict)
    nearby_agents: List[AgentSnapshot] = field(default_factory=list)

    turn: int = 0
    observation_metadata: Dict[str, Any] = field(default_factory=dict)
    last_reflex_fired_turn: Dict[str, int] = field(default_factory=dict)
    available_action_types: Set[str] = field(default_factory=set)
    received_signals: List[ReflexSignal] = field(default_factory=list)

    # L5 mode-trigger fields (populated by env adapter)
    active_modes: Set[str] = field(default_factory=set)
    # Lightweight numeric trend slots the env fills when relevant
    trends: Dict[str, float] = field(default_factory=dict)
    # Group-level scalars (0..1) — resource pressure across nearby agents etc.
    group_metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# UtilityTrace — 5-axis record written after every reflex fire
# ---------------------------------------------------------------------------

@dataclass
class UtilityTrace:
    """Per-fire multi-axis utility record.

    Axes (signed, unit-free but comparable within the same kind):
      u_cost  : LLM inference cost saved (≥0)
      u_surv  : survival/homeostasis contribution (negative if destabilising)
      u_task  : main-task support (aligned with env metric_delta)
      u_disp  : displacement penalty already baked in as a NEGATIVE number
                (i.e. u_disp <= 0). Scoring code adds it directly.
      u_coord : pair closure or group coordination contribution
    """
    turn: int = 0
    agent_id: str = ""
    template_id: str = ""
    kind: str = ReflexKind.SELF.value
    u_cost: float = 0.0
    u_surv: float = 0.0
    u_task: float = 0.0
    u_disp: float = 0.0
    u_coord: float = 0.0
    # The outcome channel (string success/failure/partial) for confidence update
    outcome: str = "success"

    def total(self) -> float:
        return self.u_cost + self.u_surv + self.u_task + self.u_disp + self.u_coord


# ---------------------------------------------------------------------------
# ReflexTemplate — shareable rule definition
# ---------------------------------------------------------------------------

class TemplateStatus(str, Enum):
    LOCKED = "locked"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass
class ReflexTemplate:
    """A shareable conditioned-reflex rule.

    Level 4/5 fields (all default-valued for backward compat):
      kind          : ReflexKind (SELF default preserves old behaviour)
      vitality      : time-varying adoption-visibility score 0..1
      pair_id       : bound pair identifier (non-empty iff kind==PAIR)
      pair_role     : EMITTER / RESPONDER (only meaningful for pair)
      mode_id       : mode tag activated when this template fires (MODE only)
      ttl_turns     : mode-window length (MODE only)
      exclusive_with: list of mode_ids this mode cannot co-exist with (MODE)
      overlay_name  : key into env overlay YAML for prompt bias (MODE)
    """
    template_id: str
    name: str
    description: str
    condition: dict
    action_program: dict
    expected_free: bool = False
    priority: int = 50
    cooldown_turns: int = 3
    max_fires: int = -1
    tags: List[str] = field(default_factory=list)
    creator_id: str = ""
    created_turn: int = 0
    confidence: float = 0.3
    schema_version: str = "1.0"
    status: str = TemplateStatus.ACTIVE

    likes: int = 0
    dislikes: int = 0
    adopt_count: int = 0

    # --- L4/L5 extensions ---
    kind: str = ReflexKind.SELF.value
    vitality: float = 1.0
    pair_id: str = ""
    pair_role: str = ""
    mode_id: str = ""
    ttl_turns: int = 0
    exclusive_with: List[str] = field(default_factory=list)
    overlay_name: str = ""

    # Running aggregates used by kind-aware confidence. Updated from
    # UtilityTrace via AutoTriggerStore.apply_utility_trace().
    positive_fires: int = 0     # successes / pair closures / mode ΔU>0
    negative_fires: int = 0     # failures / missed closures / mode ΔU<=0
    total_fires: int = 0        # unique agent-weighted fires observed by store
    last_seen_turn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "action_program": self.action_program,
            "expected_free": self.expected_free,
            "priority": self.priority,
            "cooldown_turns": self.cooldown_turns,
            "max_fires": self.max_fires,
            "tags": list(self.tags),
            "creator_id": self.creator_id,
            "created_turn": self.created_turn,
            "confidence": self.confidence,
            "schema_version": self.schema_version,
            "status": self.status,
            "likes": self.likes,
            "dislikes": self.dislikes,
            "adopt_count": self.adopt_count,
            "kind": self.kind,
            "vitality": self.vitality,
            "pair_id": self.pair_id,
            "pair_role": self.pair_role,
            "mode_id": self.mode_id,
            "ttl_turns": self.ttl_turns,
            "exclusive_with": list(self.exclusive_with),
            "overlay_name": self.overlay_name,
            "positive_fires": self.positive_fires,
            "negative_fires": self.negative_fires,
            "total_fires": self.total_fires,
            "last_seen_turn": self.last_seen_turn,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReflexTemplate":
        return cls(
            template_id=d.get("template_id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            condition=d.get("condition", {}),
            action_program=d.get("action_program", {}),
            expected_free=d.get("expected_free", False),
            priority=d.get("priority", 50),
            cooldown_turns=d.get("cooldown_turns", 3),
            max_fires=d.get("max_fires", -1),
            tags=d.get("tags", []),
            creator_id=d.get("creator_id", ""),
            created_turn=d.get("created_turn", 0),
            confidence=d.get("confidence", 0.3),
            schema_version=d.get("schema_version", "1.0"),
            status=d.get("status", TemplateStatus.ACTIVE),
            likes=d.get("likes", 0),
            dislikes=d.get("dislikes", 0),
            adopt_count=d.get("adopt_count", 0),
            kind=d.get("kind", ReflexKind.SELF.value),
            vitality=d.get("vitality", 1.0),
            pair_id=d.get("pair_id", ""),
            pair_role=d.get("pair_role", ""),
            mode_id=d.get("mode_id", ""),
            ttl_turns=d.get("ttl_turns", 0),
            exclusive_with=d.get("exclusive_with", []),
            overlay_name=d.get("overlay_name", ""),
            positive_fires=d.get("positive_fires", 0),
            negative_fires=d.get("negative_fires", 0),
            total_fires=d.get("total_fires", 0),
            last_seen_turn=d.get("last_seen_turn", 0),
        )


# ---------------------------------------------------------------------------
# ReflexPairTemplate — binds an emitter + responder template
# ---------------------------------------------------------------------------

@dataclass
class ReflexPairTemplate:
    """Descriptor for a cooperative pair.

    Both referenced templates must already be published with kind=PAIR and
    matching pair_id. The pair is the reward-attribution unit: closure =
    emitter fires → responder consumes the same emit_id within the window.
    """
    pair_id: str
    name: str
    description: str
    emitter_template_id: str
    responder_template_id: str
    signal_type: str
    closure_window: int = 3
    tags: List[str] = field(default_factory=list)

    # Aggregate stats maintained by PairClosureTracker / store
    emissions: int = 0
    closures: int = 0
    misses: int = 0
    confidence: float = 0.3
    vitality: float = 1.0
    last_emit_turn: int = -1
    last_closure_turn: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "name": self.name,
            "description": self.description,
            "emitter_template_id": self.emitter_template_id,
            "responder_template_id": self.responder_template_id,
            "signal_type": self.signal_type,
            "closure_window": self.closure_window,
            "tags": list(self.tags),
            "emissions": self.emissions,
            "closures": self.closures,
            "misses": self.misses,
            "confidence": self.confidence,
            "vitality": self.vitality,
            "last_emit_turn": self.last_emit_turn,
            "last_closure_turn": self.last_closure_turn,
        }


# ---------------------------------------------------------------------------
# ModeTemplate — reflex whose fire activates a behavioural mode
# ---------------------------------------------------------------------------

@dataclass
class ModeTemplate:
    """Descriptor for a mode reflex (Level 5).

    A ModeTemplate maps to an underlying ReflexTemplate with kind=MODE,
    whose action_program is a single ``activate_mode`` node. The mode
    activation injects a prompt overlay (looked up by overlay_name) for
    ttl_turns turns, and is mutually exclusive with any mode_id in
    exclusive_with.
    """
    mode_id: str
    name: str
    description: str
    template_id: str
    overlay_name: str
    ttl_turns: int = 10
    cooldown_turns: int = 20
    horizon_turns: int = 10
    exclusive_with: List[str] = field(default_factory=list)
    priority: int = 50
    tags: List[str] = field(default_factory=list)

    # Aggregates maintained per mode template across activations
    activations: int = 0
    positive_activations: int = 0   # ΔU > 0 over horizon
    negative_activations: int = 0   # ΔU <= 0 over horizon

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "name": self.name,
            "description": self.description,
            "template_id": self.template_id,
            "overlay_name": self.overlay_name,
            "ttl_turns": self.ttl_turns,
            "cooldown_turns": self.cooldown_turns,
            "horizon_turns": self.horizon_turns,
            "exclusive_with": list(self.exclusive_with),
            "priority": self.priority,
            "tags": list(self.tags),
            "activations": self.activations,
            "positive_activations": self.positive_activations,
            "negative_activations": self.negative_activations,
        }


@dataclass
class ModeActivation:
    """Runtime record of an active mode window for one agent."""
    agent_id: str
    mode_id: str
    template_id: str
    activated_turn: int
    expires_turn: int
    horizon_turns: int
    # Utility accumulators populated per turn within the window
    u_cost: float = 0.0
    u_surv: float = 0.0
    u_task: float = 0.0
    u_coord: float = 0.0
    u_thrash: float = 0.0

    def delta_total(self) -> float:
        return self.u_cost + self.u_surv + self.u_task + self.u_coord - self.u_thrash


# ---------------------------------------------------------------------------
# AgentReflexEntry — per-agent adoption record
# ---------------------------------------------------------------------------

@dataclass
class AgentReflexEntry:
    """Per-agent record of an adopted reflex template."""
    template: ReflexTemplate
    adopted_turn: int = 0
    fire_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    last_fired_turn: int = -1
    enabled: bool = True
    program_state: Dict[str, Any] = field(default_factory=dict)
    local_overrides: Dict[str, Any] = field(default_factory=dict)

    # L4/L5 per-agent state
    # Rolling utility traces (keep last N for sliding window aggregates)
    recent_traces: List[UtilityTrace] = field(default_factory=list)
    # For pair emitters: maps emit_id -> turn_emitted for pending closures
    pending_emits: Dict[str, int] = field(default_factory=dict)
    # Last time this entry contributed a positive outcome (for vitality)
    last_positive_turn: int = -1

    def record_trace(self, trace: UtilityTrace, max_keep: int = 20) -> None:
        self.recent_traces.append(trace)
        if len(self.recent_traces) > max_keep:
            self.recent_traces = self.recent_traces[-max_keep:]


# ---------------------------------------------------------------------------
# TriggerResult — output of reflex evaluation
# ---------------------------------------------------------------------------

@dataclass
class TriggerResult:
    """Result of evaluating reflex templates for one agent in one turn."""
    fired: bool = False
    template_id: str = ""
    template_name: str = ""
    action: Optional[Dict[str, Any]] = None
    expected_free: bool = False
    matched_signal: Optional[ReflexSignal] = None
    # L5: when the winning reflex is a mode, this carries the mode_id just
    # activated. Callers should treat this as a "no world action" fire.
    activated_mode: str = ""
    # L4: adoption score snapshot for telemetry/debugging
    adoption_score: float = 0.0
    kind: str = ReflexKind.SELF.value
