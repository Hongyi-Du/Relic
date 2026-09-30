"""
Observation DTOs: the environment's output to the agent each turn.

Consumed by:
  - PerceptorV2 (percept.py)  — renders signals into the LLM perception prompt
  - Signal metadata for tool gating and CM retrieval
  - NatureEnvAdapter (adapter.py) — builds EnvObservation per turn
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Signal ontology slots
# ---------------------------------------------------------------------------

class SignalLevel:
    """Semantic intensity levels for pressure-like slots."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def choices(cls):
        # returns list[str]
        return [cls.NONE, cls.LOW, cls.MEDIUM, cls.HIGH, cls.CRITICAL]


class ProgressState:
    """Lifecycle of task progress, independent from environment vocabulary."""
    STALLED = "stalled"
    ACTIVE = "active"
    ADVANCING = "advancing"
    COMPLETED = "completed"

    @classmethod
    def choices(cls):
        # returns list[str]
        return [cls.STALLED, cls.ACTIVE, cls.ADVANCING, cls.COMPLETED]


def _coerce_enum(value, allowed, fallback):
    # returns str
    text = str(value or "").strip().lower()
    return text if text in allowed else fallback


def _coerce_keywords(value):
    # returns tuple[str, ...]
    if value is None:
        return ()
    if isinstance(value, tuple):
        raw = value
    elif isinstance(value, list):
        raw = tuple(value)
    else:
        raw = (value,)
    out = []
    for item in raw:
        token = str(item or "").strip().lower()
        if token and token not in out:
            out.append(token)
    return tuple(out)


@dataclass(frozen=True)
class DomainSignalsV1:
    """
    Environment-agnostic signal ontology (v1, constrained slots).

    Produced by env-specific signal mappers (e.g. signal_mapper.py).
    Consumed by:
      - Used for tool gating and collective memory queries
      - PerceptorV2._build_signals() — non-default slots rendered into LLM prompt
    """
    ontology_version: str = "1.0"

    # -- Semantic pressure/state slots --
    # Each is mapped from env-specific state by the signal mapper.
    resource_pressure: str = SignalLevel.NONE   # resource availability pressure
    danger_level: str = SignalLevel.NONE        # environmental danger assessment
    coordination_need: str = SignalLevel.NONE   # nearby living agents
    progress_state: str = ProgressState.ACTIVE  # last action outcome

    # -- Controlled keyword channels --
    # evidence_keywords: boolean env-state tags used for tool gating and collective memory queries
    #   (e.g. "near_storage" unlocks store/retrieve).
    # intent_keywords: action-hint tags derived from situational conditions.
    intent_keywords: tuple = ()
    evidence_keywords: tuple = ()

    # -- Free-form situational keywords (not filtered by controlled set) --
    # Used exclusively for CM retrieval enrichment. Populated by env-specific
    # signal mappers with dynamic content: inventory items, nearby entity types,
    # area quadrants, etc.
    context_keywords: tuple = ()

    def __post_init__(self):
        object.__setattr__(
            self,
            "resource_pressure",
            _coerce_enum(self.resource_pressure, set(SignalLevel.choices()), SignalLevel.NONE),
        )
        object.__setattr__(
            self,
            "danger_level",
            _coerce_enum(self.danger_level, set(SignalLevel.choices()), SignalLevel.NONE),
        )
        object.__setattr__(
            self,
            "coordination_need",
            _coerce_enum(self.coordination_need, set(SignalLevel.choices()), SignalLevel.NONE),
        )
        object.__setattr__(
            self,
            "progress_state",
            _coerce_enum(self.progress_state, set(ProgressState.choices()), ProgressState.ACTIVE),
        )
        object.__setattr__(self, "intent_keywords", _coerce_keywords(self.intent_keywords))
        object.__setattr__(self, "evidence_keywords", _coerce_keywords(self.evidence_keywords))
        object.__setattr__(self, "context_keywords", _coerce_keywords(self.context_keywords))

    def to_dict(self):
        # returns Dict[str, Any]
        return {
            "ontology_version": self.ontology_version,
            "resource_pressure": self.resource_pressure,
            "danger_level": self.danger_level,
            "coordination_need": self.coordination_need,
            "progress_state": self.progress_state,
            "intent_keywords": list(self.intent_keywords),
            "evidence_keywords": list(self.evidence_keywords),
            "context_keywords": list(self.context_keywords),
        }


# ---------------------------------------------------------------------------
# Agent view (other agents visible to this agent)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentView:
    """Minimal representation of another agent visible in the observation."""
    agent_id: str
    name: str
    x: int
    y: int
    distance: int


# ---------------------------------------------------------------------------
# Top-level observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvObservation:
    """
    The complete observation delivered to an agent each turn.

    * ``view`` — pre-rendered perception text from the environment
    * ``signals`` — structured domain signals for the intent router
    * ``feedback`` — textual result of the agent's previous action
    * ``turn`` — current simulation turn number
    * ``agent_views`` — structured list of visible agents
    * ``possible_actions`` — interaction options available this turn
    * ``metadata`` — arbitrary env-specific extras (e.g. season name)
    """
    view: str = ""
    signals: DomainSignalsV1 = field(default_factory=DomainSignalsV1)
    feedback: str = ""
    turn: int = 0
    agent_views: tuple = ()         # Tuple[AgentView, ...]
    possible_actions: tuple = ()    # Tuple[Dict, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    unlocked_tools: tuple = ()     # Tuple[ToolUnlock, ...]
    library_announcements: tuple = ()   # tuple of LibraryAnnouncement (agent_sdk.contracts.announcement); duck-typed to avoid agent_sdk->harness_sdk import (B1)
    # Slim view used in the Reflect phase of the same turn. Strips the
    # world-physical sections (status, atlas, inventory, nearby resources,
    # etc.) that the Act-phase user_msg already showed — keeps only the
    # comm-state sections that may have shifted between Act commits and
    # Reflect (broadcasts, sessions, invitations, DMs, evolved menu).
    # When empty, callers fall back to ``view``.
    reflect_view: str = ""
