"""
Asset types for Gene+Event collective memory model.
Schema version: 1.6.0
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum


SCHEMA_VERSION = "1.6.0"


class AssetCategory(str, Enum):
    # Canonical 4 (2026-04-20): what LLMs are now asked to produce.
    OBSERVATION = "observation"
    STRATEGY = "strategy"
    WARNING = "warning"
    RECIPE = "recipe"
    # Pre-1.6 values (kept for backward compat loading of old stores).
    REPAIR = "repair"
    OPTIMIZE = "optimize"
    INNOVATE = "innovate"
    # T1.4 additive extensions: harness_sdk component categories.
    # Existing readers comparing by .value continue to work unchanged.
    COMPONENT_COMM_TOPOLOGY = "component_comm_topology"
    COMPONENT_PLAYBOOK_CHANNEL = "component_playbook_channel"
    COMPONENT_REFLEX_TEMPLATE = "component_reflex_template"
    # Line B addition:
    FIX_PATCH = "fix_patch"   # immune-memory stored corrective patch


# Bidirectional aliases: old category names → canonical 4.
# Unknown strings pass through unchanged so custom tags survive.
_CATEGORY_ALIAS: Dict[str, str] = {
    "observation": AssetCategory.OBSERVATION.value,
    "strategy": AssetCategory.STRATEGY.value,
    "warning": AssetCategory.WARNING.value,
    "recipe": AssetCategory.RECIPE.value,
    "repair": AssetCategory.REPAIR.value,
    "optimize": AssetCategory.OPTIMIZE.value,
    "innovate": AssetCategory.INNOVATE.value,
    # Short-form aliases an LLM may produce.
    "obs": AssetCategory.OBSERVATION.value,
    "strat": AssetCategory.STRATEGY.value,
    "warn": AssetCategory.WARNING.value,
    "rec": AssetCategory.RECIPE.value,
}


def canonicalize_category(raw: Any) -> str:
    """Return a canonical category string (plain str, lowercase).

    Why not return the enum member? ``AssetCategory(str, Enum)`` on
    Python 3.12 renders ``str(AssetCategory.RECIPE) == 'AssetCategory.RECIPE'``
    (qualified name), which silently poisons hashes and frontend chips.
    Returning ``.value`` makes category a stable plain str everywhere.
    """
    if raw is None:
        return AssetCategory.OBSERVATION.value
    raw_str = str(raw).strip().lower()
    if raw_str.startswith("assetcategory."):
        raw_str = raw_str[len("assetcategory."):]
    return _CATEGORY_ALIAS.get(raw_str, raw_str)


class AssetStatus(str, Enum):
    ACTIVE = "active"
    DISPUTED = "disputed"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    # T1.4 additive extensions: harness_sdk component lifecycle statuses.
    # Existing readers comparing by .value continue to work unchanged.
    TRIALING = "trialing"
    STABLE = "stable"
    NICHE_SPECIALIST = "niche_specialist"
    RETIRED = "retired"


class FeedbackType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"


class DisputeStatus(str, Enum):
    RECEIPT = "receipt"
    BATCH = "batch"
    OPEN = "open"
    DISCUSS = "discuss"
    PENDING_OWNER = "pending_owner"
    CLOSED = "closed"


class OwnerAction(str, Enum):
    DELETE = "delete"
    EDIT = "edit"
    SPLIT = "split"
    DEFER = "defer"


def _coerce_coordinate(raw: Any) -> Optional[Tuple[int, int]]:
    """Coerce serialized coord (list|tuple|None) into an (int,int) tuple.

    Defensive: bad shapes silently become None so pre-1.6 stores load
    cleanly even if the field was missing or garbage.
    """
    if raw is None:
        return None
    try:
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return (int(raw[0]), int(raw[1]))
    except (ValueError, TypeError):
        return None
    return None


@dataclass
class Gene:
    """Transferable strategy — what/when/why.

    `summary` is a short one-line title (agent-supplied `name`); `body` is the
    longer narrative description the agent wrote on upload (may repeat/expand
    `summary`). Both are round-tripped through persistence so the frontend can
    render the full entry. `body` defaults to empty for backward compatibility
    with pre-1.5 serialized stores that only had `summary`.

    `origin_coordinate` (1.6.0): where the upload happened in the spatial
    env. Non-spatial envs pass ``None`` → spatial multiplier decays to 1.0
    at retrieve time. Stored as a 2-tuple; serialized as a 2-element list.
    """
    asset_id: str
    schema_version: str = SCHEMA_VERSION
    category: str = AssetCategory.INNOVATE
    signals_match: List[str] = field(default_factory=list)
    summary: str = ""
    body: str = ""
    origin_coordinate: Optional[Tuple[int, int]] = None
    contributor_id: str = ""
    turn_created: int = 0
    event_ids: List[str] = field(default_factory=list)
    confidence: float = 0.3
    action_related: bool = False
    status: str = AssetStatus.ACTIVE
    revision: int = 1
    superseded_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Gene",
            "asset_id": self.asset_id,
            "schema_version": self.schema_version,
            "category": self.category,
            "signals_match": self.signals_match,
            "summary": self.summary,
            "body": self.body,
            "origin_coordinate": (
                list(self.origin_coordinate)
                if self.origin_coordinate is not None
                else None
            ),
            "contributor_id": self.contributor_id,
            "turn_created": self.turn_created,
            "event_ids": list(self.event_ids),
            "confidence": self.confidence,
            "action_related": self.action_related,
            "status": self.status,
            "revision": self.revision,
            "superseded_by": self.superseded_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Gene":
        return cls(
            asset_id=d["asset_id"],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            category=d.get("category", AssetCategory.INNOVATE),
            signals_match=d.get("signals_match", []),
            summary=d.get("summary", ""),
            body=d.get("body", ""),
            origin_coordinate=_coerce_coordinate(d.get("origin_coordinate")),
            contributor_id=d.get("contributor_id", ""),
            turn_created=d.get("turn_created", 0),
            event_ids=d.get("event_ids", []),
            confidence=d.get("confidence", 0.3),
            action_related=d.get("action_related", False),
            status=d.get("status", AssetStatus.ACTIVE),
            revision=d.get("revision", 1),
            superseded_by=d.get("superseded_by"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Event:
    """Single application evidence — context, trigger, outcome."""
    asset_id: str
    schema_version: str = SCHEMA_VERSION
    gene_id: str = ""
    trigger: List[str] = field(default_factory=list)
    summary: str = ""
    contributor_id: str = ""
    turn_created: int = 0
    confidence: float = 0.3
    action_related: bool = False
    domain_signals: Dict[str, Any] = field(default_factory=dict)
    intent_action: str = ""
    env_result: str = ""
    env_fingerprint: Dict[str, Any] = field(default_factory=dict)
    trace_hash: str = ""
    outcome: Dict[str, Any] = field(default_factory=lambda: {"status": "unknown", "score": 0.0})
    blast_radius: Dict[str, Any] = field(default_factory=dict)
    success_streak: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    auto_attributed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Event",
            "asset_id": self.asset_id,
            "schema_version": self.schema_version,
            "gene_id": self.gene_id,
            "trigger": self.trigger,
            "summary": self.summary,
            "contributor_id": self.contributor_id,
            "turn_created": self.turn_created,
            "confidence": self.confidence,
            "action_related": self.action_related,
            "domain_signals": self.domain_signals,
            "intent_action": self.intent_action,
            "env_result": self.env_result,
            "env_fingerprint": self.env_fingerprint,
            "trace_hash": self.trace_hash,
            "outcome": self.outcome,
            "blast_radius": self.blast_radius,
            "success_streak": self.success_streak,
            "metadata": self.metadata,
            "auto_attributed": self.auto_attributed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        return cls(
            asset_id=d["asset_id"],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            gene_id=d.get("gene_id", ""),
            trigger=d.get("trigger", []),
            summary=d.get("summary", ""),
            contributor_id=d.get("contributor_id", ""),
            turn_created=d.get("turn_created", 0),
            confidence=d.get("confidence", 0.3),
            action_related=d.get("action_related", False),
            domain_signals=d.get("domain_signals", {}),
            intent_action=d.get("intent_action", ""),
            env_result=d.get("env_result", ""),
            env_fingerprint=d.get("env_fingerprint", {}),
            trace_hash=d.get("trace_hash", ""),
            outcome=d.get("outcome", {"status": "unknown", "score": 0.0}),
            blast_radius=d.get("blast_radius", {}),
            success_streak=d.get("success_streak", 0),
            metadata=d.get("metadata", {}),
            auto_attributed=d.get("auto_attributed", True),
        )


@dataclass
class FeedbackRecord:
    voter_id: str
    asset_id: str
    feedback_type: str  # "like" | "dislike"
    turn: int
    reason: str = ""


@dataclass
class AssetFeedbackStats:
    like_count: int = 0
    dislike_count: int = 0
    unique_voters: Set[str] = field(default_factory=set)


@dataclass
class AgentCredibility:
    task_credibility: float = 0.5
    social_credibility: float = 0.5


@dataclass
class Comment:
    agent_id: str
    content: str
    turn: int
    is_representative: bool = False


@dataclass
class DisputeThread:
    asset_id: str
    status: str = DisputeStatus.RECEIPT
    op_agent_id: str = ""
    original_snapshot: Dict[str, Any] = field(default_factory=dict)
    like_representatives: List[str] = field(default_factory=list)
    dislike_representatives: List[str] = field(default_factory=list)
    comments: List[Comment] = field(default_factory=list)
    created_turn: int = 0
    resolved_turn: Optional[int] = None
    resolution: Optional[str] = None
    resolution_params: Optional[Dict[str, Any]] = None


@dataclass
class RankedGene:
    gene: Gene
    score: float = 0.0
    vitality: Optional[float] = None
