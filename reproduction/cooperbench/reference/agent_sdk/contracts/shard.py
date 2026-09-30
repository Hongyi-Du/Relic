"""ExperienceShard + supporting DTOs.

B8:  auto-mint in __post_init__; mismatched explicit shard_id raises ValueError.
B20: shard IDs prefixed "shard:sha256:" via mint_id(namespace="shard").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional

from .hashing import mint_id


@dataclass(frozen=True)
class ShardRating:
    valence: Literal["good", "bad", "neutral"]
    confidence: float
    reason: str

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


@dataclass(frozen=True)
class MetricBound:
    min: float
    max: float
    direction: Literal["higher_better", "lower_better"]


@dataclass(frozen=True)
class ExperienceShard:
    shard_id: str
    agent_id: str
    turn: int
    # DEPRECATED after Step E (populated None during dual-write window)
    task: Optional[str]
    obstacle: Optional[str]
    desired_help: Optional[str]
    planner_text: str
    component_id_used: Optional[str]
    rating: Optional[ShardRating]
    metric_vector_delta: Mapping[str, float]
    metric_bounds: Mapping[str, MetricBound]
    cohort_members: tuple[str, ...]
    # MODIFIED: triggered_by extended to 4-way literal; "update_plan" preserved
    triggered_by: Literal["update_plan", "planner_update", "share_experience", "gene_upload"]
    emission_ts: str
    group_id: Optional[int] = None
    # NEW: universal open-tasks header — included in every shard regardless of terminal
    open_tasks_snapshot: tuple[dict, ...] = field(default_factory=tuple)
    # NEW: per-terminal payload (frozen MappingProxyType)
    terminal_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Freeze mappings as read-only proxies.
        object.__setattr__(
            self, "metric_vector_delta", MappingProxyType(dict(self.metric_vector_delta))
        )
        object.__setattr__(
            self, "metric_bounds", MappingProxyType(dict(self.metric_bounds))
        )
        object.__setattr__(
            self, "terminal_payload", MappingProxyType(dict(self.terminal_payload))
        )
        # Mint ID from canonical payload and enforce B8 mint-or-raise.
        payload = self._mint_payload()
        minted = mint_id(payload, namespace="shard")
        if not self.shard_id:
            object.__setattr__(self, "shard_id", minted)
        elif self.shard_id != minted:
            raise ValueError(
                f"shard_id {self.shard_id!r} does not match canonical mint {minted!r}"
            )

    def to_dict(self) -> dict:
        """JSON-safe serialization — converts MappingProxyType fields to plain dicts."""
        return {
            "shard_id": self.shard_id,
            "agent_id": self.agent_id,
            "turn": self.turn,
            "task": self.task,
            "obstacle": self.obstacle,
            "desired_help": self.desired_help,
            "planner_text": self.planner_text,
            "component_id_used": self.component_id_used,
            "rating": (
                {
                    "valence": self.rating.valence,
                    "confidence": self.rating.confidence,
                    "reason": self.rating.reason,
                }
                if self.rating is not None
                else None
            ),
            "metric_vector_delta": dict(self.metric_vector_delta),
            "metric_bounds": {
                k: {"min": v.min, "max": v.max, "direction": v.direction}
                for k, v in self.metric_bounds.items()
            },
            "cohort_members": list(self.cohort_members),
            "triggered_by": self.triggered_by,
            "emission_ts": self.emission_ts,
            # NEW fields
            "open_tasks_snapshot": [dict(t) for t in self.open_tasks_snapshot],
            "terminal_payload": dict(self.terminal_payload),
        }

    def _mint_payload(self) -> dict:
        payload = {
            "agent_id": self.agent_id,
            "turn": self.turn,
            "planner_text": self.planner_text,
            "rating": self.rating,
            "metric_vector_delta": dict(self.metric_vector_delta),
        }
        # Only include group_id in minting payload when explicitly set (backcompat).
        if self.group_id is not None:
            payload["group_id"] = self.group_id
        return payload
