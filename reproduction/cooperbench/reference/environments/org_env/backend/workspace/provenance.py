"""Provenance + content-hash utilities (DESIGN env_org §21/§39, acceptance ⑩).

Every shareable object carries a content_hash + a provenance chain so the event
graph can trace "where did this file/result/claim come from". Deterministic:
same content -> same hash (reproducibility, acceptance ⑫).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


def content_hash(payload: Any) -> str:
    """Stable short hash of any JSON-serializable payload."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class ProvenanceStep:
    tick: int
    actor_id: str
    action: str                       # created | edited | shared | derived_from | reviewed | ...
    source_object_id: str = ""        # for derived/shared-from links
    note: str = ""


@dataclass
class Provenance:
    """An append-only chain of who-did-what to an object."""
    origin_actor_id: str = ""
    origin_tick: int = 0
    steps: List[ProvenanceStep] = field(default_factory=list)

    def add(self, *, tick: int, actor_id: str, action: str,
            source_object_id: str = "", note: str = "") -> "Provenance":
        self.steps.append(ProvenanceStep(tick=tick, actor_id=actor_id, action=action,
                                         source_object_id=source_object_id, note=note))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_actor_id": self.origin_actor_id,
            "origin_tick": self.origin_tick,
            "steps": [s.__dict__ for s in self.steps],
        }


__all__ = ["content_hash", "Provenance", "ProvenanceStep"]
