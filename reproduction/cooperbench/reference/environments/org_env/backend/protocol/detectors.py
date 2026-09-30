"""Protocol detectors (DESIGN env_org §59). v0 supports at least task_ownership,
experiment_tracking, meeting_cadence (acceptance ㉔).

A detector scans WORLD state for *structural* signals that an institution is
forming — independent of whether an agent explicitly proposed it — and returns a
``DetectionResult`` with an emergence level. If a matching explicit Protocol
exists in the registry, the registry's richer classification takes precedence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionResult:
    protocol_type: str
    detected: bool
    emergence_level: str          # none|weak|strong
    evidence: dict = field(default_factory=dict)


class ProtocolDetector:
    protocol_type = ""

    def detect(self, world: Any, registry: Any) -> DetectionResult:  # pragma: no cover
        raise NotImplementedError

    def _registry_level(self, registry: Any) -> str:
        for p in registry.protocols.values():
            if p.protocol_type == self.protocol_type:
                return registry.classify_emergence(p.protocol_id)
        return "none"


class TaskOwnershipDetector(ProtocolDetector):
    """owner labels appear + reused + unowned tasks shrink (§59.1)."""
    protocol_type = "task_ownership"

    def detect(self, world, registry):
        owners = dict(world.board.owners)
        owned = len(owners)
        unowned = sum(1 for t in world.tasks.values() if not t.owner_id)
        reuse = Counter(owners.values())
        repeated = any(c >= 2 for c in reuse.values())
        detected = owned >= 3 and repeated
        level = self._registry_level(registry)
        if level == "none" and detected:
            level = "weak"
        return DetectionResult(self.protocol_type, detected, level,
                               {"owned": owned, "unowned": unowned, "repeated_owner": repeated})


class ExperimentTrackingDetector(ProtocolDetector):
    """a tracker doc exists + multiple results logged to it (§59.2)."""
    protocol_type = "experiment_logging"

    def detect(self, world, registry):
        tracker = any(getattr(d, "doc_type", "") == "experiment_tracker"
                      for d in world.documents.values())
        logged = sum(1 for r in world.sandbox_system.results.values() if r.logged_to_tracker)
        detected = tracker and logged >= 2
        level = self._registry_level(registry)
        if level == "none" and detected:
            level = "weak"
        return DetectionResult(self.protocol_type, detected, level,
                               {"tracker_exists": tracker, "results_logged": logged})


class MeetingCadenceDetector(ProtocolDetector):
    """repeated meetings of one type + notes produced (§59.3)."""
    protocol_type = "daily_sync_cadence"

    def detect(self, world, registry):
        ms = world.meeting_system
        by_type = Counter(m.meeting_type for m in ms.meetings.values())
        recurring = [t for t, c in by_type.items() if c >= 2]
        notes = sum(1 for m in ms.meetings.values() if m.notes_doc_id)
        detected = bool(recurring) and notes >= 2
        level = self._registry_level(registry)
        if level == "none" and detected:
            level = "weak"
        return DetectionResult(self.protocol_type, detected, level,
                               {"recurring_types": recurring, "meetings_with_notes": notes})


DEFAULT_DETECTORS = (TaskOwnershipDetector(), ExperimentTrackingDetector(), MeetingCadenceDetector())


def run_all_detectors(world, registry):
    return [d.detect(world, registry) for d in DEFAULT_DETECTORS]


__all__ = [
    "DetectionResult", "ProtocolDetector", "TaskOwnershipDetector",
    "ExperimentTrackingDetector", "MeetingCadenceDetector",
    "DEFAULT_DETECTORS", "run_all_detectors",
]
