"""Small, stable contracts shared by OrgEnv candidate producers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CandidateSource(str, Enum):
    """Origin of a feasible action candidate, retained for policy telemetry."""

    ENVIRONMENT = "environment"
    NEED = "need"
    PERSONA = "persona"
    SOCIAL = "social"
    INSTITUTION = "institution"
    WISH = "wish"


@dataclass
class ActionCandidate:
    """A feasible action before SDL scoring and selection."""

    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source: CandidateSource = CandidateSource.ENVIRONMENT
    rationale: str = ""
    target_uid: str | None = None

    def to_action_dict(self) -> dict[str, Any]:
        return {"type": self.action_type, "parameters": dict(self.parameters)}
