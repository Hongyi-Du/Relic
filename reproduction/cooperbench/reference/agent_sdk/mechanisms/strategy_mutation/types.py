"""
Strategy Mutation types.

PersonaTraits and MutationResult are the vocabulary between the Agent SDK
and any trait-mutation backend.  All fields are environment-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PersonaTraits:
    """Mutable personality traits that influence agent behaviour."""
    selfishness: float = 3.0          # 1-5 scale
    explorativeness: float = 3.0      # 1-5 scale
    social_disposition: float = 3.0   # 1-5 scale

    def to_dict(self) -> Dict[str, float]:
        return {
            "selfishness": self.selfishness,
            "explorativeness": self.explorativeness,
            "social_disposition": self.social_disposition,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonaTraits":
        return cls(
            selfishness=float(d.get("selfishness", 3.0)),
            explorativeness=float(d.get("explorativeness", 3.0)),
            social_disposition=float(d.get("social_disposition", 3.0)),
        )


@dataclass
class MutationResult:
    """Output of a mutation pass."""
    new_attributes: Dict[str, Any] = field(default_factory=dict)
    mutations_applied: Dict[str, Any] = field(default_factory=dict)   # trait -> delta
    special_mutations: List[str] = field(default_factory=list)        # e.g. ["RECIPE_UNLOCK:stone_axe"]
