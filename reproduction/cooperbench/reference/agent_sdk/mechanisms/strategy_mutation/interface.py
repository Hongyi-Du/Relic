"""
StrategyMutationPort — protocol for pluggable trait/persona mutation.

The Agent SDK calls this port during reproduction to:
  1. Apply mutations to parent attributes → child attributes
  2. Convert persona traits to prompt text
  3. Modify lifecycle parameters (reproduction cost, lifespan)

Concrete implementation: ``environments/nature_env/backend/agents/enhanced_mutations.py``
"""
from __future__ import annotations

from typing import Dict, Any, Protocol, runtime_checkable

from agent_sdk.mechanisms.strategy_mutation.types import MutationResult, PersonaTraits


@runtime_checkable
class StrategyMutationPort(Protocol):
    """
    Minimal contract for trait mutation during reproduction.
    """

    def apply_mutations(
        self,
        parent_attributes: Dict[str, Any],
        mutation_config: Dict[str, Any],
    ) -> MutationResult:
        """
        Apply mutations to parent attributes and return the result.

        Returns MutationResult with new_attributes, mutations_applied delta,
        and special_mutations list.
        """
        ...

    def get_persona_prompt(self, persona: PersonaTraits) -> str:
        """
        Convert persona trait scores into natural-language prompt text.

        Used to inject personality-coloured instructions into the agent's
        system prompt.
        """
        ...

    def format_special_mutations(self, mutations: list) -> str:
        """
        Format special mutation tags (e.g. RECIPE_UNLOCK, STRATEGIC_VISION)
        into prompt-injectable text for the new agent's initial context.
        """
        ...

    def get_modified_reproduction_cost(self, base_cost: float) -> float:
        """Return the mutated reproduction energy cost."""
        ...

    def get_modified_lifespan(self, base_lifespan: int) -> int:
        """Return the mutated max lifespan (in turns)."""
        ...
