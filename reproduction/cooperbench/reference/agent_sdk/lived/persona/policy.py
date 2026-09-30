"""Profile-to-Policy (§13) — now the Persona-Conditioned Bounded Softmax Policy.

The decision scorer is the mechanism the whole research claim hinges on (§17.3):
the profile must *actually* change the utility, verifiable by counterfactual
editing. As of the PCBSP rewrite, the real model lives in
:mod:`agent_sdk.lived.persona.pcbsp` (SurvivalScore + TraitScore + RelationScore +
MoodShift + EpisodeValue − CostScore, with §6 transforms, §13 survival guard,
§14 satisficing and §15 seeded softmax).

This module is kept as the stable public surface / back-compat shim:
  * ``ProfileToPolicy``      → alias of :class:`~agent_sdk.lived.persona.pcbsp.PCBSPPolicy`
  * ``profile_to_weights``   → re-exported from :mod:`agent_sdk.lived.core.wmatrix`
  * ``DecisionContext`` / ``PolicyTrace`` → re-exported from pcbsp
  * ``TriggerContext``       → retained (controller passes it through; PCBSP
    realizes thresholds via the survival guard + relation + mood terms instead)
  * ``mood_shift`` / ``sampling_temperature`` → thin function shims over PCBSP

The LLM proposes candidates + rationales upstream; this scorer only re-weights
and samples. It never invents actions (§0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_sdk.lived.core.contracts import ActionCandidate, MoodState, ProfileVector
# Real model:
from agent_sdk.lived.persona.pcbsp import (  # noqa: F401
    DecisionContext,
    PCBSPPolicy,
    PolicyTrace,
    ProfileToPolicy,
)
# Registered weight matrix (raw-trait, §8):
from agent_sdk.lived.core.wmatrix import profile_to_weights  # noqa: F401

_DEFAULT_POLICY = PCBSPPolicy()


@dataclass
class TriggerContext:
    """Back-compat: state scalars the old threshold rules consulted. PCBSP no
    longer uses explicit threshold boosts (the survival guard + RelationScore +
    MoodShift cover them), but the controller still constructs/threads this, so
    it is retained as an inert carrier."""
    observed_inequality: float = 0.0
    leader_legitimacy: float = 1.0
    target_has_betrayed: bool = False
    external_threat: float = 0.0
    unknown_resource_nearby: bool = False


def mood_shift(mood: MoodState, candidate: ActionCandidate) -> float:
    """Shim: §10 MoodShift for one candidate (delegates to PCBSP)."""
    return _DEFAULT_POLICY.mood_shift(mood, candidate)


def sampling_temperature(profile: ProfileVector, mood: MoodState, base: float = 0.0) -> float:
    """Shim: §15.1 temperature (delegates to PCBSP). ``base`` is ignored
    (PCBSP uses its own TAU_BASE) and kept only for signature compatibility."""
    return _DEFAULT_POLICY.temperature(profile, mood)
