"""Action annotation (PCBSP §4/§5) + shared candidate-pool helpers (§3).

Two pieces:

  * :func:`build_candidate_pool` — wraps a list of *state-derived* action types
    into :class:`ActionCandidate` objects. Candidate generation is env-specific
    and profile-INDEPENDENT (§3.1): all agents in the same state see the same
    pool. This helper just enforces that shape so the shared-pool audit (§17.5)
    can assert it.

  * :class:`RuleBasedAnnotator` — a :class:`FeatureExtractorPort` that maps
    common action types to fixed :class:`ActionFeatures` (§19 Decision E:
    rule-based for common actions, LLM only for novel/wish actions). It enforces
    the §5 annotation rules: only schema features, only the discrete intensities
    {0,.25,.5,.75,1.0}, and (critically) annotation does NOT depend on persona.

:func:`snap_intensity` / :func:`sanitize_features` let an LLM annotator's raw
output be coerced to the legal discrete grid before it enters the policy.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_sdk.lived.core.contracts import ActionCandidate, ActionFeatures, CandidateSource
from agent_sdk.lived.core.schema import FEATURE_INTENSITIES, FEATURE_SPECS


# --------------------------------------------------------------------------- #
# §3 shared candidate pool helper
# --------------------------------------------------------------------------- #
def build_candidate_pool(
    action_specs: List[Any],
    *,
    default_source: CandidateSource = CandidateSource.ENVIRONMENT,
) -> List[ActionCandidate]:
    """Wrap state-derived action specs into candidates. Each spec is either an
    ``action_type`` str or a ``(action_type, parameters, target_uid)`` tuple /
    dict. The result is profile-independent (§3.1)."""
    pool: List[ActionCandidate] = []
    for spec in action_specs:
        if isinstance(spec, str):
            pool.append(ActionCandidate(action_type=spec, source=default_source))
        elif isinstance(spec, dict):
            pool.append(ActionCandidate(
                action_type=spec["action_type"],
                parameters=dict(spec.get("parameters", {})),
                source=spec.get("source", default_source),
                target_uid=spec.get("target_uid"),
            ))
        elif isinstance(spec, (tuple, list)):
            at = spec[0]
            params = spec[1] if len(spec) > 1 else {}
            tgt = spec[2] if len(spec) > 2 else None
            pool.append(ActionCandidate(action_type=at, parameters=dict(params or {}),
                                        source=default_source, target_uid=tgt))
    return pool


# --------------------------------------------------------------------------- #
# §5 intensity hygiene
# --------------------------------------------------------------------------- #
def snap_intensity(x: float) -> float:
    """Snap a raw value to the nearest legal discrete intensity (§5)."""
    return min(FEATURE_INTENSITIES, key=lambda c: abs(c - max(0.0, min(1.0, x))))


def sanitize_features(d: Dict[str, float]) -> ActionFeatures:
    """Build an :class:`ActionFeatures` from a raw {feature: value} mapping,
    dropping unknown keys and snapping every value to the discrete grid (§5
    rules 1+2). This is the gate an LLM annotation must pass through."""
    clean: Dict[str, float] = {}
    for k, v in (d or {}).items():
        if k in FEATURE_SPECS:
            try:
                clean[k] = snap_intensity(float(v))
            except (TypeError, ValueError):
                continue
    return ActionFeatures(**clean)


# --------------------------------------------------------------------------- #
# §4 rule-based annotation table for common actions (discrete intensities)
# --------------------------------------------------------------------------- #
# Keyed by action_type. Values use only schema features + {0,.25,.5,.75,1.0}.
# These are deterministic and persona-INDEPENDENT (§5 rule 3).
DEFAULT_ACTION_FEATURES: Dict[str, Dict[str, float]] = {
    # physical
    "move": {"energy_cost": 0.25, "time_cost": 0.25},
    "gather": {"resource_gain": 0.75, "energy_cost": 0.5, "time_cost": 0.5},
    "eat": {"survival_gain": 0.75, "energy_gain": 0.75},
    "rest": {"energy_gain": 0.75, "time_cost": 0.5},
    "return_home": {"survival_gain": 0.25, "energy_cost": 0.25, "time_cost": 0.25},
    "deposit": {"public_good_gain": 0.5, "future_security_gain": 0.5, "storage_value": 0.25, "time_cost": 0.25},
    "withdraw": {"resource_gain": 0.5, "private_gain": 0.5},
    "inspect": {"information_gain": 0.75, "experiment_value": 0.5, "time_cost": 0.25},
    "pick_up": {"resource_gain": 0.25, "time_cost": 0.25},
    "drop": {"time_cost": 0.25},
    # communication
    "comm_local": {"information_gain": 0.25, "time_cost": 0.25},
    "ask_help": {"reciprocity_gain": 0.25, "reputation_risk": 0.25, "time_cost": 0.25},
    "tell_info": {"altruistic_gain": 0.25, "information_gain": 0.25, "trust_gain": 0.25},
    "camp_announce": {"coordination_gain": 0.75, "dominance_gain": 0.5, "reputation_risk": 0.25},
    "public_mark": {"violation_detection": 0.5, "reputation_gain": 0.5, "institution_gain": 0.5},
    # affordance / invention
    "use_basket": {"resource_gain": 0.5, "future_security_gain": 0.25},
    "repair_basket": {"material_cost": 0.5, "skill_gain": 0.25, "time_cost": 0.5},
    "attempt_material_session": {"experiment_value": 0.75, "novelty_value": 0.75, "skill_gain": 0.5, "uncertainty_cost": 0.5, "time_cost": 0.5},
    "test_prototype": {"experiment_value": 0.75, "information_gain": 0.5, "uncertainty_cost": 0.5, "skill_gain": 0.5},
    # episode
    "continue_session": {"experiment_value": 0.5, "skill_gain": 0.25, "time_cost": 0.5},
    "abandon_session": {"time_cost": 0.0, "autonomy_gain": 0.25},
    "continue_teaching": {"altruistic_gain": 0.5, "skill_gain": 0.5, "reputation_gain": 0.25, "time_cost": 0.5},
    "support_trial_rule": {"rule_compliance": 0.75, "institution_gain": 0.5, "public_good_gain": 0.5},
    # civic
    "support_proposal": {"institution_gain": 0.5, "public_good_gain": 0.5, "conformity_cost": 0.0},
    "oppose_proposal": {"autonomy_gain": 0.5, "conflict_risk": 0.5, "conformity_cost": 0.5},
    "accuse_violation": {"violation_detection": 0.75, "fairness_gain": 0.5, "conflict_risk": 0.75, "reputation_risk": 0.5},
    "request_compensation": {"fairness_gain": 0.5, "reciprocity_gain": 0.5, "conflict_risk": 0.5},
    "appeal": {"fairness_gain": 0.5, "institution_gain": 0.25, "time_cost": 0.5},
    # wish
    "express_need": {"information_gain": 0.25, "reputation_risk": 0.25},
    "start_material_session": {"experiment_value": 0.75, "novelty_value": 0.5, "uncertainty_cost": 0.5, "time_cost": 0.5},
    "start_civic_session": {"institution_gain": 0.5, "public_good_gain": 0.5, "conflict_risk": 0.25},
    # self-interest
    "hoard_food": {"private_gain": 0.75, "opportunistic_gain": 0.5, "fairness_cost": 0.5},
    "over_withdraw": {"private_gain": 0.75, "opportunistic_gain": 0.75, "fairness_cost": 0.75, "public_harm": 0.5, "violation_gain": 0.5},
}


class RuleBasedAnnotator:
    """Deterministic, persona-independent feature annotator for common actions
    (§19 Decision E). Implements :class:`FeatureExtractorPort`.

    Unknown action types fall back to ``fallback`` (another extractor, e.g. an
    LLM annotator) if provided, else an all-zero vector. All emitted values are
    on the legal discrete grid (§5)."""

    def __init__(self, table: Optional[Dict[str, Dict[str, float]]] = None,
                 fallback: Optional[Any] = None):
        self.table = dict(DEFAULT_ACTION_FEATURES)
        if table:
            self.table.update(table)
        self.fallback = fallback

    def extract(self, *, agent_id: str, candidate: ActionCandidate, state: Any) -> ActionFeatures:
        spec = self.table.get(candidate.action_type)
        if spec is not None:
            return sanitize_features(spec)
        if self.fallback is not None:
            return self.fallback.extract(agent_id=agent_id, candidate=candidate, state=state)
        return ActionFeatures()
