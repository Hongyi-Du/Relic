"""Internal Growth Module — agent skill growth, domain reputation, informal authority.

Agents accumulate professional skill + domain reputation from event evidence; informal
authority is derived from skill+reputation+usage+ownership and softly biases reviewer
routing / objection weight / task preference. Core personality profile stays fixed.
See docs/design/env_org.md §61.
"""
from environments.org_env.growth.appraiser import GrowthAppraiser
from environments.org_env.growth.authority import (
    availability,
    domain_for_action,
    objection_weight,
    policy_authority_bonus,
    select_reviewers,
)
from environments.org_env.growth.objects import (
    REPUTATION_DOMAINS,
    SKILL_DOMAINS,
    GrowthEvent,
    GrowthSignal,
    new_authority,
    new_reputation,
)
from environments.org_env.growth.reconciler import GrowthReconciler

__all__ = [
    "GrowthAppraiser", "GrowthReconciler", "GrowthSignal", "GrowthEvent",
    "SKILL_DOMAINS", "REPUTATION_DOMAINS", "new_reputation", "new_authority",
    "availability", "domain_for_action", "objection_weight", "policy_authority_bonus",
    "select_reviewers",
]
