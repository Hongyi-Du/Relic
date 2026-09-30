"""Informal-authority influence (spec §11 / §15.3) — SOFT priors only.

Authority never decides an outcome; it shifts probabilities: who is more likely to be
asked to review, whose objection more likely escalates, which domain task an agent
leans toward. High fatigue/stress/overload lowers availability so authority can't make
one agent the answer to everything (anti "强者恒强").
"""
from __future__ import annotations

import math
from typing import Any, List, Optional

from environments.org_env.growth.objects import POLICY_AUTHORITY_DELTA, REVIEWER_TAU


def availability(agent) -> float:
    """§15.3: 1 - 0.5 fatigue - 0.3 stress - 0.2 overload, clipped to [0.1, 1.0]."""
    v = getattr(agent, "vitals", None)
    get = (lambda k: float(v.get(k, 0.0))) if v is not None and hasattr(v, "get") else \
        (lambda k: float(getattr(agent, "vitals", {}).get(k, 0.0)) if isinstance(getattr(agent, "vitals", None), dict) else 0.0)
    try:
        fatigue = get("fatigue")
        stress = get("stress")
        overload = get("workload") if get("workload") else get("overload")
    except Exception:
        fatigue = stress = overload = 0.0
    a = 1.0 - 0.5 * fatigue - 0.3 * stress - 0.2 * overload
    return max(0.1, min(1.0, a))


def authority_in(agent, domain: str) -> float:
    return float(getattr(agent, "authority", {}).get(domain, 0.0) or 0.0)


def domain_for_action(action_type: str) -> Optional[str]:
    """The dominant reputation domain an action contributes to (for routing/policy)."""
    from environments.org_env.growth.appraiser import _ACTION_DOMAINS
    rep = _ACTION_DOMAINS.get(action_type, ({}, {}))[1]
    return max(rep, key=rep.get) if rep else None


def select_reviewers(world: Any, *, domain: Optional[str], exclude: List[str], k: int = 1,
                     rng=None) -> List[str]:
    """§11.1: authority-weighted (softmax over a_{i,d} x availability) reviewer pick.
    Falls back to a reliability-leaning default when no domain / no authority signal."""
    agents = getattr(world, "agents", {}) or {}
    cand = [a for aid, a in agents.items() if aid not in exclude]
    if not cand:
        return []
    if domain:
        weights = []
        for a in cand:
            score = math.exp(REVIEWER_TAU * authority_in(a, domain)) * availability(a)
            weights.append(max(1e-6, score))
        order = sorted(range(len(cand)), key=lambda i: -weights[i])
        picked = [cand[i].id for i in order[:k]]
        if picked:
            return picked
    # fallback: reliability/quality-leaning roster (not always founders)
    pref = [a.id for a in cand if a.id in ("calvin", "victor", "will")]
    return (pref or [a.id for a in cand])[:k]


def objection_weight(agent, domain: Optional[str]) -> float:
    """§11.2: 0.5 + 0.5 a_{i,d}. High-authority objection more likely to escalate."""
    if not domain:
        return 0.5
    return 0.5 + 0.5 * authority_in(agent, domain)


def policy_authority_bonus(agent, action_type: str, world: Any) -> float:
    """§11.3: small score bonus for acting in a domain the agent already has standing in."""
    dmn = domain_for_action(action_type)
    if not dmn:
        return 0.0
    return POLICY_AUTHORITY_DELTA * authority_in(agent, dmn)


def policy_reputation_bonus(agent, action_type: str, world: Any) -> float:
    """v8 #3.3: a second small soft prior — above-baseline domain reputation nudges an
    agent toward acting in that domain (separate from authority, surfaced in the trace).
    Capped at 0.10 so it stays a prior, never a decider."""
    dmn = domain_for_action(action_type)
    if not dmn:
        return 0.0
    rep = float((getattr(agent, "reputation", {}) or {}).get(dmn, 0.5) or 0.5)
    return max(0.0, min(0.10, 0.5 * (rep - 0.5)))


__all__ = ["availability", "authority_in", "domain_for_action", "select_reviewers",
           "objection_weight", "policy_authority_bonus", "policy_reputation_bonus"]
