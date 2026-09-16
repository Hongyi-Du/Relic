"""GrowthReconciler (spec §8-§14) — daily batch apply of skill / reputation / authority.

Runs once per day on the buffered GrowthSignals: accumulate skill deltas (diminishing,
capped), reputation deltas (visibility/credit/outcome weighted, asymmetric, decayed,
capped), recompute informal authority from skill+reputation+usage+ownership, refresh
go-to tags, and write a GrowthEvent for every non-trivial change.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from environments.org_env.growth.objects import (
    BETA_B,
    BETA_O,
    BETA_R,
    BETA_S,
    BETA_U,
    DOMAIN_SKILL_MAP,
    ETA_REP,
    ETA_SKILL,
    ETA_SKILL_NEG,
    FAIL_REF,
    GAMMA_NEG,
    GO_TO_MIN_POSITIVE_EVENTS,
    GO_TO_REL_FLOOR,
    GO_TO_REL_MARGIN,
    GROWABLE_SKILLS,
    LAMBDA_REP_DAY,
    MAX_REP_DELTA_DAY,
    MAX_REP_DELTA_EVENT,
    MAX_SKILL_DELTA_DAY,
    MAX_SKILL_DELTA_EVENT,
    MU_REP,
    RECENT_WINDOW,
    REPUTATION_DOMAINS,
    USE_REF,
    GrowthEvent,
    clip01,
    new_authority,
    new_reputation,
)


def _cap(x: float, lim: float) -> float:
    return max(-lim, min(lim, x))


class GrowthReconciler:
    def __init__(self) -> None:
        self._seq = 0

    def run(self, world: Any, tick: int) -> Dict[str, Any]:
        agents = getattr(world, "agents", {}) or {}
        signals = list(getattr(world, "_growth_signals", []) or [])
        # group signals by agent
        by_agent: Dict[str, list] = {}
        for s in signals:
            if s.agent_id in agents:
                by_agent.setdefault(s.agent_id, []).append(s)

        growth_events: List[GrowthEvent] = []
        for aid, agent in agents.items():
            self._ensure_state(agent)
            authority_before = dict(agent.authority)
            skill_delta, rep_delta = self._accumulate(agent, by_agent.get(aid, []))
            self._apply_skill(agent, skill_delta)
            self._apply_reputation(agent, rep_delta)         # includes daily decay
            self._recompute_authority(agent, world)
            self._update_recent(agent, by_agent.get(aid, []), tick)
            growth_events.extend(self._log(world, agent, aid, tick, skill_delta, rep_delta,
                                           authority_before, by_agent.get(aid, [])))
        self._refresh_go_to(world, agents, tick)
        world.growth_events.extend(growth_events)
        world._growth_signals = []                            # consumed
        return {"tick": tick, "signals": len(signals), "growth_events": len(growth_events)}

    # -- state init -------------------------------------------------------
    @staticmethod
    def _ensure_state(agent) -> None:
        if not getattr(agent, "reputation", None):
            agent.reputation = new_reputation()
        if not getattr(agent, "authority", None):
            agent.authority = new_authority()
        if not hasattr(agent, "go_to_tags"):
            agent.go_to_tags = []
        if not hasattr(agent, "_domain_events"):
            agent._domain_events = {}          # domain -> [(tick, outcome>0)]

    # -- §8/§9 accumulate deltas across the day --------------------------
    def _accumulate(self, agent, sigs) -> tuple:
        sd: Dict[str, float] = {}
        rd: Dict[str, float] = {}
        for s in sigs:
            for k, m in s.skill_weights.items():
                if k not in GROWABLE_SKILLS:
                    continue
                cur = float(agent.skills.get(k, 0.3))
                if s.outcome > 0:
                    d = ETA_SKILL * s.credit * m * s.workload * s.outcome * (1 - cur)
                else:
                    d = -ETA_SKILL_NEG * s.credit * m * s.workload * abs(s.outcome) * cur
                sd[k] = sd.get(k, 0.0) + _cap(d, MAX_SKILL_DELTA_EVENT)
            g = s.outcome if s.outcome >= 0 else GAMMA_NEG * s.outcome
            for dmn, n in s.rep_weights.items():
                if dmn not in REPUTATION_DOMAINS:
                    continue
                impact = ETA_REP * (s.credit * n * s.visibility * s.workload * g)
                rd[dmn] = rd.get(dmn, 0.0) + _cap(impact, MAX_REP_DELTA_EVENT)
        # per-day caps
        sd = {k: _cap(v, MAX_SKILL_DELTA_DAY) for k, v in sd.items()}
        rd = {k: _cap(v, MAX_REP_DELTA_DAY) for k, v in rd.items()}
        return sd, rd

    @staticmethod
    def _apply_skill(agent, skill_delta) -> None:
        for k, d in skill_delta.items():
            agent.skills[k] = clip01(float(agent.skills.get(k, 0.3)) + d)

    @staticmethod
    def _apply_reputation(agent, rep_delta) -> None:
        for dmn in REPUTATION_DOMAINS:
            r = float(agent.reputation.get(dmn, MU_REP))
            r = MU_REP + (1 - LAMBDA_REP_DAY) * (r - MU_REP)     # §9 daily decay
            agent.reputation[dmn] = clip01(r + rep_delta.get(dmn, 0.0))

    # -- §10 authority derivation ----------------------------------------
    def _recompute_authority(self, agent, world) -> None:
        for dmn in REPUTATION_DOMAINS:
            r = float(agent.reputation.get(dmn, MU_REP))
            s_bar = sum(w * float(agent.skills.get(k, 0.3))
                        for k, w in DOMAIN_SKILL_MAP.get(dmn, {}).items())
            u = self._recent_use(agent, dmn)
            o = self._ownership(agent, world, dmn)
            b = self._recent_fail(agent, dmn)
            a = (BETA_R * r + BETA_S * s_bar + BETA_U * u + BETA_O * o - BETA_B * b)
            agent.authority[dmn] = clip01(a)

    def _recent_use(self, agent, dmn) -> float:
        ev = agent._domain_events.get(dmn, [])
        n = sum(1 for _, ok in ev if ok)
        return min(1.0, math.log(1 + n) / math.log(1 + USE_REF))

    def _recent_fail(self, agent, dmn) -> float:
        ev = agent._domain_events.get(dmn, [])
        n = sum(1 for _, ok in ev if not ok)
        return min(1.0, n / FAIL_REF)

    @staticmethod
    def _ownership(agent, world, dmn) -> float:
        from environments.org_env.product.objects import artifact_purpose
        from environments.org_env.growth.appraiser import _PURPOSE_DOMAINS
        active = resolved = 0
        for t in (getattr(world, "tasks", {}) or {}).values():
            if getattr(t, "owner_id", None) != agent.id:
                continue
            arts = getattr(t, "linked_artifacts", []) or []
            in_domain = any(dmn in _PURPOSE_DOMAINS.get(artifact_purpose(a), ({}, {}))[1]
                            for a in arts)
            if not in_domain:
                continue
            st = getattr(t.status, "value", str(t.status))
            if st in ("done", "merged", "released"):
                resolved += 1
            else:
                active += 1
        return min(1.0, 0.2 * active + 0.3 * resolved)

    def _update_recent(self, agent, sigs, tick) -> None:
        for s in sigs:
            for dmn in s.rep_weights:
                if dmn in REPUTATION_DOMAINS:
                    agent._domain_events.setdefault(dmn, []).append((tick, s.outcome > 0))
        # prune outside the window
        lo = tick - RECENT_WINDOW
        for dmn, ev in agent._domain_events.items():
            agent._domain_events[dmn] = [(t, ok) for (t, ok) in ev if t >= lo]

    # -- §12 evidence log ------------------------------------------------
    def _log(self, world, agent, aid, tick, skill_delta, rep_delta, authority_before, sigs):
        out: List[GrowthEvent] = []
        if not (skill_delta or rep_delta):
            return out
        # one GrowthEvent per reputation domain that moved (carries related skill deltas)
        moved = {d for d, v in rep_delta.items() if abs(v) > 1e-4} or \
            {d for d in REPUTATION_DOMAINS if abs(agent.authority[d] - authority_before.get(d, 0)) > 1e-3}
        for dmn in moved:
            self._seq += 1
            rel_skills = {k: round(v, 5) for k, v in skill_delta.items()
                          if k in DOMAIN_SKILL_MAP.get(dmn, {})}
            src = next((s for s in sigs if dmn in s.rep_weights), None)
            out.append(GrowthEvent(
                growth_event_id=f"growth_{self._seq}", tick=tick, agent_id=aid, domain=dmn,
                source_event_id=getattr(src, "source_event_id", ""),
                source_action_id=getattr(src, "source_action_id", ""),
                skill_deltas=rel_skills, reputation_deltas={dmn: round(rep_delta.get(dmn, 0.0), 5)},
                authority_before=round(authority_before.get(dmn, 0.0), 4),
                authority_after=round(agent.authority[dmn], 4),
                visibility=round(getattr(src, "visibility", 0.0), 3),
                outcome_score=round(getattr(src, "outcome", 0.0), 3),
                credit=round(getattr(src, "credit", 0.0), 3),
                reason=getattr(src, "reason", "") or f"{dmn} updated"))
        return out

    # -- §14 go-to tags (relative rule) ----------------------------------
    def _refresh_go_to(self, world, agents, tick) -> None:
        """v8e #2: the go-to is derived from the SAME authority values as the authority
        leaderboard (no separate standing blend), so the two are always consistent. The
        authority formula itself is competence-weighted (BETA_S high, ownership low), which
        is what lets a domain expert lead rather than whoever was most active / a founder."""
        for dmn in REPUTATION_DOMAINS:
            ranked = sorted(agents.values(), key=lambda a: -a.authority.get(dmn, 0.0))
            leader = ranked[0] if ranked else None
            second = ranked[1].authority.get(dmn, 0.0) if len(ranked) > 1 else 0.0
            for agent in agents.values():
                tag = f"go_to_{dmn}"
                a = agent.authority.get(dmn, 0.0)
                ev = agent._domain_events.get(dmn, [])
                pos = sum(1 for _, ok in ev if ok)
                neg = sum(1 for _, ok in ev if not ok)
                # relative: clear authority leader, distinct enough, with NET-positive
                # evidence (a stale failure shouldn't permanently block a clear leader).
                qualifies = (agent is leader
                             and (a >= GO_TO_REL_FLOOR or a >= second + GO_TO_REL_MARGIN)
                             and pos >= GO_TO_MIN_POSITIVE_EVENTS
                             and pos > neg)
                if qualifies and tag not in agent.go_to_tags:
                    agent.go_to_tags.append(tag)
                elif not qualifies and tag in agent.go_to_tags:
                    agent.go_to_tags.remove(tag)


__all__ = ["GrowthReconciler"]
