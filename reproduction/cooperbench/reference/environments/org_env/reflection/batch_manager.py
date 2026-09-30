"""ReflectionBatchManager — half-day batch reflection (preflight §3-§6).

Reflection is NOT per-action. Every REFLECTION_BATCH_INTERVAL_TICKS (a half-day)
the manager looks at the last window of events, picks at most a few agents that
actually had something worth reflecting on (salience-ranked, §5), and runs ONE
reflection each through the ReflectionManager (which then integrates ideas into
sparse/deduped wishes, §7-§9). Caps bound cost + avoid spam.

Generic: selection uses role / event signals / work-state, never a specific id.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

REFLECTION_BATCH_INTERVAL_TICKS = 6
MAX_REFLECTIONS_PER_BATCH = 2          # keep reflections sparse (~<=8 / 50 ticks, §20#2)
MAX_REFLECTIONS_PER_AGENT_PER_DAY = 1  # an agent reflects at most once per day
MAX_REFLECTIONS_PER_EPISODE = 2
MIN_REFLECTION_GAP_PER_AGENT = 12
MAX_NEW_WISHES_PER_BATCH = 2
# only genuinely-salient agents reflect (a closed episode / product change /
# challenge / rejection — not mere stress), keeping reflections sparse (§20).
MIN_REFLECTION_SALIENCE = 1.5

# event "type" families that signal an agent had a reflection-worthy half-day.
_CHALLENGE_SUBTYPES = {"changes_requested", "challenged", "claim_dispute",
                       "requested_action", "violation", "rejected"}


class ReflectionBatchManager:
    def __init__(self) -> None:
        self._last_reflect_tick: Dict[str, int] = {}
        self._day_count: Dict[Tuple[str, int], int] = {}
        self._episode_reflect_count: Dict[str, int] = {}
        self.batches_run = 0

    def maybe_run_batch(self, world: Any, tick: int, llm_client: Any = None) -> List[Any]:
        if tick == 0 or tick % REFLECTION_BATCH_INTERVAL_TICKS != 0:
            return []
        return self.run_batch(world, tick, llm_client)

    def run_batch(self, world: Any, tick: int, llm_client: Any = None) -> List[Any]:
        self.batches_run += 1
        window = self.collect_window(world, tick - REFLECTION_BATCH_INTERVAL_TICKS, tick)
        selected = self.select_agents(world, window, tick)
        budget = [MAX_NEW_WISHES_PER_BATCH]
        out = []
        for aid, episode in selected[:MAX_REFLECTIONS_PER_BATCH]:
            refl = world.reflection_manager.reflect(
                aid, world, episode=episode, reason="half_day_batch", batch_budget=budget)
            out.append(refl)
            self._last_reflect_tick[aid] = tick
            day = tick // 24
            self._day_count[(aid, day)] = self._day_count.get((aid, day), 0) + 1
            if episode is not None:
                self._episode_reflect_count[episode.episode_id] = \
                    self._episode_reflect_count.get(episode.episode_id, 0) + 1
        return out

    # -- window ------------------------------------------------------------
    def collect_window(self, world: Any, start: int, end: int) -> List[Dict[str, Any]]:
        return [e for e in getattr(world, "events", []) if start < int(e.get("tick", -1)) <= end]

    # -- selection (§5) ----------------------------------------------------
    def select_agents(self, world: Any, window: List[Dict[str, Any]], tick: int) -> List[Tuple[str, Any]]:
        agents = list(getattr(world, "agents", {}).keys())
        day = tick // 24
        mgr = getattr(world, "episode_manager", None)
        # closed episodes touched in the window
        closed_eps = []
        if mgr is not None:
            for ep in mgr.episodes.values():
                if ep.status != "open" and start_in_window(ep, window, tick):
                    closed_eps.append(ep)
        # per-agent salience + a candidate episode for context
        scored: List[Tuple[float, str, Any]] = []
        for aid in agents:
            # gap + per-day caps
            last = self._last_reflect_tick.get(aid)
            if last is not None and tick - last < MIN_REFLECTION_GAP_PER_AGENT:
                continue
            if self._day_count.get((aid, day), 0) >= MAX_REFLECTIONS_PER_AGENT_PER_DAY:
                continue
            sal, ep = self._salience(aid, world, window, closed_eps)
            if sal < MIN_REFLECTION_SALIENCE:
                continue
            scored.append((sal, aid, ep))
        scored.sort(key=lambda x: -x[0])
        # respect per-episode cap
        out: List[Tuple[str, Any]] = []
        for sal, aid, ep in scored:
            if ep is not None and self._episode_reflect_count.get(ep.episode_id, 0) >= MAX_REFLECTIONS_PER_EPISODE:
                ep = None
            out.append((aid, ep))
        return out

    def _salience(self, aid: str, world: Any, window: List[Dict[str, Any]],
                  closed_eps: List[Any]) -> Tuple[float, Any]:
        sal = 0.0
        ep_for_ctx = None
        # participated in a closed episode (strongest signal)
        for ep in closed_eps:
            if aid in getattr(ep, "participants", []):
                sal += 2.0
                ep_for_ctx = ep_for_ctx or ep
        # else use an OPEN episode the agent is in, for richer reflection context
        if ep_for_ctx is None:
            mgr = getattr(world, "episode_manager", None)
            if mgr is not None:
                for ep in mgr.episodes.values():
                    if ep.status == "open" and aid in getattr(ep, "participants", []):
                        ep_for_ctx = ep
                        break
        # product artifact changed by this agent
        if any(e.get("type") == "product_event" and e.get("agent_id") == aid for e in window):
            sal += 1.5
        # received a challenge / change-request / dispute / rejection nearby
        if any(e.get("agent_id") == aid and (e.get("subtype") in _CHALLENGE_SUBTYPES) for e in window) \
                or any(aid in (e.get("recipients") or []) or e.get("target") == aid for e in window):
            sal += 1.5
        # an action of theirs was rejected this window
        decs = [d for d in getattr(world, "action_decisions", [])
                if d.agent_id == aid and d.validation_status == "rejected"]
        if decs:
            sal += 1.2
        # stress / fatigue load
        a = world.agents.get(aid)
        if a is not None and hasattr(a, "work_state"):
            ws = a.work_state.snapshot()
            sal += 1.0 * float(ws.get("stress") or 0.0)
            sal += 0.6 * float(ws.get("burnout_risk") or 0.0)
        # carries an unresolved blocker in memory
        mem = getattr(world, "agent_memories", {}).get(aid)
        if mem is not None and getattr(mem, "repeated_blockers", []):
            sal += 1.0
        return sal, ep_for_ctx


def start_in_window(ep: Any, window: List[Dict[str, Any]], tick: int) -> bool:
    """An episode counts if it became closed recently (within ~this half-day)."""
    last = int(getattr(ep, "last_event_tick", 0) or getattr(ep, "end_tick", 0) or 0)
    return last >= tick - REFLECTION_BATCH_INTERVAL_TICKS


__all__ = ["ReflectionBatchManager", "REFLECTION_BATCH_INTERVAL_TICKS",
           "MAX_REFLECTIONS_PER_BATCH", "MAX_NEW_WISHES_PER_BATCH"]
