"""
auto_trigger/mode.py — Mode Reflex runtime (Level 5).

A ModeManager maintains per-agent active modes with TTL, cooldowns and
mutual exclusion. A mode is activated by a ``ModeTemplate`` reflex: the
reflex's action_program is a single ``activate_mode`` node that calls
``ModeManager.activate()`` via the AutoTriggerPort.

Three guarantees:
  * A mode cannot re-activate while its cooldown has not elapsed for an
    agent.
  * Activating mode X evicts any currently-active mode listed in its
    ``exclusive_with``.
  * When a mode's TTL expires, the window is settled — accumulated
    utility is turned into a UtilityTrace back to the underlying
    ReflexTemplate so the store can update its kind-aware confidence.

The ModeManager also exposes the set of active mode_ids for a given
agent: adapters prepend this set onto the ReflexContext so condition
evaluation (``mode_active``) and prompt overlay injection
(``assemble_mode_overlay``) can read it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from .scoring import accumulate_mode_window, apply_trace_to_template
from .types import ModeActivation, ModeTemplate, ReflexKind, ReflexTemplate, UtilityTrace

logger = logging.getLogger(__name__)


@dataclass
class _AgentModeState:
    active: Dict[str, ModeActivation] = field(default_factory=dict)
    # mode_id -> cooldown_until_turn (inclusive)
    cooldown_until: Dict[str, int] = field(default_factory=dict)
    # Rolling counter of distinct activations this agent has seen; used to
    # compute thrash penalties.
    activations_recent: List[int] = field(default_factory=list)


class ModeManager:
    """Engine-singleton managing per-agent mode activations.

    The engine owns ONE ModeManager. It is injected into the
    AutoTriggerPort so that ActionProgramExecutor's ``activate_mode`` node
    can call ``manager.activate()`` from inside a reflex fire.
    """

    def __init__(self) -> None:
        self._modes: Dict[str, ModeTemplate] = {}
        self._per_agent: Dict[str, _AgentModeState] = {}
        # Optional callback (template_id -> None) triggered after a window
        # settles. Useful for tests and telemetry.
        self._settlement_listener: Optional[Callable[[ModeActivation, UtilityTrace], None]] = None
        # Template registry lookup function — injected by the port so the
        # manager can update the underlying ReflexTemplate aggregates when
        # a window settles without importing the store directly.
        self._template_lookup: Optional[Callable[[str], Optional[ReflexTemplate]]] = None

    # -- Registration ---------------------------------------------------

    def register_mode(self, mode: ModeTemplate) -> ModeTemplate:
        if not mode.mode_id:
            raise ValueError("ModeTemplate.mode_id is required")
        self._modes[mode.mode_id] = mode
        logger.info(
            "[ModeManager] registered mode %s (template=%s ttl=%d cd=%d overlay=%s)",
            mode.mode_id, mode.template_id, mode.ttl_turns, mode.cooldown_turns,
            mode.overlay_name,
        )
        return mode

    def set_template_lookup(self, fn: Callable[[str], Optional[ReflexTemplate]]) -> None:
        self._template_lookup = fn

    def set_settlement_listener(
        self, fn: Optional[Callable[[ModeActivation, UtilityTrace], None]]
    ) -> None:
        self._settlement_listener = fn

    def get_mode(self, mode_id: str) -> Optional[ModeTemplate]:
        return self._modes.get(mode_id)

    def all_modes(self) -> List[ModeTemplate]:
        return list(self._modes.values())

    # -- Activation -----------------------------------------------------

    def activate(
        self, agent_id: str, mode_id: str, turn: int,
        ttl_override: Optional[int] = None,
    ) -> Optional[ModeActivation]:
        """Activate ``mode_id`` for ``agent_id`` at ``turn``.

        Returns the fresh ModeActivation on success, or None if:
          * the mode is unknown,
          * the agent is still in cooldown for this mode,
          * the mode is already active on the agent.
        """
        mode = self._modes.get(mode_id)
        if mode is None:
            logger.warning("[ModeManager] unknown mode_id=%s", mode_id)
            return None
        state = self._per_agent.setdefault(agent_id, _AgentModeState())

        # Cooldown guard
        cd = state.cooldown_until.get(mode_id, -1)
        if turn <= cd:
            logger.debug(
                "[ModeManager] activate blocked by cooldown agent=%s mode=%s turn=%d cd=%d",
                agent_id, mode_id, turn, cd,
            )
            return None

        # Already active -> refresh TTL instead of re-activating
        existing = state.active.get(mode_id)
        if existing is not None:
            ttl = int(ttl_override if ttl_override is not None else mode.ttl_turns)
            existing.expires_turn = turn + max(1, ttl)
            logger.debug(
                "[ModeManager] refresh mode=%s agent=%s new_expiry=%d",
                mode_id, agent_id, existing.expires_turn,
            )
            return existing

        # Mutual exclusion — evict exclusive_with modes by settling them first
        for other_id in list(mode.exclusive_with):
            if other_id in state.active:
                self._settle(state, agent_id, other_id, turn, reason="evicted")

        ttl = int(ttl_override if ttl_override is not None else mode.ttl_turns)
        activation = ModeActivation(
            agent_id=agent_id,
            mode_id=mode_id,
            template_id=mode.template_id,
            activated_turn=turn,
            expires_turn=turn + max(1, ttl),
            horizon_turns=int(mode.horizon_turns),
        )
        state.active[mode_id] = activation
        mode.activations += 1
        state.activations_recent.append(turn)
        if len(state.activations_recent) > 50:
            state.activations_recent = state.activations_recent[-50:]
        logger.info(
            "[ModeManager] ACTIVATE mode=%s agent=%s ttl=%d expires=%d",
            mode_id, agent_id, ttl, activation.expires_turn,
        )
        return activation

    # -- Query ----------------------------------------------------------

    def active_mode_ids(self, agent_id: str, turn: int) -> Set[str]:
        state = self._per_agent.get(agent_id)
        if state is None:
            return set()
        self.settle_expired(agent_id, turn)
        return set(state.active.keys())

    def active_overlays(self, agent_id: str, turn: int) -> List[str]:
        """Return the overlay names of active modes, highest priority first."""
        state = self._per_agent.get(agent_id)
        if state is None:
            return []
        self.settle_expired(agent_id, turn)
        items = []
        for mid, activ in state.active.items():
            mode = self._modes.get(mid)
            if mode is not None and mode.overlay_name:
                items.append((mode.priority, mode.overlay_name))
        items.sort(reverse=True)
        return [name for (_p, name) in items]

    def is_active(self, agent_id: str, mode_id: str, turn: int) -> bool:
        state = self._per_agent.get(agent_id)
        if state is None:
            return False
        self.settle_expired(agent_id, turn)
        return mode_id in state.active

    # -- Window utility feeds -------------------------------------------

    def record_turn_metrics(
        self,
        agent_id: str,
        turn: int,
        *,
        llm_cost_saved: float = 0.0,
        energy_delta_pct: float = 0.0,
        hp_delta_pct: float = 0.0,
        metric_delta: float = 0.0,
        pair_closure_delta: float = 0.0,
        thrash_delta: float = 0.0,
    ) -> None:
        """Feed per-turn numeric signals into any open mode windows.

        This should be called by the agent core right after an action
        executes, regardless of whether that action was a reflex fire or
        a regular LLM decision.
        """
        state = self._per_agent.get(agent_id)
        if not state or not state.active:
            return
        for activation in state.active.values():
            accumulate_mode_window(
                activation,
                llm_cost_saved=llm_cost_saved,
                energy_delta_pct=energy_delta_pct,
                hp_delta_pct=hp_delta_pct,
                metric_delta=metric_delta,
                pair_closure_delta=pair_closure_delta,
                thrash_delta=thrash_delta,
            )

    # -- Expiry / settlement --------------------------------------------

    def settle_expired(self, agent_id: str, turn: int) -> List[ModeActivation]:
        """Close out modes whose TTL has elapsed for ``agent_id``."""
        state = self._per_agent.get(agent_id)
        if state is None:
            return []
        settled: List[ModeActivation] = []
        for mid in list(state.active.keys()):
            activation = state.active[mid]
            if turn >= activation.expires_turn:
                closed = self._settle(state, agent_id, mid, turn, reason="ttl")
                if closed is not None:
                    settled.append(closed)
        return settled

    def _settle(
        self, state: _AgentModeState, agent_id: str, mode_id: str,
        turn: int, *, reason: str,
    ) -> Optional[ModeActivation]:
        activation = state.active.pop(mode_id, None)
        if activation is None:
            return None
        mode = self._modes.get(mode_id)
        if mode is None:
            return activation
        state.cooldown_until[mode_id] = turn + int(mode.cooldown_turns)

        total = activation.delta_total()
        if total > 0:
            mode.positive_activations += 1
        else:
            mode.negative_activations += 1

        trace = UtilityTrace(
            turn=turn,
            agent_id=agent_id,
            template_id=mode.template_id,
            kind=ReflexKind.MODE.value,
            u_cost=activation.u_cost,
            u_surv=activation.u_surv,
            u_task=activation.u_task,
            u_disp=-activation.u_thrash,   # thrash penalty lives in u_disp
            u_coord=activation.u_coord,
            outcome="success" if total > 0 else "partial",
        )
        if self._template_lookup is not None:
            tmpl = self._template_lookup(mode.template_id)
            if tmpl is not None:
                apply_trace_to_template(tmpl, trace)
        if self._settlement_listener is not None:
            try:
                self._settlement_listener(activation, trace)
            except Exception as e:
                logger.debug("[ModeManager] settlement listener error: %s", e)
        logger.info(
            "[ModeManager] SETTLE mode=%s agent=%s reason=%s totalU=%.3f",
            mode_id, agent_id, reason, total,
        )
        return activation

    # -- Introspection --------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "modes": {mid: m.to_dict() for mid, m in self._modes.items()},
            "per_agent": {
                aid: {
                    "active": {mid: {
                        "activated_turn": a.activated_turn,
                        "expires_turn": a.expires_turn,
                        "uC": a.u_cost, "uS": a.u_surv, "uT": a.u_task,
                        "uCo": a.u_coord, "uTh": a.u_thrash,
                    } for mid, a in s.active.items()},
                    "cooldown_until": dict(s.cooldown_until),
                    "recent_activations": list(s.activations_recent[-10:]),
                }
                for aid, s in self._per_agent.items()
            },
        }


__all__ = ["ModeManager"]
