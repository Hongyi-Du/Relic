"""
auto_trigger/pair.py — Cooperative Reflex Pair registry & closure tracker.

Level 4 machinery: bind an Emitter template and a Responder template by
pair_id, observe every emit / consume on the signal bus, and award
closure credit when a responder consumes the same emit_id within a time
window. Pair objects, not individual templates, are the unit of adoption
reward for cooperative reflexes.

Usage inside the engine-side port:

    registry = PairRegistry()
    registry.register(pair_template)

    tracker = PairClosureTracker(registry)
    # Emitter fires ->
    tracker.on_emit(pair_id, emit_id, emitter_agent_id, turn)
    # Responder consumes the signal ->
    result = tracker.on_consume(pair_id, emit_id, responder_agent_id, turn)
    # Book-keeping sweep each turn ->
    misses = tracker.sweep_timeouts(turn)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import ReflexKind, ReflexPairTemplate, ReflexTemplate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class PairRegistry:
    """Registry of cooperative pairs by pair_id.

    Also provides reverse look-up from an individual template_id to the
    pair it belongs to (for reward attribution) and from the signal_type
    that an emitter uses to the pair descriptor (for responder dispatch).
    """

    def __init__(self) -> None:
        self._pairs: Dict[str, ReflexPairTemplate] = {}
        self._template_to_pair: Dict[str, str] = {}
        # signal_type -> [pair_id, ...] (multiple pairs may share a signal)
        self._signal_to_pairs: Dict[str, List[str]] = {}

    def register(self, pair: ReflexPairTemplate) -> ReflexPairTemplate:
        """Register a pair; overrides any previous entry with the same id."""
        if not pair.pair_id:
            raise ValueError("ReflexPairTemplate.pair_id is required")
        self._pairs[pair.pair_id] = pair
        if pair.emitter_template_id:
            self._template_to_pair[pair.emitter_template_id] = pair.pair_id
        if pair.responder_template_id:
            self._template_to_pair[pair.responder_template_id] = pair.pair_id
        if pair.signal_type:
            self._signal_to_pairs.setdefault(pair.signal_type, [])
            if pair.pair_id not in self._signal_to_pairs[pair.signal_type]:
                self._signal_to_pairs[pair.signal_type].append(pair.pair_id)
        logger.info(
            "[PairRegistry] registered pair %s (emitter=%s responder=%s signal=%s)",
            pair.pair_id, pair.emitter_template_id, pair.responder_template_id,
            pair.signal_type,
        )
        return pair

    def get(self, pair_id: str) -> Optional[ReflexPairTemplate]:
        return self._pairs.get(pair_id)

    def get_by_template(self, template_id: str) -> Optional[ReflexPairTemplate]:
        pid = self._template_to_pair.get(template_id)
        return self._pairs.get(pid) if pid else None

    def get_by_signal(self, signal_type: str) -> List[ReflexPairTemplate]:
        pids = self._signal_to_pairs.get(signal_type, [])
        return [self._pairs[p] for p in pids if p in self._pairs]

    def stamp_templates(self, templates: Dict[str, ReflexTemplate]) -> None:
        """After registration, annotate the referenced ReflexTemplates.

        Sets ``kind=PAIR``, ``pair_id`` and ``pair_role`` so the evaluator
        knows how to classify them. Idempotent.
        """
        for pair in self._pairs.values():
            e = templates.get(pair.emitter_template_id)
            if e is not None:
                e.kind = ReflexKind.PAIR.value
                e.pair_id = pair.pair_id
                from .types import PairRole as _PR
                e.pair_role = _PR.EMITTER.value
            r = templates.get(pair.responder_template_id)
            if r is not None:
                r.kind = ReflexKind.PAIR.value
                r.pair_id = pair.pair_id
                from .types import PairRole as _PR
                r.pair_role = _PR.RESPONDER.value

    def all_pairs(self) -> List[ReflexPairTemplate]:
        return list(self._pairs.values())

    def to_dict(self) -> Dict[str, dict]:
        return {pid: p.to_dict() for pid, p in self._pairs.items()}


# ---------------------------------------------------------------------------
# Closure tracker
# ---------------------------------------------------------------------------

@dataclass
class _PendingEmit:
    pair_id: str
    emit_id: str
    emitter_agent_id: str
    turn_emitted: int


@dataclass
class ClosureEvent:
    """Returned by :meth:`PairClosureTracker.on_consume` — summary of one closure."""
    pair_id: str
    emit_id: str
    emitter_agent_id: str
    responder_agent_id: str
    turn_emitted: int
    turn_closed: int
    within_window: bool


class PairClosureTracker:
    """Track pending emits and settle them when responders consume the signal.

    A ``sweep_timeouts(turn)`` call should be invoked every turn by the
    engine; emits that exceed ``pair.closure_window`` become misses and
    contribute a negative closure signal to the pair's confidence.
    """

    def __init__(self, registry: PairRegistry):
        self._registry = registry
        self._pending: Dict[str, _PendingEmit] = {}
        # Telemetry for recent events (engine/UI can inspect)
        self._recent_closures: List[ClosureEvent] = []
        self._recent_misses: List[Tuple[str, str, str, int]] = []
        # History size caps — we only keep the latest N for introspection.
        self._max_history = 200

    # -- Emit side -------------------------------------------------------

    def on_emit(
        self, pair_id: str, emit_id: str, emitter_agent_id: str, turn: int
    ) -> None:
        if not pair_id or not emit_id:
            return
        pair = self._registry.get(pair_id)
        if pair is None:
            return
        self._pending[emit_id] = _PendingEmit(
            pair_id=pair_id, emit_id=emit_id,
            emitter_agent_id=emitter_agent_id, turn_emitted=turn,
        )
        pair.emissions += 1
        pair.last_emit_turn = turn
        logger.debug(
            "[PairClosure] EMIT pair=%s emit=%s by %s @ turn %d",
            pair_id, emit_id, emitter_agent_id, turn,
        )

    # -- Consume side ----------------------------------------------------

    def on_consume(
        self, pair_id: str, emit_id: str, responder_agent_id: str, turn: int
    ) -> Optional[ClosureEvent]:
        """Return a ClosureEvent if the consumption matches a pending emit."""
        if not pair_id or not emit_id:
            return None
        pair = self._registry.get(pair_id)
        if pair is None:
            return None
        pending = self._pending.pop(emit_id, None)
        if pending is None:
            return None
        within = (turn - pending.turn_emitted) <= int(pair.closure_window)
        if within:
            pair.closures += 1
            pair.last_closure_turn = turn
        else:
            pair.misses += 1
        event = ClosureEvent(
            pair_id=pair_id, emit_id=emit_id,
            emitter_agent_id=pending.emitter_agent_id,
            responder_agent_id=responder_agent_id,
            turn_emitted=pending.turn_emitted,
            turn_closed=turn,
            within_window=within,
        )
        self._recent_closures.append(event)
        self._trim(self._recent_closures)
        logger.info(
            "[PairClosure] CLOSE pair=%s emit=%s em=%s rp=%s dt=%d within=%s",
            pair_id, emit_id, pending.emitter_agent_id, responder_agent_id,
            turn - pending.turn_emitted, within,
        )
        self._update_pair_quality(pair)
        return event

    # -- Timeout sweep ---------------------------------------------------

    def sweep_timeouts(self, turn: int) -> List[Tuple[str, str, str, int]]:
        """Move timed-out pending emits to the miss ledger.

        Returns a list of ``(pair_id, emit_id, emitter_agent_id, turn_emitted)``
        for the misses detected this sweep.
        """
        dropped: List[Tuple[str, str, str, int]] = []
        for emit_id in list(self._pending.keys()):
            p = self._pending[emit_id]
            pair = self._registry.get(p.pair_id)
            if pair is None:
                self._pending.pop(emit_id, None)
                continue
            if (turn - p.turn_emitted) > int(pair.closure_window):
                self._pending.pop(emit_id, None)
                pair.misses += 1
                dropped.append((p.pair_id, emit_id, p.emitter_agent_id, p.turn_emitted))
                self._update_pair_quality(pair)
        if dropped:
            self._recent_misses.extend(dropped)
            self._trim(self._recent_misses)
            logger.info("[PairClosure] SWEEP misses=%d at turn=%d", len(dropped), turn)
        return dropped

    # -- Stats -----------------------------------------------------------

    def closure_rate(self, pair_id: str) -> float:
        pair = self._registry.get(pair_id)
        if pair is None or pair.emissions <= 0:
            return 0.0
        return pair.closures / float(pair.emissions)

    def recent_closures(self, limit: int = 50) -> List[ClosureEvent]:
        return list(self._recent_closures[-limit:])

    def recent_misses(self, limit: int = 50) -> List[Tuple[str, str, str, int]]:
        return list(self._recent_misses[-limit:])

    def pending_count(self) -> int:
        return len(self._pending)

    # -- Internal --------------------------------------------------------

    def _update_pair_quality(self, pair: ReflexPairTemplate) -> None:
        """Recompute Laplace-smoothed pair confidence + decay vitality."""
        pos = int(pair.closures)
        neg = int(pair.misses)
        pair.confidence = (pos + 1.0) / (pos + neg + 2.0)
        # Pair-level vitality mirrors template vitality dynamics but uses
        # closure vs miss as the reinforcement signal.
        pair.vitality = max(0.0, min(1.0, pair.vitality * 0.98
                                     + (0.10 if pos > neg else -0.05)))

    def _trim(self, buf: list) -> None:
        if len(buf) > self._max_history:
            del buf[:len(buf) - self._max_history]


__all__ = [
    "PairRegistry",
    "PairClosureTracker",
    "ClosureEvent",
]
