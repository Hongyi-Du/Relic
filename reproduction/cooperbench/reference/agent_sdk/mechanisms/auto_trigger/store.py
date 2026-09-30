"""
auto_trigger/store.py — AutoTriggerStore.

Shared template registry with three-pool Top-K visibility, exploration
quotas for low-exposure high-tier reflexes, utility-trace ingestion and
kind-aware Laplace confidence. Modeled after CollectiveMemory's
AssetStore — same reputation philosophy, but with an expanded object
model (SELF / PAIR / MODE).

The store is intentionally the only place where aggregate stats are
mutated. Agents push ``UtilityTrace`` objects via ``apply_utility_trace``;
the store updates the template's confidence and vitality in a single
atomic step, then optionally forwards the trace to pair / mode
sub-systems owned by the engine.
"""
from __future__ import annotations

import logging
import random
import uuid
from typing import Any, Callable, Dict, List, Optional

from .scoring import apply_trace_to_template
from .types import (
    AgentReflexEntry,
    ReflexKind,
    ReflexTemplate,
    TemplateStatus,
    UtilityTrace,
)

logger = logging.getLogger(__name__)


class _VoteRecord:
    __slots__ = ("template_id", "voter_id", "vote", "turn")

    def __init__(self, template_id: str, voter_id: str, vote: str, turn: int):
        self.template_id = template_id
        self.voter_id = voter_id
        self.vote = vote
        self.turn = turn


DEFAULT_EXPLORATION_QUOTA: Dict[str, float] = {
    ReflexKind.SELF.value: 0.10,
    ReflexKind.PAIR.value: 0.25,
    ReflexKind.MODE.value: 0.25,
}


class AutoTriggerStore:
    """Shared template store supporting the Level 4/5 reflex lifecycle.

    Three-pool retrieval:
        * ``list_templates(kind=..., top_k=n)`` partitions by kind.
        * ``get_top_templates_by_kind(kind, n)`` returns the kind pool.

    Exploration quotas:
        Each call to ``list_templates`` with ``explore_fraction>0`` replaces
        a fraction of the tail slots with low-exposure picks (templates whose
        ``total_fires`` lie in the bottom 25% of the pool).

    Utility traces:
        ``apply_utility_trace`` updates aggregate counters, confidence and
        vitality through :func:`scoring.apply_trace_to_template`, and
        forwards the trace to any registered listener (engine pair/mode
        side effects).

    Backward compatibility:
        Original signatures (``list_templates(tags=, top_k=)``,
        ``apply_vote(...)``, ``get_top_templates(n)``, ``get_active_templates_for_adoption``)
        still behave as before; new behaviour is additive.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.templates: Dict[str, ReflexTemplate] = {}
        self._votes: List[_VoteRecord] = []
        self._voter_set: Dict[str, set] = {}
        self._deprecation_threshold: int = int(
            self.config.get("deprecation_dislike_threshold", 5)
        )
        self._initial_confidence: float = float(
            self.config.get("initial_confidence", 0.3)
        )
        self._builtin_auto_unlock: List[str] = list(
            self.config.get("builtin_auto_unlock", [])
        )
        # Exploration quota per kind; merge user config on top of defaults.
        quota_cfg = dict(DEFAULT_EXPLORATION_QUOTA)
        user_quota = self.config.get("exploration_quota", {}) or {}
        if isinstance(user_quota, dict):
            quota_cfg.update({k: float(v) for k, v in user_quota.items()})
        self._exploration_quota: Dict[str, float] = quota_cfg

        # Listener: called after apply_utility_trace so engine can forward to
        # pair/mode sub-systems without introducing import cycles.
        self._trace_listener: Optional[Callable[[ReflexTemplate, UtilityTrace], None]] = None
        self._rng = random.Random(0xA11CE)

    # ── Listeners ──────────────────────────────────────────────────────

    def set_trace_listener(
        self, fn: Optional[Callable[[ReflexTemplate, UtilityTrace], None]]
    ) -> None:
        self._trace_listener = fn

    # ── Publish ────────────────────────────────────────────────────────

    def publish_template(self, template: ReflexTemplate) -> ReflexTemplate:
        if not template.template_id:
            template.template_id = f"reflex_{uuid.uuid4().hex[:8]}"
        if not template.confidence:
            template.confidence = self._initial_confidence
        if template.name in self._builtin_auto_unlock:
            template.status = TemplateStatus.ACTIVE
        self.templates[template.template_id] = template
        logger.info(
            "[AutoTriggerStore] Published template '%s' (id=%s kind=%s status=%s)",
            template.name, template.template_id, template.kind, template.status,
        )
        return template

    # ── Retrieval ──────────────────────────────────────────────────────

    def get_template(self, template_id: str) -> Optional[ReflexTemplate]:
        return self.templates.get(template_id)

    def get_template_by_name(self, name: str) -> Optional[ReflexTemplate]:
        for t in self.templates.values():
            if t.name == name:
                return t
        return None

    def list_templates(
        self,
        tags: Optional[List[str]] = None,
        top_k: int = 20,
        *,
        kind: Optional[str] = None,
        explore_fraction: Optional[float] = None,
    ) -> List[ReflexTemplate]:
        """List templates ordered by (vitality * confidence) with exploration.

        When ``kind`` is provided, only that kind is returned. ``tags`` still
        filters. ``explore_fraction`` overrides the per-kind default quota.
        """
        pool: List[ReflexTemplate] = []
        for t in self.templates.values():
            if t.status == TemplateStatus.DEPRECATED:
                continue
            if kind is not None and t.kind != kind:
                continue
            if tags and not any(tag in t.tags for tag in tags):
                continue
            pool.append(t)
        if not pool:
            return []

        # Score = vitality * confidence; falls back to confidence alone for
        # templates without L4 vitality (unit=1.0 by default).
        pool.sort(key=lambda t: (float(t.vitality) * float(t.confidence), t.priority), reverse=True)
        top_k = max(1, int(top_k))

        # Exploration quota — reserve some slots for low-exposure candidates
        if kind is not None:
            frac = self._exploration_quota.get(kind, 0.0) if explore_fraction is None else float(explore_fraction)
        else:
            frac = float(explore_fraction or 0.0)
        if frac > 0.0 and len(pool) > top_k:
            n_explore = max(1, int(round(top_k * frac)))
            exploit = pool[: top_k - n_explore]
            tail = pool[top_k - n_explore:]
            # Pick low-exposure from tail: bottom 25% by total_fires, then random
            tail_sorted_by_exposure = sorted(tail, key=lambda t: float(t.total_fires))
            cutoff = max(1, int(len(tail_sorted_by_exposure) * 0.25))
            low_exposure = tail_sorted_by_exposure[:cutoff] or tail_sorted_by_exposure
            explore = self._rng.sample(low_exposure, k=min(n_explore, len(low_exposure)))
            return exploit + explore

        return pool[:top_k]

    def get_top_templates(self, n: int = 5) -> List[ReflexTemplate]:
        active = [
            t for t in self.templates.values()
            if t.status == TemplateStatus.ACTIVE
        ]
        active.sort(key=lambda t: (float(t.vitality) * float(t.confidence), t.priority), reverse=True)
        return active[:n]

    def get_top_templates_by_kind(self, kind: str, n: int = 5) -> List[ReflexTemplate]:
        return self.list_templates(kind=kind, top_k=n)

    def get_active_templates_for_adoption(self) -> List[ReflexTemplate]:
        return [
            t for t in self.templates.values()
            if t.status == TemplateStatus.ACTIVE
        ]

    # ── Update / Delete (creator only) ─────────────────────────────────

    def update_template(
        self, template_id: str, updates: Dict[str, Any], requester_id: str
    ) -> Optional[ReflexTemplate]:
        t = self.templates.get(template_id)
        if t is None:
            return None
        if t.creator_id and t.creator_id != requester_id:
            logger.warning(
                "[AutoTriggerStore] Agent %s tried to edit template %s owned by %s",
                requester_id, template_id, t.creator_id,
            )
            return None
        allowed_fields = {
            "name", "description", "condition", "action_program",
            "priority", "cooldown_turns", "max_fires", "tags", "expected_free",
            # L4/L5 tunables that creators may adjust
            "ttl_turns", "exclusive_with", "overlay_name",
        }
        for k, v in updates.items():
            if k in allowed_fields:
                setattr(t, k, v)
        return t

    def delete_template(self, template_id: str, requester_id: str) -> bool:
        t = self.templates.get(template_id)
        if t is None:
            return False
        if t.creator_id and t.creator_id != requester_id:
            logger.warning(
                "[AutoTriggerStore] Agent %s tried to delete template %s owned by %s",
                requester_id, template_id, t.creator_id,
            )
            return False
        t.status = TemplateStatus.DEPRECATED
        return True

    # ── Voting ─────────────────────────────────────────────────────────

    def apply_vote(
        self, template_id: str, vote: str, voter_id: str, turn: int
    ) -> float:
        t = self.templates.get(template_id)
        if t is None:
            return 0.0
        voter_key = f"{template_id}:{voter_id}"
        if voter_key in self._voter_set.get(template_id, set()):
            return 0.0
        self._voter_set.setdefault(template_id, set()).add(voter_key)
        self._votes.append(_VoteRecord(template_id, voter_id, vote, turn))

        if vote in ("like", "up"):
            t.likes += 1
            t.positive_fires += 1
        else:
            t.dislikes += 1
            t.negative_fires += 1

        # Kind-aware Laplace confidence.
        t.confidence = (t.positive_fires + 1.0) / (t.positive_fires + t.negative_fires + 2.0)

        # Deprecation check
        if vote in ("dislike", "down") and t.status == TemplateStatus.ACTIVE:
            consecutive = self._count_consecutive_dislikes(template_id)
            if consecutive >= self._deprecation_threshold:
                t.status = TemplateStatus.DEPRECATED
                logger.info(
                    "[AutoTriggerStore] Template '%s' deprecated after %d consecutive dislikes",
                    t.name, consecutive,
                )

        return t.confidence

    def _count_consecutive_dislikes(self, template_id: str) -> int:
        count = 0
        for rec in reversed(self._votes):
            if rec.template_id != template_id:
                continue
            if rec.vote in ("dislike", "down"):
                count += 1
            else:
                break
        return count

    # ── Utility trace ingestion (L4/L5) ────────────────────────────────

    def apply_utility_trace(
        self,
        trace: UtilityTrace,
        *,
        pair_closed: Optional[bool] = None,
    ) -> Optional[ReflexTemplate]:
        """Feed one UtilityTrace into the aggregates of its template.

        Returns the updated template (or None if the template_id is not
        known). Also forwards to any registered ``trace_listener`` so the
        engine can react (e.g. update mode / pair stats for telemetry).
        """
        t = self.templates.get(trace.template_id)
        if t is None:
            return None
        apply_trace_to_template(t, trace, pair_closed=pair_closed)
        if self._trace_listener is not None:
            try:
                self._trace_listener(t, trace)
            except Exception as e:
                logger.debug("[AutoTriggerStore] trace listener error: %s", e)
        return t

    # ── Unlock ─────────────────────────────────────────────────────────

    def unlock_templates(self, template_ids: List[str]) -> None:
        for tid in template_ids:
            t = self.templates.get(tid)
            if t is None:
                t = self.get_template_by_name(tid)
            if t and t.status == TemplateStatus.LOCKED:
                t.status = TemplateStatus.ACTIVE
                logger.info(
                    "[AutoTriggerStore] Unlocked template '%s' (id=%s)",
                    t.name, t.template_id,
                )

    # ── Serialization ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "templates": {
                tid: t.to_dict() for tid, t in self.templates.items()
            },
            "votes": [
                {
                    "template_id": v.template_id,
                    "voter_id": v.voter_id,
                    "vote": v.vote,
                    "turn": v.turn,
                }
                for v in self._votes
            ],
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> "AutoTriggerStore":
        store = cls(config=config)
        for tid, td in data.get("templates", {}).items():
            store.templates[tid] = ReflexTemplate.from_dict(td)
        for vd in data.get("votes", []):
            store._votes.append(
                _VoteRecord(vd["template_id"], vd["voter_id"], vd["vote"], vd["turn"])
            )
            store._voter_set.setdefault(vd["template_id"], set()).add(
                f"{vd['template_id']}:{vd['voter_id']}"
            )
        return store
