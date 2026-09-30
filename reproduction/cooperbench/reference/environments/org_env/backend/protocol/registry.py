"""ProtocolRegistry — explicit protocol lifecycle + emergence classification
(DESIGN env_org §57-§59, reuses core emergence levels [core §24]).

Lifecycle: propose -> support/oppose -> adopt -> use* -> violation* ->
enforcement* -> (amend/obsolete). ``classify_emergence`` maps the event history
to none / weak / strong:

  weak   = proposed + adopted + repeated_use(>=USE_MIN) + persistence(>=PERSIST_MIN)
  strong = weak + third-party enforcement + typed governed-object transition
           + an independently evaluated outcome improvement
"""
from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from environments.org_env.backend.protocol.objects import (
    GOVERNED_OBJECT_LIFECYCLE_STATES,
    GOVERNED_OBJECT_TYPES,
    INDEPENDENT_OUTCOME_ORACLE_TYPES,
    GovernedObjectSnapshot,
    IndependentOutcomeOracle,
    Protocol,
    ProtocolEvent,
)
from environments.org_env.experiments.provenance import stable_fingerprint

USE_MIN = 2          # "repeated use"
PERSIST_MIN = 48     # ticks (~2 days) the protocol stayed active
REVIEW_MIN_TICKS = 3      # preflight v3 §7: no same-tick adoption (review latency)
ADOPT_MIN_SUPPORTERS = 2  # distinct supporters required to adopt


def protocol_is_live(protocol: Any) -> bool:
    """Whether a rule may still participate in the native lifecycle.

    Native OrgEnv historically permits both proposed and adopted active rules
    to receive support/use/violation/enforcement events.  Preserve that behavior
    exactly; only terminal or internally inconsistent lifecycle states fail
    closed.
    """

    return bool(
        protocol is not None
        and getattr(protocol, "status", None) == "active"
        and getattr(protocol, "adoption_status", None)
        in {"proposed", "adopted"}
    )


def effective_min_supporters(roster_size: int) -> int:
    """The endorsement threshold, capped at what the roster can supply.

    Requiring two supporters encodes "more than one party endorsed this", which
    is what separates an organizational object from a private note. A one-person
    condition cannot satisfy it at any level of effort, so that rung reports a
    hard zero that is indistinguishable from a solo founder who tried and
    failed - the rung stops measuring anything. Capping at the roster keeps the
    rule's meaning (everyone who could endorse did) and keeps the ladder
    ordered, since B1-B3 still owe two.
    """

    return max(1, min(ADOPT_MIN_SUPPORTERS, int(roster_size or 1)))

_BOOKKEEPING_STATE_KEYS = frozenset(
    {
        "blocked",
        "enforcement_count",
        "enforcement_recorded",
        "protocol_id",
        "state_impact_ref",
        "violation_event_id",
    }
)


class ProtocolRegistry:
    def __init__(self, *, min_supporters: int = ADOPT_MIN_SUPPORTERS):
        self.protocols: Dict[str, Protocol] = {}
        self.events: List[ProtocolEvent] = []
        # Set from the roster once the world is built; see
        # effective_min_supporters for why this is not a fixed 2.
        self.min_supporters = max(1, int(min_supporters))
        # A transfer arm's evaluation window measures whether an INHERITED
        # capability gets used, which only means something while the
        # organization cannot compile a new one. Set by
        # capability_transfer.freeze_capability_compilation and lifted when the
        # window closes; false for every ordinary run.
        self.compilation_frozen = False
        self._seq = 0

    def _ev(self, event_type: str, protocol_id: str, actor_id: str, tick: int,
            data: Optional[dict] = None) -> ProtocolEvent:
        self._seq += 1
        e = ProtocolEvent(event_id=f"pev_{self._seq}", event_type=event_type,
                          protocol_id=protocol_id, actor_id=actor_id, tick=tick, data=data or {})
        self.events.append(e)
        return e

    # -- lifecycle ----------------------------------------------------------
    def propose(self, *, proposer_id: str, protocol_type: str, rule_summary: str,
                scope: str = "review", target_process: str = "", tick: int = 0,
                protocol_id: Optional[str] = None) -> Protocol:
        pid = protocol_id or f"proto_{protocol_type}"
        e = self._ev("proposal", pid, proposer_id, tick)
        p = Protocol(protocol_id=pid, protocol_type=protocol_type, proposer_id=proposer_id,
                     proposal_event_id=e.event_id, rule_summary=rule_summary, scope=scope,
                     target_process=target_process, supporters=[proposer_id],
                     first_tick=tick, last_active_tick=tick)
        self.protocols[pid] = p
        return p

    def support(self, agent_id: str, protocol_id: str, tick: int = 0) -> None:
        p = self.protocols[protocol_id]
        if not protocol_is_live(p):
            raise ValueError("protocol_not_live")
        if agent_id not in p.supporters:
            p.supporters.append(agent_id)
        self._ev("support", protocol_id, agent_id, tick)
        self._touch(p, tick)
        # O1.7 §34.15: adoption EMERGES from the support chain (>=2 supporters),
        # never from a single artifact-creation auto-adopt. Preflight v3 §7: it must
        # also clear a review latency, so it can't adopt the same tick it was proposed.
        if self._can_adopt(p, tick):
            self.adopt(protocol_id, tick)

    def _can_adopt(self, p, tick: int) -> bool:
        return (protocol_is_live(p) and p.adoption_status == "proposed"
                and len(set(p.supporters)) >= self.min_supporters
                and tick - int(getattr(p, "first_tick", 0) or 0) >= REVIEW_MIN_TICKS
                and len(set(p.opposers)) < len(set(p.supporters)))

    def tick_adoptions(self, tick: int = 0) -> None:
        """Per-tick sweep (preflight v3 §7): adopt protocols that gathered enough
        support earlier and have now cleared the review-latency window."""
        for pid, p in list(self.protocols.items()):
            if self._can_adopt(p, tick):
                self.adopt(pid, tick)
        # ``emergence_level`` is only written by classify_emergence, which the
        # metrics path does not call — it recomputes from the events and keeps
        # the answer to itself. So the attribute stayed at its initial "none"
        # for the whole run while the run reported the protocol as emerged, and
        # perception hands that attribute to the agents: in one run a rule the
        # whole roster had adopted and invoked 334 times was shown to its own
        # supporters as not having emerged. Recompute where the registry
        # already sweeps, so the stored value is the one the metric would give.
        for pid in self.protocols:
            self.classify_emergence(pid)

    def oppose(self, agent_id: str, protocol_id: str, tick: int = 0) -> None:
        p = self.protocols[protocol_id]
        if not protocol_is_live(p):
            raise ValueError("protocol_not_live")
        if agent_id not in p.opposers:
            p.opposers.append(agent_id)
        self._ev("oppose", protocol_id, agent_id, tick)
        self._touch(p, tick)

    def adopt(self, protocol_id: str, tick: int = 0, *, approver_id: Optional[str] = None,
              force: bool = False) -> bool:
        """v8 #2: adoption records a concrete approver (never an empty actor) and cannot
        happen the same tick it was proposed — unless ``force`` (a mirror of an already-
        governed ProtocolSpec, which carries its own latency + distinct approvers)."""
        p = self.protocols[protocol_id]
        if (
            getattr(p, "status", None) != "active"
            or p.adoption_status != "proposed"
        ):
            return False
        # The freeze applies to `force` too. The forced path mirrors an
        # already-governed spec, and letting it through would give the
        # organization a second door into the capability the window is holding
        # shut. Injection sets the freeze only after its own forced adoptions.
        if self.compilation_frozen:
            return False
        if not force:
            if tick - int(getattr(p, "first_tick", 0) or 0) < REVIEW_MIN_TICKS:
                return False
            if len(set(p.supporters)) < self.min_supporters:
                return False
        p.adoption_status = "adopted"
        actor = approver_id or (sorted(set(p.supporters))[0] if p.supporters else "")
        self._ev("adoption", protocol_id, actor, tick)
        self._touch(p, tick)
        return True

    def use(
        self,
        agent_id: str,
        protocol_id: str,
        tick: int = 0,
        *,
        context_id: str | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
    ) -> ProtocolEvent:
        p = self.protocols[protocol_id]
        if not protocol_is_live(p):
            raise ValueError("protocol_not_live")
        e = self._ev(
            "use",
            protocol_id,
            agent_id,
            tick,
            {
                key: value
                for key, value in {
                    "context_id": context_id,
                    "episode_id": episode_id,
                    "task_id": task_id,
                }.items()
                if value
            },
        )
        p.usage_events.append(e.event_id)
        if agent_id not in p.supporters:          # v8 #2: a distinct user endorses the norm
            p.supporters.append(agent_id)
        self._touch(p, tick)
        # repeated use by DISTINCT agents establishes the norm (chain-based adoption,
        # §34.15), still subject to the review-latency gate + >=2 distinct supporters.
        if (p.adoption_status == "proposed" and len(p.usage_events) >= USE_MIN
                and tick - int(getattr(p, "first_tick", 0) or 0) >= REVIEW_MIN_TICKS):
            self.adopt(protocol_id, tick, approver_id=agent_id)
        return e

    def violate(
        self,
        agent_id: str,
        protocol_id: str,
        tick: int = 0,
        *,
        context_id: str | None = None,
    ) -> ProtocolEvent:
        p = self.protocols[protocol_id]
        if not protocol_is_live(p):
            raise ValueError("protocol_not_enforceable")
        e = self._ev(
            "violation",
            protocol_id,
            agent_id,
            tick,
            {"context_id": context_id} if context_id else {},
        )
        p.violation_events.append(e.event_id)
        self._touch(p, tick)
        return e

    def enforce(
        self,
        agent_id: str,
        protocol_id: str,
        tick: int = 0,
        *,
        violation_event_id: str | None = None,
        state_impact_ref: str | None = None,
        blocked: bool = False,
        context_id: str | None = None,
        state_before_hash: str | None = None,
        state_after_hash: str | None = None,
        state_before: Mapping[str, Any] | None = None,
        state_after: Mapping[str, Any] | None = None,
        governed_object_before: GovernedObjectSnapshot | None = None,
        governed_object_after: GovernedObjectSnapshot | None = None,
    ) -> ProtocolEvent:
        p = self.protocols[protocol_id]
        if not protocol_is_live(p):
            raise ValueError("protocol_not_enforceable")
        if (governed_object_before is None) != (governed_object_after is None):
            raise ValueError(
                "enforcement_governed_transition_requires_both_snapshots"
            )
        if (state_before is None) != (state_after is None):
            raise ValueError(
                "enforcement_state_transition_requires_both_states"
            )
        if governed_object_before is not None and state_before is not None:
            raise ValueError("enforcement_transition_sources_are_mutually_exclusive")

        governed_object_type = None
        governed_object_id = None
        typed_governed_object = False
        if governed_object_before is not None:
            self._validate_governed_transition(
                governed_object_before,
                governed_object_after,
            )
            state_before_payload = self._snapshot_payload(
                governed_object_before
            )
            state_after_payload = self._snapshot_payload(governed_object_after)
            governed_object_type = governed_object_before.object_type
            governed_object_id = governed_object_before.object_id
            if state_impact_ref not in (None, governed_object_id):
                raise ValueError("enforcement_state_impact_ref_identity_mismatch")
            state_impact_ref = governed_object_id
            typed_governed_object = True
        else:
            state_before_payload = (
                dict(state_before) if state_before is not None else None
            )
            state_after_payload = (
                dict(state_after) if state_after is not None else None
            )
        if state_before_payload is not None:
            derived_before_hash = stable_fingerprint(
                state_before_payload
            )
            derived_after_hash = stable_fingerprint(state_after_payload)
            if (
                state_before_hash is not None
                and state_before_hash != derived_before_hash
            ) or (
                state_after_hash is not None
                and state_after_hash != derived_after_hash
            ):
                raise ValueError("enforcement_state_hash_mismatch")
            state_before_hash = derived_before_hash
            state_after_hash = derived_after_hash
        e = self._ev(
            "enforcement",
            protocol_id,
            agent_id,
            tick,
            {
                key: value
                for key, value in {
                    "violation_event_id": violation_event_id,
                    "state_impact_ref": state_impact_ref,
                    "blocked": bool(blocked),
                    "context_id": context_id,
                    "state_before_hash": state_before_hash,
                    "state_after_hash": state_after_hash,
                    "state_before": state_before_payload,
                    "state_after": state_after_payload,
                    "governed_object_type": governed_object_type,
                    "governed_object_id": governed_object_id,
                    "typed_governed_object": typed_governed_object,
                }.items()
                if value not in (None, "", False)
            },
        )
        p.enforcement_events.append(e.event_id)
        self._touch(p, tick)
        return e

    @staticmethod
    def _snapshot_payload(snapshot: GovernedObjectSnapshot) -> dict:
        return {
            "object_type": snapshot.object_type,
            "object_id": snapshot.object_id,
            "lifecycle_state": snapshot.lifecycle_state,
            "observed_tick": int(snapshot.observed_tick),
            "provenance_ref": snapshot.provenance_ref,
            "attributes": dict(snapshot.attributes),
        }

    @staticmethod
    def _validate_governed_transition(
        before: GovernedObjectSnapshot,
        after: GovernedObjectSnapshot,
    ) -> None:
        if not isinstance(before, GovernedObjectSnapshot) or not isinstance(
            after, GovernedObjectSnapshot
        ):
            raise TypeError("governed_transition_requires_typed_snapshots")
        if before.object_type not in GOVERNED_OBJECT_TYPES:
            raise ValueError("unsupported_governed_object_type")
        if after.object_type != before.object_type:
            raise ValueError("governed_object_type_changed")
        if not before.object_id or after.object_id != before.object_id:
            raise ValueError("governed_object_identity_changed")
        allowed_states = GOVERNED_OBJECT_LIFECYCLE_STATES[before.object_type]
        if (
            before.lifecycle_state not in allowed_states
            or after.lifecycle_state not in allowed_states
        ):
            raise ValueError("invalid_governed_object_lifecycle_state")
        if before.observed_tick < 0 or after.observed_tick < before.observed_tick:
            raise ValueError("governed_object_observation_order_invalid")
        if (
            not before.provenance_ref
            or not after.provenance_ref
            or before.provenance_ref == after.provenance_ref
        ):
            raise ValueError("governed_object_provenance_is_not_paired")
        before_attributes = dict(before.attributes)
        after_attributes = dict(after.attributes)
        if (
            before.lifecycle_state == after.lifecycle_state
            and before_attributes == after_attributes
        ):
            raise ValueError("governed_object_state_did_not_change")
        changed_keys = {
            key
            for key in set(before_attributes) | set(after_attributes)
            if before_attributes.get(key) != after_attributes.get(key)
        }
        if (
            before.lifecycle_state == after.lifecycle_state
            and changed_keys
            and changed_keys <= _BOOKKEEPING_STATE_KEYS
        ):
            raise ValueError("bookkeeping_only_state_transition")

    def record_independent_outcome(
        self,
        protocol_id: str,
        observation: IndependentOutcomeOracle,
    ) -> None:
        """Attach an external outcome attestation without deriving it from events."""

        if not protocol_is_live(self.protocols.get(protocol_id)):
            raise ValueError("protocol_not_enforceable")
        if not isinstance(observation, IndependentOutcomeOracle):
            raise TypeError("independent_outcome_requires_typed_oracle")
        if observation.oracle_type not in INDEPENDENT_OUTCOME_ORACLE_TYPES:
            raise ValueError("unsupported_independent_outcome_oracle_type")
        if observation.favorable_direction not in {"increase", "decrease"}:
            raise ValueError("invalid_outcome_favorable_direction")
        if not all(
            math.isfinite(float(value))
            for value in (observation.baseline_value, observation.observed_value)
        ):
            raise ValueError("non_finite_outcome_value")
        improved = (
            observation.observed_value > observation.baseline_value
            if observation.favorable_direction == "increase"
            else observation.observed_value < observation.baseline_value
        )
        if not improved:
            raise ValueError("independent_outcome_did_not_improve")
        if not observation.oracle_id or not observation.metric_name:
            raise ValueError("independent_outcome_identity_missing")
        if not observation.evaluator_id:
            raise ValueError("independent_outcome_evaluator_missing")
        if not observation.evidence_ref or observation.evidence_ref.startswith(
            ("pev_", "protocol_event:")
        ):
            raise ValueError("outcome_oracle_must_not_reference_protocol_ledger")
        if not re.fullmatch(r"[0-9a-f]{64}", observation.evidence_hash):
            raise ValueError("invalid_outcome_oracle_evidence_hash")

        enforcement = next(
            (
                event
                for event in self.events
                if event.event_id == observation.enforcement_event_id
                and event.protocol_id == protocol_id
                and event.event_type == "enforcement"
            ),
            None,
        )
        if enforcement is None:
            raise ValueError("outcome_oracle_enforcement_not_found")
        if not self._has_valid_typed_transition(enforcement):
            raise ValueError("outcome_oracle_enforcement_has_no_typed_transition")
        if (
            observation.governed_object_type
            != enforcement.data.get("governed_object_type")
            or observation.governed_object_id
            != enforcement.data.get("governed_object_id")
        ):
            raise ValueError("outcome_oracle_governed_object_mismatch")
        if (
            observation.governed_state_after_hash
            != enforcement.data.get("state_after_hash")
        ):
            raise ValueError("outcome_oracle_governed_state_mismatch")
        violation = next(
            (
                event
                for event in self.events
                if event.event_id
                == enforcement.data.get("violation_event_id")
            ),
            None,
        )
        excluded_evaluators = {enforcement.actor_id}
        if violation is not None:
            excluded_evaluators.add(violation.actor_id)
        if observation.evaluator_id in excluded_evaluators:
            raise ValueError("outcome_oracle_evaluator_is_not_independent")
        if observation.observed_tick < enforcement.tick:
            raise ValueError("outcome_oracle_precedes_enforcement")

        protocol = self.protocols[protocol_id]
        if any(
            item.oracle_id == observation.oracle_id
            for item in protocol.independent_outcome_oracles
        ):
            raise ValueError("duplicate_independent_outcome_oracle")
        protocol.independent_outcome_oracles.append(observation)
        self._ev(
            "impact",
            protocol_id,
            observation.evaluator_id,
            observation.observed_tick,
            {
                "evidence_role": "independent_outcome_oracle",
                "oracle_id": observation.oracle_id,
                "oracle_type": observation.oracle_type,
                "evaluator_id": observation.evaluator_id,
                "governed_object_type": observation.governed_object_type,
                "governed_object_id": observation.governed_object_id,
                "enforcement_event_id": observation.enforcement_event_id,
                "governed_state_after_hash": (
                    observation.governed_state_after_hash
                ),
                "metric_name": observation.metric_name,
                "baseline_value": observation.baseline_value,
                "observed_value": observation.observed_value,
                "favorable_direction": observation.favorable_direction,
                "observed_tick": observation.observed_tick,
                "evidence_ref": observation.evidence_ref,
                "evidence_hash": observation.evidence_hash,
            },
        )

    @classmethod
    def _has_valid_typed_transition(cls, event: ProtocolEvent) -> bool:
        data = event.data
        before_payload = data.get("state_before")
        after_payload = data.get("state_after")
        if (
            data.get("typed_governed_object") is not True
            or not isinstance(before_payload, Mapping)
            or not isinstance(after_payload, Mapping)
        ):
            return False
        try:
            before = GovernedObjectSnapshot(
                object_type=str(before_payload["object_type"]),
                object_id=str(before_payload["object_id"]),
                lifecycle_state=str(before_payload["lifecycle_state"]),
                observed_tick=int(before_payload["observed_tick"]),
                provenance_ref=str(before_payload["provenance_ref"]),
                attributes=dict(before_payload.get("attributes", {})),
            )
            after = GovernedObjectSnapshot(
                object_type=str(after_payload["object_type"]),
                object_id=str(after_payload["object_id"]),
                lifecycle_state=str(after_payload["lifecycle_state"]),
                observed_tick=int(after_payload["observed_tick"]),
                provenance_ref=str(after_payload["provenance_ref"]),
                attributes=dict(after_payload.get("attributes", {})),
            )
            cls._validate_governed_transition(before, after)
        except (KeyError, TypeError, ValueError):
            return False
        return bool(
            data.get("governed_object_type") == before.object_type
            and data.get("governed_object_id") == before.object_id
            and data.get("state_impact_ref") == before.object_id
            and re.fullmatch(
                r"[0-9a-f]{64}", str(data.get("state_before_hash") or "")
            )
            and re.fullmatch(
                r"[0-9a-f]{64}", str(data.get("state_after_hash") or "")
            )
            and stable_fingerprint(dict(before_payload))
            == data.get("state_before_hash")
            and stable_fingerprint(dict(after_payload))
            == data.get("state_after_hash")
            and data.get("state_before_hash") != data.get("state_after_hash")
        )

    @staticmethod
    def _is_valid_independent_outcome(
        observation: IndependentOutcomeOracle,
        enforcement: ProtocolEvent,
    ) -> bool:
        if observation.oracle_type not in INDEPENDENT_OUTCOME_ORACLE_TYPES:
            return False
        if observation.favorable_direction == "increase":
            improved = observation.observed_value > observation.baseline_value
        elif observation.favorable_direction == "decrease":
            improved = observation.observed_value < observation.baseline_value
        else:
            return False
        return bool(
            improved
            and math.isfinite(float(observation.baseline_value))
            and math.isfinite(float(observation.observed_value))
            and observation.enforcement_event_id == enforcement.event_id
            and observation.governed_object_type
            == enforcement.data.get("governed_object_type")
            and observation.governed_object_id
            == enforcement.data.get("governed_object_id")
            and observation.governed_state_after_hash
            == enforcement.data.get("state_after_hash")
            and observation.evaluator_id
            not in {enforcement.actor_id, ""}
            and observation.observed_tick >= enforcement.tick
            and bool(observation.evidence_ref)
            and not observation.evidence_ref.startswith(
                ("pev_", "protocol_event:")
            )
            and bool(re.fullmatch(r"[0-9a-f]{64}", observation.evidence_hash))
        )

    @staticmethod
    def _outcome_from_event(
        event: ProtocolEvent,
    ) -> IndependentOutcomeOracle | None:
        data = event.data
        if (
            event.event_type != "impact"
            or data.get("evidence_role") != "independent_outcome_oracle"
        ):
            return None
        try:
            return IndependentOutcomeOracle(
                oracle_id=str(data["oracle_id"]),
                oracle_type=str(data["oracle_type"]),
                evaluator_id=str(data["evaluator_id"]),
                governed_object_type=str(data["governed_object_type"]),
                governed_object_id=str(data["governed_object_id"]),
                enforcement_event_id=str(data["enforcement_event_id"]),
                governed_state_after_hash=str(
                    data["governed_state_after_hash"]
                ),
                metric_name=str(data["metric_name"]),
                baseline_value=float(data["baseline_value"]),
                observed_value=float(data["observed_value"]),
                favorable_direction=str(data["favorable_direction"]),
                observed_tick=int(data["observed_tick"]),
                evidence_ref=str(data["evidence_ref"]),
                evidence_hash=str(data["evidence_hash"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def set_impact(self, protocol_id: str, metrics: Dict[str, float]) -> None:
        protocol = self.protocols[protocol_id]
        if not protocol_is_live(protocol):
            raise ValueError("protocol_not_enforceable")
        protocol.impact_metrics.update(metrics)

    def amend(
        self,
        agent_id: str,
        protocol_id: str,
        tick: int = 0,
        *,
        revision_kind: str = "extend",
        source_proposal_id: str | None = None,
    ) -> ProtocolEvent:
        p = self.protocols[protocol_id]
        if not protocol_is_live(p):
            raise ValueError("protocol_not_enforceable")
        event = self._ev(
            "amendment",
            protocol_id,
            agent_id,
            tick,
            {
                key: value
                for key, value in {
                    "revision_kind": revision_kind,
                    "source_proposal_id": source_proposal_id,
                }.items()
                if value
            },
        )
        self._touch(p, tick)
        return event

    def obsolete(
        self,
        agent_id: str,
        protocol_id: str,
        tick: int = 0,
        *,
        source_proposal_id: str | None = None,
    ) -> ProtocolEvent:
        p = self.protocols[protocol_id]
        if not protocol_is_live(p):
            raise ValueError("protocol_not_enforceable")
        p.status = "obsolete"
        p.adoption_status = "obsolete"
        event = self._ev(
            "obsolete",
            protocol_id,
            agent_id,
            tick,
            (
                {"source_proposal_id": source_proposal_id}
                if source_proposal_id
                else {}
            ),
        )
        self._touch(p, tick)
        return event

    def attribute_impact(self, tick: int = 0) -> None:
        """Compute a same-ledger diagnostic that never establishes strong impact.

        Protocol-local and substrate-agnostic: every violation event IS an
        occurrence of the friction the protocol targets (its
        ``violation_condition``), so the targeted friction rate is
        violations / governed activity. Split each adopted protocol's
        post-adoption activity at its tick midpoint; a lower violation rate in
        the late half means the institution measurably reduced its targeted
        friction — the generic success metric ("the targeted failure/friction
        rate drops on subsequent cycles"). Writes the signed delta to
        ``impact_metrics`` (improvement is negative); a non-improvement writes
        0.0 so a protocol whose compliance regressed honestly drops back below
        the diagnostic signal. Because both numerator and denominator come from
        this registry's own event ledger, this value cannot serve as the
        independent outcome oracle required for strong institutional evidence.
        Emits an auditable diagnostic ``impact`` event on change.
        """
        adoption_ticks = {e.protocol_id: e.tick for e in self.events
                          if e.event_type == "adoption"}
        for pid, p in self.protocols.items():
            if (
                not protocol_is_live(p)
                or p.adoption_status != "adopted"
                or pid not in adoption_ticks
            ):
                continue
            t0 = adoption_ticks[pid]
            acts = [(e.tick, e.event_type) for e in self.events
                    if e.protocol_id == pid and e.tick >= t0
                    and e.event_type in ("use", "violation")]
            uses = sum(1 for _, kind in acts if kind == "use")
            violations = sum(1 for _, kind in acts if kind == "violation")
            if uses < USE_MIN or violations < 1:
                continue                      # nothing measurable yet
            last = max(t for t, _ in acts)
            if last <= t0:
                continue                      # no post-adoption window to split
            mid = (t0 + last) / 2.0
            early = [kind for t, kind in acts if t <= mid]
            late = [kind for t, kind in acts if t > mid]
            if not early or not late:
                continue
            early_rate = early.count("violation") / len(early)
            late_rate = late.count("violation") / len(late)
            delta = round(late_rate - early_rate, 4)
            value = delta if delta < 0 else 0.0
            if p.impact_metrics.get("violation_rate_delta") != value:
                self.set_impact(pid, {"violation_rate_delta": value})
                self._ev("impact", pid, "system", tick,
                         {"violation_rate_delta": value,
                          "early_violation_rate": round(early_rate, 4),
                          "late_violation_rate": round(late_rate, 4),
                          "window": [t0, last],
                          "evidence_role": "same_ledger_diagnostic"})

    def _touch(self, p: Protocol, tick: int) -> None:
        p.last_active_tick = max(p.last_active_tick, tick)
        p.persistence_ticks = p.last_active_tick - p.first_tick

    # -- emergence ----------------------------------------------------------
    def emergence_evidence(self, protocol_id: str) -> dict:
        """Return the conservative event-level evidence used by classification."""
        p = self.protocols[protocol_id]
        protocol_events = [
            event for event in self.events if event.protocol_id == protocol_id
        ]
        return self._emergence_evidence_from_order(p, protocol_events)

    def _emergence_evidence_from_order(
        self,
        p: Protocol,
        protocol_events: List[ProtocolEvent],
    ) -> dict:
        event_position = {
            event.event_id: index for index, event in enumerate(protocol_events)
        }
        proposal = next(
            (
                event
                for event in protocol_events
                if event.event_type == "proposal"
                and event.event_id == p.proposal_event_id
            ),
            None,
        )
        adoption_events = [
            event
            for event in protocol_events
            if event.event_type == "adoption"
            and proposal is not None
            and event_position[event.event_id] > event_position[proposal.event_id]
            and event.tick - proposal.tick >= REVIEW_MIN_TICKS
        ]
        adoption = adoption_events[0] if adoption_events else None
        pre_adoption_endorsers = (
            {
                event.actor_id
                for event in protocol_events
                if event.actor_id
                and event_position[event.event_id]
                <= event_position[adoption.event_id]
                and event.event_type in {"proposal", "support", "use"}
            }
            if adoption is not None
            else set()
        )
        post_adoption_uses = (
            [
                event
                for event in protocol_events
                if event.event_type == "use"
                and event_position[event.event_id] > event_position[adoption.event_id]
                and event.tick >= adoption.tick
            ]
            if adoption is not None
            else []
        )
        use_units = {
            (
                event.actor_id,
                event.data.get("episode_id")
                or event.data.get("task_id")
                or event.data.get("context_id")
                or f"tick:{event.tick}",
            )
            for event in post_adoption_uses
        }
        explicit_contexts = {
            value
            for event in post_adoption_uses
            for value in (
                event.data.get("episode_id"),
                event.data.get("task_id"),
                event.data.get("context_id"),
            )
            if value
        }
        episode_ids = {
            event.data.get("episode_id")
            for event in post_adoption_uses
            if event.data.get("episode_id")
        }
        distinct_use_ticks = {event.tick for event in post_adoption_uses}
        repeated_use = (
            len(post_adoption_uses) >= USE_MIN
            and len(use_units) >= USE_MIN
            and len(distinct_use_ticks) >= USE_MIN
        )
        persistent = bool(
            adoption is not None
            and repeated_use
            and max(distinct_use_ticks) - adoption.tick >= PERSIST_MIN
        )
        post_adoption_span_ticks = (
            max(distinct_use_ticks) - adoption.tick
            if adoption is not None and distinct_use_ticks
            else 0
        )
        violations_by_id = {
            event.event_id: event
            for event in protocol_events
            if event.event_type == "violation"
        }
        valid_enforcements = [
            event
            for event in protocol_events
            if event.event_type == "enforcement"
            and event.data.get("violation_event_id") in violations_by_id
            and event_position[
                violations_by_id[event.data["violation_event_id"]].event_id
            ]
            < event_position[event.event_id]
            and violations_by_id[event.data["violation_event_id"]].tick
            <= event.tick
            and violations_by_id[event.data["violation_event_id"]].actor_id
            != event.actor_id
            and bool(event.data.get("state_impact_ref"))
            and bool(event.data.get("blocked"))
            and self._has_valid_typed_transition(event)
        ]
        valid_enforcements_by_id = {
            event.event_id: event for event in valid_enforcements
        }
        outcome_attestations = [
            (event, observation)
            for event in protocol_events
            if (observation := self._outcome_from_event(event)) is not None
        ]
        valid_outcome_oracles = [
            observation
            for event, observation in outcome_attestations
            if observation.enforcement_event_id in valid_enforcements_by_id
            and event.actor_id == observation.evaluator_id
            and event.tick == observation.observed_tick
            and event_position[event.event_id]
            > event_position[observation.enforcement_event_id]
            and self._is_valid_independent_outcome(
                observation,
                valid_enforcements_by_id[observation.enforcement_event_id],
            )
            and observation.evaluator_id
            != violations_by_id[
                valid_enforcements_by_id[
                    observation.enforcement_event_id
                ].data["violation_event_id"]
            ].actor_id
        ]
        amendment_events = [
            event
            for event in protocol_events
            if event.event_type == "amendment"
            and adoption is not None
            and event_position[event.event_id] > event_position[adoption.event_id]
        ]
        repair_events = [
            event
            for event in amendment_events
            if event.data.get("revision_kind") in {"relax", "repair"}
        ] + [
            event
            for event in protocol_events
            if event.event_type == "obsolete"
            and adoption is not None
            and event_position[event.event_id] > event_position[adoption.event_id]
        ]
        emergence_tick = None
        if adoption is not None:
            cumulative_uses: list[ProtocolEvent] = []
            for event in post_adoption_uses:
                cumulative_uses.append(event)
                cumulative_units = {
                    (
                        item.actor_id,
                        item.data.get("episode_id")
                        or item.data.get("task_id")
                        or item.data.get("context_id")
                        or f"tick:{item.tick}",
                    )
                    for item in cumulative_uses
                }
                cumulative_ticks = {item.tick for item in cumulative_uses}
                if (
                    len(cumulative_uses) >= USE_MIN
                    and len(cumulative_units) >= USE_MIN
                    and len(cumulative_ticks) >= USE_MIN
                    and max(cumulative_ticks) - adoption.tick >= PERSIST_MIN
                ):
                    emergence_tick = max(cumulative_ticks)
                    break
        return {
            "proposed": proposal is not None,
            "proposal_event_id": proposal.event_id if proposal else None,
            "proposal_tick": proposal.tick if proposal else None,
            "adopted": (
                p.adoption_status == "adopted"
                and adoption is not None
                and len(pre_adoption_endorsers) >= self.min_supporters
            ),
            "adoption_event_id": adoption.event_id if adoption else None,
            "adoption_tick": adoption.tick if adoption else None,
            "pre_adoption_endorser_count": len(pre_adoption_endorsers),
            "post_adoption_use_count": len(post_adoption_uses),
            "independent_use_unit_count": len(use_units),
            "distinct_use_tick_count": len(distinct_use_ticks),
            "explicit_context_count": len(explicit_contexts),
            "episode_count": len(episode_ids),
            "repeated_use": repeated_use,
            "cross_context_reuse": len(explicit_contexts) >= USE_MIN,
            "cross_episode_reuse": len(episode_ids) >= USE_MIN,
            "persistent": persistent,
            "post_adoption_span_ticks": post_adoption_span_ticks,
            "emergence_tick": emergence_tick,
            "valid_enforcement_count": len(valid_enforcements),
            "valid_enforcement_event_ids": [
                event.event_id for event in valid_enforcements
            ],
            "has_paired_state_impact": bool(valid_enforcements),
            "valid_independent_outcome_count": len(valid_outcome_oracles),
            "valid_independent_outcome_oracle_ids": [
                observation.oracle_id for observation in valid_outcome_oracles
            ],
            "has_independent_outcome_impact": bool(valid_outcome_oracles),
            "has_measured_outcome_impact": bool(valid_outcome_oracles),
            "has_same_ledger_diagnostic_impact": any(
                float(value) != 0.0 for value in p.impact_metrics.values()
            ),
            "amendment_count": len(amendment_events),
            "repair_count": len(repair_events),
        }

    def shuffled_event_history_replay(
        self,
        protocol_id: str,
        *,
        seed: int,
    ) -> dict:
        """Replay the detector on a shuffled event order without changing runtime.

        This is the NC3 detector control. Agents act in the unmodified world;
        only the evaluator's causal ordering is permuted after the run. The
        control therefore measures order sensitivity without confounding the
        treatment with a different friction schedule or action history.
        """

        p = self.protocols[protocol_id]
        observed = [
            event for event in self.events if event.protocol_id == protocol_id
        ]
        shuffled = list(observed)
        random.Random(
            f"protocol_event_history:{protocol_id}:{int(seed)}"
        ).shuffle(shuffled)
        if len(shuffled) > 1 and [
            event.event_id for event in shuffled
        ] == [event.event_id for event in observed]:
            shuffled = shuffled[1:] + shuffled[:1]
        observed_evidence = self._emergence_evidence_from_order(p, observed)
        shuffled_evidence = self._emergence_evidence_from_order(p, shuffled)
        return {
            "protocol_id": protocol_id,
            "seed": int(seed),
            "runtime_mutated": False,
            "observed_event_order": [
                event.event_id for event in observed
            ],
            "shuffled_event_order": [
                event.event_id for event in shuffled
            ],
            "observed_evidence": observed_evidence,
            "shuffled_evidence": shuffled_evidence,
            "observed_level": self._level_from_evidence(observed_evidence),
            "shuffled_level": self._level_from_evidence(shuffled_evidence),
        }

    @staticmethod
    def _level_from_evidence(evidence: dict) -> str:
        weak = all(
            evidence[key]
            for key in ("proposed", "adopted", "repeated_use", "persistent")
        )
        strong = (
            weak
            and evidence["has_paired_state_impact"]
            and evidence["has_independent_outcome_impact"]
        )
        return "strong" if strong else ("weak" if weak else "none")

    def classify_emergence(self, protocol_id: str) -> str:
        p = self.protocols[protocol_id]
        evidence = self.emergence_evidence(protocol_id)
        p.emergence_level = self._level_from_evidence(evidence)
        return p.emergence_level

    def emerged(self, level: str = "weak") -> List[Protocol]:
        order = {"none": 0, "weak": 1, "strong": 2}
        want = order[level]
        out = []
        for pid in self.protocols:
            self.classify_emergence(pid)
        for p in self.protocols.values():
            if order[p.emergence_level] >= want:
                out.append(p)
        return out


__all__ = [
    "ADOPT_MIN_SUPPORTERS",
    "effective_min_supporters",
    "PERSIST_MIN",
    "REVIEW_MIN_TICKS",
    "USE_MIN",
    "ProtocolRegistry",
    "protocol_is_live",
]
