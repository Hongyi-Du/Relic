"""Content-addressed organizational-capability evidence for one OrgEnv run."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from environments.org_env.experiments.provenance import stable_fingerprint
from environments.org_env.backend.protocol.objects import (
    IndependentOutcomeOracle,
    Protocol,
    ProtocolEvent,
)
from environments.org_env.backend.protocol.registry import (
    ADOPT_MIN_SUPPORTERS,
    ProtocolRegistry,
    effective_min_supporters,
)


CAPABILITY_EVIDENCE_SCHEMA_VERSION = "orgenv_capability_evidence_v1"


def _event_payload(event: Any) -> dict[str, Any]:
    return {
        "event_id": str(getattr(event, "event_id", "")),
        "event_type": str(getattr(event, "event_type", "")),
        "protocol_id": str(getattr(event, "protocol_id", "")),
        "actor_id": str(getattr(event, "actor_id", "")),
        "tick": int(getattr(event, "tick", 0) or 0),
        "data": dict(getattr(event, "data", {}) or {}),
    }


def _mean(values: Sequence[float]) -> float | None:
    """Mean of the observations, or None when there were none.

    `time_to_emergence` and `persistence` are both "lower is better" readings of
    something that happened. With no protocol reaching emergence there is
    nothing to time, and 0.0 there reads as instant formation — the condition
    that never formed a capability scored best. Measured on one pilot pair: the
    arm with no protocols read 0.0 against 107.8 for the arm that formed two.
    """
    return sum(values) / len(values) if values else None


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Every metric here is named "protocol", so only protocols may enter it.
    # Document carriers are graded by the same criteria and belong in the
    # per-capability breakdown, but folding them into a rate called
    # ``weak_protocol_emergence_rate`` would make the name untrue and silently
    # change what every existing contrast measures.
    carrier_rows = list(rows)
    rows = [row for row in carrier_rows
            if str(row.get("carrier_kind") or "protocol") == "protocol"]
    protocol_count = len(rows)
    weak_count = sum(row["emergence_level"] in {"weak", "strong"} for row in rows)
    strong_count = sum(row["emergence_level"] == "strong" for row in rows)
    denominator = protocol_count or 1
    emergence_delays = [
        float(row["evidence"]["emergence_tick"])
        - float(row["evidence"]["proposal_tick"])
        for row in rows
        if row["evidence"].get("emergence_tick") is not None
        and row["evidence"].get("proposal_tick") is not None
    ]
    persistence_spans = [
        float(row["evidence"]["post_adoption_span_ticks"])
        for row in rows
        if row["evidence"].get("adopted")
    ]
    metrics = {
        "weak_protocol_emergence_rate": weak_count / denominator,
        "strong_protocol_emergence_rate": strong_count / denominator,
        "time_to_emergence": _mean(emergence_delays),
        "persistence": _mean(persistence_spans),
        "repeated_protocol_use_rate": (
            sum(row["evidence"]["repeated_use"] for row in rows) / denominator
        ),
        "cross_context_protocol_reuse_rate": (
            sum(row["evidence"]["cross_context_reuse"] for row in rows) / denominator
        ),
        "protocol_persistence_rate": (
            sum(row["evidence"]["persistent"] for row in rows) / denominator
        ),
        "valid_third_party_enforcement_rate": (
            sum(row["evidence"]["has_paired_state_impact"] for row in rows)
            / denominator
        ),
        "protocol_amendment_rate": (
            sum(row["evidence"]["amendment_count"] > 0 for row in rows) / denominator
        ),
        "protocol_repair_rate": (
            sum(row["evidence"]["repair_count"] > 0 for row in rows) / denominator
        ),
        "measurable_impact": (
            sum(row["evidence"]["has_measured_outcome_impact"] for row in rows)
            / denominator
        ),
        # The preregistration's MIDDLE tier: used beyond the episode it was
        # adopted in. It was never evaluated because nothing stamps an episode
        # id on an event, so the published cross-episode number came from a key
        # that is always unset and the reported figure answered a different
        # question (cross-CONTEXT reuse). Episode start/end ticks recover it
        # without inventing an identifier.
        "episode_transfer_rate": (
            sum(
                bool((row.get("episode_transfer") or {}).get(
                    "transferred_beyond_originating_episode"))
                for row in rows
            )
            / denominator
        ),
        # A rate of 0.0 must be readable as "did not transfer", never as "could
        # not be measured", so the measurable share is reported beside it.
        "episode_transfer_measurable_rate": (
            sum(
                bool((row.get("episode_transfer") or {}).get(
                    "episode_transfer_measurable"))
                for row in rows
            )
            / denominator
        ),
        # Every rate above divides by `protocol_count or 1`, so a run with no
        # protocols reports 0.0 across the board — indistinguishable from a run
        # whose protocols all failed to form. Four of those rates are components
        # of the organizational capability composite, and three of the four
        # conditions disable institutionalization by definition, so "not
        # applicable" and "tried and did not form" had the same reading. The
        # denominator has to travel with them, in `metrics`, because that is
        # what the run record copies and what the analysis consumes.
        "protocol_carrier_count": float(protocol_count),
    }
    return {
        "protocol_count": protocol_count,
        "weak_protocol_count": weak_count,
        "strong_protocol_count": strong_count,
        "carrier_count": len(carrier_rows),
        "document_carrier_count": len(carrier_rows) - protocol_count,
        # None survives rounding: it is the reading for "no observations", and
        # coercing it to 0.0 is exactly the confusion _mean exists to remove.
        "metrics": {
            key: (None if value is None else round(float(value), 12))
            for key, value in metrics.items()
        },
        # The per-capability claim spans every carrier: seven of the ten
        # preregistered capabilities are embedded in documents, not protocols.
        "by_capability": _by_capability(carrier_rows),
    }


# A capability cannot form faster than proposal -> adoption -> USE_MIN uses ->
# PERSIST_MIN ticks of persistence. Measured time_to_emergence on 336-tick runs
# is ~133 ticks, and a 168-tick sweep produced weak emergence 0.00 in EVERY
# condition with protocol_use_count 0 — a run too short to contain a formation
# reports the same all-zero table as a system where formation is broken, and a
# reader cannot tell those apart.
MIN_TICKS_FOR_FORMATION = 240


def institutionalization_caveat(world: Any) -> str:
    """Why a zero capability table may mean "not applicable" in this arm.

    P0..P4 all set institutionalization_enabled=False, which switches off the
    proposal -> adoption chain itself: those arms produce no protocols at all,
    so every emergence rate is 0.00 with protocol_count 0. That is the arm's
    DEFINITION, not a measurement of it, and it reads identically to a P5 run
    where formation was attempted and failed. Measured across a six-condition
    336-tick sweep: five arms reported protocol_use_count 0 for this reason.
    """
    if getattr(world, "institutionalization_enabled", True):
        return ""
    return (
        "institutionalization_disabled_in_this_condition: the proposal-to-"
        "adoption chain is switched off by the arm's definition, so capability "
        "formation is UNDEFINED here rather than measured as zero"
    )


def formation_horizon_caveat(world_tick: int) -> str:
    """Why an all-zero capability table may mean "too short", not "did not form"."""
    from environments.org_env.backend.protocol.registry import PERSIST_MIN, USE_MIN

    if int(world_tick) >= MIN_TICKS_FOR_FORMATION:
        return ""
    return (
        f"run_too_short_for_capability_formation:{int(world_tick)}_ticks; "
        f"a protocol needs adoption plus {USE_MIN} uses plus {PERSIST_MIN} ticks "
        f"of persistence, and measured time_to_emergence is ~133 ticks, so "
        f"emergence rates near zero here are not evidence that formation failed"
    )


# OrgEnv names the protocols it seeds after the RULE ("review before merge");
# the capability registry names them after the STANDING OBJECT the rule creates
# ("review gate"). Same thing, two vocabularies, so the join needs a spelling
# table or every seeded protocol lands in `unmapped` and the breakdown is empty.
#
# Only literals this codebase itself writes belong here — each one is a fact
# about our own naming, checkable by grep. Protocol types the LLM invents at
# runtime are deliberately NOT aliased: deciding that
# "beta_launch_gate_protocol" means release_governance would be an argument
# about text, and that is exactly the inference this metric must not make.
_PROTOCOL_TYPE_TO_CAPABILITY = {
    "experiment_logging": "experiment_tracker",
    "review_before_merge": "review_gate",
    # task_ownership was aliased while the codebase handed agents a written
    # ownership rule to accept. It writes no such literal now — an organization
    # that wants one has to arrive at it itself — so aliasing it would be a
    # guess about text, which is the inference this table exists to avoid.
}


def _canonical_capability_slug(protocol_type: str, carrier_kind: str = "protocol") -> str:
    from environments.org_env.experiments.capability_carriers import (
        canonical_capability,
    )

    mapped = canonical_capability(protocol_type, carrier_kind)
    return mapped or protocol_type


def _episode_transfer_evidence(
    world: Any,
    evidence: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Whether the capability was used OUTSIDE the episode it was adopted in.

    The preregistration's middle tier is "transfer beyond the originating
    episode", and it was never evaluated: nothing stamps an episode id on an
    event, so ``episode_count`` counted a key that is never set and the
    published number (cross-context reuse) answered a different question.

    Episodes do carry start/end ticks. Intersecting a use tick with those
    windows recovers the episode without inventing an identifier. Episodes
    overlap, so both sides are sets and transfer means a use episode the
    adoption tick was not inside.
    """
    from environments.org_env.experiments.capability_carriers import (
        episode_windows,
        episodes_covering,
    )

    windows = episode_windows(world)
    adoption_tick = evidence.get("adoption_tick")
    if not windows or adoption_tick is None:
        return {
            "originating_episode_ids": [],
            "transfer_episode_ids": [],
            "transferred_beyond_originating_episode": False,
            "episode_transfer_measurable": bool(windows),
        }
    origin = episodes_covering(windows, int(adoption_tick))
    use_episodes: set[str] = set()
    for row in event_rows:
        if str(row.get("event_type") or "") != "use":
            continue
        tick = row.get("tick")
        if tick is None:
            continue
        if int(tick) <= int(adoption_tick):
            continue
        use_episodes |= episodes_covering(windows, int(tick))
    transferred = sorted(use_episodes - origin)
    return {
        "originating_episode_ids": sorted(origin),
        "transfer_episode_ids": transferred,
        "transferred_beyond_originating_episode": bool(transferred),
        "episode_transfer_measurable": True,
    }


def _document_carrier_rows(world: Any) -> list[dict[str, Any]]:
    """Capability rows for shared documents, derived the same way protocols are.

    Seven of the ten preregistered capabilities have no protocol that carries
    them; they live in trackers and checklists. Without this channel they were
    not "absent from the data" but structurally unobservable, and a reader
    cannot tell those apart from a zero.

    The document's history is expressed as real ProtocolEvents and handed to the
    same ProtocolRegistry that grades protocols, so the identical four-part
    criteria apply and the validator's recompute-and-compare invariant holds for
    both channels. Anything less would make a document-borne capability a
    hand-asserted number sitting next to event-derived ones.
    """
    from environments.org_env.backend.protocol.objects import Protocol, ProtocolEvent
    from environments.org_env.backend.protocol.registry import (
        ProtocolRegistry,
        effective_min_supporters,
    )
    from environments.org_env.experiments.capability_carriers import (
        CARRIER_KIND_DOCUMENT,
        collect_document_carriers,
    )

    # Same roster cap the live registry uses: on a one-person roster the author
    # is the only agent who can ever touch a document, so a two-user rule would
    # make every document-borne capability unobservable rather than absent.
    min_supporters = effective_min_supporters(len(getattr(world, "agents", {}) or {}))
    rows: list[dict[str, Any]] = []
    for carrier in collect_document_carriers(world):
        events: list[ProtocolEvent] = []

        def _add(event_type: str, tick: int, actor: str, **data: Any) -> None:
            events.append(ProtocolEvent(
                event_id=f"{carrier.carrier_id}:{event_type}:{len(events)}",
                event_type=event_type,
                protocol_id=carrier.carrier_id,
                actor_id=actor or "unknown",
                tick=int(tick),
                data=dict(data),
            ))

        author = carrier.use_actors[0] if carrier.use_actors else "unknown"
        _add("proposal", carrier.created_tick, author)
        # A shared document has no ratification step of its own, so its
        # supporters are the distinct agents who went on to use it: a document
        # only its author ever touched is a private note, not an organizational
        # object - unless the author is the whole organization.
        supporters = list(dict.fromkeys(carrier.use_actors))[:max(min_supporters, 1)]
        for supporter in supporters:
            _add("support", carrier.created_tick, supporter)
        if len(supporters) >= min_supporters:
            _add("adoption", carrier.adopted_tick or carrier.created_tick, author)
        for index, tick in enumerate(carrier.use_ticks):
            actor = carrier.use_actors[index % len(carrier.use_actors)] if carrier.use_actors else ""
            context = carrier.use_contexts[index % len(carrier.use_contexts)] if carrier.use_contexts else ""
            _add("use", tick, actor, **({"context_id": context} if context else {}))

        protocol = Protocol(
            protocol_id=carrier.carrier_id,
            protocol_type=carrier.carrier_type,
            proposal_event_id=events[0].event_id if events else "",
            scope="team",
            target_process="",
            adoption_status="adopted" if len(supporters) >= min_supporters else "proposed",
            status="active",
        )
        registry = ProtocolRegistry(min_supporters=min_supporters)
        registry.protocols[protocol.protocol_id] = protocol
        registry.events = list(events)
        evidence = dict(registry.emergence_evidence(protocol.protocol_id))
        level = registry._level_from_evidence(evidence)
        event_rows = [_event_payload(event) for event in events]
        rows.append({
            "protocol_id": protocol.protocol_id,
            "protocol_type": protocol.protocol_type,
            "carrier_kind": CARRIER_KIND_DOCUMENT,
            "scope": protocol.scope,
            "target_process": "",
            "status": protocol.status,
            "adoption_status": protocol.adoption_status,
            "emergence_level": level,
            "impact_metrics": {},
            "evidence": evidence,
            "episode_transfer": _episode_transfer_evidence(world, evidence, event_rows),
            "events": event_rows,
            "event_order_hash": stable_fingerprint(event_rows),
        })
    return rows


def _by_capability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Break the protocol rows down by named organizational capability.

    The thesis is about specific capabilities forming, so "three protocols
    reached weak emergence" is not the claim a reader needs — "the review gate
    formed, the experiment tracker did not" is. A rate over an unnamed pool
    cannot distinguish those.

    Classification reuses ``society_core``'s registry unchanged, which matches
    exact slugs only and drops everything else. That is deliberately
    conservative: an open-vocabulary protocol type the LLM invented is left
    unmapped rather than argued into a bucket, so this can understate capability
    formation but never overstate it. ``unmapped_protocol_types`` reports what
    fell out, so the shortfall is visible instead of silently absorbed.
    """
    try:
        from society_core.organizational_capabilities import (
            CAPABILITY_KIND_ORGANIZATIONAL,
            ORGANIZATIONAL_CAPABILITY_LABELS,
            classify_capability_kind,
        )
    except Exception:
        return {}

    buckets: dict[str, dict[str, Any]] = {}
    unmapped: dict[str, int] = {}
    for row in rows:
        protocol_type = _canonical_capability_slug(
            str(row.get("protocol_type") or ""),
            str(row.get("carrier_kind") or "protocol"))
        if classify_capability_kind(protocol_type) != CAPABILITY_KIND_ORGANIZATIONAL:
            unmapped[protocol_type] = unmapped.get(protocol_type, 0) + 1
            continue
        bucket = buckets.setdefault(
            protocol_type,
            {"label": ORGANIZATIONAL_CAPABILITY_LABELS[protocol_type],
             "protocol_count": 0, "weak_count": 0, "strong_count": 0,
             "protocol_ids": []},
        )
        bucket["protocol_count"] += 1
        bucket["weak_count"] += int(row["emergence_level"] in {"weak", "strong"})
        bucket["strong_count"] += int(row["emergence_level"] == "strong")
        bucket["protocol_ids"].append(str(row.get("protocol_id") or ""))
    from society_core.organizational_capabilities import (
        EXPLORATORY_CAPABILITIES,
        PREREGISTERED_CAPABILITIES,
    )

    formed = {
        key for key, bucket in buckets.items() if bucket["weak_count"] > 0
    }
    return {
        "capabilities": {key: buckets[key] for key in sorted(buckets)},
        # The preregistered ten are the denominator a reader sees. Reporting it
        # against the whole registry would quietly enlarge the base with
        # capabilities the study never claimed to measure.
        "preregistered_capability_count": len(PREREGISTERED_CAPABILITIES),
        "preregistered_formed": sorted(formed & set(PREREGISTERED_CAPABILITIES)),
        "preregistered_formed_count": len(formed & set(PREREGISTERED_CAPABILITIES)),
        "preregistered_not_formed": sorted(
            set(PREREGISTERED_CAPABILITIES) - formed
        ),
        "exploratory_formed": sorted(formed & set(EXPLORATORY_CAPABILITIES)),
        "named_capability_count": len(ORGANIZATIONAL_CAPABILITY_LABELS),
        "observed_capability_count": len(buckets),
        "unmapped_protocol_count": sum(unmapped.values()),
        "unmapped_protocol_types": dict(sorted(unmapped.items())),
    }


def _final_evaluator_outcomes(
    payload: Mapping[str, Any],
) -> tuple[str, str, tuple[dict[str, Any], ...]]:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        result = payload.get("evaluation")
    if not isinstance(result, Mapping):
        result = payload
    dataset_id = str(result.get("dataset_id") or payload.get("dataset_id") or "")
    result_hash = str(result.get("result_hash") or payload.get("result_hash") or "")
    raw = result.get("outcomes")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return dataset_id, result_hash, ()
    outcomes = tuple(dict(row) for row in raw if isinstance(row, Mapping))
    return dataset_id, result_hash, outcomes


def _linked_outcomes(
    enforcement: ProtocolEvent,
    outcomes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    after = enforcement.data.get("state_after")
    if not isinstance(after, Mapping):
        return ()
    attributes = after.get("attributes")
    if not isinstance(attributes, Mapping):
        return ()
    issue_ids = {
        str(value) for value in (attributes.get("issue_ids") or ()) if str(value)
    }
    oracle_ids = {
        str(value)
        for value in (attributes.get("acceptance_oracle_ids") or ())
        if str(value)
    }
    if not issue_ids and not oracle_ids:
        return ()
    linked = []
    for outcome in outcomes:
        test_id = str(outcome.get("test_id") or "")
        outcome_issues = {
            str(value) for value in (outcome.get("issue_ids") or ()) if str(value)
        }
        if test_id in oracle_ids or issue_ids.intersection(outcome_issues):
            linked.append(dict(outcome))
    return tuple(linked)


def attach_independent_outcomes(
    capability_evidence: Mapping[str, Any],
    final_evaluator: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind post-rollout evaluator outcomes to typed enforcement transitions.

    This enriches the evaluator-facing packet only. It does not mutate the
    simulation world and does not treat an observed improvement as causal.
    """

    validate_organizational_capability_evidence(capability_evidence)
    dataset_id, result_hash, outcomes = _final_evaluator_outcomes(final_evaluator)
    if not dataset_id or len(result_hash) != 64 or not outcomes:
        # Final evaluation ran but yielded no linkable oracle outcomes: mark it
        # so a persistent 0.0 strong rate is attributable (distinct from the
        # in-sim "pending" stage) instead of silently ambiguous.
        unchanged = {
            key: value
            for key, value in dict(capability_evidence).items()
            if key != "evidence_hash"
        }
        unchanged.setdefault("min_supporters", ADOPT_MIN_SUPPORTERS)
        unchanged["oracle_attachment"] = "final_evaluation_without_oracle_outcomes"
        unchanged["evidence_hash"] = stable_fingerprint(unchanged)
        validate_organizational_capability_evidence(unchanged)
        return unchanged

    rows: list[dict[str, Any]] = []
    linked_oracle_ids: list[str] = []
    detector_replays: list[dict[str, Any]] = []
    replay_seed_by_protocol = {
        str(row.get("protocol_id") or ""): int(row.get("seed", 0))
        for row in (capability_evidence.get("detector_replays") or ())
        if isinstance(row, Mapping)
    }
    # `pev_N` is one sequence shared by every protocol -- a live registry hands
    # out ids from a single counter, so one protocol's events can run pev_2..290
    # and the next one's start at pev_291. Attaching an outcome below rebuilds a
    # throwaway registry PER protocol, and seeding its counter from that
    # protocol's own ids alone would mint an id another protocol already owns:
    # the run then dies in validation with
    # capability_evidence_event_ids_not_global_unique after all 336 ticks are
    # done, losing the record of a completed run (seen on a nine-protocol arm).
    # So the counter starts at the global maximum and carries forward across
    # protocols.
    next_sequence = max(
        (
            int(match.group(1))
            for candidate in (capability_evidence.get("protocols") or ())
            if isinstance(candidate, Mapping)
            for event in (candidate.get("events") or ())
            if isinstance(event, Mapping)
            and (match := re.fullmatch(r"pev_(\d+)", str(event.get("event_id") or "")))
        ),
        default=0,
    )
    for raw_row in capability_evidence.get("protocols") or ():
        row = dict(raw_row)
        protocol_id = str(row["protocol_id"])
        protocol = Protocol(
            protocol_id=protocol_id,
            protocol_type=str(row.get("protocol_type") or ""),
            proposal_event_id=str(
                (row.get("evidence") or {}).get("proposal_event_id") or ""
            ),
            scope=str(row.get("scope") or ""),
            target_process=str(row.get("target_process") or ""),
            adoption_status=str(row.get("adoption_status") or ""),
            impact_metrics=dict(row.get("impact_metrics") or {}),
            status=str(row.get("status") or ""),
        )
        registry = ProtocolRegistry(
            min_supporters=int(
                capability_evidence.get("min_supporters", ADOPT_MIN_SUPPORTERS) or 1
            )
        )
        registry.protocols[protocol_id] = protocol
        registry.events = [
            ProtocolEvent(**dict(event))
            for event in (row.get("events") or ())
            if isinstance(event, Mapping)
        ]
        # Global, not per-protocol: see next_sequence above.
        registry._seq = max(next_sequence, len(registry.events))
        existing_enforcement_ids = {
            str(event.data.get("enforcement_event_id") or "")
            for event in registry.events
            if event.event_type == "impact"
            and event.data.get("evidence_role") == "independent_outcome_oracle"
        }
        for enforcement in tuple(registry.events):
            if (
                enforcement.event_type != "enforcement"
                or enforcement.event_id in existing_enforcement_ids
                or not registry._has_valid_typed_transition(enforcement)
            ):
                continue
            linked = _linked_outcomes(enforcement, outcomes)
            if not linked:
                continue
            baseline_pass_rate = sum(
                item.get("baseline_status") == "passed" for item in linked
            ) / len(linked)
            candidate_pass_rate = sum(
                item.get("candidate_status") == "passed" for item in linked
            ) / len(linked)
            if candidate_pass_rate <= baseline_pass_rate:
                continue
            evidence_payload = {
                "dataset_id": dataset_id,
                "result_hash": result_hash,
                "enforcement_event_id": enforcement.event_id,
                "linked_outcomes": linked,
            }
            evidence_hash = stable_fingerprint(evidence_payload)
            oracle_id = f"outcome_{evidence_hash[:24]}"
            observation = IndependentOutcomeOracle(
                oracle_id=oracle_id,
                oracle_type="acceptance_test",
                evaluator_id=f"frozen_time_machine_evaluator:{dataset_id}",
                governed_object_type=str(enforcement.data["governed_object_type"]),
                governed_object_id=str(enforcement.data["governed_object_id"]),
                enforcement_event_id=enforcement.event_id,
                governed_state_after_hash=str(enforcement.data["state_after_hash"]),
                metric_name="linked_acceptance_oracle_pass_rate",
                baseline_value=baseline_pass_rate,
                observed_value=candidate_pass_rate,
                favorable_direction="increase",
                observed_tick=max(
                    int(capability_evidence.get("world_tick", 0) or 0),
                    enforcement.tick,
                ),
                evidence_ref=(
                    f"evaluator://{dataset_id}/{result_hash}/{enforcement.event_id}"
                ),
                evidence_hash=evidence_hash,
            )
            registry.record_independent_outcome(
                protocol_id,
                observation,
            )
            linked_oracle_ids.append(oracle_id)

        event_rows = [_event_payload(event) for event in registry.events]
        # Carry the counter forward so the next protocol cannot reuse an id this
        # one just minted.
        next_sequence = max(next_sequence, registry._seq)
        evidence = registry.emergence_evidence(protocol_id)
        row.update(
            {
                "emergence_level": registry._level_from_evidence(evidence),
                "evidence": evidence,
                # episode_transfer is deliberately NOT rewritten here: attaching
                # an oracle adds outcome records, not use events, so the
                # episodes a capability was used in cannot have changed. The
                # value already on the row stays authoritative.
                "events": event_rows,
                "event_order_hash": stable_fingerprint(event_rows),
            }
        )
        rows.append(row)
        if protocol_id in replay_seed_by_protocol:
            detector_replays.append(
                registry.shuffled_event_history_replay(
                    protocol_id,
                    seed=replay_seed_by_protocol[protocol_id],
                )
            )

    aggregate = _aggregate(rows)
    event_ids = [str(event["event_id"]) for row in rows for event in row["events"]]
    enriched = {
        key: value
        for key, value in capability_evidence.items()
        if key
        not in {
            "protocol_count",
            "weak_protocol_count",
            "strong_protocol_count",
            "metrics",
            "protocols",
            "capability_event_ids",
            "detector_replays",
            "evidence_hash",
            "independent_outcome_linkage",
        }
    }
    enriched.update(
        {
            **aggregate,
            "protocols": rows,
            "capability_event_ids": list(dict.fromkeys(event_ids)),
            "detector_replays": detector_replays,
            "independent_outcome_linkage": {
                "dataset_id": dataset_id,
                "result_hash": result_hash,
                "linked_oracle_ids": list(linked_oracle_ids),
                "linked_oracle_count": len(linked_oracle_ids),
                "causal_claim": False,
            },
            "oracle_attachment": "attached",
        }
    )
    enriched["evidence_hash"] = stable_fingerprint(enriched)
    validate_organizational_capability_evidence(enriched)
    return enriched


def build_organizational_capability_evidence(
    world: Any,
    *,
    shuffled_replay_seed: int | None = None,
) -> dict[str, Any]:
    """Build a self-contained evidence packet from the protocol event ledger.

    The packet is evaluator-facing. It records the event order used by the
    detector, the exact evidence predicates, aggregate metrics, and an
    integrity hash. No capability is inferred from labels or artifacts alone.
    """

    registry = getattr(world, "protocol_registry", None)
    protocols = list(getattr(registry, "protocols", {}).values()) if registry else []
    rows: list[dict[str, Any]] = []
    all_event_ids: list[str] = []
    detector_replays: list[dict[str, Any]] = []
    for protocol in sorted(protocols, key=lambda item: item.protocol_id):
        events = [
            event
            for event in (getattr(registry, "events", ()) or ())
            if getattr(event, "protocol_id", None) == protocol.protocol_id
        ]
        event_rows = [_event_payload(event) for event in events]
        evidence = dict(registry.emergence_evidence(protocol.protocol_id))
        level = registry._level_from_evidence(evidence)
        transfer = _episode_transfer_evidence(world, evidence, event_rows)
        all_event_ids.extend(row["event_id"] for row in event_rows)
        rows.append(
            {
                "protocol_id": str(protocol.protocol_id),
                "protocol_type": str(protocol.protocol_type),
                "scope": str(getattr(protocol, "scope", "")),
                "target_process": str(getattr(protocol, "target_process", "")),
                "status": str(getattr(protocol, "status", "")),
                "adoption_status": str(getattr(protocol, "adoption_status", "")),
                "emergence_level": level,
                "impact_metrics": {
                    str(key): float(value)
                    for key, value in sorted(
                        (getattr(protocol, "impact_metrics", {}) or {}).items()
                    )
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                },
                "evidence": evidence,
                # Beside `evidence`, never inside it: the validator recomputes
                # `evidence` from the events and demands exact equality, which
                # is what guarantees no predicate was hand-asserted. Transfer is
                # derived from episode WINDOWS rather than from the event
                # stream, so it rides alongside instead of weakening that check.
                "episode_transfer": transfer,
                "events": event_rows,
                "event_order_hash": stable_fingerprint(event_rows),
            }
        )
        if shuffled_replay_seed is not None:
            detector_replays.append(
                registry.shuffled_event_history_replay(
                    protocol.protocol_id,
                    seed=int(shuffled_replay_seed),
                )
            )

    # Seven of the ten preregistered capabilities are carried by shared
    # documents rather than protocols; without this channel they could never
    # appear in a report at all.
    document_rows = _document_carrier_rows(world)
    rows.extend(document_rows)
    for row in document_rows:
        all_event_ids.extend(item["event_id"] for item in row["events"])
    aggregate = _aggregate(rows)
    payload: dict[str, Any] = {
        "schema_version": CAPABILITY_EVIDENCE_SCHEMA_VERSION,
        "run_id": str(getattr(world, "run_id", "")),
        "world_tick": int(getattr(world, "world_tick", 0) or 0),
        # Travels with the packet because the validator recomputes every
        # predicate from the stored events alone and has no roster to cap
        # against; without it a solo run's rows would fail their own recompute.
        "min_supporters": effective_min_supporters(
            len(getattr(world, "agents", {}) or {})
        ),
        **aggregate,
        "protocols": rows,
        "capability_event_ids": list(dict.fromkeys(all_event_ids)),
        "detector_replays": detector_replays,
        # Explicit stage marker: strong emergence and measurable_impact REQUIRE
        # an evaluator-owned independent oracle that only exists at final
        # evaluation. "pending" tells any mid-run reader that a 0.0 strong rate
        # is expected at this stage, not evidence of a broken chain.
        "oracle_attachment": "pending",
        "claim_boundary": {
            "weak": (
                "proposal, valid adoption, repeated independent post-adoption "
                "use, and at least 48 ticks of persistence"
            ),
            "strong": (
                "weak evidence plus linked later third-party enforcement, a "
                "typed before/after transition of the same governed object, "
                "and a directionally improved independent evaluator oracle"
            ),
            "causal_impact": (
                "not established by this packet; requires a registered "
                "counterfactual contrast"
            ),
            # Empty unless the run was too short to CONTAIN a formation. Without
            # it an all-zero table from a 168-tick run is indistinguishable from
            # one produced by a system where formation is broken.
            "formation_horizon": formation_horizon_caveat(
                int(getattr(world, "world_tick", 0) or 0)
            ),
            # Empty unless the arm switches the formation chain off entirely.
            # Without it, P0..P4's structural zero is indistinguishable from a
            # P5 run where formation was attempted and did not happen.
            "institutionalization": institutionalization_caveat(world),
        },
    }
    payload["evidence_hash"] = stable_fingerprint(payload)
    validate_organizational_capability_evidence(payload)
    return payload


def validate_organizational_capability_evidence(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != CAPABILITY_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unknown_capability_evidence_schema")
    declared_hash = payload.get("evidence_hash")
    if not isinstance(declared_hash, str) or len(declared_hash) != 64:
        raise ValueError("invalid_capability_evidence_hash")
    unhashed = dict(payload)
    unhashed.pop("evidence_hash", None)
    if stable_fingerprint(unhashed) != declared_hash:
        raise ValueError("capability_evidence_hash_mismatch")
    protocols = payload.get("protocols")
    if not isinstance(protocols, Sequence) or isinstance(protocols, (str, bytes)):
        raise ValueError("capability_evidence_protocols_must_be_a_list")
    # ``protocols`` carries every capability carrier; ``protocol_count`` counts
    # only the protocol-borne ones, because every metric keyed on it is named
    # after protocols. Both are checked so neither can drift from the rows.
    carrier_rows = list(protocols)
    protocol_rows = [
        row for row in carrier_rows
        if isinstance(row, Mapping)
        and str(row.get("carrier_kind") or "protocol") == "protocol"
    ]
    if int(payload.get("protocol_count", -1)) != len(protocol_rows):
        raise ValueError("capability_evidence_protocol_count_mismatch")
    if int(payload.get("carrier_count", len(carrier_rows))) != len(carrier_rows):
        raise ValueError("capability_evidence_carrier_count_mismatch")
    ids = [row.get("protocol_id") for row in protocols if isinstance(row, Mapping)]
    if len(ids) != len(protocols) or len(set(ids)) != len(ids):
        raise ValueError("capability_evidence_protocol_ids_invalid")
    min_supporters = int(payload.get("min_supporters", ADOPT_MIN_SUPPORTERS) or 1)
    if min_supporters < 1 or min_supporters > ADOPT_MIN_SUPPORTERS:
        raise ValueError("capability_evidence_min_supporters_invalid")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("capability_evidence_metrics_required")
    for name, value in metrics.items():
        if not isinstance(name, str):
            raise ValueError("capability_evidence_metrics_must_be_finite")
        # None is a legal reading: "no observations to average". A mean over an
        # empty set is not zero, and reporting it as zero made the condition
        # that formed no capability score best on every lower-is-better metric.
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("capability_evidence_metrics_must_be_finite")
    rebuilt_rows: list[dict[str, Any]] = []
    all_ids: list[str] = []
    replay_by_protocol = {
        str(row.get("protocol_id")): row
        for row in (payload.get("detector_replays") or ())
        if isinstance(row, Mapping)
    }
    for row in protocols:
        if not isinstance(row, Mapping):
            raise ValueError("capability_evidence_protocol_row_invalid")
        events_raw = row.get("events")
        if not isinstance(events_raw, Sequence) or isinstance(
            events_raw,
            (str, bytes),
        ):
            raise ValueError("capability_evidence_events_must_be_a_list")
        events = []
        for item in events_raw:
            if (
                not isinstance(item, Mapping)
                or item.get("protocol_id") != row.get("protocol_id")
                or not str(item.get("event_id") or "")
                or not isinstance(item.get("data"), Mapping)
            ):
                raise ValueError("capability_evidence_event_invalid")
            events.append(ProtocolEvent(**dict(item)))
        event_ids = [event.event_id for event in events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("capability_evidence_duplicate_event_id")
        all_ids.extend(event_ids)
        protocol = Protocol(
            protocol_id=str(row["protocol_id"]),
            protocol_type=str(row.get("protocol_type") or ""),
            proposal_event_id=str(
                (row.get("evidence") or {}).get("proposal_event_id") or ""
            ),
            scope=str(row.get("scope") or ""),
            target_process=str(row.get("target_process") or ""),
            adoption_status=str(row.get("adoption_status") or ""),
            impact_metrics=dict(row.get("impact_metrics") or {}),
            status=str(row.get("status") or ""),
        )
        registry = ProtocolRegistry(min_supporters=min_supporters)
        registry.protocols[protocol.protocol_id] = protocol
        registry.events = events
        observed = registry._emergence_evidence_from_order(
            protocol,
            events,
        )
        level = registry._level_from_evidence(observed)
        if observed != row.get("evidence"):
            raise ValueError("capability_evidence_predicates_not_event_derived")
        if level != row.get("emergence_level"):
            raise ValueError("capability_emergence_level_not_event_derived")
        if stable_fingerprint(list(events_raw)) != row.get("event_order_hash"):
            raise ValueError("capability_event_order_hash_mismatch")
        replay = replay_by_protocol.get(protocol.protocol_id)
        if replay is not None:
            expected_replay = registry.shuffled_event_history_replay(
                protocol.protocol_id,
                seed=int(replay.get("seed", 0)),
            )
            if dict(replay) != expected_replay:
                raise ValueError("capability_detector_replay_not_event_derived")
        rebuilt_rows.append(dict(row))
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("capability_evidence_event_ids_not_global_unique")
    if list(dict.fromkeys(all_ids)) != list(payload.get("capability_event_ids") or ()):
        raise ValueError("capability_event_id_index_mismatch")
    aggregate = _aggregate(rebuilt_rows)
    for field in (
        "protocol_count",
        "weak_protocol_count",
        "strong_protocol_count",
        "metrics",
    ):
        if payload.get(field) != aggregate[field]:
            raise ValueError(f"capability_evidence_aggregate_mismatch:{field}")


__all__ = [
    "CAPABILITY_EVIDENCE_SCHEMA_VERSION",
    "attach_independent_outcomes",
    "build_organizational_capability_evidence",
    "validate_organizational_capability_evidence",
]
