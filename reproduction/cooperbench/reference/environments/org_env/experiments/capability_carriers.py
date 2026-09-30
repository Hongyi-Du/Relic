"""Capability carriers: the objects a capability can actually be embedded in.

The thesis defines organizational embeddedness as a capability living in "a
shared artifact, workflow, role relation, protocol, review gate, tracker or
governance object". The evidence builder only ever looked at protocols, so
seven of the ten preregistered capabilities had no channel through which they
could be observed at all — not "did not form", but *unobservable*. A real run
carried a document literally named ``doc_customer_triage`` and a
``doc_experiment_tracker`` that events recorded being logged to, and neither
could ever appear in a capability report.

A carrier is therefore a protocol OR a shared document. Both are reduced to the
same four-part evidence the protocol path already uses — proposed, adopted,
repeatedly used, persisted — so the two channels stay comparable and a reader
cannot tell from the number which kind of object carried it.

Carrier types map to capabilities through tables of literals THIS codebase
writes. An LLM-invented protocol type or a free-text document is deliberately
left unmapped rather than argued into a bucket: deciding that
"beta_launch_gate_protocol" means release_governance is an inference about text,
and a capability metric must not make it. Unmapped carriers are counted and
reported so the shortfall stays visible.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CARRIER_KIND_PROTOCOL",
    "CARRIER_KIND_DOCUMENT",
    "DOCUMENT_TYPE_TO_CAPABILITY",
    "PROTOCOL_TYPE_TO_CAPABILITY",
    "Carrier",
    "canonical_capability",
    "collect_document_carriers",
    "episode_windows",
    "episodes_covering",
]

CARRIER_KIND_PROTOCOL = "protocol"
CARRIER_KIND_DOCUMENT = "document"

# OrgEnv names the protocols it seeds after the RULE ("review before merge");
# the capability registry names them after the STANDING OBJECT the rule creates
# ("review gate"). Same concept, two vocabularies, so the join needs a spelling
# table or every seeded protocol lands unmapped.
PROTOCOL_TYPE_TO_CAPABILITY: dict[str, str] = {
    "experiment_logging": "experiment_tracker",
    "review_before_merge": "review_gate",
    "task_ownership": "ownership_map",
}

# Document types this codebase writes, mapped to the capability the document
# embodies. Only literals emitted by our own code appear here — each is
# grep-checkable, and a test pins that a mapped type is still written somewhere.
DOCUMENT_TYPE_TO_CAPABILITY: dict[str, str] = {
    "experiment_tracker": "experiment_tracker",
    "customer_feedback_summary": "customer_triage",
    "design_doc": "shared_memory",
    "claim_tracker": "claim_evidence_binding",
    "source_tracker": "source_credibility_workflow",
    "evidence_checklist": "evidence_workflow",
    "budget_policy": "budget_rule",
    "release_checklist": "release_governance",
}


def canonical_capability(carrier_type: str, kind: str) -> str:
    """The capability a carrier type embodies, or "" when it maps to none."""
    table = (PROTOCOL_TYPE_TO_CAPABILITY if kind == CARRIER_KIND_PROTOCOL
             else DOCUMENT_TYPE_TO_CAPABILITY)
    carrier_type = str(carrier_type or "")
    if carrier_type in table:
        return table[carrier_type]
    # A type that already IS a registry slug needs no translation.
    from society_core.organizational_capabilities import (
        ORGANIZATIONAL_CAPABILITY_LABELS,
    )

    return carrier_type if carrier_type in ORGANIZATIONAL_CAPABILITY_LABELS else ""


@dataclass(frozen=True)
class Carrier:
    """One object a capability is embedded in, with its event-level evidence."""

    carrier_id: str
    kind: str
    carrier_type: str
    capability: str
    created_tick: int
    adopted_tick: int | None
    use_ticks: tuple[int, ...] = ()
    use_actors: tuple[str, ...] = ()
    use_contexts: tuple[str, ...] = ()

    @property
    def use_count(self) -> int:
        return len(self.use_ticks)

    @property
    def last_use_tick(self) -> int | None:
        return max(self.use_ticks) if self.use_ticks else None

    def persistence_span(self) -> int:
        """Ticks between adoption and the last recorded use."""
        if self.adopted_tick is None or not self.use_ticks:
            return 0
        return max(0, max(self.use_ticks) - int(self.adopted_tick))


def episode_windows(world: Any) -> tuple[tuple[str, int, int], ...]:
    """``(episode_id, start_tick, end_tick)`` for every episode the run opened.

    Episodes are not stamped on events, which is why cross-episode reuse was
    computed from context ids and the proposal's transfer tier was never
    evaluated. The windows have always been there; intersecting a use tick with
    them recovers the episode without inventing an identifier.
    """
    manager = getattr(world, "episode_manager", None)
    episodes = getattr(manager, "episodes", None) or {}
    out: list[tuple[str, int, int]] = []
    for episode_id, episode in episodes.items():
        start = getattr(episode, "start_tick", None)
        if start is None:
            continue
        end = getattr(episode, "end_tick", None)
        if end is None:
            end = getattr(episode, "updated_at_tick", None)
        out.append((str(episode_id), int(start), int(end if end is not None else start)))
    return tuple(sorted(out, key=lambda row: (row[1], row[0])))


def episodes_covering(windows: Sequence[tuple[str, int, int]], tick: int) -> frozenset[str]:
    """Episode ids whose window contains ``tick``. Episodes overlap, so this is a set."""
    return frozenset(
        episode_id for episode_id, start, end in windows if start <= int(tick) <= end
    )


def _document_fields(document: Any) -> Mapping[str, Any]:
    return document if isinstance(document, Mapping) else getattr(document, "__dict__", {}) or {}


def _document_use_events(world: Any, document_id: str) -> list[Mapping[str, Any]]:
    """Events that record this document being USED, not merely existing.

    Creation is not use — a tracker nobody logs to has not become a capability,
    which is exactly the distinction the four-part criteria exist to draw.
    """
    hits: list[Mapping[str, Any]] = []
    for raw in getattr(world, "events", []) or []:
        event = raw if isinstance(raw, Mapping) else getattr(raw, "__dict__", {}) or {}
        if not event:
            continue
        referenced = any(
            str(event.get(key) or "") == document_id
            for key in ("doc_id", "document_id", "target_id", "artifact_id", "tracker_id")
        )
        if not referenced:
            continue
        hits.append(event)
    return hits


def collect_document_carriers(world: Any) -> tuple[Carrier, ...]:
    """Shared documents that embody a registry capability, with their use evidence."""
    documents = getattr(world, "documents", None) or {}
    carriers: list[Carrier] = []
    for document_id, document in documents.items():
        fields = _document_fields(document)
        capability = canonical_capability(
            str(fields.get("doc_type") or ""), CARRIER_KIND_DOCUMENT)
        if not capability:
            continue
        uses = _document_use_events(world, str(document_id))
        ticks = tuple(sorted(int(e.get("tick") or 0) for e in uses))
        actors = tuple(sorted({str(e.get("agent_id") or e.get("actor_id") or "") for e in uses} - {""}))
        contexts = tuple(sorted({
            str(e.get("context_id") or e.get("task_id") or "") for e in uses} - {""}))
        created = int(fields.get("created_tick") or fields.get("last_updated_tick") or 0)
        carriers.append(Carrier(
            carrier_id=str(document_id),
            kind=CARRIER_KIND_DOCUMENT,
            carrier_type=str(fields.get("doc_type") or ""),
            capability=capability,
            created_tick=created,
            # A shared document's adoption is its creation: unlike a protocol it
            # carries no separate ratification step, so requiring one would make
            # every document-borne capability unobservable for the wrong reason.
            adopted_tick=created,
            use_ticks=ticks,
            use_actors=actors,
            use_contexts=contexts,
        ))
    return tuple(sorted(carriers, key=lambda c: c.carrier_id))
