"""P3 Organization-as-a-Service façade.

The transparent P2 seat view is deliberately rich enough for a person to
inspect.  P3 is a different contract: by default the organization is hidden
behind a concise liaison state.  This module is therefore a *projection of a
filtered P2 seat view*, never a second reader of the world or inspector.

Opaque evidence handles are bound to the holder's seat token.  The server
resolves a handle by rebuilding the currently visible P2 brief and accepts it
only if it still names a current P2 source.  A handle consequently cannot be
used to guess an object id, cross a seat boundary, or retain access after a
seat is released.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from environments.org_env.human.organization_brief import build_organization_brief
from environments.org_env.human.seat import SeatUnavailable
from environments.org_env.human.visibility import FORBIDDEN_IN_SEAT_VIEW
from environments.org_env.human.working_agent import (
    MAX_STEPS,
    build_recent_activity_snapshot,
    build_team_progress_snapshot,
)


P3_FIXED_SEAT_ID = "victor"
LIAISON_MODEL_ATTEMPTS = 3

_LIAISON_ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["ordinary", "grounded_answer", "attention_preference", "meeting_context",
                      "meeting_plan", "open_resource", "clarification"],
        },
        "message_delivery": {
            "type": "string", "enum": ["", "all_messages", "secretary_triage"],
        },
        "meeting_id": {"type": "string"},
        "decision": {"type": "string", "enum": ["", "attend", "skip", "delegate"]},
        "viewpoint": {"type": "string"},
        "return_focus": {"type": "string"},
        # Resource identity is chosen semantically by the model from the
        # current, seat-visible resource candidates.  It is never accepted
        # from a browser endpoint: the browser receives only an opaque ri_ ref.
        "resource_kind": {"type": "string"},
        "resource_id": {"type": "string"},
        "resource_section": {
            "type": "string",
            "enum": ["overview", "review", "ci", "commits", "patches", "files", "diff", "provenance"],
        },
        "reply": {"type": "string"},
        "references": {
            "type": "array",
            "maxItems": 24,
            "items": {
                "type": "object",
                "properties": {
                    "object_id": {"type": "string"},
                    "kind": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["object_id"],
            },
        },
    },
    "required": ["route", "reply"],
    "allOf": [{
        "if": {
            "properties": {"route": {"const": "open_resource"}},
            "required": ["route"],
        },
        "then": {
            "required": ["resource_kind", "resource_id", "resource_section"],
        },
    }],
}


def _opaque_ref(token: str, source: Dict[str, Any]) -> str:
    """Return a deterministic, seat-token-bound opaque evidence handle."""
    material = f"{source.get('kind', '')}\x1f{source.get('id', '')}".encode("utf-8")
    digest = hmac.new(token.encode("utf-8"), material, hashlib.sha256).hexdigest()
    return f"ev_{digest[:32]}"


def _resource_ref(token: str, kind: str, resource_id: str) -> str:
    """Return an opaque handle for one resource, bound to exactly one token."""
    material = f"resource\x1f{kind}\x1f{resource_id}".encode("utf-8")
    digest = hmac.new(token.encode("utf-8"), material, hashlib.sha256).hexdigest()
    return f"ri_{digest[:32]}"


class LiaisonFacade:
    """P3-only API boundary over a ``HumanApi`` instance.

    The facade may delegate seat lifecycle, clock, and draft operations to
    ``HumanApi``.  Its own reads intentionally do not delegate to ``view()``,
    because that route returns the transparent P2 view with raw collections.
    """

    def __init__(self, human_api: Any) -> None:
        self.human = human_api
        # This is HCI-local delivery state, not a world notification channel.
        # It contains only summaries derived from an already filtered P2 view
        # and is dropped when its token's seat is released.
        self._notification_history: Dict[str, List[Dict[str, Any]]] = {}
        self._notification_seen: Dict[str, Dict[str, None]] = {}
        # P3's secretary mediates the human's attention.  These records are
        # local to the authenticated liaison session: they are not organization
        # objects, do not create a new member, and never confer an action the
        # Victor seat does not already have through P2.
        self._meeting_plans: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._meeting_history: Dict[str, List[Dict[str, Any]]] = {}
        self._message_history: Dict[str, List[Dict[str, Any]]] = {}
        self._message_seen: Dict[str, Dict[str, None]] = {}
        self._message_delivery: Dict[str, str] = {}
        # A resource focus is UI-local metadata for one already-recorded
        # secretary reply.  It does not grant access: opening always resolves
        # its ri_ handle through current token and P2 visibility again.
        self._resource_focus: Dict[str, Dict[str, Dict[str, str]]] = {}
        # A failed semantic-router call is retryable against the original,
        # already-recorded human message; it never falls through to heuristics.
        self._failed_routes: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # -- P3 fixed seat lifecycle / clock isolation ------------------------
    def session(self, token: str = "") -> Dict[str, Any]:
        """Create or resume P3's one fixed Victor seat.

        The browser never chooses an ``agent_id``.  A supplied token is only
        reusable when it already authenticates Victor; tokens for every other
        P1/P2 seat are rejected before any P3 state can be read.  With no
        token, another P3 client attaches to the already-held Victor seat.
        This is deliberate: Codex webviews and ordinary browsers do not share
        localStorage, but P3 represents one human perspective and one thread.
        """
        if token:
            try:
                seat = self.human._seat(token)
            except (SeatUnavailable, RuntimeError) as exc:
                return {"error": str(exc)}
            if seat.agent_id != P3_FIXED_SEAT_ID:
                return {"error": f"p3_fixed_seat_required:{P3_FIXED_SEAT_ID}"}
            return {"agent_id": seat.agent_id, "token": seat.token,
                    "claimed_at": seat.claimed_at, "reused": True}

        try:
            existing = self.human.runtime().seats.get(P3_FIXED_SEAT_ID)
        except RuntimeError as exc:
            return {"error": str(exc)}
        if existing is not None:
            existing.touch()
            return {"agent_id": existing.agent_id, "token": existing.token,
                    "claimed_at": existing.claimed_at, "reused": True,
                    "shared_fixed_session": True}

        claimed = self.human.claim(P3_FIXED_SEAT_ID, "P3 human")
        if claimed.get("error"):
            return claimed
        return {**claimed, "reused": False, "shared_fixed_session": True}

    def release(self, token: str) -> Dict[str, Any]:
        self._notification_history.pop(token, None)
        self._notification_seen.pop(token, None)
        self._meeting_plans.pop(token, None)
        self._meeting_history.pop(token, None)
        self._message_history.pop(token, None)
        self._message_seen.pop(token, None)
        self._message_delivery.pop(token, None)
        self._resource_focus.pop(token, None)
        self._failed_routes.pop(token, None)
        return self.human.release(token)

    def runtime_status(self) -> Dict[str, Any]:
        return self.human.runtime_status()

    def runtime_start(self, seconds_per_tick: Optional[float] = None) -> Dict[str, Any]:
        return self.human.runtime_start(seconds_per_tick)

    def runtime_pause(self) -> Dict[str, Any]:
        return self.human.runtime_pause()

    # -- visibility-bounded read model -------------------------------------
    def _authenticated_view(self, token: str) -> Tuple[Any, Dict[str, Any]]:
        """Get the cached P2 view for this token, without exposing it."""
        seat = self.human._seat(token)
        if seat.agent_id != P3_FIXED_SEAT_ID:
            raise SeatUnavailable(f"p3_fixed_seat_required:{P3_FIXED_SEAT_ID}")
        runtime = self.human.runtime()
        try:
            return seat, runtime.seat_view(seat.agent_id)
        except KeyError:
            runtime.refresh_views()
            return seat, runtime.seat_view(seat.agent_id)

    def _brief(self, token: str) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        seat, view = self._authenticated_view(token)
        # The brief is rebuilt only from this completed, filtered P2 view.  It
        # is not allowed to look at the world to enrich or explain an item.
        return seat, view, build_organization_brief(view)

    def _source_index(self, token: str, brief: Dict[str, Any],
                      view: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Map only P2-visible sources to this token's opaque handles.

        Brief citations are the normal path.  P3 notifications additionally
        cite filtered task/event summaries, so a person can follow a visible
        update through the same evidence -> trace boundary without receiving a
        raw feed record or an agent's private reflection.
        """
        by_identity: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in list(brief.get("what_is_happening") or []) + list(
                brief.get("needs_your_decision") or []):
            for source in item.get("refs") or []:
                key = (str(source.get("kind", "")), str(source.get("id", "")))
                if key[0] and key[1]:
                    by_identity.setdefault(key, {"source": source, "item": item})
        for rows in (view.get("objects") or {}).values():
            for obj in rows or []:
                source = {"kind": str(obj.get("kind") or "object"),
                          "id": str(obj.get("id") or ""),
                          "title": str(obj.get("title") or "Visible item")}
                key = (source["kind"], source["id"])
                if key[1]:
                    why = "This item is currently visible to the human seat."
                    if source["kind"] == "meeting":
                        agenda = "; ".join(str(row) for row in (obj.get("agenda") or []) if row)
                        participants = ", ".join(self._meeting_names(view, obj))
                        why = (f"Agenda: {agenda or 'no public agenda recorded'}. "
                               f"Participants: {participants or 'none listed'}. "
                               f"Victor RSVP: {'attend' if obj.get('attended') else 'skip' if obj.get('declined') else 'pending' }.")
                    by_identity.setdefault(key, {"source": source, "item": {
                        "title": source["title"], "what": source["title"],
                        "status": str(obj.get("status") or "visible"), "why": why,
                    }})
        # Directed-message summaries cite the ordinary P2-visible message, but
        # P3 never returns the raw thread collection.  Indexing it here makes
        # the normal evidence -> trace disclosure path available on demand.
        for message in self._visible_message_rows(token, view):
            message_id = str(message.get("id") or "")
            if not message_id:
                continue
            actor = self._member_name(view, message.get("sender"))
            source = {"kind": "message", "id": message_id,
                      "title": f"Message from {actor}"}
            by_identity.setdefault(("message", message_id), {
                "source": source,
                "item": {
                    "title": source["title"],
                    "what": str(message.get("text") or "Visible message"),
                    "status": str(message.get("urgency") or
                                  message.get("importance") or "normal"),
                    "why": ("This message was visible to Victor's ordinary P2 seat and "
                            "was routed through the secretary's configured attention policy."),
                },
            })
        for event in self._visible_event_rows(token, view):
            projected = self._event_source(event, view)
            if projected is None:
                continue
            source, item = projected
            resource_identity = self._event_resource_identity(event, view)
            if resource_identity is not None:
                source = dict(source)
                source["resource_kind"], source["resource_id"] = resource_identity
            by_identity.setdefault((source["kind"], source["id"]),
                                   {"source": source, "item": item,
                                    "event_record": self._p2_event_record(event)})
        return {_opaque_ref(token, row["source"]): row for row in by_identity.values()}

    @staticmethod
    def _p2_event_record(event: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize the complete event already admitted by the P2 feed.

        The P2 feed is the visibility boundary.  The small forbidden-key guard
        is defence in depth for a malformed legacy event, not a second event
        classifier.  JSON normalization mirrors the eventual HTTP payload and
        avoids exposing a mutable reference to the runtime journal.
        """
        permitted = {key: value for key, value in event.items()
                     if key not in FORBIDDEN_IN_SEAT_VIEW}
        if str(event.get("type") or "") == "reflection_event":
            # P2 admits the existence/timing of a visible reflection cycle, not
            # its private record identity, prose, memory links, or tool trace.
            permitted = {key: permitted[key] for key in (
                "type", "subtype", "agent_id", "tick", "status",
            ) if key in permitted}
        return json.loads(json.dumps(permitted, default=str))

    @staticmethod
    def _event_related_current_objects(event: Dict[str, Any], view: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return current P2 rows explicitly named by an already visible event."""
        named = {
            str(event.get(key) or "")
            for key in (
                "object_id", "task_id", "doc_id", "proposal_id", "protocol_id",
                "experiment_id", "result_id", "meeting_id", "pr_id",
                "pull_request_id", "candidate_id", "release_id", "issue_id",
                "message_id", "branch_id",
            )
        }
        named.discard("")
        candidate_id = str(event.get("candidate_id") or "")
        tick = event.get("tick")
        if candidate_id and isinstance(tick, int):
            # A readiness event can be paired with same-tick blocker events
            # that name the actual visible task(s).  This is an explicit P2
            # event relation, not a title-based world search.
            for companion in ((view.get("feed") or {}).get("events") or []):
                if (str(companion.get("candidate_id") or "") != candidate_id
                        or companion.get("tick") != tick):
                    continue
                named.update(str(companion.get(key) or "") for key in (
                    "object_id", "task_id", "doc_id", "proposal_id", "protocol_id",
                    "experiment_id", "result_id", "meeting_id", "pr_id",
                    "pull_request_id", "candidate_id", "release_id", "issue_id",
                    "message_id", "branch_id",
                ))
            named.discard("")
        if not named:
            return []
        related: List[Dict[str, Any]] = []
        seen = set()
        for rows in (view.get("objects") or {}).values():
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                identity = str(row.get("id") or "")
                if identity not in named or identity in seen:
                    continue
                # These are full, already-shaped P2 rows rather than a raw
                # domain object lookup.  JSON normalization provides an
                # immutable response-safe copy.
                related.append(json.loads(json.dumps(row, default=str)))
                seen.add(identity)
        return related

    @staticmethod
    def _event_resource_identity(event: Dict[str, Any], view: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Find a current, already-visible object named by a filtered event.

        Events themselves remain lightweight historical evidence.  An ``Open
        source`` request may only target a current parent object which is still
        in the P2 seat view; it never turns an event id into a world lookup.
        """
        parameter_kinds = (
            ("pr_id", "pull_request"), ("pull_request_id", "pull_request"),
            ("task_id", "task"),
            ("doc_id", "document"), ("proposal_id", "proposal"),
            ("protocol_id", "protocol"), ("experiment_id", "experiment"),
            ("result_id", "result"), ("meeting_id", "meeting"),
            ("candidate_id", "release_candidate"), ("release_id", "release"),
            ("issue_id", "issue"), ("message_id", "message"),
            ("branch_id", "branch"),
        )
        visible = [obj for rows in (view.get("objects") or {}).values()
                   for obj in (rows or []) if isinstance(obj, dict)]
        for field, kind in parameter_kinds:
            value = str(event.get(field) or "")
            if any(str(obj.get("id") or "") == value and
                   str(obj.get("kind") or "") == kind for obj in visible):
                return kind, value
        generic_id = str(event.get("object_id") or "")
        if generic_id:
            matches = [obj for obj in visible if str(obj.get("id") or "") == generic_id]
            if len(matches) == 1:
                return str(matches[0].get("kind") or "object"), generic_id
        candidate_id = str(event.get("candidate_id") or "")
        tick = event.get("tick")
        if candidate_id and isinstance(tick, int):
            for related in ((view.get("feed") or {}).get("events") or []):
                if (str(related.get("candidate_id") or "") != candidate_id
                        or related.get("tick") != tick):
                    continue
                for field, kind in parameter_kinds:
                    value = str(related.get(field) or "")
                    if any(str(obj.get("id") or "") == value and
                           str(obj.get("kind") or "") == kind for obj in visible):
                        return kind, value
        return None

    def _resource_index(self, token: str, view: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Build token-bound resource handles from the *current* P2 view only."""
        resources: Dict[Tuple[str, str], Dict[str, str]] = {}
        for rows in (view.get("objects") or {}).values():
            for obj in rows or []:
                if not isinstance(obj, dict):
                    continue
                kind = str(obj.get("kind") or "")
                resource_id = str(obj.get("id") or "")
                if kind and resource_id:
                    resources.setdefault((kind, resource_id), {
                        "kind": kind, "id": resource_id,
                        "title": str(obj.get("title") or resource_id),
                    })
        # Event cards can open a real parent resource, but only while the
        # parent remains visible.  Historical event access stays in trace().
        for event in self._visible_event_rows(token, view):
            identity = self._event_resource_identity(event, view)
            if identity is None:
                continue
            kind, resource_id = identity
            resources.setdefault((kind, resource_id), {
                "kind": kind, "id": resource_id, "title": resource_id,
            })
        return {_resource_ref(token, row["kind"], row["id"]): row
                for row in resources.values()}

    def _resource_ref_for_source(self, token: str, source: Dict[str, Any],
                                 view: Dict[str, Any]) -> str:
        """Associate an object evidence source with its same visibility-bound resource."""
        kind = str(source.get("resource_kind") or source.get("kind") or "")
        resource_id = str(source.get("resource_id") or source.get("id") or "")
        for ref, row in self._resource_index(token, view).items():
            if row["kind"] == kind and row["id"] == resource_id:
                return ref
        return ""

    def _resources_for_evidence(
            self, token: str, view: Dict[str, Any],
            evidence_records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Resolve exact read evidence to resources already visible to this seat."""
        index = self._resource_index(token, view)
        by_object_id = {row["id"]: {"ref": ref, "object_id": row["id"],
                                    "kind": row["kind"], "title": row["title"]}
                        for ref, row in index.items()}
        matched_ids = []
        for evidence in evidence_records:
            if evidence.get("tool") == "read_object":
                object_id = str((evidence.get("args") or {}).get("object_id") or "")
                if object_id in by_object_id and object_id not in matched_ids:
                    matched_ids.append(object_id)

        repo_paths = {
            str((evidence.get("args") or {}).get("path") or "").replace("\\", "/")
            for evidence in evidence_records
            if evidence.get("tool") == "read_repo"
        }
        repo_paths.discard("")
        if repo_paths:
            runtime = self.human.runtime()
            for row in index.values():
                if row["kind"] != "task":
                    continue
                resource = runtime.seat_resource(
                    P3_FIXED_SEAT_ID, "task", row["id"], "overview")
                linked_paths = {
                    str(linked.get("path") or "").replace("\\", "/")
                    for linked in ((resource.get("content") or {}).get("linked_files") or [])
                }
                if repo_paths.intersection(linked_paths) and row["id"] not in matched_ids:
                    matched_ids.append(row["id"])
        return [by_object_id[object_id] for object_id in matched_ids]

    @staticmethod
    def _event_identity(event: Dict[str, Any]) -> str:
        """Server-only stable identity for a filtered event; never return it."""
        encoded = json.dumps(event, sort_keys=True, default=str,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    @staticmethod
    def _visible_task(view: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
        for task in ((view.get("objects") or {}).get("tasks") or []):
            if str(task.get("id") or "") == task_id:
                return task
        return None

    @staticmethod
    def _visible_object_for_event(view: Dict[str, Any], event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find an event's named object without reading outside the seat view."""
        wanted = {
            str(event.get(key) or "")
            for key in ("object_id", "proposal_id", "meeting_id", "experiment_id",
                        "pr_id", "pull_request_id", "protocol_id", "decision_id",
                        "candidate_id")
        }
        wanted.discard("")
        if not wanted:
            return None
        for rows in (view.get("objects") or {}).values():
            for obj in rows or []:
                if str(obj.get("id") or "") in wanted:
                    return obj
        return None

    def _actor_name(self, view: Dict[str, Any], event: Dict[str, Any]) -> str:
        subtype = str(event.get("subtype") or "").lower()
        # Approval/rejection/change events often retain the original actor in
        # ``agent_id`` but explicitly name the person who made the decision.
        # Prefer that visible roster member when it is present.
        if any(word in subtype for word in ("approve", "reject", "change")):
            for key in ("approver", "reviewer", "changed_by"):
                named = self._roster_index(view).get(str(event.get(key) or ""), {})
                if named:
                    return str(named.get("name") or "The organization")
        actor = self._roster_index(view).get(str(event.get("agent_id") or ""), {})
        return str(actor.get("name") or "The organization")

    def _member_name(self, view: Dict[str, Any], member_id: Any) -> str:
        member = self._roster_index(view).get(str(member_id or ""), {})
        return str(member.get("name") or "an unassigned visible member")

    @staticmethod
    def _event_tick(event: Dict[str, Any]) -> Optional[int]:
        """Keep the organizational tick as metadata, never as a feed event."""
        tick = event.get("tick")
        return tick if isinstance(tick, int) and not isinstance(tick, bool) else None

    @staticmethod
    def _event_phrase(value: Any) -> str:
        return str(value or "").replace("_", " ").strip()

    def _event_item(self, event: Dict[str, Any], view: Dict[str, Any], *, kind: str,
                    title: str, status: str = "activity", why: str = "",
                    dedup: str = "", object_title: str = "", slot: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build a safe event card from fields already exposed by the P2 view."""
        event_id = self._event_identity(event)
        source = {"kind": kind, "id": event_id, "title": object_title or title}
        return source, {
            "title": title, "what": object_title or title, "status": status,
            "why": why or title, "actor": self._actor_name(view, event),
            "event_type": self._event_phrase(event.get("type")),
            "event_subtype": self._event_phrase(event.get("subtype")),
            "event_tick": self._event_tick(event),
            "_dedup_key": dedup or f"{kind}:{title}", "_event_source": True,
            "_slot_key": slot,
        }

    @staticmethod
    def _notification_priority(item: Dict[str, Any]) -> int:
        """Rank initial-summary facts by the amount of human attention they need."""
        kind = str(item.get("kind") or "")
        status = str(item.get("status") or "").lower()
        if kind == "release_readiness" and status in {"blocked", "failed", "attention"}:
            return 100
        if kind == "task_completion":
            return 95
        if kind == "proposal_event" and status in {"under_review", "attention", "approved", "rejected"}:
            return 90
        if kind in {"evidence_request", "claim_dispute_resolved"}:
            return 85
        if kind == "product_dogfood" and status not in {"ok", "passed", "success"}:
            return 85
        if kind == "release_blocker_cleared":
            return 80
        if kind == "experiment_recorded":
            return 75
        if kind == "human_related_event":
            return 78
        if kind in {"meeting_event", "reflection_cycle"}:
            # The P3 contract explicitly promises a human-facing recap after
            # meetings and reflection cycles, including on first entry.
            return 80
        if kind == "product_dogfood":
            return 60
        if kind == "protocol_activity":
            return 55
        return 40

    def _event_source(self, event: Dict[str, Any], view: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Turn a meaningful, P2-filtered event into a concise public update.

        Event ids and ticks are deliberately not identities for notification
        purposes: simulation bookkeeping often emits the same fact repeatedly.
        Unknown events are not turned into content-free chat cards.
        """
        event_type = f"{event.get('type') or ''} {event.get('subtype') or ''}".lower()
        subtype = str(event.get("subtype") or "").lower()
        if (str(event.get("type") or "").lower() == "tick" or subtype == "read_feed"
                # Funding remains part of OrgWorld's resource dynamics, but an
                # ordinary scheduled tranche is not a Victor-facing liaison
                # event. A concrete blocker or explicit human decision caused
                # by funding can still surface through those semantic objects.
                or str(event.get("type") or "").lower() == "funding_event"
                or (subtype == "wish_clustering"
                    and not (event.get("open_wishes") or event.get("themes")))
                or str(event.get("type") or "").lower() in {
                    "overtime_event", "recovery_event", "wish_event"
                }
                or (str(event.get("type") or "").lower() == "communication_event"
                    and subtype in {"surface", "reactive_ack", "deferred"})
                or ("background" in event_type and not any(event.get(key) for key in
                    ("result", "outcome", "finding", "status")))):
            return None
        actor = self._actor_name(view, event)
        if subtype == "wish_clustering":
            wishes = int(event.get("open_wishes") or 0)
            themes = int(event.get("themes") or 0)
            return self._event_item(
                event, view, kind="wish_clustering",
                title=f"The organization clustered {wishes} open wishes into {themes} themes.",
                status="activity",
                why="This visible clustering has non-empty organizational follow-up.",
                dedup=f"wish-clustering:{wishes}:{themes}")
        if subtype == "blocker_to_task":
            candidate = str(event.get("candidate_id") or "")
            tick = self._event_tick(event)
            if any(str(row.get("subtype") or "") == "readiness_check"
                   and self._event_tick(row) == tick
                   and str(row.get("candidate_id") or "") == candidate
                   for row in ((view.get("feed") or {}).get("events") or [])):
                # The readiness card carries the paired task/owner facts. Do
                # not make a one-check/four-task burst crowd it out.
                return None
        task = self._visible_task(view, str(event.get("task_id") or ""))
        if task is not None:
            # Deliberately do not reuse the task object's source key here: a
            # task notification must resolve to its event-safe provenance, not
            # silently become a richer default task trace.
            source = {"kind": "task_event", "id": self._event_identity(event),
                      "title": str(task.get("title") or "Visible task")}
            status = str(task.get("status") or "active")
            complete = status.lower() in {"merged", "released", "done"}
            owner = self._member_name(view, task.get("owner"))
            if subtype == "blocker_to_task":
                gate = self._event_phrase(event.get("gate"))
                source["kind"] = "release_blocker_task"
                return source, {
                    "title": f"{actor} created blocker task {source['title']} for {owner}; status: {status}.",
                    "what": source["title"], "status": status,
                    "why": f"This visible task addresses release gate {gate or 'a recorded blocker'} and is owned by {owner}.",
                    "actor": actor, "event_type": self._event_phrase(event.get("type")),
                    "event_subtype": self._event_phrase(event.get("subtype")),
                    "event_tick": self._event_tick(event),
                    "_dedup_key": f"blocker-task:{source['title']}:{owner}:{status}:{gate}",
                    "_event_source": True,
                }
            if complete:
                title = f"Completed: {source['title']} is {status}."
                why = f"{actor} reported the visible task owned by {owner} in terminal delivery status {status}."
            elif subtype in {"assigned", "owned", "owner_changed"}:
                title = f"{source['title']} is owned by {owner}; status: {status}."
                why = f"{actor} recorded the visible task assignment. Assignment is not evidence that work is active or complete."
            else:
                title = f"{actor} updated {source['title']}; status: {status}."
                why = f"The visible task is owned by {owner} and remains {status}; this is not completion evidence."
            return source, {"title": title, "what": source["title"], "status": status,
                            "why": why, "actor": actor, "event_type": self._event_phrase(event.get("type")),
                            "event_subtype": self._event_phrase(event.get("subtype")),
                            "event_tick": self._event_tick(event),
                            "_dedup_key": (f"completed:{source['title']}" if complete else
                                            f"task:{source['title']}:{status}:{subtype}"),
                            "_event_source": True}
        if subtype == "readiness_check":
            status = str(event.get("status") or "under_review")
            blockers = [self._event_phrase(gate) for gate in (event.get("blockers") or []) if gate]
            candidate = str(event.get("candidate_id") or "")
            tick = self._event_tick(event)
            follow_up_owners = []
            for follow_up in ((view.get("feed") or {}).get("events") or []):
                if (str(follow_up.get("subtype") or "") != "blocker_to_task"
                        or self._event_tick(follow_up) != tick
                        or str(follow_up.get("candidate_id") or "") != candidate):
                    continue
                follow_task = self._visible_task(view, str(follow_up.get("task_id") or ""))
                if follow_task is not None:
                    follow_up_owners.append(
                        f"{self._member_name(view, follow_up.get('owner_id') or follow_task.get('owner'))} — "
                        f"{follow_task.get('title') or 'visible blocker task'}")
            if status == "blocked":
                title = f"{actor}'s release readiness check is blocked by {len(blockers)} gates."
                why = ("Blocked gates: " + "; ".join(blockers)
                       if blockers else "The visible release readiness check reported a blocked status.")
            else:
                title = f"{actor} ran a release readiness check; status: {self._event_phrase(status)}."
                why = "The visible release candidate is currently " + self._event_phrase(status) + "."
            if follow_up_owners:
                follow_up_names = [entry.split(" — ", 1)[0] for entry in follow_up_owners]
                owner_counts: Dict[str, int] = {}
                for name in follow_up_names:
                    owner_counts[name] = owner_counts.get(name, 0) + 1
                title += " Follow-ups: " + ", ".join(
                    f"{name} ×{count}" for name, count in owner_counts.items()) + "."
                why += " Follow-up owners: " + "; ".join(follow_up_owners) + "."
            return self._event_item(event, view, kind="release_readiness", title=title,
                                    status=status, why=why,
                                    dedup=(f"readiness:{candidate}:{status}:{'|'.join(blockers)}:"
                                           f"{'|'.join(follow_up_owners)}"))
        if subtype == "blocker_to_task" and event.get("task_id"):
            blocker_task = self._visible_task(view, str(event.get("task_id")))
            if blocker_task is None:
                return None
            task_title = str(blocker_task.get("title") or "Visible blocker task")
            owner = self._member_name(view, event.get("owner_id") or blocker_task.get("owner"))
            gate = self._event_phrase(event.get("gate"))
            status = str(blocker_task.get("status") or "open")
            return self._event_item(
                event, view, kind="release_blocker_task",
                title=f"{actor} created blocker task {task_title} for {owner}; status: {status}.",
                status=status,
                why=f"This visible task addresses release gate {gate or 'a recorded blocker'} and is owned by {owner}.",
                dedup=f"blocker-task:{task_title}:{owner}:{status}:{gate}", object_title=task_title)
        if subtype == "blocker_to_issue":
            # The paired blocker task explains the work and owner; an issue-only
            # card is redundant and usually has no additional visible outcome.
            return None
        if subtype == "dogfood":
            outcome = self._event_phrase(event.get("outcome") or "recorded")
            query = str(event.get("query") or "the product")
            finding = str(event.get("finding") or "").strip()
            title = f"{actor} dogfooded the product for \"{query}\": {outcome}."
            why = f"Outcome: {outcome}." + (f" Finding: {finding}" if finding else "")
            return self._event_item(event, view, kind="product_dogfood", title=title,
                                    status=str(event.get("outcome") or "activity"), why=why,
                                    dedup=f"dogfood:{query}:{outcome}:{finding}")
        if str(event.get("type") or "") == "experiment_event" and subtype == "logged_to_tracker":
            return self._event_item(
                event, view, kind="experiment_recorded",
                title=f"{actor} logged an experiment result to the shared tracker.",
                status="recorded",
                why="A seat-visible experiment event records that a result entered the shared tracker; this does not by itself validate the result.",
                dedup=f"experiment-tracker:{event.get('result_id') or ''}:{event.get('doc_id') or ''}")
        if str(event.get("type") or "") == "protocol_use_event":
            protocol = self._visible_object_for_event(view, event)
            if protocol is None:
                return None
            protocol_title = str(protocol.get("title") or "a visible protocol")
            protocol_status = str(protocol.get("status") or "visible")
            use_count = protocol.get("use_count")
            count_text = f" Visible use count: {use_count}." if isinstance(use_count, int) else ""
            return self._event_item(
                event, view, kind="protocol_activity",
                title=f"{actor} used protocol {protocol_title}; registry status: {protocol_status}.",
                status=protocol_status,
                why=f"The visible protocol registry records this use.{count_text} Use is activity, not evidence that the protocol solved the problem.",
                dedup=f"protocol-use:{protocol.get('id') or protocol_title}:{use_count}",
                object_title=protocol_title,
                slot=f"protocol-use:{protocol.get('id') or protocol_title}")
        if str(event.get("type") or "") == "claim_dispute_event" and subtype == "resolved":
            resolver = self._member_name(view, event.get("resolved_by"))
            resolution = self._event_phrase(event.get("resolution") or "recorded")
            return self._event_item(
                event, view, kind="claim_dispute_resolved",
                title=f"{resolver} resolved a result-evidence dispute: {resolution}.",
                status="resolved",
                why=f"The visible dispute record names {resolver} and records the resolution as {resolution}.",
                dedup=f"claim-dispute:{event.get('dispute_id') or ''}:resolved:{resolution}")
        if str(event.get("type") or "") == "speech_act_event" and subtype in {
                "challenge_result", "ask_for_evidence", "request_reproduction"}:
            action = {
                "challenge_result": "challenged a result",
                "ask_for_evidence": "asked for evidence",
                "request_reproduction": "requested a reproduction",
            }[subtype]
            return self._event_item(
                event, view, kind="evidence_request",
                title=f"{actor} {action}; a result needs more evidence.",
                status="attention",
                why="This visible communication records an evidence challenge, not a failed or completed result.",
                dedup=f"evidence-request:{actor}:{subtype}:{self._event_tick(event)}")
        if subtype == "blocker_cleared":
            gate = self._event_phrase(event.get("gate"))
            if gate.startswith("gate "):
                gate = gate[5:]
            return self._event_item(event, view, kind="release_blocker_cleared",
                                    title=f"{actor} cleared a release blocker{f': {gate}' if gate else ''}.",
                                    status="cleared", why="The recorded release blocker is no longer active.",
                                    dedup=f"blocker-cleared:{gate}")
        obj = self._visible_object_for_event(view, event)
        object_title = str((obj or {}).get("title") or "")
        object_status = str((obj or {}).get("status") or event.get("status") or "activity")
        if "reflection" in event_type:
            tick = self._event_tick(event)
            follow_up = next((self._visible_object_for_event(view, candidate)
                              for candidate in ((view.get("feed") or {}).get("events") or [])
                              if self._event_tick(candidate) == tick
                              and "proposal" in f"{candidate.get('type') or ''} {candidate.get('subtype') or ''}".lower()), None)
            follow_title = str((follow_up or {}).get("title") or "")
            suffix = f"; public follow-up: {follow_title}." if follow_title else "."
            return self._event_item(event, view, kind="reflection_cycle",
                                    title=f"{actor} completed a reflection cycle{suffix}", status="activity",
                                    why=("Private reflection content is not exposed."
                                         + (f" A visible follow-up proposal is {follow_title}." if follow_title else "")),
                                    dedup=f"reflection:{actor}:{follow_title}", object_title=follow_title)
        if "meeting" in event_type:
            meeting_action = ("automatically convened" if subtype == "auto_convened"
                              else self._event_phrase(subtype) or "updated")
            return self._event_item(event, view, kind="meeting_event",
                                    title=f"{actor} {meeting_action} meeting {object_title or 'a visible meeting'}; status: {object_status}.",
                                    status=object_status,
                                    why="The visible meeting changed organizational state; it is not completion evidence.",
                                    dedup=f"meeting:{object_title}:{subtype}:{object_status}", object_title=object_title)
        family = str(event.get("type") or "").lower()
        if any(word in event_type for word in ("proposal", "protocol", "experiment", "decision", "repo", "pull_request", "pr_", "ci")):
            noun = ("pull request" if (family in {"repo_event", "pull_request_event", "pr_event"}
                                        or subtype.startswith("pr_") or subtype == "ci") else
                    "proposal" if "proposal" in event_type else
                    "protocol" if "protocol" in event_type else
                    "experiment" if "experiment" in event_type else "decision")
            result = self._event_phrase(subtype or event.get("status") or "updated")
            object_slot = next((str(event.get(key) or "") for key in (
                "proposal_id", "protocol_id", "experiment_id", "decision_id",
                "pr_id", "pull_request_id", "object_id") if event.get(key)), object_title)
            if not object_title and noun in {"proposal", "protocol", "experiment", "decision"}:
                return None
            if noun == "proposal" and subtype == "approved" and object_status != "approved":
                title = f"{actor} recorded an approval on proposal {object_title}; it remains {self._event_phrase(object_status)}."
            elif noun == "pull request":
                pr_label = object_title or object_slot
                author_id = str((obj or {}).get("author") or "")
                author = self._member_name(view, author_id) if author_id else ""
                possessed = (f"{author}'s pull request" if author else "pull request")
                named_pr = f"{possessed}{f' {pr_label}' if pr_label else ''}"
                semantic_status = {
                    "public_tests_failed": "failed",
                    "ci_failed": "failed",
                    "pr_reviewed": "reviewed",
                    "review_pr": "reviewed",
                    "pr_approved": "approved",
                    "approved": "approved",
                    "pr_merged": "merged",
                    "merged": "merged",
                    "ci_verdict_retired": "retired",
                    "request_changes": "changes_requested",
                    "changes_requested": "changes_requested",
                    "pr_opened": "opened",
                    "opened": "opened",
                }.get(subtype, subtype or "updated")
                if subtype in {"public_tests_failed", "ci_failed"}:
                    title = f"Public tests failed for {named_pr}."
                elif subtype in {"pr_reviewed", "review_pr"}:
                    title = (f"{actor} reviewed {named_pr}." if actor != "The organization"
                             else f"The organization reviewed {named_pr}.")
                elif subtype in {"pr_approved", "approved"}:
                    title = f"{actor} approved {named_pr}."
                elif subtype in {"pr_merged", "merged"}:
                    title = f"{actor} merged {named_pr}."
                elif subtype == "ci_verdict_retired":
                    merged_pr = str(event.get("merged_pr_id") or "")
                    suffix = f" after {merged_pr} merged" if merged_pr else ""
                    title = f"The CI verdict for {named_pr} was retired{suffix}."
                else:
                    title = f"{actor} recorded {result} for {named_pr}."
                role_facts = []
                if author:
                    role_facts.append(f"PR author: {author}")
                if actor != "The organization":
                    role_facts.append(f"event actor: {actor}")
                current = str((obj or {}).get("status") or "")
                if current:
                    role_facts.append(f"current PR status: {self._event_phrase(current)}")
                why = ("This is the immutable seat-visible repo event. "
                       + ("; ".join(role_facts) + ". " if role_facts else "")
                       + "Review participation is not PR authorship, and this event alone is not task completion evidence.")
                return self._event_item(
                    event, view, kind="pull_request_event", title=title,
                    status=semantic_status, why=why,
                    dedup=f"pull-request-event:{self._event_identity(event)}",
                    object_title=pr_label,
                )
            else:
                title = f"{actor} {result} {noun} {object_title or 'visible work'}; status: {object_status}."
            return self._event_item(event, view, kind=f"{noun.replace(' ', '_')}_event",
                                    title=title,
                                    status=object_status,
                                    why=f"The visible {noun} is {object_status}; this is not task completion evidence.",
                                    dedup=f"{noun}:{object_slot}:{object_status}", object_title=object_title,
                                    slot=f"{noun}:{object_slot}")
        seat_id = self._seat_id(view)
        human_related = (
            str(event.get("agent_id") or "") == seat_id
            or seat_id in (event.get("participants") or [])
            or any(str(event.get(key) or "") == seat_id for key in (
                "owner_id", "assignee_id", "target_agent", "reviewer", "approver",
            ))
        )
        if human_related:
            action = self._event_phrase(event.get("action_type") or subtype)
            if not action:
                return None
            status = self._event_phrase(event.get("status") or
                                        event.get("outcome") or "recorded")
            subject = object_title or self._event_phrase(
                event.get("title") or event.get("task_title") or "Victor's organization work")
            detail_pairs = []
            for label, key in (("result", "result"), ("reason", "reason"),
                               ("failure", "failure_reason"), ("finding", "finding")):
                value = self._event_phrase(event.get(key))
                if value:
                    detail_pairs.append(f"{label}: {value}")
            detail = "; ".join(detail_pairs)
            if str(event.get("agent_id") or "") == seat_id:
                title = f"Your {action} was recorded for {subject}; status: {status}."
            else:
                title = f"{actor} {action} work involving Victor; status: {status}."
            why = ("This update explicitly names Victor as actor, participant, owner, assignee, "
                   "target, reviewer, or approver in the already filtered seat-visible event."
                   + (f" {detail}." if detail else ""))
            return self._event_item(
                event, view, kind="human_related_event", title=title, status=status,
                why=why, dedup=f"human:{action}:{subject}:{status}:{detail}",
                object_title=subject,
            )
        # A generic event with no safe actor/action/object/result is useful only
        # in the deeper P2 inspection view, never as a P3 conversation card.
        return None

    def _notifications(self, token: str, view: Dict[str, Any]) -> List[Dict[str, Any]]:
        history = self._notification_history.setdefault(token, [])
        is_initial_snapshot = token not in self._notification_seen
        seen = self._notification_seen.setdefault(token, {})
        candidates: Dict[str, Dict[str, Any]] = {}
        candidate_slots: Dict[str, str] = {}
        for event in self._visible_event_rows(token, view):
            projected = self._event_source(event, view)
            if projected is None:
                continue
            source, item = projected
            complete = (source["kind"] == "task_event"
                        and str(item["status"]).lower() in {"merged", "released", "done"})
            # A terminal delivery is a fact about one task, not a fresh
            # completion every time a later event mentions that task.
            key = (f"completed:{source['title']}" if complete
                   else str(item.get("_dedup_key") or f"{source['kind']}:{source['id']}"))
            slot = str(item.get("_slot_key") or "")
            if slot and slot in candidate_slots:
                # A snapshot may retain created -> review -> approved for one
                # visible object. P3 reports the latest state, not its entire
                # internal state-transition history.
                candidates.pop(candidate_slots[slot], None)
            ref = _opaque_ref(token, source)
            resource_identity = self._event_resource_identity(event, view)
            resource_ref = (_resource_ref(token, resource_identity[0], resource_identity[1])
                            if resource_identity is not None else "")
            # Pop/move makes repeated terminal events retain the latest
            # provenance while still yielding only one completion notice.
            candidates.pop(key, None)
            candidates[key] = {"_key": key, "_source": source,
                               "id": f"nt_{ref[3:]}", "kind": source["kind"],
                               "summary": item["title"], "status": item["status"],
                               "evidence_refs": [ref], "at": time.time(),
                               "tick": item.get("event_tick"),
                               # A notification must retain the substance which
                               # justified surfacing it.  These are generated
                               # only from a P2-filtered event projection, not
                               # from a generic "organization updated" shell.
                               "what": str(item.get("what") or item["title"]),
                               "why": str(item.get("why") or item["title"]),
                               "actor": str(item.get("actor") or "The organization"),
                               **({"resource_ref": resource_ref} if resource_ref else {}),
                               "_priority": self._notification_priority({
                                   "kind": source["kind"], "status": item["status"],
                               })}
            if slot:
                candidate_slots[slot] = key

        # Establishing the liaison should not replay an entire organization
        # feed as a chat transcript.  Record the baseline as seen, then give
        # the human only a compact latest-summary window. Later polls append
        # exclusively unseen keys, even after the 60-row history is trimmed.
        if is_initial_snapshot:
            ranked = sorted(enumerate(candidates.values()),
                            key=lambda pair: (pair[1]["_priority"], pair[0]),
                            reverse=True)[:8]
            additions = [row for _, row in sorted(ranked, key=lambda pair: pair[0])]
        else:
            additions = [row for key, row in candidates.items() if key not in seen]
        history.extend(additions)
        for key in candidates:
            seen[key] = None
        while len(seen) > 512:
            del seen[next(iter(seen))]
        # Bound the local secretary record.  It never writes to the world and
        # can never become a hidden event archive for another seat.
        del history[:-60]
        return [{key: value for key, value in row.items() if not key.startswith("_")}
                for row in history]

    # -- P3 attention mediation -------------------------------------------
    @staticmethod
    def _seat_id(view: Dict[str, Any]) -> str:
        return str((view.get("seat") or {}).get("agent_id") or P3_FIXED_SEAT_ID)

    def _visible_event_rows(self, token: str,
                            view: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Merge the current P2 surface with its bounded HCI delivery journal."""
        rows: Dict[str, Dict[str, Any]] = {}
        getter = getattr(self.human, "liaison_visible_events", None)
        try:
            journal_seat = self.human._seat(token)
        except (SeatUnavailable, RuntimeError):
            journal_seat = None
        if callable(getter) and journal_seat is not None:
            for event in getter(self._seat_id(view)):
                rows[self._event_identity(event)] = event
        for event in ((view.get("feed") or {}).get("events") or []):
            rows[self._event_identity(event)] = event
        return list(rows.values())

    def _visible_message_rows(self, token: str,
                              view: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: Dict[str, Dict[str, Any]] = {}
        getter = getattr(self.human, "liaison_visible_messages", None)
        try:
            journal_seat = self.human._seat(token)
        except (SeatUnavailable, RuntimeError):
            journal_seat = None
        if callable(getter) and journal_seat is not None:
            for message in getter(self._seat_id(view)):
                identity = str(message.get("id") or self._event_identity(message))
                rows[identity] = message
        for message in self._flat_messages(view):
            identity = str(message.get("id") or self._event_identity(message))
            rows[identity] = message
        return list(rows.values())

    @staticmethod
    def _flat_messages(view: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return each P2-visible message once, without exposing the feed."""
        rows: Dict[str, Dict[str, Any]] = {}
        for thread in ((view.get("feed") or {}).get("threads") or []):
            for message in thread.get("messages") or []:
                message_id = str(message.get("id") or "")
                if message_id:
                    rows[message_id] = message
        return list(rows.values())

    @staticmethod
    def _directed_message(message: Dict[str, Any], seat_id: str) -> bool:
        """A real direct/mentioned message, not ordinary shared-channel traffic."""
        if bool(message.get("mentions_me")) or seat_id in (message.get("mentions") or []):
            return True
        # ``dm_id`` is reserved by the communication model even though most
        # current scenarios use mentions.  Recipients alone do not imply a DM:
        # channel sends list every channel member as a recipient.
        return bool(message.get("dm_id") and seat_id in (message.get("recipients") or []))

    @staticmethod
    def _important_message(message: Dict[str, Any]) -> bool:
        return (str(message.get("urgency") or "").lower() in {
                    "high", "urgent", "incident"
                } or str(message.get("importance") or "").lower() in {
                    "decision_relevant", "blocker", "policy_relevant"
                })

    def _message_updates(self, token: str, view: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Route messages through the secretary's explicit delivery policy.

        The default is not a raw inbox: only directed, high-signal messages are
        delivered.  ``all_messages`` deliberately mirrors every P2-visible
        message through the secretary thread, but still cannot read a channel
        or DM that Victor's seat could not read.
        """
        history = self._message_history.setdefault(token, [])
        seen = self._message_seen.setdefault(token, {})
        mode = self._message_delivery.setdefault(token, "secretary_triage")
        seat_id = self._seat_id(view)
        candidates: List[Dict[str, Any]] = []
        for message in self._visible_message_rows(token, view):
            message_id = str(message.get("id") or "")
            if not message_id or message_id in seen or str(message.get("sender") or "") == seat_id:
                continue
            directed = self._directed_message(message, seat_id)
            if mode == "secretary_triage" and not (directed and self._important_message(message)):
                # Do not mark held messages seen. If Victor later asks to see
                # everything, the messages still present in the P2 working
                # surface can be delivered at that point.
                continue
            actor = self._member_name(view, message.get("sender"))
            importance = str(message.get("importance") or "useful")
            urgency = str(message.get("urgency") or "normal")
            text = str(message.get("text") or "(no message text)")
            source = {"kind": "message", "id": message_id,
                      "title": f"Message from {actor}"}
            prefix = (f"{actor} sent you an important message"
                      if mode == "secretary_triage" else
                      f"Message from {actor} in {message.get('channel') or 'a visible channel'}")
            candidates.append({
                "id": f"msg_{_opaque_ref(token, source)[3:]}",
                "role": "liaison", "kind": "incoming_message",
                "threadable": True,
                "text": f"{prefix}: {text}", "summary": f"{prefix}: {text}",
                "status": urgency if urgency != "normal" else importance,
                "importance": importance, "urgency": urgency,
                "delivery_mode": mode,
                "details": (f"Secretary assessment: importance={importance}; urgency={urgency}. "
                            "The source remains the ordinary message visible to Victor."),
                "evidence_refs": [_opaque_ref(token, source)],
                "at": message.get("at") or time.time(),
            })
            seen[message_id] = None
        # Avoid turning a newly opened P3 window into a replay of a large
        # all-message inbox. The remainder remains available in P2/on request.
        limit = 30 if mode == "all_messages" else 8
        history.extend(candidates[-limit:])
        del history[:-100]
        while len(seen) > 1024:
            del seen[next(iter(seen))]
        return list(history)

    @staticmethod
    def _meeting_names(view: Dict[str, Any], meeting: Dict[str, Any]) -> List[str]:
        roster = LiaisonFacade._roster_index(view)
        return [str(roster.get(str(pid), {}).get("name") or pid)
                for pid in (meeting.get("participants") or [])]

    @staticmethod
    def _meeting_public_outcome(meeting: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize only outcome fields already allowlisted by the P2 view."""
        outcome = dict(meeting.get("outcome") or {})
        notes = str(outcome.get("notes_summary") or meeting.get("notes_summary") or "")
        decisions = list(outcome.get("decisions") or meeting.get("decision_summaries") or [])
        actions = list(outcome.get("action_items") or meeting.get("action_items") or [])
        unresolved = list(outcome.get("unresolved_questions") or
                          meeting.get("unresolved_questions") or [])
        return {"notes_summary": notes, "decisions": decisions,
                "action_items": actions, "unresolved_questions": unresolved}

    @staticmethod
    def _meeting_outcome_text(outcome: Dict[str, Any]) -> str:
        parts: List[str] = []
        if outcome.get("notes_summary"):
            parts.append(f"Public notes: {outcome['notes_summary']}")
        decisions = [str(row.get("summary") if isinstance(row, dict) else row)
                     for row in outcome.get("decisions") or []]
        decisions = [row for row in decisions if row]
        if decisions:
            parts.append("Decisions: " + "; ".join(decisions))
        actions = []
        for row in outcome.get("action_items") or []:
            if isinstance(row, dict):
                detail = str(row.get("description") or "follow-up")
                if row.get("assignee"):
                    detail += f" — {row['assignee']}"
                actions.append(detail)
            elif row:
                actions.append(str(row))
        if actions:
            parts.append("Follow-ups: " + "; ".join(actions))
        unresolved = [str(row) for row in outcome.get("unresolved_questions") or [] if row]
        if unresolved:
            parts.append("Open questions: " + "; ".join(unresolved))
        return " ".join(parts) or "The visible meeting record contains no notes or decisions."

    def _meeting_updates(self, token: str, view: Dict[str, Any]) -> Tuple[List[Dict[str, Any]],
                                                                          List[Dict[str, Any]]]:
        """Create one pre-start decision turn and one post-meeting report."""
        plans = self._meeting_plans.setdefault(token, {})
        history = self._meeting_history.setdefault(token, [])
        seat_id = self._seat_id(view)
        active_cards: List[Dict[str, Any]] = []
        meetings = ((view.get("objects") or {}).get("meetings") or [])
        for meeting in meetings:
            meeting_id = str(meeting.get("id") or "")
            if not meeting_id or seat_id not in (meeting.get("participants") or []):
                continue
            status = str(meeting.get("status") or "").lower()
            plan = plans.get(meeting_id)
            if plan is None and status in {"scheduled", "active"}:
                title = str(meeting.get("title") or meeting.get("meeting_type") or "Meeting")
                agenda = [str(item) for item in (meeting.get("agenda") or []) if item]
                names = self._meeting_names(view, meeting)
                source = {"kind": "meeting", "id": meeting_id, "title": title}
                resource_ref = _resource_ref(token, "meeting", meeting_id)
                plan = {
                    "meeting_id": meeting_id, "title": title, "decision": "pending",
                    "viewpoint": "", "return_focus": "", "draft_ids": [],
                    "viewpoint_draft_id": "", "attendance_draft_id": "",
                    "viewpoint_confirmed": False, "attendance_confirmed": False,
                    "report_delivered": False, "created_at": time.time(),
                    "evidence_ref": _opaque_ref(token, source),
                    "resource_ref": resource_ref,
                }
                # The browser receives a token-bound opaque plan handle, never
                # the meeting id or the component draft ids used internally.
                plan["plan_id"] = plan["evidence_ref"]
                plans[meeting_id] = plan
                agenda_text = "; ".join(agenda) or "No public agenda was recorded."
                people_text = ", ".join(names) or "No participants were listed."
                history.append({
                    "id": f"meeting_msg_{plan['evidence_ref'][3:]}",
                    "role": "liaison", "kind": "meeting_invitation",
                    "threadable": True,
                    "status": "decision_required", "at": plan["created_at"],
                    "text": (f"Before \"{title}\" starts, I need your attendance decision. "
                             f"Agenda: {agenda_text} Participants: {people_text}. "
                             "You can say that you will attend; say you will not attend and "
                             "tell me what exact viewpoint to relay, what information to bring "
                             "back, either one, or neither; or ask me about the context first."),
                    "summary": f"Attendance decision needed before {title} starts.",
                    "details": ("The meeting is held in scheduled state while Victor's claimed "
                                "human seat has no confirmed attend/skip response. Other "
                                "organizational work continues."),
                    "meeting": {"title": title, "agenda": agenda, "participants": names,
                                "status": status},
                    "evidence_refs": [plan["evidence_ref"]],
                    "resource_ref": resource_ref,
                })
            if plan is None:
                continue

            # Reconcile confirmed world state in case RSVP was submitted from
            # P1/P2 rather than this P3 conversation.
            if meeting.get("attended"):
                plan["attendance_confirmed"] = True
                if plan["decision"] == "pending":
                    plan["decision"] = "attend"
            if meeting.get("declined"):
                plan["attendance_confirmed"] = True
                if plan["decision"] == "pending":
                    plan["decision"] = "skip"

            if status in {"scheduled", "active"}:
                active_cards.append({
                    "plan_id": plan["plan_id"],
                    "title": plan["title"], "status": status,
                    "decision": plan["decision"],
                    "viewpoint": plan["viewpoint"],
                    "return_focus": plan["return_focus"],
                    "attendance_confirmed": bool(plan["attendance_confirmed"]),
                    "confirmation_required": bool(
                        plan["decision"] != "pending" and not plan["attendance_confirmed"]),
                    "evidence_refs": [plan["evidence_ref"]],
                    "resource_ref": plan["resource_ref"],
                })

            if status in {"completed", "closed", "cancelled"} and not plan["report_delivered"]:
                outcome = self._meeting_public_outcome(meeting)
                detail = self._meeting_outcome_text(outcome)
                return_focus = str(plan.get("return_focus") or "")
                if plan["decision"] == "skip" and not return_focus:
                    report = (f"\"{plan['title']}\" ended. As requested, I did not prepare a "
                              "detailed return brief.")
                else:
                    requested = (f" You asked me to bring back: {return_focus}."
                                 if return_focus else "")
                    relayed = (" Your confirmed viewpoint was relayed before the meeting."
                               if plan.get("viewpoint_confirmed") else
                               " No viewpoint was relayed on your behalf.")
                    report = f"\"{plan['title']}\" ended.{requested}{relayed} {detail}"
                history.append({
                    "id": f"meeting_report_{plan['evidence_ref'][3:]}",
                    "role": "liaison", "kind": "meeting_report", "status": status,
                    "threadable": True,
                    "at": time.time(), "text": report, "summary": report,
                    "details": detail, "requested_return": return_focus,
                    "evidence_refs": [plan["evidence_ref"]],
                    "resource_ref": plan["resource_ref"],
                })
                plan["report_delivered"] = True
        del history[:-80]
        return list(history), active_cards

    @staticmethod
    def _decision_details(decision_type: str, item: Dict[str, Any],
                          view: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Enrich a decision at the P3 boundary without widening the brief API.

        ``build_organization_brief`` is a stable, compact seat-view contract.
        P3 needs a richer card, so resolve the exact already-visible awaiting
        object here, where the liaison can also mint its resource handle.
        """
        awaiting = ((view.get("member") or {}).get("awaiting_me") or {})
        bucket = {
            "proposal_approval": "proposals",
            "pull_request_review": "reviews",
            "meeting_attendance": "meetings",
        }.get(decision_type, "")
        item_id = str(item.get("id") or "")
        source = next((row for row in awaiting.get(bucket) or []
                       if str(row.get("id") or "") == item_id), {})
        if decision_type == "proposal_approval":
            return (
                "Your response determines whether this proposal can be adopted, revised, or rejected.",
                {"proposal_type": source.get("proposal_type"),
                 "proposer": source.get("author"),
                 "required_approvers": source.get("approvers"),
                 "approved_by": source.get("approved_by"),
                 "current_status": source.get("status")},
            )
        if decision_type == "pull_request_review":
            return (
                "Your recorded review can approve this change or return it for revision; merge and release remain separate gated actions.",
                {"author": source.get("author"),
                 "reviewers": source.get("reviewers"),
                 "ci_passed": source.get("ci_passed"),
                 "current_status": source.get("status")},
            )
        if decision_type == "meeting_attendance":
            return (
                "Your RSVP determines whether Victor attends. Leaving it pending holds only this meeting while other organization work continues.",
                {"agenda": source.get("agenda"),
                 "participants": source.get("participants"),
                 "scheduled_tick": source.get("scheduled_tick"),
                 "current_status": source.get("status")},
            )
        return ("", {})

    def _item(self, token: str, item: Dict[str, Any],
              view: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """P3 default workstream: no raw object/member/feed fields or ids."""
        status = str(item.get("status") or "attention")
        title = str(item.get("title") or item.get("what") or "Visible workstream")
        projected = {
            "title": title,
            "status": status,
            "summary": f"{title} is currently {status.replace('_', ' ')}.",
            "evidence_refs": [_opaque_ref(token, source)
                              for source in item.get("refs") or []],
        }
        resource_refs = []
        if view is None:
            # Internal callers normally pass the snapshot they are already
            # projecting.  The fallback remains seat-filtered for legacy
            # callers, never a world lookup.
            _seat, view = self._authenticated_view(token)
        for source in item.get("refs") or []:
            resource_ref = self._resource_ref_for_source(token, source, view)
            if resource_ref and resource_ref not in resource_refs:
                resource_refs.append(resource_ref)
        if resource_refs:
            projected["resource_refs"] = resource_refs
            # A primary ref keeps the thin client simple; plural refs preserve
            # a decision's full visible provenance when it has several inputs.
            projected["resource_ref"] = resource_refs[0]
        decision_type = str(item.get("decision_type") or "")
        if decision_type:
            default_impact, default_context = self._decision_details(
                decision_type, item, view)
            projected.update({
                "id": f"decision_{projected['evidence_refs'][0]}",
                "kind": str(item.get("kind") or "decision"),
                "what": str(item.get("what") or f"Decide: {title}"),
                "why": str(item.get("why") or "Your judgment is required."),
                "impact": str(item.get("impact") or default_impact),
                "context": dict(item.get("context") or default_context),
                "decision_type": decision_type,
                "preferred_resource_tab": (
                    "review" if decision_type in {
                        "proposal_approval", "pull_request_review"} else "summary"),
                "requires_human_decision": True,
                "decision_options": self._decision_options(decision_type, title),
            })
            # ``options`` is the direct decision-card contract; retain the
            # older name for existing clients while making the semantic field
            # explicit for new P3 cards.
            projected["options"] = list(projected["decision_options"])
            # A decision inbox row must say what the human is deciding, not
            # collapse a governance obligation into "currently attention".
            projected["summary"] = projected["what"]
        return projected

    def _validated_decision_context(self, token: str, view: Dict[str, Any],
                                    submitted: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Resolve one opaque card selection against the current decision inbox."""
        if not submitted:
            return [], {}
        if not isinstance(submitted, dict):
            raise ValueError("invalid_decision_reference")
        resource_ref = str(submitted.get("resource_ref") or "")
        resource = self._resource_index(token, view).get(resource_ref)
        if resource is None:
            raise ValueError("stale_or_invisible_decision_reference")
        brief = build_organization_brief(view)
        item = next((row for row in brief.get("needs_your_decision") or []
                     if str(row.get("id") or "") == str(resource.get("id") or "")), None)
        if item is None:
            raise ValueError("decision_is_no_longer_pending")
        decision_type = str(item.get("decision_type") or "")
        option_id = str(submitted.get("option_id") or "")
        options = self._decision_options(decision_type, str(item.get("title") or "Decision"))
        option = next((row for row in options if str(row.get("id") or "") == option_id), None)
        if option_id and option is None:
            raise ValueError("invalid_decision_option")
        public = {
            "resource_ref": resource_ref,
            "title": str(item.get("title") or resource.get("title") or "Decision"),
            "kind": str(item.get("kind") or resource.get("kind") or "decision"),
            "decision_type": decision_type,
            "option_id": option_id,
            "option_label": str((option or {}).get("label") or ""),
        }
        return [{"object_id": str(resource["id"])}], public

    @staticmethod
    def _decision_options(decision_type: str, title: str) -> List[Dict[str, str]]:
        """Natural-language composer starters, never direct action endpoints."""
        quoted = f'“{title}”'
        context = {
            "id": "context", "label": "Ask for context",
            "instruction": (f"Explain {quoted}, show me the recorded evidence and tradeoffs, "
                            "and tell me exactly what decision is required. Do not take action yet."),
        }
        if decision_type == "proposal_approval":
            return [
                context,
                {"id": "approve", "label": "Draft approval",
                 "instruction": (f"I want to approve the proposal {quoted}. Prepare the exact "
                                 "approval action for my confirmation.")},
                {"id": "changes", "label": "Request changes",
                 "instruction": f"I want to request changes to the proposal {quoted}: "},
                {"id": "reject", "label": "Draft rejection",
                 "instruction": (f"I want to reject the proposal {quoted}. Prepare the exact "
                                 "rejection action for my confirmation.")},
            ]
        if decision_type == "pull_request_review":
            return [
                {"id": "review", "label": "Open review evidence",
                 "instruction": (f"Review {quoted}. Show me the actual diff, review comments, "
                                 "and CI evidence before I decide. Do not approve or merge yet.")},
                {"id": "approve", "label": "Draft approval",
                 "instruction": (f"I want to approve {quoted} after checking its current review "
                                 "and CI evidence. Prepare the review action for my confirmation.")},
                {"id": "changes", "label": "Request changes",
                 "instruction": f"I want to request changes on {quoted}: "},
            ]
        if decision_type == "meeting_attendance":
            return [
                context,
                {"id": "attend", "label": "Attend",
                 "instruction": f"I will attend {quoted}. Prepare that attendance decision for confirmation."},
                {"id": "delegate", "label": "Delegate or skip",
                 "instruction": (f"I will not attend {quoted}. Ask what viewpoint, if any, you should relay "
                                 "and what information, if any, you should bring back before preparing the decision.")},
            ]
        return [context]

    @staticmethod
    def _roster_index(view: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        members = (view.get("member") or {}).get("members") or []
        return {str(member.get("agent_id") or ""): member for member in members}

    def _expanded_item(self, token: str, item: Dict[str, Any],
                       view: Dict[str, Any]) -> Dict[str, Any]:
        out = self._item(token, item, view)
        owner = self._roster_index(view).get(str(item.get("owner") or ""))
        if owner:
            out["owner"] = {"name": str(owner.get("name") or ""),
                            "role": str(owner.get("role") or "member")}
        out["why"] = str(item.get("why") or "Visible organizational state.")
        return out

    def _project(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        happening = list(brief.get("what_is_happening") or [])
        decisions = list(brief.get("needs_your_decision") or [])
        blocked = [item for item in happening
                   if str(item.get("status") or "") in {"blocked", "failed", "changes_requested"}]
        status = "blocked" if blocked else "needs_attention" if decisions else "active"
        organization = brief.get("organization") or {}
        return {
            "title": str(organization.get("product_name") or
                         organization.get("company_name") or "Organization"),
            "status": status,
            "summary": str(brief.get("summary") or "No public summary is available."),
            "milestone": str(organization.get("product_stage") or ""),
        }

    def _organization(self, token: str, brief: Dict[str, Any],
                      view: Dict[str, Any]) -> Dict[str, Any]:
        roster = (view.get("member") or {}).get("members") or []
        event_rows: Dict[str, Dict[str, Any]] = {}
        event_slots: Dict[str, str] = {}
        for event in ((view.get("feed") or {}).get("events") or []):
            projected = self._event_source(event, view)
            if projected is None:
                continue
            source, item = projected
            key = str(item.get("_dedup_key") or f"{source['kind']}:{source['id']}")
            slot = str(item.get("_slot_key") or "")
            if slot and slot in event_slots:
                event_rows.pop(event_slots[slot], None)
            event_rows.pop(key, None)
            ref = _opaque_ref(token, source)
            resource_identity = self._event_resource_identity(event, view)
            resource_ref = (_resource_ref(token, resource_identity[0], resource_identity[1])
                            if resource_identity is not None else "")
            event_rows[key] = {
                "type": source["kind"], "actor": str(item.get("actor") or "organization"),
                "summary": item["title"], "status": item["status"],
                "tick": item.get("event_tick"),
                "what": str(item.get("what") or item["title"]),
                "why": str(item.get("why") or item["title"]),
                "evidence_refs": [ref],
                **({"resource_ref": resource_ref} if resource_ref else {}),
            }
            if slot:
                event_slots[slot] = key
        events = list(event_rows.values())[-12:]
        return {
            "summary": str(brief.get("summary") or "No public summary is available."),
            "workstreams": [self._expanded_item(token, item, view)
                            for item in brief.get("what_is_happening") or []],
            "decision_inbox": [self._expanded_item(token, item, view)
                               for item in brief.get("needs_your_decision") or []],
            "agents": [{"name": str(member.get("name") or ""),
                        "role": str(member.get("role") or "member"),
                        "online": bool(member.get("online", True))}
                       for member in roster],
            "recent_activity": events,
        }

    @staticmethod
    def _conversation(agent_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Hide tool output and all draft parameters from P3's default state."""
        messages = []
        for message in agent_state.get("messages") or []:
            role = message.get("role")
            identity = {
                "id": str(message.get("message_id") or ""),
                "thread_id": str(message.get("thread_id") or ""),
                "parent_id": str(message.get("parent_id") or ""),
                "threadable": True,
            }
            if role == "human":
                public = {"role": "human", "text": str(message.get("text") or ""),
                          "at": message.get("at"), **identity}
                if message.get("decision_context"):
                    public["decision_context"] = dict(message["decision_context"])
                messages.append(public)
            elif role == "agent":
                text = str(message.get("text") or "")
                # Draft lifecycle chatter belongs to the P2 working-agent
                # transcript.  P3 renders one semantic confirmation card and
                # must not repeat implementation details such as action names.
                if (text.startswith("Prepared: ") or text.startswith("Sent ")
                        or text.startswith("That was refused:")):
                    continue
                public = {"role": "liaison", "text": str(message.get("text") or ""),
                          "at": message.get("at"),
                          "references": list(message.get("references") or []),
                          **identity}
                kind = str(message.get("kind") or "")
                if kind in {"clarification", "grounded_answer", "model_error", "meeting_instruction", "attention_preference",
                             "execution_plan", "execution_summary", "execution_progress", "task_review", "resource_opened",
                            "action_interpretation", "action_result"}:
                    public["kind"] = kind
                if kind == "clarification":
                    public["clarification"] = public["text"]
                messages.append(public)
        return messages

    def _register_thread_roots(self, token: str, view: Dict[str, Any], worker: Any,
                               *card_groups: List[Dict[str, Any]]) -> None:
        """Make every public card a thread root with its current visible object anchors."""
        resource_index = self._resource_index(token, view)
        for cards in card_groups:
            for card in cards:
                message_id = str(card.get("id") or "")
                text = str(card.get("text") or card.get("summary") or "")
                if message_id and text:
                    handles = [card.get("resource_ref")]
                    handles.extend(card.get("resource_refs") or [])
                    references = []
                    for handle in handles:
                        ref = str(handle.get("ref") if isinstance(handle, dict) else handle or "")
                        resource = resource_index.get(ref)
                        if resource and not any(row["object_id"] == resource["id"]
                                                for row in references):
                            references.append({"object_id": resource["id"],
                                               "kind": resource["kind"]})
                    worker.register_context_root(
                        message_id, text, kind=str(card.get("kind") or "event"),
                        at=float(card.get("at") or time.time()),
                        references=references,
                    )

    @staticmethod
    def _display_action_params(view: Dict[str, Any],
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve structure ids to the human-readable symbols already visible."""
        labels: Dict[str, str] = {}
        for member in ((view.get("member") or {}).get("members") or []):
            member_id = str(member.get("agent_id") or "")
            if member_id:
                labels[member_id] = str(member.get("name") or member_id)
        for channel in ((view.get("feed") or {}).get("channels") or []):
            channel_id = str(channel.get("id") or "")
            if channel_id:
                labels[channel_id] = str(channel.get("name") or channel.get("title")
                                         or channel_id)
        for rows in (view.get("objects") or {}).values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                object_id = str(row.get("id") or "")
                if object_id:
                    labels[object_id] = str(row.get("title") or row.get("name") or object_id)
        display: Dict[str, Any] = {}
        for key, value in params.items():
            display_key = key[:-3] if key.endswith("_id") else key
            if isinstance(value, str) and value in labels:
                display[display_key] = f"{labels[value]} ({value})"
            else:
                display[display_key] = value
        return display

    def _pending_actions(self, token: str, agent_state: Dict[str, Any],
                         view: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the exact normalized action a human is being asked to confirm."""
        meeting_drafts = {
            draft_id
            for plan in self._meeting_plans.get(token, {}).values()
            for draft_id in (plan.get("draft_ids") or [])
        }
        return [{"draft_id": str(draft.get("draft_id") or ""),
                 "label": str(draft.get("label") or draft.get("action_type") or ""),
                 "action_type": str(draft.get("action_type") or ""),
                 "thread_id": str(draft.get("thread_id") or ""),
                 "params": dict(draft.get("params") or {}),
                 "display_params": self._display_action_params(
                     view, dict(draft.get("params") or {})),
                 "rationale": str(draft.get("rationale") or "")}
                for draft in agent_state.get("drafts") or []
                if str(draft.get("draft_id") or "") not in meeting_drafts]

    def state(self, token: str, since: int = 0) -> Dict[str, Any]:
        """The default P3 payload: organization façade plus liaison dialogue."""
        try:
            seat, view, brief = self._brief(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        workstreams = [self._item(token, item, view)
                       for item in brief.get("what_is_happening") or []]
        blockers = [item for item in workstreams
                    if item.get("status") in {"blocked", "failed", "changes_requested"}]
        notifications = self._notifications(token, view)
        meeting_updates, meeting_cards = self._meeting_updates(token, view)
        message_updates = self._message_updates(token, view)
        if self._meeting_plans.get(token):
            # Meeting invitations/reports are the human-facing source of truth;
            # do not also emit the old generic "meeting changed state" card.
            notifications = [row for row in notifications
                             if row.get("kind") != "meeting_event"]
        worker = self.human._agent(seat.agent_id)
        decision_cards = [self._item(token, item, view)
                          for item in brief.get("needs_your_decision") or []]
        self._register_thread_roots(
            token, view, worker, notifications, meeting_updates, message_updates,
            decision_cards)
        # HumanApi appends only the authenticated seat's real execution jobs.
        # Public cards above are registered only as conversation roots; they
        # never become organization objects or synthetic execution activity.
        agent_state = self.human.agent_state(token, since)
        conversation = self._conversation(agent_state)
        resource_by_object_id = {
            row["id"]: {"ref": ref, "kind": row["kind"], "title": row["title"]}
            for ref, row in self._resource_index(token, view).items()
        }
        raw_messages = {str(message.get("message_id") or ""): message
                        for message in list(agent_state.get("messages") or [])}
        for message in conversation:
            resources = [resource_by_object_id[str(reference.get("object_id") or "")]
                         for reference in list(message.get("references") or [])
                         if str(reference.get("object_id") or "") in resource_by_object_id]
            if not resources:
                raw_message = raw_messages.get(str(message.get("id") or "")) or {}
                resources = self._resources_for_evidence(
                    token, view, list(raw_message.get("evidence_records") or []))
            if resources:
                message["resource_ref"] = resources[0]["ref"]
                message["resource_refs"] = [
                    {key: resource[key] for key in ("ref", "kind", "title")}
                    for resource in resources
                ]
        focuses = self._resource_focus.get(token, {})
        for message in conversation:
            focus = focuses.get(str(message.get("id") or ""))
            if focus:
                message["resource_focus"] = dict(focus)
        # Completion/activity feedback is a safe, durable liaison utterance.
        # It is never a raw world event nor an agent reflection.
        conversation.extend({"id": notice["id"], "threadable": True,
                             "role": "liaison", "text": notice["summary"],
                             "kind": "event", "event_type": notice["kind"],
                              "status": notice["status"], "at": notice["at"],
                              "summary": notice["summary"],
                              "evidence_refs": notice["evidence_refs"],
                              "tick": notice.get("tick"),
                              "what": notice.get("what"),
                              "why": notice.get("why"),
                              "actor": notice.get("actor"),
                              **({"resource_ref": notice["resource_ref"]}
                                 if notice.get("resource_ref") else {})}
                            for notice in notifications)
        conversation.extend(meeting_updates)
        conversation.extend(message_updates)
        conversation.sort(key=lambda row: float(row.get("at") or 0))
        failed_requests = list(agent_state.get("failed_requests") or [])
        failed_requests.extend({
            key: row[key] for key in (
                "thread_id", "original_request", "error", "step", "max_steps",
                "next_step", "source_count", "sources",
            )
        } for row in self._failed_routes.get(token, {}).values())
        execution_jobs = []
        for source_job in list(agent_state.get("execution_jobs") or []):
            job = dict(source_job)
            resources = [resource_by_object_id[str(reference.get("object_id") or "")]
                         for reference in list(job.get("references") or [])
                         if str(reference.get("object_id") or "") in resource_by_object_id]
            if resources:
                job["resource_ref"] = resources[0]["ref"]
                job["resource_refs"] = resources
            execution_jobs.append(job)
        return {
            "mode": "organization_as_a_service",
            "project_summary": self._project(brief),
            "workstreams": workstreams,
            "blocker": blockers[0] if blockers else None,
            "decision_inbox": decision_cards,
            "organization_available": True,
            "conversation": conversation,
            "conversation_threads": list(agent_state.get("threads") or []),
            "active_task_focus": dict(agent_state.get("active_task_focus") or {}),
            "pending_actions": self._pending_actions(token, agent_state, view),
            # A delegated RSVP is one human decision even when it compiles to
            # multiple ordinary P2 actions.  Keep those mechanics server-side.
            "meeting_plans": [card for card in meeting_cards
                              if card.get("confirmation_required")],
            "notifications": notifications,
            "meeting_invitations": meeting_cards,
            "execution_jobs": execution_jobs,
            "execution_agents": list(agent_state.get("execution_agents") or []),
            "working": {
                **dict(agent_state.get("working") or {}),
                "queued_requests": int(agent_state.get("queue_depth") or 0),
            },
            "failed_request": failed_requests[-1] if failed_requests else None,
            "failed_requests": failed_requests,
            "last_model_error": agent_state.get("error"),
            "attention_preferences": {
                "message_delivery": self._message_delivery.setdefault(
                    token, "secretary_triage"),
                "description": ("The secretary screens directed messages and immediately "
                                "delivers high-signal items. Ask to see all messages at any time."
                                if self._message_delivery[token] == "secretary_triage" else
                                "Every message visible to Victor's seat is relayed through the secretary."),
            },
            "busy": bool(agent_state.get("busy")),
        }

    def organization(self, token: str) -> Dict[str, Any]:
        """A read-only façade-only organization update, without dialogue."""
        try:
            _seat, view, brief = self._brief(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        return {"organization": self._organization(token, brief, view)}

    def _resolve(self, token: str, ref: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        try:
            _seat, view, brief = self._brief(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return None, {"error": str(exc)}
        row = self._source_index(token, brief, view).get(ref or "")
        if row is None:
            return None, {"error": "unknown_evidence_ref"}
        row["view"] = view
        return row, None

    def evidence(self, token: str, ref: str) -> Dict[str, Any]:
        """Resolve one opaque handle to its already-cited P2 explanation."""
        row, error = self._resolve(token, ref)
        if error:
            return error
        item = row["item"]
        source = row["source"]
        evidence = {
            "ref": ref,
            "title": str(item.get("title") or item.get("what") or "Visible source"),
            "status": str(item.get("status") or "attention"),
            "why": str(item.get("why") or "Visible organizational state."),
            "source": {"kind": str(source.get("kind") or "object"),
                       "title": str(source.get("title") or item.get("title") or "Visible source")},
        }
        event_record = row.get("event_record")
        if item.get("_event_source") and isinstance(event_record, dict):
            # This is the complete event record which the P2 feed already
            # admitted for this seat, not a summary reconstructed by the
            # secretary.  Current related objects remain separate because an
            # event can outlive or cease to name a visible source.
            evidence["event"] = event_record
            evidence["event_summary"] = {
                "actor": str(item.get("actor") or "The organization"),
                "action": str(item.get("event_subtype") or item.get("event_type") or "recorded activity"),
                "organization_tick": item.get("event_tick"),
            }
            evidence["related_current_visible_objects"] = self._event_related_current_objects(
                event_record, row["view"])
        resource_ref = self._resource_ref_for_source(token, source, row["view"])
        if resource_ref:
            evidence["resource_ref"] = resource_ref
        return {"disclosure_level": "evidence", "evidence": evidence}

    def resource(self, token: str, ref: str, section: str = "overview") -> Dict[str, Any]:
        """Open one token-bound, current-seat-visible read-only resource.

        The opaque ref is deliberately the only browser input.  The resource
        seam owns all raw-domain serialization; this façade neither reads the
        world nor offers a fallback object when that seam is unavailable.
        """
        try:
            seat, view = self._authenticated_view(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        row = self._resource_index(token, view).get(str(ref or ""))
        if row is None:
            # Do not distinguish a guessed id, an old/released token, and an
            # object which just left this seat's visibility boundary.
            return {"error": "unknown_resource_ref"}
        requested_section = str(section or "overview")
        try:
            runtime = self.human.runtime()
            resolver = getattr(runtime, "seat_resource", None)
            if not callable(resolver):
                return {"error": "resource_adapter_unavailable"}
            result = resolver(seat.agent_id, row["kind"], row["id"], requested_section)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        except Exception:  # The adapter must not leak its underlying object ids/errors.
            return {"error": "resource_unavailable"}
        if not isinstance(result, dict):
            return {"error": "resource_unavailable"}
        if result.get("error"):
            # A race between index construction and the short-lock adapter call
            # is a visibility failure, not a detail the browser may diagnose.
            return {"error": "unknown_resource_ref"}
        execution_jobs = []
        secretary_reports = []
        secretary_evidence = []
        agent_state = self.human.agent_state(token)
        if not agent_state.get("error"):
            object_id = str(row.get("id") or "")
            execution_jobs = [
                job for job in list(agent_state.get("execution_jobs") or [])
                if any(str(reference.get("object_id") or "") == object_id
                       for reference in list(job.get("references") or []))
            ]
            for message in list(agent_state.get("messages") or []):
                if message.get("role") != "agent" or not message.get("evidence_records"):
                    continue
                message_resources = self._resources_for_evidence(
                    token, view, list(message.get("evidence_records") or []))
                linked = (any(str(reference.get("object_id") or "") == object_id
                              for reference in list(message.get("references") or []))
                          or any(resource["object_id"] == object_id
                                 for resource in message_resources))
                if linked:
                    secretary_reports.append({
                        "message_id": str(message.get("message_id") or ""),
                        "kind": str(message.get("kind") or ""),
                        "text": str(message.get("text") or ""),
                        "evidence_records": list(message.get("evidence_records") or []),
                    })
            secretary_evidence = [
                {**evidence, "message_id": report["message_id"]}
                for report in secretary_reports
                for evidence in report["evidence_records"]
            ]
        return {
            "resource_ref": str(ref), "resource_kind": row["kind"],
            "resource_section": requested_section,
            "execution_context": {"jobs": execution_jobs},
            "secretary_context": {
                "reports": secretary_reports,
                "evidence_records": secretary_evidence,
            },
            **result,
        }

    @staticmethod
    def _visible_object(view: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """Allowlist a currently visible P2 object's trace-safe fields."""
        wanted_id = str(source.get("id") or "")
        safe_fields = (
            "id", "kind", "title", "status", "owner", "author", "reviewers",
            "approved_by", "ci_passed", "progress", "priority", "proposal_type",
            "protocol_type", "emergence", "decision_type", "age_hours",
            "meeting_type", "attended", "declined", "agenda", "notes_summary",
            "decision_summaries", "unresolved_questions", "action_items",
        )
        for rows in (view.get("objects") or {}).values():
            for obj in rows or []:
                if str(obj.get("id") or "") == wanted_id:
                    return {key: obj.get(key) for key in safe_fields if key in obj}
        return {"id": wanted_id, "kind": str(source.get("kind") or "object"),
                "title": str(source.get("title") or wanted_id)}

    def trace(self, token: str, ref: str) -> Dict[str, Any]:
        """A deliberately small provenance trace; it cannot become an event dump."""
        row, error = self._resolve(token, ref)
        if error:
            return error
        item = row["item"]
        source = row["source"]
        view = row["view"]
        if item.get("_event_source"):
            # ``event_record`` was already admitted by the P2 feed.  The trace
            # adds the current P2-visible objects it explicitly names so the
            # user can distinguish what happened then from what is true now.
            event_record = row.get("event_record") if isinstance(
                row.get("event_record"), dict) else {}
            related = self._event_related_current_objects(event_record, view)
            steps: List[Dict[str, Any]] = [
                {"kind": "visible_event",
                 "summary": str(item.get("title") or "Visible organization activity."),
                 "actor": str(item.get("actor") or "The organization"),
                 "organization_tick": item.get("event_tick"),
                 "event": event_record},
                {"kind": "visible_evidence",
                 "summary": str(item.get("why") or "Visible organization activity."),
                 "status": str(item.get("status") or "activity")},
            ]
            for obj in related:
                steps.append({
                    "kind": "organizational_object",
                    "summary": (f"{obj.get('kind', 'object')} "
                                f"{obj.get('title') or obj.get('id')} is currently "
                                f"{obj.get('status') or 'visible'}."),
                    "object": obj,
                })
            result = {"disclosure_level": "organizational_trace", "trace": {
                "ref": ref,
                "source": {"kind": str(source.get("kind") or "organization_event"),
                           "title": str(source.get("title") or "Visible update")},
                "steps": steps,
            }}
            resource_ref = self._resource_ref_for_source(token, source, view)
            if resource_ref:
                result["resource_ref"] = resource_ref
            return result
        obj = self._visible_object(view, source)
        roster = self._roster_index(view)
        owner_id = str(obj.get("owner") or obj.get("author") or "")
        if owner_id in roster:
            obj["responsible_member"] = {"name": str(roster[owner_id].get("name") or ""),
                                         "role": str(roster[owner_id].get("role") or "member")}
            obj.pop("owner", None)
            obj.pop("author", None)
        steps: List[Dict[str, Any]] = [
            {"kind": "summary", "summary": str(item.get("what") or item.get("title") or "Visible work")},
            {"kind": "evidence", "summary": str(item.get("why") or "Visible organizational state.")},
            {"kind": "organizational_object", "summary": (
                f"{obj.get('kind', 'object')} {obj.get('title') or obj.get('id')} "
                f"is {obj.get('status') or 'visible'}."), "object": obj},
        ]
        wanted_id = str(source.get("id") or "")
        for message in self._visible_message_rows(token, view):
            attached = {str(att.get("object_id") or "")
                        for att in message.get("attachments") or []}
            if str(message.get("id") or "") == wanted_id or wanted_id in attached:
                actor = roster.get(str(message.get("sender") or ""), {})
                steps.append({"kind": "visible_message",
                              "summary": str(message.get("text") or ""),
                              "actor": str(actor.get("name") or "organization")})
        result = {"disclosure_level": "organizational_trace", "trace": {
            "ref": ref,
            "source": {"kind": str(source.get("kind") or "object"),
                       "id": str(source.get("id") or ""),
                       "title": str(source.get("title") or "")},
            "steps": steps,
        }}
        resource_ref = self._resource_ref_for_source(token, source, view)
        if resource_ref:
            result["resource_ref"] = resource_ref
        return result

    # -- liaison interaction: natural-language attention plans + P2 gate ---
    def _route_human_instruction(self, token: str, text: str,
                                 view: Dict[str, Any], *,
                                 reply_to: str = "", thread_id: str = "",
                                 exclude_message_id: str = "",
                                 decision_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ask the model what the human means before any P3-local handler runs.

        This is deliberately semantic model routing. No keyword/regex intent
        detector gets to intercept a review request merely because it contains
        a phrase such as "you go" or "report back".
        """
        worker = self.human._agent(self._seat_id(view))
        if worker.llm is None:
            return {"error": "liaison_model_required_for_human_instruction"}
        self._meeting_updates(token, view)
        meeting_rows = {
            str(row.get("id") or ""): row
            for row in ((view.get("objects") or {}).get("meetings") or [])
            if row.get("id")
        }
        active_meetings = []
        for plan in self._meeting_plans.get(token, {}).values():
            if plan.get("attendance_confirmed") or plan.get("report_delivered"):
                continue
            meeting = meeting_rows.get(str(plan.get("meeting_id") or ""), {})
            active_meetings.append({
                "meeting_id": str(plan.get("meeting_id") or ""),
                "title": str(plan.get("title") or meeting.get("title") or "Meeting"),
                "status": str(meeting.get("status") or plan.get("status") or "scheduled"),
                "agenda": list(meeting.get("agenda") or []),
                "participants": self._meeting_names(view, meeting),
                "current_decision": str(plan.get("decision") or "pending"),
            })
        context = worker.conversation_context(
            reply_to=reply_to, thread_id=thread_id, limit=12,
            exclude_message_id=exclude_message_id)
        if context.get("error"):
            return {"error": str(context["error"])}
        resolved_thread = str(context.get("thread_id") or "")
        system = (
            "You are the semantic router for Victor's organization secretary. Every human "
            "utterance comes here before any local handler. Classify its meaning, not its "
            "words. `ordinary` covers all code, task, review, testing, approval, proposal, "
            "merge, release, reporting, delegation, or other work requests; it will be passed "
            "to the complete P1/P2 action compiler. Delegating a choice (including a short "
            "follow-up like '你选一个就行') is ordinary: retain the preceding work request "
            "and let the compiler choose a current eligible task and arrange execution. "
            "Do not ask the human to name a task after they delegated that choice. "
            "`grounded_answer` is only a read-only "
            "question that can be completely answered from project_context, runtime_context, "
            "team_progress, recent_activity, recent_secretary_updates, and the active "
            "conversation root supplied below. For grounded_answer, write the "
            "actual useful answer in reply, including exact task/PR ids, owners, statuses and "
            "event actor roles; never return an acknowledgement or ask the worker to look again. "
            "`team_progress.unassigned_tasks` contains the exact visible ID, title, status, and "
            "priority for every currently unassigned non-terminal task; never replace those rows "
            "with only a count or claim that their details are unavailable. "
            "When that answer enumerates visible objects the human may refer to later, include "
            "references in the same order, using exact object_id values from team_progress, "
            "recent_activity, or the visible conversation context. Never invent an ID; the "
            "server validates every reference against Victor's current seat view. "
            "`submitted_decision_context`, when present, is the server-validated decision card "
            "and option that Victor attached to this exact utterance. Treat it as an exact "
            "referent and explicit choice, route consequential options through `ordinary`, and "
            "never rematch its title to a different object or action family. "
            "A phrase such as 'you "
            "go inspect it and report back' is ordinary work, not a meeting response. "
            "`attention_preference` is only an explicit request to change whether the human "
            "personally sees every directed message or lets the secretary triage them. "
            "`meeting_context` is only a question about one active meeting. `meeting_plan` "
            "is only an RSVP or an instruction for the secretary to relay Victor's exact "
            "viewpoint and/or return specified meeting information. Never invent a meeting, "
            "viewpoint, return focus, or message preference. Use an exact visible meeting_id. "
            "`open_resource` means the human wants original currently-visible material behind "
            "a PR, task, document, result, review, file, or other resource. Select exact "
            "resource_kind and resource_id from resource_candidates plus one section. Never "
            "invent an id or use this route to take an action; ambiguous targets require "
            "clarification. "
            "PR ownership comes only from the PR object's author. A reviewer or event actor is "
            "not the PR author. A historical event and a current object status are separate "
            "facts; never claim an event was false merely because no active object exists now. "
            "If a meeting instruction lacks a required human choice, use `clarification` and "
            "ask only for that choice. For `meeting_plan`, decision=attend means Victor goes; "
            "skip means Victor does not go and requests neither relay nor return brief; "
            "delegate means Victor does not go and provides viewpoint and/or return_focus. "
            "Write `reply` in the human's language. Return only the requested JSON object."
        )
        user = json.dumps({
            "human_instruction": text,
            "submitted_decision_context": dict(decision_context or {}),
            "project_context": {
                key: value for key, value in
                dict(((view.get("member") or {}).get("company") or {})).items()
                if key in {"product_name", "company_name", "product_stage"}
            },
            "runtime_context": self.human.runtime_context(),
            "active_meetings": active_meetings,
            "message_delivery": self._message_delivery.setdefault(
                token, "secretary_triage"),
            "conversation_scope": resolved_thread or "main",
            "recent_dialogue": list(context.get("messages") or []),
            "team_progress": build_team_progress_snapshot(view),
            "recent_activity": build_recent_activity_snapshot(view),
            "resource_candidates": list(self._resource_index(token, view).values()),
            "recent_secretary_updates": [
                {"id": str(row.get("id") or ""),
                 "kind": str(row.get("kind") or "event"),
                 "summary": str(row.get("summary") or row.get("text") or ""),
                 "status": str(row.get("status") or ""),
                 "tick": row.get("tick")}
                for row in (
                    list(self._notification_history.get(token, []))
                    + list(self._meeting_history.get(token, []))
                    + list(self._message_history.get(token, []))
                )[-20:]
            ],
        }, ensure_ascii=False, default=str)
        last_error: Exception = RuntimeError("liaison_intent_model_failed")
        for _attempt in range(LIAISON_MODEL_ATTEMPTS):
            try:
                decision = worker.llm.generate_json(
                    system, user, _LIAISON_ROUTE_SCHEMA,
                    temperature=0.0, max_tokens=700)
                if not isinstance(decision, dict):
                    raise ValueError("liaison_intent_model_returned_non_object")
                route = str(decision.get("route") or "")
                if route not in {
                    "ordinary", "grounded_answer", "attention_preference", "meeting_context",
                    "meeting_plan", "open_resource", "clarification",
                }:
                    raise ValueError("liaison_intent_model_returned_invalid_route")
                return dict(decision)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        return {"error": f"liaison_intent_model_failed:{last_error!r}"}

    def _record_local_exchange(self, seat_id: str, human_text: str,
                               liaison_text: str, *, kind: str = "",
                               reply_to: str = "", thread_id: str = "",
                               human_message_id: str = "",
                               references: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        worker = self.human._agent(seat_id)
        if human_message_id:
            return worker.record_reply_to_human(
                human_message_id, liaison_text, kind=kind,
                references=references or [])
        return worker.record_exchange(
            human_text, liaison_text, kind=kind,
            reply_to=reply_to, thread_id=thread_id,
            references=references or [])

    def _select_meeting_plan(self, token: str, meeting_id: str,
                             view: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        self._meeting_updates(token, view)  # initialize invitations even if ask precedes poll
        active = [plan for plan in self._meeting_plans.get(token, {}).values()
                  if not plan.get("attendance_confirmed") and not plan.get("report_delivered")]
        exact = [plan for plan in active
                 if str(plan.get("meeting_id") or "") == str(meeting_id or "")]
        if len(exact) == 1:
            return exact[0], ""
        if len(active) == 1:
            return active[0], ""
        if not active:
            return None, "There is no visible meeting awaiting your attendance decision."
        names = ", ".join(str(plan.get("title") or "Meeting") for plan in active)
        return None, f"More than one meeting is waiting. Please name one: {names}."

    def _meeting_context_reply(self, plan: Dict[str, Any], view: Dict[str, Any]) -> str:
        meeting = next((row for row in ((view.get("objects") or {}).get("meetings") or [])
                        if str(row.get("id") or "") == plan["meeting_id"]), {})
        agenda = "; ".join(str(row) for row in (meeting.get("agenda") or []) if row)
        people = ", ".join(self._meeting_names(view, meeting))
        return (f"\"{plan['title']}\" is currently {meeting.get('status') or 'scheduled'}. "
                f"Agenda: {agenda or 'no public agenda was recorded'}. "
                f"Participants: {people or 'none listed'}. The organization will keep this "
                "meeting scheduled until Victor's claimed seat confirms attend or skip; other "
                "work continues. You may keep asking questions before deciding.")

    def _handle_model_routed_instruction(
            self, token: str, text: str, view: Dict[str, Any],
            routing: Dict[str, Any], *, reply_to: str = "",
            thread_id: str = "", human_message_id: str = "") -> Dict[str, Any]:
        """Apply only the P3-local meaning selected by the model."""
        seat_id = self._seat_id(view)
        route = str(routing.get("route") or "")
        model_reply = str(routing.get("reply") or "").strip()
        if route == "grounded_answer":
            if not model_reply:
                return {"error": "liaison_intent_model_omitted_grounded_reply"}
            recorded = self._record_local_exchange(
                seat_id, text, model_reply, kind="grounded_answer",
                reply_to=reply_to, thread_id=thread_id,
                human_message_id=human_message_id,
                references=routing.get("references") or [])
            return {**recorded, "handled": "grounded_answer"}
        if route == "clarification":
            reply = model_reply or "Please add the missing context; nothing has been routed."
            recorded = self._record_local_exchange(
                seat_id, text, reply, kind="clarification",
                reply_to=reply_to, thread_id=thread_id,
                human_message_id=human_message_id)
            return {**recorded, "handled": "meeting_clarification"}

        if route == "attention_preference":
            preference = str(routing.get("message_delivery") or "")
            if preference not in {"all_messages", "secretary_triage"}:
                return {"error": "liaison_intent_model_omitted_message_delivery"}
            self._message_delivery[token] = preference
            reply = model_reply or (
                "I will relay every message visible to Victor's seat."
                if preference == "all_messages" else
                "I will screen directed messages and immediately relay high-signal items.")
            recorded = self._record_local_exchange(
                seat_id, text, reply, kind="attention_preference",
                reply_to=reply_to, thread_id=thread_id,
                human_message_id=human_message_id)
            if recorded.get("error"):
                return recorded
            self.human._log_hci_event("liaison_attention_preference", {
                "agent_id": seat_id, "message_delivery": preference,
            })
            return {**recorded, "handled": "attention_preference"}

        if route == "open_resource":
            resource_kind = str(routing.get("resource_kind") or "")
            resource_id = str(routing.get("resource_id") or "")
            section = str(routing.get("resource_section") or "overview")
            allowed_sections = {
                "overview", "review", "ci", "commits", "patches", "files", "diff",
                "provenance",
            }
            resource_ref = next((ref for ref, row in self._resource_index(token, view).items()
                                 if row["kind"] == resource_kind and row["id"] == resource_id), "")
            if not resource_ref or section not in allowed_sections:
                if not resource_kind or not resource_id:
                    reply = "I could not open a source because the requested resource identity was missing."
                elif section not in allowed_sections:
                    reply = "I could not open that source because the requested section is not supported."
                else:
                    reply = "I could not open that source because it is not currently visible to this seat."
                recorded = self._record_local_exchange(
                    seat_id, text, reply,
                    kind="clarification", reply_to=reply_to, thread_id=thread_id,
                    human_message_id=human_message_id)
                return {**recorded, "handled": "resource_clarification"}
            opened = self.resource(token, resource_ref, section)
            if opened.get("error"):
                # A resolver error may be current-seat revalidation or an
                # adapter load failure; neither permits an opening claim.
                recorded = self._record_local_exchange(
                    seat_id, text, "The source could not be loaded; nothing was opened.",
                    kind="clarification", reply_to=reply_to, thread_id=thread_id,
                    human_message_id=human_message_id)
                return {**recorded, "handled": "resource_clarification"}
            reply = model_reply or "I opened the currently visible source in the read-only inspector."
            recorded = self._record_local_exchange(
                seat_id, text, reply, kind="resource_opened",
                reply_to=reply_to, thread_id=thread_id,
                human_message_id=human_message_id,
                references=[{"object_id": resource_id, "kind": resource_kind}])
            if recorded.get("error"):
                return recorded
            focus = {"ref": resource_ref, "kind": resource_kind, "section": section}
            self._resource_focus.setdefault(token, {})[str(recorded["message_id"])] = focus
            self.human._log_hci_event("liaison_resource_opened", {
                "agent_id": seat_id, "resource_kind": resource_kind, "section": section,
            })
            return {**recorded, "handled": "open_resource", "resource_focus": focus}

        if route not in {"meeting_context", "meeting_plan"}:
            return {"error": f"unsupported_liaison_route:{route}"}
        plan, error = self._select_meeting_plan(
            token, str(routing.get("meeting_id") or ""), view)
        if plan is None:
            recorded = self._record_local_exchange(
                seat_id, text, model_reply or error, kind="clarification",
                reply_to=reply_to, thread_id=thread_id,
                human_message_id=human_message_id)
            return {**recorded, "handled": "meeting_clarification"}
        if route == "meeting_context":
            reply = model_reply or self._meeting_context_reply(plan, view)
            recorded = self._record_local_exchange(
                seat_id, text, reply, kind="clarification",
                reply_to=reply_to, thread_id=thread_id,
                human_message_id=human_message_id)
            return {**recorded, "handled": "meeting_context"}

        worker = self.human._agent(seat_id)
        decision = str(routing.get("decision") or "")
        viewpoint = str(routing.get("viewpoint") or "").strip()
        return_focus = str(routing.get("return_focus") or "").strip()
        if decision not in {"attend", "skip", "delegate"}:
            return {"error": "liaison_intent_model_omitted_meeting_decision"}
        if decision == "delegate" and not (viewpoint or return_focus):
            reply = model_reply or (
                "What exact viewpoint should I relay, what should I bring back, "
                "or should I do neither?")
            recorded = self._record_local_exchange(
                seat_id, text, reply, kind="clarification",
                reply_to=reply_to, thread_id=thread_id,
                human_message_id=human_message_id)
            return {**recorded, "handled": "meeting_clarification"}
        reply = model_reply or "I prepared that meeting plan for your confirmation."
        recorded = self._record_local_exchange(
            seat_id, text, reply, kind="meeting_instruction",
            reply_to=reply_to, thread_id=thread_id,
            human_message_id=human_message_id)
        if recorded.get("error"):
            return recorded
        resolved_thread = str(recorded.get("thread_id") or "")
        plan.update({"decision": decision, "viewpoint": viewpoint,
                     "return_focus": return_focus, "interpreted_at": time.time()})
        drafts = []
        meeting = next((row for row in ((view.get("objects") or {}).get("meetings") or [])
                        if str(row.get("id") or "") == plan["meeting_id"]), {})
        if decision == "attend":
            draft = worker.add_draft(
                "attend_meeting", {"meeting_id": plan["meeting_id"]},
                rationale=("Confirm Victor's attendance through the same meeting action and "
                           "current-state gateway used by P2."),
                thread_id=resolved_thread)
            if draft:
                plan["attendance_draft_id"] = draft.draft_id
                drafts.append(draft.draft_id)
        else:
            if viewpoint:
                room = str(meeting.get("room_channel") or f"meeting_{plan['meeting_id']}")
                draft = worker.add_draft(
                    "send_message", {"channel_id": room, "text": viewpoint,
                                     "importance": "decision_relevant"},
                    rationale=("Relay only the exact viewpoint supplied by Victor into the "
                               "ordinary visible meeting room. The secretary is not a member."),
                    thread_id=resolved_thread)
                if draft:
                    plan["viewpoint_draft_id"] = draft.draft_id
                    drafts.append(draft.draft_id)
            draft = worker.add_draft(
                "skip_meeting", {"meeting_id": plan["meeting_id"]},
                rationale=("Record that Victor will not attend. This must be confirmed only "
                           "after any requested viewpoint has been relayed."),
                thread_id=resolved_thread)
            if draft:
                plan["attendance_draft_id"] = draft.draft_id
                drafts.append(draft.draft_id)
        plan["draft_ids"] = drafts
        self.human._log_hci_event("liaison_meeting_plan_interpreted", {
            "agent_id": seat_id, "meeting_id": plan["meeting_id"],
            "decision": decision, "has_viewpoint": bool(viewpoint),
            "has_return_focus": bool(return_focus),
        })
        return {**recorded, "handled": "meeting_instruction"}

    def ask(self, token: str, text: str, *, reply_to: str = "",
            thread_id: str = "", decision_context: Any = None) -> Dict[str, Any]:
        try:
            _seat, view = self._authenticated_view(token)  # authenticate before accepting text
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        if not (text or "").strip():
            return {"error": "empty_message"}
        notifications = self._notifications(token, view)
        meeting_updates, _meeting_cards = self._meeting_updates(token, view)
        message_updates = self._message_updates(token, view)
        self._register_thread_roots(
            token, view, self.human._agent(self._seat_id(view)),
            notifications, meeting_updates, message_updates,
        )
        worker = self.human._agent(self._seat_id(view))
        try:
            submitted_references, validated_decision_context = self._validated_decision_context(
                token, view, decision_context)
        except ValueError as exc:
            return {"error": str(exc)}
        recorded_human = worker.record_human_message(
            text.strip(), reply_to=reply_to, thread_id=thread_id,
            references=submitted_references,
            decision_context=validated_decision_context)
        if recorded_human.get("error"):
            return recorded_human
        human_message_id = str(recorded_human["message_id"])
        route_key = str(recorded_human.get("thread_id") or "") or "__main__"
        self._failed_routes.setdefault(token, {}).pop(route_key, None)
        routing = self._route_human_instruction(
            token, text.strip(), view, reply_to=reply_to, thread_id=thread_id,
            exclude_message_id=human_message_id,
            decision_context=validated_decision_context)
        if routing.get("error"):
            worker.last_error = str(routing["error"])
            self._failed_routes.setdefault(token, {})[route_key] = {
                "thread_id": str(recorded_human.get("thread_id") or ""),
                "original_request": text.strip(),
                "error": str(routing["error"]),
                "step": 0, "max_steps": MAX_STEPS, "next_step": 1,
                "source_count": 0, "sources": [],
                "human_message_id": human_message_id,
                "reply_to": str(reply_to or ""),
            }
            recorded = self._record_local_exchange(
                self._seat_id(view), text.strip(),
                "秘书模型没有完成这次解释。你的原消息已经保留；系统没有用正则或模板替代模型，"
                "也没有准备或执行任何动作。模型恢复后可以直接重试。",
                kind="model_error", reply_to=reply_to, thread_id=thread_id,
                human_message_id=human_message_id,
            )
            if recorded.get("error"):
                return recorded
            self.human._log_hci_event("liaison_model_error", {
                "agent_id": self._seat_id(view), "reason": str(routing["error"]),
            })
            return {**recorded, "handled": "model_error", "retryable": True}
        if routing.get("route") == "ordinary":
            return worker.enqueue_recorded_message(
                human_message_id, require_model=True)
        return self._handle_model_routed_instruction(
            token, text.strip(), view, routing,
            reply_to=reply_to, thread_id=thread_id,
            human_message_id=human_message_id)

    def execution_start(self, token: str, job_id: str) -> Dict[str, Any]:
        """Confirm the one semantic start of an already model-planned job."""
        try:
            self._authenticated_view(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        return self.human.execution_start(token, job_id)

    def cancel_request(self, token: str) -> Dict[str, Any]:
        try:
            self._authenticated_view(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        return self.human.agent_cancel_request(token)

    def retry_request(self, token: str, *, thread_id: str = "") -> Dict[str, Any]:
        try:
            _seat, view = self._authenticated_view(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        key = str(thread_id or "") or "__main__"
        failed_route = self._failed_routes.get(token, {}).get(key)
        if failed_route is not None:
            text = str(failed_route["original_request"])
            message_id = str(failed_route["human_message_id"])
            routing = self._route_human_instruction(
                token, text, view,
                reply_to=str(failed_route.get("reply_to") or ""),
                thread_id=str(failed_route.get("thread_id") or ""),
                exclude_message_id=message_id,
            )
            if routing.get("error"):
                failed_route["error"] = str(routing["error"])
                return {"accepted": False, "retryable": True,
                        "thread_id": str(failed_route.get("thread_id") or "")}
            self._failed_routes[token].pop(key, None)
            worker = self.human._agent(self._seat_id(view))
            if routing.get("route") == "ordinary":
                return worker.enqueue_recorded_message(message_id, require_model=True)
            return self._handle_model_routed_instruction(
                token, text, view, routing,
                reply_to=str(failed_route.get("reply_to") or ""),
                thread_id=str(failed_route.get("thread_id") or ""),
                human_message_id=message_id,
            )
        return self.human.agent_retry_request(token, thread_id=thread_id)

    def execution_cancel(self, token: str, job_id: str) -> Dict[str, Any]:
        try:
            self._authenticated_view(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        return self.human.execution_cancel(token, job_id)

    def prepare(self, token: str, action_type: str, params: Optional[Dict[str, Any]] = None,
                rationale: str = "") -> Dict[str, Any]:
        """Create, but never submit, any ordinary P2 member-action draft.

        This is the authenticated contract used by an integrated liaison
        parser.  It deliberately accepts no privileged action set: the P2
        gateway still performs the exact parameter, role, visibility, and
        current-state validation upon explicit ``confirm``.
        """
        try:
            seat, _view = self._authenticated_view(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        if not (action_type or "").strip():
            return {"error": "empty_action_type"}
        draft = self.human._agent(seat.agent_id).add_draft(
            str(action_type), dict(params or {}), rationale=str(rationale or ""))
        if draft is None:
            return {"error": f"unknown_action_type:{action_type}"}
        return {"draft": draft.public()}

    def confirm(self, token: str, draft_id: str) -> Dict[str, Any]:
        """Confirm through P2's current-state gateway validation, exactly once."""
        try:
            self._authenticated_view(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        linked_plan = next((plan for plan in self._meeting_plans.get(token, {}).values()
                            if draft_id in (plan.get("draft_ids") or [])), None)
        if (linked_plan and draft_id == linked_plan.get("attendance_draft_id")
                and linked_plan.get("viewpoint_draft_id")
                and not linked_plan.get("viewpoint_confirmed")):
            return {"error": ("confirm_meeting_viewpoint_first:"
                              f"{linked_plan['viewpoint_draft_id']}")}
        result = self.human.agent_confirm(token, draft_id)
        if linked_plan is not None and result.get("ok"):
            if draft_id == linked_plan.get("viewpoint_draft_id"):
                linked_plan["viewpoint_confirmed"] = True
            if draft_id == linked_plan.get("attendance_draft_id"):
                linked_plan["attendance_confirmed"] = True
                linked_plan["confirmed_at"] = time.time()
            self.human._log_hci_event("liaison_meeting_plan_confirmed", {
                "agent_id": P3_FIXED_SEAT_ID, "meeting_id": linked_plan["meeting_id"],
                "decision": linked_plan["decision"], "draft_id": draft_id,
            })
        return result

    def confirm_meeting_plan(self, token: str, plan_id: str) -> Dict[str, Any]:
        """Apply one human RSVP plan through its ordinary P2 actions.

        The plan may compile to a message followed by an RSVP, but P3 presents
        and confirms it as one semantic decision.  Ordering stays server-side
        so a person cannot hit the RSVP first and receive a raw draft error.
        """
        try:
            seat, _view = self._authenticated_view(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        plan = next((row for row in self._meeting_plans.get(token, {}).values()
                     if hmac.compare_digest(str(row.get("plan_id") or ""),
                                            str(plan_id or ""))), None)
        if plan is None:
            return {"error": "meeting_plan_not_found"}
        if plan.get("decision") == "pending":
            return {"error": "meeting_plan_needs_instruction"}
        if plan.get("attendance_confirmed"):
            return {"error": "meeting_plan_already_confirmed"}

        ordered = []
        viewpoint_draft = str(plan.get("viewpoint_draft_id") or "")
        attendance_draft = str(plan.get("attendance_draft_id") or "")
        if viewpoint_draft and not plan.get("viewpoint_confirmed"):
            ordered.append(viewpoint_draft)
        if attendance_draft:
            ordered.append(attendance_draft)
        if not ordered:
            return {"error": "meeting_plan_has_no_pending_action"}

        # One semantic confirmation must not relay the human's words and only
        # afterwards discover that the RSVP is already invalid. Preflight the
        # complete component set against one current-world snapshot before the
        # first mutation. The ordinary gateway is still run again by each
        # confirmation, so its exactly-once and current-state guarantees stay
        # authoritative.
        from environments.org_env.human import gateway
        working = self.human._agent(seat.agent_id)
        try:
            with self.human.runtime().lock:
                for draft_id in ordered:
                    draft = working.drafts.get(draft_id)
                    if draft is None or draft.status != "pending":
                        return {"error": f"unknown_or_settled_draft:{draft_id}",
                                "completed_steps": []}
                    gateway.validate(self.human.runtime().world, seat.agent_id,
                                     draft.action_type, draft.params)
        except gateway.ActionRefused as exc:
            return {"error": str(exc), "completed_steps": []}

        completed = []
        for draft_id in ordered:
            result = self.confirm(token, draft_id)
            if not result.get("ok"):
                return {"error": str(result.get("error") or result.get("failure_reason")
                                     or "meeting_plan_confirmation_failed"),
                        "completed_steps": completed}
            completed.append(draft_id)

        relayed = bool(plan.get("viewpoint_confirmed"))
        decision = str(plan.get("decision") or "")
        plan_language = (str(plan.get("viewpoint") or "")
                         + str(plan.get("return_focus") or ""))
        if any("\u3400" <= char <= "\u9fff" for char in plan_language):
            summary = ("会议安排已确认：观点已转达，你将不参会。"
                       if relayed else
                       "会议安排已确认。" + ("你将亲自参会。" if decision == "attend"
                                            else "你将不参会。"))
        else:
            summary = ("Meeting plan confirmed: your viewpoint was relayed and you will not "
                       "attend." if relayed else
                       "Meeting plan confirmed: " + ("you will attend."
                                                      if decision == "attend"
                                                      else "you will not attend."))
        self.human._agent(seat.agent_id)._append(
            "agent", summary, kind="meeting_instruction")
        return {"ok": True, "plan_id": plan_id, "decision": decision,
                "viewpoint_relayed": relayed,
                "attendance_confirmed": bool(plan.get("attendance_confirmed")),
                "message": summary}

    def discard_meeting_plan(self, token: str, plan_id: str) -> Dict[str, Any]:
        """Discard every still-pending component without exposing its ids."""
        try:
            seat, _view = self._authenticated_view(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        plan = next((row for row in self._meeting_plans.get(token, {}).values()
                     if hmac.compare_digest(str(row.get("plan_id") or ""),
                                            str(plan_id or ""))), None)
        if plan is None:
            return {"error": "meeting_plan_not_found"}
        discarded = []
        worker = self.human._agent(seat.agent_id)
        for draft_id in list(plan.get("draft_ids") or []):
            draft = worker.drafts.get(draft_id)
            if draft is not None and draft.status == "pending":
                result = worker.discard(draft_id)
                if result.get("discarded"):
                    discarded.append(draft_id)
        plan.update({"decision": "pending", "viewpoint": "", "return_focus": "",
                     "draft_ids": [], "viewpoint_draft_id": "",
                     "attendance_draft_id": "", "viewpoint_confirmed": False,
                     "attendance_confirmed": False})
        worker._append("agent", "会议计划已取消，你可以继续追问或重新说明安排。",
                       kind="meeting_instruction")
        return {"ok": True, "plan_id": plan_id, "discarded_count": len(discarded)}

    def discard(self, token: str, draft_id: str) -> Dict[str, Any]:
        try:
            self._authenticated_view(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        result = self.human.agent_discard(token, draft_id)
        linked_plan = next((plan for plan in self._meeting_plans.get(token, {}).values()
                            if draft_id in (plan.get("draft_ids") or [])), None)
        if linked_plan is not None and result.get("discarded"):
            linked_plan.update({"decision": "pending", "viewpoint": "", "return_focus": "",
                                "draft_ids": [], "viewpoint_draft_id": "",
                                "attendance_draft_id": "", "viewpoint_confirmed": False,
                                "attendance_confirmed": False})
        return result


__all__ = ["LiaisonFacade", "P3_FIXED_SEAT_ID"]
