"""OrgEpisodeManager — turns the event stream into causal organizational episodes.

Observes either an ``ExecutionResult`` (richest: action_type + created/modified
objects) or a world-level event dict (meeting/commitment/background), normalizes
it into an :class:`EpEvent`, then OPENS a new episode on a trigger signal, ATTACHES
related events to open episodes (object overlap / shared participant / proximity /
type compatibility), and CLOSES episodes once an outcome object appears or the
episode goes quiet. Template summaries (no LLM) describe each episode.

Generic only: no per-agent-id branches anywhere.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from environments.org_env.episodes.episode import (
    ABSORB_INTO,
    attach_only_episode_type,
    EPISODE_COMPATIBLE_EVENT_TYPES,
    EPISODE_TITLE,
    WORKFLOW_TYPES,
    EpEvent,
    OrgEpisode,
    classify_object,
    is_debug_failure,
    trigger_episode_type,
)

# attachment / closure tuning
ATTACH_THRESHOLD = 3
PROXIMITY = 12          # ticks within which an event is "near" an episode
CLOSE_QUIET = 24        # ticks of silence before an open episode resolves/abandons
# size guards — an episode that grows past any of these is closed (effectively a
# split: the next trigger opens a fresh episode), so nothing absorbs unboundedly.
MAX_EVENTS = 40
MAX_OBJECTS = 30
MAX_DURATION_TICKS = 48

# object-id keys that may appear in a raw event payload
_ID_KEYS = ("message_id", "task_id", "doc_id", "meeting_id", "pr_id", "commit_id",
            "branch_id", "result_id", "job_id", "protocol_id", "post_id", "issue_id",
            "commitment_id", "dispute_id", "request_id", "artifact_id", "object_id",
            "search_id", "notes_doc_id", "target")


class OrgEpisodeManager:
    def __init__(self) -> None:
        self.episodes: Dict[str, OrgEpisode] = {}
        self.events_by_id: Dict[str, EpEvent] = {}
        self._ep_seq = 0
        self._ev_seq = 0

    # ====================== observation entry points ====================== #
    def observe_result(self, result: Any, world: Any) -> List[OrgEpisode]:
        """Observe every event a chosen action produced (ExecutionResult)."""
        touched: List[OrgEpisode] = []
        created = list(getattr(result, "created_objects", []) or [])
        modified = list(getattr(result, "modified_objects", []) or [])
        channel = (getattr(result, "state_delta", {}) or {}).get("channel_id")
        for raw in (getattr(result, "events", []) or []):
            ev = self._ep_event_from_raw(raw, world, action_type=result.action_type,
                                         default_actor=result.agent_id, created=created,
                                         modified=modified, success=result.success, channel=channel)
            ep = self._observe(ev, world)
            if ep is not None:
                touched.append(ep)
        return touched

    def observe_world_event(self, raw: Dict[str, Any], world: Any) -> Optional[OrgEpisode]:
        """Observe a world-level event (meeting lifecycle / commitment violation /
        background completion) not produced by a single ExecutionResult."""
        ev = self._ep_event_from_raw(raw, world, action_type=raw.get("action_type", ""),
                                     default_actor=raw.get("agent_id"), created=[], modified=[],
                                     channel=raw.get("channel_id"))
        return self._observe(ev, world)

    def update_open_episodes(self, world: Any) -> None:
        tick = int(getattr(world, "world_tick", 0))
        for ep in list(self.episodes.values()):
            if ep.status == "open":
                self.maybe_close_episode(ep, world, tick)

    # ====================== normalization ================================ #
    def _ep_event_from_raw(self, raw, world, *, action_type, default_actor, created,
                           modified, success=True, channel=None) -> EpEvent:
        self._ev_seq += 1
        family = raw.get("type", "action_event")
        subtype = raw.get("subtype", "")
        actor = raw.get("agent_id") or default_actor
        tick = int(raw.get("tick", getattr(world, "world_tick", 0)))
        at = raw.get("action_type", action_type) or ""
        chan = raw.get("channel_id") or channel
        # object ids: explicit payload ids + the action's created/modified objects
        oids: List[str] = []
        for k in _ID_KEYS:
            v = raw.get(k)
            if isinstance(v, str) and v:
                oids.append(v)
        for v in list(created) + list(modified):
            if isinstance(v, str) and v:
                oids.append(v)
        # an agent the event targets (for participant linkage)
        target_id = None
        for k in ("target_agent", "recipient_id", "owner_id", "assignee_id", "target"):
            v = raw.get(k)
            if isinstance(v, str) and v in getattr(world, "agents", {}):
                target_id = v
                break
        # dedupe, drop the actor's own id
        seen, clean = set(), []
        for o in oids:
            if o and o not in seen and o not in getattr(world, "agents", {}):
                seen.add(o)
                clean.append(o)
        primary = self._primary_object(raw, clean, created)
        return EpEvent(event_id=f"epev{self._ev_seq}", tick=tick, family=family, subtype=subtype,
                       action_type=at, actor_id=actor, target_id=target_id, object_ids=clean,
                       primary_object_id=primary, channel_id=chan, success=bool(success), raw=dict(raw))

    @staticmethod
    def _primary_object(raw, clean, created) -> Optional[str]:
        for k in ("result_id", "dispute_id", "protocol_id", "post_id", "pr_id",
                  "task_id", "doc_id", "meeting_id", "commitment_id", "request_id"):
            v = raw.get(k)
            if isinstance(v, str) and v:
                return v
        tgt = raw.get("target")
        if isinstance(tgt, str) and tgt and not tgt.startswith("ep"):
            return tgt
        if created:
            return created[0]
        return clean[0] if clean else None

    # ====================== core observe ================================= #
    def _observe(self, ev: EpEvent, world: Any) -> Optional[OrgEpisode]:
        self.events_by_id[ev.event_id] = ev
        trig = trigger_episode_type(ev)
        primary: Optional[OrgEpisode] = None
        if trig:
            # anti-spam / workflow absorption: attach to an already-open episode of a
            # related type (ABSORB_INTO) if it matches; otherwise open a fresh one.
            absorb = [ep for ep in self._open() if ep.episode_type in ABSORB_INTO.get(trig, {trig})]
            if trig == "launch_crunch_episode":
                # a launch is a COMPANY-WIDE time window, not object-scoped — collapse
                # all launch signals into the open launch episode instead of opening
                # dozens of one-event launch episodes (preflight review #2).
                live = [e for e in absorb if not self._is_full(e, ev.tick)]
                primary = max(live, key=lambda e: e.updated_at_tick) if live \
                    else self.open_episode(trig, ev, world)
            else:
                best, score = self._best_match(ev, absorb, world)
                if best is not None and score >= ATTACH_THRESHOLD:
                    primary = best
                else:
                    primary = self.open_episode(trig, ev, world)
            self.attach_event_to_episode(ev, primary, world)
        elif attach_only_episode_type(ev):
            # C-fix: attach-only trigger (outbound org post/reply). Join an OPEN absorbable
            # feedback/triage loop if one matches; otherwise DO NOTHING (never open a fresh
            # episode) so outbound chatter can't spawn zero-action episodes that then abandon.
            atype = attach_only_episode_type(ev)
            absorb = [ep for ep in self._open() if ep.episode_type in ABSORB_INTO.get(atype, {atype})]
            best, score = self._best_match(ev, absorb, world)
            if best is not None and score >= ATTACH_THRESHOLD:
                primary = best
                self.attach_event_to_episode(ev, primary, world)
        else:
            best, score = self._best_match(ev, self._open(), world)
            if best is not None and score >= ATTACH_THRESHOLD:
                primary = best
                self.attach_event_to_episode(ev, primary, world)
        # cross-link: any OTHER open episode that shares an object with this event
        if ev.object_ids:
            for ep in self._open():
                if ep is primary or self._is_full(ep, ev.tick):
                    continue
                if set(ev.object_ids) & set(ep.linked_object_ids):
                    if ev.event_id not in ep.linked_event_ids:
                        ep.linked_event_ids.append(ev.event_id)
                        ep.timeline.append(ev.compact())
                    if primary is not None:
                        self._relate(primary, ep)
        return primary

    def _open(self) -> List[OrgEpisode]:
        return [ep for ep in self.episodes.values() if ep.status == "open"]

    def _best_match(self, ev, episodes, world):
        best, best_score = None, 0
        for ep in episodes:
            if self._is_full(ep, ev.tick):        # full episodes don't absorb (split)
                continue
            s = self.score_event_episode_match(ev, ep, world)
            if s > best_score:
                best, best_score = ep, s
        return best, best_score

    def score_event_episode_match(self, ev: EpEvent, ep: OrgEpisode, world: Any) -> int:
        obj = bool(set(ev.object_ids) & set(ep.linked_object_ids))
        chan = bool(ev.channel_id and ev.channel_id in ep.linked_channels)
        act = bool(ev.actor_id and ev.actor_id in ep.participants)
        tgt = bool(ev.target_id and ev.target_id in ep.participants)
        # LINKAGE GATE: an event must share an object, a thread/channel, or a
        # participant with the episode. Proximity + type-compatibility ALONE never
        # attach. Workflow episodes (feedback / triage / launch) are stricter: they
        # require object OR channel overlap (a shared participant is not enough), so
        # they don't vacuum up everyone who happens to act nearby.
        if ep.episode_type in WORKFLOW_TYPES:
            if not (obj or chan):
                return -1
        elif not (obj or chan or act or tgt):
            return -1
        score = (3 if obj else 0) + (2 if chan else 0) + (1 if act else 0) + (1 if tgt else 0)
        if abs(ev.tick - ep.updated_at_tick) <= PROXIMITY:
            score += 1
        if ev.family in EPISODE_COMPATIBLE_EVENT_TYPES.get(ep.episode_type, set()):
            score += 1
        if ev.tick - ep.updated_at_tick > CLOSE_QUIET:
            score -= 3
        return score

    def _is_full(self, ep: OrgEpisode, tick: int) -> bool:
        return (len(ep.linked_event_ids) >= MAX_EVENTS
                or len(ep.linked_object_ids) >= MAX_OBJECTS
                or (tick - ep.start_tick) > MAX_DURATION_TICKS)

    # ====================== open / attach ================================ #
    def open_episode(self, etype: str, ev: EpEvent, world: Any) -> OrgEpisode:
        self._ep_seq += 1
        eid = f"ep_{etype.split('_')[0]}_{self._ep_seq}"
        ep = OrgEpisode(
            episode_id=eid, episode_type=etype, start_tick=ev.tick, status="open",
            trigger_event_id=ev.event_id, trigger_object_id=ev.primary_object_id,
            primary_agent_id=ev.actor_id, created_at_tick=ev.tick, updated_at_tick=ev.tick,
            title=EPISODE_TITLE.get(etype, etype))
        ep.problem_statement = self._problem(etype, ev, world)
        self.episodes[eid] = ep
        if etype == "debugging_episode":
            self._enrich_debugging(ep, ev, world)
        return ep

    def _enrich_debugging(self, ep: OrgEpisode, ev: EpEvent, world: Any) -> None:
        """Anchor a debugging episode on the failure log + suspected module + blocker issue,
        so the subsequent fix loop (edit -> commit -> PR -> CI -> rerun gate) attaches to it
        and the whole 'how we fixed gate X' story is captured for later recall."""
        raw = ev.raw or {}
        # 1. failure log
        err = str(getattr(world, "_build_error", "") or "")
        blockers = [str(b) for b in (raw.get("blockers") or []) if b]
        log = err or (str(raw.get("status", "")) + ((": " + ", ".join(blockers)) if blockers else ""))
        ep.failure_log = log[:600]
        ep.failing_gates = blockers or ([raw.get("subtype")] if raw.get("subtype") else [])
        # 2. suspected module: the .py named in the build error -> its artifact
        art = self._artifact_for_error(world, err)
        if art is not None:
            ep.suspected_module = getattr(art, "linked_file_path", "") or art.artifact_id
            ep.link_object(art.artifact_id)
        # 3. the smoke/CI blocker issue (so the fix task + edits cross-link here)
        for iid, a in (getattr(world, "product_artifacts", {}) or {}).items():
            if getattr(a, "artifact_type", "") == "issue" and getattr(a, "status", "") == "open" \
                    and str(iid).startswith("rel_blocker_") \
                    and any(k in str(iid).lower() for k in ("smoke", "ci", "eval", "test", "build")):
                ep.link_object(iid)
        ep.updated_at_tick = ev.tick

    @staticmethod
    def _artifact_for_error(world: Any, err: str):
        import re
        m = re.match(r"\s*([\w./\\-]+\.py)", err or "")
        if not m:
            return None
        base = m.group(1).replace("\\", "/").split("/")[-1]
        for a in (getattr(world, "product_artifacts", {}) or {}).values():
            fp = (getattr(a, "linked_file_path", "") or "").replace("\\", "/")
            if fp and fp.split("/")[-1] == base:
                return a
        return None

    def ensure_product_episode(self, result: Any, world: Any) -> OrgEpisode:
        """Preflight v3 §6: a product change with no causal episode gets one — reuse an
        open product/launch episode if present, else open a product_work_episode — so
        every artifact revision is anchored in an explainable organizational process."""
        for ep in self.episodes.values():
            if ep.status == "open" and ep.episode_type in (
                    "product_work_episode", "launch_crunch_episode", "claim_evidence_episode",
                    "docs_alignment_episode", "product_maintenance_episode"):
                return ep
        raw = {"type": "product_event", "subtype": "revised", "agent_id": result.agent_id,
               "tick": int(getattr(world, "world_tick", 0)), "action_type": result.action_type}
        ev = self._ep_event_from_raw(
            raw, world, action_type=result.action_type, default_actor=result.agent_id,
            created=list(getattr(result, "created_objects", []) or []),
            modified=list(getattr(result, "modified_objects", []) or []),
            success=getattr(result, "success", True),
            channel=(getattr(result, "state_delta", {}) or {}).get("channel_id"))
        ep = self.open_episode("product_work_episode", ev, world)
        self.attach_event_to_episode(ev, ep, world)
        return ep

    def attach_event_to_episode(self, ev: EpEvent, ep: OrgEpisode, world: Any) -> None:
        ep.add_participant(ev.actor_id)
        if ev.target_id:
            ep.add_participant(ev.target_id)
        if ev.event_id not in ep.linked_event_ids:
            ep.linked_event_ids.append(ev.event_id)
            ep.timeline.append(ev.compact())
        for oid in ev.object_ids:
            ep.link_object(oid)
        if ev.channel_id and ev.channel_id not in ep.linked_channels:
            ep.linked_channels.append(ev.channel_id)
        self._record_produced(ev, ep, world)
        ep.updated_at_tick = max(ep.updated_at_tick, ev.tick)
        self.events_by_id[ev.event_id] = ev

    def _relate(self, a: OrgEpisode, b: OrgEpisode) -> None:
        if b.episode_id not in a.related_episode_ids:
            a.related_episode_ids.append(b.episode_id)
        if a.episode_id not in b.related_episode_ids:
            b.related_episode_ids.append(a.episode_id)

    def _record_produced(self, ev: EpEvent, ep: OrgEpisode, world: Any) -> None:
        """Track outcome objects (artifacts / protocols / tasks / product changes)."""
        # protocol proposal / adoption
        if ev.family in ("protocol_proposal_event",) or ev.action_type == "propose_protocol":
            pid = ev.primary_object_id or next((o for o in ev.object_ids
                                                if classify_object(o) == "protocol"), None)
            if pid and pid not in ep.produced_protocols:
                ep.produced_protocols.append(pid)
        # public-facing update / launch post
        if ev.family == "external_signal_event" and ev.subtype in ("company_update",):
            tag = f"public_update@{ev.tick}"
            if tag not in ep.produced_product_changes:
                ep.produced_product_changes.append(tag)
        # v5 §P1-1: patches/merges/releases are concrete product changes
        if ev.family in ("product_event", "release_event") and ev.subtype in (
                "patch_applied", "merged_to_mainline", "published"):
            obj = ev.primary_object_id or (ev.object_ids[0] if ev.object_ids else "")
            tag = f"{ev.subtype}:{obj}@{ev.tick}" if obj else f"{ev.subtype}@{ev.tick}"
            if tag not in ep.produced_product_changes:
                ep.produced_product_changes.append(tag)
        # v11: a debugging episode records each fix attempt (edit / commit)
        if ep.episode_type == "debugging_episode" and (
                ev.action_type in ("edit_repo_file", "commit_patch", "write_design_note")
                or ev.subtype in ("patch_applied", "commit")):
            tag = f"{ev.action_type or ev.subtype}@{ev.tick}"
            if tag not in ep.attempted_patches:
                ep.attempted_patches.append(tag)
        # created objects by kind
        for oid in ev.object_ids:
            kind = classify_object(oid)
            if kind in ("doc", "artifact") and oid not in ep.produced_artifacts:
                ep.produced_artifacts.append(oid)
            elif kind == "task" and oid not in ep.produced_tasks:
                ep.produced_tasks.append(oid)
            elif kind == "repo" and oid not in ep.produced_product_changes:
                ep.produced_product_changes.append(oid)

    # ====================== closure ====================================== #
    def maybe_close_episode(self, ep: OrgEpisode, world: Any, tick: int) -> bool:
        outcome = self._outcome_reached(ep, world)
        quiet = tick - ep.updated_at_tick > CLOSE_QUIET
        full = self._is_full(ep, tick)            # size/duration guard -> close (split)
        # workflow episodes span many agents/steps — keep them OPEN until activity
        # stops (quiet) or they hit a size guard, so the whole chain (task → artifact
        # → response) is absorbed; focused episodes also close on their terminal outcome.
        if ep.episode_type in WORKFLOW_TYPES:
            if not (quiet or full):
                return False
        elif not (outcome or quiet or full):
            return False
        ep.end_tick = tick
        produced = bool(ep.produced_artifacts or ep.produced_protocols
                        or ep.produced_tasks or ep.produced_product_changes)
        ep.status = "resolved" if (outcome or produced) else "abandoned"
        self.summarize_episode(ep, world)
        return True

    def _outcome_reached(self, ep: OrgEpisode, world: Any) -> bool:
        t = ep.episode_type
        if t == "experiment_episode":
            return self._any_result_tracked(ep, world) or bool(ep.produced_artifacts)
        if t == "claim_dispute_episode":
            return (self._any_dispute_closed(ep, world) or bool(ep.produced_artifacts)
                    or bool(ep.produced_protocols))
        if t == "feedback_ingestion_episode":
            return bool(ep.produced_tasks or ep.produced_artifacts or ep.produced_product_changes)
        if t == "protocol_formation_episode":
            return self._any_protocol_settled(ep, world)
        if t == "launch_crunch_episode":
            return bool(ep.produced_product_changes)
        if t == "customer_triage_episode":
            return bool(ep.produced_artifacts or ep.produced_tasks or ep.produced_product_changes)
        if t == "debugging_episode":
            # resolved once a fix landed AND the technical failure cleared: world._build_error
            # is empty (smoke re-passed) and the linked blocker issue is no longer open.
            fixed = bool(ep.produced_product_changes)
            cleared = not (getattr(world, "_build_error", "") or "")
            blocker_closed = self._debug_blocker_closed(ep, world)
            return (fixed and cleared) or blocker_closed
        return False

    def _debug_blocker_closed(self, ep: OrgEpisode, world: Any) -> bool:
        arts = getattr(world, "product_artifacts", {}) or {}
        blk = [o for o in ep.linked_object_ids if str(o).startswith("rel_blocker_")]
        if not blk:
            return False
        return all(getattr(arts.get(b), "status", "open") != "open" for b in blk if b in arts)

    def _any_result_tracked(self, ep, world) -> bool:
        ss = getattr(world, "sandbox_system", None)
        for rid in ep.linked_sandbox_result_ids:
            r = getattr(ss, "results", {}).get(rid) if ss else None
            if r is not None and getattr(r, "logged_to_tracker", False):
                return True
        return False

    def _any_dispute_closed(self, ep, world) -> bool:
        cr = getattr(world, "commitment_registry", None)
        disputes = [o for o in ep.linked_object_ids if classify_object(o) == "speech_object"
                    and o.startswith("dispute_")]
        for did in disputes:
            d = getattr(cr, "disputes", {}).get(did) if cr else None
            if d is not None and d.status != "open":
                return True
        return False

    def _any_protocol_settled(self, ep, world) -> bool:
        reg = getattr(world, "protocol_registry", None)
        for pid in ep.linked_protocol_ids:
            p = getattr(reg, "protocols", {}).get(pid) if reg else None
            if p is None:
                continue
            if getattr(p, "adoption_status", "") == "adopted":
                return True
            if getattr(p, "violation_events", None) and getattr(p, "enforcement_events", None):
                return True
        return False

    # ====================== summaries (template, no LLM) ================= #
    def _name(self, world, aid) -> str:
        a = getattr(world, "agents", {}).get(aid)
        return getattr(a, "name", aid) if a else (aid or "someone")

    def _problem(self, etype: str, ev: EpEvent, world: Any) -> str:
        who = self._name(world, ev.actor_id)
        obj = ev.primary_object_id or "an object"
        return {
            "feedback_ingestion_episode": f"External signal {obj} entered the company; who responds and what changes?",
            "claim_dispute_episode": f"{who} challenged {obj} — is the result reproducible / sufficiently evidenced?",
            "experiment_episode": f"{who} ran an experiment ({obj}); will the result be shared and tracked?",
            "protocol_formation_episode": f"A rule was proposed ({obj}); will it be adopted and enforced?",
            "launch_crunch_episode": f"Launch pressure ({who} {ev.action_type or ev.subtype}); can the team ship without breaking quality?",
            "customer_triage_episode": f"A customer issue needs triage and a response ({obj}).",
            "debugging_episode": f"A technical gate failed ({obj}); localize the bug, patch it, and re-pass the gate.",
        }.get(etype, f"{etype} triggered by {who}.")

    def summarize_episode(self, ep: OrgEpisode, world: Any) -> None:
        names = ", ".join(self._name(world, a) for a in ep.participants) or "—"
        # conflict
        cr = getattr(world, "commitment_registry", None)
        disputes = [d for d in getattr(cr, "disputes", {}).values()
                    if d.dispute_id in ep.linked_object_ids] if cr else []
        if disputes:
            d = disputes[0]
            ep.conflict_summary = (f"{self._name(world, d.challenger_id)} disputed {d.target_object_id} "
                                   f"(demanded evidence/reproducibility).")
        elif any(self.events_by_id.get(e) and self.events_by_id[e].subtype == "changes_requested"
                 for e in ep.linked_event_ids):
            ep.conflict_summary = "Reviewer requested changes / flagged quality concerns."
        elif ep.episode_type == "launch_crunch_episode":
            ep.conflict_summary = "Speed-vs-quality tension during launch crunch."
        # decision
        if ep.produced_protocols:
            ep.decision_summary = f"Protocol(s) proposed/established: {', '.join(ep.produced_protocols)}."
        elif any("tracker" in a for a in ep.produced_artifacts):
            ep.decision_summary = "Decided to track results in a shared tracker."
        elif ep.linked_meeting_ids:
            ep.decision_summary = f"Discussed in meeting(s): {', '.join(ep.linked_meeting_ids)}."
        # outcome — produced objects + world-state outcomes (tracked / resolved / adopted)
        parts = []
        if ep.produced_artifacts:
            parts.append(f"artifacts: {', '.join(ep.produced_artifacts)}")
        if ep.produced_protocols:
            parts.append(f"protocols: {', '.join(ep.produced_protocols)}")
        if ep.produced_tasks:
            parts.append(f"tasks: {', '.join(ep.produced_tasks)}")
        if ep.produced_product_changes:
            parts.append(f"product changes: {', '.join(ep.produced_product_changes)}")
        if ep.episode_type == "experiment_episode" and self._any_result_tracked(ep, world):
            parts.append("result(s) logged to the shared tracker")
        if ep.episode_type == "claim_dispute_episode" and self._any_dispute_closed(ep, world):
            parts.append("dispute resolved")
        if ep.episode_type == "protocol_formation_episode" and self._any_protocol_settled(ep, world):
            parts.append("protocol adopted/enforced")
        if ep.episode_type == "debugging_episode":
            landed = ep.produced_product_changes or ep.attempted_patches
            gates = ", ".join(g for g in ep.failing_gates if g) or "smoke"
            mod = ep.suspected_module or "the failing module"
            if ep.status == "resolved":
                ep.resolution = (f"Fixed {mod}"
                                 + (f" via {', '.join(landed)}" if landed else "")
                                 + f"; gate(s) [{gates}] re-passed.")
            else:
                ep.resolution = (f"Unresolved — {mod} still failing gate(s) [{gates}]"
                                 + (f"; tried {', '.join(landed)}" if landed else "") + ".")
            ep.outcome_summary = ep.resolution + f" Participants: {names}."
        elif parts:
            ep.outcome_summary = ("Produced " + "; ".join(parts) + ". "
                                  f"Participants: {names}. "
                                  f"({ep.start_tick}→{ep.end_tick}, status={ep.status}).")
        else:
            ep.outcome_summary = (f"No concrete outcome; episode {ep.status}. "
                                  f"Participants: {names}.")
        # deltas
        ep.state_delta = {"participants": len(ep.participants),
                          "events": len(ep.linked_event_ids),
                          "messages": len(ep.linked_message_ids),
                          "duration_ticks": (ep.end_tick or ep.updated_at_tick) - ep.start_tick}
        ep.graph_delta = self._graph_delta(ep)

    def _graph_delta(self, ep: OrgEpisode) -> Dict[str, Any]:
        edges = []
        if ep.trigger_event_id:
            edges.append([ep.episode_id, "triggered_by", ep.trigger_event_id])
        for a in ep.participants:
            edges.append([ep.episode_id, "involved", a])
        for o in (ep.produced_artifacts + ep.produced_protocols + ep.produced_tasks
                  + ep.produced_product_changes):
            edges.append([ep.episode_id, "produced", o])
        for rid in ep.related_episode_ids:
            edges.append([ep.episode_id, "related_to", rid])
        return {"node": {"id": ep.episode_id, "type": "episode",
                         "episode_type": ep.episode_type, "status": ep.status},
                "edges": edges}

    # ====================== debugging recall (v11 memory) ================ #
    def recall_resolved_debugging(self, *, gate: str = "", file_hint: str = "",
                                  exclude_id: str = "") -> Optional[Dict[str, Any]]:
        """Coding memory: 'how did we fix this gate/file last time?'. Returns the most
        recent RESOLVED debugging episode matching the gate keyword and/or suspected file,
        with a compact recall (suspected module + resolution + patches)."""
        gk = (gate or "").lower()
        fk = (file_hint or "").lower()
        best = None
        for ep in self.episodes.values():
            if ep.episode_type != "debugging_episode" or ep.status != "resolved":
                continue
            if ep.episode_id == exclude_id or not ep.resolution:
                continue
            hay = (" ".join(ep.failing_gates) + " " + ep.suspected_module + " "
                   + ep.failure_log).lower()
            if gk and gk not in hay:
                continue
            if fk and fk not in (ep.suspected_module or "").lower():
                continue
            if best is None or (ep.end_tick or ep.updated_at_tick) > (best.end_tick or best.updated_at_tick):
                best = ep
        if best is None:
            return None
        return {
            "episode_id": best.episode_id,
            "suspected_module": best.suspected_module,
            "failing_gates": list(best.failing_gates),
            "resolution": best.resolution,
            "attempted_patches": list(best.attempted_patches),
            "end_tick": best.end_tick or best.updated_at_tick,
        }

    def recall_brief(self, *, gate: str = "", file_hint: str = "", exclude_id: str = "") -> str:
        """One-line recall string for prompts / blocker issues (empty if no prior fix)."""
        r = self.recall_resolved_debugging(gate=gate, file_hint=file_hint, exclude_id=exclude_id)
        if not r:
            return ""
        return (f"Prior fix ({r['episode_id']}): {r['resolution']}"
                + (f" [touched {r['suspected_module']}]" if r['suspected_module'] else ""))

    # ====================== snapshot ===================================== #
    def snapshot(self) -> Dict[str, Any]:
        items = [self._episode_item(ep) for ep in self.episodes.values()]
        items.sort(key=lambda x: x["start_tick"])
        by_type: Dict[str, int] = {}
        for ep in self.episodes.values():
            by_type[ep.episode_type] = by_type.get(ep.episode_type, 0) + 1
        return {
            "items": items,
            "open_count": sum(1 for ep in self.episodes.values() if ep.status == "open"),
            "closed_count": sum(1 for ep in self.episodes.values() if ep.status != "open"),
            "by_type": by_type,
            "total": len(self.episodes),
        }

    def _episode_item(self, ep: OrgEpisode) -> Dict[str, Any]:
        return {
            "episode_id": ep.episode_id, "episode_type": ep.episode_type, "title": ep.title,
            "status": ep.status, "start_tick": ep.start_tick, "end_tick": ep.end_tick,
            "participants": list(ep.participants), "primary_agent_id": ep.primary_agent_id,
            "trigger_event_id": ep.trigger_event_id, "trigger_object_id": ep.trigger_object_id,
            "problem_statement": ep.problem_statement, "conflict_summary": ep.conflict_summary,
            "decision_summary": ep.decision_summary, "outcome_summary": ep.outcome_summary,
            "failure_log": ep.failure_log, "suspected_module": ep.suspected_module,
            "failing_gates": list(ep.failing_gates), "attempted_patches": list(ep.attempted_patches),
            "resolution": ep.resolution,
            "produced_artifacts": list(ep.produced_artifacts),
            "produced_protocols": list(ep.produced_protocols),
            "produced_tasks": list(ep.produced_tasks),
            "produced_product_changes": list(ep.produced_product_changes),
            "linked_event_ids": list(ep.linked_event_ids),
            "linked_message_ids": list(ep.linked_message_ids),
            "linked_meeting_ids": list(ep.linked_meeting_ids),
            "linked_task_ids": list(ep.linked_task_ids),
            "linked_doc_ids": list(ep.linked_doc_ids),
            "linked_repo_ids": list(ep.linked_repo_ids),
            "linked_sandbox_result_ids": list(ep.linked_sandbox_result_ids),
            "linked_protocol_ids": list(ep.linked_protocol_ids),
            "linked_external_signal_ids": list(ep.linked_external_signal_ids),
            "linked_artifact_ids": list(ep.linked_artifact_ids),
            "linked_object_ids": list(ep.linked_object_ids),
            "linked_channels": list(ep.linked_channels),
            "related_episode_ids": list(ep.related_episode_ids),
            "linked_reflection_ids": list(ep.linked_reflection_ids),
            "linked_wish_ids": list(ep.linked_wish_ids),
            "linked_proposal_ids": list(ep.linked_proposal_ids),
            "state_delta": dict(ep.state_delta), "graph_delta": dict(ep.graph_delta),
            "timeline": [dict(t) for t in ep.timeline],
            "linked_objects_count": len(ep.linked_object_ids),
        }


__all__ = ["OrgEpisodeManager", "ATTACH_THRESHOLD", "CLOSE_QUIET"]
