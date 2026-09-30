"""OrgPerceptionAdapter — OrgWorld state -> OrgPerceptionPacket (DESIGN env_org
§70, O1 §6).

Enforces information asymmetry (O1 §2 #3/#4, §6.2): an agent only perceives
public/shared objects, its own private objects, messages in channels it belongs
to (with read_status), objects shared to it / to meetings it attends, and its
own branch/sandbox local objects. It never sees others' private workspace /
unshared sandbox results / unread message content / unopened attachments.

Perception keeps OBJECT REFERENCES (id + short summary), not expanded content
(§6.3), so downstream prompts/policies reference objects rather than copy them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class OrgPerceptionPacket:
    agent_id: str
    agent_name: str
    tick: int
    clock_state: Dict[str, Any] = field(default_factory=dict)
    self_state: Dict[str, float] = field(default_factory=dict)
    availability_state: Dict[str, Any] = field(default_factory=dict)
    visible_tasks: List[dict] = field(default_factory=list)
    assigned_tasks: List[dict] = field(default_factory=list)
    blocked_tasks: List[dict] = field(default_factory=list)
    unowned_critical_tasks: List[dict] = field(default_factory=list)
    visible_messages: List[dict] = field(default_factory=list)
    unread_mentions: List[dict] = field(default_factory=list)
    visible_docs: List[dict] = field(default_factory=list)
    visible_files: List[dict] = field(default_factory=list)
    visible_prs: List[dict] = field(default_factory=list)
    prs_awaiting_my_review: List[dict] = field(default_factory=list)
    visible_experiments: List[dict] = field(default_factory=list)
    visible_results: List[dict] = field(default_factory=list)
    local_unshared_results: List[dict] = field(default_factory=list)
    visible_meetings: List[dict] = field(default_factory=list)
    visible_protocols: List[dict] = field(default_factory=list)
    visible_budget_summary: Dict[str, Any] = field(default_factory=dict)
    visible_payroll_summary: Dict[str, Any] = field(default_factory=dict)
    visible_external_posts: List[dict] = field(default_factory=list)
    visible_search_results: List[dict] = field(default_factory=list)
    private_workspace_summary: Dict[str, Any] = field(default_factory=dict)
    sandbox_summary: Dict[str, Any] = field(default_factory=dict)
    repo_branch_summary: List[dict] = field(default_factory=list)
    recent_events: List[dict] = field(default_factory=list)
    memory_relevant_observations: List[dict] = field(default_factory=list)
    # reflection-memory decision context (so past reflection influences future acts)
    recent_reflections: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    unresolved_needs: List[str] = field(default_factory=list)
    open_wishes: List[str] = field(default_factory=list)
    open_wish_needs: List[str] = field(default_factory=list)


class OrgPerceptionAdapter:
    RECENT_EVENT_WINDOW = 30

    def build_perception(self, agent_id: str, org_world: Any, tick: int) -> OrgPerceptionPacket:
        w = org_world
        agent = w.agents[agent_id]
        av = w.time.availability.get(agent_id)
        clk = w.time.clock

        pkt = OrgPerceptionPacket(
            agent_id=agent_id, agent_name=agent.name, tick=tick,
            clock_state=clk.snapshot(),
            self_state=dict(agent.vitals.variables),
            availability_state={
                "status": av.current_availability_status if av else "available",
                "is_online": av.is_online() if av else True,
                "after_hours_responsiveness": av.after_hours_responsiveness if av else 0.4,
                "weekend_work_tendency": av.weekend_work_tendency if av else 0.3,
            },
        )

        self._fill_tasks(pkt, w, agent_id)
        self._fill_messages(pkt, w, agent_id)
        self._fill_docs_files(pkt, w, agent_id)
        self._fill_repo(pkt, w, agent_id)
        self._fill_sandbox_experiments(pkt, w, agent_id)
        self._fill_meetings(pkt, w, agent_id)
        self._fill_protocols(pkt, w, agent_id)
        self._fill_budget_payroll(pkt, w, agent, agent_id)
        self._fill_external(pkt, w, agent_id)
        self._fill_search(pkt, w, agent_id)
        self._fill_private(pkt, w, agent_id)
        self._fill_events(pkt, w, agent_id)
        self._fill_reflections(pkt, w, agent_id)
        return pkt

    # -- tasks (team-visible board; ownership emerges) ---------------------
    def _fill_tasks(self, pkt, w, aid):
        for tid, t in w.tasks.items():
            if getattr(t, "visibility", "team") not in ("team", "public") and t.owner_id != aid:
                continue
            # description carries the linked issue's reported text, so a task ref
            # without it named the work without stating it.
            ref = {"task_id": tid, "title": t.title, "status": getattr(t.status, "value", str(t.status)),
                   "owner": t.owner_id, "priority": getattr(t, "priority", 3),
                   "description": getattr(t, "description", "") or ""}
            from environments.org_env.cooperbench.visibility import is_strict_coop, visible_feature_brief
            if is_strict_coop(w) and str(tid).startswith("task_oss_cooper_feature_"):
                feature_id = str(tid).removeprefix("task_oss_")
                ref["description"] = visible_feature_brief(w, aid, feature_id) or "Private feature brief; request the owner's explicit share."
            pkt.visible_tasks.append(ref)
            # Finished work remains visible on the team board, but it is no
            # longer an actionable assignment for its former owner.
            if t.owner_id == aid and ref["status"] not in (
                    "merged", "released", "done", "abandoned"):
                pkt.assigned_tasks.append(ref)
            if getattr(t.status, "value", str(t.status)) == "blocked" and t.owner_id == aid:
                pkt.blocked_tasks.append(ref)
            if not t.owner_id and getattr(t, "priority", 3) >= 4:
                pkt.unowned_critical_tasks.append(ref)

    # -- messages (only channels the agent is in; read_status preserved) ---
    def _fill_messages(self, pkt, w, aid):
        # _process_inbox marks the inbox read just before this runs, so
        # read_status is True for everything and cannot distinguish anything.
        # What the agent has not seen before is what it took in on this pass.
        newly_read = (w.__dict__.get("_newly_read_messages") or {}).get(aid) or set()
        for m in w.comm.perceivable_messages(aid):
            read = aid in m.read_by
            ref = {"type": "message", "message_id": m.message_id, "sender": m.sender_id,
                   "newly_read": m.message_id in newly_read,
                   "channel": m.channel_id, "summary": m.text_summary,
                   "attachments": [a.object_id for a in m.attachments],
                   "attachment_types": [a.attachment_type for a in m.attachments],
                   "urgency": m.urgency, "importance": m.importance,
                   "mentions": list(m.mentions),
                    "read_status": "read" if read else "unread"}
            if getattr(w, "_cooperbench_delivery_focus", False) and read:
                ref["text"] = m.full_text if m.full_text is not None else m.text_summary
            pkt.visible_messages.append(ref)
            if aid in m.mentions and not read:
                pkt.unread_mentions.append(ref)

    # -- docs / files (company-visible + own private) ----------------------
    def _fill_docs_files(self, pkt, w, aid):
        for f in w.visible_files_for(aid):
            pkt.visible_files.append({"file_id": f.object_id, "file_type": f.file_type,
                                      "title": f.title, "owner": f.owner_id,
                                      "visibility": getattr(f.visibility, "value", str(f.visibility))})
        for did, d in w.documents.items():
            if getattr(d, "visibility", "team") in ("team", "public") or getattr(d, "owner_id", None) == aid:
                pkt.visible_docs.append({"doc_id": did, "title": d.title,
                                         "doc_type": getattr(d, "doc_type", "doc"),
                                         "owner": getattr(d, "owner_id", None)})

    # -- repo (own branches; PRs authored/reviewing) -----------------------
    def _fill_repo(self, pkt, w, aid):
        repo = w.repo_system.repo
        for bid, b in repo.branches.items():
            if b.owner_id == aid:
                pkt.repo_branch_summary.append({"branch_id": bid, "status": getattr(b.status, "value", str(b.status)),
                                                "commits": len(b.commit_ids),
                                                "uncommitted": b.uncommitted_changes,
                                                "linked_task": b.linked_task})
        for pid, pr in repo.pull_requests.items():
            involved = pr.author_id == aid or aid in pr.reviewers
            if not involved:
                continue
            # A CI failure does not change a request's status, so a blocked
            # request looked exactly like an approved one that was about to
            # land. Say whether it can merge, and if not, say why: the reason is
            # the only thing that tells the author which module to go fix.
            ref = {"pr_id": pid, "author": pr.author_id, "status": getattr(pr.status, "value", str(pr.status)),
                   "reviewers": list(pr.reviewers), "reviewed": pr.reviewed,
                   "source_branch": pr.source_branch,
                   "ci_passed": bool(getattr(pr, "ci_passed", False)),
                   "blocked_by": str(getattr(pr, "ci_brief", "") or ""),
                   "requested_changes": [
                       str(item)[:1000]
                       for item in (getattr(pr, "requested_changes", []) or [])[-3:]
                       if str(item).strip()
                   ],
                   "review_comments": [
                       {
                           "reviewer": str((item or {}).get("reviewer") or ""),
                           "approve": bool((item or {}).get("approve", False)),
                           "comment": str((item or {}).get("comment") or "")[:1000],
                       }
                       for item in (getattr(pr, "review_comments", []) or [])[-3:]
                       if isinstance(item, dict)
                       and str(item.get("comment") or "").strip()
                   ]}
            pkt.visible_prs.append(ref)
            if aid in pr.reviewers and getattr(pr.status, "value", str(pr.status)) in (
                    "open", "review_requested", "changes_requested"):
                pkt.prs_awaiting_my_review.append(ref)

    # -- sandbox / experiments (own sandbox private; exported = shared) ----
    def _fill_sandbox_experiments(self, pkt, w, aid):
        ss = w.sandbox_system
        sb = ss.sandboxes.get(f"sandbox_{aid}")
        if sb:
            pkt.sandbox_summary = {"sandbox_id": sb.sandbox_id, "status": sb.sandbox_status,
                                   "completed_jobs": len(sb.completed_jobs),
                                   "failed_jobs": len(sb.failed_jobs),
                                   "cached_results": len(sb.cached_result_ids),
                                   "cost_spent": round(sb.cost_spent, 2)}
        for r in ss.visible_results_for(aid):
            ref = {"result_id": r.result_id, "experiment_id": r.experiment_id,
                   "metrics": dict(r.metrics), "logged_to_tracker": r.logged_to_tracker,
                   "reproducibility": r.reproducibility_status}
            pkt.visible_results.append(ref)
            if not r.logged_to_tracker and (sb and r.result_id in sb.cached_result_ids):
                pkt.local_unshared_results.append(ref)
        for eid, e in w.experiments.items():
            if getattr(e, "owner_id", None) == aid:
                pkt.visible_experiments.append({"experiment_id": eid, "title": getattr(e, "title", ""),
                                                "status": getattr(getattr(e, "status", ""), "value", str(getattr(e, "status", "")))})

    # -- meetings (participant only) ---------------------------------------
    def _fill_meetings(self, pkt, w, aid):
        for mid, m in w.meeting_system.meetings.items():
            if aid not in m.participants:
                continue
            pkt.visible_meetings.append({"meeting_id": mid, "type": m.meeting_type, "title": m.title,
                                         "status": getattr(m.status, "value", str(m.status)),
                                         "scheduled_tick": m.scheduled_tick,
                                         "attended": aid in m.attendees})

    # -- protocols (org-public once proposed) ------------------------------
    def _fill_protocols(self, pkt, w, aid):
        for pid, p in w.protocol_registry.protocols.items():
            pkt.visible_protocols.append({"protocol_id": pid, "type": p.protocol_type,
                                          "status": p.adoption_status,
                                          "supporters": list(p.supporters),
                                          "emergence": p.emergence_level})

    # -- budget / payroll (role-gated) -------------------------------------
    def _fill_budget_payroll(self, pkt, w, agent, aid):
        b = w.budget_system.budget
        # founders see full budget; everyone sees pressure/runway summary
        pkt.visible_budget_summary = {"budget_pressure": b.budget_pressure, "runway_days": b.runway_days,
                                      "cost_multiplier": b.cost_multiplier}
        if agent.is_founder:
            pkt.visible_budget_summary.update({"cash_balance": round(b.cash_balance, 1),
                                               "remaining_budget": round(b.remaining_budget, 1)})
        comp = w.budget_system.comp.get(aid)
        payroll = w.budget_system.payroll
        if comp:
            pkt.visible_payroll_summary = {"unpaid_salary": comp.unpaid_salary,
                                           "trust_in_company": round(comp.trust_in_company, 2),
                                           "retention_risk": comp.retention_risk,
                                           "payroll_status": payroll.payroll_status}

    # -- external posts (only ones the agent has read/saved) ---------------
    def _fill_external(self, pkt, w, aid):
        pw = w.personal.get(aid)
        seen_post_ids = set(pw.saved_post_ids) if pw else set()
        # posts referenced by messages this agent has READ also become perceivable
        for m in w.comm.known_messages(aid):
            for a in m.attachments:
                if a.attachment_type == "external_post":
                    seen_post_ids.add(a.object_id)
        # Iterate the community feed (insertion-ordered dict = chronological), newest
        # first, NOT the set itself: set iteration order depends on PYTHONHASHSEED and
        # leaked hash order into candidate generation (the mapper caps posts at [:2]),
        # making whole runs diverge across interpreter hash seeds. Newest-first also
        # keeps fresh external signals perceivable instead of forever shadowed by the
        # two oldest saved posts.
        for pid, p in reversed(list(w.community.posts.items())):
            if pid in seen_post_ids:
                pkt.visible_external_posts.append({"post_id": pid, "author": p.author_id,
                                                   "topic": p.topic, "summary": p.content_summary,
                                                   "stance": getattr(p, "stance", 0.0)})

    # -- search (own search logs) ------------------------------------------
    def _fill_search(self, pkt, w, aid):
        for log in w.search_system.logs[-self.RECENT_EVENT_WINDOW:]:
            if log.agent_id == aid:
                pkt.visible_search_results.append({"search_id": log.search_id, "query": log.query,
                                                   "domain": log.search_domain,
                                                   "results": list(log.retrieved_object_ids)})

    def _fill_private(self, pkt, w, aid):
        pw = w.personal.get(aid)
        if pw:
            pkt.private_workspace_summary = {"notes": len(pw.personal_notes),
                                             "todos": len(pw.private_todos),
                                             "local_files": len(pw.local_files),
                                             "saved_posts": len(pw.saved_post_ids),
                                             "current_focus": pw.current_focus}

    def _fill_events(self, pkt, w, aid):
        for e in w.events[-self.RECENT_EVENT_WINDOW:]:
            if e.get("agent_id") in (aid, None) or aid in (e.get("participants") or []):
                pkt.recent_events.append(e)

    def _fill_reflections(self, pkt, w, aid):
        """Inject reflection-memory so past reflection shapes future decisions (spec §5).
        Also populates the previously-dead memory_relevant_observations field."""
        rm = getattr(w, "reflection_manager", None)
        if rm is None:
            return
        ctx = rm.context_for_decision(aid, w)
        pkt.recent_reflections = ctx.get("recent_reflections", [])
        pkt.lessons_learned = ctx.get("lessons_learned", [])
        pkt.unresolved_needs = ctx.get("unresolved_needs", [])
        pkt.open_wishes = ctx.get("open_wishes", [])
        pkt.open_wish_needs = ctx.get("open_wish_needs", [])
        pkt.memory_relevant_observations = [
            {"kind": "reflection", "text": t} for t in pkt.recent_reflections]
        phase_context = ctx.get("programbench_workflow_state")
        if isinstance(phase_context, dict):
            pkt.memory_relevant_observations.append(
                {"kind": "programbench_workflow_state", **phase_context}
            )
        retrieved = ctx.get("programbench_targeted_public_retrievals")
        if isinstance(retrieved, list) and retrieved:
            pkt.memory_relevant_observations.append(
                {
                    "kind": "programbench_targeted_public_retrievals",
                    "surfaces": retrieved,
                }
            )

    # -- DomainAdapter-protocol compatibility shim (skeleton test) ---------
    def build_perception_from_state(self, *, agent_id: str, state: Any) -> Any:
        raise NotImplementedError("O1 perception is world-driven; use build_perception(agent_id, org_world, tick)")


__all__ = ["OrgPerceptionPacket", "OrgPerceptionAdapter"]
