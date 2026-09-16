"""StateReconciler (Internal Pipeline Completion spec #1).

Keeps global product state consistent: when a capability is built, a task completes,
or a PR merges, the corresponding known_gap / issue / product-readiness state must
follow — no "task done but gap still active", no "resolved capability still listed
as an active gap". Runs end-of-day, after a PR merge, after a task update, before a
release gate, and at final-snapshot time.
"""
from __future__ import annotations

from typing import Any, Dict, List

from environments.org_env.product.objects import artifact_purpose

_DONE_TASK = {"done", "merged", "released"}
_INPROGRESS_TASK = {"in_progress", "review", "implementation_done", "review_pending"}


def _sv(x) -> str:
    return getattr(getattr(x, "status", None), "value", str(getattr(x, "status", "")))


class StateReconciler:
    def reconcile(self, world: Any, *, reason: str = "", tick: int = None) -> Dict[str, Any]:
        tick = int(getattr(world, "world_tick", 0) or 0) if tick is None else tick
        warnings: List[str] = []
        self._reconcile_gaps(world, tick, warnings)
        self._reconcile_issues(world, tick, warnings)
        self._sync_surface(world, tick)        # v8h P0: keep artifact/product summaries honest
        self._close_stale_blockers(world, tick)  # v8h P0: drop release-blocker issues whose gate passes
        readiness = self._derive_readiness(world)
        world.product_readiness = readiness
        world.institution_context = self._build_institution_context(world)
        from environments.org_env.experiments.ablations import CAPABILITY_MEMORY, mechanism_disabled
        if mechanism_disabled(world, CAPABILITY_MEMORY):
            world.company_skills = []          # ablation: no persistent capability ledger
        else:
            from environments.org_env.backend.skills import detect_company_skills
            world.company_skills = detect_company_skills(world)     # spec #8
        world._reconcile_warnings = warnings
        return {"reason": reason, "tick": tick, "warnings": warnings, "readiness": readiness}

    @staticmethod
    def _build_institution_context(world) -> Dict[str, Any]:
        """spec #4: the live set of adopted protocols + active tools governing behaviour,
        with their use/enforcement counts (drives the report + company-skill detector)."""
        pm = getattr(world, "proposal_manager", None)
        protocols = [{
            "protocol_id": s.protocol_id, "name": s.name, "status": s.status,
            "revision": getattr(s, "revision", 0), "use_count": getattr(s, "use_count", 0),
            "enforcement_count": getattr(s, "enforcement_count", 0),
            "violation_count": getattr(s, "violation_count", 0),
        } for s in (pm.protocol_specs.values() if pm else []) if s.status == "adopted"]
        tools = [{"tool_id": t.tool_id, "name": t.name, "status": t.status}
                 for t in (pm.tools.values() if pm else []) if t.status == "active"]
        return {"protocols": protocols, "tools": tools,
                "active_protocol_count": len(protocols), "active_tool_count": len(tools)}

    # -- known gaps --------------------------------------------------------- #
    def _reconcile_gaps(self, world, tick, warnings) -> None:
        gaps = getattr(world, "known_gaps", {}) or {}
        arts = getattr(world, "product_artifacts", {}) or {}
        for g in gaps.values():
            art = arts.get(g.linked_artifact_id) if g.linked_artifact_id else None
            if art is None:                       # re-link: the resolving artifact may have
                art = self._relink(world, g)       # been created after seed (e.g. quality gate)
                if art is not None:
                    g.linked_artifact_id = art.artifact_id
            built = self._capability_built(g, art)
            merged = bool(art and int(getattr(art, "mainline_revision", 0) or 0) > 0)
            prev = g.status
            if built and merged:
                new = "resolved"
            elif built:
                new = "mitigated"
            elif prev in ("resolved", "mitigated"):
                new = "regressed"
            else:
                new = "active"
            if new != prev:
                g.status = new
                g.last_update_tick = tick
                g.evidence.append({"tick": tick, "from": prev, "to": new})
                if art is not None and new in ("mitigated", "resolved"):
                    self._credit(g, art)
                if new == "resolved":
                    # v8g P0: a resolved gap must not linger on the product surface
                    # (README change-list / product summary) — keep the demo state honest.
                    if art is not None and g.description in (getattr(art, "known_gaps", []) or []):
                        art.known_gaps.remove(g.description)
                    ps = getattr(world, "product", None)
                    if ps is not None and g.description in (getattr(ps, "known_systemic_issues", []) or []):
                        ps.known_systemic_issues.remove(g.description)
            # invariant check for the report
            if new in ("active", "regressed") and built and merged:
                warnings.append(f"{g.gap_id}: capability built+merged but gap still {new}")

    # -- v8h P0: surface text sync (no stale "incomplete" once a gap is resolved) --------
    def _sync_surface(self, world, tick) -> None:
        """Rebuild artifact + product summaries from CURRENT capabilities + active gaps, so a
        resolved capability never still reads as 'does not score credibility / no evidence gate /
        overpromises'. Only overwrites once an artifact has real built capabilities (un-built
        artifacts keep their seed text)."""
        arts = getattr(world, "product_artifacts", {}) or {}
        for a in arts.values():
            if getattr(a, "artifact_type", "") == "issue":
                continue
            caps = [str(c) for c in (getattr(a, "capabilities", []) or [])]
            gaps = [str(g) for g in (getattr(a, "known_gaps", []) or [])]
            if not caps:
                continue
            name = getattr(a, "title", None) or getattr(a, "linked_file_path", None) or a.artifact_id
            merged = int(getattr(a, "mainline_revision", 0) or 0) > 0
            s = f"{name}: now provides {', '.join(caps[:5])}."
            s += (f" Open gaps: {'; '.join(gaps[:3])}." if gaps
                  else (" Merged to mainline; no open gaps." if merged else " No open gaps (pending merge)."))
            a.summary = s
        ps = getattr(world, "product", None)
        if ps is not None:
            active = [str(g) for g in (getattr(ps, "known_systemic_issues", []) or [])]
            stage = getattr(ps, "stage", "prototype")
            name = getattr(ps, "name", None) or "Current OSS workload"
            if active:
                ps.summary = (
                    f"{name} ({str(stage).replace('_', ' ')}). "
                    f"Remaining gaps: {'; '.join(active[:4])}."
                )
            else:
                ps.summary = (
                    f"{name} ({str(stage).replace('_', ' ')}). "
                    "No critical gaps remain."
                )

    def _close_stale_blockers(self, world, tick) -> None:
        """v8h P0: a rel_blocker_* issue whose gate now PASSES (and whose blocker task is done)
        must not linger open after release_ready — close it."""
        from environments.org_env.backend.repo.release import evaluate_release_gates
        from environments.org_env.backend.entities import COMPLETED_TASK_STATUSES
        arts = getattr(world, "product_artifacts", {}) or {}
        rcs = getattr(getattr(world, "repo_system", None), "repo", None)
        rcs = list(getattr(rcs, "release_candidates", {}).values()) if rcs is not None else []
        rc = next((r for r in rcs if getattr(r, "status", "") in ("approved", "released", "under_review")), None) \
            or (rcs[-1] if rcs else None)
        if rc is None:
            return
        try:
            results = {r["gate"]: r["passed"] for r in evaluate_release_gates(world, rc)}
        except Exception:
            return
        for iid, a in list(arts.items()):
            if not str(iid).startswith("rel_blocker_") or getattr(a, "status", "") == "resolved":
                continue
            gate = getattr(a, "linked_gate", None) or str(iid).replace("rel_blocker_", "")
            t = (getattr(world, "tasks", {}) or {}).get(f"task_rel_blocker_{gate}")
            task_done = t is None or getattr(t.status, "value", str(t.status)) in COMPLETED_TASK_STATUSES
            if results.get(gate, False) and task_done:
                a.status = "resolved"
                a.issue_status = "resolved"
                a.updated_at_tick = tick
                ps = getattr(world, "product", None)
                if ps is not None and iid in (getattr(ps, "open_issue_ids", []) or []):
                    ps.open_issue_ids.remove(iid)

    def _relink(self, world, g):
        from environments.org_env.product.known_gap import infer_gap_meta
        purpose, _, _ = infer_gap_meta(g.description)
        if not purpose:
            return None
        for a in (getattr(world, "product_artifacts", {}) or {}).values():
            if getattr(a, "artifact_type", "") == "issue":
                continue
            key = getattr(a, "linked_file_path", "") or getattr(a, "artifact_id", "")
            if artifact_purpose(key) == purpose:
                return a
        return None

    @staticmethod
    def _capability_built(g, art) -> bool:
        if art is None:
            return False
        caps = [str(c).lower() for c in (getattr(art, "capabilities", []) or [])]
        revised = int(getattr(art, "revision", 0) or 0) > 0
        cap = (g.required_capability or "").strip().lower()
        # per-artifact gap whose source string a patch already removed -> built
        if g.scope == "artifact":
            listed = [str(x).strip().lower() for x in (getattr(art, "known_gaps", []) or [])]
            if revised and g.description.strip().lower() not in listed:
                return True
        if cap:
            return any(cap in c for c in caps)
        return revised                              # no-capability gap closes on a real revision

    @staticmethod
    def _credit(g, art) -> None:
        for pid in (getattr(art, "patch_history_ids", []) or [])[-3:]:
            if pid not in g.resolved_by_patch_ids:
                g.resolved_by_patch_ids.append(pid)
        for tid in (getattr(art, "linked_task_ids", []) or []):
            if tid not in g.resolved_by_task_ids:
                g.resolved_by_task_ids.append(tid)
        for prid in (getattr(art, "linked_pr_ids", []) or []):
            if prid not in g.resolved_by_pr_ids:
                g.resolved_by_pr_ids.append(prid)

    # -- issues ------------------------------------------------------------- #
    def _reconcile_issues(self, world, tick, warnings) -> None:
        arts = getattr(world, "product_artifacts", {}) or {}
        tasks = getattr(world, "tasks", {}) or {}
        gaps = list((getattr(world, "known_gaps", {}) or {}).values())

        def _linked_tasks(issue_id, issue_obj):
            tids = set(getattr(issue_obj, "linked_task_ids", []) or [])
            rt = getattr(issue_obj, "related_task_id", None)
            if rt:
                tids.add(rt)
            for t in tasks.values():
                if issue_id in (getattr(t, "linked_issues", []) or []):
                    tids.add(t.task_id)
            return [tasks[t] for t in tids if t in tasks]

        def _related_gaps(issue_obj):
            # #1 follow-up: tie an issue to the gap(s) that share its concern, so a
            # mitigated/resolved gap moves the issue even without a directly linked task.
            from environments.org_env.product.known_gap import infer_gap_meta
            text = f"{getattr(issue_obj, 'title', '')} {getattr(issue_obj, 'problem', '')} " \
                   f"{getattr(issue_obj, 'summary', '')} {getattr(issue_obj, 'description', '')}"
            purpose, _, _ = infer_gap_meta(text)
            if not purpose:
                return []
            return [g for g in gaps if infer_gap_meta(g.description)[0] == purpose]

        def _classify(issue_id, issue_obj, set_status, get_status):
            lt = _linked_tasks(issue_id, issue_obj)
            rel_gaps = _related_gaps(issue_obj)
            if not lt and not rel_gaps:
                return
            done = [t for t in lt if _sv(t) in _DONE_TASK]
            inprog = [t for t in lt if _sv(t) in _INPROGRESS_TASK]
            gap_resolved = [g for g in rel_gaps if g.status == "resolved"]
            gap_mitig = [g for g in rel_gaps if g.status == "mitigated"]
            gap_active = [g for g in rel_gaps if g.status in ("active", "regressed")]
            pr_ids, patch_ids, task_ids = [], [], []
            merged_arts = []
            for t in done + inprog:
                for aid in (getattr(t, "linked_artifacts", []) or []):
                    a = arts.get(aid)
                    if a and int(getattr(a, "mainline_revision", 0) or 0) > 0:
                        merged_arts.append(aid)
                        pr_ids += list(getattr(a, "linked_pr_ids", []) or [])
                        patch_ids += list(getattr(a, "patch_history_ids", []) or [])
            for g in gap_resolved + gap_mitig:        # carry the gap's resolution evidence
                pr_ids += list(g.resolved_by_pr_ids); patch_ids += list(g.resolved_by_patch_ids)
                task_ids += list(g.resolved_by_task_ids)
            prev = get_status()
            # fully resolved: a relevant gap resolved, OR a linked task merged
            if gap_resolved or (done and merged_arts):
                new = "resolved"; task_ids += [t.task_id for t in done]
            # partial: relevant gap mitigated, or implementation done but not merged
            elif gap_mitig or done:
                new = "partially_resolved"; task_ids += [t.task_id for t in done]
            elif inprog or (gap_active and (patch_ids or task_ids)):
                new = "in_progress"
            elif prev in ("resolved", "partially_resolved"):
                new = "reopened"
            else:
                new = prev or "open"
            if new != prev:
                set_status(new, sorted(set(task_ids)), sorted(set(pr_ids)), sorted(set(patch_ids)))
            if new == "resolved" and not (task_ids or pr_ids or patch_ids):
                warnings.append(f"{issue_id}: marked resolved without evidence links")

        for iid, iss in (getattr(world, "issues", {}) or {}).items():
            def _set(new, tids, prids, pids, _i=iss, _t=tick):
                _i.status = new; _i.last_status_tick = _t
                _i.resolved_by_task_ids = tids; _i.resolved_by_pr_ids = prids
                _i.resolved_by_patch_ids = pids
                if new == "resolved" and getattr(_i, "resolved_tick", None) is None:
                    _i.resolved_tick = _t
            _classify(iid, iss, _set, lambda _i=iss: getattr(_i, "status", "open"))

        # product-issue artifacts: rich lifecycle on `issue_status`; `status` stays
        # open/closed for compat (closed only when fully resolved).
        for aid, a in arts.items():
            if getattr(a, "artifact_type", "") != "issue":
                continue
            def _set(new, tids, prids, pids, _a=a, _t=tick):
                _a.issue_status = new
                _a.resolved_by_task_ids = tids; _a.resolved_by_pr_ids = prids
                _a.resolved_by_patch_ids = pids
                _a.status = "closed" if new == "resolved" else "open"
                _a.updated_at_tick = _t
            _classify(aid, a, _set, lambda _a=a: getattr(_a, "issue_status", "open"))

    # -- product readiness (derived, never manually set) -------------------- #
    def _derive_readiness(self, world) -> Dict[str, Any]:
        gaps = list((getattr(world, "known_gaps", {}) or {}).values())
        arts = getattr(world, "product_artifacts", {}) or {}

        def _active_purpose(p):
            return any(artifact_purpose(getattr(a, "linked_file_path", "") or a.artifact_id) == p
                       and getattr(a, "status", "") not in ("deprecated", "closed")
                       for a in arts.values())

        rs = getattr(world, "repo_system", None)
        prs = (rs.repo.pull_requests.values() if rs else [])
        merged = [p for p in prs if _sv(p) == "merged"]
        releases = list((rs.repo.releases.values() if rs else []))
        active_critical = [g for g in gaps if g.status in ("active", "regressed") and g.critical]

        def _gap_ok(*keys):
            for g in gaps:
                if any(k in g.description.lower() for k in keys):
                    if g.status in ("active", "regressed"):
                        return False
            return True

        # #4: release_ready (a beta can ship) must NOT imply "all gaps cleared". Surface the FULL open
        # systemic-gap count + the OSS objective split (tested-fix vs untested) as separate signals so
        # readiness and known_gaps can't silently contradict each other.
        open_gaps_all = sum(1 for g in gaps if g.status in ("active", "regressed"))
        oss_hidden = getattr(world, "_oss_release_hidden", None) or {}
        return {
            "demo_path_exists": _active_purpose("onboarding") or _active_purpose("demo"),
            "critical_gaps_remaining": len(active_critical),
            "open_systemic_gaps": open_gaps_all,
            "gaps_cleared": open_gaps_all == 0,          # explicit: release_ready does NOT imply this
            "tested_issue_fix_rate": oss_hidden.get("issue_fix_rate"),
            "hidden_test_pass_rate": oss_hidden.get("pass_rate"),
            "release_candidate_ready": len(active_critical) == 0 and len(merged) >= 2,
            "release_ready": bool(releases),            # a beta can ship even with open (non-critical) gaps
            "external_claim_safe": _gap_ok("overpromis", "overclaim") and _gap_ok("evidence link", "claim-evidence"),
            "merged_pr_count": len(merged),
        }


__all__ = ["StateReconciler"]
