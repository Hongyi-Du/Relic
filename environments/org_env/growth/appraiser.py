"""GrowthAppraiser (spec §2-§7) — turn an ExecutionResult into credited GrowthSignals.

Rule-based: classifies the result's outcome, where it landed (visibility), how big it
was (workload), which skill/reputation domains it touches, and who gets credit. LLM may
add rationale elsewhere but never the numbers. Signals are buffered on the world and
applied in batch by the GrowthReconciler.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from environments.org_env.coding.profile import action_coding_skill
from environments.org_env.growth.objects import (
    CREDIT_ROLE,
    MAX_TOTAL_CREDIT,
    OUTCOME_SCORE,
    VISIBILITY,
    GrowthSignal,
    workload_weight,
)

# action_type -> (skill_weights m_{e,k}, reputation_weights n_{e,d}); §7 fallback table.
_ACTION_DOMAINS: Dict[str, Tuple[Dict[str, float], Dict[str, float]]] = {
    "edit_repo_file": ({"debugging": 0.6, "rapid_prototyping": 0.4}, {"engineering_execution": 1.0}),
    "commit_patch": ({"debugging": 0.5, "rapid_prototyping": 0.5}, {"engineering_execution": 1.0}),
    "open_pr": ({"debugging": 0.5, "claim_evidence_review": 0.5}, {"engineering_execution": 1.0}),
    "review_pr": ({"claim_evidence_review": 0.5, "debugging": 0.5},
                  {"engineering_execution": 0.6, "reliability_evidence": 0.4}),
    "approve_pr": ({"claim_evidence_review": 0.5, "debugging": 0.5},
                   {"engineering_execution": 0.6, "reliability_evidence": 0.4}),
    "merge_pr": ({"debugging": 0.5, "rapid_prototyping": 0.5}, {"engineering_execution": 1.0}),
    "request_changes": ({"claim_evidence_review": 0.6, "debugging": 0.4},
                        {"reliability_evidence": 0.5, "engineering_execution": 0.5}),
    "review_doc": ({"claim_evidence_review": 0.5, "documentation_quality": 0.5},
                   {"reliability_evidence": 0.4, "documentation_quality": 0.4, "product_judgment": 0.2}),
    "audit_readme_claims": ({"documentation_quality": 0.6, "claim_evidence_review": 0.4},
                            {"documentation_quality": 0.6, "product_judgment": 0.4}),
    "challenge_result": ({"reproducibility_tracking": 0.6, "claim_evidence_review": 0.4},
                         {"reliability_evidence": 1.0}),
    "ask_for_evidence": ({"reproducibility_tracking": 0.6, "claim_evidence_review": 0.4},
                         {"reliability_evidence": 1.0}),
    "request_reproduction": ({"reproducibility_tracking": 0.7, "claim_evidence_review": 0.3},
                             {"reliability_evidence": 1.0}),
    "run_cheap_pilot": ({"eval_design": 0.7, "reproducibility_tracking": 0.3},
                        {"reliability_evidence": 0.6, "engineering_execution": 0.4}),
    "run_eval_stub": ({"eval_design": 0.8, "reproducibility_tracking": 0.2},
                      {"reliability_evidence": 0.6, "engineering_execution": 0.4}),
    "create_eval_stub": ({"eval_design": 0.8, "debugging": 0.2},
                         {"reliability_evidence": 0.5, "engineering_execution": 0.5}),
    "export_result_to_tracker": ({"reproducibility_tracking": 0.8, "eval_design": 0.2},
                                 {"reliability_evidence": 1.0}),
    "create_experiment_tracker": ({"reproducibility_tracking": 0.7, "workflow_design": 0.3},
                                  {"reliability_evidence": 0.7, "workflow_design": 0.3}),
    "update_claim_tracker": ({"claim_evidence_review": 0.6, "debugging": 0.4},
                             {"reliability_evidence": 0.6, "engineering_execution": 0.4}),
    "update_source_tracker": ({"source_credibility": 0.7, "debugging": 0.3}, {"reliability_evidence": 1.0}),
    "create_doc": ({"documentation_quality": 0.6, "workflow_design": 0.4},
                   {"documentation_quality": 0.6, "workflow_design": 0.4}),
    "write_design_note": ({"documentation_quality": 0.5, "workflow_design": 0.5},
                          {"documentation_quality": 0.5, "workflow_design": 0.5}),
    "create_report_quality_checklist": ({"workflow_design": 0.5, "report_quality": 0.5},
                                        {"workflow_design": 0.6, "documentation_quality": 0.4}),
    "create_report_template": ({"report_quality": 0.6, "workflow_design": 0.4},
                               {"workflow_design": 0.5, "documentation_quality": 0.5}),
    "create_onboarding_doc": ({"documentation_quality": 0.6, "customer_research": 0.4},
                              {"documentation_quality": 0.5, "customer_sensing": 0.5}),
    "open_issue": ({"customer_research": 0.5, "claim_evidence_review": 0.5},
                   {"product_judgment": 0.6, "customer_sensing": 0.4}),
    "create_issue": ({"customer_research": 0.5, "claim_evidence_review": 0.5},
                     {"product_judgment": 0.6, "customer_sensing": 0.4}),
    "monitor_customer_feedback": ({"customer_research": 0.8, "documentation_quality": 0.2},
                                  {"customer_sensing": 1.0}),
    "share_external_signal": ({"customer_research": 0.7, "coordination": 0.3},
                              {"customer_sensing": 1.0}),
    "schedule_meeting": ({"coordination": 0.7, "workflow_design": 0.3}, {"coordination_leadership": 1.0}),
    "summarize_decision": ({"coordination": 0.6, "documentation_quality": 0.4},
                           {"coordination_leadership": 0.7, "documentation_quality": 0.3}),
    "assign_action_item": ({"coordination": 0.8, "workflow_design": 0.2}, {"coordination_leadership": 1.0}),
    "coordinate_followup": ({"coordination": 0.8, "workflow_design": 0.2}, {"coordination_leadership": 1.0}),
    "propose_protocol": ({"workflow_design": 0.6, "coordination": 0.4},
                         {"coordination_leadership": 0.6, "workflow_design": 0.4}),
    "approve_proposal": ({"workflow_design": 0.5, "coordination": 0.5},
                         {"coordination_leadership": 0.6, "workflow_design": 0.4}),
    "create_runway_update": ({"cost_governance": 0.8, "workflow_design": 0.2}, {"cost_governance": 1.0}),
    "create_cost_ledger": ({"cost_governance": 0.8, "workflow_design": 0.2}, {"cost_governance": 1.0}),
    "run_launch_readiness_check": ({"workflow_design": 0.5, "coordination": 0.5},
                                   {"coordination_leadership": 0.6, "workflow_design": 0.4}),
    "create_release_candidate": ({"workflow_design": 0.5, "coordination": 0.5},
                                 {"coordination_leadership": 0.6, "workflow_design": 0.4}),
    "approve_release_candidate": ({"coordination": 0.6, "workflow_design": 0.4},
                                  {"coordination_leadership": 1.0}),
    "publish_product_release": ({"coordination": 0.6, "workflow_design": 0.4},
                                {"coordination_leadership": 1.0}),
    "use_tool": ({"workflow_design": 0.5, "reproducibility_tracking": 0.5},
                 {"reliability_evidence": 0.5, "workflow_design": 0.5}),
}

# product-artifact purpose -> skill/reputation nudge (refines the action mapping).
# Keys MUST match environments.org_env.product.objects.artifact_purpose() outputs.
_PURPOSE_DOMAINS: Dict[str, Tuple[Dict[str, float], Dict[str, float]]] = {
    "eval": ({"eval_design": 1.0}, {"reliability_evidence": 1.0}),
    "claim_tracker": ({"claim_evidence_review": 1.0}, {"reliability_evidence": 1.0}),
    "source_tracker": ({"source_credibility": 1.0}, {"reliability_evidence": 1.0}),
    "report_quality": ({"report_quality": 1.0}, {"workflow_design": 1.0}),
    "report_writer": ({"report_quality": 1.0}, {"workflow_design": 1.0}),
    "report_template": ({"report_quality": 1.0}, {"workflow_design": 1.0}),
    "readme": ({"documentation_quality": 1.0}, {"documentation_quality": 0.6, "product_judgment": 0.4}),
    "product_design": ({"documentation_quality": 0.5, "workflow_design": 0.5},
                       {"documentation_quality": 0.5, "product_judgment": 0.5}),
    "research_loop": ({"debugging": 0.5, "rapid_prototyping": 0.5}, {"engineering_execution": 1.0}),
    "onboarding": ({"customer_research": 1.0}, {"customer_sensing": 1.0}),
    "design_note": ({"workflow_design": 1.0}, {"workflow_design": 1.0}),
    "demo": ({"rapid_prototyping": 1.0}, {"engineering_execution": 1.0}),
}


def _norm(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    return {k: v / s for k, v in d.items()} if s > 0 else d


class GrowthAppraiser:
    def collect(self, result: Any, world: Any, actor_id: str) -> List[GrowthSignal]:
        """Append GrowthSignals for this result to world._growth_signals; return them."""
        sink = world.__dict__.setdefault("_growth_signals", [])
        sigs = self._signals(result, world, actor_id)
        sink.extend(sigs)
        return sigs

    def _signals(self, result, world, actor_id) -> List[GrowthSignal]:
        outcome_key, outcome = self._outcome(result)
        if outcome == 0.0:                       # nothing growth-relevant happened
            return []
        skill_w, rep_w = self._domains(result, world)
        if not skill_w and not rep_w:
            return []
        vis = self._visibility(result)
        wl = workload_weight(self._workload(result))
        tick = int(getattr(world, "world_tick", 0) or 0)
        eid = (result.events[0].get("event_id") if result.events else None) or \
            f"{result.action_type}@t{tick}"
        out: List[GrowthSignal] = []
        # primary actor (credit 1.0)
        out.append(GrowthSignal(
            agent_id=actor_id, outcome=outcome, visibility=vis, workload=wl,
            credit=CREDIT_ROLE["primary_actor"], skill_weights=skill_w, rep_weights=rep_w,
            source_event_id=str(eid), source_action_id=result.action_id, tick=tick,
            reason=f"{result.action_type} -> {outcome_key}"))
        # reviewers credited on a merge (their earlier approval paid off)
        if outcome_key == "pr_merged":
            for r in self._merge_reviewers(result, world, actor_id):
                out.append(GrowthSignal(
                    agent_id=r, outcome=outcome * 0.6, visibility=vis, workload=min(wl, 1.0),
                    credit=CREDIT_ROLE["approving_reviewer"], skill_weights=skill_w,
                    rep_weights=rep_w, source_event_id=str(eid), source_action_id=result.action_id,
                    tick=tick, reason="approved a PR that merged"))
        # cap total credit (§6)
        tot = sum(s.credit for s in out)
        if tot > MAX_TOTAL_CREDIT:
            scale = MAX_TOTAL_CREDIT / tot
            for s in out:
                s.credit *= scale
        return out

    def _outcome(self, result) -> Tuple[str, float]:
        if not result.success:
            reason = (result.failure_reason or "").lower()
            if "duplicate" in reason:
                return ("patch_rejected_duplicate", OUTCOME_SCORE["patch_rejected_duplicate"])
            if "patch_rejected" in reason or "invalid" in reason or "fake" in reason:
                return ("patch_rejected_invalid", OUTCOME_SCORE["patch_rejected_invalid"])
            return ("action_failed", -0.15)
        subs = {e.get("subtype") for e in result.events}
        types = {e.get("type") for e in result.events}
        if "merged_to_mainline" in subs or "pr_merged" in subs:
            return ("pr_merged", OUTCOME_SCORE["pr_merged"])
        if "published" in subs or "release_published" in subs:
            return ("release_gate_passed", OUTCOME_SCORE["release_gate_passed"])
        if "gate_blocked" in subs or "release_blocked" in subs:
            return ("release_gate_blocked", OUTCOME_SCORE["release_gate_blocked"])
        if "pr_reviewed" in subs or "pr_approved" in subs:
            return ("pr_approved", OUTCOME_SCORE["pr_approved"])
        if "changes_requested" in subs:
            return ("pr_changes_requested", OUTCOME_SCORE["pr_changes_requested"])
        if "patch_applied" in subs:
            return ("patch_applied", OUTCOME_SCORE["patch_applied"])
        if "merged" in subs:                                 # task moved to merged
            return ("task_merged", OUTCOME_SCORE["task_merged"])
        if "protocol_use_event" in types or "protocol_enforcement_event" in types:
            return ("protocol_used", OUTCOME_SCORE["protocol_used"])
        if result.action_type in ("challenge_result", "ask_for_evidence", "request_reproduction"):
            return ("valid_challenge", OUTCOME_SCORE["valid_challenge"])
        # a generic successful productive action -> small positive; coding actions
        # (ACTION_CODING_SKILL) count even without an _ACTION_DOMAINS row, so the
        # coding skills that outcome gates consume are exercisable at all.
        if result.action_type in _ACTION_DOMAINS or \
                action_coding_skill(result.action_type) is not None:
            return ("productive_action", 0.25)
        return ("trivial", 0.0)

    def _domains(self, result, world) -> Tuple[Dict[str, float], Dict[str, float]]:
        sk, rp = _ACTION_DOMAINS.get(result.action_type, ({}, {}))
        sk, rp = dict(sk), dict(rp)
        # Bridge the coding-skill taxonomy (coding/profile.py::ACTION_CODING_SKILL is
        # the single source of truth) into the growth signal: the two taxonomies are
        # independent classifications of the same action, so absent other information
        # the coding channel gets an equal share of the skill mass (max-entropy split;
        # full share when the action has no org-skill row). Without this bridge, skills
        # that outcome gates consume via agent.skill() — e.g. test_writing gating
        # commit quality flags — are growth-dead and their gates permanently frozen.
        coding_skill = action_coding_skill(result.action_type)
        if coding_skill is not None:
            sk[coding_skill] = sk.get(coding_skill, 0.0) + (sum(sk.values()) or 1.0)
        # refine by the product-artifact purpose of what was touched
        try:
            from environments.org_env.product.objects import artifact_purpose
            arts = getattr(world, "product_artifacts", {}) or {}
            for oid in (result.modified_objects + result.created_objects):
                a = arts.get(oid)
                if a is None:
                    continue
                purpose = artifact_purpose(getattr(a, "linked_file_path", None) or oid)
                psk, prp = _PURPOSE_DOMAINS.get(purpose, ({}, {}))
                for k, v in psk.items():
                    sk[k] = sk.get(k, 0.0) + v
                for d, v in prp.items():
                    rp[d] = rp.get(d, 0.0) + v
        except Exception:
            pass
        return _norm(sk), _norm(rp)

    def _visibility(self, result) -> float:
        subs = {e.get("subtype") for e in result.events}
        types = {e.get("type") for e in result.events}
        if subs & {"published", "release_published", "gate_blocked", "release_blocked"} \
                or "release_event" in types:
            return VISIBILITY["company_wide"]
        if subs & {"merged_to_mainline", "pr_merged"}:
            return VISIBILITY["merged_mainline"]
        if subs & {"pr_opened", "pr_reviewed", "pr_approved", "ci_run", "changes_requested"} \
                or "repo_event" in types:
            return VISIBILITY["pr_review_ci"]
        if any(str(o).startswith(("issue", "task")) for o in result.modified_objects):
            return VISIBILITY["shared_object"]
        if result.messages:
            return VISIBILITY["team_channel"]
        if "search_event" in types or "background_job_event" in types:
            return VISIBILITY["private"]
        return VISIBILITY["team_channel"]

    def _workload(self, result) -> float:
        sd = getattr(result, "state_delta", {}) or {}
        pid = sd.get("patch_id")
        patch = None
        if pid and hasattr(result, "agent_id"):
            patch = None       # patch object lives on world; appraiser keeps it light
        # prefer the action's duration (already workload-derived in v7) else a default
        dur = sd.get("duration")
        if isinstance(dur, (int, float)) and dur > 0:
            return float(dur)
        return 2.0

    def _merge_reviewers(self, result, world, actor_id) -> List[str]:
        pr_id = next((e.get("pr_id") for e in result.events if e.get("pr_id")), None)
        repo = getattr(world, "repo_system", None)
        if not pr_id or repo is None:
            return []
        pr = getattr(repo.repo, "pull_requests", {}).get(pr_id) if hasattr(repo, "repo") else None
        revs = list(getattr(pr, "reviewers", []) or []) if pr is not None else []
        return [r for r in revs if r != actor_id and r in getattr(world, "agents", {})]


__all__ = ["GrowthAppraiser"]
