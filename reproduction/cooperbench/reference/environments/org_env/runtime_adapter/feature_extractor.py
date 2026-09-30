"""OrgFeatureExtractor — (candidate, perception, world) -> OrgFeatures (DESIGN
env_org ?48 + core ?32.4, O1 ?8).

OrgFeatures is the org-domain feature vector (task / skill / social / cost / repo /
budget / protocol / external dims). The OrgPolicy (O1 ?9) dots these with
profile-derived weights so the persona *actually* changes the chosen action.

This module also hosts OrgEventAppraisal (kept here for the DomainAdapter import
surface); the richer appraisal logic lives in appraisal.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

from environments.org_env.backend.actions import action_category

# action_type -> the skill that gates its success (for skill_match / failure_risk).
ACTION_SKILL = {
    "work_on_task": "core_coding", "debug_code": "debugging", "commit_changes": "core_coding",
    "edit_file": "core_coding", "edit_repo_file": "core_coding",
    "open_pr": "core_coding", "run_experiment": "experimental_design",
    "run_cheap_pilot": "experimental_design", "run_paper_baseline": "experimental_design",
    "review_pr": "review_quality", "review_doc": "review_quality", "review_result": "review_quality",
    "request_changes": "review_quality", "request_doc_changes": "claim_wording",
    "create_doc": "documentation", "edit_doc": "documentation", "publish_doc": "documentation",
    "create_experiment_tracker": "experiment_logging", "update_experiment_tracker": "experiment_logging",
    "export_result_to_tracker": "reproducibility_tracking", "save_result": "reproducibility_tracking",
    "create_cost_ledger": "cost_logging", "add_cost_ledger_entry": "cost_logging",
    "create_claim_evidence_table": "claim_evidence_review", "add_claim_evidence_row": "claim_evidence_review",
    "create_workflow_artifact": "artifact_design", "create_customer_triage_sheet": "customer_triage",
    "create_review_checklist": "review_quality",
    "share_external_post": "external_community_sensing", "read_feed": "external_community_sensing",
    "monitor_customer_feedback": "customer_sense", "reply_customer": "customer_sense",
    "post_company_update": "public_narrative", "schedule_meeting": "meeting_facilitation",
    "record_meeting_notes": "documentation", "propose_protocol": "protocol_design",
    "present_report": "public_narrative",
}

# action_type -> role best-suited to it (role_affinity).
ACTION_ROLE = {
    "run_experiment": "reliability", "run_cheap_pilot": "fast_engineer", "commit_changes": "fast_engineer",
    "review_pr": "reliability", "export_result_to_tracker": "reliability",
    "create_experiment_tracker": "reliability", "review_doc": "editorial",
    "request_doc_changes": "editorial", "share_external_post": "community",
    "create_customer_triage_sheet": "community", "monitor_customer_feedback": "community",
    "create_workflow_artifact": "artifact_design", "create_doc": "external_voice",
    "publish_doc": "external_voice", "propose_protocol": "cofounder",
    "post_company_update": "founder", "schedule_meeting": "founder",
}


@dataclass
class OrgFeatures:
    # ?8.1 task / progress
    progress_gain: float = 0.0
    deadline_urgency: float = 0.0
    task_priority: float = 0.0
    blocker_resolution: float = 0.0
    dependency_unlock: float = 0.0
    demo_relevance: float = 0.0
    customer_relevance: float = 0.0
    # ?8.2 skill fit
    skill_match: float = 0.0
    role_affinity: float = 0.0
    learning_gain: float = 0.0
    failure_risk_from_low_skill: float = 0.0
    # ?8.3 social / coordination
    coordination_gain: float = 0.0
    clarity_gain: float = 0.0
    trust_gain: float = 0.0
    trust_risk: float = 0.0
    conflict_risk: float = 0.0
    visibility_gain: float = 0.0
    reputation_gain: float = 0.0
    reputation_risk: float = 0.0
    # ?8.4 cost / workload
    attention_cost: float = 0.0
    fatigue_delta: float = 0.0
    stress_delta: float = 0.0
    context_switch_cost: float = 0.0
    overtime_penalty: float = 0.0
    weekend_penalty: float = 0.0
    burnout_risk_delta: float = 0.0
    # ?8.5 repo / experiment
    repo_health_gain: float = 0.0
    technical_debt_risk: float = 0.0
    review_quality_gain: float = 0.0
    reproducibility_gain: float = 0.0
    untracked_result_risk: float = 0.0
    claim_evidence_gain: float = 0.0
    # ?8.6 budget / payroll
    api_cost: float = 0.0
    compute_cost: float = 0.0
    budget_cost: float = 0.0
    runway_risk_delta: float = 0.0
    payroll_trust_gain: float = 0.0
    retention_risk_delta: float = 0.0
    # ?8.7 protocol
    protocol_creation_potential: float = 0.0
    protocol_use_potential: float = 0.0
    protocol_violation_risk: float = 0.0
    norm_enforcement_gain: float = 0.0
    institutional_memory_gain: float = 0.0
    # governance (proposal approval — persona decides; §13.x)
    proposal_endorsement: float = 0.0   # pull toward approving (rises with usefulness/score)
    proposal_skepticism: float = 0.0    # pull toward rejecting/changes (rises with risk/low use)
    # ?8.8 external community
    external_signal_value: float = 0.0
    customer_pressure: float = 0.0
    expert_advice_value: float = 0.0
    competitor_pressure: float = 0.0
    recruiting_pressure: float = 0.0
    public_reputation_effect: float = 0.0
    # O1.6 work-state + routine (?11-?12)
    routine_prior: float = 0.0          # set by RoutineScheduler (phase prior)
    recovery_value: float = 0.0         # rest/sleep value rises with fatigue/burnout
    capacity_strain: float = 0.0        # heavy work under low attention (penalty)
    org_grievance_pull: float = 0.0     # ask_payroll/consider_offer under low trust/comp stress

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def names(cls):
        return tuple(f.name for f in fields(cls))


class OrgFeatureExtractor:
    def extract(self, candidate: Any, perception: Any, org_world: Any) -> OrgFeatures:
        w = org_world
        agent = w.agents[perception.agent_id]
        at = candidate.action_type
        cat = action_category(at)
        params = candidate.parameters or {}
        f = OrgFeatures()

        # -- cost (from category hint + clock overtime/weekend) ------------
        hint = w.time  # clock for overtime/weekend
        clk = hint.clock
        from environments.org_env.backend.actions import CATEGORY_COST_HINT
        ch = CATEGORY_COST_HINT.get(cat, {})
        f.attention_cost = max(0.0, ch.get("attention", 0.06))
        f.fatigue_delta = max(0.0, ch.get("fatigue", 0.02))
        f.stress_delta = max(0.0, ch.get("stress", 0.01))
        from environments.org_env.cooperbench.work_schedule import compressed_schedule_enabled
        compressed_schedule = compressed_schedule_enabled(w)
        if not compressed_schedule and (clk.is_after_hours or clk.is_late_night):
            f.overtime_penalty = 0.4
            f.burnout_risk_delta += 0.2
        if not compressed_schedule and clk.is_weekend:
            f.weekend_penalty = 0.5
            f.burnout_risk_delta += 0.2
        # context switch: leaving an assigned/focus task to do something else
        if cat in ("comm", "search", "bridge") and perception.assigned_tasks:
            f.context_switch_cost = 0.3

        # -- skill fit -----------------------------------------------------
        skill_name = ACTION_SKILL.get(at)
        if skill_name is not None:
            sm = agent.skill(skill_name, 0.3)
            f.skill_match = sm
            f.failure_risk_from_low_skill = max(0.0, 0.8 - sm)
            f.learning_gain = 0.3 * (1.0 - sm)
        role_for = ACTION_ROLE.get(at)
        if role_for is not None:
            f.role_affinity = 1.0 if agent.role == role_for else 0.2

        # -- category / action-specific semantics --------------------------
        self._semantics(f, at, cat, params, perception, agent, w)
        # -- O1.6 work-state effects (?12) ---------------------------------
        self._work_state_effects(f, at, cat, perception)
        return f

    def _work_state_effects(self, f, at, cat, perception) -> None:
        """Current attention/fatigue/stress/burnout/comp-stress/trust shift the
        utility of recovery / heavy-work / grievance actions (spec ?12)."""
        ss = perception.self_state or {}
        attn = float(ss.get("attention", 1.0))
        fatigue = float(ss.get("fatigue", 0.0))
        burnout = float(ss.get("burnout_risk", 0.0))
        comp_stress = float(ss.get("compensation_stress", 0.0))
        trust = float(ss.get("trust_in_company", 0.7))
        retention = float(ss.get("retention_risk", 0.0))
        if at in ("rest_offline", "sleep"):
            f.recovery_value = min(1.5, fatigue + burnout + max(0.0, 0.6 - attn))
        if cat in ("work", "repo", "sandbox", "doc", "artifact") and attn < 0.4:
            f.capacity_strain = (0.4 - attn) + 0.5 * fatigue
        if at in ("ask_about_payroll", "consider_external_offer", "resign"):
            f.org_grievance_pull = comp_stress + max(0.0, 0.7 - trust) + retention

    def _semantics(self, f, at, cat, params, perception, agent, w):
        clk = w.time.clock
        # task / work
        if at in (
            "work_on_task",
            "pick_task",
            "debug_code",
            "commit_changes",
            "edit_file",
            "edit_repo_file",
        ):
            f.progress_gain = 0.7 if at in ("work_on_task", "commit_changes") else 0.4
            tid = params.get("task_id")
            t = w.tasks.get(tid) if tid else None
            if t is not None:
                f.task_priority = getattr(t, "priority", 3) / 5.0
                f.deadline_urgency = min(1.0, getattr(t, "priority", 3) / 5.0)
            if at in ("commit_changes", "edit_file", "edit_repo_file"):
                f.repo_health_gain = 0.3
                f.technical_debt_risk = 0.5 if agent.skill("test_writing", 0.3) < 0.4 else 0.2
            if "demo" in (params.get("module", "") + str(params.get("title", ""))).lower():
                f.demo_relevance = 0.7
        if at == "pick_task":
            f.dependency_unlock = 0.3
        if at in ("update_task_status", "update_tracker", "inspect_task_board"):
            f.progress_gain = 0.2
            f.institutional_memory_gain = 0.2
        if at == "assign_task_owner":
            f.coordination_gain = 0.5
            f.institutional_memory_gain = 0.3
            f.protocol_creation_potential = 0.4   # ownership norm seed

        # communication
        if cat == "comm":
            f.coordination_gain = 0.5
            f.clarity_gain = 0.3 if at in ("ask_for_clarification", "send_async_update") else 0.1
            f.visibility_gain = 0.3
            if at in ("escalate_incident",):
                f.conflict_risk = 0.4
            if at in ("share_external_post", "share_search_result", "share_json_result",
                      "share_experiment_result", "share_doc", "share_file"):
                f.coordination_gain = 0.6
                f.institutional_memory_gain = 0.2
            if at == "share_external_post":
                f.external_signal_value = 0.7
                f.customer_pressure = 0.4
            # founder rude-message failure mode -> conflict risk
            if "rude_or_uncivil_message" in agent.failure_modes and at == "send_message":
                f.conflict_risk += 0.2

        # meeting
        if cat == "meeting":
            f.coordination_gain = 0.7
            f.visibility_gain = 0.4
            if at == "record_meeting_notes":
                f.institutional_memory_gain = 0.6
                f.protocol_use_potential = 0.4
            if at == "schedule_meeting":
                f.context_switch_cost = 0.3   # interrupts others

        # repo review
        if at in ("review_pr", "request_changes", "approve_pr"):
            f.review_quality_gain = 0.7
            f.repo_health_gain = 0.4
            f.trust_gain = 0.2
            if at == "request_changes":
                f.conflict_risk = 0.3
                f.protocol_use_potential = 0.3   # review-before-merge norm
        if at == "merge_pr":
            f.progress_gain = 0.7              # merging is what ships work to the mainline
            f.repo_health_gain = 0.4
            pr_id = params.get("pr_id")
            pr = w.repo_system.repo.pull_requests.get(pr_id) if pr_id else None
            if pr is not None and not pr.reviewed:
                f.protocol_violation_risk = 0.6   # merging unreviewed
                f.technical_debt_risk = 0.5

        # v5 repo workflow promotion: committing/PR/CI are real progress toward shipping,
        # so the policy advances the chain instead of re-editing the same artifact.
        if at == "commit_patch":
            f.progress_gain = 0.6
            f.repo_health_gain = 0.3
        if at == "push_commit":
            f.progress_gain = 0.2
        if at == "open_pr":
            f.progress_gain = 0.5
            f.coordination_gain = 0.4
            f.review_quality_gain = 0.2
        if at == "run_ci":
            f.progress_gain = 0.35
            f.reproducibility_gain = 0.25

        # v5 release lifecycle promotion
        if at == "create_release_candidate":
            f.progress_gain = 0.6
            f.coordination_gain = 0.3
            f.public_reputation_effect = 0.2
        if at == "run_launch_readiness_check":
            f.review_quality_gain = 0.5
            f.reproducibility_gain = 0.2
        if at == "approve_release_candidate":
            f.coordination_gain = 0.4
            f.review_quality_gain = 0.3
        if at == "block_release_candidate":
            f.review_quality_gain = 0.4
            f.conflict_risk = 0.2
        if at == "publish_product_release":
            f.progress_gain = 0.8
            f.public_reputation_effect = 0.6
            f.visibility_gain = 0.4
        if at == "collect_post_launch_feedback":
            f.customer_pressure = 0.4
            f.external_signal_value = 0.4

        # sandbox / experiment
        if at in ("run_experiment", "run_cheap_pilot", "run_paper_baseline", "run_script"):
            f.progress_gain = 0.6 if at != "run_paper_baseline" else 0.4
            f.api_cost = 0.3 if at == "run_cheap_pilot" else 0.7
            f.compute_cost = 0.3 if at == "run_cheap_pilot" else 0.6
            f.budget_cost = f.api_cost
            f.runway_risk_delta = f.budget_cost * w.budget_system.budget.cost_multiplier * 0.3
        if at in ("export_result_to_tracker", "save_result"):
            f.reproducibility_gain = 0.7
            f.institutional_memory_gain = 0.5
            f.protocol_use_potential = 0.5      # experiment-logging norm
        if perception.local_unshared_results and at not in (
                "export_result_to_tracker", "share_sandbox_output", "save_result"):
            f.untracked_result_risk = 0.5

        # docs / artifacts
        if at in ("create_doc", "edit_doc", "publish_doc"):
            f.institutional_memory_gain = 0.4
            f.public_reputation_effect = 0.3 if at == "publish_doc" else 0.1
        if at in ("review_doc", "request_doc_changes"):
            f.review_quality_gain = 0.6
            f.claim_evidence_gain = 0.4
            if at == "request_doc_changes":
                f.conflict_risk = 0.2
        if at == "create_experiment_tracker":
            f.reproducibility_gain = 0.6
            f.protocol_creation_potential = 0.6
            f.institutional_memory_gain = 0.6
        if at == "create_cost_ledger":
            f.protocol_creation_potential = 0.5
            f.runway_risk_delta = -0.2
        if at in ("create_claim_evidence_table", "add_claim_evidence_row"):
            f.claim_evidence_gain = 0.7
            f.reproducibility_gain = 0.3
        if at == "create_customer_triage_sheet":
            f.customer_relevance = 0.7
            f.protocol_creation_potential = 0.4
        if at in ("create_workflow_artifact", "create_review_checklist"):
            f.protocol_creation_potential = 0.4
            f.institutional_memory_gain = 0.3

        # protocol
        if cat == "protocol":
            if at == "propose_protocol":
                f.protocol_creation_potential = 0.8
                f.institutional_memory_gain = 0.5
                f.visibility_gain = 0.4
            elif at in ("support_protocol", "follow_protocol"):
                f.protocol_use_potential = 0.6
            elif at == "enforce_protocol":
                f.norm_enforcement_gain = 0.7
                f.conflict_risk = 0.3
            elif at == "violate_protocol":
                f.protocol_violation_risk = 0.8
                f.reputation_risk = 0.3
            elif at == "oppose_protocol":
                f.conflict_risk = 0.4

        # governance — approve / reject / request changes on a pending proposal.
        # Persona drives the choice: usefulness/score pulls endorsement; risk + low
        # usefulness pulls skepticism. (weights modulated per-trait in policy.py)
        if cat == "governance":
            pid = params.get("proposal_id")
            pr = (w.proposal_manager.proposals.get(pid)
                  if pid and getattr(w, "proposal_manager", None) else None)
            useful = float(getattr(pr, "usefulness_score", None) or 0.5)
            risk = float(getattr(pr, "risk_score", None) or 0.3)
            adopt = float(getattr(pr, "adoption_score", None) or 0.5)
            if at == "approve_proposal":
                f.proposal_endorsement = 0.5 * adopt + 0.5 * useful
                f.coordination_gain = 0.3
            elif at == "reject_proposal":
                f.proposal_skepticism = risk + max(0.0, 0.5 - useful)
                f.conflict_risk = 0.3
            elif at == "request_proposal_changes":
                f.proposal_skepticism = 0.4 * risk + 0.2
                f.review_quality_gain = 0.4
                f.conflict_risk = 0.15

        # external bridge
        if cat == "bridge":
            f.external_signal_value = 0.5
            if at == "monitor_customer_feedback":
                f.customer_pressure = 0.5
                f.customer_relevance = 0.5
            if at == "ask_external_expert":
                f.expert_advice_value = 0.6
            if at == "read_feed":
                f.external_signal_value = 0.4
            if at == "post_company_update":
                f.public_reputation_effect = 0.5
                f.reputation_risk = 0.2

        # time / availability
        if cat == "time":
            if at in ("rest_offline", "sleep"):
                f.fatigue_delta = -0.3
                f.stress_delta = -0.2
                f.burnout_risk_delta = -0.3
            if at in ("work_overtime", "weekend_work"):
                f.progress_gain = 0.5
                f.burnout_risk_delta += 0.3

        # payroll / retention
        if cat == "payroll":
            if at == "ask_about_payroll":
                comp = w.budget_system.comp.get(agent.id)
                f.retention_risk_delta = (comp.retention_risk if comp else 0.0)
                f.conflict_risk = 0.2
            if at == "create_runway_update":
                f.payroll_trust_gain = 0.5
                f.institutional_memory_gain = 0.3
                f.protocol_creation_potential = 0.4
            if at == "announce_payroll_delay":
                f.payroll_trust_gain = 0.3
                f.conflict_risk = 0.3
            if at == "consider_external_offer":
                f.retention_risk_delta = 0.5
            if at == "run_payroll":
                f.payroll_trust_gain = 0.6


class OrgEventAppraisal:
    """Re-exported here for the DomainAdapter import surface; real logic in
    appraisal.py (OrgEventAppraisalImpl)."""
    def __init__(self):
        from environments.org_env.runtime_adapter.appraisal import OrgEventAppraisalImpl
        self._impl = OrgEventAppraisalImpl()

    def appraise(self, execution_result: Any, org_world: Any):
        return self._impl.appraise(execution_result, org_world)


__all__ = ["OrgFeatures", "OrgFeatureExtractor", "OrgEventAppraisal"]
