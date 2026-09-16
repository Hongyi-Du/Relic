"""Persona mechanism ontology + mapping rules (NO agent-specific data).

The PersonaGraphBuilder is forbidden from hardcoding any individual agent's graph
(no per-agent-id branches). It may ONLY use the GENERIC rules + ontology here:

* ``TRAIT_TO_FEATURE_RULES`` — trait/skill -> policy feature (reuses the REAL
  policy coefficients so the graph reflects how the agent is actually scored).
* ``ACTION_METADATA`` / ``SPEECH_METADATA`` — per-action/speech ontology
  (feature_tags / required_skills / risk_tags / routine_tags / style_tags) so the
  builder can derive which traits/skills lead to which behaviors.
* ``STATE_TO_TENDENCY`` — work-state -> action tendency.
* ``FAILURE_KEYWORD_RISK`` — failure-mode keyword -> risk feature/action.

A speed-biased agent's "speed_bias -> run_cheap_pilot" tendency emerges from these
generic rules + that agent's own profile, never from a per-agent table.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from environments.org_env.runtime_adapter.policy import PROFILE_COEFFS

# (trait_or_skill, feature, coeff) — the real policy modulation (single source of truth).
TRAIT_TO_FEATURE_RULES: List[Tuple[str, str, float]] = list(PROFILE_COEFFS)

# features where higher = WORSE (risk/cost). A trait that *reduces* one of these
# (negative coeff) must not be read as "disinclined toward the mitigating action",
# so the tendency scorer skips risk features in feature_tags (use risk_tags +
# driver_traits instead).
RISK_FEATURES = {
    "technical_debt_risk", "protocol_violation_risk", "conflict_risk", "untracked_result_risk",
    "runway_risk_delta", "retention_risk_delta", "failure_risk_from_low_skill", "burnout_risk_delta",
    "reputation_risk", "trust_risk", "weekend_penalty", "overtime_penalty", "attention_cost",
    "fatigue_delta", "stress_delta", "capacity_strain",
}

# action_type -> ontology metadata. families: experiment / engineering / review /
# tracking / governance / recovery / payroll / external / doc / comm.
ACTION_METADATA: Dict[str, Dict[str, object]] = {
    "run_cheap_pilot": {"family": "experiment", "feature_tags": ["progress_gain", "demo_relevance"],
                        "required_skills": ["rapid_prototyping", "experimental_design", "core_coding"],
                        "risk_tags": ["untracked_result_risk"], "routine_tags": ["deep_work"]},
    "run_experiment": {"family": "experiment", "feature_tags": ["progress_gain", "reproducibility_gain"],
                       "required_skills": ["experimental_design", "multi_agent_eval"],
                       "risk_tags": ["untracked_result_risk"], "routine_tags": ["deep_work"]},
    "commit_changes": {"family": "engineering", "feature_tags": ["progress_gain", "repo_health_gain"],
                       "required_skills": ["core_coding"], "risk_tags": ["technical_debt_risk"]},
    "edit_file": {"family": "engineering", "feature_tags": ["progress_gain"], "required_skills": ["core_coding"]},
    "open_pr": {"family": "engineering", "feature_tags": ["repo_health_gain", "review_quality_gain"],
                "required_skills": ["core_coding"]},
    "work_overtime": {"family": "engineering", "feature_tags": ["progress_gain"],
                      "required_skills": [], "risk_tags": ["burnout_risk_delta"], "routine_tags": ["overtime"]},
    "debug_failure": {"family": "engineering", "feature_tags": ["progress_gain", "repo_health_gain"],
                      "required_skills": ["debugging", "core_coding"], "routine_tags": ["deep_work"]},
    "review_pr": {"family": "review", "feature_tags": ["review_quality_gain"],
                  "required_skills": ["review_quality"]},
    "formal_pr_review": {"family": "review", "feature_tags": ["review_quality_gain", "claim_evidence_gain"],
                         "required_skills": ["review_quality", "claim_evidence_review"],
                         "risk_tags": ["conflict_risk"]},
    "request_changes": {"family": "review", "feature_tags": ["review_quality_gain"],
                        "required_skills": ["review_quality"], "risk_tags": ["conflict_risk"]},
    "review_doc": {"family": "review", "feature_tags": ["review_quality_gain", "clarity_gain"],
                   "required_skills": ["clarity_review", "review_quality"]},
    "export_result_to_tracker": {"family": "tracking", "feature_tags": ["reproducibility_gain",
                                 "institutional_memory_gain"],
                                 "required_skills": ["reproducibility_tracking", "experiment_logging",
                                                     "institutional_memory", "claim_evidence_review"]},
    "create_experiment_tracker": {"family": "tracking", "feature_tags": ["institutional_memory_gain",
                                  "reproducibility_gain"],
                                  "required_skills": ["experiment_logging", "reproducibility_tracking",
                                                      "institutional_memory", "protocol_design"]},
    "update_experiment_tracker": {"family": "tracking", "feature_tags": ["institutional_memory_gain"],
                                  "required_skills": ["experiment_logging", "institutional_memory"]},
    "create_doc": {"family": "doc", "feature_tags": ["clarity_gain", "claim_evidence_gain"],
                   "required_skills": ["documentation"]},
    "edit_doc": {"family": "doc", "feature_tags": ["clarity_gain"],
                 "required_skills": ["documentation", "clarity_review"]},
    "publish_doc": {"family": "doc", "feature_tags": ["clarity_gain", "visibility_gain"],
                    "required_skills": ["documentation", "external_docs", "launch_materials"]},
    # design / artifact tendencies (designers, template/artifact-biased agents)
    "create_workflow_artifact": {"family": "artifact", "feature_tags": ["institutional_memory_gain",
                                 "clarity_gain"], "required_skills": ["artifact_design", "template_design",
                                 "documentation", "product_sense"]},
    "revise_workflow_artifact": {"family": "artifact", "feature_tags": ["institutional_memory_gain"],
                                 "required_skills": ["artifact_design", "template_design"]},
    "create_review_checklist": {"family": "artifact", "feature_tags": ["review_quality_gain",
                                "institutional_memory_gain"], "required_skills": ["template_design",
                                "review_quality", "artifact_design"]},
    "create_customer_triage_sheet": {"family": "artifact", "feature_tags": ["customer_relevance",
                                     "institutional_memory_gain"], "required_skills": ["customer_sense",
                                     "template_design", "documentation"]},
    # external voice / launch / customer tendencies
    "respond_to_public_comment": {"family": "external", "feature_tags": ["public_reputation_effect",
                                  "customer_relevance"], "required_skills": ["external_community_sensing",
                                  "brand_consistency"]},
    "prepare_customer_feedback_summary": {"family": "external", "feature_tags": ["customer_pressure",
                                          "clarity_gain"], "required_skills": ["customer_sense",
                                          "documentation"]},
    "dm_external_contact": {"family": "external", "feature_tags": ["external_signal_value"],
                            "required_skills": ["external_community_sensing", "brand_consistency"]},
    "post_company_update": {"family": "external", "feature_tags": ["public_reputation_effect",
                            "visibility_gain"], "required_skills": ["public_narrative"]},
    "share_external_post": {"family": "external", "feature_tags": ["external_signal_value",
                            "customer_relevance"], "required_skills": ["external_community_sensing"]},
    "monitor_customer_feedback": {"family": "external", "feature_tags": ["customer_pressure",
                                  "external_signal_value"], "required_skills": ["customer_sense"]},
    "rest_offline": {"family": "recovery", "feature_tags": ["recovery_value"], "routine_tags": ["recovery"]},
    "sleep": {"family": "recovery", "feature_tags": ["recovery_value"], "routine_tags": ["recovery"]},
    "ask_about_payroll": {"family": "payroll", "feature_tags": ["org_grievance_pull"]},
    "consider_external_offer": {"family": "payroll", "feature_tags": ["org_grievance_pull"],
                                "risk_tags": ["retention_risk_delta"]},
    "schedule_meeting": {"family": "governance", "feature_tags": ["coordination_gain", "visibility_gain"],
                         "required_skills": ["meeting_facilitation"]},
}

# speech_act -> ontology metadata.
SPEECH_METADATA: Dict[str, Dict[str, object]] = {
    "challenge_result": {"feature_tags": ["claim_evidence_gain", "reproducibility_gain"],
                         "style_tags": ["directness", "evidence_demand"],
                         "required_skills": ["review_quality", "claim_evidence_review"],
                         "risk_tags": ["conflict_risk"]},
    "request_reproduction": {"feature_tags": ["reproducibility_gain"], "style_tags": ["evidence_demand"],
                             "required_skills": ["reproducibility_tracking"]},
    "ask_for_evidence": {"feature_tags": ["claim_evidence_gain"], "style_tags": ["evidence_demand"],
                         "required_skills": ["claim_evidence_review"]},
    "warn_about_risk": {"feature_tags": ["review_quality_gain"], "style_tags": ["directness"],
                        "required_skills": ["technical_judgment", "review_quality"],
                        "risk_tags": ["technical_debt_risk", "conflict_risk"],
                        # risk-averse / quality / long-term agents proactively flag risk
                        "driver_traits": [("risk_aversion", 0.4), ("quality_bar", 0.4),
                                          ("long_termism", 0.3), ("distrust_sensitivity", 0.3)]},
    "promise_work": {"feature_tags": ["progress_gain", "demo_relevance"], "style_tags": ["urgency"],
                     "required_skills": ["rapid_prototyping"]},
    "suggest_rewrite": {"feature_tags": ["clarity_gain", "claim_evidence_gain"], "style_tags": ["directness"],
                        "required_skills": ["clarity_review", "claim_wording"]},
    "propose_protocol": {"feature_tags": ["protocol_creation_potential", "norm_enforcement_gain"],
                         "style_tags": ["normativity"], "required_skills": ["protocol_design"]},
    "enforce_protocol": {"feature_tags": ["norm_enforcement_gain", "protocol_use_potential"],
                         "style_tags": ["normativity", "directness"], "risk_tags": ["conflict_risk"]},
    "deescalate": {"feature_tags": ["coordination_gain", "trust_gain"], "style_tags": ["warmth"],
                   "required_skills": ["team_morale"]},
    "push_team": {"feature_tags": ["progress_gain", "deadline_urgency"], "style_tags": ["urgency", "directness"]},
    "coordinate_followup": {"feature_tags": ["coordination_gain"], "style_tags": ["warmth"],
                            "required_skills": ["meeting_facilitation"]},
    "share_external_signal": {"feature_tags": ["external_signal_value"], "style_tags": ["warmth"],
                              "required_skills": ["external_community_sensing"]},
}

# work-state vital -> action tendencies (sign): high value pushes(+)/suppresses(-).
STATE_TO_TENDENCY: Dict[str, List[Tuple[str, float]]] = {
    "fatigue": [("rest_offline", 0.6), ("work_overtime", -0.5), ("debug_failure", -0.3)],
    "burnout_risk": [("rest_offline", 0.7), ("consider_external_offer", 0.4), ("work_overtime", -0.6)],
    "stress": [("rest_offline", 0.3), ("deescalate", -0.3)],
    "compensation_stress": [("ask_about_payroll", 0.7), ("consider_external_offer", 0.6)],
    "retention_risk": [("consider_external_offer", 0.7)],
}

# role -> preferred action families / speech acts (role/context gating). Keyed by a
# substring of the agent's role (generic, NOT per-agent-id). A tendency that fits the
# role is boosted in summary salience; an off-role low-skill tendency is damped.
ROLE_PRIORS: Dict[str, Dict[str, set]] = {
    "founder":     {"families": {"external", "governance"},
                    "speech": {"push_team", "coordinate_followup", "commit_to_direction", "defend_demo_progress"}},
    "cofounder":   {"families": {"review", "tracking", "governance"},
                    "speech": {"ask_for_evidence", "propose_protocol", "challenge_result"}},
    "architect":   {"families": {"review", "tracking", "governance"},
                    "speech": {"ask_for_evidence", "propose_protocol", "challenge_result"}},
    "reliability": {"families": {"tracking", "review"},
                    "speech": {"request_reproduction", "warn_about_risk", "challenge_result", "enforce_protocol"}},
    "community":   {"families": {"external"},
                    "speech": {"deescalate", "share_external_signal", "coordinate_followup"}},
    "external":    {"families": {"external", "doc"},
                    "speech": {"share_external_signal", "deescalate"}},
    "editorial":   {"families": {"review", "doc"},
                    "speech": {"suggest_rewrite", "warn_about_risk"}},
    "artifact":    {"families": {"artifact", "doc"}, "speech": {"propose_protocol"}},
    "design":      {"families": {"artifact", "doc"}, "speech": set()},
    "engineer":    {"families": {"engineering", "experiment"}, "speech": {"promise_work", "push_team"}},
}

# work-state vitals that are scheduling / bookkeeping, NOT persona-explanatory — they
# must never enter summary salience (kept only in raw / work-state detail panel).
SUMMARY_STATE_BAN = {
    "next_available_tick", "aux_speech_slots_remaining", "daily_message_count",
    "daily_meeting_count", "deep_work_blocks_used_today", "current_meeting_id",
}


def role_priors_for(role: str) -> Dict[str, set]:
    """Union of priors for every keyword that appears in the role (e.g.
    'artifact_design' matches both 'artifact' and 'design')."""
    fams: set = set()
    sp: set = set()
    rl = (role or "").lower()
    for key, pri in ROLE_PRIORS.items():
        if key in rl:
            fams |= pri.get("families", set())
            sp |= pri.get("speech", set())
    return {"families": fams, "speech": sp}


# failure-mode keyword -> (risk feature, risky action). Generic substring match.
FAILURE_KEYWORD_RISK: Dict[str, Tuple[str, str]] = {
    "review": ("review_quality_gain", "bypass_review"),
    "process": ("protocol_violation_risk", "skip_protocol"),
    "track": ("untracked_result_risk", "skip_tracker"),
    "scope": ("technical_debt_risk", "scope_creep"),
    "conflict": ("conflict_risk", "escalate_conflict"),
    "rude": ("conflict_risk", "uncivil_message"),
    "escalat": ("conflict_risk", "escalate_conflict"),
    "doc": ("clarity_gain", "missing_docs"),
    "maintenance": ("technical_debt_risk", "maintenance_blindness"),
    "overwork": ("burnout_risk_delta", "overwork"),
    "overcommit": ("retention_risk_delta", "overcommit"),
    "centraliz": ("conflict_risk", "centralize_control"),
    "evidence": ("claim_evidence_gain", "weak_evidence"),
    "operational": ("technical_debt_risk", "skip_ops_detail"),
}


# -- reverse maps (derived; not hardcoded per agent) ----------------------- #
def _feature_to(meta: Dict[str, Dict[str, object]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name, m in meta.items():
        for feat in m.get("feature_tags", []) or []:
            out.setdefault(feat, []).append(name)
    return out


FEATURE_TO_ACTIONS = _feature_to(ACTION_METADATA)
FEATURE_TO_SPEECH = _feature_to(SPEECH_METADATA)


def features_for(key: str) -> List[Tuple[str, float]]:
    """Policy features a trait/skill modulates (feature, coeff)."""
    return [(f, c) for (k, f, c) in TRAIT_TO_FEATURE_RULES if k == key]


def actions_using_skill(skill: str) -> List[str]:
    return [a for a, m in ACTION_METADATA.items() if skill in (m.get("required_skills", []) or [])]


def speech_using_skill(skill: str) -> List[str]:
    return [a for a, m in SPEECH_METADATA.items() if skill in (m.get("required_skills", []) or [])]


__all__ = [
    "TRAIT_TO_FEATURE_RULES", "RISK_FEATURES", "ACTION_METADATA", "SPEECH_METADATA",
    "STATE_TO_TENDENCY", "FAILURE_KEYWORD_RISK", "ROLE_PRIORS", "SUMMARY_STATE_BAN",
    "FEATURE_TO_ACTIONS", "FEATURE_TO_SPEECH",
    "features_for", "actions_using_skill", "speech_using_skill", "role_priors_for",
]
