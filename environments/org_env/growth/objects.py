"""Internal Growth Module — domains, mappings, constants, GrowthEvent (spec §1-§15).

Agents accumulate professional skill + domain reputation + informal authority from
event evidence. Core personality/professional-profile dims (speed_bias, risk_aversion,
process_commitment, …) are NOT touched here — only skills, reputation, authority, go-to.
All numbers are rule-based + bounded + event-grounded (no free-form LLM drift).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# -- §1.1 skill domains (operate on agent.skills; overlap with policy skill names) --
SKILL_DOMAINS = (
    "rapid_prototyping", "debugging", "eval_design", "reproducibility_tracking",
    "claim_evidence_review", "documentation_quality", "source_credibility",
    "report_quality", "workflow_design", "coordination", "customer_research",
    "cost_governance",
)

# Skills the reconciler may move: the org-skill taxonomy plus the coding-skill
# taxonomy (single source: coding/profile.py::CODING_SKILLS). Outcome gates read
# coding skills via agent.skill() (e.g. test_writing in commit quality flags); a
# skill a gate consumes but growth cannot move makes that gate permanently
# unpassable for members seeded below its threshold.
def _growable_skills() -> tuple:
    from environments.org_env.coding.profile import CODING_SKILLS
    return tuple(SKILL_DOMAINS) + tuple(k for k in CODING_SKILLS if k not in SKILL_DOMAINS)


GROWABLE_SKILLS = _growable_skills()

# -- §1.1b consumption-vocabulary aliases -------------------------------------
# The seed team (backend/agents/seed_team.py) and several mechanical consumption
# points (feature_extractor.ACTION_SKILL, the sandbox pilot skill in execution.py,
# meeting sub-action scoring in world.py, policy profile-conditioning) use a
# free-form skill vocabulary that predates the growth taxonomy. A name a gate or
# score reads but growth cannot write is frozen at its seed value for the whole
# run: growth moves the canonical name while the consumer reads the alias.
# Every mechanical read must therefore go through effective_skill(), which
# resolves an alias to its canonical growable skill and returns the max over the
# equivalence class — grown competence becomes visible at every read point while
# seeded heterogeneity (e.g. calvin experimental_design=0.86) is preserved.
SKILL_ALIASES: Dict[str, str] = {
    "experimental_design": "eval_design",
    "experiment_design": "eval_design",
    "core_coding": "patch_generation",
    "documentation": "documentation_quality",
    "meeting_facilitation": "coordination",
    "review_quality": "code_review",
    "claim_wording": "claim_evidence_review",
    "experiment_logging": "reproducibility_tracking",
    "cost_logging": "cost_governance",
    "customer_triage": "customer_research",
    "customer_sense": "customer_research",
    "external_community_sensing": "customer_research",
    "public_narrative": "report_quality",
    "protocol_design": "workflow_design",
    "artifact_design": "workflow_design",
}


def _skill_classes() -> Dict[str, tuple]:
    classes: Dict[str, list] = {}
    for alias, canon in SKILL_ALIASES.items():
        classes.setdefault(canon, [canon]).append(alias)
    return {c: tuple(names) for c, names in classes.items()}


_SKILL_CLASS = _skill_classes()


def resolve_skill(name: str) -> str:
    """Canonical growable name for a (possibly aliased) skill name."""
    return SKILL_ALIASES.get(name, name)


def effective_skill(skills: Any, name: str, default: float = 0.0) -> float:
    """Effective value of ``name`` over its alias equivalence class: the max of
    every present entry (canonical + aliases), else ``default``. Growth writes
    only canonical names; seeds may only carry aliases — max makes both visible."""
    canon = SKILL_ALIASES.get(name, name)
    vals = [v for k in _SKILL_CLASS.get(canon, (name,))
            if (v := skills.get(k)) is not None]
    return float(max(vals)) if vals else float(default)

# -- §1.2 reputation domains (org-recognised "trusted to do X reliably") --
REPUTATION_DOMAINS = (
    "engineering_execution", "reliability_evidence", "documentation_quality",
    "product_judgment", "customer_sensing", "workflow_design",
    "coordination_leadership", "cost_governance",
)

# -- §10.1 reputation-domain -> skill aggregate weights A_{d,k} (rows ~sum to 1) --
DOMAIN_SKILL_MAP: Dict[str, Dict[str, float]] = {
    "engineering_execution": {"rapid_prototyping": 0.4, "debugging": 0.4, "eval_design": 0.2},
    "reliability_evidence": {"reproducibility_tracking": 0.35, "claim_evidence_review": 0.30,
                             "eval_design": 0.20, "source_credibility": 0.15},
    "documentation_quality": {"documentation_quality": 0.6, "claim_evidence_review": 0.25,
                              "report_quality": 0.15},
    "product_judgment": {"claim_evidence_review": 0.4, "documentation_quality": 0.3,
                         "customer_research": 0.3},
    "customer_sensing": {"customer_research": 0.8, "documentation_quality": 0.2},
    "workflow_design": {"workflow_design": 0.6, "report_quality": 0.2, "coordination": 0.2},
    "coordination_leadership": {"coordination": 0.7, "workflow_design": 0.3},
    "cost_governance": {"cost_governance": 0.8, "workflow_design": 0.2},
}

# -- §3 outcome scores o_e in [-1, 1] (rule-based; LLM may only add rationale) --
OUTCOME_SCORE: Dict[str, float] = {
    "patch_applied": 0.4, "patch_merged_mainline": 0.8, "pr_approved": 0.6, "pr_merged": 0.9,
    "ci_passed": 0.5, "task_implementation_done": 0.4, "task_merged": 0.8,
    "gap_mitigated": 0.5, "gap_resolved": 0.9, "issue_partially_resolved": 0.5, "issue_resolved": 0.9,
    "valid_challenge": 0.6, "protocol_used": 0.4, "release_gate_passed": 1.0,
    "patch_rejected_duplicate": -0.2, "patch_rejected_invalid": -0.5, "pr_changes_requested": -0.4,
    "ci_failed": -0.6, "claim_disputed_unresolved": -0.4, "result_non_reproducible": -0.7,
    "protocol_violation": -0.8, "release_gate_blocked": -0.7,
}

# -- §4 visibility weights v_e in [0, 1] --
VISIBILITY: Dict[str, float] = {
    "private": 0.2, "direct_message": 0.3, "team_channel": 0.5, "shared_object": 0.6,
    "pr_review_ci": 0.8, "merged_mainline": 0.9, "company_wide": 1.0,
}

# -- §6 credit by role in event --
CREDIT_ROLE: Dict[str, float] = {
    "primary_actor": 1.0, "approving_reviewer": 0.4, "problem_catching_reviewer": 0.6,
    "issue_opener": 0.3, "signal_linker": 0.3, "proposer": 0.4, "tool_user": 0.3,
    "blocker_resolver": 0.7, "mentioned": 0.1, "unrelated": 0.0,
}
MAX_TOTAL_CREDIT = 2.0          # §6: scale down if a single event over-credits

# -- §8/§9 learning rates + decay --
ETA_SKILL = 0.015               # positive skill learning rate
ETA_SKILL_NEG = 0.004           # negative (smaller — one failure shouldn't gut a skill)
MU_REP = 0.5                    # reputation baseline every domain regresses toward
LAMBDA_REP_DAY = 0.005          # reputation decay per day
ETA_REP = 0.02                  # reputation learning rate
GAMMA_NEG = 1.3                 # negative reputation impact slightly > positive

# -- §15.2 bounds --
MAX_SKILL_DELTA_EVENT = 0.02
MAX_REP_DELTA_EVENT = 0.03
MAX_SKILL_DELTA_DAY = 0.05
MAX_REP_DELTA_DAY = 0.08

# -- §5 workload reference --
W_REF = 4.0
MAX_WORKLOAD_WEIGHT = 1.5

# -- §10 authority derivation weights --
# v8e #2: competence-weighted so domain SKILL (not transient activity or founder ownership)
# drives authority. Skill weight raised (0.30->0.42), recent-use + ownership lowered
# (0.15->0.10, 0.10->0.04 — ownership correlates with founder/cofounder role and was
# letting founders outrank specialist reviewers), and the failure penalty softened
# (0.20->0.12) so a stale negative doesn't crush a high-skill reviewer. go-to derives
# from these same authority values (leaderboard-consistent).
BETA_R, BETA_S, BETA_U, BETA_O, BETA_B = 0.42, 0.42, 0.10, 0.04, 0.12
USE_REF = 5.0                   # §10.2 N_ref for recent successful use
FAIL_REF = 3.0                  # §10.4 N_fail_ref
RECENT_WINDOW = 200             # last_K ticks for use/fail

# -- §11 authority influence --
REVIEWER_TAU = 2.0              # §11.1 reviewer routing softmax temperature
POLICY_AUTHORITY_DELTA = 0.07   # §11.3 score' bonus weight (in [0.05, 0.10])

# -- §14 go-to thresholds --
# Relative rule (growth-run finding: absolute 0.70 was unreachable in 240t, no go-to formed).
# A tentative go-to needs: domain rank #1 AND (authority >= floor OR >= 0.05 over #2) AND
# >=2 recent positive evidence AND no recent failure. The absolute 0.70 becomes a "strong"
# confirmation only (kept for reference / future longer-horizon promotion).
GO_TO_AUTHORITY = 0.70           # strong (long-horizon) confirmation threshold
# v8 300t finding: authority is damped to ~0.4-0.5 (skills start at 0.3, fail term subtracts),
# so a 0.50 floor + neg==0 produced only 1 go-to despite clear reputation divergence. Lower
# the floor + margin and require NET-positive evidence (pos>neg) instead of zero failures.
GO_TO_REL_FLOOR = 0.45           # tentative: must clear at least this
GO_TO_REL_MARGIN = 0.03          # ... or lead the next agent by this margin
GO_TO_MIN_POSITIVE_EVENTS = 2
GO_TO_TOP_N = 2


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def workload_weight(workload_score: float) -> float:
    """§5: log-growth so big tasks don't over-reward; capped at 1.5x."""
    import math
    if workload_score <= 0:
        return 0.3
    return min(MAX_WORKLOAD_WEIGHT, math.log(1 + workload_score) / math.log(1 + W_REF))


@dataclass
class GrowthSignal:
    """One credited agent's stake in one growth-relevant event (pre-apply)."""
    agent_id: str
    outcome: float
    visibility: float
    workload: float
    credit: float
    skill_weights: Dict[str, float] = field(default_factory=dict)   # m_{e,k}
    rep_weights: Dict[str, float] = field(default_factory=dict)     # n_{e,d}
    source_event_id: str = ""
    source_action_id: str = ""
    tick: int = 0
    reason: str = ""


@dataclass
class GrowthEvent:
    """§12: persistent evidence for every state change (no black box)."""
    growth_event_id: str
    tick: int
    agent_id: str
    domain: str
    source_event_id: str = ""
    source_action_id: str = ""
    skill_deltas: Dict[str, float] = field(default_factory=dict)
    reputation_deltas: Dict[str, float] = field(default_factory=dict)
    authority_before: float = 0.0
    authority_after: float = 0.0
    visibility: float = 0.0
    outcome_score: float = 0.0
    credit: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.__dict__.items()}


def new_reputation() -> Dict[str, float]:
    return {d: MU_REP for d in REPUTATION_DOMAINS}


def new_authority() -> Dict[str, float]:
    return {d: 0.0 for d in REPUTATION_DOMAINS}


__all__ = [
    "SKILL_DOMAINS", "GROWABLE_SKILLS", "SKILL_ALIASES", "resolve_skill", "effective_skill",
    "REPUTATION_DOMAINS", "DOMAIN_SKILL_MAP",
    "OUTCOME_SCORE", "VISIBILITY",
    "CREDIT_ROLE", "MAX_TOTAL_CREDIT", "ETA_SKILL", "ETA_SKILL_NEG", "MU_REP", "LAMBDA_REP_DAY",
    "ETA_REP", "GAMMA_NEG", "MAX_SKILL_DELTA_EVENT", "MAX_REP_DELTA_EVENT", "MAX_SKILL_DELTA_DAY",
    "MAX_REP_DELTA_DAY", "W_REF", "BETA_R", "BETA_S", "BETA_U", "BETA_O", "BETA_B", "USE_REF",
    "FAIL_REF", "RECENT_WINDOW", "REVIEWER_TAU", "POLICY_AUTHORITY_DELTA", "GO_TO_AUTHORITY",
    "GO_TO_MIN_POSITIVE_EVENTS", "GO_TO_TOP_N", "GrowthSignal", "GrowthEvent",
    "clip01", "workload_weight", "new_reputation", "new_authority",
]
