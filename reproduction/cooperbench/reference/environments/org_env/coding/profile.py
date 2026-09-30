"""Per-agent coding profile (v11 §8): 'everyone can code' -> 'different people do different
kinds of code work'. Each role emphasizes different coding SKILLS, and the policy gives a
small utility nudge toward coding actions that match the agent's profile — so Sean lands fast
patches, Calvin/SL drive tests + CI-failure analysis, Victor does design/refactor fixes, and
Paul focuses on review + final merge / release judgment.
"""
from __future__ import annotations

from typing import Any, Optional

# 8 coding-skill dimensions (the individual coding-ability axes)
CODING_SKILLS = (
    "bug_localization", "patch_generation", "test_writing", "debugging",
    "refactoring", "integration_awareness", "release_engineering", "code_review",
)
DEFAULT_EMPHASIS = 0.3

# role -> coding-skill emphasis in [0,1] (only non-default emphases listed)
ROLE_CODING_PROFILE = {
    "fast_engineer":   {"patch_generation": 0.9, "bug_localization": 0.7, "debugging": 0.6,
                        "integration_awareness": 0.5},
    "reliability":     {"test_writing": 0.9, "debugging": 0.8, "bug_localization": 0.7,
                        "code_review": 0.7, "release_engineering": 0.6},
    "cofounder":       {"refactoring": 0.8, "integration_awareness": 0.8, "code_review": 0.6,
                        "patch_generation": 0.5},      # architecture / design fixes (Victor)
    "founder":         {"release_engineering": 0.8, "code_review": 0.7},  # final merge / release (Paul)
    "artifact_design": {"refactoring": 0.6, "integration_awareness": 0.6, "test_writing": 0.4},
    "editorial":       {"code_review": 0.5, "test_writing": 0.3},
}

# coding ACTION -> the skill dim it exercises (utility nudges follow the agent's emphasis)
ACTION_CODING_SKILL = {
    "edit_repo_file": "patch_generation",
    "commit_patch": "patch_generation",
    "write_design_note": "refactoring",
    "run_eval_stub": "test_writing", "create_eval_stub": "test_writing",
    "update_claim_tracker": "test_writing",
    "run_ci": "release_engineering",
    "run_launch_readiness_check": "release_engineering",
    "create_release_candidate": "release_engineering",
    "publish_product_release": "release_engineering",
    "approve_release_candidate": "release_engineering",
    "review_pr": "code_review", "approve_pr": "code_review",
    "request_changes": "code_review", "merge_pr": "code_review",
    "dogfood_product": "test_writing",   # using/QA-ing the product is a testing skill
}

# centered-emphasis -> policy utility nudge (cf. growth POLICY_AUTHORITY_DELTA=0.07)
CODING_BONUS_SCALE = 0.2


def role_coding_profile(role: Optional[str]) -> dict:
    prof = {k: DEFAULT_EMPHASIS for k in CODING_SKILLS}
    prof.update(ROLE_CODING_PROFILE.get(role or "", {}))
    return prof


def action_coding_skill(action_type: str) -> Optional[str]:
    return ACTION_CODING_SKILL.get(action_type)


def coding_affinity(role: Optional[str], action_type: str) -> float:
    """Raw emphasis [0,1] of this role's profile for a coding action; 0.0 if not a coding action."""
    skill = ACTION_CODING_SKILL.get(action_type)
    if skill is None:
        return 0.0
    return role_coding_profile(role).get(skill, DEFAULT_EMPHASIS)


def coding_policy_bonus(agent: Any, action_type: str, world: Any = None) -> float:
    """Small additive utility nudge: above-average coding affinity is rewarded, below-average
    slightly discouraged, so profile-matched agents prefer their kind of code work."""
    skill = ACTION_CODING_SKILL.get(action_type)
    if skill is None:
        return 0.0
    emphasis = role_coding_profile(getattr(agent, "role", "")).get(skill, DEFAULT_EMPHASIS)
    return (emphasis - DEFAULT_EMPHASIS) * CODING_BONUS_SCALE


__all__ = [
    "CODING_SKILLS", "ROLE_CODING_PROFILE", "ACTION_CODING_SKILL", "DEFAULT_EMPHASIS",
    "role_coding_profile", "action_coding_skill", "coding_affinity", "coding_policy_bonus",
]
