"""Messy product substrate — the concrete artifacts the team fights over.

The company is building a research-agent prototype (LanternScout) from 0→1. The
substrate is deliberately MESSY (overpromising README, vague eval, claim tracker
without evidence enforcement, …) so that conflict / reflection / wishes / proposals
/ tools / protocols form around real code/doc/issue/eval/report objects rather than
abstract talk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ARTIFACT_TYPES = ("repo_file", "doc", "eval", "template", "issue", "demo", "report", "tool_stub")
ARTIFACT_STATUS = ("draft", "active", "broken", "needs_review", "deprecated", "open", "closed")

# canonical "purpose" of an artifact, derived from its id / file path. Used for
# purpose-level dedup of create-class doc actions and task-specific evidence matching.
_PURPOSE_PATTERNS = (
    ("onboarding", ("onboarding",)),
    ("report_quality", ("report_quality", "quality_checklist")),
    ("report_writer", ("report_writer",)),
    ("report_template", ("report_template",)),
    ("readme", ("readme",)),
    ("claim_tracker", ("claim_tracker",)),
    ("source_tracker", ("source_tracker",)),
    ("eval", ("eval_stub", "eval/", "art_eval", "_eval_")),
    ("research_loop", ("research_loop",)),
    ("product_design", ("product_design",)),
    ("design_note", ("design_note",)),
    ("demo", ("demo",)),
)


def artifact_purpose(artifact_id_or_path: str) -> str:
    s = (artifact_id_or_path or "").lower()
    for purpose, pats in _PURPOSE_PATTERNS:
        if any(p in s for p in pats):
            return purpose
    return "other"


@dataclass
class ProductState:
    product_id: str
    name: str
    stage: str                              # legacy RELEASE lifecycle: messy->internal_release->beta_released
    summary: str
    milestone_stage: str = "discovery"      # v13 WORK-MODE: discovery->stabilization->integration->market_validation
    known_systemic_issues: List[str] = field(default_factory=list)
    repo_id: Optional[str] = None
    open_issue_ids: List[str] = field(default_factory=list)
    artifact_ids: List[str] = field(default_factory=list)
    # which product substrate seeded this world (OSS time-machine brief §6): synthetic_lanternscout
    # (default / debug) | oss_time_machine (real OSS history). substrate_meta carries non-secret
    # metadata only (dataset_id / anonymized product name) — never reference code or hidden tests.
    substrate_type: str = "synthetic_lanternscout"
    substrate_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}


@dataclass
class ProductArtifact:
    artifact_id: str
    artifact_type: str
    title: str
    status: str
    owner_agent_id: Optional[str] = None
    linked_repo_id: Optional[str] = None
    linked_file_path: Optional[str] = None
    summary: str = ""
    content: str = ""                       # real WORKING-tree file text — grows per accepted patch
    mainline_content: str = ""              # real MAINLINE file text — advances only on PR merge
    known_gaps: List[str] = field(default_factory=list)
    priority: str = "unclear"               # for issues
    problem: str = ""                       # for issues
    linked_episode_ids: List[str] = field(default_factory=list)
    linked_wish_ids: List[str] = field(default_factory=list)
    linked_proposal_ids: List[str] = field(default_factory=list)
    linked_action_ids: List[str] = field(default_factory=list)   # always set on a change
    linked_task_ids: List[str] = field(default_factory=list)
    revision: int = 0                       # working/candidate revision (bumps on accepted patch)
    capabilities: List[str] = field(default_factory=list)   # v5 §P0-5: what's already built
    # v5 repo workflow: mainline only advances when a PR merges the carrying commit(s)
    mainline_revision: int = 0
    linked_branch_ids: List[str] = field(default_factory=list)
    linked_commit_ids: List[str] = field(default_factory=list)
    linked_pr_ids: List[str] = field(default_factory=list)
    created_at_tick: int = 0
    updated_at_tick: int = 0
    # v4 §5.2: every revision is a concrete patch, awaiting a second party's review
    patch_history_ids: List[str] = field(default_factory=list)
    change_summaries: List[str] = field(default_factory=list)
    awaiting_review: bool = False
    last_reviewed_tick: Optional[int] = None
    # issue closure evidence (v4 review §3): closing must record WHY + WHAT resolved it
    close_reason: str = ""
    resolved_by_action_ids: List[str] = field(default_factory=list)
    resolved_by_patch_ids: List[str] = field(default_factory=list)
    resolved_by_task_ids: List[str] = field(default_factory=list)
    # Internal Pipeline #1 follow-up: rich issue lifecycle (the open/closed `status` vocab
    # is kept for compat; issue_status carries open/in_progress/partially_resolved/resolved).
    issue_status: str = "open"
    linked_gap_ids: List[str] = field(default_factory=list)
    # Appended for positional-constructor and legacy snapshot compatibility.
    # True only for a repository path introduced during the run. Until its
    # carrying PR merges, it exists on the working branch but not on mainline.
    created_as_new_file: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            k: (list(v) if isinstance(v, list) else v)
            for k, v in self.__dict__.items()
        }
        # A checkpoint pickled before this field existed has no instance
        # attribute for it.  Runtime getattr() correctly falls back to the
        # dataclass default, but snapshots must also make that safe default
        # explicit so the next replay is on the current schema.
        payload.setdefault("created_as_new_file", False)
        return payload


__all__ = ["ProductState", "ProductArtifact", "ARTIFACT_TYPES", "ARTIFACT_STATUS", "artifact_purpose"]
