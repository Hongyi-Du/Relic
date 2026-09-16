"""RepoLite objects (DESIGN env_org §27-§31/§41). v0: summary + state, no real diff."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class BranchStatus(str, Enum):
    CLEAN = "clean"
    DIRTY = "dirty"
    READY_FOR_PR = "ready_for_pr"
    UNDER_REVIEW = "under_review"
    MERGED = "merged"
    STALE = "stale"
    CONFLICTED = "conflicted"
    ABANDONED = "abandoned"


class PRStatus(str, Enum):
    OPEN = "open"
    REVIEW_REQUESTED = "review_requested"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"
    MERGED = "merged"
    CLOSED = "closed"
    STALE = "stale"


@dataclass
class Commit:
    commit_id: str
    author_id: str
    branch_id: str
    message: str = ""
    changed_files: List[str] = field(default_factory=list)
    linked_task_id: Optional[str] = None
    linked_task_ids: List[str] = field(default_factory=list)   # v6 P0.4: full set
    linked_issue_ids: List[str] = field(default_factory=list)  # v6 P0.4
    linked_experiment_id: Optional[str] = None
    timestamp: int = 0
    quality_flags: List[str] = field(default_factory=list)   # quick_hack|missing_tests|schema_change|doc_only|risky_refactor|hotfix|experimental
    test_status: str = "unknown"
    risk_level: str = "low"
    # v5 repo workflow: the execution-layer patches this commit carries (so a merge
    # can apply them to the mainline product artifact) + lifecycle status.
    patch_ids: List[str] = field(default_factory=list)
    artifact_ids: List[str] = field(default_factory=list)
    status: str = "local"   # local | pushed | included_in_pr | merged | reverted


@dataclass
class Branch:
    branch_id: str
    owner_id: str
    base_branch: str = "main"
    status: BranchStatus = BranchStatus.CLEAN
    commit_ids: List[str] = field(default_factory=list)
    uncommitted_changes: int = 0
    linked_issue: Optional[str] = None
    linked_task: Optional[str] = None
    linked_experiment: Optional[str] = None
    merge_conflict_risk: float = 0.0
    last_sync_tick: int = 0


@dataclass
class PullRequest:
    pr_id: str
    author_id: str
    source_branch: str
    target_branch: str = "main"
    linked_issue: Optional[str] = None
    linked_task: Optional[str] = None
    linked_task_ids: List[str] = field(default_factory=list)   # v6 P0.4: full set
    linked_issue_ids: List[str] = field(default_factory=list)  # v6 P0.4
    reviewers: List[str] = field(default_factory=list)
    status: PRStatus = PRStatus.OPEN
    risk_level: str = "low"
    test_status: str = "unknown"
    review_comments: List[dict] = field(default_factory=list)
    approved_by: List[str] = field(default_factory=list)
    requested_changes: List[str] = field(default_factory=list)
    opened_tick: int = 0                                       # v6 P0.4: review/merge latency
    approved_tick: Optional[int] = None
    merged_tick: Optional[int] = None
    merge_conflict: bool = False
    reviewed: bool = False   # any review (approve/request_changes) happened
    # v5 repo workflow: commits/patches carried by the PR + CI state
    commit_ids: List[str] = field(default_factory=list)
    patch_ids: List[str] = field(default_factory=list)
    ci_run_ids: List[str] = field(default_factory=list)
    ci_passed: bool = False


@dataclass
class CIResult:
    ci_id: str
    pr_id: str
    commit_id: Optional[str] = None
    created_at_tick: int = 0
    status: str = "pending"   # pending | passed | failed | skipped
    checks: List[dict] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class ReleaseGateResult:
    gate: str
    passed: bool
    detail: str = ""


@dataclass
class ReleaseCandidate:
    candidate_id: str
    version: str
    created_by: str
    created_at_tick: int = 0
    last_updated_tick: int = 0          # v8d P0a: bumped when later merges are folded in
    included_pr_ids: List[str] = field(default_factory=list)
    included_commit_ids: List[str] = field(default_factory=list)
    included_artifact_ids: List[str] = field(default_factory=list)
    included_task_ids: List[str] = field(default_factory=list)
    required_gates: List[str] = field(default_factory=list)
    gate_results: List[dict] = field(default_factory=list)   # [{gate,passed,detail}]
    status: str = "draft"   # draft | under_review | approved | blocked | released
    blockers: List[str] = field(default_factory=list)
    approvals: List[str] = field(default_factory=list)
    waived_gates: List[str] = field(default_factory=list)
    linked_episode_ids: List[str] = field(default_factory=list)


@dataclass
class ProductRelease:
    release_id: str
    version: str
    candidate_id: str
    released_by: str
    released_at_tick: int = 0
    release_tag: str = ""
    public_summary: str = ""
    included_artifacts: List[str] = field(default_factory=list)
    # What the release shipped, fixed at publish time. The candidate keeps the
    # live view and is still being refreshed while it is open, so a release
    # that only pointed at its candidate did not record its own contents;
    # anything asking "what went out in this release" had to join through
    # candidate_id and got no answer if it asked the release directly.
    included_pr_ids: List[str] = field(default_factory=list)
    included_tasks: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    post_launch_feedback_ids: List[str] = field(default_factory=list)


@dataclass
class CompanyRepo:
    repo_id: str = "product_repo"
    name: str = "OSS product"
    main_branch: str = "main"
    modules: List[str] = field(default_factory=list)
    branches: dict = field(default_factory=dict)            # branch_id -> Branch
    commits: dict = field(default_factory=dict)             # commit_id -> Commit
    pull_requests: dict = field(default_factory=dict)       # pr_id -> PullRequest
    ci_runs: dict = field(default_factory=dict)             # ci_id -> CIResult
    release_candidates: dict = field(default_factory=dict)  # candidate_id -> ReleaseCandidate
    releases: dict = field(default_factory=dict)            # release_id -> ProductRelease
    current_version: str = "0.0.1"
    main_commit_ids: List[str] = field(default_factory=list)
    build_status: str = "unknown"
    release_tags: List[str] = field(default_factory=list)
    technical_debt: float = 0.3


__all__ = ["BranchStatus", "PRStatus", "Commit", "Branch", "PullRequest", "CIResult",
           "ReleaseGateResult", "ReleaseCandidate", "ProductRelease", "CompanyRepo"]
