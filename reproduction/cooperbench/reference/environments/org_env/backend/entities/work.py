"""OrgEnv work objects — Task / Issue / Document / Experiment (DESIGN env_org §33.1).

Frozen dataclass schema only (skeleton). The "work" group covers what agents
produce/track day-to-day; economy & governance objects live in sibling modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    REVIEW = "review"
    # Internal Pipeline spec #2: a product task is NOT done on an accepted patch alone —
    # it advances implementation_done -> review_pending -> merged -> released. "done" is
    # reserved for non-product tasks / terminal; completion = merged or released.
    IMPLEMENTATION_DONE = "implementation_done"
    REVIEW_PENDING = "review_pending"
    MERGED = "merged"
    RELEASED = "released"
    DONE = "done"
    ABANDONED = "abandoned"


# a finished product task: reached the mainline (merged) or shipped (released); "done" is
# kept for non-product tasks that have no merge step.
COMPLETED_TASK_STATUSES = ("merged", "released", "done")


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    task_id: str
    title: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.OPEN
    priority: int = 3
    owner_id: Optional[str] = None
    required_skills: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    deadline_tick: Optional[int] = None
    estimated_effort: float = 0.0
    actual_effort: float = 0.0
    linked_docs: List[str] = field(default_factory=list)
    linked_issues: List[str] = field(default_factory=list)
    linked_experiments: List[str] = field(default_factory=list)
    linked_artifacts: List[str] = field(default_factory=list)
    visibility: str = "team"
    history: List[Dict[str, Any]] = field(default_factory=list)
    # preflight v3 §4: gate completion on real evidence (no single weak action -> done)
    progress_score: float = 0.0
    completion_requirements: List[str] = field(
        default_factory=lambda: ["artifact_revised", "gaps_cleared", "reviewed", "multi_evidence"])
    progress_evidence: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Issue:
    issue_id: str
    title: str = ""
    description: str = ""
    severity: str = "minor"
    owner_id: Optional[str] = None
    status: str = "open"
    source: str = "internal"   # internal | customer | external_post | bug_report | competitor_signal
    related_task_id: Optional[str] = None
    customer_impact: str = ""
    created_tick: int = 0
    resolved_tick: Optional[int] = None
    # v6 P0.3: external feedback is clustered, not re-filed. A repeat of the same pain
    # bumps support_count / last_seen_tick and records who raised it + which signal,
    # instead of creating a near-duplicate issue (the v5 "13 identical issues" bug).
    topic: str = ""
    support_count: int = 1
    supporting_agent_ids: List[str] = field(default_factory=list)
    source_signal_ids: List[str] = field(default_factory=list)
    last_seen_tick: int = 0
    # Internal Pipeline spec #1: richer status (open / in_progress / partially_resolved /
    # resolved / blocked / reopened) + resolution evidence, set by the StateReconciler from
    # linked task / PR / patch state — never "resolved" without an evidence link.
    linked_task_ids: List[str] = field(default_factory=list)
    linked_gap_ids: List[str] = field(default_factory=list)
    resolved_by_patch_ids: List[str] = field(default_factory=list)
    resolved_by_task_ids: List[str] = field(default_factory=list)
    resolved_by_pr_ids: List[str] = field(default_factory=list)
    last_status_tick: int = 0


@dataclass
class Document:
    doc_id: str
    title: str = ""
    doc_type: str = "design_doc"   # design_doc|experiment_note|customer_summary|SOP|checklist|tracker|meeting_note|API_note
    author_id: Optional[str] = None
    owner_id: Optional[str] = None
    content_summary: str = ""
    version: int = 1
    visibility: str = "team"
    linked_tasks: List[str] = field(default_factory=list)
    linked_protocols: List[str] = field(default_factory=list)
    last_updated_tick: int = 0
    trust_level: float = 0.5


@dataclass
class Experiment:
    experiment_id: str
    title: str = ""
    hypothesis: str = ""
    owner_id: Optional[str] = None
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    compute_cost: float = 0.0
    api_cost: float = 0.0
    dataset: str = ""
    method: str = ""
    result_summary: str = ""
    linked_task_id: Optional[str] = None
    logged_to_tracker: bool = False
    reproducibility_status: str = "unknown"
    failure_reason: str = ""


__all__ = ["TaskStatus", "ExperimentStatus", "Task", "Issue", "Document", "Experiment"]
