"""Types for the frozen OSS time-machine substrate."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

OSS_TIME_MACHINE = "oss_time_machine"
SUBSTRATE_TYPES = (OSS_TIME_MACHINE,)


@dataclass(frozen=True)
class HistoricalIssue:
    """A real historical issue extracted from the project's history (brief §5).

    ``hidden=True`` marks a *held-out* issue used only for transfer evaluation — it must NOT be
    released during the early capability-formation phase.

    ``body`` is the AGENT-VISIBLE text (the rewritten / user-style phrasing). The provenance fields
    (``source_url`` / ``source_id`` / ``text_original`` / ``labels`` / ``release`` …) are evaluator-
    side metadata: they identify the real GitHub PR/issue/release the friction came from and may
    reveal the future fix, so they are NOT placed on the agent-visible product artifact.
    """

    issue_id: str
    title: str                        # AGENT-VISIBLE: a user-facing problem statement (NOT the PR title)
    body: str
    issue_type: str = "bug"          # bug | docs | usability | performance | compatibility | feature
    severity: str = "medium"          # minor | medium | major | critical
    component: str = ""
    created_tick: int = 0
    source: str = "historical_github"
    hidden: bool = False              # held-out (transfer test), not released during formation
    acceptance_hint: str = ""         # agent-visible, BEHAVIORAL acceptance (never the fix/solution)
    linked_hidden_test_ids: List[str] = field(default_factory=list)
    # provenance (evaluator-side; real GitHub linkage — requirement: source_url/id, original text, …)
    source_url: str = ""
    source_id: str = ""
    source_type: str = ""             # pull_request | issue | release_note | commit
    created_at: str = ""
    labels: List[str] = field(default_factory=list)
    title_original: str = ""          # the real upstream PR/issue title (reveals the fix) — evaluator only
    text_original: str = ""           # the raw upstream text (may reveal identity/fix) — evaluator only
    text_rewritten: str = ""          # the agent-visible rewrite (== body)
    release: str = ""                 # which release the fix landed in
    # Repo paths the issue is expected to touch. Machine-generated datasets name
    # ``component`` logically ("public_node_api") and carry the real paths here,
    # so keeping the hints is what lets a component be resolved when the manifest
    # declares no component_map.
    candidate_path_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}

    def provenance(self) -> Dict[str, Any]:
        """Evaluator-side provenance record (never exposed to agents)."""
        return {
            "issue_id": self.issue_id, "source_url": self.source_url, "source_id": self.source_id,
            "source_type": self.source_type, "created_at": self.created_at, "labels": list(self.labels),
            "title_original": self.title_original, "text_original": self.text_original,
            "release": self.release, "component": self.component,
        }


@dataclass(frozen=True)
class HiddenTestSpec:
    """An evaluator-only behavior test (brief §5). ``evaluator_only`` MUST stay True — its source
    is never exported into the agent-visible repo, perception, search, prompts or normal snapshot."""

    test_id: str
    name: str
    command: List[str]
    issue_ids: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    rel_path: str = ""               # path inside the dataset's hidden-tests dir
    evaluator_only: bool = True
    introduced_in: str = ""          # upstream release that first contains the behavior
    source_url: str = ""             # evaluator-only provenance for the behavior

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class OSSSubstrateSpec:
    """Resolved dataset paths + manifest for an OSS time-machine substrate (brief §5)."""

    project_id: str
    product_name: str
    dataset_dir: str
    starter_repo_dir: str
    reference_repo_dir: str          # evaluator-only; never exported to agent-visible surfaces
    public_issues_dir: str
    heldout_issues_dir: str
    hidden_tests_dir: str            # evaluator-only
    public_tests_dir: str = ""
    external_signals_dir: str = ""
    contamination_dir: str = ""
    manifest: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        """A summary safe to surface (counts / ids only — NO reference or hidden-test content)."""
        return {
            "project_id": self.project_id,
            "product_name": self.product_name,
            "dataset_dir": self.dataset_dir,
            "starter_repo_dir": self.starter_repo_dir,
        }


__all__ = [
    "OSS_TIME_MACHINE", "SUBSTRATE_TYPES",
    "HistoricalIssue", "HiddenTestSpec", "OSSSubstrateSpec",
]
