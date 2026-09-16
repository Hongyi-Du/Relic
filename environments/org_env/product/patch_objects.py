"""Execution-layer patch objects (v4 §2.2-2.3).

ActionDecision decides *what* to do; the execution LLM (DocEditorLLM / CodeEditorLLM)
decides *how* to change the artifact, producing a concrete, inspectable patch. The
PatchValidator decides whether the patch may enter the world. A patch is the ONLY way
a product artifact's revision is bumped (no more bare ``revision += 1``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DOC_PATCH_TYPES = ("doc_patch", "doc_create", "doc_review", "doc_audit")
CODE_PATCH_TYPES = ("code_patch", "schema_change", "validator_change", "stub_update")


@dataclass
class DocPatch:
    patch_id: str
    target_object_id: str
    actor_id: str
    tick: int
    patch_type: str = "doc_patch"
    edit_goal: str = ""
    new_content: str = ""            # full updated file text (real document content)
    unified_diff: str = ""           # real old->new diff, computed on apply
    changed_sections: List[Dict[str, str]] = field(default_factory=list)  # [{section,before,after}]
    change_summary: str = ""
    removed_overclaims: List[str] = field(default_factory=list)
    added_limitations: List[str] = field(default_factory=list)
    added_requirements: List[str] = field(default_factory=list)
    remaining_risks: List[str] = field(default_factory=list)
    # doc_create content (v4 §1): the actual sections/items a new doc introduces
    checklist_items: List[str] = field(default_factory=list)
    workflow_sections: List[str] = field(default_factory=list)
    # which of the artifact's known_gaps this patch actually resolves (v4 §4) — only
    # these are removed on apply, instead of blindly popping the first gap.
    resolved_gaps: List[str] = field(default_factory=list)
    related_issue_ids: List[str] = field(default_factory=list)
    related_task_ids: List[str] = field(default_factory=list)
    related_episode_ids: List[str] = field(default_factory=list)
    related_wish_ids: List[str] = field(default_factory=list)
    related_proposal_ids: List[str] = field(default_factory=list)
    validation_status: str = "pending"            # pending / accepted / rejected
    rejection_reason: Optional[str] = None
    applied_tick: Optional[int] = None            # v8 #4: tick the patch entered the artifact
    # Appended so replay of positional legacy constructors keeps field order.
    creates_file: bool = False        # patch introduces a repository path
    base_mainline_revision: int = 0   # optimistic-concurrency base for merge/replay

    def is_empty(self) -> bool:
        return not (self.new_content or self.changed_sections or self.removed_overclaims
                    or self.added_limitations or self.added_requirements or self.checklist_items
                    or self.workflow_sections)

    def to_dict(self) -> Dict[str, Any]:
        return _patch_to_dict(self)


@dataclass
class CodePatch:
    patch_id: str
    target_object_id: str
    actor_id: str
    tick: int
    patch_type: str = "code_patch"
    edit_goal: str = ""
    new_content: str = ""            # full updated file text (real source code)
    unified_diff: str = ""           # real old->new diff, computed on apply
    files_changed: List[str] = field(default_factory=list)
    pseudo_diff: str = ""
    change_summary: str = ""
    added_fields: List[str] = field(default_factory=list)
    added_checks: List[str] = field(default_factory=list)
    changed_behavior: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    resolved_gaps: List[str] = field(default_factory=list)   # v4 §4
    related_issue_ids: List[str] = field(default_factory=list)
    related_task_ids: List[str] = field(default_factory=list)
    related_episode_ids: List[str] = field(default_factory=list)
    related_wish_ids: List[str] = field(default_factory=list)
    related_proposal_ids: List[str] = field(default_factory=list)
    validation_status: str = "pending"
    rejection_reason: Optional[str] = None
    applied_tick: Optional[int] = None            # v8 #4: tick the patch entered the artifact
    # Appended so replay of positional legacy constructors keeps field order.
    creates_file: bool = False        # patch introduces a repository path
    base_mainline_revision: int = 0   # optimistic-concurrency base for merge/replay

    def is_empty(self) -> bool:
        return not (self.new_content or self.pseudo_diff or self.added_fields
                    or self.added_checks or self.changed_behavior)

    def to_dict(self) -> Dict[str, Any]:
        return _patch_to_dict(self)


def _patch_to_dict(p) -> Dict[str, Any]:
    """v8 #4: dump the patch with canonical, self-consistent telemetry keys so a
    patch is never missing artifact_id / status / created_tick / applied_tick."""
    d = {k: (list(v) if isinstance(v, list) else v) for k, v in p.__dict__.items()}
    d["artifact_id"] = p.target_object_id        # canonical alias of target_object_id
    d["status"] = p.validation_status            # canonical alias of validation_status
    d["created_tick"] = p.tick                   # patch birth tick
    d.setdefault("applied_tick", getattr(p, "applied_tick", None))
    # Old pickles do not have the Gate 1 creation fields in ``__dict__``.
    # Persist their conservative defaults explicitly when the object is next
    # snapshotted so replay never has to infer create semantics.
    d.setdefault("creates_file", False)
    d.setdefault("base_mainline_revision", 0)
    return d


__all__ = ["DocPatch", "CodePatch", "DOC_PATCH_TYPES", "CODE_PATCH_TYPES"]
