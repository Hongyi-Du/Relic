"""Workspace objects — WorkspaceObject base + FileObject (DESIGN env_org §21/§39).

Real behaviour: visibility resolution + content_hash + provenance. The
visibility model is the spine of OrgEnv's information asymmetry (acceptance ㉗,
§19/§70): a private object is visible only to its owner + explicitly granted
agents; team/public objects are visible to all company members.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from environments.org_env.backend.workspace.provenance import Provenance, content_hash


class Visibility(str, Enum):
    PRIVATE = "private"     # owner + explicitly-granted only
    TEAM = "team"           # all company members
    PUBLIC = "public"       # all members (alias of team inside the company)
    CHANNEL = "channel"     # members of a specific channel (set via location)


@dataclass
class WorkspaceObject:
    """Base for any shareable object. Tracks owner, visibility, an explicit
    ``granted_to`` set (for shares, O-Infra-2), provenance + content_hash."""
    object_id: str
    object_type: str = "object"
    owner_id: Optional[str] = None
    visibility: Visibility = Visibility.PRIVATE
    location: Optional[str] = None          # workspace id / channel id
    granted_to: Set[str] = field(default_factory=set)
    provenance: Provenance = field(default_factory=Provenance)
    content_hash: str = ""
    created_tick: int = 0
    last_modified_tick: int = 0

    def visible_to(self, agent_id: str, *, members: Optional[Set[str]] = None) -> bool:
        """Is ``agent_id`` allowed to see this object?

        private: owner or explicitly granted. team/public: any company member.
        channel: granted_to (channel membership). Unknown members set -> treat
        team/public as visible to anyone (skeleton-safe default).
        """
        if agent_id == self.owner_id or agent_id in self.granted_to:
            return True
        if self.visibility in (Visibility.TEAM, Visibility.PUBLIC):
            return members is None or agent_id in members
        return False  # PRIVATE / CHANNEL without grant

    def grant(self, agent_id: str) -> None:
        self.granted_to.add(agent_id)

    def recompute_hash(self, payload: Any) -> str:
        self.content_hash = content_hash(payload)
        return self.content_hash


@dataclass
class FileObject(WorkspaceObject):
    """Unified file/shareable-object model (§21). file_type drives downstream
    handling; raw_payload optional (summary + hash are authoritative)."""
    file_type: str = "doc"   # doc|markdown_note|json_result|experiment_report|cost_report|benchmark_config|meeting_note|customer_feedback|external_snapshot|proposal|checklist|tracker|ledger|presentation|code_patch_summary|launch_blog|protocol_doc
    title: str = ""
    creator_id: Optional[str] = None
    version: int = 1
    content_summary: str = ""
    raw_payload: Optional[Any] = None
    linked_task_ids: List[str] = field(default_factory=list)
    linked_experiment_ids: List[str] = field(default_factory=list)
    linked_protocol_ids: List[str] = field(default_factory=list)
    review_status: str = "draft"   # draft|in_review|approved|changes_requested|archived
    trust_level: float = 0.5

    def __post_init__(self):
        self.object_type = "file"
        if self.creator_id and not self.owner_id:
            self.owner_id = self.creator_id
        if not self.content_hash:
            self.recompute_hash({"title": self.title, "summary": self.content_summary,
                                 "payload": self.raw_payload, "version": self.version})


__all__ = ["Visibility", "WorkspaceObject", "FileObject"]
