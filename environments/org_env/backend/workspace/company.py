"""CompanyWorkspace — shared company workspace (DESIGN env_org §24/§40).

Holds shared/team-visible objects + the member set used by visibility checks.
Real behaviour: register files, resolve what an agent can see, share (grant).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from environments.org_env.backend.workspace.objects import FileObject, Visibility, WorkspaceObject


@dataclass
class CompanyWorkspace:
    workspace_id: str = "relic_org"
    company_name: str = "Relic organization"
    members: Set[str] = field(default_factory=set)
    files: Dict[str, FileObject] = field(default_factory=dict)
    shared_doc_ids: List[str] = field(default_factory=list)
    shared_artifact_ids: List[str] = field(default_factory=list)
    shared_board_ids: List[str] = field(default_factory=list)
    external_signal_board: List[str] = field(default_factory=list)
    workspace_events: List[dict] = field(default_factory=list)

    def add_member(self, agent_id: str) -> None:
        self.members.add(agent_id)

    def register_file(self, f: FileObject) -> FileObject:
        self.files[f.object_id] = f
        if f.visibility in (Visibility.TEAM, Visibility.PUBLIC) and f.file_type in ("doc", "tracker", "ledger", "checklist"):
            if f.object_id not in self.shared_doc_ids:
                self.shared_doc_ids.append(f.object_id)
        return f

    def share(self, object_id: str, agent_id: str, *, tick: int = 0, actor_id: str = "") -> bool:
        """Grant an agent visibility of a (private) object. Records provenance."""
        obj = self.files.get(object_id)
        if obj is None:
            return False
        obj.grant(agent_id)
        obj.provenance.add(tick=tick, actor_id=actor_id or "", action="shared",
                           note=f"granted_to {agent_id}")
        return True

    def visible_files_for(self, agent_id: str) -> List[FileObject]:
        return [f for f in self.files.values() if f.visible_to(agent_id, members=self.members)]


__all__ = ["CompanyWorkspace"]
