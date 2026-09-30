"""PersonalWorkspace — per-agent private workspace (DESIGN env_org §25/§40).

NOT public: other agents cannot see its contents unless the owner explicitly
shares an object into the company workspace / a channel (O-Infra-2). This is the
source of "I have a local result but haven't shared it" friction (§26/最终原则).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from environments.org_env.backend.workspace.objects import FileObject


@dataclass
class PersonalWorkspace:
    agent_id: str
    personal_workspace_id: str = ""
    personal_notes: List[str] = field(default_factory=list)
    private_todos: List[str] = field(default_factory=list)
    local_files: Dict[str, FileObject] = field(default_factory=dict)
    draft_docs: List[str] = field(default_factory=list)
    saved_post_ids: List[str] = field(default_factory=list)
    downloaded_doc_ids: List[str] = field(default_factory=list)
    local_branch_ids: List[str] = field(default_factory=list)
    local_experiment_ids: List[str] = field(default_factory=list)
    sandbox_id: Optional[str] = None
    open_questions: List[str] = field(default_factory=list)
    current_focus: Optional[str] = None
    external_contacts: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.personal_workspace_id:
            self.personal_workspace_id = f"pw_{self.agent_id}"

    def add_local_file(self, f: FileObject) -> FileObject:
        """Store a private file (owner-only until shared)."""
        f.owner_id = f.owner_id or self.agent_id
        self.local_files[f.object_id] = f
        return f


__all__ = ["PersonalWorkspace"]
