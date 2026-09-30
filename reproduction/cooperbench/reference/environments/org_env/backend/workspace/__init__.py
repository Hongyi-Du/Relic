"""OrgEnv workspace + object system (DESIGN env_org §21/§24/§25/§39/§40).

  provenance.py — content_hash + Provenance chain
  objects.py    — Visibility / WorkspaceObject / FileObject (visibility resolution)
  company.py    — CompanyWorkspace (shared, member-scoped)
  personal.py   — PersonalWorkspace (per-agent private)
"""
from environments.org_env.backend.workspace.company import CompanyWorkspace
from environments.org_env.backend.workspace.objects import (
    FileObject,
    Visibility,
    WorkspaceObject,
)
from environments.org_env.backend.workspace.personal import PersonalWorkspace
from environments.org_env.backend.workspace.provenance import (
    Provenance,
    ProvenanceStep,
    content_hash,
)

__all__ = [
    "content_hash", "Provenance", "ProvenanceStep",
    "Visibility", "WorkspaceObject", "FileObject",
    "CompanyWorkspace", "PersonalWorkspace",
]
