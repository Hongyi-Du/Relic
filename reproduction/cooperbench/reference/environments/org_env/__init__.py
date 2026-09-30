"""environments.org_env — OrgEnv organization domain (DESIGN env_org).

A second SocioGenesis domain alongside nature_env: an organization (internal
full-lived company agents) embedded in an external professional community
(lightweight community agents). Shares the env-agnostic lived Core via the
``agent_sdk.lived.domain`` adapter seam (DESIGN core §32).

Directory architecture mirrors nature_env (see DESIGN env_org §35.1):
  backend/         organization internals
    entities/      work · economy · governance objects
    community/     external professional-community layer
    agents/        internal full-lived agents
    actions/       action registry
    simulation/    the organization world
  runtime_adapter/ DomainAdapter implementation (perception/execution/feature/replay)
  config/          scenarios + metrics

Status: 📐 skeleton — structure + frozen schema + adapter stubs. Behaviour lands
in Stage O2–O5 (DESIGN env_org §35.2). Entry point: ``OrgEnvAdapter``.
"""
from environments.org_env.runtime_adapter.adapter import OrgEnvAdapter

__all__ = ["OrgEnvAdapter"]
