"""OrgEnv replay formatter — stub (conforms to DomainReplayFormatter).

Renders a DomainState into a JSON-serializable frame for the Lived Inspector
(core §28). SKELETON: minimal frame; rich org views (board/budget/protocol
lifecycle) land in Stage O2+.
"""
from __future__ import annotations

from typing import Any, Dict

from agent_sdk.lived.domain.interfaces import DomainState


class OrgReplayFormatter:
    def format_frame(self, *, state: DomainState) -> Dict[str, Any]:
        return {
            "domain": "org",
            "run_id": state.run_id,
            "tick": state.world_tick,
            "agents": sorted(state.agents.keys()),
            "entities": len(state.entities),
            "public_records": len(state.public_records),
        }


__all__ = ["OrgReplayFormatter"]
