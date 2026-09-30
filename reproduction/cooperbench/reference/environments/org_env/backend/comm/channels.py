"""Channels (DESIGN env_org §16/§38)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

# Default company channels (§16).
DEFAULT_CHANNELS = (
    "team_general", "engineering", "experiments", "customer_feedback",
    "launch", "external_signals", "incidents", "random",
)


@dataclass
class Channel:
    channel_id: str
    channel_type: str = "team_general"
    members: Set[str] = field(default_factory=set)
    message_ids: List[str] = field(default_factory=list)
    pinned_item_ids: List[str] = field(default_factory=list)

    def add_member(self, agent_id: str) -> None:
        self.members.add(agent_id)


__all__ = ["DEFAULT_CHANNELS", "Channel"]
