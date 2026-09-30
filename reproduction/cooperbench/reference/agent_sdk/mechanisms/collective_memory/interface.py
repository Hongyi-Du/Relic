"""
CollectiveMemoryPort — protocol for pluggable collective memory backends.

The Agent SDK interacts with collective memory only through this interface.
Concrete implementation: agent_sdk.mechanisms.collective_memory.store.AssetStore
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from .types import Gene, Event


@runtime_checkable
class CollectiveMemoryPort(Protocol):
    """
    Minimal contract for Gene+Event collective memory.

    Methods map to agent skills:
      upload_event  → upload evidence of strategy application
      upload_gene   → upload reusable strategy (passive: called by CompactionEngine)
      apply_vote    → like/dislike (all assets, drives confidence + vote-based deprecation)
      get_gene / get_event → direct access
    """

    def upload_event(
        self, agent_id: str, event_data: Dict[str, Any], turn: int,
    ) -> Event: ...

    def upload_gene(
        self, agent_id: str, gene_data: Dict[str, Any], turn: int,
    ) -> Gene: ...

    def apply_vote(
        self, asset_id: str, vote_type: str, voter_id: str,
        turn: int, reason: str = "",
    ) -> float: ...

    def get_gene(self, asset_id: str) -> Optional[Gene]: ...

    def get_event(self, asset_id: str) -> Optional[Event]: ...

    def get_events_for_gene(self, gene_id: str) -> List[Event]: ...

    def to_dict(self) -> Dict[str, Any]: ...
