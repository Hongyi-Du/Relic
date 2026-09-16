"""Engagement metrics across posts and users."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol

@dataclass
class PostEngagement:
    group_id: str
    post_id: int
    author_id: int
    author_username: str
    date: str
    reactions_by_emoji: Dict[str, int] = field(default_factory=dict)
    replies: int = 0
    comments: int = 0

    def as_dict(self) -> Dict[str, Any]:
        raise NotImplementedError('as_dict is not implemented yet')

class EngagementSource(Protocol):

    def iter_posts(self, group_id: str, since: datetime, until: datetime) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError('iter_posts is not implemented yet')

def _coerce_dt(value: Any) -> datetime:
    raise NotImplementedError('_coerce_dt is not implemented yet')

def engagement_report(group_ids: Iterable[str], since: Optional[datetime]=None, until: Optional[datetime]=None, source: Optional[EngagementSource]=None, now: Optional[datetime]=None) -> Dict[str, Any]:
    """Emit per-post engagement and per-user message frequency across groups."""
    raise NotImplementedError('engagement_report is not implemented yet')
