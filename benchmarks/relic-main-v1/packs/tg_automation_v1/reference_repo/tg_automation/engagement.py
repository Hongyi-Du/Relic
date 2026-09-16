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
        return {
            "group_id": self.group_id,
            "post_id": self.post_id,
            "author_id": self.author_id,
            "author_username": self.author_username,
            "date": self.date,
            "reactions_by_emoji": dict(self.reactions_by_emoji),
            "reactions_total": sum(self.reactions_by_emoji.values()),
            "replies": self.replies,
            "comments": self.comments,
        }


class EngagementSource(Protocol):
    def iter_posts(self, group_id: str, since: datetime, until: datetime) -> Iterable[Dict[str, Any]]: ...


def _coerce_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)


def engagement_report(
    group_ids: Iterable[str],
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    source: Optional[EngagementSource] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Emit per-post engagement and per-user message frequency across groups."""
    if source is None:
        raise ValueError("engagement_report requires a 'source' implementing iter_posts")

    now = now or datetime.now(timezone.utc)
    if until is None:
        until = now
    if since is None:
        since = until - timedelta(days=30)

    posts: List[PostEngagement] = []
    user_message_count: Counter[int] = Counter()
    user_username: Dict[int, str] = {}

    for group_id in group_ids:
        for raw in source.iter_posts(group_id, since, until):
            ts = _coerce_dt(raw["date"])
            if ts < since or ts >= until:
                continue
            author_id = int(raw["author_id"])
            username = str(raw.get("author_username", ""))
            user_message_count[author_id] += 1
            if username:
                user_username[author_id] = username
            reactions_in = raw.get("reactions", {}) or {}
            reactions = {str(k): int(v) for k, v in reactions_in.items()}
            replies = int(raw.get("replies", 0))
            comments = int(raw.get("comments", 0))
            posts.append(
                PostEngagement(
                    group_id=group_id,
                    post_id=int(raw["post_id"]),
                    author_id=author_id,
                    author_username=username,
                    date=ts.isoformat(),
                    reactions_by_emoji=reactions,
                    replies=replies,
                    comments=comments,
                )
            )

    window_days = max(1, (until - since).days or 1)
    user_freq = [
        {
            "user_id": uid,
            "username": user_username.get(uid, ""),
            "messages_sent": cnt,
            "messages_per_day": round(cnt / window_days, 4),
        }
        for uid, cnt in sorted(user_message_count.items())
    ]
    return {
        "posts": [p.as_dict() for p in posts],
        "user_frequency": user_freq,
        "window_start": since.isoformat(),
        "window_end": until.isoformat(),
    }
