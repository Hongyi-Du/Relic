"""Per-user interaction analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol

from tg_automation.retry import retry_with_backoff

LOG = logging.getLogger(__name__)


@dataclass
class UserInteraction:
    user_id: int
    username: str
    messages_sent: int = 0
    reactions_given: int = 0
    last_active_at: Optional[datetime] = None
    active_days: int = 0
    _seen_days: set = field(default_factory=set, repr=False, compare=False)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "messages_sent": self.messages_sent,
            "reactions_given": self.reactions_given,
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else "",
            "active_days": self.active_days,
        }


class TelegramSource(Protocol):
    def iter_messages(self, group_id: str, since: datetime, until: datetime) -> Iterable[Dict[str, Any]]: ...


def _coerce_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)


def analyze_interactions(
    group_id: str,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    source: Optional[TelegramSource] = None,
    now: Optional[datetime] = None,
) -> List[UserInteraction]:
    """Return per-user interaction records for ``group_id`` over ``[since, until)``.

    Defaults to the trailing 30 days when the window is unspecified.
    """
    if source is None:
        raise ValueError("analyze_interactions requires a 'source' implementing iter_messages")

    now = now or datetime.now(timezone.utc)
    if until is None:
        until = now
    if since is None:
        since = until - timedelta(days=30)

    if since >= until:
        raise ValueError("'since' must precede 'until'")

    records: Dict[int, UserInteraction] = {}

    def _pull() -> List[Dict[str, Any]]:
        return list(source.iter_messages(group_id, since, until))

    messages = retry_with_backoff(_pull)
    for msg in messages:
        ts = _coerce_dt(msg["date"])
        if ts < since or ts >= until:
            continue
        user_id = int(msg["sender_id"])
        username = str(msg.get("username", ""))
        rec = records.get(user_id)
        if rec is None:
            rec = UserInteraction(user_id=user_id, username=username)
            records[user_id] = rec
        kind = msg.get("kind", "message")
        if kind == "message":
            rec.messages_sent += 1
        elif kind == "reaction":
            rec.reactions_given += int(msg.get("count", 1))
        else:
            continue
        day_key = ts.date().isoformat()
        rec._seen_days.add(day_key)
        if rec.last_active_at is None or ts > rec.last_active_at:
            rec.last_active_at = ts
        if username and not rec.username:
            rec.username = username

    for rec in records.values():
        rec.active_days = len(rec._seen_days)

    return sorted(records.values(), key=lambda r: r.user_id)
