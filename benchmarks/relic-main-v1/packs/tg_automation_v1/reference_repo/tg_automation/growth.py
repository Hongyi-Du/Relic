"""Group growth: invite links, bulk invites, trend dashboard."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol

from tg_automation.retry import FloodWaitError, NetworkError, retry_with_backoff

LOG = logging.getLogger(__name__)


@dataclass
class InviteLink:
    group_id: str
    url: str
    expires_at: Optional[datetime] = None
    join_limit: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "url": self.url,
            "expires_at": self.expires_at.isoformat() if self.expires_at else "",
            "join_limit": self.join_limit if self.join_limit is not None else "",
        }


@dataclass
class InviteResult:
    group_id: str
    username: str
    status: str
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "username": self.username,
            "status": self.status,
            "error": self.error,
        }


class GrowthClient(Protocol):
    def create_invite_link(
        self, group_id: str, expires_at: Optional[datetime], join_limit: Optional[int]
    ) -> str: ...

    def invite_user(self, group_id: str, username: str) -> None: ...

    def iter_membership_events(
        self, group_id: str, since: datetime, until: datetime
    ) -> Iterable[Dict[str, Any]]: ...


def create_invite_link(
    group_id: str,
    client: GrowthClient,
    *,
    expires_at: Optional[datetime] = None,
    join_limit: Optional[int] = None,
) -> InviteLink:
    if join_limit is not None and join_limit <= 0:
        raise ValueError("join_limit must be positive")
    url = retry_with_backoff(client.create_invite_link, group_id, expires_at, join_limit)
    return InviteLink(group_id=group_id, url=url, expires_at=expires_at, join_limit=join_limit)


def _read_usernames_csv(path: str) -> List[tuple[str, str]]:
    rows: List[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "username" not in reader.fieldnames:
            raise ValueError("bulk-invite CSV must have a 'username' column")
        has_group = "group_id" in reader.fieldnames
        for row in reader:
            uname = (row.get("username") or "").strip()
            if not uname:
                continue
            gid = (row.get("group_id") or "").strip() if has_group else ""
            rows.append((gid, uname))
    return rows


def schedule_bulk_invites(
    group_id: str,
    csv_path: str,
    client: GrowthClient,
) -> List[InviteResult]:
    """Invite every username in ``csv_path`` to ``group_id``.

    The CSV must contain a ``username`` column; if it also contains a
    ``group_id`` column, that overrides the default ``group_id`` per row.
    FloodWaitError is honored via retry_with_backoff. NetworkError retries
    up to 5 attempts then logs and skips.
    """
    rows = _read_usernames_csv(csv_path)
    results: List[InviteResult] = []
    for default_gid, username in rows:
        target_gid = default_gid or group_id
        try:
            retry_with_backoff(client.invite_user, target_gid, username)
            results.append(InviteResult(group_id=target_gid, username=username, status="invited"))
        except NetworkError as exc:
            LOG.error("giving up on %s after retries: %s", username, exc)
            results.append(
                InviteResult(
                    group_id=target_gid, username=username, status="skipped", error=str(exc)
                )
            )
        except FloodWaitError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOG.error("skipping %s after invite error: %s", username, exc)
            results.append(
                InviteResult(
                    group_id=target_gid, username=username, status="skipped", error=str(exc)
                )
            )
    return results


def _coerce_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)


def trend_dashboard(
    group_id: str,
    client: GrowthClient,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Return a list of daily ``{date, new_members, active_members}`` rows."""
    now = now or datetime.now(timezone.utc)
    if until is None:
        until = now
    if since is None:
        since = until - timedelta(days=30)
    if since >= until:
        raise ValueError("'since' must precede 'until'")

    events = list(retry_with_backoff(lambda: list(client.iter_membership_events(group_id, since, until))))
    by_day: Dict[str, Dict[str, set | int]] = {}
    cursor = since
    while cursor < until:
        by_day[cursor.date().isoformat()] = {"new": 0, "active": set()}
        cursor = cursor + timedelta(days=1)

    for ev in events:
        ts = _coerce_dt(ev["date"])
        if ts < since or ts >= until:
            continue
        day = ts.date().isoformat()
        bucket = by_day.setdefault(day, {"new": 0, "active": set()})
        kind = ev.get("kind")
        uid = int(ev.get("user_id", 0))
        if kind == "join":
            bucket["new"] = int(bucket["new"]) + 1  # type: ignore[operator]
        # Any membership/activity event counts the user as active for the day.
        bucket["active"].add(uid)  # type: ignore[union-attr]

    rows: List[Dict[str, Any]] = []
    for day in sorted(by_day):
        bucket = by_day[day]
        rows.append(
            {
                "date": day,
                "new_members": int(bucket["new"]),  # type: ignore[arg-type]
                "active_members": len(bucket["active"]),  # type: ignore[arg-type]
            }
        )
    return rows
