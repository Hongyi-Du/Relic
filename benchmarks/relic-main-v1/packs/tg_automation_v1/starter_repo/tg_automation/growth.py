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
        raise NotImplementedError('as_dict is not implemented yet')

@dataclass
class InviteResult:
    group_id: str
    username: str
    status: str
    error: str = ''

    def as_dict(self) -> Dict[str, Any]:
        raise NotImplementedError('as_dict is not implemented yet')

class GrowthClient(Protocol):

    def create_invite_link(self, group_id: str, expires_at: Optional[datetime], join_limit: Optional[int]) -> str:
        raise NotImplementedError('create_invite_link is not implemented yet')

    def invite_user(self, group_id: str, username: str) -> None:
        raise NotImplementedError('invite_user is not implemented yet')

    def iter_membership_events(self, group_id: str, since: datetime, until: datetime) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError('iter_membership_events is not implemented yet')

def create_invite_link(group_id: str, client: GrowthClient, *, expires_at: Optional[datetime]=None, join_limit: Optional[int]=None) -> InviteLink:
    raise NotImplementedError('create_invite_link is not implemented yet')

def _read_usernames_csv(path: str) -> List[tuple[str, str]]:
    raise NotImplementedError('_read_usernames_csv is not implemented yet')

def schedule_bulk_invites(group_id: str, csv_path: str, client: GrowthClient) -> List[InviteResult]:
    """Invite every username in ``csv_path`` to ``group_id``.

    The CSV must contain a ``username`` column; if it also contains a
    ``group_id`` column, that overrides the default ``group_id`` per row.
    FloodWaitError is honored via retry_with_backoff. NetworkError retries
    up to 5 attempts then logs and skips.
    """
    raise NotImplementedError('schedule_bulk_invites is not implemented yet')

def _coerce_dt(value: Any) -> datetime:
    raise NotImplementedError('_coerce_dt is not implemented yet')

def trend_dashboard(group_id: str, client: GrowthClient, since: Optional[datetime]=None, until: Optional[datetime]=None, now: Optional[datetime]=None) -> List[Dict[str, Any]]:
    """Return a list of daily ``{date, new_members, active_members}`` rows."""
    raise NotImplementedError('trend_dashboard is not implemented yet')
