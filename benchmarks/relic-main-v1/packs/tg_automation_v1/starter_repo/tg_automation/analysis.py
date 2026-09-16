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
        raise NotImplementedError('as_dict is not implemented yet')

class TelegramSource(Protocol):

    def iter_messages(self, group_id: str, since: datetime, until: datetime) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError('iter_messages is not implemented yet')

def _coerce_dt(value: Any) -> datetime:
    raise NotImplementedError('_coerce_dt is not implemented yet')

def analyze_interactions(group_id: str, since: Optional[datetime]=None, until: Optional[datetime]=None, source: Optional[TelegramSource]=None, now: Optional[datetime]=None) -> List[UserInteraction]:
    """Return per-user interaction records for ``group_id`` over ``[since, until)``.

    Defaults to the trailing 30 days when the window is unspecified.
    """
    raise NotImplementedError('analyze_interactions is not implemented yet')
