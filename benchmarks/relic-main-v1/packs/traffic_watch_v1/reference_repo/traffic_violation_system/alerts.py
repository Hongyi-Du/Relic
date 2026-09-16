"""Real-time alert dispatching: live in-memory feed + optional webhook."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional

from .violations import ViolationRecord


@dataclass
class Alert:
    record: ViolationRecord

    def to_payload(self) -> dict:
        return self.record.to_dict()


class AlertDispatcher:
    """Holds a rolling live-feed buffer and optionally POSTs to a webhook.

    The webhook poster is injected so tests can pass a fake.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        webhook_poster: Optional[Callable[[str, dict], None]] = None,
        feed_size: int = 100,
    ) -> None:
        self._webhook_url = webhook_url
        self._poster = webhook_poster
        self._feed: Deque[Alert] = deque(maxlen=feed_size)

    def dispatch(self, record: ViolationRecord) -> Alert:
        alert = Alert(record=record)
        self._feed.append(alert)
        if self._webhook_url and self._poster is not None:
            try:
                self._poster(self._webhook_url, alert.to_payload())
            except Exception:
                pass
        return alert

    def feed(self) -> List[Alert]:
        return list(self._feed)