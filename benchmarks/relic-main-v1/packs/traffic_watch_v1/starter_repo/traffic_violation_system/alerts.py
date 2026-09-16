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
        raise NotImplementedError('to_payload is not implemented yet')

class AlertDispatcher:
    """Holds a rolling live-feed buffer and optionally POSTs to a webhook.

    The webhook poster is injected so tests can pass a fake.
    """

    def __init__(self, webhook_url: Optional[str]=None, webhook_poster: Optional[Callable[[str, dict], None]]=None, feed_size: int=100) -> None:
        raise NotImplementedError('__init__ is not implemented yet')

    def dispatch(self, record: ViolationRecord) -> Alert:
        raise NotImplementedError('dispatch is not implemented yet')

    def feed(self) -> List[Alert]:
        raise NotImplementedError('feed is not implemented yet')
