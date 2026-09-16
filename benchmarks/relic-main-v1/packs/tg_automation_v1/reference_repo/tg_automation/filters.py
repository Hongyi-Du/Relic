"""Filter user records by activity mode."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

from tg_automation.analysis import UserInteraction


_ALLOWED_FIELDS = {"messages_sent", "reactions_given", "active_days"}
_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "!=": lambda a, b: a != b,
    "==": lambda a, b: a == b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "=": lambda a, b: a == b,
}
_PRED_RE = re.compile(r"^\s*(\w+)\s*(>=|<=|!=|==|>|<|=)\s*(-?\d+)\s*$")


@dataclass
class _Predicate:
    field: str
    op: str
    value: int

    def __call__(self, rec: UserInteraction) -> bool:
        return _OPS[self.op](getattr(rec, self.field), self.value)


def parse_custom_predicate(expr: str) -> Callable[[UserInteraction], bool]:
    """Parse a single comparator like ``messages_sent>=5``."""
    m = _PRED_RE.match(expr)
    if not m:
        raise ValueError(f"invalid predicate: {expr!r}")
    field, op, raw_value = m.group(1), m.group(2), m.group(3)
    if field not in _ALLOWED_FIELDS:
        raise ValueError(f"unknown field: {field}")
    return _Predicate(field=field, op=op, value=int(raw_value))


def filter_users(
    users: Iterable[UserInteraction],
    mode: str,
    *,
    custom: Sequence[str] | None = None,
    active_threshold: int = 1,
) -> List[UserInteraction]:
    """Filter ``users`` by ``mode``.

    Modes:
        * ``active``   - records with messages_sent + reactions_given >= active_threshold.
        * ``inactive`` - the complement of ``active``.
        * ``custom``   - all predicates in ``custom`` must hold.
    """
    if mode not in ("active", "inactive", "custom"):
        raise ValueError(f"unknown filter mode: {mode}")

    users = list(users)
    if mode == "active":
        return [u for u in users if (u.messages_sent + u.reactions_given) >= active_threshold]
    if mode == "inactive":
        return [u for u in users if (u.messages_sent + u.reactions_given) < active_threshold]
    if not custom:
        raise ValueError("custom mode requires at least one predicate")
    predicates = [parse_custom_predicate(c) for c in custom]
    return [u for u in users if all(p(u) for p in predicates)]
