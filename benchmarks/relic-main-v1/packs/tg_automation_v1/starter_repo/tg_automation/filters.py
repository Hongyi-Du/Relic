"""Filter user records by activity mode."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence
from tg_automation.analysis import UserInteraction
_ALLOWED_FIELDS = {'messages_sent', 'reactions_given', 'active_days'}
_OPS = {'>=': lambda a, b: a >= b, '<=': lambda a, b: a <= b, '!=': lambda a, b: a != b, '==': lambda a, b: a == b, '>': lambda a, b: a > b, '<': lambda a, b: a < b, '=': lambda a, b: a == b}
_PRED_RE = re.compile('^\\s*(\\w+)\\s*(>=|<=|!=|==|>|<|=)\\s*(-?\\d+)\\s*$')

@dataclass
class _Predicate:
    field: str
    op: str
    value: int

    def __call__(self, rec: UserInteraction) -> bool:
        raise NotImplementedError('__call__ is not implemented yet')

def parse_custom_predicate(expr: str) -> Callable[[UserInteraction], bool]:
    """Parse a single comparator like ``messages_sent>=5``."""
    raise NotImplementedError('parse_custom_predicate is not implemented yet')

def filter_users(users: Iterable[UserInteraction], mode: str, *, custom: Sequence[str] | None=None, active_threshold: int=1) -> List[UserInteraction]:
    """Filter ``users`` by ``mode``.

    Modes:
        * ``active``   - records with messages_sent + reactions_given >= active_threshold.
        * ``inactive`` - the complement of ``active``.
        * ``custom``   - all predicates in ``custom`` must hold.
    """
    raise NotImplementedError('filter_users is not implemented yet')
