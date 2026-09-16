"""Acceptance check for issue_stop_before_deadline.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: With a budget D and a fixed pause P, a run driven by `stop_before_delay(D)` stops at the last attempt whose elapsed time plus P is still below D, so the total pause time requested never reaches D; `stop_after_delay` still overshoots and the attempt-count stop is unaffected.
"""
from datetime import timedelta
from types import SimpleNamespace

from tenacity import stop_after_attempt, stop_after_delay, stop_before_delay


def test_stop_before_delay_stops_before_a_fixed_pause_reaches_budget():
    budget = timedelta(seconds=10)
    pause = 3
    stop = stop_before_delay(budget)

    requested_pauses = []
    for elapsed in (0, 3, 6):
        state = SimpleNamespace(seconds_since_start=elapsed, upcoming_sleep=pause)
        assert stop(state) is False
        requested_pauses.append(pause)

    state = SimpleNamespace(seconds_since_start=9, upcoming_sleep=pause)
    assert stop(state) is True
    assert sum(requested_pauses) == 9
    assert sum(requested_pauses) < budget.total_seconds()


def test_existing_delay_and_attempt_stops_keep_their_behavior():
    state = SimpleNamespace(seconds_since_start=9, upcoming_sleep=3, attempt_number=2)

    assert stop_after_delay(10)(state) is False
    assert stop_after_attempt(3)(state) is False

    state.attempt_number = 3
    assert stop_after_attempt(3)(state) is True
