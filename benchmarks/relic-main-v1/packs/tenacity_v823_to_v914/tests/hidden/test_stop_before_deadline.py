"""Withheld contract: a deadline-bounded stop condition gives up before overrunning its budget.

The retry loop is driven through the public ``tenacity.Retrying`` entry point with an
injected sleep function, so the contract observes only how many attempts happened and how
much delay was requested -- never how the decision is computed.
"""

from __future__ import annotations

import time

import pytest

import tenacity

DEADLINE = 0.5
STEP = 0.2


def _drive(stop):
    requested_delays = []
    attempts = []

    def record_and_sleep(seconds):
        requested_delays.append(float(seconds))
        time.sleep(float(seconds))

    def always_fails():
        attempts.append(1)
        raise ValueError("upstream unavailable")

    retrying = tenacity.Retrying(
        stop=stop,
        wait=tenacity.wait_fixed(STEP),
        sleep=record_and_sleep,
        reraise=True,
    )

    with pytest.raises(ValueError):
        retrying(always_fails)

    return attempts, requested_delays


def test_the_delay_budget_is_never_overrun() -> None:
    attempts, requested_delays = _drive(tenacity.stop_before_delay(DEADLINE))

    assert len(attempts) == 3
    assert requested_delays == [STEP, STEP]
    assert sum(requested_delays) < DEADLINE


def test_the_permissive_deadline_keeps_overrunning_as_before() -> None:
    attempts, requested_delays = _drive(tenacity.stop_after_delay(DEADLINE))

    assert len(attempts) == 4
    assert requested_delays == [STEP, STEP, STEP]
    assert sum(requested_delays) > DEADLINE


def test_attempt_capped_stopping_is_unaffected() -> None:
    attempts, requested_delays = _drive(tenacity.stop_after_attempt(2))

    assert len(attempts) == 2
    assert requested_delays == [STEP]
