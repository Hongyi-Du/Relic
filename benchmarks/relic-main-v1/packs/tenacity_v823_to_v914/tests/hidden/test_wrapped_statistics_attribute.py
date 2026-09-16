"""Withheld contract: a decorated function reports the statistics of its own last run.

Only the decorated callable is inspected, so the contract passes for any bookkeeping
arrangement that keeps per-call statistics reachable from the function itself.
"""

from __future__ import annotations

import asyncio

import tenacity


def test_last_run_statistics_are_readable_from_the_decorated_function() -> None:
    attempts = []

    @tenacity.retry(
        retry=tenacity.retry_if_result(lambda value: value is None),
        wait=tenacity.wait_fixed(0),
        stop=tenacity.stop_after_attempt(6),
    )
    def poll():
        attempts.append(1)
        return "done" if len(attempts) >= 3 else None

    assert poll() == "done"
    assert poll.statistics["attempt_number"] == 3
    assert set(poll.statistics) >= {"attempt_number", "idle_for", "start_time"}


def test_statistics_are_refreshed_by_the_following_run() -> None:
    failures_left = [2]

    @tenacity.retry(
        retry=tenacity.retry_if_result(lambda value: value is None),
        wait=tenacity.wait_fixed(0),
        stop=tenacity.stop_after_attempt(6),
    )
    def poll():
        if failures_left[0] > 0:
            failures_left[0] -= 1
            return None
        return "done"

    assert poll() == "done"
    assert poll.statistics["attempt_number"] == 3

    assert poll() == "done"
    assert poll.statistics["attempt_number"] == 1


def test_async_decorated_functions_report_the_same_way() -> None:
    attempts = []

    @tenacity.retry(
        retry=tenacity.retry_if_result(lambda value: value is None),
        wait=tenacity.wait_fixed(0),
        stop=tenacity.stop_after_attempt(6),
    )
    async def poll():
        attempts.append(1)
        return "done" if len(attempts) >= 4 else None

    assert asyncio.run(poll()) == "done"
    assert poll.statistics["attempt_number"] == 4


def test_the_configuration_object_is_still_exposed_for_patching() -> None:
    @tenacity.retry(stop=tenacity.stop_after_attempt(4), wait=tenacity.wait_fixed(0))
    def always_fails():
        raise ValueError("nope")

    assert isinstance(always_fails.retry, tenacity.BaseRetrying)
    assert callable(always_fails.retry_with)

    always_fails.retry.stop = tenacity.stop_after_attempt(2)
    try:
        always_fails()
    except tenacity.RetryError as error:
        assert error.last_attempt.attempt_number == 2
    else:
        raise AssertionError("expected RetryError once the attempt budget ran out")
