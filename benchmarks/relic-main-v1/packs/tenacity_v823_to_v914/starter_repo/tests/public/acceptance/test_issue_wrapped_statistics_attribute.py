"""Acceptance check for issue_wrapped_statistics_attribute.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: After calling a decorated function, synchronous or asynchronous, `fn.statistics['attempt_number']` reports the attempts that particular call took and is refreshed by the next call; `fn.retry` is still the configuration object, patching it still changes behaviour, and `fn.retry_with` still produces a reconfigured variant.
"""
import asyncio

import pytest
from tenacity import RetryError, retry, stop_after_attempt, wait_none


def test_sync_retry_statistics_are_published_on_wrapper_and_retry_remains_configurable():
    state = {"failures": 1, "attempts": 0}

    @retry(stop=stop_after_attempt(3), wait=wait_none())
    def operation():
        state["attempts"] += 1
        if state["failures"]:
            state["failures"] -= 1
            raise ValueError("try again")
        return "done"

    retry_configuration = operation.retry

    assert operation() == "done"
    assert operation.statistics["attempt_number"] == 2
    assert "idle_for" in operation.statistics
    assert "start_time" in operation.statistics

    state["failures"] = 0
    state["attempts"] = 0
    assert operation() == "done"
    assert operation.statistics["attempt_number"] == 1

    assert operation.retry is retry_configuration
    operation.retry.stop = stop_after_attempt(1)
    state["failures"] = 1
    state["attempts"] = 0
    with pytest.raises(RetryError):
        operation()
    assert state["attempts"] == 1

    reconfigured = operation.retry_with(stop=stop_after_attempt(2))
    state["failures"] = 1
    state["attempts"] = 0
    assert reconfigured() == "done"
    assert state["attempts"] == 2


def test_async_retry_statistics_are_published_on_wrapper():
    state = {"failures": 1}

    @retry(stop=stop_after_attempt(3), wait=wait_none())
    async def operation():
        if state["failures"]:
            state["failures"] -= 1
            raise ValueError("try again")
        return "done"

    assert asyncio.run(operation()) == "done"
    assert operation.statistics["attempt_number"] == 2
    assert "idle_for" in operation.statistics
    assert "start_time" in operation.statistics

    state["failures"] = 0
    assert asyncio.run(operation()) == "done"
    assert operation.statistics["attempt_number"] == 1
