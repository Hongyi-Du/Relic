"""Acceptance check for issue_trio_native_sleep.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: A retried coroutine with a non-zero wait completes under Trio with no sleep function configured anywhere, still completes under asyncio, and an explicitly supplied pause function is still the one that gets used.
"""
import asyncio

import trio
from tenacity import retry, stop_after_attempt, wait_fixed


def test_retry_with_nonzero_wait_uses_trio_when_no_sleep_is_configured():
    attempts = 0

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.01))
    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry")
        return "completed"

    async def run():
        return await operation()

    assert trio.run(run) == "completed"
    assert attempts == 2


def test_retry_with_nonzero_wait_still_works_under_asyncio():
    attempts = 0

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.01))
    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry")
        return "completed"

    async def run():
        return await operation()

    assert asyncio.run(run()) == "completed"
    assert attempts == 2


def test_retry_uses_explicitly_supplied_sleep_function():
    pauses = []
    attempts = 0

    async def supplied_sleep(delay):
        pauses.append(delay)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(0.01),
        sleep=supplied_sleep,
    )
    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry")
        return "completed"

    assert asyncio.run(operation()) == "completed"
    assert pauses == [0.01]
