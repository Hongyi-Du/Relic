"""Acceptance check for issue_async_retry_predicate.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: An asynchronous retried function configured with a coroutine `retry=` predicate stops as soon as that predicate resolves to false and returns the last result to the caller, and the predicate observes attempt numbers 1, 2, 3, … just as an ordinary predicate would.
"""
import asyncio

from tenacity import retry as tenacity_retry
from tenacity import stop_after_attempt


def test_async_retry_predicate_is_awaited_and_receives_attempt_numbers():
    observed_attempts = []

    async def should_retry(retry_state):
        observed_attempts.append(retry_state.attempt_number)
        return retry_state.outcome.result() < 3

    @tenacity_retry(retry=should_retry, stop=stop_after_attempt(5))
    async def operation():
        return len(observed_attempts) + 1

    assert asyncio.run(operation()) == 3
    assert observed_attempts == [1, 2, 3]
