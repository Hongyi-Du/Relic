"""Acceptance check for issue_async_lifecycle_callbacks.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: On the asynchronous path, coroutine `before` / `after` / `before_sleep` hooks run in the same order and see the same attempt numbers as the equivalent ordinary hooks, and ordinary hooks keep working exactly as they do today.
"""
import asyncio

from tenacity import retry, stop_after_attempt, wait_none


def test_async_retry_hooks_are_awaited_in_attempt_order():
    events = []
    attempts = 0

    async def before(retry_state):
        events.append(("before", retry_state.attempt_number))

    async def after(retry_state):
        events.append(("after", retry_state.attempt_number))

    async def before_sleep(retry_state):
        events.append(("before_sleep", retry_state.attempt_number))

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_none(),
        before=before,
        after=after,
        before_sleep=before_sleep,
        reraise=True,
    )
    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry")
        return "done"

    assert asyncio.run(operation()) == "done"
    assert events == [
        ("before", 1),
        ("after", 1),
        ("before_sleep", 1),
        ("before", 2),
    ]
