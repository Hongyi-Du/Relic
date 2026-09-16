"""Withheld contract: coroutine lifecycle hooks are actually awaited by async retrying.

The event loop is driven by this module (``asyncio.run``) because the evaluator runs with
pytest plugin autoloading disabled. Only ``tenacity.retry`` and the documented hook
parameters are used, so any dispatch that awaits the hooks satisfies the contract.
"""

from __future__ import annotations

import asyncio

import tenacity

EXPECTED_TRACE = [
    ("before", 1),
    ("after", 1),
    ("before_sleep", 1),
    ("before", 2),
    ("after", 2),
    ("before_sleep", 2),
    ("before", 3),
]


def test_coroutine_hooks_are_awaited() -> None:
    async def scenario():
        trace = []

        async def note_before(retry_state):
            trace.append(("before", retry_state.attempt_number))

        async def note_after(retry_state):
            trace.append(("after", retry_state.attempt_number))

        async def note_before_sleep(retry_state):
            trace.append(("before_sleep", retry_state.attempt_number))

        attempts = []

        @tenacity.retry(
            before=note_before,
            after=note_after,
            before_sleep=note_before_sleep,
            wait=tenacity.wait_fixed(0),
            stop=tenacity.stop_after_attempt(3),
        )
        async def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("cold cache")
            return "warm"

        return await flaky(), attempts, trace

    result, attempts, trace = asyncio.run(scenario())

    assert result == "warm"
    assert len(attempts) == 3
    assert trace == EXPECTED_TRACE


def test_plain_hooks_still_work_on_the_async_path() -> None:
    async def scenario():
        trace = []

        def note_before(retry_state):
            trace.append(("before", retry_state.attempt_number))

        def note_after(retry_state):
            trace.append(("after", retry_state.attempt_number))

        def note_before_sleep(retry_state):
            trace.append(("before_sleep", retry_state.attempt_number))

        attempts = []

        @tenacity.retry(
            before=note_before,
            after=note_after,
            before_sleep=note_before_sleep,
            wait=tenacity.wait_fixed(0),
            stop=tenacity.stop_after_attempt(3),
        )
        async def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("cold cache")
            return "warm"

        return await flaky(), attempts, trace

    result, attempts, trace = asyncio.run(scenario())

    assert result == "warm"
    assert len(attempts) == 3
    assert trace == EXPECTED_TRACE
