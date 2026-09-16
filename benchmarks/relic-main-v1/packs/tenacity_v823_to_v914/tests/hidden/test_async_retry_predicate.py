"""Withheld contract: a coroutine retry predicate decides the loop instead of being ignored.

Everything goes through ``tenacity.retry(retry=...)``; the contract only observes how many
times the target ran and which attempt numbers the predicate saw.
"""

from __future__ import annotations

import asyncio

import tenacity


def test_coroutine_predicate_terminates_the_loop() -> None:
    async def scenario():
        seen = []

        async def keep_polling(retry_state):
            seen.append(retry_state.attempt_number)
            return retry_state.attempt_number < 3

        calls = []

        @tenacity.retry(
            retry=keep_polling,
            wait=tenacity.wait_fixed(0),
            stop=tenacity.stop_after_attempt(10),
        )
        async def probe():
            calls.append(1)
            return "pending"

        return await probe(), calls, seen

    result, calls, seen = asyncio.run(scenario())

    assert result == "pending"
    assert len(calls) == 3
    assert seen == [1, 2, 3]


def test_plain_predicate_still_terminates_the_loop() -> None:
    async def scenario():
        seen = []

        def keep_polling(retry_state):
            seen.append(retry_state.attempt_number)
            return retry_state.attempt_number < 3

        calls = []

        @tenacity.retry(
            retry=keep_polling,
            wait=tenacity.wait_fixed(0),
            stop=tenacity.stop_after_attempt(10),
        )
        async def probe():
            calls.append(1)
            return "pending"

        return await probe(), calls, seen

    result, calls, seen = asyncio.run(scenario())

    assert result == "pending"
    assert len(calls) == 3
    assert seen == [1, 2, 3]
