"""Withheld contract: a coroutine give-up callback supplies the caller's fallback value.

The contract asserts on the value the caller receives, so any dispatch that awaits the
callback and returns its result passes.
"""

from __future__ import annotations

import asyncio

import tenacity


def test_coroutine_give_up_callback_result_reaches_the_caller() -> None:
    async def scenario():
        observed = []

        async def fall_back(retry_state):
            observed.append(retry_state.attempt_number)
            return {"status": "degraded", "attempts": retry_state.attempt_number}

        @tenacity.retry(
            retry_error_callback=fall_back,
            wait=tenacity.wait_fixed(0),
            stop=tenacity.stop_after_attempt(3),
        )
        async def always_fails():
            raise ValueError("no capacity")

        return await always_fails(), observed

    result, observed = asyncio.run(scenario())

    assert result == {"status": "degraded", "attempts": 3}
    assert observed == [3]


def test_plain_give_up_callback_result_still_reaches_the_caller() -> None:
    async def scenario():
        observed = []

        def fall_back(retry_state):
            observed.append(retry_state.attempt_number)
            return {"status": "degraded", "attempts": retry_state.attempt_number}

        @tenacity.retry(
            retry_error_callback=fall_back,
            wait=tenacity.wait_fixed(0),
            stop=tenacity.stop_after_attempt(3),
        )
        async def always_fails():
            raise ValueError("no capacity")

        return await always_fails(), observed

    result, observed = asyncio.run(scenario())

    assert result == {"status": "degraded", "attempts": 3}
    assert observed == [3]
