"""Withheld contract: an awaitable pause function makes the whole retry loop awaitable.

The contract drives its own event loop and only observes the awaited result plus the
delays the injected pause function was actually asked to perform.
"""

from __future__ import annotations

import asyncio

import tenacity

STEP = 0.75


def test_an_awaitable_pause_function_is_honoured_for_a_plain_callable() -> None:
    async def scenario():
        requested_delays = []

        async def pause(seconds):
            requested_delays.append(float(seconds))

        calls = []

        @tenacity.retry(
            sleep=pause,
            wait=tenacity.wait_fixed(STEP),
            stop=tenacity.stop_after_attempt(5),
            retry=tenacity.retry_if_result(lambda value: value is None),
        )
        def poll():
            calls.append(1)
            return "ready" if len(calls) >= 3 else None

        return await poll(), calls, requested_delays

    result, calls, requested_delays = asyncio.run(scenario())

    assert result == "ready"
    assert len(calls) == 3
    assert requested_delays == [STEP, STEP]


def test_a_blocking_pause_function_keeps_the_call_synchronous() -> None:
    requested_delays = []
    calls = []

    @tenacity.retry(
        sleep=lambda seconds: requested_delays.append(float(seconds)),
        wait=tenacity.wait_fixed(STEP),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_result(lambda value: value is None),
    )
    def poll():
        calls.append(1)
        return "ready" if len(calls) >= 3 else None

    assert poll() == "ready"
    assert len(calls) == 3
    assert requested_delays == [STEP, STEP]


def test_an_awaitable_pause_function_still_works_for_a_coroutine() -> None:
    async def scenario():
        requested_delays = []

        async def pause(seconds):
            requested_delays.append(float(seconds))

        calls = []

        @tenacity.retry(
            sleep=pause,
            wait=tenacity.wait_fixed(STEP),
            stop=tenacity.stop_after_attempt(5),
            retry=tenacity.retry_if_result(lambda value: value is None),
        )
        async def poll():
            calls.append(1)
            return "ready" if len(calls) >= 3 else None

        return await poll(), calls, requested_delays

    result, calls, requested_delays = asyncio.run(scenario())

    assert result == "ready"
    assert len(calls) == 3
    assert requested_delays == [STEP, STEP]
