"""Withheld contract: the built-in backoff pause follows whichever async library is running.

Both event loops are driven by this module because the evaluator runs with pytest plugin
autoloading disabled. No sleep function is supplied by the caller in either case.
"""

from __future__ import annotations

import asyncio

import trio

import tenacity


def _build_flaky():
    calls = []

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(0.01))
    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("connection refused")
        return "settled"

    return flaky, calls


def test_backoff_pause_works_under_trio_without_configuration() -> None:
    flaky, calls = _build_flaky()

    assert trio.run(flaky) == "settled"
    assert len(calls) == 3


def test_backoff_pause_still_works_under_asyncio() -> None:
    flaky, calls = _build_flaky()

    assert asyncio.run(flaky()) == "settled"
    assert len(calls) == 3


def test_an_explicit_sleep_function_is_still_honoured() -> None:
    requested_delays = []

    async def record_sleep(seconds):
        requested_delays.append(float(seconds))

    calls = []

    @tenacity.retry(
        sleep=record_sleep,
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_fixed(0.01),
    )
    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("connection refused")
        return "settled"

    assert trio.run(flaky) == "settled"
    assert requested_delays == [0.01, 0.01]
