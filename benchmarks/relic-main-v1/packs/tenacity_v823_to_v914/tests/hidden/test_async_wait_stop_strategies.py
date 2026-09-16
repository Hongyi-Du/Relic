"""Withheld contract: coroutine wait and stop strategies steer the async retry loop.

The delays are captured through an injected sleep callable rather than by reading any
internal state, so the contract holds for any implementation that awaits the strategies.
"""

from __future__ import annotations

import asyncio

import tenacity


def _expected_delays():
    return [0.25, 0.5, 0.75]


def test_coroutine_wait_and_stop_strategies_are_awaited() -> None:
    async def scenario():
        requested_delays = []

        async def record_sleep(seconds):
            requested_delays.append(float(seconds))

        async def ladder(retry_state):
            return retry_state.attempt_number * 0.25

        async def stop_at_fourth(retry_state):
            return retry_state.attempt_number >= 4

        attempts = []

        @tenacity.retry(
            sleep=record_sleep, wait=ladder, stop=stop_at_fourth, reraise=True
        )
        async def always_fails():
            attempts.append(1)
            raise ValueError("still broken")

        try:
            await always_fails()
        except ValueError:
            pass
        else:
            raise AssertionError("expected the final failure to be re-raised")

        return attempts, requested_delays

    attempts, requested_delays = asyncio.run(scenario())

    assert len(attempts) == 4
    assert requested_delays == _expected_delays()


def test_plain_wait_and_stop_strategies_still_steer_the_async_loop() -> None:
    async def scenario():
        requested_delays = []

        async def record_sleep(seconds):
            requested_delays.append(float(seconds))

        def ladder(retry_state):
            return retry_state.attempt_number * 0.25

        def stop_at_fourth(retry_state):
            return retry_state.attempt_number >= 4

        attempts = []

        @tenacity.retry(
            sleep=record_sleep, wait=ladder, stop=stop_at_fourth, reraise=True
        )
        async def always_fails():
            attempts.append(1)
            raise ValueError("still broken")

        try:
            await always_fails()
        except ValueError:
            pass
        else:
            raise AssertionError("expected the final failure to be re-raised")

        return attempts, requested_delays

    attempts, requested_delays = asyncio.run(scenario())

    assert len(attempts) == 4
    assert requested_delays == _expected_delays()
