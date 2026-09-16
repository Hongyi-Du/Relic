"""Withheld contract: predicate retry strategies accept coroutines and compose with the sync ones.

The async strategy namespace is resolved inside the tests rather than at module import so
this file still collects against a tree that does not provide it yet.
"""

from __future__ import annotations

import asyncio

import tenacity


def test_result_predicate_strategy_accepts_a_coroutine() -> None:
    from tenacity.asyncio.retry import retry_if_result

    async def scenario():
        polls = []

        async def not_ready(value):
            return value != "ready"

        @tenacity.retry(
            retry=retry_if_result(not_ready),
            wait=tenacity.wait_fixed(0),
            stop=tenacity.stop_after_attempt(6),
        )
        async def poll():
            polls.append(1)
            return "ready" if len(polls) >= 3 else "pending"

        return await poll(), polls

    result, polls = asyncio.run(scenario())

    assert result == "ready"
    assert len(polls) == 3


def test_a_sync_strategy_composes_with_an_async_one() -> None:
    from tenacity.asyncio.retry import retry_if_exception

    async def scenario():
        attempts = []

        async def is_transient(exception):
            return isinstance(exception, TimeoutError)

        combined = tenacity.retry_if_exception_type(ValueError) | retry_if_exception(
            is_transient
        )

        @tenacity.retry(
            retry=combined,
            wait=tenacity.wait_fixed(0),
            stop=tenacity.stop_after_attempt(6),
        )
        async def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise ValueError("bad payload")
            if len(attempts) == 2:
                raise TimeoutError("slow upstream")
            return "settled"

        return await flaky(), attempts

    result, attempts = asyncio.run(scenario())

    assert result == "settled"
    assert len(attempts) == 3


def test_sync_strategy_composition_is_unchanged() -> None:
    attempts = []
    combined = tenacity.retry_if_exception_type(ValueError) | tenacity.retry_if_result(
        lambda value: value is None
    )

    @tenacity.retry(
        retry=combined, wait=tenacity.wait_fixed(0), stop=tenacity.stop_after_attempt(6)
    )
    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("bad payload")
        if len(attempts) == 2:
            return None
        return "settled"

    assert flaky() == "settled"
    assert len(attempts) == 3
