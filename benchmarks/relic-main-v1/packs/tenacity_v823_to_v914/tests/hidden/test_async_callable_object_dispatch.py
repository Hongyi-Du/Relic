"""Withheld contract: a callable object whose call is a coroutine is retried asynchronously.

The contract decorates an instance -- never a plain ``async def`` -- and asserts on the
awaited result and the number of times the body actually ran.
"""

from __future__ import annotations

import asyncio
import functools

import tenacity


class FlakyService:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise ValueError("socket reset")
        return "connected"


def test_instance_with_coroutine_call_is_retried() -> None:
    service = FlakyService(failures=2)
    decorated = tenacity.retry(
        stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_fixed(0)
    )(service)

    assert asyncio.run(decorated()) == "connected"
    assert service.calls == 3


def test_partial_over_a_coroutine_keeps_being_retried() -> None:
    service = FlakyService(failures=2)
    decorated = tenacity.retry(
        stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_fixed(0)
    )(functools.partial(service.__call__))

    assert asyncio.run(decorated()) == "connected"
    assert service.calls == 3


def test_plain_callables_keep_their_established_dispatch() -> None:
    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(0))
    def sync_target():
        return "sync"

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(0))
    async def async_target():
        return "async"

    assert sync_target() == "sync"
    assert asyncio.run(async_target()) == "async"
