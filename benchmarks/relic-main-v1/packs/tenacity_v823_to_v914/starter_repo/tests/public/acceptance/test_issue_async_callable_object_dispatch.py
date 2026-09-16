"""Acceptance check for issue_async_callable_object_dispatch.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: Decorating an instance whose `__call__` is a coroutine yields an awaitable that retries the body until it succeeds; plain synchronous functions still return their value directly and plain coroutine functions still behave exactly as before.
"""
import asyncio

from tenacity import retry, stop_after_attempt


def test_retry_decorated_async_callable_instance_retries():
    class Client:
        def __init__(self):
            self.calls = 0

        async def __call__(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient failure")
            return "success"

    client = Client()
    decorated_client = retry(stop=stop_after_attempt(2))(client)

    assert asyncio.run(decorated_client()) == "success"
    assert client.calls == 2
