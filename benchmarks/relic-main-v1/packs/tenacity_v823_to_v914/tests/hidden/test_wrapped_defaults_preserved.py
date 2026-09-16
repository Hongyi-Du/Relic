"""Withheld contract: a retry wrapper keeps the argument defaults of the callable it wraps.

Exercised only through ``tenacity.retry`` / ``tenacity.Retrying`` and the standard
introspection attributes that every Python callable carries, so any wrapper construction
that carries the defaults across satisfies the contract.
"""

from __future__ import annotations

import asyncio

import tenacity


def test_positional_defaults_survive_the_sync_wrapper() -> None:
    @tenacity.retry(stop=tenacity.stop_after_attempt(2))
    def render(width=96, unit="px"):
        return f"{width}{unit}"

    assert render.__defaults__ == (96, "px")
    assert render() == "96px"


def test_keyword_only_defaults_survive_the_sync_wrapper() -> None:
    retrying = tenacity.Retrying(stop=tenacity.stop_after_attempt(2))

    def summarize(payload, *, indent=4, sort_keys=True):
        return (payload, indent, sort_keys)

    wrapped = retrying.wraps(summarize)

    assert wrapped.__kwdefaults__ == {"indent": 4, "sort_keys": True}
    assert wrapped("body") == ("body", 4, True)


def test_defaults_survive_the_async_wrapper() -> None:
    @tenacity.retry(stop=tenacity.stop_after_attempt(2))
    async def fetch(timeout=2.5, *, retries=3):
        return (timeout, retries)

    assert fetch.__defaults__ == (2.5,)
    assert fetch.__kwdefaults__ == {"retries": 3}
    assert asyncio.run(fetch()) == (2.5, 3)


def test_established_wrapper_metadata_remains_intact() -> None:
    @tenacity.retry(stop=tenacity.stop_after_attempt(2))
    def documented(value=0):
        """Docstring that must survive wrapping."""
        return value

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Docstring that must survive wrapping."
    assert callable(documented.retry_with)
    assert documented(7) == 7
