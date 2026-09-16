"""Tests for tg_automation.retry."""

from __future__ import annotations

import pytest

from tg_automation.retry import FloodWaitError, NetworkError, retry_with_backoff


def test_floodwait_sleeps_for_requested_seconds():
    sleeps: list[float] = []
    calls = {"n": 0}

    def f():
        calls["n"] += 1
        if calls["n"] == 1:
            raise FloodWaitError(seconds=4)
        return "ok"

    out = retry_with_backoff(f, sleep=sleeps.append)
    assert out == "ok"
    assert sleeps == [4.0]


def test_network_error_retries_with_exponential_backoff():
    sleeps: list[float] = []
    calls = {"n": 0}

    def f():
        calls["n"] += 1
        if calls["n"] < 3:
            raise NetworkError("transient")
        return "ok"

    out = retry_with_backoff(f, base_delay=1.0, sleep=sleeps.append)
    assert out == "ok"
    assert sleeps == [1.0, 2.0]


def test_network_error_gives_up_after_max_attempts():
    sleeps: list[float] = []

    def f():
        raise NetworkError("always")

    with pytest.raises(NetworkError):
        retry_with_backoff(f, max_attempts=5, sleep=sleeps.append)
    assert len(sleeps) == 4  # 4 backoff sleeps before final failure
