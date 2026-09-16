"""Withheld contract: a result backend gets a recovery callback before each retry.

The callback is what lets a backend drop the connection state that caused the failure;
without it the retry loop reuses the same broken handle and burns through its budget.
"""

from __future__ import annotations

import pytest

from celery import Celery, states
from celery.backends.base import BaseBackend


class RecordingBackend(BaseBackend):
    """A backend that records every recovery callback it receives."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.recovered: list[BaseException] = []
        self.slept: list[float] = []

    def exception_safe_to_retry(self, exc: BaseException) -> bool:
        return isinstance(exc, TimeoutError)

    def on_backend_retryable_error(self, exc: BaseException) -> None:
        self.recovered.append(exc)

    def _sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


def build_backend(**overrides) -> RecordingBackend:
    app = Celery("contract", broker="memory://", backend="cache+memory://")
    app.conf.result_backend_always_retry = True
    app.conf.result_backend_max_retries = 4
    for key, value in overrides.items():
        setattr(app.conf, key, value)
    return RecordingBackend(app=app)


def test_reading_task_meta_recovers_before_every_retry() -> None:
    backend = build_backend()
    outcomes = [
        TimeoutError("first"),
        TimeoutError("second"),
        {"status": states.SUCCESS, "result": "contract"},
    ]

    def read(task_id):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    backend._get_task_meta_for = read

    assert backend.get_task_meta("contract-hook-1")["result"] == "contract"
    assert [str(exc) for exc in backend.recovered] == ["first", "second"]


def test_storing_a_result_recovers_before_every_retry() -> None:
    backend = build_backend()
    backend._get_task_meta_for = lambda task_id: {"status": states.PENDING}
    attempts = []
    outcomes = [TimeoutError("write failed"), None]

    def write(*args, **kwargs):
        attempts.append(args[:1])
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    backend._store_result = write

    assert backend.store_result("contract-hook-2", 42, states.SUCCESS) == 42
    assert len(attempts) == 2
    assert [str(exc) for exc in backend.recovered] == ["write failed"]


def test_a_failing_recovery_callback_does_not_abort_the_retry_loop() -> None:
    backend = build_backend()
    backend.on_backend_retryable_error = lambda exc: 1 / 0
    outcomes = [TimeoutError("first"), {"status": states.SUCCESS, "result": 7}]

    def read(task_id):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    backend._get_task_meta_for = read

    assert backend.get_task_meta("contract-hook-3")["result"] == 7


def test_errors_the_backend_calls_unrecoverable_are_not_retried() -> None:
    backend = build_backend()

    def read(task_id):
        raise ValueError("permanent")

    backend._get_task_meta_for = read

    with pytest.raises(ValueError):
        backend.get_task_meta("contract-hook-4")

    assert backend.recovered == []
    assert backend.slept == []
