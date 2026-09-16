"""Withheld contract: an incoming request resolves ignore_result against the task.

The worker's decision to persist a failure is read off the request object; when the
message carries no preference the task's own declaration has to be honoured, and when it
carries one that preference wins.
"""

from __future__ import annotations

from celery import Celery
from celery.worker.request import Request


class Message:
    """A decoded broker message, reduced to what a request reads from it."""

    def __init__(self, headers: dict, body: tuple) -> None:
        self.headers = headers
        self.body = body
        self.payload = body
        self.content_type = "application/json"
        self.content_encoding = "utf-8"
        self.properties = {"correlation_id": headers["id"], "reply_to": ""}
        self.delivery_info = {"exchange": "", "routing_key": "celery"}


def build_request(*, task_ignores: bool, message_says=None) -> Request:
    app = Celery("contract", broker="memory://", backend="cache+memory://")

    @app.task(name="contract.job", ignore_result=task_ignores, shared=False)
    def job():  # pragma: no cover - never executed
        return None

    envelope = app.amqp.as_task_v2("contract-req", "contract.job", args=(), kwargs={})
    headers = dict(envelope.headers)
    if message_says is None:
        headers.pop("ignore_result", None)
    else:
        headers["ignore_result"] = message_says

    return Request(
        Message(headers, envelope.body), app=app, task=job, decoded=True
    )


def test_a_message_without_a_preference_follows_a_task_that_ignores_results() -> None:
    request = build_request(task_ignores=True)

    assert request.ignore_result is True
    assert request.store_errors is False


def test_a_message_without_a_preference_follows_a_task_that_keeps_results() -> None:
    request = build_request(task_ignores=False)

    assert request.ignore_result is False
    assert request.store_errors is True


def test_a_message_asking_for_results_overrides_a_task_that_ignores_them() -> None:
    request = build_request(task_ignores=True, message_says=False)

    assert request.ignore_result is False
    assert request.store_errors is True


def test_a_message_asking_to_ignore_overrides_a_task_that_keeps_them() -> None:
    request = build_request(task_ignores=False, message_says=True)

    assert request.ignore_result is True
    assert request.store_errors is False


def test_an_explicitly_empty_preference_falls_back_to_the_task() -> None:
    request = build_request(task_ignores=True, message_says=None)

    assert request.store_errors is False
