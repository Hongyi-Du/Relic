"""Withheld contract: a retry signature does not carry dead-letter bookkeeping headers.

Built through ``Task.signature_from_request``, the public entry point a retry goes
through, so any implementation that strips the headers before republishing passes.
"""

from __future__ import annotations

from celery import Celery

DEAD_LETTER_HEADERS = {
    "x-death": [{"count": 4, "queue": "contract_delayed_3", "reason": "expired"}],
    "x-first-death-exchange": "contract_delayed_3",
    "x-first-death-queue": "contract_delayed_3",
    "x-first-death-reason": "expired",
    "x-last-death-exchange": "contract_delayed_7",
    "x-last-death-queue": "contract_delayed_7",
    "x-last-death-reason": "rejected",
}


def build_task():
    app = Celery("contract", broker="memory://", backend="cache+memory://")

    @app.task(bind=True, name="contract.flaky", shared=False)
    def flaky(self):  # pragma: no cover - never executed
        return None

    return flaky


def retry_headers(request_headers: dict) -> dict:
    task = build_task()
    task.push_request()
    try:
        task.request.headers = request_headers
        return dict(task.signature_from_request().options["headers"] or {})
    finally:
        task.pop_request()


def test_dead_letter_bookkeeping_is_not_republished_on_retry() -> None:
    headers = retry_headers(dict(DEAD_LETTER_HEADERS))

    assert headers == {}


def test_application_headers_survive_alongside_dead_letter_bookkeeping() -> None:
    headers = retry_headers(
        {"tenant": "acme", "trace-id": "7f3c", **DEAD_LETTER_HEADERS}
    )

    assert headers == {"tenant": "acme", "trace-id": "7f3c"}


def test_stripping_does_not_mutate_the_incoming_request_headers() -> None:
    task = build_task()
    task.push_request()
    try:
        incoming = {"tenant": "acme", **DEAD_LETTER_HEADERS}
        task.request.headers = incoming
        task.signature_from_request()

        assert set(incoming) == {"tenant", *DEAD_LETTER_HEADERS}
    finally:
        task.pop_request()


def test_a_request_without_headers_still_produces_a_usable_signature() -> None:
    task = build_task()
    task.push_request()
    try:
        signature = task.signature_from_request()

        assert signature.task == "contract.flaky"
        assert not signature.options.get("headers")
    finally:
        task.pop_request()
