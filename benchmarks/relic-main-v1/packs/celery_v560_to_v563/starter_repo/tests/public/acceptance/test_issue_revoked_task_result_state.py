"""Withheld contract: revoking a task publishes REVOKED to the result store.

Observed through ``AsyncResult``, the same surface a client uses, so any implementation
that records the revocation in the backend satisfies the contract.
"""

from __future__ import annotations

from celery import Celery
from celery.result import AsyncResult
from celery.worker import state as worker_state
from celery.worker.control import revoke


class PanelState:
    """The minimal remote-control panel state a command handler receives."""

    def __init__(self, app: Celery) -> None:
        self.app = app
        self.hostname = "contract@localhost"
        self.consumer = None


def build_app() -> Celery:
    app = Celery("contract", broker="memory://", backend="cache+memory://")
    app.conf.task_always_eager = True
    return app


def test_revoked_task_reports_revoked_to_a_client() -> None:
    app = build_app()
    revoke(PanelState(app), "contract-revoked-single")

    assert AsyncResult("contract-revoked-single", app=app).state == "REVOKED"


def test_every_id_in_a_batch_revocation_reaches_the_result_store() -> None:
    app = build_app()
    batch = ["contract-revoked-a", "contract-revoked-b", "contract-revoked-c"]

    revoke(PanelState(app), batch)

    assert [AsyncResult(uid, app=app).state for uid in batch] == ["REVOKED"] * 3


def test_untouched_task_ids_stay_pending() -> None:
    app = build_app()
    revoke(PanelState(app), "contract-revoked-only-this-one")

    assert AsyncResult("contract-never-revoked", app=app).state == "PENDING"


def test_revocation_is_still_recorded_in_the_worker_state() -> None:
    app = build_app()
    revoke(PanelState(app), "contract-revoked-bookkeeping")

    assert "contract-revoked-bookkeeping" in worker_state.revoked
