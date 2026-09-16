"""Withheld contract: the SQL result backend ships connection-health engine options.

Read straight off ``app.conf``, so the contract holds however the defaults are declared.
"""

from __future__ import annotations

from celery import Celery


def build_app() -> Celery:
    return Celery("contract", broker="memory://", backend="cache+memory://")


def test_engine_options_default_to_pre_ping_and_recycling() -> None:
    options = build_app().conf.database_engine_options

    assert options["pool_pre_ping"] is True
    assert options["pool_recycle"] == 3600


def test_default_is_reported_through_the_settings_lookup_helpers() -> None:
    app = build_app()

    assert app.conf.get("database_engine_options")["pool_recycle"] == 3600
    assert app.conf["database_engine_options"]["pool_pre_ping"] is True


def test_an_explicit_configuration_still_wins() -> None:
    app = build_app()
    app.conf.database_engine_options = {"pool_pre_ping": False, "echo": True}

    assert app.conf.database_engine_options == {"pool_pre_ping": False, "echo": True}


def test_reading_the_default_does_not_leak_between_applications() -> None:
    first = build_app()
    first.conf.database_engine_options["pool_recycle"] = 11

    assert build_app().conf.database_engine_options["pool_recycle"] == 3600


def test_neighbouring_database_settings_keep_their_values() -> None:
    conf = build_app().conf

    assert conf.database_short_lived_sessions is False
    assert conf.database_url is None
