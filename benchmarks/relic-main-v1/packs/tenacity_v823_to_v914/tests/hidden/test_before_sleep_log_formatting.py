"""Withheld contract: the pre-sleep log line renders the upcoming delay as a readable number.

A plain recording object stands in for the logger, so the contract exercises only the
documented ``before_sleep_log`` surface and the text it produces.
"""

from __future__ import annotations

import logging

import tenacity


class RecordingLogger:
    def __init__(self):
        self.records = []

    def log(self, level, message, *args, **kwargs):
        self.records.append((level, message))


def _run(**log_options):
    logger = RecordingLogger()

    def always_fails():
        raise ValueError("bad gateway")

    retrying = tenacity.Retrying(
        wait=tenacity.wait_fixed(0.1) + tenacity.wait_fixed(0.2),
        stop=tenacity.stop_after_attempt(2),
        sleep=lambda seconds: None,
        reraise=True,
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING, **log_options),
    )

    try:
        retrying(always_fails)
    except ValueError:
        pass
    else:
        raise AssertionError("expected the final failure to be re-raised")

    return logger.records


def test_the_default_rendering_is_compact() -> None:
    records = _run()

    assert len(records) == 1
    level, message = records[0]
    assert level == logging.WARNING
    assert "in 0.3 seconds" in message


def test_the_rendering_honours_an_explicit_format() -> None:
    assert "in 0.30 seconds" in _run(sec_format="%.2f")[0][1]


def test_the_rest_of_the_line_is_unchanged() -> None:
    _, message = _run()[0]

    assert message.startswith("Retrying ")
    assert "always_fails" in message
    assert message.endswith("as it raised ValueError: bad gateway.")
