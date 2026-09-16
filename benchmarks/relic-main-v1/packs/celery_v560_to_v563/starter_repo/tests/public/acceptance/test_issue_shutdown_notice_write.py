"""Withheld contract: shutdown notices reach the console under a cooperative os.write.

Green-thread libraries replace ``os.write`` with a version that refuses to run inside
their own event loop, which is exactly where a signal handler executes. The contract only
looks at what lands on the stream, not at how the write is performed.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from celery.apps.worker import safe_say


class CooperativeWriteInstalled:
    """Stand-in for a green-thread library that has monkey-patched ``os.write``."""

    def __enter__(self) -> None:
        self._original = os.write

        def refuse(fd: int, data: bytes) -> int:
            raise RuntimeError("do not call blocking functions from the mainloop")

        os.write = refuse

    def __exit__(self, *exc_info: object) -> None:
        os.write = self._original


def written(message: str, *, patched: bool) -> bytes:
    with tempfile.TemporaryFile() as stream:
        if patched:
            with CooperativeWriteInstalled():
                safe_say(message, stream)
        else:
            safe_say(message, stream)
        stream.seek(0)
        return stream.read()


def test_notice_is_emitted_even_when_os_write_refuses_to_block() -> None:
    assert written("warm shutdown (MainProcess)", patched=True) == (
        b"\nwarm shutdown (MainProcess)\n"
    )


def test_two_notices_in_a_row_both_survive() -> None:
    with tempfile.TemporaryFile() as stream:
        with CooperativeWriteInstalled():
            safe_say("worker: Hitting Ctrl+C again", stream)
            safe_say("worker: Cold shutdown", stream)
        stream.seek(0)

        assert stream.read() == (
            b"\nworker: Hitting Ctrl+C again\n\nworker: Cold shutdown\n"
        )


def test_ordinary_runtime_output_is_unchanged() -> None:
    assert written("celery@contract ready.", patched=False) == (
        b"\ncelery@contract ready.\n"
    )


def test_stream_without_a_descriptor_is_ignored_rather_than_raising() -> None:
    class NoDescriptor:
        def fileno(self) -> None:
            return None

    safe_say("no descriptor here", NoDescriptor())


@pytest.mark.parametrize("message", ["", "unicode \u2014 dash"])
def test_message_body_is_framed_by_newlines(message: str) -> None:
    assert written(message, patched=True) == f"\n{message}\n".encode()
